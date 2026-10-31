"""The CARLA half: draws what SUMO computed.

* Vehicles are spawned with physics off and placed every frame (ApplyTransform), so CARLA
  never drives them: their motion is SUMO's, interpolated between SUMO steps (interp.py).
* Traffic lights are frozen (CARLA's own cycle is stopped) and set from SUMO's signal state.
* Synchronous mode with a fixed frame time: frame k shows simulated time t0 + k * dt however
  long the GPU takes to render it, so a recording is smooth even on a slow GPU.
"""

from __future__ import annotations

import hashlib
import math
import logging

import carla

from cosim.blueprints import candidates_from_library, choose
from cosim.convert import tl_state
from cosim.mapping import Frame, sumo_angle_to_carla_yaw
from cosim.tlmap import Links

log = logging.getLogger("cosim.carla")
SPAWN_UP_M = 500.0
# Height above the pavement given to ApplyWalkerState. CARLA raises a walker by its capsule
# half-height itself (the API then reports the actor 1.10 m above the given point), so the
# pavement point is passed as it is. Measured with tools/probe_walkers.py on the server: with
# 0 the lowest foot bone is 4-6 cm above the ground under it (the sole) for 28/32 walkers in
# Town10 (the other 4 stand under an overhang that the ray hits first) and 113/113 in
# Ingolstadt7; with the capsule half-height added here (0.95) they floated 0.6-0.8 m.
WALKER_UP_M = 0.0
KERB_M = 0.15  # the pavement above the lanes, until CARLA's own ground height for the spot is known
SPAWN_TRIES = 3


LARGE_MAP_SPAN_M = 3000.0   # spawn points spread wider than this: a large (tiled) map
LARGE_ACTIVE_M = 20000.0    # covers all of a 10 km map: no dormant actors
LARGE_STREAM_M = 1500.0     # tiles kept around the spectator (default 3000 m); less VRAM on a 4 GB GPU


def is_large_map(cmap: carla.Map) -> bool:
    pts = cmap.get_spawn_points()
    if not pts:
        return False
    xs = [p.location.x for p in pts]
    ys = [p.location.y for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys)) > LARGE_MAP_SPAN_M


class CarlaSide:
    def __init__(self, client: carla.Client, world: carla.World, frame: Frame, fps: float,
                 links: Links, synchronous: bool = True):
        """links: landmark id -> the SUMO links it shows (tlmap.py). synchronous=False (free-fly,
        cosim/freefly.py): the server draws as fast as it can and nobody waits for a tick."""
        self.client = client
        self.world = world
        self.frame = frame
        self.dt = 1.0 / fps
        settings = world.get_settings()
        settings.synchronous_mode = synchronous
        settings.fixed_delta_seconds = self.dt if synchronous else None
        settings.no_rendering_mode = False
        if is_large_map(world.get_map()):
            # Large maps (Town12/13, 10 x 10 km) stream ~2 km tiles around the spectator and
            # make actors further than actor_active_distance dormant. SUMO moves every vehicle
            # of the city; we draw only a district (--draw-radius), but keep whatever is drawn
            # and every sensor active wherever the camera goes.
            settings.actor_active_distance = LARGE_ACTIVE_M
            settings.tile_stream_distance = LARGE_STREAM_M
        world.apply_settings(settings)
        self.library = world.get_blueprint_library()
        self.cands = candidates_from_library(self.library)
        self.actors: dict[str, int] = {}          # SUMO id -> CARLA actor id
        self.lights: dict[str, int] = {}          # SUMO id -> last light bits sent
        self.failed: set[str] = set()
        self.ground_levels: list[float] | None = None  # use_road_levels()
        self._tries: dict[str, int] = {}
        self.walkers: dict[str, int] = {}         # SUMO person id -> CARLA walker id
        self._walker_bps: list = []
        self.map = world.get_map()
        self._tl_links = self._map_traffic_lights(links)
        world.freeze_all_traffic_lights(True)
        self.tl_conflicts = 0
        self.tl_checks = self.tl_mismatches = 0
        self._tl_set: dict[int, carla.TrafficLightState] = {}

    # -- traffic lights --------------------------------------------------------------------

    def _map_traffic_lights(self, links: Links) -> dict[int, tuple[carla.TrafficLight, list[tuple[str, int]]]]:
        """CARLA traffic-light actor -> the SUMO links (tls id, link index) it shows.

        Counted for the report: landmarks without a traffic-light actor (`tl_no_actor`) and
        landmarks no SUMO link refers to (`tl_unlinked`, drawn but left as CARLA froze them)."""
        out: dict[int, tuple[carla.TrafficLight, list[tuple[str, int]]]] = {}
        self.tl_no_actor = self.tl_unlinked = 0
        for lm in self.map.get_all_landmarks_of_type("1000001"):
            sumo_links = links.get(lm.id)
            tl = self.world.get_traffic_light(lm)
            if tl is None:
                self.tl_no_actor += 1
                continue
            if not sumo_links:
                self.tl_unlinked += 1
                continue
            entry = out.setdefault(tl.id, (tl, []))
            entry[1].extend(p for p in sumo_links if p not in entry[1])
        log.info("traffic lights: %d CARLA actors showing %d SUMO links (%d landmarks without actor, "
                 "%d without SUMO link)", len(out), sum(len(v[1]) for v in out.values()),
                 self.tl_no_actor, self.tl_unlinked)
        return out

    @property
    def traffic_light_count(self) -> int:
        return len(self._tl_links)

    def set_traffic_lights(self, sumo_tls: dict[str, str]) -> None:
        # First check that what was set last step is what the server shows now (frozen lights
        # must not run their own cycle): counted per actor and step in `tl_mismatches`.
        for tl_id, want in self._tl_set.items():
            self.tl_checks += 1
            if self._tl_links[tl_id][0].get_state() != want:
                self.tl_mismatches += 1
        if getattr(self, "lights_off", False):
            # each head drawn with its own direction instead (cosim/signal_heads.py): the pole's lamps dark
            for tl, _links in self._tl_links.values():
                if self._tl_set.get(tl.id) != carla.TrafficLightState.Off:
                    tl.set_state(carla.TrafficLightState.Off)
                    self._tl_set[tl.id] = carla.TrafficLightState.Off
            return
        for tl, links in self._tl_links.values():
            chars = [sumo_tls[t][i] for t, i in links if t in sumo_tls and i < len(sumo_tls[t])]
            if not chars:
                continue
            state, conflict = tl_state(chars)
            self.tl_conflicts += conflict
            if self._tl_set.get(tl.id) != state:
                tl.set_state(state)
                self.tl_set_calls = getattr(self, "tl_set_calls", 0) + 1
            self._tl_set[tl.id] = state

    # -- ground --------------------------------------------------------------------------------

    def ground(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        """(z, pitch) of the driving lane under CARLA point (x, y) for a vehicle facing `yaw`.
        SUMO's corridor networks are flat; CARLA's towns are not."""
        wp = (self._wp_on_levels(x, y, yaw) if self.ground_levels else
              self.map.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True,
                                    lane_type=carla.LaneType.Driving))
        if wp is None:
            return 0.0, 0.0
        r = wp.transform.rotation
        same_way = abs((yaw - r.yaw + 180.0) % 360.0 - 180.0) <= 90.0
        # Waypoint pitch comes wrapped (Town05: 360, 359.4, -360, 366.6 for roads within a few
        # degrees of level). Unwrapped, Poser's blend from 0 to 360 passed through 180 and the car
        # turned upside down for a moment, so it is brought into [-180, 180) here.
        pitch = (r.pitch + 180.0) % 360.0 - 180.0
        return wp.transform.location.z, (pitch if same_way else -pitch)

    def use_road_levels(self, step_m: float = 5.0, spacing_m: float = 20.0) -> list[float]:
        """For towns with roads above roads (Town05's raised ring road, 10 m up): the heights at
        which ground() looks for the lane, every `step_m` between the lowest and highest road.
        SUMO's network from netconvert_carla is flat, and a lookup from z = 0 picks the road
        below a bridge: measured on Town05, 58 of 3164 lane segments (the ring road) were put
        10 m too low. Off by default: the films and datasets were made with the plain lookup."""
        zs = [w.transform.location.z for w in self.map.generate_waypoints(spacing_m)]
        lo, hi = min(zs), max(zs)
        n = int((hi - lo) // step_m) + 1
        self.ground_levels = [lo + k * step_m for k in range(n + 1)]
        log.info("ground lookup on %d levels, z %.1f to %.1f", len(self.ground_levels), lo, hi)
        return self.ground_levels

    def _wp_on_levels(self, x: float, y: float, yaw: float):
        """The driving lane under (x, y) on whichever level fits best: nearest in plan, and
        running the vehicle's way (or the opposite way) rather than across it."""
        best, best_score = None, None
        for z in self.ground_levels:
            wp = self.map.get_waypoint(carla.Location(x, y, z), project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            loc = wp.transform.location
            d = ((loc.x - x) ** 2 + (loc.y - y) ** 2) ** 0.5
            turn = abs((yaw - wp.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            score = d + (0.0 if turn < 45.0 or turn > 135.0 else 20.0)
            if best_score is None or score < best_score - 1e-6:
                best, best_score = wp, score
        return best

    # -- vehicles ----------------------------------------------------------------------------

    def spawn(self, new: list[tuple[str, str, carla.Transform]]) -> None:
        """new: (SUMO id, vclass, first transform)."""
        batch, ids = [], []
        for vid, vclass, tf in new:
            bp_id, colour = choose(self.cands, vclass, vid)
            bp = self.library.find(bp_id)
            if colour and bp.has_attribute("color"):
                bp.set_attribute("color", colour)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "sumo")
            # Spawn high above the road (no collision with neighbours or buildings on the
            # spawn check), then placed exactly on the next frame. 50 m was not enough in
            # Town10: one spawn failed against a building ("collision at spawn position").
            up = carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + SPAWN_UP_M),
                                 tf.rotation)
            batch.append(carla.command.SpawnActor(bp, up).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
            ids.append(vid)
        if not batch:
            return
        for vid, resp in zip(ids, self.client.apply_batch_sync(batch, False), strict=True):
            if resp.error:
                # retried on the next SUMO steps; given up after SPAWN_TRIES
                self._tries[vid] = self._tries.get(vid, 0) + 1
                if self._tries[vid] >= SPAWN_TRIES:
                    self.failed.add(vid)
                log.warning("could not spawn %s (try %d): %s", vid, self._tries[vid], resp.error)
            else:
                self.actors[vid] = resp.actor_id

    def move(self, poses: dict[str, carla.Transform], lights: dict[str, carla.VehicleLightState]) -> None:
        batch = [carla.command.ApplyTransform(self.actors[v], tf) for v, tf in poses.items() if v in self.actors]
        for v, ls in lights.items():
            if v in self.actors and self.lights.get(v) != int(ls):
                self.lights[v] = int(ls)
                batch.append(carla.command.SetVehicleLightState(self.actors[v], ls))
        if batch:
            self.client.apply_batch(batch)

    def light_changes(self, lights: dict[str, carla.VehicleLightState]) -> list[tuple[int, int]]:
        """(actor, light state as int) of the vehicles whose lights changed, noted as set: for another process
        to send with its poses (one call to the server instead of two)."""
        out = []
        for v, ls in lights.items():
            aid = self.actors.get(v)
            if aid is not None and self.lights.get(v) != int(ls):
                self.lights[v] = int(ls)
                out.append((aid, int(ls)))
        return out

    def set_lights(self, lights: dict[str, carla.VehicleLightState]) -> None:
        """Only the lights (`move` sends them with the poses): when another process sends the poses."""
        batch = []
        for v, ls in lights.items():
            if v in self.actors and self.lights.get(v) != int(ls):
                self.lights[v] = int(ls)
                batch.append(carla.command.SetVehicleLightState(self.actors[v], ls))
        if batch:
            self.client.apply_batch(batch)

    def destroy(self, vids: list[str]) -> None:
        batch = [carla.command.DestroyActor(self.actors.pop(v)) for v in vids if v in self.actors]
        for v in vids:
            self.lights.pop(v, None)
        if batch:
            self.client.apply_batch(batch)

    # -- pedestrians ------------------------------------------------------------------------

    PAVEMENT_GLIDE_M = 0.6   # per call (one SUMO step, 0.5 s): how fast a person's pavement offset may change

    def walker_pose(self, x: float, y: float, angle: float, road: str = "", pid: str | None = None) -> carla.Transform:
        """CARLA transform of a SUMO person at (x, y) heading `angle` (SUMO convention), for
        ApplyWalkerState: the point on the pavement under it (see WALKER_UP_M).

        A person who is not on a crossing is kept on CARLA's pavement: SUMO's walking areas reach past
        CARLA's kerbs, so people waiting for the walk signal stood on the road's shoulder, in the
        street (measured on Town05: 952 of 1 778 waiting samples on a shoulder or driving lane). Such a
        point is moved onto the nearest CARLA sidewalk within 8 m, 0.5 m inside its edges (_onto_pavement).
        With `pid`, call once per SUMO step and person: the move onto the pavement then glides (see below);
        forget(pid) when the person is gone."""
        cx, cy = self.frame.to_carla_xy(x, y)
        on_crossing = road.startswith(":") and "_c" in road
        tx, ty = (cx, cy) if on_crossing else self._onto_pavement(cx, cy)
        if pid is None:
            cx, cy = tx, ty
        else:
            # Glide: the offset onto the pavement changes by at most PAVEMENT_GLIDE_M a step. Without it a
            # person stepping off a crossing (not moved) onto a corner (moved) jumped up to 5-10 m in one
            # step (tools/check_ped_pavement.py: 900 jumps over 1 m in 6 min).
            if not hasattr(self, "_pave_off"):
                self._pave_off = {}
            want = (tx - cx, ty - cy)
            have = self._pave_off.get(pid, want)
            d = math.hypot(want[0] - have[0], want[1] - have[1])
            if d > self.PAVEMENT_GLIDE_M:
                k = self.PAVEMENT_GLIDE_M / d
                have = (have[0] + (want[0] - have[0]) * k, have[1] + (want[1] - have[1]) * k)
            else:
                have = want
            self._pave_off[pid] = have
            cx, cy = cx + have[0], cy + have[1]
        wp = self.map.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
        z = wp.transform.location.z if wp is not None else 0.0
        z = self._surface_z(cx, cy, z, KERB_M if not on_crossing else 0.0)
        rot = carla.Rotation(0.0, sumo_angle_to_carla_yaw(angle), 0.0)
        return carla.Transform(carla.Location(cx, cy, z + WALKER_UP_M), rot)

    def _surface_z(self, cx: float, cy: float, z_lane: float, guess: float = 0.0) -> float:
        """Height of what is actually under a walker: the map's lanes are flat at road level, but the
        pavement is a kerb higher, so walkers placed at the lane's height stood in it to the shins.
        CARLA's ground projection from 2 m above, kept per 0.5 m square (the ground does not move);
        anything that is not ground (a car, a walker, a pole) falls back to the lane's height.

        The projection is a call to the server that waits for its next picture. It was made on the SUMO
        thread, a walking person entering a new square every step or so: with CARLA at 30 FPS (a camera
        open, the night) the SUMO thread fell behind and the vehicles juddered. Now a square not looked up
        yet answers at once with the lane's height plus `guess` (the kerb, off a crossing), and a thread
        of its own asks the server; its answer is used from then on."""
        if not hasattr(self, "_zcache"):
            self._zcache = {}
        key = (round(cx * 2.0), round(cy * 2.0), round(z_lane))
        z = self._zcache.get(key)
        if z is not None:
            return z
        self._ask_ground(key, cx, cy, z_lane)
        return z_lane + guess

    def _ask_ground(self, key, cx: float, cy: float, z_lane: float) -> None:
        """With `ground_server` = (host, port) set, the server is asked by a process of its own
        (cosim/ground_proc.py: the call keeps Python's lock while it waits for the server, which held up every
        thread of the free-fly viewer); else by a thread."""
        import queue
        import threading

        if not hasattr(self, "_zq"):
            self._zasked: set = set()
            server = getattr(self, "ground_server", None)
            if server is not None:
                import multiprocessing as mp

                from cosim import ground_proc

                ctx = mp.get_context("spawn")
                self._zq, zout = ctx.Queue(), ctx.Queue()
                self._zproc = ctx.Process(target=ground_proc.run, args=(*server, self._zq, zout), daemon=True)
                self._zproc.start()
                threading.Thread(target=self._ground_answers, args=(zout,), daemon=True,
                                 name="ground-answers").start()
            else:
                self._zq = queue.Queue()
                threading.Thread(target=self._ground_worker, daemon=True, name="ground").start()
        if key in self._zasked:
            return
        if len(self._zasked) - len(self._zcache) >= self.GROUND_IN_FLIGHT:
            return  # the server answers ~40 a second: the rest are asked when they come up again
        self._zasked.add(key)
        self._zq.put((key, cx, cy, z_lane))

    GROUND_IN_FLIGHT = 24  # questions waiting for an answer at most (~0.6 s of the server's pictures)

    def _ground_answers(self, zout) -> None:
        while True:
            key, z = zout.get()
            self._zcache[key] = z

    def _ground_worker(self) -> None:
        from cosim.ground_proc import ground_z

        while True:
            key, cx, cy, z_lane = self._zq.get()
            self._zcache[key] = ground_z(self.world, carla, cx, cy, z_lane)

    def stop_ground(self) -> None:
        """Ends the ground process, if there is one."""
        proc = getattr(self, "_zproc", None)
        if proc is not None:
            try:
                self._zq.put(("stop",))
                proc.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            if proc.is_alive():
                proc.terminate()
            # questions still buffered for the pipe would keep this process from ending (multiprocessing
            # waits for its queue to be written out at exit: the viewer hung there once, CARLA left open)
            self._zq.cancel_join_thread()

    PAVEMENT_MARGIN_M = 0.5  # kept this far inside the sidewalk's edges (the kerb is on the road side)
    PAVEMENT_REACH_M = 8.0   # points farther than this from any sidewalk are left where they are

    def _onto_pavement(self, cx: float, cy: float) -> tuple[float, float]:
        """The nearest point on CARLA's pavement, PAVEMENT_MARGIN_M inside the sidewalk's edges.

        Measured offline (tools/check_ped_pavement.py, SUMO alone, CARLA's Town05 map, 10 min): people
        standing still were mostly at the kerb line of a junction corner, 0 m from the driving lane and
        1-2 m outside the sidewalk. The first version left them there: it only moved points that lay on a
        non-sidewalk lane, and a corner point lies on no lane at all (get_waypoint returns None there,
        and sometimes even for points well inside a sidewalk). It also kept the offset along the
        sidewalk, which at a sidewalk's end put the point past the end, into the junction. Now: the
        nearest sidewalk point, its lateral offset clamped to the sidewalk minus the margin, no offset
        along it."""
        sw = self.map.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True,
                                   lane_type=carla.LaneType.Sidewalk)
        if sw is None:
            return cx, cy
        c = sw.transform.location
        r = sw.transform.get_right_vector()
        f = sw.transform.get_forward_vector()
        rn, fn = math.hypot(r.x, r.y) or 1.0, math.hypot(f.x, f.y) or 1.0
        rx, ry, fx, fy = r.x / rn, r.y / rn, f.x / fn, f.y / fn
        dx, dy = cx - c.x, cy - c.y
        side, along = dx * rx + dy * ry, dx * fx + dy * fy
        half = max(sw.lane_width / 2.0 - self.PAVEMENT_MARGIN_M, 0.1)
        if abs(side) <= half and abs(along) <= 0.5:
            return cx, cy  # on the pavement already
        if math.hypot(dx, dy) > self.PAVEMENT_REACH_M:
            return cx, cy
        side = max(-half, min(half, side))
        return c.x + side * rx, c.y + side * ry

    def spawn_walkers(self, new: list[tuple[str, carla.Transform]]) -> None:
        if not self._walker_bps:
            self._walker_bps = sorted(self.library.filter("walker.pedestrian.*"), key=lambda b: b.id)
        batch, ids = [], []
        for pid, tf in new:
            h = int.from_bytes(hashlib.blake2b(pid.encode(), digest_size=4).digest(), "little")
            bp = self._walker_bps[h % len(self._walker_bps)]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            up = carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z + SPAWN_UP_M), tf.rotation)
            batch.append(carla.command.SpawnActor(bp, up))
            ids.append(pid)
        if not batch:
            return
        for pid, resp in zip(ids, self.client.apply_batch_sync(batch, False), strict=True):
            if resp.error:
                self._tries[pid] = self._tries.get(pid, 0) + 1
                if self._tries[pid] >= SPAWN_TRIES:
                    self.failed.add(pid)
            else:
                self.walkers[pid] = resp.actor_id

    def move_walkers(self, poses: dict[str, tuple[carla.Transform, float]]) -> None:
        """Pose and walking speed (the speed drives the walk animation)."""
        batch = [carla.command.ApplyWalkerState(self.walkers[p], tf, speed) for p, (tf, speed) in poses.items()
                 if p in self.walkers]
        if batch:
            self.client.apply_batch(batch)

    def forget_walkers(self, keep) -> None:
        """Drop the pavement offsets of people no longer drawn."""
        off = getattr(self, "_pave_off", None)
        if off:
            for p in [p for p in off if p not in keep]:
                del off[p]

    def destroy_walkers(self, pids: list[str]) -> None:
        batch = [carla.command.DestroyActor(self.walkers.pop(p)) for p in pids if p in self.walkers]
        if batch:
            self.client.apply_batch(batch)

    def tick(self) -> int:
        return self.world.tick()

    def close(self) -> None:
        self.destroy(list(self.actors))
        self.destroy_walkers(list(self.walkers))
        try:
            self.world.tick()
        except RuntimeError:
            pass
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.world.freeze_all_traffic_lights(False)

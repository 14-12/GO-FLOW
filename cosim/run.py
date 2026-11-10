"""Run a SUMO scenario and render it in CARLA.

    python -m cosim.run --scenario town10 --controller webster --duration 60 --record out/town10.mp4
    python -m cosim.run --scenario ingolstadt7 --controller webster --camera orbit --record out/i7.mp4

(1) Load the CARLA map: a town by name, or the map derived from the SUMO network (OpenDRIVE
standalone mode). (2) Run SUMO, warming up without rendering. (3) For every SUMO step, render
`fps` frames per simulated second with vehicles interpolated between the step's start and
end states, traffic lights set from SUMO, and the sun placed for the scenario clock.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import carla

from cosim import scenario as scenario_mod
from cosim.camera import NearView, Pose, Recorder, arc_pose, chase_pose, clear_arc, orbit_pose, smooth
from cosim.carla_side import CarlaSide, is_large_map
from cosim.convert import light_state, vehicle_transform
from cosim.interp import State, hermite
from cosim.mapping import sumo_angle_to_carla_yaw
from cosim.solar import carla_sun_azimuth, scenario_sun
from cosim.sumo_side import StepResult, SumoSide

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("cosim")

WEATHER = {
    "clear": dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, fog_density=0.0),
    "cloudy": dict(cloudiness=80.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, fog_density=5.0),
    "rain": dict(cloudiness=90.0, precipitation=60.0, precipitation_deposits=50.0, wetness=80.0, fog_density=2.0),
    "fog": dict(cloudiness=60.0, precipitation=0.0, precipitation_deposits=0.0, wetness=20.0, fog_density=45.0),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="town10")
    ap.add_argument("--controller", default="webster", choices=["webster", "actuated", "default", "max_pressure"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--fps", type=float, default=20.0, help="rendered frames per simulated second")
    ap.add_argument("--warmup", type=float, default=300.0, help="simulated seconds before rendering")
    ap.add_argument("--duration", type=float, default=60.0, help="simulated seconds to render")
    ap.add_argument("--camera", default="orbit", help="orbit[:tlsID] | tour[:N] | chase | fixed:x,y,z,yaw,pitch")
    ap.add_argument("--record", default=None, help="write an MP4 here")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fov", type=float, default=75.0)
    ap.add_argument("--weather", default="clear", choices=list(WEATHER))
    ap.add_argument("--hour", type=float, default=None, help="override the clock for the light (visual only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sumo-port", type=int, default=8813)
    ap.add_argument("--report", default=None, help="write a JSON report of the run here")
    ap.add_argument("--extras", action="store_true",
                    help="add the main project's pedestrians and emergency vehicles (cosim/extras.py)")
    ap.add_argument("--emergency-every", type=float, default=None,
                    help="with --extras: mean simulated seconds between emergency vehicles (main project: 75)")
    ap.add_argument("--draw-radius", type=float, default=None,
                    help="draw only the vehicles and pedestrians within this many metres of the camera's "
                         "junction (large maps: SUMO runs the whole city, CARLA draws the district filmed)")
    ap.add_argument("--stills", default=None, help="also save a PNG every --stills-every simulated seconds here")
    ap.add_argument("--stills-every", type=float, default=2.0)
    return ap.parse_args(argv)


def load_world(client: carla.Client, scen: scenario_mod.Scenario) -> carla.World:
    t0 = time.time()
    if scen.kind == "carla-town":
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != scen.carla_map:
            world = client.load_world(scen.carla_map)
        log.info("town %s ready in %.1f s", scen.carla_map, time.time() - t0)
        return world
    params = carla.OpendriveGenerationParameters(
        vertex_distance=1.0, max_road_length=50.0, wall_height=0.0, additional_width=0.6,
        smooth_junctions=True, enable_mesh_visibility=True, enable_pedestrian_navigation=True)
    world = client.generate_opendrive_world(scen.xodr.read_text(encoding="utf-8"), params)
    log.info("OpenDRIVE world generated in %.1f s", time.time() - t0)
    return world


CENTRE_BLOCK_MAX = 0.15  # large maps: largest share of the picture's centre something may hide
ORBIT_DIST_M = (45.0 ** 2 + 20.5 ** 2) ** 0.5  # camera (45 m out, 22 m up) to the target (1.5 m up)
NEAR_VIEW_MAX = 0.03  # largest share of the picture something solid within 15 m may cover
LARGE_MAP_M = 3000.0  # wider than this (Town12, Town13: ~10 km) = a large map, streamed in tiles


def tour_stops(centres: dict, arg: str) -> list[str]:
    """tour:N = the N busiest junctions; tour:id1,id2,... = these junctions in this order."""
    if arg and not arg.isdigit():
        ids = [t for t in arg.split(",") if t]
        missing = [t for t in ids if t not in centres]
        if missing:
            raise SystemExit(f"tour: no signalised junction {missing}")
        return ids
    return list(centres)[: int(arg or 4)]


def stream_in(world, tick, cx: float, cy: float, z0: float = 0.0, min_ticks: int = 60, min_s: float = 20.0) -> int:
    """Large maps load their ~2 km tiles around the spectator (WorldSettings.spectator_as_ego):
    put it above (cx, cy) and tick for at least min_ticks and min_s seconds (tiles were measured
    to stream in within 4 s, tools/probe_large_map.py). Returns the ticks.

    No ray is cast to check (World.cast_ray): on Town12, during a co-simulation, a ray blocked
    the 0.9.16 server for good, three runs out of three (15 Oct: 300 s and 900 s client
    timeouts, GPU idle), 20 ticks after the camera jumped and also 60 ticks / 20 s after it."""
    world.get_spectator().set_transform(carla.Transform(carla.Location(cx, cy, z0 + 60.0), carla.Rotation(-90, 0, 0)))
    t0 = time.time()
    n = 0
    while n < min_ticks or time.time() - t0 < min_s:
        tick()
        n += 1
    return n


def settle(tick, fast_s: float = 1.0, need: int = 10, max_ticks: int = 600) -> int:
    """Tick until `need` ticks in a row each take less than `fast_s` (the server has finished
    what it does after a map load or a settings change). Returns the ticks made."""
    n = run = 0
    while n < max_ticks and run < need:
        t = time.time()
        tick()
        n += 1
        run = run + 1 if time.time() - t < fast_s else 0
    return n


def ground_z(cmap, cx: float, cy: float) -> float:
    """Height of the road at a junction. The camera heights in camera.py are above the ground
    at z = 0, as in Town10 and the maps made from SUMO networks; Town12's centre lies ~360 m
    up (its terrain), where a camera at z = 22 m would film the underside of the map."""
    wp = cmap.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True)
    if wp is None:
        return 0.0
    wp2 = cmap.get_waypoint(carla.Location(cx, cy, wp.transform.location.z), project_to_road=True)
    return float((wp2 or wp).transform.location.z)


def lift(pose: Pose, z0: float) -> Pose:
    return dataclasses.replace(pose, z=pose.z + z0) if z0 else pose


def camera_arcs(world, centres: dict, kind: str, arg: str, tick=None, fov: float = 75.0,
                stats: dict | None = None, zs: dict | None = None) -> dict:
    """For orbit and tour cameras: the clear arc around each junction filmed (camera.py). With
    `tick` (synchronous mode), each candidate view is also rendered and rejected if something
    solid within 15 m covers more than NEAR_VIEW_MAX of it. `zs`, if given, receives the road
    height at each junction (ground_z): every pose and ray is lifted by it."""
    if kind == "orbit":
        ids = [arg or next(iter(centres))]
    elif kind == "tour":
        ids = tour_stops(centres, arg)
    else:
        return {}
    spawn = world.get_map().get_spawn_points()
    mx = sum(p.location.x for p in spawn) / max(len(spawn), 1)
    my = sum(p.location.y for p in spawn) / max(len(spawn), 1)
    # A large map (kilometres across) has no meaningful "inside": the map-centre rule that
    # keeps a Town10 camera from filming the void beyond the town is not used there.
    span = max((max(p.location.x for p in spawn) - min(p.location.x for p in spawn)) if spawn else 0.0,
               (max(p.location.y for p in spawn) - min(p.location.y for p in spawn)) if spawn else 0.0)
    large = span > LARGE_MAP_M

    cmap = world.get_map()

    view = NearView(world, tick, fov) if tick is not None else None
    arcs = {}
    try:
        for tid in ids:
            cx, cy, _ = centres[tid]
            z0 = ground_z(cmap, cx, cy)
            if zs is not None:
                zs[tid] = z0
            if tick is not None and large:
                stream_in(world, tick, cx, cy, z0)  # the tiles round it must be loaded first
            seen = {}

            def labels(a, b, z0=z0):
                if large:  # no rays on large maps (see stream_in): the rendered view decides
                    return []
                hits = world.cast_ray(carla.Location(a[0], a[1], a[2] + z0), carla.Location(b[0], b[1], b[2] + z0))
                return [getattr(h.label, "name", str(h.label)) for h in hits]

            def view_ok(deg, cx=cx, cy=cy, seen=seen, z0=z0):
                pose = lift(orbit_pose(cx, cy, 0.0, start_deg=deg, deg_per_s=0.0), z0)
                if not large:
                    seen[deg] = view.near_fraction(pose)
                    return seen[deg] <= NEAR_VIEW_MAX
                near, blocked = view.assess(pose, target_m=ORBIT_DIST_M)
                seen[deg] = max(near, blocked * NEAR_VIEW_MAX / CENTRE_BLOCK_MAX)  # one score, same threshold
                return near <= NEAR_VIEW_MAX and blocked <= CENTRE_BLOCK_MAX

            arcs[tid] = clear_arc(labels, cx, cy, (mx - cx, my - cy) if spawn and not large else None,
                                  view_ok=view_ok if view else None)
            rejected = sorted(d for d, f in seen.items() if f > NEAR_VIEW_MAX)
            if stats is not None:
                stats["camera_views_rendered"] = stats.get("camera_views_rendered", 0) + len(seen)
            log.info("camera arc at %s (road at z=%.1f): %s (views rendered %d, rejected for a close obstacle: %s)",
                     tid, z0, arcs[tid], len(seen), rejected)
    finally:
        if view:
            view.close()
    return arcs


def within(step: StepResult, frame, cx: float, cy: float, radius: float, keep: set[str],
           margin: float = 60.0) -> StepResult:
    """`step` with only the vehicles and pedestrians within `radius` of CARLA point (cx, cy);
    those in `keep` (drawn already) stay until `radius + margin`, so nothing flickers at the edge."""
    def near(oid, x, y):
        px, py = frame.to_carla_xy(x, y)
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        r = radius + margin if oid in keep else radius
        return d2 <= r * r
    return dataclasses.replace(
        step, vehicles={k: v for k, v in step.vehicles.items() if near(k, v.x, v.y)},
        persons={k: p for k, p in step.persons.items() if near(k, p.x, p.y)})


class Poser:
    """Vehicle transforms for one SUMO step: Hermite between the step's start and end, and
    the ground (z, pitch) taken from the CARLA map at both ends and blended linearly.

    heading="tangent" (the films): the body faces along the Hermite curve. heading="turn"
    (free-fly): the body turns from SUMO's heading at the start of the step to the one at the
    end, the short way round, at an even rate. Measured on 300 s of Town05 traffic at 30 poses a
    second: the tangent heading reversed its turning 2827 times (210 of 252 vehicles: they shook
    in turns and lane changes, where SUMO's heading and the curve disagree) and jumped by more
    than 15 deg between two poses 271 times; turning evenly: 127 reversals, no jump.

    heading="trail" (free-fly since 19 Oct): the front follows a Catmull-Rom curve through SUMO's
    positions, driven at SUMO's speeds made continuous across steps (interp.Timed, since 22 Oct) (`pre`, prev, cur, `nxt`: the step before and the step after, so it runs a step behind),
    and the body points along the front's own path (interp.Trails, kept by the caller across steps).
    The even turn still shook: its rate jumped at every step and the body slid sideways against
    the motion (p99.9 slip 32 deg); see interp.Trails for the numbers."""

    def __init__(self, frame, ground=None, heading: str = "tangent"):
        self.frame = frame
        self.ground = ground  # (x, y, yaw) -> (z, pitch), CARLA coordinates; None: flat
        self.heading = heading
        self._g: dict[str, tuple[float, float]] = {}
        self.pre: dict | None = None  # vehicles of the step before prev (heading="trail")
        self.nxt: dict | None = None  # vehicles of the step after cur (heading="trail")

    def begin_step(self, prev: StepResult | None, cur: StepResult) -> None:
        old = self._g
        self._g = {}
        self._from = {}
        for vid, v in cur.vehicles.items():
            p = prev.vehicles.get(vid) if prev else None
            o = old.get(vid)
            if o is not None and p is not None and (p.x, p.y, p.angle, p.length) == (v.x, v.y, v.angle, v.length):
                self._g[vid] = o  # (10 Nov) standing still: the same ground, not looked up again (exact; 6 cars in
                #   10 stand at the evening peak, and the lookups were a quarter of the viewer's SUMO thread)
            else:
                self._g[vid] = self._ground_of(v)
            self._from[vid] = o or (self._ground_of(p) if p else self._g[vid])
        self.prev, self.cur = prev, cur

    def _ground_of(self, v) -> tuple[float, float]:
        if self.ground is None:
            return 0.0, 0.0
        tf = vehicle_transform(self.frame, v.x, v.y, v.angle, v.length)
        return self.ground(tf.location.x, tf.location.y, sumo_angle_to_carla_yaw(v.angle))

    def poses(self, s: float, dt: float, trails=None) -> dict[str, carla.Transform]:
        out = {}
        for vid, v in self.cur.vehicles.items():
            p = self.prev.vehicles.get(vid) if self.prev else None
            if p is None:
                x, y, ang = v.x, v.y, v.angle
                if self.heading == "trail" and trails is not None:
                    ang = trails.update(vid, x, y, v.length, v.angle)
            elif self.heading == "trail" and trails is not None:
                from cosim.interp import Timed

                if not hasattr(self, "_timed"):
                    self._timed = {}
                seg = self._timed.get(vid)
                if seg is None:  # once per step and vehicle: the curve and its timing (interp.Timed)
                    q0 = (self.pre or {}).get(vid)
                    q3 = (self.nxt or {}).get(vid)
                    p1, p2 = (p.x, p.y), (v.x, v.y)
                    p0 = (q0.x, q0.y) if q0 is not None else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
                    p3 = (q3.x, q3.y) if q3 is not None else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])
                    seg = self._timed[vid] = Timed(p0, p1, p2, p3)
                x, y = seg.at(s)
                if getattr(v, "fixed", False):   # (8 Nov) a heading given (cosim/parking_motion.py)
                    ang = trails.fixed(vid, x, y, v.length,
                                       p.angle + ((v.angle - p.angle + 180.0) % 360.0 - 180.0) * min(max(s, 0.0), 1.0))
                else:
                    ang = trails.update(vid, x, y, v.length, v.angle)
            else:
                x, y, ang = hermite(State(p.x, p.y, p.angle, p.speed), State(v.x, v.y, v.angle, v.speed), s, dt)
                if self.heading == "turn":
                    ang = (p.angle + ((v.angle - p.angle + 180.0) % 360.0 - 180.0) * min(max(s, 0.0), 1.0)) % 360.0
            (z0, p0), (z1, p1) = self._from[vid], self._g[vid]
            out[vid] = vehicle_transform(self.frame, x, y, ang, v.length, z0 + (z1 - z0) * s, p0 + (p1 - p0) * s)
        return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    scen = scenario_mod.load(args.scenario, ROOT)
    centres = scenario_mod.tls_centres(scen.net_file, scen.frame)
    log.info("scenario %s (%s), %d signals, sun for %.4f N %.4f E", scen.name, scen.kind, len(centres),
             scen.lat, scen.lon)

    client = carla.Client(args.host, args.port)
    client.set_timeout(300.0)
    server_version = client.get_server_version()
    log.info("CARLA server %s, client %s", server_version, client.get_client_version())
    world = load_world(client, scen)
    large = is_large_map(world.get_map())
    if large:
        # Town12/13: the first calls after loading and after the large-map settings (every actor
        # of the 10 km map made active) were measured to block the server for 1-3 minutes.
        client.set_timeout(900.0)
    landmark_ids = [lm.id for lm in world.get_map().get_all_landmarks_of_type("1000001")]
    links = scen.links(landmark_ids)

    sumo = SumoSide(scen.folder, args.controller, seed=args.seed, port=args.sumo_port, extras=args.extras,
                     emergency_every=args.emergency_every)
    dt_sumo = sumo.step_length
    sub = max(1, round(args.fps * dt_sumo))
    fps = sub / dt_sumo
    begin = sumo.time
    prev = None
    while sumo.time < begin + args.warmup:
        prev = sumo.step()
    log.info("warm-up done at t=%.0f with %d vehicles", sumo.time, len(prev.vehicles) if prev else 0)

    t_settle = time.time()
    carla_side = CarlaSide(client, world, scen.frame, fps=fps, links=links)
    if large:
        n = settle(carla_side.tick)
        log.info("large map: settings applied and server settled in %.0f s (%d ticks)", time.time() - t_settle, n)
    poser = Poser(scen.frame, carla_side.ground)
    w, h = (int(v) for v in args.res.lower().split("x"))
    rec = Recorder(world, w, h, fov=args.fov, path=args.record, fps=fps)

    weather = carla.WeatherParameters(**WEATHER[args.weather])
    stats = {"frames": 0, "sumo_steps": 0, "max_vehicles": 0, "spawn_failures": 0,
             "render_s": 0.0, "sim_begin": sumo.time, "fps_video": fps}
    kind, _, cam_arg = args.camera.partition(":")
    zs: dict[str, float] = {}
    arcs = camera_arcs(world, centres, kind, cam_arg, tick=carla_side.tick, fov=args.fov, stats=stats, zs=zs)
    cam, chase_id = None, None

    def focus(tsim: float) -> tuple[float, float]:
        """Where the camera looks at time tsim (CARLA x, y): what --draw-radius is centred on."""
        if kind == "orbit":
            cx, cy, _ = centres[cam_arg or next(iter(centres))]
            return cx, cy
        if kind == "tour":
            stops = tour_stops(centres, cam_arg)
            cx, cy, _ = centres[stops[min(int(tsim / (args.duration / len(stops))), len(stops) - 1)]]
            return cx, cy
        if kind == "fixed":
            x, y = (float(v) for v in cam_arg.split(",")[:2])
            return x, y
        return (cam.x, cam.y) if cam is not None else next(iter(centres.values()))[:2]

    wall0 = time.time()
    try:
        end = sumo.time + args.duration
        while sumo.time < end:
            cur = sumo.step()
            stats["sumo_steps"] += 1
            secs = args.hour * 3600 if args.hour is not None else scen.clock_start_s + cur.time
            az, el = scenario_sun(secs % 86400, scen.lat, scen.lon)
            weather.sun_altitude_angle = el
            weather.sun_azimuth_angle = carla_sun_azimuth(az)  # convention measured, see solar.py
            world.set_weather(weather)
            night = el < 3.0

            carla_side.destroy(cur.arrived)
            stats["sumo_vehicles_max"] = max(stats.get("sumo_vehicles_max", 0), len(cur.vehicles))
            if args.draw_radius:
                fx, fy = focus(cur.time - stats["sim_begin"])
                cur = within(cur, scen.frame, fx, fy, args.draw_radius, set(carla_side.actors) | set(carla_side.walkers))
                prev = within(prev, scen.frame, fx, fy, args.draw_radius + 200.0, set()) if prev else prev
                carla_side.destroy([vid for vid in carla_side.actors if vid not in cur.vehicles])
                stats["drawn_max"] = max(stats.get("drawn_max", 0), len(cur.vehicles))
            poser.begin_step(prev, cur)
            first = poser.poses(0.0, dt_sumo)
            new = [(vid, v.vclass, first[vid]) for vid, v in cur.vehicles.items()
                   if vid not in carla_side.actors and vid not in carla_side.failed]
            carla_side.spawn(new)
            stats["spawn_failures"] = len(carla_side.failed)
            stats["max_vehicles"] = max(stats["max_vehicles"], len(carla_side.actors))
            carla_side.set_traffic_lights(cur.tls)
            lights = {vid: light_state(v.signals, night, emergency_run=sumo.on_run(vid))
                      for vid, v in cur.vehicles.items()}
            carla_side.destroy_walkers(cur.persons_arrived + [p for p in carla_side.walkers if p not in cur.persons])
            carla_side.spawn_walkers([(pid, carla_side.walker_pose(pp.x, pp.y, pp.angle))
                                      for pid, pp in cur.persons.items()
                                      if pid not in carla_side.walkers and pid not in carla_side.failed])
            stats["max_pedestrians"] = max(stats.get("max_pedestrians", 0), len(carla_side.walkers))
            if args.extras:
                ev = [vid for vid, v in cur.vehicles.items() if v.vclass in ("emergency", "authority")]
                stats.setdefault("emergency_seen", set()).update(ev)
                stats["emergency_on_run_max"] = max(stats.get("emergency_on_run_max", 0),
                                                    sum(sumo.on_run(v) for v in ev))

            for k in range(1, sub + 1):
                s = k / sub
                poses = poser.poses(s, dt_sumo)
                carla_side.move(poses, lights)
                if cur.persons:
                    walk = {}
                    for pid, pp in cur.persons.items():
                        q = prev.persons.get(pid) if prev else None
                        x, y = (pp.x, pp.y) if q is None else (q.x + (pp.x - q.x) * s, q.y + (pp.y - q.y) * s)
                        walk[pid] = (carla_side.walker_pose(x, y, pp.angle), pp.speed)
                    carla_side.move_walkers(walk)
                tsim = cur.time - dt_sumo + s * dt_sumo - stats["sim_begin"]
                if kind == "orbit":
                    tid = cam_arg or next(iter(centres))
                    cx, cy, _ = centres[tid]
                    pose = lift(arc_pose(cx, cy, tsim, arcs[tid]), zs.get(tid, 0.0))
                elif kind == "tour":
                    # The N busiest junctions in turn, an equal share of the run each, cut
                    # from one to the next.
                    stops = tour_stops(centres, cam_arg)
                    i = min(int(tsim / (args.duration / len(stops))), len(stops) - 1)
                    cx, cy, _ = centres[stops[i]]
                    pose = lift(arc_pose(cx, cy, tsim, arcs[stops[i]]), zs.get(stops[i], 0.0))
                elif kind == "chase":
                    if chase_id not in poses:
                        moving = [vid for vid, v in cur.vehicles.items() if v.speed > 6 and vid in carla_side.actors]
                        chase_id = moving[0] if moving else next(iter(poses), None)
                    pose = chase_pose(poses[chase_id]) if chase_id else cam
                else:
                    x, y, z, yaw, pitch = (float(v) for v in cam_arg.split(","))
                    pose = Pose(x, y, z, yaw, pitch)
                # tour: a cut from one junction to the next; a glide between them flew the camera
                # through buildings (seen in the Town10 tour, 14 Oct)
                cam = smooth(cam, pose, {"chase": 0.25}.get(kind, 1.0))
                if cam is not None:
                    rec.place(cam)
                t1 = time.time()
                fid = carla_side.tick()
                still = None
                if args.stills and stats["frames"] % max(1, round(args.stills_every * fps)) == 0:
                    still = str(Path(args.stills) / f"t{tsim:06.1f}.png")
                rec.collect(fid, still=still)
                stats["render_s"] += time.time() - t1
                stats["frames"] += 1
            prev = cur
            if stats["sumo_steps"] % max(1, round(10 / dt_sumo)) == 0:
                log.info("t=%.0f vehicles=%d frames=%d (%.3f s/frame) tl-conflicts=%d", cur.time,
                         len(carla_side.actors), stats["frames"], stats["render_s"] / max(stats["frames"], 1),
                         carla_side.tl_conflicts)
    finally:
        if "emergency_seen" in stats:
            stats["emergency_seen"] = sorted(stats["emergency_seen"])
        stats.update(
            sim_end=sumo.time, wall_s=round(time.time() - wall0, 1),
            s_per_frame=round(stats["render_s"] / max(stats["frames"], 1), 4),
            tl_actors=carla_side.traffic_light_count, tl_no_actor=carla_side.tl_no_actor,
            tl_unlinked=carla_side.tl_unlinked, tl_conflict_updates=carla_side.tl_conflicts,
            tl_state_checks=carla_side.tl_checks, tl_state_mismatches=carla_side.tl_mismatches,
            controlled_tls=len(sumo.controlled), video_frames=rec.written, scenario=scen.name,
            kind=scen.kind, controller=args.controller, camera=args.camera, weather=args.weather,
            res=args.res, server_version=server_version, client_version=client.get_client_version(),
            map=scen.map_info.__dict__ if scen.map_info else scen.meta)
        rec.close()
        carla_side.close()
        sumo.close()
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("done: %s", json.dumps({k: v for k, v in stats.items() if k != "map"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

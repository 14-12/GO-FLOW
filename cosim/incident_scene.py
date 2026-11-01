"""What an accident looks like in CARLA (23-24 Oct): the wrecks as they came to rest, debris, the drivers out and on
the pavement, the police car and its officer guiding the traffic past the cones, the recovery truck backing in,
tilting its deck and winching each wreck up (cosim/towing.py). The traffic side (who crashed, who comes, when) is
cosim/incidents.py, in SUMO's thread; this follows its snapshot once per SUMO step, in the viewer's pose feeder
(FreeFly.apply), and makes no server call itself: everything goes to the pose process (cosim/pose_proc.py), which
moves the actors with the traffic and makes the few server calls a scene needs (props, doors, the officer's arm).

The cars cannot be deformed in CARLA 0.9.16 (no damage model, and the runtime texture API reaches only an actor's
static meshes, not a car's body). Damage is shown by what a crash does to a car's position and surroundings:
turned and pushed by the impact (more for a harder one, no further than its lane holds), tilted and sunk on a broken
suspension, on its side after a hard side impact, hazard lights on, glass and plastic on the road around it (more
for a harder one), oil.

What the marker says follows what can be seen: POLICE ON SCENE once the officer stands at his post (a police car
held up further back sends him on foot), the recovery truck ON THE WAY until it backs in, then LOADING.
"""

from __future__ import annotations

import logging
import math
import random
import time

from cosim.incidents import block_at

log = logging.getLogger("incident_scene")

POLICE_BP = ("walker.pedestrian.0030", "walker.pedestrian.0032")
WORKER_BP = ("walker.pedestrian.0052",)
CIVIL_BP = ("walker.pedestrian.0002", "walker.pedestrian.0005", "walker.pedestrian.0014", "walker.pedestrian.0026",
            "walker.pedestrian.0028", "walker.pedestrian.0035")
DOOR_MODELS = {"vehicle.dodge.charger_2020", "vehicle.dodge.charger_police_2020", "vehicle.ford.crown",
               "vehicle.lincoln.mkz_2020", "vehicle.mercedes.coupe_2020", "vehicle.mercedes.sprinter",
               "vehicle.mini.cooper_s_2021", "vehicle.nissan.patrol_2021", "vehicle.volkswagen.t2_2021",
               "vehicle.ford.ambulance", "vehicle.carlamotors.firetruck", "vehicle.carlamotors.european_hgv"}
WALK_MS = 1.4
DOOR_OPEN_S = 2.6                   # a door opened to get out is shut again (it would reach into a lane)
POST_BEHIND_M = 4.0                 # the officer's post: between his car (8 m back) and the wrecks, inside the cones,
                                    # the car shielding him from anyone still coming up the closed lane
GLASS = ("static.prop.brokentile01", "static.prop.brokentile02", "static.prop.brokentile03", "static.prop.brokentile04")
BITS = ("static.prop.garbage05", "static.prop.garbage06", "static.prop.garbage02")
FLUID = ("static.prop.dirtdebris01", "static.prop.dirtdebris02", "static.prop.dirtdebris03")
STATUS_COLOUR = {"ACCIDENT": "crash", "BREAKDOWN": "crash", "POLICE ON THE WAY": "police en route", "POLICE ARRIVING": "police en route",
                 "POLICE ON SCENE": "secured", "TOW TRUCK ON THE WAY": "recovery", "LOADING THE WRECK": "recovery",
                 "CLEARING": "clearing"}


class SceneWalkers:
    """The walkers a scene uses (officers, the trucks' crew, drivers), parked out of sight while loading."""

    def __init__(self, ids_by_role: dict[str, list[int]]):
        self.free = {k: list(v) for k, v in ids_by_role.items()}

    def take(self, role: str) -> int | None:
        pool = self.free.get(role) or []
        return pool.pop() if pool else None

    def give(self, role: str, aid: int) -> None:
        self.free.setdefault(role, []).append(aid)


def _local(x: float, y: float, yaw: float, lx: float, ly: float) -> tuple[float, float]:
    """A point lx forward, ly right of (x, y) facing yaw (CARLA: left-handed, y to the right)."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return x + lx * c - ly * s, y + lx * s + ly * c


def _inside(x: float, y: float, body: tuple, margin: float) -> bool:
    """(x, y) within `margin` of a vehicle's box (CARLA centre x, y, yaw in degrees, half length, half width)."""
    cx, cy, yaw, hl, hw = body
    a = math.radians(yaw)
    dx, dy = x - cx, y - cy
    along = dx * math.cos(a) + dy * math.sin(a)
    across = -dx * math.sin(a) + dy * math.cos(a)
    return abs(along) <= hl + margin and abs(across) <= hw + margin


class IncidentScene:
    def __init__(self, frame, ground, net, send, walkers: SceneWalkers, cs, models: dict | None = None):
        """frame: SUMO <-> CARLA; ground(x, y, yaw) -> (z, pitch) in CARLA coordinates; net: sumolib net (lane shapes);
        send(msg): to the pose process; cs: the pooled CarlaSide (its actors: SUMO id -> actor)."""
        self.frame, self.ground, self.net, self.send, self.walkers, self.cs = frame, ground, net, send, walkers, cs
        self.models = models or {}
        self.state: dict[int, dict] = {}
        self.dmg: dict[str, dict] = {}          # wreck vid -> damage, with its wall-clock start
        self.carried: dict[int, tuple] = {}     # wreck actor -> (truck vid, pool key)
        self.markers: list[dict] = []           # for the overlay (CARLA x, y, z, label, ...)
        self.decks_free: list[int] = []         # the recovery trucks' decks (cosim/plate_proc.py makes them)
        self.decks_on: dict[str, tuple] = {}    # truck vid -> (truck actor, deck)
        self.extra_tows = None                  # () -> vids of recovery trucks not at an accident (the yard's: cosim/depots.py)
        self.pending_release: list = []
        self.closes: list = []                  # (time, actor, door): doors to shut again
        self.later: list = []                   # (time, message): sent when due (a walker shows from his first key on)
        self.taken: set[str] = set()            # wrecks on a truck: never drawn again by the traffic side

    # ------------------------------------------------------------------ geometry
    def _carla(self, x: float, y: float, angle: float, length: float):
        """Centre (CARLA x, y, z) and yaw of a vehicle from SUMO's front position and angle."""
        from cosim.mapping import front_to_centre, sumo_angle_to_carla_yaw

        cx, cy = front_to_centre(x, y, angle, length / 2)
        ux, uy = self.frame.to_carla_xy(cx, cy)
        yaw = sumo_angle_to_carla_yaw(angle)
        z = self.ground(ux, uy, yaw)[0]
        return ux, uy, z, yaw

    def _lane_point(self, lane: str, pos: float, lateral: float):
        """CARLA (x, y, z, yaw) at `pos` along a SUMO lane, `lateral` m to its left (SUMO's left)."""
        from sumolib.geomhelper import positionAtShapeOffset

        ln = self.net.getLane(lane)
        shape = ln.getShape()
        geo = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(shape, shape[1:], strict=False))
        k = geo / max(ln.getLength(), 0.01)
        p = min(max(pos, 0.0), ln.getLength()) * k
        x, y = positionAtShapeOffset(shape, p)
        x2, y2 = positionAtShapeOffset(shape, min(p + 0.5, geo))
        if math.hypot(x2 - x, y2 - y) < 1e-3:
            x2, y2 = x, y
            x, y = positionAtShapeOffset(shape, max(p - 0.5, 0.0))
        dx, dy = x2 - x, y2 - y
        n = math.hypot(dx, dy) or 1.0
        lx, ly = -dy / n, dx / n  # SUMO: left of the direction of travel
        ux, uy = self.frame.to_carla_xy(x + lx * lateral, y + ly * lateral)
        from cosim.mapping import sumo_angle_to_carla_yaw

        yaw = sumo_angle_to_carla_yaw(math.degrees(math.atan2(dx, dy)))
        return ux, uy, self.ground(ux, uy, yaw)[0], yaw

    def _wreck_pose(self, w: dict):
        """A wreck as drawn: CARLA (x, y, z, yaw), pushed and turned by its damage."""
        x, y, z, yaw = self._carla(w["x"], w["y"], w["angle"], w["length"])
        d = w["damage"]
        x, y = _local(x, y, yaw, d.get("dx", 0.0), 0.0)
        return x, y, z, yaw + d.get("yaw", 0.0)

    def _half_width(self, model: str | None) -> float:
        info = self.models.get(model or "")
        return info["extent"][1] * 0.92 if info else 0.95

    # ------------------------------------------------------------------ per SUMO step
    def damage_for(self, vid: str) -> dict | None:
        return self.dmg.get(vid)

    def add_decks(self, ids: list[int]) -> None:
        self.decks_free += list(ids)

    def update(self, snapshot: list[dict], vehicles: dict) -> None:
        """snapshot: IncidentManager.snapshot(); vehicles: this step's SUMO vehicles near you (incident ones always)."""
        now = time.time()
        live = {i["id"] for i in snapshot}
        for iid in [k for k in self.state if k not in live]:
            self._finish(iid)
        markers = []
        for inc in snapshot:
            st = self.state.get(inc["id"])
            if st is None:
                st = self.state[inc["id"]] = {"seen": now, "debris": False, "cones": False, "officer": None,
                                              "at_post": None, "drivers": {}, "towing": {}, "workers": {},
                                              "rng": random.Random(inc["id"] * 7919), "phase": None}
                self._crash(inc, st, now, vehicles)
            else:
                self._joined(inc, st, now)
            self._step(inc, st, now, vehicles)
            markers.append(self._marker(inc, st, now))
        self._decks(snapshot)
        self._doors(now)
        due = [m for t, m in self.later if now >= t]
        self.later = [(t, m) for t, m in self.later if now < t]
        for m in due:
            self.send(m)
        self._release_carried()
        self.markers = markers

    def _door(self, aid: int, which: str, now: float, shut_after: float | None = DOOR_OPEN_S) -> None:
        self.send(("door", aid, True, which))
        if shut_after is not None:
            self.closes.append((now + shut_after, aid, which))

    def _doors(self, now: float) -> None:
        keep = []
        for t, aid, which in self.closes:
            if now >= t:
                self.send(("door", aid, False, which))
            else:
                keep.append((t, aid, which))
        self.closes = keep

    def _decks(self, snapshot: list[dict]) -> None:
        """Each recovery truck carries its deck while it is drawn (the accidents' and the yard's); the deck goes back
        when the truck is gone."""
        vids = [t["vid"] for inc in snapshot for t in inc["tows"]]
        if self.extra_tows is not None:
            vids += [v for v in self.extra_tows() if v not in vids]
        for vid in vids:
            aid = self.cs.actors.get(vid)
            if aid is None or vid in self.decks_on or not self.decks_free:
                continue
            d = self.decks_free.pop()
            self.decks_on[vid] = (aid, d)
            self.send(("deck", aid, d))
        for vid, (aid, d) in list(self.decks_on.items()):
            if self.cs.actors.get(vid) != aid:
                self.send(("deck_off", aid))
                self.decks_on.pop(vid)
                self.decks_free.append(d)

    def _crash(self, inc: dict, st: dict, now: float, vehicles: dict) -> None:
        for w in inc["wrecks"]:
            if w["vid"] not in self.dmg:
                self.dmg[w["vid"]] = dict(w["damage"], t0=now)
        if inc["kind"] == "breakdown":  # (31 Oct) nothing hit anything: no debris
            return
        self._debris(inc, st, inc["wrecks"])

    def _joined(self, inc: dict, st: dict, now: float) -> None:
        """Cars that ran into the wrecks later (a pile-up, 31 Oct; before, they stood there undamaged), or into a
        stalled car: their damage, and debris where they hit."""
        new = [w for w in inc["wrecks"] if w["vid"] not in self.dmg and not w["removed"]]
        for w in new:
            self.dmg[w["vid"]] = dict(w["damage"], t0=now)
        if inc["kind"] == "breakdown":
            return
        if not st["debris"]:
            self._debris(inc, st, inc["wrecks"])
        elif new:
            self._debris(inc, st, new)

    def _debris(self, inc: dict, st: dict, wrecks: list) -> None:
        pts = [self._carla(w["x"], w["y"], w["angle"], w["length"]) for w in wrecks]
        if not pts:
            return
        # the debris: where the cars met (between them), glass and plastic, oil for the harder crashes
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = min(p[2] for p in pts)
        k = max(w["damage"].get("severity", 0.3) for w in wrecks)
        r = st["rng"]
        props = []
        n_glass, n_bits, n_fluid = 3 + int(9 * k), 1 + int(4 * k), (1 if k > 0.35 else 0) + (1 if k > 0.75 else 0)
        spread = 1.0 + 1.6 * k
        for kinds, n, rad in ((GLASS, n_glass, spread), (BITS, n_bits, spread * 1.2), (FLUID, n_fluid, 1.0)):
            for _ in range(n):
                a, d = r.uniform(0, 2 * math.pi), rad * math.sqrt(r.random())
                props.append((r.choice(kinds), cx + d * math.cos(a), cy + d * math.sin(a), cz + 0.02, r.uniform(0, 360)))
        self.send(("props", f"inc{inc['id']}_debris", props))
        st["debris"] = True

    # ------------------------------------------------------------------ the people
    def _drivers_out(self, inc: dict, st: dict, now: float) -> None:
        """Each driver gets out on the pavement's side (the door facing it), shuts the door and goes to wait on the
        pavement beside the car (cosim/incidents.py holds the traffic when an open lane is in the way)."""
        for w in inc["wrecks"]:
            ex = w.get("exit")
            vid = w["vid"]
            if not ex or not ex["out"] or w["removed"] or vid in st["drivers"]:
                continue
            aid = self.cs.actors.get(vid)
            wid = self.walkers.take("civil")
            if aid is None or wid is None:
                if wid is not None:
                    self.walkers.give("civil", wid)
                st["drivers"][vid] = None  # no walker free: stays in the car
                continue
            x, y, z, yaw = self._wreck_pose(w)
            right = abs(w["damage"].get("yaw", 0.0)) < 90.0  # the pavement is on the car's right (SUMO lane 0)
            side = 1.0 if right else -1.0
            half = self._half_width(w.get("model"))
            hl = w["length"] / 2.0
            door = _local(x, y, yaw, 0.35, side * (half + 0.45))
            others = [o for o in inc["wrecks"] if o["vid"] != vid and not o["removed"]]
            pts = [door]
            if any(self._near_wreck(o, door, 0.5) for o in others):
                # that door is against the other car: out on the other side, round the back of the car
                side = -side
                pts = [_local(x, y, yaw, 0.35, side * (half + 0.45)), _local(x, y, yaw, -hl - 1.0, side * (half + 0.5)),
                       _local(x, y, yaw, -hl - 1.0, -side * (half + 0.5))]
            tx, ty = self.frame.to_carla_xy(*ex["target"])
            tz = self.ground(tx, ty, yaw)[0]
            if w.get("model") in DOOR_MODELS:
                self._door(aid, "FR" if side > 0 else "FL", now)
            t0 = now + 1.0
            look = math.degrees(math.atan2(y - ty, x - tx))
            pts.append((tx, ty))
            keys, t = [], t0
            for i, p in enumerate(pts):
                if i:
                    t += max(math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) / 1.3, 0.6)
                nxt = pts[i + 1] if i + 1 < len(pts) else None
                hd = math.degrees(math.atan2(nxt[1] - p[1], nxt[0] - p[0])) if nxt else look
                keys.append((t, p[0], p[1], z if nxt else tz, hd))
            keys.append((t + 0.4, tx, ty, tz, look))
            self.later.append((t0 - 0.1, ("walk", wid, keys)))
            st["drivers"][vid] = wid
            st.setdefault("driver_spot", {})[vid] = (ex.get("walk_lane"), ex.get("walk_pos"), tx, ty, tz)

    def _near_wreck(self, w: dict, p: tuple, margin: float) -> bool:
        """Point p within `margin` of wreck w's body (as it lies)."""
        x, y, _, yaw = self._wreck_pose(w)
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        dx, dy = p[0] - x, p[1] - y
        lx, ly = dx * c + dy * s, -dx * s + dy * c
        return abs(lx) < w["length"] / 2.0 + margin and abs(ly) < self._half_width(w.get("model")) + margin

    def _drivers_leave(self, st: dict, now: float) -> None:
        """At clearing the drivers walk off along the pavement and are put away."""
        for vid, wid in list(st["drivers"].items()):
            if wid is None:
                continue
            spot = st.get("driver_spot", {}).get(vid)
            if spot and spot[0]:
                x, y, z, yaw = self._lane_point(spot[0], spot[1] + 14.0, 0.0)
                self.send(("walk", wid, [(now, spot[2], spot[3], spot[4], yaw), (now + 10.0, x, y, z, yaw)]))
            self.pending_release.append((now + 10.5, wid, "civil"))
            st["drivers"][vid] = None

    def _post(self, inc: dict, pv, vehicles: dict | None = None) -> tuple | None:
        """The officer's post: in the closed lane POST_BEHIND_M behind the wrecks (in front of his car), just inside the
        cone line on the open lane's side, facing the oncoming traffic. (x, y, z, face, side of the open lane as he faces the traffic)."""
        closed = inc.get("closed") or []
        live = [w for w in inc["wrecks"] if not w["removed"] and w["lane"] and not w["lane"].startswith(":")]
        if not closed or not live:
            if pv is None:
                return None
            px, py, pz, pyaw = self._carla(pv.x, pv.y, pv.angle, pv.length)
            fx, fy = _local(px, py, pyaw, pv.length / 2 + 2.0, -1.45)  # ahead of his car, on its driver's side
            return fx, fy, pz, (pyaw + 180.0) % 360.0, "R"
        lane = closed[0]
        on = [w for w in live if w["lane"] == lane] or live
        rear = min(w["pos"] - w["length"] for w in on)
        ln = self.net.getLane(lane)
        half = ln.getWidth() / 2.0
        side = 1.0
        open_lane = inc.get("open_lane")
        if open_lane:
            side = 1.0 if int(open_lane.rsplit("_", 1)[1]) > int(lane.rsplit("_", 1)[1]) else -1.0
        # in front of his car when it stands right behind the wrecks (it shields him); else beside the last wreck's
        # back, where anyone still coming up the closed lane has stopped behind it
        pat = inc.get("police_at")
        shield = pv is not None and pat is not None and pat[0] == lane and rear - 16.0 < pat[1] <= rear
        side_s = "R" if side > 0 else "L"  # facing back down the road, an open lane to the traffic's left is on his right
        if shield:
            x, y, z, yaw = self._lane_point(lane, max(rear - POST_BEHIND_M, 1.0), side * (half - 0.75))
            if vehicles:  # a car stopped between his car and the wrecks (or a wreck in the junction): clear of it
                busy = self._bodies(vehicles, {inc.get("police")} | {w["vid"] for w in inc["wrecks"] if w["removed"]})
                for back in (0.0, 1.5, 3.0, -1.5, 4.5):
                    s_ = max(rear - POST_BEHIND_M - back, 1.0)
                    px, py, pz, _ = self._lane_point(lane, s_, side * (half - 0.75))
                    if not any(_inside(px, py, b, 0.6) for b in busy):
                        x, y, z = px, py, pz
                        break
            return x, y, z, (yaw + 180.0) % 360.0, side_s
        # else just off the last wreck's back corner on the open lane's side (as it lies, turned by the crash): the
        # open lane's drivers keep a metre from the wreck, anyone still on the closed lane stops behind it
        last = min(on, key=lambda w: w["pos"])
        wx, wy, wz, wyaw = self._wreck_pose(last)
        hl, hw = last["length"] / 2.0, self._half_width(last.get("model"))
        bx, by, _, lyaw = self._lane_point(lane, max(rear, 0.5), 0.0)
        ox, oy, _, _ = self._lane_point(lane, max(rear, 0.5), side * 1.0)
        u = (ox - bx, oy - by)  # 1 m towards the open lane
        fwd = (math.cos(math.radians(lyaw)), math.sin(math.radians(lyaw)))
        corners = [_local(wx, wy, wyaw, a, b) for a in (-hl, hl) for b in (-hw, hw)]
        back = sorted(corners, key=lambda p: (p[0] - bx) * fwd[0] + (p[1] - by) * fwd[1])[:2]
        cx, cy = max(back, key=lambda p: (p[0] - bx) * u[0] + (p[1] - by) * u[1])
        # 0.6 m out from it (the open lane's drivers keep a metre from the wreck) and 1.5 m back (it swings straight
        # when winched)
        x, y = cx + u[0] * 0.6 - fwd[0] * 1.5, cy + u[1] * 0.6 - fwd[1] * 1.5
        # never out in the open lane: a wreck pushed over the lane line put him there, and the cars went through him
        # (seed 7, 24 Oct); then he stands further back, behind its corner
        lat = (x - bx) * u[0] + (y - by) * u[1]
        if lat > half - 0.4:
            k = lat - (half - 0.4)
            x, y = x - u[0] * k, y - u[1] * k
        # not where a car stands (one stopped right behind the wreck, pulling out): along the wreck's side instead
        # (seed 26, watched: the officer walked into the car behind the wreck)
        if vehicles:
            busy = self._bodies(vehicles, {last["vid"], inc.get("police")} | {w["vid"] for w in inc["wrecks"] if w["removed"]})
            for step in (0.0, 1.0, 2.0, 3.0, -1.5, 4.0):
                px, py = x + fwd[0] * step, y + fwd[1] * step
                if not any(_inside(px, py, b, 0.6) for b in busy):
                    x, y = px, py
                    break
        return x, y, wz, (lyaw + 180.0) % 360.0, side_s

    def _bodies(self, vehicles: dict, skip: set) -> list:
        """(CARLA centre x, y, yaw, half length, half width) of the vehicles near you but those in `skip`."""
        from cosim.mapping import front_to_centre, sumo_angle_to_carla_yaw

        out = []
        for vid, v in vehicles.items():
            if vid in skip:
                continue
            cx, cy = front_to_centre(v.x, v.y, v.angle, v.length / 2)
            ux, uy = self.frame.to_carla_xy(cx, cy)
            out.append((ux, uy, sumo_angle_to_carla_yaw(v.angle), v.length / 2, 1.0))
        return out

    def _officer_out(self, inc: dict, st: dict, pv, now: float, vehicles: dict | None = None) -> None:
        """The officer gets out of the police car and walks to his post; from a car held up further back he walks
        the whole way (the marker says POLICE ON SCENE only once he is there)."""
        wid = self.walkers.take("police")
        if wid is None:
            return
        post = self._post(inc, pv, vehicles)
        if post is None:
            self.walkers.give("police", wid)
            return
        pts = []
        if pv is not None:
            px, py, pz, pyaw = self._carla(pv.x, pv.y, pv.angle, pv.length)
            pts.append(_local(px, py, pyaw, 0.3, -1.45))  # beside the driver's door
            # round his car, not through it: along its side to a corner, then to his post
            fx, fy = post[0] - px, post[1] - py
            ahead = fx * math.cos(math.radians(pyaw)) + fy * math.sin(math.radians(pyaw)) > 0
            right = -fx * math.sin(math.radians(pyaw)) + fy * math.cos(math.radians(pyaw)) > 0
            if right:  # his post is on the car's other side: round its front or its back
                end = (pv.length / 2 + 1.0) * (1 if ahead else -1)
                pts += [_local(px, py, pyaw, end, -1.45), _local(px, py, pyaw, end, 1.45)]
            pa = self.cs.actors.get(inc["police"])
            if pa is not None:
                self._door(pa, "FL", now)
            t0 = now + 1.2
        else:  # no police car could get there: he comes on foot along the closed lane
            pts.append(_local(post[0], post[1], post[3], 40.0, 0.0))
            pz, t0 = post[2], now
        pts.append(post[:2])
        keys, t = [], t0
        for i, p in enumerate(pts):
            if i:
                t += max(math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) / WALK_MS, 0.6)
            nxt = pts[i + 1] if i + 1 < len(pts) else p
            yaw = math.degrees(math.atan2(nxt[1] - p[1], nxt[0] - p[0])) if nxt != p else post[3]
            keys.append((t, p[0], p[1], pz if i < len(pts) - 1 else post[2], yaw))
        t1 = t
        keys.append((t1 + 0.6, post[0], post[1], post[2], post[3]))
        start = pts[0]
        self.later.append((t0 - 0.1, ("walk", wid, keys)))
        st["officer"], st["post"], st["face"], st["side"] = wid, post[:3], post[3], post[4]
        st["at_post"] = t1 + 0.6
        st["officer_path"] = [(k[1], k[2], k[3]) for k in keys]
        st["officer_from"] = (start[0], start[1], pz)

    def _worker(self, inc: dict, st: dict, t: dict, now: float) -> None:
        """The truck's crew, by the loading's timeline (cosim/towing.py): out of the cab when the truck is in place, to
        the deck's controls behind the cab, to the wreck's front to hook the winch, back to the controls, along the
        loaded wreck to strap it, back into the cab. On the side away from the traffic."""
        from cosim import towing

        wid = self.walkers.take("worker")
        if wid is None:
            return
        lx, ly, lz, lyaw = self._carla(*t["load"])
        tl = t["tl"]
        side = 1.0  # the side away from the open lane: right of the truck when the open lane is on its left
        closed, open_lane = inc.get("closed") or [], inc.get("open_lane")
        if closed and open_lane:
            side = 1.0 if int(open_lane.rsplit("_", 1)[1]) > int(closed[0].rsplit("_", 1)[1]) else -1.0
        ex, _ = towing.deck_end(towing.deck_pose(1.0, 1.0))

        def at(fx: float, fy: float):
            x, y = _local(lx, ly, lyaw, fx, fy * side)
            return x, y, lz

        t0 = now
        cab, lever = at(1.9, 1.75), at(-1.2, 1.95)
        hook, strap = at(ex - 0.6, 1.45), at(-2.4, 1.95)
        face_in = lyaw - 90.0 * side
        keys = [(t0 + tl["reverse"] + 0.5, *cab, lyaw), (t0 + tl["reverse"] + 3.0, *lever, face_in),
                (t0 + tl["tilt"], *lever, face_in), (t0 + tl["tilt"] + 4.0, *hook, lyaw),
                (t0 + tl["align"] + 0.5, *hook, lyaw), (t0 + tl["align"] + 3.5, *lever, face_in),
                (t0 + tl["slide_back"], *lever, face_in), (t0 + tl["slide_back"] + 1.5, *strap, face_in),
                (t0 + tl["strap"], *strap, face_in), (t0 + tl["strap"] + 2.0, *cab, lyaw)]
        # parked out of sight until he gets out of the cab, and put away again once back in it
        self.later.append((keys[0][0] - 0.1, ("walk", wid, keys)))
        st["workers"][t["vid"]] = wid
        self.pending_release.append((t0 + tl["strap"] + 2.5, wid, "worker"))

    # ------------------------------------------------------------------ one step of one incident
    def _step(self, inc: dict, st: dict, now: float, vehicles: dict) -> None:
        phase = inc["phase"]
        self._drivers_out(inc, st, now)
        police = inc.get("police")
        pv = vehicles.get(police) if police else None
        if phase in ("secured", "recovery") and st["officer"] is None and not st.get("officer_done") and \
                (pv is None or pv.speed < 0.05):  # once the police car stands
            self._officer_out(inc, st, pv, now, vehicles)
        if st["officer"] is not None and st["at_post"] is not None and now >= st["at_post"]:
            if not st["cones"]:
                self._cones(inc, st)
            # waving the traffic past; the raised palm while it is held (people crossing, a truck backing in)
            mode = "stop" if inc.get("holding") else "pass"
            if st.get("gesture") != mode:
                self.send(("gesture", st["officer"], st["side"], mode))
                st["gesture"] = mode
        # recovery trucks: backing in, the deck, the wreck winched up, carried off
        wrecks = {w["vid"]: w for w in inc["wrecks"]}
        for t in inc["tows"]:
            wv = t["wreck"]
            if t["state"] in ("backing", "loading", "loaded", "leaving") and wv not in st["towing"] and t["load"]:
                wa, ta = self.cs.actors.get(wv), self.cs.actors.get(t["vid"])
                w = wrecks.get(wv)
                if wa is None or ta is None or w is None:
                    continue
                x, y, z, yaw = self._carla(*t["load"])
                # on SUMO's clock: the loading started t_load (SUMO time), the truck is held for its timeline
                t0 = now - max(inc.get("now", t["t_load"]) - t["t_load"], 0.0)
                self.send(("tow", wa, ta, {"t0": t0, "load": (x, y, z, yaw), "car_len": w["length"], "tl": t["tl"],
                                           "truck_vid": t["vid"]}))
                st["towing"][wv] = wa
                self.taken.add(wv)
                key = self.cs.vclass_of.get(wv, "passenger")
                self.cs.actors.pop(wv, None)       # the truck has it now: never parked by the traffic side
                self.cs.lights.pop(wv, None)
                self.carried[wa] = (t["vid"], key)
                self.dmg.pop(wv, None)
                if t["vid"] not in st["workers"]:
                    self._worker(inc, st, t, t0)
        if phase == "clearing" and st.get("phase") != "clearing":
            if st["cones"]:
                self.send(("props_clear", f"inc{inc['id']}_cones"))
                st["cones"] = False
            self._drivers_leave(st, now)
            if st["officer"] is not None:
                self.send(("gesture_stop", st["officer"]))
                path = list(reversed(st.get("officer_path") or []))
                if len(path) >= 2:  # back the way he came, round his car
                    keys, t = [], now
                    for i, p in enumerate(path):
                        if i:
                            t += max(math.hypot(p[0] - path[i - 1][0], p[1] - path[i - 1][1]) / WALK_MS, 0.6)
                        nxt = path[i + 1] if i + 1 < len(path) else p
                        yaw = math.degrees(math.atan2(nxt[1] - p[1], nxt[0] - p[0])) if nxt != p else st["face"]
                        keys.append((t, p[0], p[1], p[2], yaw))
                    self.send(("walk", st["officer"], keys))
                    st["officer_back"] = t + 0.5
                else:
                    st["officer_back"] = now
        if st.get("officer_back") and now >= st["officer_back"]:
            self.send(("release", [st["officer"]], "w"))
            self.walkers.give("police", st["officer"])
            pa = self.cs.actors.get(police) if police else None
            if pa is not None:
                self._door(pa, "FL", now, DOOR_OPEN_S * 0.6)  # he gets in
            st["officer"], st["officer_back"], st["officer_done"] = None, None, True
        st["phase"] = phase

    def _cones(self, inc: dict, st: dict) -> None:
        """Cones closing the wrecks' lane: a taper from the road's edge to the open lane, then a line along it past the
        wrecks, on the lane line; the warning sign further back. Not ahead of the wrecks: the recovery truck backs in
        there from the open lane."""
        closed = inc.get("closed") or []
        live = [w for w in inc["wrecks"] if not w["removed"] and w["lane"] and not w["lane"].startswith(":")]
        if not closed or not live:
            st["cones"] = True  # nothing to cone off (a junction): the police car's beacons do
            return
        lane = closed[0]
        on = [w for w in live if w["lane"] == lane] or live
        rear = min(w["pos"] - w["length"] for w in on)
        front = max(w["pos"] for w in on)
        ln = self.net.getLane(lane)
        half = ln.getWidth() / 2.0
        open_lane = inc.get("open_lane")
        side = 1.0
        if open_lane:
            side = 1.0 if int(open_lane.rsplit("_", 1)[1]) > int(lane.rsplit("_", 1)[1]) else -1.0
        props = []
        taper = [(rear - 34.0 + 4.0 * i, (-half + 0.4 + (2 * half - 0.9) * i / 5.0) * side) for i in range(6)]
        # the line 0.5 m inside the closed lane: the cars passing on the open lane keep their usual distance to it
        line = [(rear - 10.0 + 4.5 * i, (half - 0.5) * side) for i in range(int((front - rear + 10.5) / 4.5) + 1)]
        first = block_at(rear) - 1.0  # none before where SUMO shuts the lane (a short road: cars move over there)
        for s, lat in taper + line:
            if max(0.5, first) <= s <= ln.getLength() - 0.5:
                x, y, z, yaw = self._lane_point(lane, s, lat)
                props.append(("static.prop.constructioncone", x, y, z, yaw))
        if rear - 52.0 > 1.0:
            x, y, z, yaw = self._lane_point(lane, rear - 52.0, -side * (half - 0.6))
            props.append(("static.prop.warningaccident", x, y, z, yaw + 180.0))
        self.send(("props", f"inc{inc['id']}_cones", props))
        st["cones"] = True

    def carrying(self, truck_vid: str) -> bool:
        """The truck has a wreck on its deck."""
        return any(tv == truck_vid for tv, _ in self.carried.values())

    def unload(self, truck_vid: str) -> int:
        """The truck's wreck is taken off at the recovery yard (the viewer calls it out of sight): it goes back to the
        pool, the truck is free for the next accident."""
        n = 0
        for wa, (tv, key) in list(self.carried.items()):
            if tv == truck_vid:
                self.send(("release", [wa], "v"))
                self.cs.idle.setdefault(key, []).append(wa)
                self.carried.pop(wa)
                n += 1
        return n

    def _release_carried(self) -> None:
        """A carried wreck goes back to the pool when its truck is no longer drawn (it left), and walkers that got into
        a cab or walked off are put away."""
        now = time.time()
        for wa, (truck_vid, key) in list(self.carried.items()):
            if truck_vid not in self.cs.actors:
                self.send(("release", [wa], "v"))
                self.cs.idle.setdefault(key, []).append(wa)
                self.carried.pop(wa)
        keep = []
        for t, wid, role in self.pending_release:
            if now >= t:
                self.send(("release", [wid], "w"))
                self.walkers.give(role, wid)
            else:
                keep.append((t, wid, role))
        self.pending_release = keep

    def _finish(self, iid: int) -> None:
        st = self.state.pop(iid)
        self.send(("props_clear", f"inc{iid}_debris"))
        if st.get("cones"):
            self.send(("props_clear", f"inc{iid}_cones"))
        if st.get("officer") is not None:
            self.send(("gesture_stop", st["officer"]))
            self.send(("release", [st["officer"]], "w"))
            self.walkers.give("police", st["officer"])
        for wid in st["drivers"].values():
            if wid is not None and not any(w == wid for _, w, _ in self.pending_release):
                self.send(("release", [wid], "w"))
                self.walkers.give("civil", wid)

    def status(self, inc: dict, st: dict, now: float) -> str:
        phase = inc["phase"]
        if phase == "crash":
            return "BREAKDOWN" if inc["kind"] == "breakdown" else "ACCIDENT"
        if phase == "police en route":
            return "POLICE ON THE WAY"
        if phase in ("clearing", "closed"):
            return "CLEARING"
        tows = inc["tows"]
        if any(t["state"] in ("backing", "loading", "loaded") for t in tows):
            return "LOADING THE WRECK"
        if st["at_post"] is None or now < st["at_post"]:
            return "POLICE ARRIVING"
        if any(t["state"] in ("en route", "there") for t in tows):
            return "TOW TRUCK ON THE WAY"
        return "POLICE ON SCENE"

    def _marker(self, inc: dict, st: dict, now: float) -> dict:
        live = [w for w in inc["wrecks"] if not w["removed"]] or inc["wrecks"]
        pts = [self._carla(w["x"], w["y"], w["angle"], w["length"]) for w in live]
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        z = max(p[2] for p in pts)
        status = self.status(inc, st, now)
        aids = [self.cs.actors.get(w["vid"]) or st["towing"].get(w["vid"]) for w in inc["wrecks"]]
        return {"id": inc["id"], "x": x, "y": y, "z": z + 1.8, "status": status, "label": inc["label"],
                "kind": inc["kind"], "kmh": round(inc["severity"] * 3.6), "cars": len(inc["wrecks"]),
                "vids": [w["vid"] for w in inc["wrecks"]], "plate_ids": [a for a in aids if a is not None],
                "phase": STATUS_COLOUR.get(status, "crash"), "sumo_phase": inc["phase"]}

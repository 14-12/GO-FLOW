"""The free-fly viewer's vehicles and pedestrians, posed once per CARLA picture, in a process of its own
(`--poses frame`, the default since 22 Oct).

The viewer worked out the poses at its own rate, 60 a second at best, while CARLA draws 50-120 pictures
a second on its own clock: some pictures showed a vehicle in a new place, others in the same place again,
and the motion juddered. In the viewer's process the poses also waited for Python's lock behind the SUMO
thread and the overlay (measured: 41-44 a second instead of 60).

Here, as the free-fly camera does (cosim/fly_proc.py): CARLA calls this process once per picture (tick
callback); the poses sent are those of the moment the *next* picture is shown (now + one picture's time),
since the server applies a pose a picture after the one it was worked out from. The viewer sends one
`segment` per SUMO step (cosim.freefly.FrameFeeder), 0.2 s before it starts: when it starts, the vehicles'
curve points and heights, the pedestrians' start and end, and the actors to park. It takes over at its
start, to the picture (sent at its start, the last pictures of every step were held: 15-18 a second). This
process is then the only one moving those actors (parking included), so nothing can pull them back. An
actor that has not moved since the last picture is not sent again (cars at a red light, people waiting).

First run on the GTX 1650 laptop (scripted 60 s flight, tools/freefly_demo_compare.bat): judder 61 % of
moving-vehicle pictures with the viewer's thread, 16 % per picture; but CARLA 60 -> 44 FPS. Hence:
normal priority (it was above normal) and unchanged actors skipped. Fair second run (both from 12:00,
same flight): judder 84 % -> 6.9 % (1-4 % once SUMO keeps up), CARLA 53.6 -> 47.5 FPS; hence distant
vehicles every 2nd / 4th picture (NEAR_M, FAR_M).

It also measures what CARLA drew: from each picture's snapshot, how far every moving vehicle went since
the picture before. A step that is a repeat (no move) or a double of the one before reads as a judder;
`judder_share` is the share of moving vehicle-pictures where the speed changed by over half from one
picture to the next (a real car changes it by about 0.05 m/s a picture), for vehicles within NEAR_M of
the camera. `--poses thread` runs the same
measurement only (`measure_only`), for comparison.
"""

from __future__ import annotations

import collections
import math
import time
import typing

from cosim.interp import Timed, Trails
from cosim.mapping import Frame

PARK = (0.0, 0.0, -1000.0)
JUDDER_MIN_SPEED = 1.5   # m/s: slower vehicles are not counted (a creeping car moves a few mm a picture)
JUDDER_SHARE = 0.5       # a speed change over half of the speed, from one picture to the next
# Far from the camera a vehicle moves under a pixel a picture: it is sent every 2nd picture beyond NEAR_M and
# every 4th beyond FAR_M (each on its own pictures, so the load stays even). The per-picture poses had cost
# CARLA 62.6 -> 53.0 FPS at street level (fair run, 12:00, same flight), most of them for distant vehicles.
NEAR_M, FAR_M = 120.0, 250.0
SHOW_AFTER = 2   # a batch sent at a picture's callback is in the picture after the next one (Lag, measured)
WAITING_MAX = 2  # batches sent but not shown yet: at this many, the next picture's is not sent (see run.tick)


PLATE_M = 55.0  # plates are posed with their car this close to the camera (beyond: a few pixels; parked)
# drivers (cosim/drivers.py): the pool's walkers seated in the cars nearest the camera within DRIVER_M; a car keeps
# its driver until it is DRIVER_KEEP_M away (no swapping at the edge)
DRIVER_M, DRIVER_KEEP_M = 45.0, 52.0


def _ease(k: float) -> float:
    k = min(max(k, 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


def damage_share(dmg: dict, t_show: float | None, t_sim: float | None = None) -> float:
    """How far into the slide and spin of the impact a wreck is (0..1, eased over dmg["ramp_s"]): from the hit itself
    (dmg["t_hit"], SUMO time) at the SUMO time the picture shows (t_sim). 3 Nov: before, from when the scene first saw
    the crash (dmg["t0"], wall clock), 0.7 s before the cars were drawn touching: the struck car was pushed and turned
    before it was hit (tools/measure_impact.py: done before the contact for 24 of 64 wrecks). That is kept only for a
    damage without t_hit."""
    ramp = max(dmg.get("ramp_s", 0.4), 0.05)
    if t_sim is not None and dmg.get("t_hit") is not None:
        return _ease((t_sim - dmg["t_hit"]) / ramp)
    if t_show is not None and dmg.get("t0") is not None:
        return _ease((t_show - dmg["t0"]) / ramp)
    return 1.0


def lay_wreck(tf, dmg: dict, t_show: float | None, t_sim: float | None = None):
    """A wreck as it came to rest: turned, tilted, pushed, sunk (dmg from cosim/incidents.IncidentManager._damage),
    reached in dmg["ramp_s"] after the hit (damage_share): the slide and spin of the impact."""
    import carla

    k = damage_share(dmg, t_show, t_sim)
    yaw = tf.rotation.yaw + dmg.get("yaw", 0.0) * k
    f = tf.get_forward_vector()
    dx = dmg.get("dx", 0.0) * k
    loc = carla.Location(tf.location.x + f.x * dx, tf.location.y + f.y * dx, tf.location.z + dmg.get("dz", 0.0) * k)
    roll = dmg.get("roll", 0.0) * k
    if dmg.get("rollover"):
        roll = 88.0 * k * (1 if dmg.get("roll", 0.0) >= 0 else -1)
        loc.z += 0.85 * k  # on its side: the body's half width above the road
    return carla.Transform(loc, carla.Rotation(tf.rotation.pitch + dmg.get("pitch", 0.0) * k, yaw, roll))


def hit_fraction(dmg: dict | None, t_end: float | None, dt: float) -> float | None:
    """The fraction of the step (ending at SUMO time t_end) at which a wreck hit (dmg["t_hit"], 3 Nov), if it hit in
    that step; else None."""
    if not dmg or t_end is None or dmg.get("t_hit") is None:
        return None
    f = (dmg["t_hit"] - (t_end - dt)) / dt
    if -1e-6 <= f <= 1.0 + 1e-6:
        return min(max(f, 0.0), 1.0)
    return None


def _rot(p: float, y: float, r: float):
    """Unreal's rotation matrix (FRotationMatrix): rows are the rotated X, Y, Z axes."""
    cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
    cy, sy = math.cos(math.radians(y)), math.sin(math.radians(y))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    return ((cp * cy, cp * sy, sp),
            (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp),
            (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp))


def compose(parent, rel: tuple):
    """parent (carla.Transform) * rel (x, y, z, pitch, yaw[, roll]) -> carla.Transform: a part fixed to a body."""
    import carla

    x, y, z, p, yw = rel[:5]
    r = rel[5] if len(rel) > 5 else 0.0
    loc = parent.transform(carla.Location(x, y, z))
    a = _rot(p, yw, r)
    b = _rot(parent.rotation.pitch, parent.rotation.yaw, parent.rotation.roll)
    m = [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, m[0][2]))))
    if abs(m[0][2]) > 0.99999:  # X straight up or down (a deck mesh built standing): only yaw - roll is defined
        yaw = 0.0
        roll = math.degrees(math.atan2(m[1][0], m[1][1])) if m[0][2] > 0 else math.degrees(math.atan2(-m[1][0], m[1][1]))
    else:
        yaw = math.degrees(math.atan2(m[0][1], m[0][0]))
        roll = math.degrees(math.atan2(-m[1][2], m[2][2]))
    return carla.Transform(carla.Location(loc.x, loc.y, loc.z), carla.Rotation(pitch, yaw, roll))


def deck_tf(truck_tf, rel: tuple):
    """The recovery truck's deck prop for a truck posed truck_tf, the deck at rel (cosim/towing.py)."""
    from cosim import towing

    return compose(compose(truck_tf, rel), towing.MESH_ROT)


def as_rel(tf) -> tuple:
    """A carla.Transform as a rel tuple (x, y, z, pitch, yaw, roll) for compose()."""
    return (tf.location.x, tf.location.y, tf.location.z, tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll)


def blend(a, b, k: float):
    """carla.Transform a -> b, k 0..1 (eased), rotations the short way round."""
    import carla

    e = _ease(k)

    def ang(u, v):
        return u + ((v - u + 180.0) % 360.0 - 180.0) * e

    la, lb = a.location, b.location
    return carla.Transform(carla.Location(la.x + (lb.x - la.x) * e, la.y + (lb.y - la.y) * e, la.z + (lb.z - la.z) * e),
                           carla.Rotation(ang(a.rotation.pitch, b.rotation.pitch), ang(a.rotation.yaw, b.rotation.yaw),
                                          ang(a.rotation.roll, b.rotation.roll)))


class TowPlan:
    """A wreck loaded onto a recovery truck's deck (cosim/towing.py), picture by picture: the truck backing in, the
    deck sliding and tilting, the wreck dragged straight and winched up, the deck back level; then carried."""

    def __init__(self, wa: int, ta: int, plan: dict, truck_tf, car_tf):
        import carla

        from cosim import towing

        self.wa, self.ta = wa, ta
        self.truck_vid = plan.get("truck_vid")
        self.t0, self.car_len = plan["t0"], plan["car_len"]
        self.tl = plan["tl"]
        x, y, z, yaw = plan["load"]
        self.load = carla.Transform(carla.Location(x, y, z), carla.Rotation(0.0, yaw, 0.0))
        self.truck0 = truck_tf
        self.car0 = car_tf
        self.path = towing.Path()
        self.s0, self.s1 = self.path.s_start(self.car_len), self.path.s_end(self.car_len)
        # loaded front first; a wreck turned round (more than 90 deg from the truck) goes up backwards
        dy = 0.0 if car_tf is None else (car_tf.rotation.yaw - yaw + 180.0) % 360.0 - 180.0
        self.flip = abs(dy) > 90.0

    def _car_rel(self, x: float, z: float, pitch: float) -> tuple:
        return (x, 0.0, z, -pitch, 180.0, 0.0) if self.flip else (x, 0.0, z, pitch, 0.0, 0.0)

    def truck(self, t: float, sumo_tf):
        """The truck's pose at plan time t, or None when SUMO's (after the plan)."""
        import carla

        from cosim import towing

        name, k = towing.phase_at(self.tl, t)
        if name == "done":
            return None
        if name == "reverse":
            if self.truck0 is None:
                return self.load
            a, b = self.truck0, self.load
            x, y, z, yaw = towing.manoeuvre((a.location.x, a.location.y, a.location.z, a.rotation.yaw),
                                            (b.location.x, b.location.y, b.location.z, b.rotation.yaw), k)
            return carla.Transform(carla.Location(x, y, z), carla.Rotation(0.0, yaw, 0.0))
        if name == "rejoin" and sumo_tf is not None:
            a, b = self.load, sumo_tf
            x, y, z, yaw = towing.manoeuvre((a.location.x, a.location.y, a.location.z, a.rotation.yaw),
                                            (b.location.x, b.location.y, b.location.z, b.rotation.yaw), k)
            return carla.Transform(carla.Location(x, y, z), carla.Rotation(b.rotation.pitch * k, yaw, 0.0))
        return self.load

    def deck(self, t: float) -> tuple:
        from cosim import towing

        return towing.deck_at(self.tl, t)

    def car(self, t: float, truck_tf, deck_rel: tuple):
        """The wreck's pose at plan time t, on a truck posed truck_tf whose deck is at deck_rel."""
        from cosim import towing

        name, k = towing.phase_at(self.tl, t)
        if name in ("reverse", "slide", "tilt"):
            return self.car0 if self.car0 is not None else compose(self.load, self._car_rel(*self.path.car(self.s0, self.car_len)))
        if name == "align":
            aligned = compose(self.load, self._car_rel(*self.path.car(self.s0, self.car_len)))
            return aligned if self.car0 is None else blend(self.car0, aligned, k)
        if name == "winch":
            s = self.s0 + (self.s1 - self.s0) * k
            return compose(self.load, self._car_rel(*self.path.car(s, self.car_len)))
        on = (towing.CAR_X, 0.0, towing.DECK_T, 0.0, 180.0, 0.0) if self.flip else (towing.CAR_X, 0.0, towing.DECK_T, 0.0, 0.0)
        return compose(compose(truck_tf, deck_rel), on)


def walker_at(keys: list, t: float):
    """(x, y, z, yaw, speed) of a walker on its keyframes [(t, x, y, z, yaw)] at time t (held at the ends)."""
    if t <= keys[0][0]:
        k = keys[0]
        return k[1], k[2], k[3], k[4], 0.0
    for a, b in zip(keys, keys[1:], strict=False):
        if a[0] <= t <= b[0]:
            f = (t - a[0]) / max(b[0] - a[0], 1e-6)
            d = math.hypot(b[1] - a[1], b[2] - a[2])
            sp = d / max(b[0] - a[0], 1e-6)
            yaw = b[4] if d < 0.05 else math.degrees(math.atan2(b[2] - a[2], b[1] - a[1]))
            return a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, yaw, sp
    k = keys[-1]
    return k[1], k[2], k[3], k[4], 0.0


class FramePoser:
    """Poses at any moment of a SUMO step, from a viewer's segment (no CARLA needed but its Transform)."""

    def __init__(self, cx: float, cy: float):
        self.frame = Frame(cx, cy)
        self.trails = Trails()
        self.seg: dict | None = None
        self._timed: dict = {}
        self.frozen_s: float | None = None
        self.held = 0  # pictures posed at the end of a step because the next step had not come yet

    def set_segment(self, seg: dict) -> None:
        self.seg = seg
        self._timed = {}
        keep = {v[1] for v in seg["veh"]}
        if len(self.trails.paths) > 2 * len(keep) + 50:
            self.trails.forget(keep)

    def s_at(self, t: float) -> float:
        seg = self.seg
        if self.frozen_s is not None:
            return self.frozen_s
        s = (t - seg["start"]) / seg["dt"]
        if s > 1.0:
            self.held += 1
        return min(max(s, 0.0), 1.0)

    def vehicles(self, s: float, t_show: float | None = None) -> list:
        """[(actor id, carla.Transform)]. Each entry of the segment: (actor, vid, p0, p1, p2, p3, length, angle, z0,
        z1, pitch0, pitch1[, lane angle 0, lane angle 1, speed 0, speed 1, damage]); the lane angles keep the body
        within its lane's direction (interp.Trails), the damage lays a wreck as it came to rest (lay_wreck)."""
        from cosim.convert import vehicle_transform

        out = []
        t_end, dt = self.seg.get("t"), self.seg.get("dt") or 0.5
        t_sim = None if t_end is None else t_end - (1.0 - s) * dt  # the SUMO time this picture shows
        for e in self.seg["veh"]:
            aid, vid, p0, p1, p2, p3, length, angle, z0, z1, pt0, pt1 = e[:12]
            la0, la1, sp0, sp1, dmg = e[12:17] if len(e) >= 17 else (None, None, None, None, None)
            hit = hit_fraction(dmg, t_end, dt)
            if p1 is None:  # no step before: where it is
                x, y = p2
            else:
                tm = self._timed.get(vid)
                if tm is None:
                    tm = self._timed[vid] = Timed(p0, p1, p2, p3, stop_at=hit)
                x, y = tm.at(s)
            la = None
            if la1 is not None:
                la = la1 if la0 is None else la0 + ((la1 - la0 + 180.0) % 360.0 - 180.0) * s
            sp = None if sp1 is None else (sp1 if sp0 is None else sp0 + (sp1 - sp0) * s)
            fixed = e[17] if len(e) >= 18 else None
            if fixed is not None:     # (8 Nov) driven into or out of a space: the heading given (cosim/parking_motion)
                ang = self.trails.fixed(vid, x, y, length, fixed[0] + ((fixed[1] - fixed[0] + 180.0) % 360.0 - 180.0) * s)
            else:
                ang = self.trails.update(vid, x, y, length, angle, la, sp)
            if dmg and (hit is None or s >= hit):
                # a wreck lies along its lane as SUMO has it (cosim/incidents.py centres it), then as the crash left it;
                # (3 Nov) on its way to the hit, as it drove
                ang = angle
            tf = vehicle_transform(self.frame, x, y, ang, length, z0 + (z1 - z0) * s, pt0 + (pt1 - pt0) * s)
            if dmg:
                tf = lay_wreck(tf, dmg, t_show, t_sim)
            out.append((aid, tf))
        return out

    def walkers(self, s: float) -> list:
        """[(actor id, carla.Transform, speed)]"""
        import carla

        out = []
        for aid, a, b, speed in self.seg["walk"]:
            x = a[0] + (b[0] - a[0]) * s
            y = a[1] + (b[1] - a[1]) * s
            z = a[2] + (b[2] - a[2]) * s
            yaw = a[3] + ((b[3] - a[3] + 180.0) % 360.0 - 180.0) * s  # the short way round
            out.append((aid, carla.Transform(carla.Location(x, y, z), carla.Rotation(0.0, yaw, 0.0)), speed))
        return out


class Driver:
    """The pose process's state between CARLA's pictures (no CARLA needed but its Transform): segments sent
    ahead and taken over at their start, parks, actors whose pose has not changed."""

    def __init__(self, cx: float, cy: float):
        self.poser = FramePoser(cx, cy)
        self.pending: list[dict] = []
        self.parks: tuple[list, list] = ([], [])
        self.last_sent: dict[int, tuple] = {}
        self.lights: list[tuple[int, int]] = []  # (actor, light state) to send with the next picture's poses
        self.plates: dict[int, list] = {}         # car actor -> [(plate actor, (x, y, z, pitch, yaw))]
        self.plates_on: set[int] = set()          # cars whose plates are posed (near the camera)
        self.tows: dict[int, TowPlan] = {}        # wreck actor -> its loading onto a recovery truck, then carried
        self.decks: dict[int, int] = {}           # recovery truck actor -> its deck (a prop, cosim/towing.py)
        self.walks: dict[int, list] = {}          # scene walker -> keyframes [(t, x, y, z, yaw)]
        self.last_tf: dict[int, object] = {}      # each vehicle actor's last pose (a loading starts there)
        self.seats: dict[int, tuple] = {}         # car actor -> its driver's place (x, y, z, 0, 0, 0)
        self.drivers_free: list[int] = []         # driver walkers not seated
        self.driver_in: dict[int, int] = {}       # car actor -> its driver walker
        self.doors_open: set[int] = set()         # cars whose driver got out (a door open at an accident scene)
        self.eyes: list = []                      # other cameras (x, y): a red-light camera site (cosim/enforcement)
        self.follow: str | None = None            # (8 Nov) the SUMO vehicle the camera follows (F), or None
        self.follow_view = "chase"                # chase | roof | drone (V)
        self.vid_aid: dict[str, int] = {}         # this segment's SUMO vehicle -> its actor
        self.follow_tf = None                     # the followed vehicle's pose in the last picture
        self.follow_new = False                   # a new car to follow: the camera flies in from where it is

    def receive(self, msg) -> None:
        kind = msg[0]
        if kind == "seg":
            seg = msg[1]
            if (len(msg) > 2 and bool(msg[2])) or self.poser.seg is None:
                self.pending.clear()  # the first one: at once
                self.activate(seg)
            else:
                self.pending.append(seg)
        elif kind == "plates":
            self.plates.update(msg[1])
        elif kind == "eyes":  # (31 Oct) [(x, y)]: plates and every picture's poses near these cameras too
            self.eyes = [tuple(e) for e in msg[1]]
        elif kind == "tow":  # (wreck actor, truck actor, plan): cosim/incident_scene.py
            _, wa, ta, plan = msg
            self.tows[wa] = TowPlan(wa, ta, plan, self.last_tf.get(ta), self.last_tf.get(wa))
        elif kind == "deck":  # (truck actor, deck prop): the deck goes with the truck from now on
            self.decks[msg[1]] = msg[2]
        elif kind == "deck_off":  # (truck actor,): its deck put away
            d = self.decks.pop(msg[1], None)
            if d is not None:
                self.parks = (self.parks[0] + [d], self.parks[1])
                self.last_sent.pop(d, None)
        elif kind == "release":  # a carried wreck (or a scene walker) put away
            for aid in msg[1]:
                self.tows.pop(aid, None)
                self.walks.pop(aid, None)
                self.last_sent.pop(aid, None)
                self.parks = (self.parks[0] + [aid], self.parks[1]) if msg[2] == "v" else (self.parks[0], self.parks[1] + [aid])
        elif kind == "walk":  # a scene walker's path (keyframes, wall clock)
            self.walks[msg[1]] = msg[2]
        elif kind == "seats":  # {car actor: driver's place}: cosim/drivers.py
            self.seats.update(msg[1])
        elif kind == "drivers":  # the drivers' walkers (posed sitting)
            self.drivers_free += [w for w in msg[1] if w not in self.drivers_free]
        elif kind == "door":  # (car actor, open?, ...): the one at the wheel got out, or back in
            (self.doors_open.add if msg[2] else self.doors_open.discard)(msg[1])
        elif kind == "follow":  # (8 Nov) (vid or None[, view]): the camera follows this vehicle (cosim/route_guidance)
            self.follow_new = msg[1] is not None and msg[1] != self.follow
            self.follow = msg[1]
            if len(msg) > 2 and msg[2]:
                self.follow_view = msg[2]
            self.follow_tf = None
        elif kind == "follow_view":
            self.follow_view = msg[1]
        elif kind == "pause":  # the viewer sends the segments again on resume; keep their parks
            for seg in self.pending:
                self.parks = (self.parks[0] + list(seg.get("park_v", ())), self.parks[1] + list(seg.get("park_w", ())))
            self.pending.clear()
            self.poser.frozen_s = msg[1]

    def activate(self, seg: dict) -> None:
        """A segment's start has come: its poses from now on, and its parks."""
        moving = {v[0] for v in seg["veh"]}
        walking = {w[0] for w in seg["walk"]}
        # parks not sent yet are kept (unless the actor is in use again)
        pv = [a for a in list(seg.get("park_v", ())) + self.parks[0] if a not in moving]
        pw = [a for a in list(seg.get("park_w", ())) + self.parks[1] if a not in walking]
        self.parks = (pv, pw)
        for a in pv + pw:
            self.last_sent.pop(a, None)
        self.lights += list(seg.get("lights", ()))
        self.poser.frozen_s = None
        self.poser.set_segment(seg)
        self.vid_aid = {v[1]: v[0] for v in seg["veh"]}

    def _changed(self, aid, tf) -> bool:
        loc, rot = tf.location, tf.rotation
        key = (round(loc.x, 3), round(loc.y, 3), round(loc.z, 3), round(rot.yaw, 2), round(rot.pitch, 2))
        if self.last_sent.get(aid) == key:
            return False
        self.last_sent[aid] = key
        return True

    def seat_drivers(self, posed: list, vs: list, pv: list, eye: tuple[float, float]) -> tuple[list, list]:
        """The cars nearest the camera get a driver (a walker of the pool, posed sitting), posed with them: within
        DRIVER_M, nearest first, as many as there are walkers; a car keeps its own until DRIVER_KEEP_M. No driver
        in a wreck, a car parked at its depot (the segment's "empty"), or one whose driver got out (door open)."""
        ex, ey = eye
        empty = set(self.poser.seg.get("empty", ())) | self.doors_open | set(self.tows)
        d2 = {}
        tfs = {}
        for a, tf in posed:
            if a in self.seats and a not in empty:
                d2[a] = (tf.location.x - ex) ** 2 + (tf.location.y - ey) ** 2
                tfs[a] = tf
        keep2, near2 = DRIVER_KEEP_M * DRIVER_KEEP_M, DRIVER_M * DRIVER_M
        for car in [c for c in self.driver_in if d2.get(c, keep2 + 1.0) > keep2]:
            w = self.driver_in.pop(car)
            self.drivers_free.append(w)
            pv = pv + [w]
            self.last_sent.pop(w, None)
        if self.drivers_free:
            for car in sorted((c for c, dd in d2.items() if dd <= near2 and c not in self.driver_in), key=d2.get):
                if not self.drivers_free:
                    break
                self.driver_in[car] = self.drivers_free.pop()
        seated = set(self.driver_in.values())
        pv = [a for a in pv if a not in seated]  # let go and seated again in the same picture: not parked
        vs = vs + [(w, compose(tfs[car], self.seats[car])) for car, w in self.driver_in.items()]
        return vs, pv

    def picture(self, t_show: float, eye: tuple[float, float] | None = None, k: int = 0):
        """For the picture shown at t_show: (vehicles [(actor, tf)] that moved, walkers [(actor, tf, speed)]
        that moved, vehicles to park, walkers to park, all vehicles posed, how many were skipped). eye: the
        camera (x, y) in CARLA coordinates, k: the picture's number, for the distant vehicles' turns."""
        while self.pending and self.pending[0]["start"] <= t_show:
            self.activate(self.pending.pop(0))
        if self.poser.seg is None:
            return [], [], [], [], [], 0
        pv, pw = self.parks
        self.parks = ([], [])
        s = self.poser.s_at(t_show)
        vs = [(a, tf) for a, tf in self.poser.vehicles(s, t_show) if a not in self.tows]
        ws = self.poser.walkers(s)
        # recovery trucks: backing in to load (their pose is the plan's, not SUMO's), their decks, the wrecks on them
        if self.tows or self.decks:
            from cosim import towing

            at = {a: i for i, (a, _) in enumerate(vs)}
            deck_rel = {}
            for wa, plan in self.tows.items():
                t = t_show - plan.t0
                sumo_tf = vs[at[plan.ta]][1] if plan.ta in at else self.last_tf.get(plan.ta)
                tf = plan.truck(t, sumo_tf)
                if tf is not None and plan.truck_vid:
                    # while the plan poses the truck, SUMO's path for it (the jump into the closed lane) is dropped:
                    # once SUMO has it again its body starts afresh along SUMO's heading (interp.Trails)
                    self.poser.trails.paths.pop(plan.truck_vid, None)
                    self.poser.trails.heading.pop(plan.truck_vid, None)
                    self.poser.trails.last.pop(plan.truck_vid, None)
                if tf is not None:
                    if plan.ta in at:
                        vs[at[plan.ta]] = (plan.ta, tf)
                    else:
                        at[plan.ta] = len(vs)
                        vs.append((plan.ta, tf))
                truck_tf = tf if tf is not None else sumo_tf
                if truck_tf is None:
                    continue
                deck_rel[plan.ta] = plan.deck(t)
                vs.append((wa, plan.car(t, truck_tf, deck_rel[plan.ta])))
            for ta, d in self.decks.items():
                if ta in at:
                    vs.append((d, deck_tf(vs[at[ta]][1], deck_rel.get(ta, towing.deck_level()))))
        for a, tf in vs:
            self.last_tf[a] = tf
        self.follow_tf = None
        if self.follow is not None:
            fa = self.vid_aid.get(self.follow)
            if fa is not None:
                self.follow_tf = self.last_tf.get(fa)
        # people at a scene (officer, drivers, the truck's crew): along their paths
        if self.walks:
            import carla

            for wa, keys in self.walks.items():
                x, y, z, yaw, sp = walker_at(keys, t_show)
                ws.append((wa, carla.Transform(carla.Location(x, y, z), carla.Rotation(0.0, yaw, 0.0)), sp))
        posed = list(vs)
        # number plates with their cars, near the camera
        if self.plates:
            near2 = PLATE_M * PLATE_M
            shown = set()
            for a, tf in vs:
                pl = self.plates.get(a)
                if not pl:
                    continue
                if eye is None or any((tf.location.x - e[0]) ** 2 + (tf.location.y - e[1]) ** 2 <= near2
                                      for e in [eye] + self.eyes):
                    shown.add(a)
                    for pa, rel in pl:
                        vs.append((pa, compose(tf, rel)))
            for a in (self.plates_on - shown) | (set(pv) & self.plates_on):
                for pa, _ in self.plates.get(a, ()):
                    pv = pv + [pa]
                    self.last_sent.pop(pa, None)
            self.plates_on = shown
        if eye is not None and (self.driver_in or (self.drivers_free and self.seats)):
            vs, pv = self.seat_drivers(posed, vs, pv, eye)
        if eye is not None:
            ex, ey = eye
            near2, far2 = NEAR_M * NEAR_M, FAR_M * FAR_M

            eyes = self.eyes

            def due(a, tf):
                d2 = (tf.location.x - ex) ** 2 + (tf.location.y - ey) ** 2
                if d2 <= near2:
                    return True
                if any((tf.location.x - e[0]) ** 2 + (tf.location.y - e[1]) ** 2 <= near2 for e in eyes):
                    return True  # (a red-light camera's view: every picture)
                return (a + k) % (2 if d2 <= far2 else 4) == 0

            vs_send = [(a, tf) for a, tf in vs if due(a, tf)]
        else:
            vs_send = vs
        v_out = [(a, tf) for a, tf in vs_send if self._changed(a, tf)]
        w_out = [(a, tf, sp) for a, tf, sp in ws if self._changed(a, tf)]
        skipped = len(vs_send) + len(ws) - len(v_out) - len(w_out)
        return v_out, w_out, pv, pw, posed, skipped


class ChaseCam:
    """(8 Nov) The camera on a followed car (cosim/route_guidance.py, F in the viewer), worked out in this process for
    the picture the car's own pose is for, so the two never drift apart. Its place is fixed to the car; only its turn
    eases after the car's (a car's heading changes at a turn faster than a viewer's eye wants). Views (V):
    chase = behind and above; roof = on the roof, looking ahead; drone = high behind, looking down.

    Transitions: when the following starts, the camera flies from where the free-fly camera was to its place on the
    car (FLY_IN_S, eased in and out, along a raised arc so it does not cut through what is between); a change of view
    glides from the last view to the new one (SWITCH_S). Both aim at the moving car, so the car never leaves the
    picture on the way."""

    VIEWS = ("chase", "roof", "drone")
    EASE_S: typing.ClassVar[dict] = {"chase": 0.35, "roof": 0.12, "drone": 0.6}
    FLY_IN_S = 1.8
    SWITCH_S = 0.9

    def __init__(self):
        self.yaw: float | None = None
        self.from_tf = None      # where a transition started (a carla.Transform)
        self.t = 0.0             # seconds into it
        self.dur = 0.0
        self.last = None         # the last pose sent
        self.view = None

    def reset(self) -> None:
        self.yaw = None
        self.from_tf = None
        self.last = None
        self.view = None

    def begin(self, from_tf, dur: float) -> None:
        """A transition from from_tf (the camera now) to the view's place, over dur seconds."""
        self.from_tf, self.t, self.dur = from_tf, 0.0, dur

    @staticmethod
    def ease(k: float) -> float:
        k = min(max(k, 0.0), 1.0)
        return k * k * k * (k * (6.0 * k - 15.0) + 10.0)   # smootherstep: no jolt at either end

    @staticmethod
    def place(view: str, ext: tuple | None) -> tuple[float, float, float]:
        """(back, up, pitch) for a car of half extents ext (x half length, y half width, z half height)."""
        ex, _, ez = ext if ext else (2.3, 1.0, 0.75)
        if view == "roof":
            return -0.15 * ex, 2.0 * ez + 0.35, -5.0
        if view == "drone":
            return 22.0 + 2.0 * ex, 26.0, -48.0
        return 4.5 + 2.0 * ex, 1.6 + 2.0 * ez, -11.0

    def update(self, car_tf, dt: float, view: str = "chase", ext: tuple | None = None):
        import carla

        if self.view is not None and view != self.view and self.last is not None:
            self.begin(self.last, self.SWITCH_S)          # a new view: glide there from the last one
        self.view = view
        tf = self._place(car_tf, dt, view, ext)
        if self.from_tf is not None and self.dur > 0.0:
            self.t += max(dt, 0.0)
            k = self.ease(self.t / self.dur)
            if k >= 1.0:
                self.from_tf = None
            else:
                a, b = self.from_tf.location, tf.location
                lift = 4.0 * math.sin(math.pi * k) * min(1.0, math.dist((a.x, a.y), (b.x, b.y)) / 40.0)
                loc = carla.Location(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.z + (b.z - a.z) * k + lift)
                # look at the car all the way, blended into the view's own turn at the end
                c = car_tf.location
                dx, dy, dz = c.x - loc.x, c.y - loc.y, c.z + 1.0 - loc.z
                look_yaw = math.degrees(math.atan2(dy, dx))
                look_pitch = math.degrees(math.atan2(dz, max(math.hypot(dx, dy), 0.1)))
                w = self.ease(min(1.0, k / 0.35))           # from the camera's own turn to looking at the car
                y0, p0 = self.from_tf.rotation.yaw, self.from_tf.rotation.pitch
                yaw = y0 + ((look_yaw - y0 + 180.0) % 360.0 - 180.0) * w
                pitch = p0 + (look_pitch - p0) * w
                m = self.ease(max(0.0, (k - 0.7) / 0.3))     # the last 30 %: into the view's own turn
                yaw = yaw + ((tf.rotation.yaw - yaw + 180.0) % 360.0 - 180.0) * m
                pitch = pitch + (tf.rotation.pitch - pitch) * m
                tf = carla.Transform(loc, carla.Rotation(pitch, yaw, 0.0))
        self.last = tf
        return tf

    def _place(self, car_tf, dt: float, view: str, ext: tuple | None):
        import carla

        target = car_tf.rotation.yaw
        if self.yaw is None:
            self.yaw = target
        else:
            k = min(1.0, max(dt, 0.0) / self.EASE_S.get(view, 0.35))
            self.yaw += ((target - self.yaw + 180.0) % 360.0 - 180.0) * k
        back, up, pitch = self.place(view, ext)
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        loc = car_tf.location
        return carla.Transform(carla.Location(loc.x - back * cy, loc.y - back * sy, loc.z + up),
                               carla.Rotation(pitch, self.yaw, 0.0))


class CrashTrace:
    """(3 Nov) What CARLA drew of each wreck around its hit, picture by picture, from the server's own snapshot: where
    the actor stood (CARLA x, y, yaw) and the SUMO time the picture shows. From the hit's step (when the wreck's damage
    reaches the pose process, up to 0.5 s before the hit) to KEEP_S after it. Appended to out/freefly/crash_trace.csv;
    read by tools/check_crash_trace.py. A failure here never stops the posing."""

    KEEP_S = 4.0
    HEADER = "sim_s,t_hit,v_hit,vid,actor,x,y,yaw,wall_s\n"

    def __init__(self, path=None):
        from pathlib import Path

        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "out" / "freefly" / "crash_trace.csv"
        self.f = None
        self.rows = 0
        self.failed = False

    def picture(self, snap, seg, t_pic: float) -> int:
        if self.failed or seg is None or seg.get("t") is None:
            return 0
        try:
            dt = seg.get("dt") or 0.5
            t_sim = seg["t"] - dt + min(max(t_pic - seg["start"], 0.0), dt)
            out = []
            for e in seg["veh"]:
                dmg = e[16] if len(e) >= 17 else None
                if not dmg or dmg.get("t_hit") is None or t_sim - dmg["t_hit"] > self.KEEP_S:
                    continue
                a = snap.find(e[0])
                if a is None:
                    continue
                tf = a.get_transform()
                out.append(f"{t_sim:.3f},{dmg['t_hit']:.3f},{dmg.get('v_hit') or 0:.2f},{e[1]},{e[0]},"
                           f"{tf.location.x:.3f},{tf.location.y:.3f},{tf.rotation.yaw:.2f},{t_pic:.3f}\n")
            if out:
                if self.f is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    new = not self.path.exists()
                    self.f = self.path.open("a", encoding="utf-8")
                    if new:
                        self.f.write(self.HEADER)
                self.f.writelines(out)
                self.f.flush()
                self.rows += len(out)
            return len(out)
        except Exception:  # noqa: BLE001 - the trace is an aid; it stops, the posing goes on
            self.failed = True
            return 0


class Judder:
    """From each picture's snapshot: how far each vehicle moved since the picture before."""

    def __init__(self):
        self.last: dict[int, tuple[float, float, float | None]] = {}  # actor -> (x, y, speed)
        self.counted = 0
        self.judders = 0

    def picture(self, snap, actors, dt: float, eye: tuple[float, float] | None = None) -> None:
        """eye: the camera (x, y); only vehicles within NEAR_M of it are counted (those you see move; the
        distant ones are sent every 2nd or 4th picture on purpose)."""
        if dt <= 0:
            return
        last, nxt = self.last, {}
        near2 = NEAR_M * NEAR_M
        for aid in actors:
            a = snap.find(aid)
            if a is None:
                continue
            loc = a.get_transform().location
            if eye is not None and (loc.x - eye[0]) ** 2 + (loc.y - eye[1]) ** 2 > near2:
                continue
            prev = last.get(aid)
            speed = None
            if prev is not None:
                speed = math.hypot(loc.x - prev[0], loc.y - prev[1]) / dt
                if speed > 60.0:  # a jump: spawned, parked or reused this picture
                    speed = None
                elif prev[2] is not None and max(prev[2], speed) >= JUDDER_MIN_SPEED:
                    self.counted += 1
                    if abs(speed - prev[2]) > JUDDER_SHARE * max(prev[2], speed):
                        self.judders += 1
            nxt[aid] = (loc.x, loc.y, speed)
        self.last = nxt

    def take(self) -> tuple[int, int]:
        out = (self.counted, self.judders)
        self.counted = self.judders = 0
        return out


class FrameClock:
    """The wall-clock moment of each picture, from the server's own frame time (the snapshot's elapsed
    seconds) rather than from when its callback happens to run here: the callback came 0-15 ms late, and a pose
    timed by it moved unevenly by that much (half a picture). The offset between the two clocks is the
    smallest seen in the last `window_s` (a late callback only makes it larger), so the server clock's slow
    drift is followed."""

    def __init__(self, window_s: float = 3.0):
        self.window_s = window_s
        self._q: collections.deque = collections.deque()  # (game time, offset), offsets increasing

    def wall(self, game_t: float, wall_now: float) -> float:
        off = wall_now - game_t
        q = self._q
        while q and q[-1][1] >= off:
            q.pop()
        q.append((game_t, off))
        while q[0][0] < game_t - self.window_s:
            q.popleft()
        return game_t + q[0][1]


class Pacing:
    """How evenly the server makes its pictures: the share of pictures that took over 1.5 times as long as the
    one before, or under 1/1.5 (CARLA's own unevenness, which no pose timing can take out)."""

    def __init__(self):
        self.last = None
        self.n = self.uneven = 0

    def picture(self, dt: float) -> None:
        if self.last is not None and dt > 0 and self.last > 0:
            self.n += 1
            r = dt / self.last
            if r > 1.5 or r < 1 / 1.5:
                self.uneven += 1
        self.last = dt

    def take(self) -> float | None:
        out = round(self.uneven / self.n, 4) if self.n else None
        self.n = self.uneven = 0
        return out


class Lag:
    """Which of the last poses sent each picture shows: 1 = those sent at the picture before (right after it
    was made), 2 = those sent one picture earlier, ... A pose that arrives at the server after it has started
    the next picture shows a picture later; when that happens only sometimes, the picture repeats a pose and
    the next one jumps two: judder, though every pose was right for its moment. Each moving vehicle checked
    counts for the sent pose nearest to where the picture has it ("other": none within 5 cm), and the
    distance to it is kept (`take` gives its median, in cm)."""

    def __init__(self, keep: int = 12, per_picture: int = 12):
        self.sent: collections.deque = collections.deque(maxlen=keep)  # newest first: {actor: (x, y)}
        self.per_picture = per_picture
        self.hist: collections.Counter = collections.Counter()
        self.off: list[float] = []

    def picture(self, snap) -> int | None:
        """Counts this picture; returns how many of the batches sent are not shown yet (0: the last one sent
        is), the median over the moving vehicles checked (len(sent) when none of them matched), or None when
        there was nothing to tell by (no vehicle moving)."""
        if len(self.sent) < 2:
            return None
        newest, before = self.sent[0], self.sent[1]
        n = 0
        behind: list[int] = []
        for aid, (x, y) in newest.items():
            if n >= self.per_picture:
                break
            b = before.get(aid)
            if b is None or abs(b[0] - x) + abs(b[1] - y) < 0.02:  # not moving: any pose would match
                continue
            a = snap.find(aid)
            if a is None:
                continue
            loc = a.get_transform().location
            n += 1
            best, best_d = None, None
            for j, entry in enumerate(self.sent):
                p = entry.get(aid)
                if p is not None:
                    d = math.hypot(p[0] - loc.x, p[1] - loc.y)
                    if best_d is None or d < best_d:
                        best, best_d = j, d
            if best is None or best_d > 0.05:
                self.hist["other"] += 1
                behind.append(len(self.sent))
            else:
                self.hist[str(best + 1)] += 1
                behind.append(best)
            if best_d is not None:
                self.off.append(best_d)
        if not behind:
            return None
        behind.sort()
        return behind[len(behind) // 2]

    def sent_now(self, vs) -> None:
        self.sent.appendleft({aid: (tf.location.x, tf.location.y) for aid, tf in vs})

    def take(self) -> dict:
        total = sum(self.hist.values())
        out = {k: round(v / total, 3) for k, v in sorted(self.hist.items(), key=lambda kv: (len(kv[0]), kv[0]))} \
            if total else {}
        if self.off:
            off = sorted(self.off)
            out["median_cm"] = round(100 * off[len(off) // 2], 2)
        self.hist.clear()
        self.off = []
        return out


class SceneOps:
    """The server calls an accident scene needs (cosim/incident_scene.py), made by this process so the viewer never
    waits on them: props spawned and destroyed (cones, the warning sign, debris), car doors, and the officer's
    gestures (his arm bones posed: the "come on, pass on the left" wave and the raised "stop" palm; bone rotations
    measured with tools/probe_incident_assets.py: the upper arm's yaw swings the arm out sideways, its pitch -80
    brings it forward). A failure is logged and the scene goes on without that part."""

    KINDS = ("props", "props_clear", "door", "gesture", "gesture_stop")
    WAVE_S = 0.55

    def __init__(self, client, world):
        self.client, self.world = client, world
        self.props: dict[str, list[int]] = {}
        self.gestures: dict[int, dict] = {}
        self.errors: list[str] = []

    def handle(self, msg) -> None:
        import carla

        kind = msg[0]
        try:
            if kind == "props":  # (key, [(blueprint, x, y, z, yaw)])
                lib = self.world.get_blueprint_library()
                batch = []
                for bp_id, x, y, z, yaw in msg[2]:
                    bp = lib.find(bp_id)
                    batch.append(carla.command.SpawnActor(bp, carla.Transform(carla.Location(x, y, z),
                                                                               carla.Rotation(0.0, yaw, 0.0))))
                res = self.client.apply_batch_sync(batch, False)
                self.props.setdefault(msg[1], []).extend(r.actor_id for r in res if not r.error)
            elif kind == "props_clear":
                ids = self.props.pop(msg[1], [])
                if ids:
                    self.client.apply_batch([carla.command.DestroyActor(a) for a in ids])
            elif kind == "door":  # (actor, open?[, "FL" | "FR" | "RL" | "RR"])
                a = self.world.get_actor(msg[1])
                if a is not None:
                    door = getattr(carla.VehicleDoor, msg[3] if len(msg) > 3 else "FL")
                    (a.open_door if msg[2] else a.close_door)(door)
            elif kind == "gesture":  # (walker, side "L"/"R" of the open lane, mode "pass"/"stop")
                g = self.gestures.get(msg[1])
                if g is None:
                    w = self.world.get_actor(msg[1])
                    if w is None:
                        return
                    rel = {b.name: b.relative for b in w.get_bones().bone_transforms
                           if b.name in ("crl_arm__R", "crl_arm__L")}
                    g = self.gestures[msg[1]] = {"walker": w, "rel": rel, "t": 0.0, "phase": 0}
                g["side"], g["mode"] = msg[2], msg[3]
            elif kind == "gesture_stop":
                g = self.gestures.pop(msg[1], None)
                if g is not None:
                    g["walker"].hide_pose()
        except Exception as e:  # noqa: BLE001 - the scene goes on
            self.errors.append(f"{kind}: {e!r}")

    def seat(self, walkers: list[int]) -> None:
        """The drivers' walkers posed sitting at the wheel (cosim/drivers.POSE), each from its own rest pose."""
        import carla

        from cosim.drivers import POSE

        for aid in walkers:
            try:
                w = self.world.get_actor(aid)
                if w is None:
                    continue
                rel = {b.name: b.relative for b in w.get_bones().bone_transforms}
                bones = []
                for name, (dp, dy, dr) in POSE.items():
                    r0 = rel.get(name)
                    if r0 is not None:
                        bones.append((name, carla.Transform(r0.location, carla.Rotation(
                            r0.rotation.pitch + dp, r0.rotation.yaw + dy, r0.rotation.roll + dr))))
                w.get_pose_from_animation()
                w.set_bones(carla.WalkerBoneControlIn(bones))
                w.show_pose()
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"seat: {e!r}")

    def animate(self) -> None:
        import carla

        now = time.time()
        for aid, g in list(self.gestures.items()):
            if now - g["t"] < self.WAVE_S:
                continue
            g["t"] = now
            g["phase"] ^= 1
            arm = "crl_arm__R" if g["side"] == "R" else "crl_arm__L"
            other = "crl_arm__L" if arm == "crl_arm__R" else "crl_arm__R"
            sgn = 1.0 if arm == "crl_arm__R" else -1.0
            r0, o0 = g["rel"].get(arm), g["rel"].get(other)
            if r0 is None or o0 is None:
                continue
            bones = [(arm, carla.Transform(r0.location, carla.Rotation(
                r0.rotation.pitch, r0.rotation.yaw + sgn * (80.0 if g["phase"] else 40.0), r0.rotation.roll)))]
            if g["mode"] == "stop":  # the other hand up towards the oncoming traffic
                up = -80.0 if other == "crl_arm__R" else 80.0  # mirrored bones (probe: R pitch -80, L pitch +80: forward)
                bones.append((other, carla.Transform(o0.location, carla.Rotation(o0.rotation.pitch + up, o0.rotation.yaw,
                                                                                 o0.rotation.roll))))
            try:
                w = g["walker"]
                w.get_pose_from_animation()
                w.set_bones(carla.WalkerBoneControlIn(bones))
                w.show_pose()
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"gesture: {e!r}")
                self.gestures.pop(aid, None)

    def clear_all(self) -> None:
        import carla

        ids = [a for v in self.props.values() for a in v]
        self.props = {}
        if ids:
            try:
                self.client.apply_batch([carla.command.DestroyActor(a) for a in ids])
            except Exception:  # noqa: BLE001
                pass


def run(host: str, port: int, q_in, q_out, cx: float, cy: float, measure_only: bool = False) -> None:
    """The process: CARLA's tick callback poses the actors of the latest segment for the next picture. A failure
    goes to the viewer's log (as {"fatal": traceback}) instead of ending the process unseen."""
    try:
        _run(host, port, q_in, q_out, cx, cy, measure_only)
    except Exception:  # noqa: BLE001 - reported, then the process ends
        import traceback

        q_out.put({"fatal": traceback.format_exc(), "measure_only": measure_only})
        raise


def _run(host: str, port: int, q_in, q_out, cx: float, cy: float, measure_only: bool = False) -> None:
    import ctypes
    import multiprocessing as mp
    import queue
    import threading

    import carla

    _ = ctypes  # normal priority: above normal took CPU from CARLA's game thread (street level 74 -> 50 FPS)
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    drv = Driver(cx, cy)
    poser = drv.poser
    judder = Judder()
    trace = CrashTrace()
    lag = Lag()
    clock = FrameClock()
    pacing = Pacing()
    lock = threading.Lock()
    spectator_id = world.get_spectator().id
    chase = ChaseCam()
    extents: dict[int, tuple] = {}   # actor -> its bounding box's half length, width, height (asked once, off the tick)
    st = {"frames": 0, "posed": 0, "ms": 0.0, "ms_max": 0.0, "frame_dt": 1 / 60, "errors": 0,
          "steep": 0, "error": None, "actors": [], "sent": 0, "skipped": 0, "waited": 0}

    def tick(snap):
        t_cb = time.perf_counter()
        try:
            dt = snap.timestamp.delta_seconds
            if 0.0 < dt < 0.5:
                st["frame_dt"] += (dt - st["frame_dt"]) * 0.1
            pacing.picture(dt)
            t_pic = clock.wall(snap.timestamp.elapsed_seconds, time.time())
            sp = snap.find(spectator_id)
            eye = None
            if sp is not None:
                loc = sp.get_transform().location
                eye = (loc.x, loc.y)
            judder.picture(snap, st["actors"], dt, eye)
            trace.picture(snap, poser.seg, t_pic)
            waiting = lag.picture(snap)
            st["frames"] += 1
            if waiting is not None and waiting >= WAITING_MAX and not measure_only:
                # The server has not shown the last WAITING_MAX batches yet: it handles client calls for only a
                # moment each picture, about one batch, so every other call (a light change, a spawn) put the
                # poses one picture further behind for good (measured: 2 pictures behind after a second, 12 and
                # more, 2-8 m, after 10 s; the delay changing from picture to picture: judder). Sending nothing
                # this picture lets it catch up.
                st["waited"] += 1
                return
            # the moment of the picture this batch will be in: the server shows it SHOW_AFTER pictures after this
            # one (it has started the next before this callback runs; measured, Lag: 95-99 %)
            t_show = t_pic + SHOW_AFTER * st["frame_dt"]
            with lock:
                if measure_only:
                    while drv.pending and drv.pending[0]["start"] <= t_show:
                        drv.activate(drv.pending.pop(0))
                    if poser.seg is not None:
                        st["actors"] = [v[0] for v in poser.seg["veh"]]
                    return
                vs, ws, pv, pw, posed, skipped = drv.picture(t_show, eye, st["frames"])
                lights, drv.lights = drv.lights, []
            if poser.seg is None:
                return
            batch = [carla.command.ApplyTransform(aid, tf) for aid, tf in vs]
            batch += [carla.command.ApplyWalkerState(aid, tf, sp) for aid, tf, sp in ws]
            park = carla.Transform(carla.Location(*PARK))
            batch += [carla.command.ApplyTransform(aid, park) for aid in pv]
            batch += [carla.command.ApplyWalkerState(aid, park, 0.0) for aid in pw]
            batch += [carla.command.SetVehicleLightState(aid, carla.VehicleLightState(ls)) for aid, ls in lights]
            if drv.follow is not None and drv.follow_tf is not None:
                # (8 Nov) the followed car's camera, in the same batch as the car: shown in the same picture
                if drv.follow_new:
                    chase.reset()
                    if sp is not None:
                        chase.begin(sp.get_transform(), ChaseCam.FLY_IN_S)
                    drv.follow_new = False
                cam = chase.update(drv.follow_tf, st["frame_dt"], drv.follow_view,
                                   extents.get(drv.vid_aid.get(drv.follow)))
                batch.append(carla.command.ApplyTransform(spectator_id, cam))
            elif drv.follow is None:
                chase.reset()
            if batch:
                client.apply_batch(batch)
            lag.sent_now(vs)
            st["sent"] += len(vs) + len(ws)
            st["skipped"] += skipped
            st["actors"] = [aid for aid, _ in posed]
            st["posed"] += 1
            decks = set(drv.decks.values())  # a deck stands on its edge as a mesh (towing.MESH_ROT): not a car tipping
            st["steep"] += sum(1 for aid, tf in posed if abs(tf.rotation.pitch) > 30.0 and aid not in decks)
        except Exception as e:  # noqa: BLE001 - never stop CARLA's callback thread
            st["errors"] += 1
            if st["error"] is None:
                import traceback

                st["error"] = f"{e!r}\n{traceback.format_exc()}"
        finally:
            ms = 1000.0 * (time.perf_counter() - t_cb)
            st["ms"] += ms
            st["ms_max"] = max(st["ms_max"], ms)

    cb = world.on_tick(tick)
    parent = mp.parent_process()
    last_report = time.time()
    scene = SceneOps(client, world)
    try:
        while parent is None or parent.is_alive():
            try:
                msg = q_in.get(timeout=0.25)
            except queue.Empty:
                msg = None
            force = False
            if msg is not None:
                kind = msg[0]
                if kind == "stop":
                    break
                if kind == "report":  # the figures since the last report, now (to measure a stretch of the viewer's)
                    force = True
                elif kind in SceneOps.KINDS:  # calls to the server for an accident scene, here, off the viewer
                    scene.handle(msg)
                    if kind == "door":
                        with lock:
                            drv.receive(msg)
                elif kind == "drivers":  # seated once (their bones), then posed with their cars
                    scene.seat(msg[1])
                    with lock:
                        drv.receive(msg)
                else:
                    with lock:
                        drv.receive(msg)
            scene.animate()
            fa = drv.vid_aid.get(drv.follow) if drv.follow is not None else None
            if fa is not None and fa not in extents:  # the followed car's size, for where its cameras sit
                try:
                    e = world.get_actor(fa).bounding_box.extent
                    extents[fa] = (e.x, e.y, e.z)
                except Exception:  # noqa: BLE001 - gone: the default size
                    extents[fa] = None
            now = time.time()
            if force or now - last_report >= 10.0:
                span = now - last_report
                counted, judders = judder.take()
                q_out.put({"span_s": round(span, 1), "pictures_per_s": round(st["frames"] / span, 1),
                           "posed_per_s": round(st["posed"] / span, 1),
                           "ms_per_picture": round(st["ms"] / max(st["frames"], 1), 2),
                           "ms_max": round(st["ms_max"], 1), "held": poser.held, "errors": st["errors"],
                           "error": st["error"], "steep": st["steep"],
                           "judder_share": round(judders / counted, 4) if counted else None,
                           "commands_per_picture": round(st["sent"] / max(st["posed"], 1), 1),
                           "unchanged_skipped_per_picture": round(st["skipped"] / max(st["posed"], 1), 1),
                           "moving_vehicle_pictures": counted, "measure_only": measure_only,
                           "shown_lag": lag.take(),
                           "pictures_waited_for_server": st["waited"], "uneven_pictures": pacing.take(),
                           "pictures": st["frames"], "forced": force, "drivers_seated": len(drv.driver_in)})
                st.update(frames=0, posed=0, ms=0.0, ms_max=0.0, error=None, sent=0, skipped=0, waited=0)
                poser.held = 0
                last_report = now
    finally:
        scene.clear_all()
        try:
            world.remove_on_tick(cb)
        except Exception:  # noqa: BLE001
            pass

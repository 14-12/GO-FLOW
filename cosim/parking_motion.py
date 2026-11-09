"""(8 Nov) Cars that drive into their space and out of it, in the viewer.

SUMO keeps a parked car in its space without driving it there: a car stops on the road beside the car park and is
in its space at the next step, 42 m away; when it leaves, it is back on the road at once (measured on the Parking du
Marché). The viewer showed those jumps. Here the viewer drives the car instead, on a path made from the car park's
own layout (scenarios/town05/places.json):

* in: from where SUMO stopped it, a few metres on along the road, a turn into the car park, along the strip inside
  its road side to the aisle of its space, down the aisle, a turn into the space; blinker as it turns in, brake
  lights as it stops;
* out: it backs out of its space into the aisle (reversing lights), stops, drives along the aisle and the strip to
  the road, its blinker on, and joins the road exactly where and when SUMO puts it back. When its time there ends
  (the stop's duration) the drive out starts that much earlier; a car taken out of its home space for a new trip
  (cosim/purposes.py) waits in its space for the drive (PurposeDemand.exit_delay). If SUMO cannot put it back yet (a
  car in the way), it waits at the road; if SUMO puts it back first, SUMO's car is held where it is until the drive
  ends;
* a car parking at the kerb glides from its lane to the kerb, and back (9 Nov, 3rd: shown a little further on than
  SUMO has it, it pulls in forwards and leaves forwards, see KERB_SHIFT_M).

Only what you see changes: SUMO decides when a car parks and leaves, as before. The aisle of a space is found from
the space itself: half a space and half an aisle behind it, the aisle running across it (Town05's painted bays are in
rows facing an aisle).

(9 Nov) The cars on drives see each other (reserve, give_way): a new drive in a car park starts when its way is clear
of the drives already on theirs (a car going out waits in its space, one coming in on the road); on its way it slows
and stands for a car in front of it (another drive held up, a car of the road); along the strip and the aisles cars
keep to the right, so two meeting pass; the cars SUMO drives up behind a car standing on the road to turn in wait
(_hold_followers); a car going out pulls onto the road in a gap. The car parks inside buildings are driven to by
their way in (scenarios/<town>/entrances.json, found in CARLA's own geometry by tools/town05_entrances_build.py),
their door opened on the way (the Galerie Atlas's shutters); a car parked inside is not drawn. Without a way in, as
before: the car appears and vanishes at the road beside the building.

(9 Nov, 3rd) No glitch: the turns are a car's (arcs), and every way is tested against what stands there (CARLA's
map, the parked cars) before it is used (_pick); see docs/WORKLOG.md.

The viewer's vehicles carry their front bumper (cosim/mapping.py); the paths here are of the car's middle.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path

AISLE_BACK_M = 5.6      # a space's middle to its aisle's middle: half a space (2.6 m) and half an aisle (3 m)
ROAD_ON_M = 9.0         # a car turning in goes on this far along the road first (turns of 4 m: a car's)
STRIP_GAP_M = 3.0       # the strip inside the car park: this far before its first row of spaces
TURN_R_M = 4.5          # turns (fillets) of at most this radius
V_DRIVE, V_TURN, V_BACK = 3.6, 2.2, 1.3     # m/s: along the strip and the aisle, in turns, backing out
ACC, DEC = 1.0, 1.4     # m/s2
PAUSE_S = 0.8           # backed out: a stop before driving off
STEP_M = 0.25           # the path is kept as points this far apart
EXIT_GUARD_S = 0.5
WIDTH_M = 1.9           # a car's width, for its body (give_way)
GAP_M = 0.5             # giving way: the room kept to another car's body
LOOK_DT, LOOK_N = 0.5, 6     # ... looking this far ahead on its way (3 s): it stands if it would meet the car
                             # within 1 s, slows if later
RESERVE_MAX_S = 60.0    # a new drive waits at most this long for its way to be clear (reserve)
END_HOLD_S = 3.0        # ... a drive's end kept clear this long after it (SUMO takes the car over there)
RATE_UP = 0.34          # ... and back to its own pace over 3 steps
GIVE_UP_S = 20.0        # ... held this long by a car held behind another drive: it goes on (by a drive: GIVE_UP_DRIVE_S)
GIVE_UP_DRIVE_S = 40.0
NEAR_M = 30.0           # the road's cars looked at: this near a drive
KEEP_RIGHT_M = 1.3      # along the strip and the aisles, cars keep to the right of the way by this (less in a
                        # narrow aisle: keep_right_m)
SPACE_HALF_M = 2.6      # half a space's depth (a parked car's nose is this far from its space's middle)
DOOR_OPEN_M = 6.0       # a building's door opens when a car on its drive is this far from it
DOOR_SHUT_M = 4.0       # ... and shuts when the car is this far past it
ON_LANE_M = 2.2         # a body this near its lane's line stands on the lane (the cars behind on it wait)
HOLD_BEHIND_M = 35.0    # ... the cars on the lane this far behind it
HOLD_GAP_M = 3.0        # ... one closer than this to a car waiting for SUMO to take it over is asked to pass it
MERGE_M, MERGE_S = 10.0, 4.0     # a drive out pulls onto the road when no car is this near behind (or this many
                                 # seconds at its speed)
OVERDUE_S = 3.0         # a drive out at the road this long after SUMO should have put it back: no longer held for
# (9 Nov, 3rd) no glitch: every drive's body, all along its way, is tested against what stands there (CARLA's
# parked cars, posts, trees, walls: scenarios/town05/parking_obstacles.json; the cars standing in their spaces), and
# the first of a few ways that keeps CLEAR_M off all of them is taken (ParkingMotion._pick)
CLEAR_M = 0.15
BACK_STRAIGHT_M = 0.8   # backed out: this far straight along the aisle after the turn, before it stops
MIN_R_M = 3.0           # a way turning tighter than this (a car cannot) is taken only if no other is clear
# at a kerb: SUMO stops a car on its lane beside its space and puts it back there; the viewer shows it parked
# KERB_SHIFT_M further on (kerb_fronts): the car stops behind its space and pulls into it forwards (an S, the cars
# behind it waiting while it is on the lane); it leaves forwards, onto its lane KERB_OUT_M past its space, where SUMO's
# car is moved (vehicle.moveTo, once that stretch of the lane is clear: the cars coming up behind wait)
KERB_SHIFT_M = 6.5
KERB_ROOM_M = 6.0       # ... the spaces as shown this far apart at least (front to front)
KERB_OUT_M = 8.0
KERB_WAIT_S = 15.0      # ... that stretch not clear this long: it backs out to where SUMO put it instead
REAR_AXLE_M = 1.0       # at a kerb a car turns about its rear axle, this far in front of its back
LC_TO_STOP = 0b001001000101    # SUMO lane change mode: strategic, cooperative, right; no speed gain, no sublane moves
PARKED_PAUSE_S = 2.0    # a car taken out for a trip on its way in: it stands this long in its space first
RIGHT, LEFT, BRAKE, BACK = 1, 2, 8, 128     # SUMO's signal bits (cosim/convert.py)
OWN_BITS = RIGHT | LEFT | BRAKE | BACK


def _unit(x, y):
    n = math.hypot(x, y) or 1.0
    return x / n, y / n


def _dir(angle_deg):
    a = math.radians(angle_deg)                 # SUMO: degrees clockwise from north
    return math.sin(a), math.cos(a)


def _angle(dx, dy):
    return math.degrees(math.atan2(dx, dy)) % 360.0


def fillet(corners, r_max=TURN_R_M):
    """The polyline through `corners` with each inner corner rounded (a quadratic curve of at most r_max either side
    of it, at most 45 % of the segments next to it), as points STEP_M apart."""
    pts = [corners[0]]
    for i in range(1, len(corners) - 1):
        a, c, b = corners[i - 1], corners[i], corners[i + 1]
        la, lb = math.dist(a, c), math.dist(c, b)
        r = min(r_max, 0.45 * la, 0.45 * lb)
        if r < 0.05:
            pts.append(c)
            continue
        ua, ub = _unit(c[0] - a[0], c[1] - a[1]), _unit(b[0] - c[0], b[1] - c[1])
        p0 = (c[0] - ua[0] * r, c[1] - ua[1] * r)
        p2 = (c[0] + ub[0] * r, c[1] + ub[1] * r)
        pts.append(p0)
        n = max(4, int(2 * r / STEP_M))
        for k in range(1, n):
            t = k / n
            pts.append(((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p2[0],
                        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p2[1]))
        pts.append(p2)
    pts.append(corners[-1])
    return resample(pts)


def arcs(corners, r=TURN_R_M, keep_end=0.5):
    """(9 Nov, 3rd) The polyline through `corners` with each inner corner a circular arc of radius r, as a car turns
    (fillet's curves are tighter than their size: a car backing out of its space turned on 1.1 m, its body swept
    through the cars beside it). Where the segments are too short for r the radius is smaller (the arcs of two
    corners share the segment between them); the last segment keeps keep_end straight (the car ends straight in its
    space). Points STEP_M apart."""
    n = len(corners)
    if n < 3:
        return resample(list(corners))
    rr = list(r) if isinstance(r, list | tuple) else [r] * n
    tan = [0.0] * n
    want = [0.0] * n
    for i in range(1, n - 1):
        a, c, b = corners[i - 1], corners[i], corners[i + 1]
        ua, ub = _unit(c[0] - a[0], c[1] - a[1]), _unit(b[0] - c[0], b[1] - c[1])
        th = math.acos(max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1])))
        th = min(th, math.radians(170.0))
        tan[i] = math.tan(th / 2.0)
        want[i] = rr[i] * tan[i]
    seg = [math.dist(corners[i], corners[i + 1]) for i in range(n - 1)]
    T = list(want)
    for _ in range(3):                                   # (each segment holds the tangents of its two ends)
        for j in range(n - 1):
            room = seg[j] - (keep_end if j == n - 2 else 0.0)
            used = (T[j] if j > 0 else 0.0) + (T[j + 1] if j + 1 < n - 1 else 0.0)
            if used > room > 0.0 and used > 1e-9:
                k = room / used
                if j > 0:
                    T[j] *= k
                if j + 1 < n - 1:
                    T[j + 1] *= k
    pts = [corners[0]]
    for i in range(1, n - 1):
        a, c, b = corners[i - 1], corners[i], corners[i + 1]
        if tan[i] < 0.01 or T[i] < 0.05:
            pts.append(c)
            continue
        ua, ub = _unit(c[0] - a[0], c[1] - a[1]), _unit(b[0] - c[0], b[1] - c[1])
        R = T[i] / tan[i]
        p0 = (c[0] - ua[0] * T[i], c[1] - ua[1] * T[i])
        left = ua[0] * ub[1] - ua[1] * ub[0] > 0.0
        nrm = (-ua[1], ua[0]) if left else (ua[1], -ua[0])
        o = (p0[0] + nrm[0] * R, p0[1] + nrm[1] * R)
        th = 2.0 * math.atan(tan[i]) * (1.0 if left else -1.0)
        a0 = math.atan2(p0[1] - o[1], p0[0] - o[0])
        m = max(2, int(abs(th) * R / (STEP_M * 0.5)))
        pts += [(o[0] + R * math.cos(a0 + th * k / m), o[1] + R * math.sin(a0 + th * k / m)) for k in range(m + 1)]
    pts.append(corners[-1])
    return resample(_dedupe(pts, 0.02))


def s_curve(p0, h, p1, r):
    """(9 Nov, 3rd) From p0 heading h to p1 (beside the line ahead of it, heading h again): an S of two arcs of radius
    r at its start (a car pulling out of a kerb place turns hard first, its nose out past the car in front), then
    straight on to p1. None when p1 is not far enough ahead."""
    n = (-h[1], h[0])
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    D, lat = dx * h[0] + dy * h[1], dx * n[0] + dy * n[1]
    if lat < 0:
        n, lat = (-n[0], -n[1]), -lat
    if lat < 0.2:
        return None
    th = math.acos(max(-1.0, 1.0 - lat / (2.0 * r))) if lat < 2.0 * r else math.pi / 2.0
    T = r * math.tan(th / 2.0)
    x2 = T + lat / max(math.tan(th), 1e-6)
    if D < x2 + T + 0.3:
        return None
    c1 = (p0[0] + h[0] * T, p0[1] + h[1] * T)
    c2 = (p0[0] + h[0] * x2 + n[0] * lat, p0[1] + h[1] * x2 + n[1] * lat)
    return arcs([p0, c1, c2, p1], r, keep_end=0.3)


def min_radius(pts):
    """The tightest turn along a path (m), from its points three apart."""
    best = float("inf")
    for i in range(3, len(pts) - 3, 2):
        a, b, c = pts[i - 3], pts[i], pts[i + 3]
        ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > 1e-6:
            best = min(best, ab * bc * ca / (2.0 * cross))
    return best


def hermite(p0, h0, p1, h1, n=None, scale=1.0):
    """A smooth curve from p0 heading h0 (unit) to p1 heading h1 (a car gliding to the kerb); scale: how long it
    keeps either heading (its tangents, times the distance)."""
    d = math.dist(p0, p1) * scale
    n = n or max(6, int(math.dist(p0, p1) * 2 / STEP_M))
    pts = []
    for k in range(n + 1):
        t = k / n
        a, b, c, e = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        pts.append((a * p0[0] + b * d * h0[0] + c * p1[0] + e * d * h1[0],
                    a * p0[1] + b * d * h0[1] + c * p1[1] + e * d * h1[1]))
    return resample(pts)


def resample(pts):
    out = [pts[0]]
    for a, b in itertools.pairwise(pts):
        seg = math.dist(a, b)
        k = 1
        while k * STEP_M < seg:
            out.append((a[0] + (b[0] - a[0]) * k * STEP_M / seg, a[1] + (b[1] - a[1]) * k * STEP_M / seg))
            k += 1
        if math.dist(out[-1], b) > 1e-6:
            out.append(b)
    return out


class Leg:
    """Part of a drive: points, timed by a speed profile (from standing, at most v_max, slower in turns, to standing).
    reverse: the car faces against its way (backing out)."""

    BLEND_M = 10.0      # an end moved (end_offset): over this last stretch of the leg

    def __init__(self, pts, v_max, reverse=False, signals=0, blink_until_m=None, pivot=0.0):
        self.pts, self.reverse, self.signals = pts, reverse, signals
        self.pivot = pivot          # (9 Nov, 3rd) pts are of a point this far behind the middle (the rear axle)
        self.end_offset = None
        self.blink_until_m = blink_until_m
        n = len(pts)
        cum = [0.0]
        for a, b in itertools.pairwise(pts):
            cum.append(cum[-1] + math.dist(a, b))
        self.cum = cum
        self.length = cum[-1]
        # the speed allowed at each point: slower where the path turns
        cap = [v_max] * n
        for i in range(1, n - 1):
            h0 = _angle(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            h1 = _angle(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            turn = abs((h1 - h0 + 180.0) % 360.0 - 180.0)
            if turn > 1.5:
                cap[i] = min(v_max, V_TURN if not reverse else v_max)
        v = [0.0] * n
        for i in range(1, n):                       # speeding up from standing
            v[i] = min(cap[i], math.sqrt(v[i - 1] ** 2 + 2 * ACC * (cum[i] - cum[i - 1])))
        v[-1] = 0.0
        for i in range(n - 2, -1, -1):              # and down to standing at the end
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * DEC * (cum[i + 1] - cum[i])))
        t = [0.0]
        for i in range(1, n):
            ds = cum[i] - cum[i - 1]
            vm = max((v[i] + v[i - 1]) / 2.0, 0.15)
            t.append(t[-1] + ds / vm)
        self.t, self.v = t, v
        self.duration = t[-1]

    def path_at(self, s):
        """The point of its path (not the car's middle, with a pivot) at s seconds into the leg."""
        x, y, face, _v, _g = self.at(s)
        if self.pivot:
            hx, hy = _dir(face)
            x, y = x - hx * self.pivot, y - hy * self.pivot
        return x, y

    def at(self, s):
        """(x, y, facing degrees, speed, signals) at s seconds into the leg."""
        t = self.t
        if s <= 0:
            i, f = 0, 0.0
        elif s >= t[-1]:
            i, f = len(t) - 2, 1.0
        else:
            lo, hi = 0, len(t) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if t[mid] <= s:
                    lo = mid
                else:
                    hi = mid
            i = lo
            f = (s - t[i]) / max(t[i + 1] - t[i], 1e-9)
        i = max(0, min(i, len(self.pts) - 2))
        a, b = self.pts[i], self.pts[i + 1]
        x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        if self.end_offset is not None:              # (SUMO put the car back elsewhere: the end moved there)
            w = min(max((self.cum[i] + (self.cum[i + 1] - self.cum[i]) * f - (self.length - self.BLEND_M))
                        / min(self.BLEND_M, self.length or 1.0), 0.0), 1.0)
            w = w * w * (3 - 2 * w)
            x, y = x + self.end_offset[0] * w, y + self.end_offset[1] * w
        j = min(i + 2, len(self.pts) - 1)             # facing: along the way where it is (half a metre)
        k = max(j - 3, 0)
        dx, dy = self.pts[j][0] - self.pts[k][0], self.pts[j][1] - self.pts[k][1]
        face = _angle(dx, dy)
        if self.reverse:
            face = (face + 180.0) % 360.0
        if self.pivot:
            hx, hy = _dir(face)
            x, y = x + hx * self.pivot, y + hy * self.pivot
        speed = self.v[i] + (self.v[i + 1] - self.v[i]) * f
        sig = self.signals
        if self.blink_until_m is not None and self.cum[i] > self.blink_until_m:
            sig &= ~(RIGHT | LEFT)
        if self.reverse:
            sig |= BACK
        if not self.reverse and speed < 2.0 and self.cum[i] > self.length - 4.0:
            sig |= BRAKE                               # stopping
        return x, y, face, speed, sig


class Drive:
    """Legs one after the other, from t0 (SUMO seconds), with pauses between. lag: the seconds it has been held up
    giving way (ParkingMotion.give_way): it is that far behind its own timetable."""

    def __init__(self, vid, kind, legs, t0, length, pauses=None):
        self.vid, self.kind, self.legs, self.t0, self.length = vid, kind, legs, t0, length
        self.area, self.why = None, None
        self.pauses = pauses or [0.0] * len(legs)
        self.duration = sum(lg.duration for lg in legs) + sum(self.pauses)
        self.lag, self.rate = 0.0, 1.0
        self.held_s = 0.0               # how long it has stood giving way (all its stops together)
        self.blocker, self.blocked_s = None, 0.0      # who it gives way to now, and for how long
        self.ignored: set[str] = set()  # cars it waited for too long (GIVE_UP_S): no longer given way to

    def progress(self, now):
        """The seconds of its drive done at `now` (its timetable less the time held up)."""
        return now - self.t0 - self.lag

    def at(self, now):
        return self.at_s(self.progress(now))

    def at_s(self, s):
        """(x, y, facing, speed, signals) s seconds into the drive."""
        for lg, pause in zip(self.legs, self.pauses, strict=False):
            if s <= lg.duration:
                return lg.at(s)
            s -= lg.duration
            if s <= pause:
                x, y, face, _v, sig = lg.at(lg.duration)
                return x, y, face, 0.0, (sig & ~BACK) | BRAKE
            s -= pause
        return self.legs[-1].at(self.legs[-1].duration)

    def done(self, now):
        return self.progress(now) >= self.duration


class ParkingMotion:
    """One per viewer run, after the town is filled. step() after PurposeDemand.step, on the step's vehicles."""

    def __init__(self, demand, conn):
        self.d, self.c = demand, conn
        self.areas = demand.places.areas
        self.drives: dict[str, Drive] = {}
        self.release: dict[str, float] = {}        # parked car -> when SUMO puts it back on the road
        self.held: set[str] = set()                # SUMO's car held at the road until the drive out ends
        self.followers: set[str] = set()           # SUMO's cars held behind a drive's car standing on their lane
        self.prev: dict[str, tuple] = {}           # vehicle -> (x, y, angle) at the step before (front bumper)
        self.seams: list[dict] = []                # each drive out: how far from SUMO's car it ended (the last 200)
        self._k = 0
        self.stats = {"in": 0, "out": 0, "kerb_in": 0, "kerb_out": 0, "waited_at_road": 0, "held_sumo": 0,
                      "seam_max_m": 0.0, "skipped_hidden": 0, "failed": 0}
        demand.exit_delay = self.exit_now
        # (9 Nov) the car parks inside buildings with a way in (scenarios/<town>/entrances.json, from CARLA's own
        # geometry: tools/town05_entrances_build.py), and the doors open on the way (door name -> cars needing it)
        self.entrances, self.gates, self.removed = load_entrances(getattr(demand.places, "path", None))
        self.doors: dict[str, set] = {}
        # (9 Nov, 3rd) what stands where the drives go (cosim/obstacles.py), and the kerbs' cars as shown
        from cosim import obstacles as po
        self._po = po
        src = getattr(demand.places, "path", None)
        f = po.FILE if src is None else Path(src).with_name("parking_obstacles.json")
        skip = set(self.gates) | set(self.removed)
        self.grid = po.Grid([o for o in po.load_file(f) if o[0] not in skip])
        self._kerb: dict[str, tuple] = {}          # parked at a kerb -> (shown middle x, y, heading unit, SUMO front)
        self._kerb_fronts: dict[str, list] = {}    # kerb -> its spaces' fronts as shown (lane positions)
        self.touches: list = []                    # the drives whose way still touches something (the last 50)
        self._outs: dict[str, tuple] = {}          # a drive out made before its time: (when, drive, area)
        self._lc: dict[str, int] = {}              # held: its lane change mode before
        self._passing: set[str] = set()            # asked to pass a car waiting at the road on the next lane
        self._lc_near: set[str] = set()            # coming to park where a car stands: kept to its lane's middle
        self._kerb_wait: dict[str, float] = {}     # put back at a kerb, waiting for its way out to be clear: since

    # ------------------------------------------------------------------ geometry
    def visible(self, aid):
        """Whether the cars of area aid are driven in and out: car parks with painted bays, kerbs, and car parks inside
        buildings with a way in (entrances.json)."""
        a = self.areas.get(aid) or {}
        if a.get("hidden"):
            return aid in self.entrances
        return "spaces" in a or bool(a.get("roadside"))

    def inside(self, vid):
        """A car standing in a car park inside a building (not on a drive): out of sight, not drawn."""
        aid = self.d.parked.get(vid)
        return aid is not None and vid not in self.drives and (self.areas.get(aid) or {}).get("hidden", False)

    def door_states(self):
        """{door name: open} for the doors on the buildings' ways in: open while a car is near it on its drive."""
        return {name: bool(cars) for name, cars in self.doors.items()}

    def _doors_step(self, now):
        for name in self.doors:
            self.doors[name] = set()
        for vid, dv in self.drives.items():
            if not hasattr(dv, "gates"):
                dv.gates = self._gates_on(dv) if self.gates else []
            s = min(max(dv.progress(now), 0.0), dv.duration)
            for name, s0, s1 in dv.gates:
                self.doors.setdefault(name, set())
                if s0 <= s <= s1:
                    self.doors[name].add(vid)

    def _road(self, aid, lane_pos):
        """(front x, y, heading unit) on the area's lane at lane_pos."""
        a = self.areas[aid]
        idx = int(a["lane"].rsplit("_", 1)[1])
        L = self.c.lane.getLength(a["lane"])
        p = min(max(lane_pos, 0.0), L)
        x0, y0 = self.c.simulation.convert2D(a["edge"], p, idx)
        x1, y1 = self.c.simulation.convert2D(a["edge"], min(p + 1.0, L), idx)
        if math.hypot(x1 - x0, y1 - y0) < 1e-3:
            x1, y1 = x0, y0
            x0, y0 = self.c.simulation.convert2D(a["edge"], max(p - 1.0, 0.0), idx)
        return (x0, y0), _unit(x1 - x0, y1 - y0)

    def keep_right_m(self, aid, bay_c, h):
        """(9 Nov, 2nd) How far right of an aisle's middle a car keeps: KEEP_RIGHT_M where the aisle has room for two,
        less in a narrow one (the Parking du Square's: 4.3 m between the cars' fronts; at 1.3 m a car going along
        it touched the parked cars' noses). The aisle's width: from this space to the row facing it across."""
        cache = self.__dict__.setdefault("_kr", {})
        key = (aid, round(bay_c[0], 1), round(bay_c[1], 1))
        if key not in cache:
            gap = None
            for sx, sy, _ang in self.areas[aid].get("spaces", ()):
                dx, dy = sx - bay_c[0], sy - bay_c[1]          # (the row across the aisle: behind this space)
                along, side = -(dx * h[0] + dy * h[1]), abs(dx * h[1] - dy * h[0])
                if along > 3.0 and side < 1.5:
                    gap = along if gap is None else min(gap, along)
            aisle = (gap - 2 * SPACE_HALF_M) if gap is not None else 99.0
            cache[key] = max(0.0, min(KEEP_RIGHT_M, (aisle - WIDTH_M) / 2.0 - 0.45))
        return cache[key]

    def aisle_back(self, aid, bay_c, h):
        """(9 Nov, 3rd) A space's middle to its aisle's middle: half way to the row facing it across the aisle
        (AISLE_BACK_M without one; the Parking du Square's aisle is 4.7 m from its spaces, 0.6 m less)."""
        gap = None
        for sx, sy, _ang in self.areas[aid].get("spaces", ()):
            dx, dy = sx - bay_c[0], sy - bay_c[1]
            along, side = -(dx * h[0] + dy * h[1]), abs(dx * h[1] - dy * h[0])
            if along > 3.0 and side < 1.5:
                gap = along if gap is None else min(gap, along)
        return gap / 2.0 if gap is not None and gap < 2 * AISLE_BACK_M + 2.0 else AISLE_BACK_M

    def narrow(self, aid):
        """Whether a car park's aisle is too narrow for two cars to pass (places.json "one_at_a_time": the Parking du
        Square's, 4.3 m between its rows' noses): its drives go one at a time."""
        return bool((self.areas.get(aid) or {}).get("one_at_a_time"))

    def _kerb_place_taken(self, vid, dv, now):
        """The car still in (or pulling out of) the kerb place a drive in goes to, or None."""
        k = self._kerb.get(vid)
        if k is None or k[4] != dv.area or "spaces" in self.areas.get(dv.area, {}):
            return None
        for oid, ok in self._kerb.items():
            if oid != vid and ok[4] == dv.area and oid not in self.d.parked and math.dist(ok[:2], k[:2]) < 3.0:
                return oid              # (put back by SUMO, waiting in its place to pull out)
        for oid, od in self.drives.items():
            if oid == vid or od.area != dv.area or od.kind != "out" or od.done(now):
                continue
            x, y, _f, _v, _g = od.at_s(min(max(od.progress(now), 0.0), od.duration))
            if math.dist((x, y), k[:2]) < dv.length + 1.0:
                return oid
        return None

    def _aisle_busy(self, vid, dv, now):
        """Another drive in the same car park on its way (or waiting to start before this one)."""
        for oid, od in self.drives.items():
            if oid == vid or od.area != dv.area or od.done(now):
                continue
            if od.progress(now) >= 0.5 or (od.t0, oid) < (dv.t0, vid):
                return True
        return False

    def lot_corners(self, aid, road_c, dr, bay_c, h):
        """Corners of the way in (road -> space), middles of the car; and the aisle's way out (towards the road)."""
        spaces = self.areas[aid]["spaces"]
        nx, ny = -dr[1], dr[0]                                   # across the road...
        if (bay_c[0] - road_c[0]) * nx + (bay_c[1] - road_c[1]) * ny < 0:
            nx, ny = -nx, -ny                                    # ...towards the car park
        n = (nx, ny)

        def along_n(p):
            return (p[0] - road_c[0]) * n[0] + (p[1] - road_c[1]) * n[1]

        def along_r(p):
            return (p[0] - road_c[0]) * dr[0] + (p[1] - road_c[1]) * dr[1]

        ab = self.aisle_back(aid, bay_c, h)
        P = (bay_c[0] - h[0] * ab, bay_c[1] - h[1] * ab)
        c0 = (road_c[0] + dr[0] * ROAD_ON_M, road_c[1] + dr[1] * ROAD_ON_M)
        aisle_across_road = abs(h[0] * dr[0] + h[1] * dr[1]) > 0.7   # spaces along the road: aisles cross it
        if aisle_across_road:
            first = min(along_n(s[:2]) - 1.5 for s in spaces)
            strip = max(first - STRIP_GAP_M, 4.0)
            g = (c0[0] + n[0] * strip, c0[1] + n[1] * strip)
            k = along_r(P) - along_r(g)
            s_ = (g[0] + dr[0] * k, g[1] + dr[1] * k)
            corners = [c0, g, s_, P]
        else:                                                    # aisles along the road
            k = along_n(P) - along_n(c0)
            t_ = (c0[0] + n[0] * k, c0[1] + n[1] * k)
            corners = [c0, t_, P]
        return corners, n, aisle_across_road

    # ------------------------------------------------------------------ the kerbs
    def kerb_fronts(self, aid):
        """The fronts of a kerb's spaces as shown (lane positions), from SUMO's (start + (i + 1) x its length):
        KERB_SHIFT_M further on, less near the lane's end (KERB_OUT_M left to pull out, KERB_ROOM_M between two),
        a place on a tree or a post moved off it by a metre or two. -> (fronts, SUMO's space length)."""
        if aid not in self._kerb_fronts:
            a = self.areas[aid]
            n = int(a.get("roadside") or 0)
            ln = (a["end"] - a["start"]) / max(n, 1)
            if a.get("shown_fronts"):
                # (9 Nov, 4th) the places clear of the pavement edge's pines and lamp posts (places.json, from
                # tools/town05_kerb_places.py): SUMO's i-th space shown at the i-th of them
                fr = list(a["shown_fronts"])
                self._kerb_fronts[aid] = ([fr[min(i, len(fr) - 1)] for i in range(n)], ln)
                return self._kerb_fronts[aid]
            try:
                lane_len = self.c.lane.getLength(a["lane"])
            except Exception:  # noqa: BLE001
                lane_len = a["end"] + KERB_OUT_M
            out, nxt = [0.0] * n, None
            for i in range(n - 1, -1, -1):
                f = a["start"] + (i + 1) * ln
                hi = lane_len - KERB_OUT_M if nxt is None else nxt - KERB_ROOM_M
                best = None
                for d in (0.0, -1.0, 1.0, -2.0, 2.0):
                    sf = max(min(f + KERB_SHIFT_M + d, hi), 5.5)
                    if not self._kerb_hits(aid, sf, 4.7):
                        best = sf
                        break
                    if best is None:
                        best = max(min(f + KERB_SHIFT_M, hi), 5.5)
                out[i] = best
                nxt = best
            if "_lat" not in a:
                return out, ln          # (its side offset not known yet: not kept)
            self._kerb_fronts[aid] = (out, ln)
        return self._kerb_fronts[aid]

    def _kerb_lat(self, aid):
        """How far beside its lane's line SUMO shows a car at this kerb (m, signed: + left of the lane's way)."""
        return self.areas[aid].get("_lat", -2.6)

    def _kerb_hits(self, aid, front, L):
        (x, y), u = self._road(aid, front - L / 2)
        lat = self._kerb_lat(aid)
        mx, my = x - u[1] * lat, y + u[0] * lat
        return self.grid.touching(self._po.car_box(mx, my, _angle(*u), L, margin=-0.05))

    def kerb_pose(self, vid, aid, v=None):
        """A car parked at a kerb as shown: (middle x, y, heading unit, SUMO's front lane position)."""
        k = self._kerb.get(vid)
        if k is not None and k[4] == aid:
            return k
        try:
            f = self.c.vehicle.getLanePosition(vid)
        except Exception:  # noqa: BLE001
            return None
        if v is None:
            v = (self._cur or {}).get(vid) if getattr(self, "_cur", None) else None
        a = self.areas[aid]
        if v is not None and "_lat" not in a and abs(v.x) < 1e5:   # (SUMO's side offset, once per kerb)
            (cx, cy), u = self._road(aid, f)
            lat = (v.x - cx) * -u[1] + (v.y - cy) * u[0]
            if abs(lat) > 1.2:                      # (a car parked there, not one on the lane)
                a["_lat"] = lat
        fronts, ln = self.kerb_fronts(aid)
        i = round((f - a["start"]) / ln) - 1
        sf = fronts[i] if 0 <= i < len(fronts) else f
        L = v.length if v is not None else 4.7
        (x, y), u = self._road(aid, sf - L / 2)
        lat = self._kerb_lat(aid)
        k = (x - u[1] * lat, y + u[0] * lat, u, f, aid, L)
        self._kerb[vid] = k
        return k

    # ------------------------------------------------------------------ no glitch
    def _standing(self, vid, box):
        """Bodies (boxes) of the cars standing in their spaces within box (x0, x1, y0, y1), but vid's."""
        out = []
        cur = getattr(self, "_cur", None) or {}
        x0, x1, y0, y1 = box
        for oid, aid in self.d.parked.items():
            if oid == vid or oid in self.drives:
                continue
            a = self.areas.get(aid) or {}
            if a.get("hidden"):
                continue
            if a.get("roadside") and "spaces" not in a:
                k = self.kerb_pose(oid, aid)
                if k is None:
                    continue
                mx, my, u, L = k[0], k[1], k[2], k[5]
                face = _angle(*u)
            else:
                v = cur.get(oid)
                if v is None or abs(v.x) > 1e5:
                    continue
                hh = _dir(v.angle)
                L = v.length
                mx, my, face = v.x - hh[0] * L / 2, v.y - hh[1] * L / 2, v.angle
            if x0 <= mx <= x1 and y0 <= my <= y1:
                out.append((oid, self._po.car_box(mx, my, face, L)))
        return out

    def _road_cars(self, vid, box):
        """Bodies of the cars SUMO drives (not parked, not on drives) within box, as they are now."""
        out = []
        cur = getattr(self, "_cur", None) or {}
        x0, x1, y0, y1 = box
        for oid, v in cur.items():
            if oid == vid or oid in self.drives or oid in self.d.parked or abs(v.x) > 1e5:
                continue
            hh = _dir(v.angle)
            mx, my = v.x - hh[0] * v.length / 2, v.y - hh[1] * v.length / 2
            if x0 <= mx <= x1 and y0 <= my <= y1:
                out.append((oid, self._po.car_box(mx, my, v.angle, v.length)))
        return out

    def _sweep(self, legs, L, vid, road=False):
        """What a drive's body touches all along its way (every half metre, CLEAR_M kept): {name: samples}, and the
        tightest turn of its way."""
        pts = [p for lg in legs for p in lg.pts]
        box = (min(p[0] for p in pts) - 8.0, max(p[0] for p in pts) + 8.0, min(p[1] for p in pts) - 8.0,
               max(p[1] for p in pts) + 8.0)
        cars = self._standing(vid, box) + (self._road_cars(vid, box) if road else [])
        names: dict[str, int] = {}
        r_min = float("inf")
        for lg in legs:
            r_min = min(r_min, min_radius(lg.pts))
            for i in range(0, len(lg.t), 2):
                x, y, face, _v, _g = lg.at(lg.t[i])
                body = self._po.car_box(x, y, face, L, margin=-CLEAR_M)
                for name in self.grid.touching(body):
                    names[name] = names.get(name, 0) + 1
                for oid, b in cars:
                    if self._po.boxes_touch(body, b):
                        names[oid] = names.get(oid, 0) + 1
        return names, r_min

    def _pick(self, kind, vid, L, cands, make, pose, road=False):
        """The first of the ways cands (make(*c) -> legs) that touches nothing (what touches its body where it stands
        in its space, pose (x, y, facing), does not count: it is there anyway) and turns no tighter than MIN_R_M;
        else the one touching least. -> legs."""
        x, y, face = pose
        here = set(self.grid.touching(self._po.car_box(x, y, face, L, margin=-CLEAR_M)))
        best = None
        for i, c in enumerate(cands):
            try:
                legs = make(*c)
            except Exception:  # noqa: BLE001, S112 - that way cannot be made here
                continue
            if not legs:
                continue
            names, r_min = self._sweep(legs, L, vid, road=road)
            bad = {k: n for k, n in names.items() if k not in here}
            score = (sum(bad.values()), 0 if r_min >= MIN_R_M else 1, i)
            if best is None or score < best[0]:
                best = (score, legs, bad, c, r_min)
            if score[0] == 0 and score[1] == 0:
                break
        if best is None:
            return None
        (n_bad, tight, i), legs, bad, c, r_min = best
        st = self.stats
        st["ways_tried"] = st.get("ways_tried", 0) + i + 1
        if n_bad:
            st["ways_touching"] = st.get("ways_touching", 0) + 1
            self.touches.append({"vid": vid, "kind": kind, "touches": bad, "way": c})
            del self.touches[:-50]
        else:
            st["ways_clear"] = st.get("ways_clear", 0) + 1
        if tight:
            st["ways_tight"] = st.get("ways_tight", 0) + 1
        return legs

    # ------------------------------------------------------------------ the drives
    def _in(self, vid, aid, now, v):
        """The drive in, from where SUMO stopped it (the step before) to its space."""
        pv = self.prev.get(vid)
        if pv is None:
            return None
        L = v.length
        dr = _dir(pv[2])
        road_c = (pv[0] - dr[0] * L / 2, pv[1] - dr[1] * L / 2)
        h = _dir(v.angle)
        if aid in self.entrances:                   # into a building, by its way in
            e = self.entrances[aid]
            c0 = (road_c[0] + dr[0] * ROAD_ON_M, road_c[1] + dr[1] * ROAD_ON_M)
            way = [road_c, c0, *map(tuple, e.get("approach", ())), tuple(e["front"]), *map(tuple, e["way"])]
            pts = fillet(_dedupe(way))
            n_ = _unit(e["front"][0] - c0[0], e["front"][1] - c0[1])
            side = RIGHT if (dr[0] * n_[1] - dr[1] * n_[0]) < 0 else LEFT
            leg = Leg(pts, V_DRIVE, signals=side, blink_until_m=ROAD_ON_M + 8.0)
            self.stats["building_in"] = self.stats.get("building_in", 0) + 1
            return self._on_road(Drive(vid, "in", [leg], now, L), aid, road_c, dr)
        if "spaces" in self.areas[aid]:
            bay_c = (v.x - h[0] * L / 2, v.y - h[1] * L / 2)       # (its middle: the space's, freefly.produce)
            corners, n, across = self.lot_corners(aid, road_c, dr, bay_c, h)
            kr = self.keep_right_m(aid, bay_c, h)
            side = RIGHT if (dr[0] * n[1] - dr[1] * n[0]) < 0 else LEFT
            P, before = corners[-1], corners[-2]
            a = _unit(P[0] - before[0], P[1] - before[1])            # down the aisle: on its right half

            way_in = self.areas[aid].get("way_in")
            if way_in:                          # (its own way from the road to its aisle: places.json)
                a = _unit(P[0] - way_in[-1][0], P[1] - way_in[-1][1])
                ax = (h[1], -h[0])
                a = ax if a[0] * ax[0] + a[1] * ax[1] >= 0 else (-ax[0], -ax[1])

            def make(off, r_bay):
                if way_in:
                    way = [road_c, *map(tuple, way_in), P, bay_c]
                else:
                    way = keep_right([road_c, *corners, bay_c], (2, 3) if across else (2,), off=kr)
                way[-2] = (P[0] + a[1] * off, P[1] - a[0] * off)
                rr = [TURN_R_M] * len(way)
                rr[-2] = r_bay
                return [Leg(arcs(way, rr, keep_end=0.6), V_DRIVE, signals=side, blink_until_m=ROAD_ON_M + 8.0)]
            cands = [(kr, 4.5), (kr, 4.0), (kr, 5.0), (kr / 2, 4.5), (0.0, 4.5), (kr, 3.5), (0.0, 4.0), (-0.5, 4.5),
                     (0.0, 3.5), (kr, 3.0), (-0.5, 3.5)]
            legs = self._pick("lot in", vid, L, cands, make, (bay_c[0], bay_c[1], v.angle))
            if legs is None:
                return None
            self.stats["in"] += 1
            return self._on_road(Drive(vid, "in", legs, now, L), aid, road_c, dr)
        # (9 Nov, 3rd) at the kerb: SUMO stopped it behind its space as shown (kerb_pose): it pulls into it forwards
        # (an S); near the lane's end, where its space is not ahead of it, it goes on past it and backs into it
        k = self.kerb_pose(vid, aid, v)
        if k is None:
            return None
        kerb_c, hk = (k[0], k[1]), k[2]
        if math.dist(kerb_c, road_c) < 0.3:
            return None
        nx, ny = kerb_c[0] - road_c[0], kerb_c[1] - road_c[1]
        side = RIGHT if (dr[0] * ny - dr[1] * nx) < 0 else LEFT
        ahead = nx * dr[0] + ny * dr[1]
        off = max(L / 2 - REAR_AXLE_M, 0.0)
        made = {}

        def make(fwd, sc):
            legs, pauses = [], []
            if fwd == "arc":                # forwards into it, an S of two arcs
                pts = s_curve(road_c, dr, kerb_c, sc)
                if pts is None:
                    return None
                legs.append(Leg(pts, 2.5, signals=side))
                made[(fwd, sc)] = (legs, [0.0])
                return legs
            if fwd is not None and fwd != "arc" and sc < 0:   # on past it, and back into it on two arcs
                start = (road_c[0] + dr[0] * (ahead + fwd), road_c[1] + dr[1] * (ahead + fwd))
                legs.append(Leg(resample([road_c, start]), 2.5, signals=side))
                pauses.append(PAUSE_S)
                ra0 = (start[0] - dr[0] * off, start[1] - dr[1] * off)
                ra1 = (kerb_c[0] - hk[0] * off, kerb_c[1] - hk[1] * off)
                fw = s_curve(ra1, hk, ra0, -sc)
                if fw is None:
                    return None
                legs.append(Leg(list(reversed(fw)), V_BACK, reverse=True, signals=side, pivot=off))
                pauses.append(0.0)
                made[(fwd, sc)] = (legs, pauses)
                return legs
            if fwd is None:                 # forwards into it
                legs.append(Leg(hermite(road_c, dr, kerb_c, hk, scale=sc), 2.5, signals=side))
                pauses.append(0.0)
            else:                           # on past it, and back into it
                start = (road_c[0] + dr[0] * (ahead + fwd), road_c[1] + dr[1] * (ahead + fwd))
                legs.append(Leg(resample([road_c, start]), 2.5, signals=side))
                pauses.append(PAUSE_S)
                ra0 = (start[0] - dr[0] * off, start[1] - dr[1] * off)
                ra1 = (kerb_c[0] - hk[0] * off, kerb_c[1] - hk[1] * off)
                legs.append(Leg(hermite(ra0, (-dr[0], -dr[1]), ra1, (-hk[0], -hk[1]), scale=sc), V_BACK,
                                reverse=True, signals=side, pivot=off))
                pauses.append(0.0)
            made[(fwd, sc)] = (legs, pauses)
            return legs
        if ahead >= 4.0:
            cands = [(None, 1.0), ("arc", 4.5), (None, 0.8), (None, 1.2), ("arc", 5.0), ("arc", 4.0), (None, 0.6),
                     (None, 1.5), (6.0, 1.0), (6.0, 0.8)]
        else:
            cands = [(6.0, 1.0), (7.0, -4.5), (6.0, 0.8), (8.0, -4.5), (7.0, 1.0), (6.0, -4.5), (9.0, -4.5),
                     (5.0, 1.0), (7.0, 0.8), (8.0, 1.0), (8.0, -4.0), (10.0, -5.0), (7.0, -4.0), (9.0, 1.0)]
            if ahead > 1.0:     # (a car standing ahead on the lane: a tighter S straight into it, if that is clear)
                cands += [(None, 1.0), (None, 0.8), (None, 1.2), (None, 0.6)]
        legs = self._pick("kerb in", vid, L, cands, make, (kerb_c[0], kerb_c[1], _angle(*hk)), road=True)
        if legs is None:
            return None
        pauses = next(p_ for lg_, p_ in made.values() if lg_ is legs)
        self.stats["kerb_in"] += 1
        return self._on_road(Drive(vid, "in", legs, now, L, pauses=pauses), aid, road_c, dr)

    def _out(self, vid, aid, t0, v):
        """The drive out of its space, starting at t0, ending on the road where SUMO puts it back."""
        L = v.length
        try:
            lane_pos = self.c.vehicle.getLanePosition(vid)
        except Exception:  # noqa: BLE001
            return None
        dep = self.areas[aid].get("depart")
        if dep is None and aid in self.entrances:
            dep = self.entrances[aid].get("depart_pos")
        if dep is not None:         # (the parkingArea's departPos: SUMO puts it back there, past its way in)
            lane_pos = dep - 0.6
        (rx, ry), dr = self._road(aid, lane_pos + 0.6)
        road_c = (rx - dr[0] * L / 2, ry - dr[1] * L / 2)
        h = _dir(v.angle)
        if aid in self.entrances:                   # out of a building: its way in, the other way
            e = self.entrances[aid]
            e0 = (road_c[0] - dr[0] * ROAD_ON_M, road_c[1] - dr[1] * ROAD_ON_M)
            way = [*map(tuple, reversed(e["way"])), tuple(e["front"]), *map(tuple, reversed(e.get("approach", ()))),
                   e0, road_c]
            pts = fillet(_dedupe(way))
            n_ = _unit(e["front"][0] - e0[0], e["front"][1] - e0[1])
            side = LEFT if (dr[0] * n_[1] - dr[1] * n_[0]) < 0 else RIGHT
            leg = Leg(pts, V_DRIVE, signals=side)
            self.stats["building_out"] = self.stats.get("building_out", 0) + 1
            return self._on_road(Drive(vid, "out", [leg], t0, L), aid, road_c, dr)
        if "spaces" in self.areas[aid]:
            bay_c = (v.x - h[0] * L / 2, v.y - h[1] * L / 2)
            corners, n, across = self.lot_corners(aid, road_c, dr, bay_c, h)
            P0 = corners[-1]
            # back out into the aisle, turning so it faces the way out
            e0 = (road_c[0] - dr[0] * ROAD_ON_M, road_c[1] - dr[1] * ROAD_ON_M)
            way_out = self.areas[aid].get("way_out")
            if way_out:                                   # (its own way from its aisle to the road: places.json)
                ax = (h[1], -h[0])
                u_out = ax if (way_out[0][0] - P0[0]) * ax[0] + (way_out[0][1] - P0[1]) * ax[1] >= 0 \
                    else (-ax[0], -ax[1])
            elif across:
                u_out = (-n[0], -n[1])                    # along the aisle, towards the road
            else:
                k = (e0[0] - P0[0]) * dr[0] + (e0[1] - P0[1]) * dr[1]
                u_out = dr if k >= 0 else (-dr[0], -dr[1])
            # (on the right half of the aisle, as it will drive off: a car coming the other way passes)
            kr = self.keep_right_m(aid, bay_c, h)
            side = LEFT if (dr[0] * n[1] - dr[1] * n[0]) < 0 else RIGHT   # (turning onto the road: the other way)

            def make(off, r_back):
                # (9 Nov, 3rd) straight back until its nose is out, then a turn of a car's radius (r_back), then on
                # straight a little: the turn was 1.1 m before, its body swept through the cars beside it
                P = (P0[0] + u_out[1] * off, P0[1] - u_out[0] * off)
                back_end = (P[0] - u_out[0] * (r_back + BACK_STRAIGHT_M), P[1] - u_out[1] * (r_back + BACK_STRAIGHT_M))
                back = Leg(arcs([bay_c, P, back_end], r_back, keep_end=BACK_STRAIGHT_M), V_BACK, reverse=True)
                if way_out:
                    fwd = [back_end, P, *map(tuple, way_out), road_c]
                elif across:
                    g = corners[1]
                    strip = (g[0] - road_c[0]) * n[0] + (g[1] - road_c[1]) * n[1]
                    s_ = corners[2]
                    g2 = (e0[0] + n[0] * strip, e0[1] + n[1] * strip)
                    fwd = keep_right([back_end, P, s_, g2, e0, road_c], (2, 3), off=kr)
                else:
                    t2 = (P[0] + dr[0] * ((e0[0] - P[0]) * dr[0] + (e0[1] - P[1]) * dr[1]),
                          P[1] + dr[1] * ((e0[0] - P[0]) * dr[0] + (e0[1] - P[1]) * dr[1]))
                    fwd = keep_right([back_end, t2, e0, road_c], (1,), off=kr) if math.dist(back_end, t2) > 1.0 \
                        else [back_end, e0, road_c]
                go = Leg(arcs(_dedupe(fwd), TURN_R_M, keep_end=1.0), V_DRIVE, signals=side)
                return [back, go]
            cands = [(kr, 4.5), (kr, 5.0), (kr, 4.0), (kr / 2, 4.5), (kr / 2, 4.0), (0.0, 4.5), (0.0, 4.0), (kr, 3.5),
                     (0.0, 3.5), (kr, 3.0), (0.0, 3.0)]
            legs = self._pick("lot out", vid, L, cands, make, (bay_c[0], bay_c[1], v.angle))
            if legs is None:
                return None
            return self._on_road(Drive(vid, "out", legs, t0, L, pauses=[PAUSE_S, 0.0]), aid, road_c, dr)
        return None         # (at a kerb: kerb_out, once SUMO has put the car back)

    def kerb_out(self, vid, aid, now, front_pos):
        """(9 Nov, 3rd) The drive out of a kerb's space, SUMO's car on its lane at front_pos (held there): forwards
        (an S about its rear axle) onto its lane, its front ending at front_pos."""
        k = self._kerb.get(vid)
        if k is None:
            return None
        kerb_c, hk, L = (k[0], k[1]), k[2], k[5]
        (rx, ry), dr = self._road(aid, front_pos)
        road_c = (rx - dr[0] * L / 2, ry - dr[1] * L / 2)
        nx, ny = road_c[0] - kerb_c[0], road_c[1] - kerb_c[1]
        side = RIGHT if (dr[0] * ny - dr[1] * nx) < 0 else LEFT
        ahead = nx * dr[0] + ny * dr[1]
        off = max(L / 2 - REAR_AXLE_M, 0.0)
        made = {}

        def make(back_m, sc):
            legs, pauses = [], []
            if back_m in ("arc", "arcb"):   # (an S of two arcs about its rear axle: a hard turn out first;
                ra0 = (kerb_c[0] - hk[0] * off, kerb_c[1] - hk[1] * off)   # arcb: backing 0.8 m first)
                ra1 = (road_c[0] - dr[0] * off, road_c[1] - dr[1] * off)
                if back_m == "arcb":
                    rb = (ra0[0] - hk[0] * 0.8, ra0[1] - hk[1] * 0.8)
                    legs.append(Leg(resample([ra0, rb]), V_BACK, reverse=True, pivot=off))
                    pauses.append(PAUSE_S)
                    ra0 = rb
                pts = s_curve(ra0, hk, ra1, sc)
                if pts is None:
                    return None
                legs.append(Leg(pts, 2.5, signals=side, pivot=off))
                pauses.append(0.0)
                made[(back_m, sc)] = (legs, pauses)
                return legs
            if ahead < 3.0:                 # (SUMO's car not ahead of it: it backs out to it)
                ra0 = (kerb_c[0] - hk[0] * off, kerb_c[1] - hk[1] * off)
                ra1 = (road_c[0] - dr[0] * off, road_c[1] - dr[1] * off)
                legs.append(Leg(hermite(ra0, (-hk[0], -hk[1]), ra1, (-dr[0], -dr[1]), scale=sc), V_BACK,
                                reverse=True, signals=side, pivot=off))
                pauses.append(0.0)
                made[(back_m, sc)] = (legs, pauses)
                return legs
            start = kerb_c
            if back_m > 0.0:
                start = (kerb_c[0] - hk[0] * back_m, kerb_c[1] - hk[1] * back_m)
                legs.append(Leg(resample([(kerb_c[0] - hk[0] * off, kerb_c[1] - hk[1] * off),
                                          (start[0] - hk[0] * off, start[1] - hk[1] * off)]), V_BACK, reverse=True,
                                pivot=off))
                pauses.append(PAUSE_S)
            ra0 = (start[0] - hk[0] * off, start[1] - hk[1] * off)
            ra1 = (road_c[0] - dr[0] * off, road_c[1] - dr[1] * off)
            legs.append(Leg(hermite(ra0, hk, ra1, dr, scale=sc), 2.5, signals=side, pivot=off))
            pauses.append(0.0)
            made[(back_m, sc)] = (legs, pauses)
            return legs
        cands = [("arc", 4.5), ("arc", 4.0), ("arcb", 4.5), ("arc", 5.0), ("arcb", 4.0), ("arc", 3.5), ("arcb", 3.5),
                 (0.0, 1.0), (0.0, 0.8), (0.0, 1.2),
                 (0.8, 1.0), (0.8, 0.8), (0.0, 0.6), (0.8, 1.2), (0.0, 1.5)]
        legs = self._pick("kerb out", vid, L, cands, make, (kerb_c[0], kerb_c[1], _angle(*hk)))
        if legs is None:
            return None
        pauses = next(p_ for lg_, p_ in made.values() if lg_ is legs)
        dv = self._on_road(Drive(vid, "out", legs, now, L, pauses=pauses), aid, road_c, dr)
        dv.area, dv.why = aid, "put back by SUMO"
        return dv

    def _kerb_clear(self, vid, aid, lo, hi):
        """Whether no car SUMO drives is on the area's lane between lane positions lo and hi (bodies), nor coming up
        behind it so fast it could not stop before lo (_kerb_blocker: which)."""
        return self._kerb_blocker(vid, aid, lo, hi) is None

    def _kerb_blocker(self, vid, aid, lo, hi):
        c = self.c
        lane = self.areas[aid]["lane"]
        try:
            for oid in c.lane.getLastStepVehicleIDs(lane):
                if oid == vid:
                    continue
                f = c.vehicle.getLanePosition(oid)
                r = f - c.vehicle.getLength(oid)
                sp = c.vehicle.getSpeed(oid)
                if f >= lo - 1.0 - sp * sp / 6.0 and r <= hi + 1.0:
                    return oid
        except Exception:  # noqa: BLE001
            return "?"
        for oid, od in self.drives.items():         # (and the cars on drives there: SUMO does not see them)
            if oid == vid or getattr(od, "road", None) is None or od.road[2] != lane \
                    or getattr(od, "waits_for", None) == vid:
                continue                            # (one waiting for this one's place: it waits)
            x, y, _f, _v, _g = od.at_s(min(max(od.progress(self._now), 0.0), od.duration))
            pos = self.on_lane(od, x, y, within=4.0)
            if pos is not None and pos - od.length / 2 <= hi + 1.0 and pos + od.length / 2 >= lo - 1.0:
                return oid
        return None

    def _kerb_way_clear(self, vid, aid):
        """Whether a kerb car's way out (its space as shown to KERB_OUT_M past it, on its lane) is clear now (or it
        has waited KERB_WAIT_S x 4 for it: SUMO's turn)."""
        k = self.kerb_pose(vid, aid)
        if k is None or self._now - self._kerb_wait.get(vid, self._now) > 4 * KERB_WAIT_S:
            return True
        sf = self._shown_front(vid, k)
        return self._kerb_clear(vid, aid, sf - k[5] - 2.0, sf + KERB_OUT_M)

    def _kerb_gate(self, now):
        """Kerb cars taken out for a trip (PurposeDemand.resume_at): SUMO puts one back only when its way out is
        clear."""
        d = self.d
        for vid, t in list(d.resume_at.items()):
            aid = d.parked.get(vid)
            if aid is None or t - now > 0.6 or "spaces" in self.areas.get(aid, {}) or not self.visible(aid):
                continue
            if not self._kerb_way_clear(vid, aid):
                d.resume_at[vid] = now + 1.0
                self.stats["kerb_waits"] = self.stats.get("kerb_waits", 0) + 1

    def _kerb_leave(self, now, vehicles, ending=()):
        """Kerb cars SUMO has just put back on their lane (or is about to drive off): held there; once the stretch
        past its space is clear SUMO's car is moved to its end (vehicle.moveTo) and the shown car pulls out to it;
        not clear within KERB_WAIT_S: it backs out to where SUMO's car is."""
        c, d = self.c, self.d
        for vid in ending:
            # (9 Nov, 3rd) a car leaving a kerb place this module had not shown parked (SUMO had it parked from
            # before the viewer's drives began, with no place): from its place as shown, as any (it appeared on
            # the lane where a car turning in still stood)
            if vid in self._kerb or vid in self.drives or vid in d.parked:
                continue
            v = vehicles.get(vid)
            try:
                lane, f = c.vehicle.getLaneID(vid), c.vehicle.getLanePosition(vid)
            except Exception:  # noqa: BLE001, S112
                continue
            aid = next((a for a in self._kerb_lanes().get(lane, ()) if self.areas[a]["start"] - 1.0 <= f
                        <= self.areas[a]["end"] + 1.0), None)
            if aid is None or v is None or not self.visible(aid):
                continue
            if self.kerb_pose(vid, aid, v) is not None:
                self.stats["kerb_untracked"] = self.stats.get("kerb_untracked", 0) + 1
        for vid, k in list(self._kerb.items()):
            if vid in d.parked or vid in self.drives:
                continue
            v = vehicles.get(vid)
            aid = k[4]
            if vid in ending:                       # (SUMO puts it back at this step: still in its space here)
                if v is not None and abs(v.x) < 1e5:
                    self._show_parked(vid, k, v)
                continue
            if v is None or not self.visible(aid):
                self._kerb.pop(vid, None)
                continue
            w = self._kerb_wait.setdefault(vid, now)
            try:
                lane_now = c.vehicle.getLaneID(vid)
                f = c.vehicle.getLanePosition(vid)
            except Exception:  # noqa: BLE001
                self._kerb.pop(vid, None)
                continue
            if lane_now != self.areas[aid]["lane"]:
                self._kerb.pop(vid, None)            # (gone on already: as SUMO has it)
                self._kerb_wait.pop(vid, None)
                continue
            if vid not in self.held:
                try:
                    c.vehicle.setSpeed(vid, 0.0)
                    self._lock(vid)
                    self.stats["held_sumo"] += 1
                except Exception:  # noqa: BLE001, S110
                    pass
            sf = self._shown_front(vid, k)
            try:
                lane_len = c.lane.getLength(self.areas[aid]["lane"])
            except Exception:  # noqa: BLE001
                lane_len = sf + KERB_OUT_M
            target = min(sf + KERB_OUT_M, lane_len - 3.0)      # (moveTo at a lane's very end: SUMO put it on)
            dv = None
            if target - sf >= 4.0 and self._kerb_clear(vid, aid, sf - k[5] - 2.0, target):
                try:
                    c.vehicle.moveTo(vid, self.areas[aid]["lane"], target)
                    c.vehicle.setSpeed(vid, 0.0)    # (again: held there until the drive ends)
                    self._lock(vid)
                    dv = self.kerb_out(vid, aid, now, target)
                    self.stats["kerb_moved_on"] = self.stats.get("kerb_moved_on", 0) + 1
                except Exception:  # noqa: BLE001
                    dv = None
            elif (now - w >= KERB_WAIT_S or target - sf < 4.0) and (self._k % 4 == 0 or now - w < KERB_WAIT_S):
                n_touch = self.stats.get("ways_touching", 0)
                dv = self.kerb_out(vid, aid, now, f)
                if dv is not None and self.stats.get("ways_touching", 0) > n_touch and now - w < 6 * KERB_WAIT_S:
                    # (9 Nov, 3rd) backing out to where SUMO put it would touch a car parked behind: it waits on
                    self.stats["ways_touching"] = n_touch
                    self.touches.pop()
                    self._show_parked(vid, k, v)
                    continue
                self.stats["kerb_backed_out"] = self.stats.get("kerb_backed_out", 0) + 1
            else:
                self._show_parked(vid, k, v)        # (still in its space as shown)
                continue
            self._kerb_wait.pop(vid, None)
            if dv is None:
                self._kerb.pop(vid, None)
                continue
            dv.end_fixed = (0.0, 0.0)               # (SUMO's car is where the drive ends, held there)
            self._kerb.pop(vid, None)
            self.reserve(dv, now)
            self.drives[vid] = dv
            self.stats["kerb_out"] += 1

    def _kerb_lanes(self):
        """lane -> the kerbs (areas) on it."""
        if not hasattr(self, "_klanes"):
            self._klanes = {}
            for aid, a in self.areas.items():
                if a.get("roadside") and "spaces" not in a and not a.get("hidden"):
                    self._klanes.setdefault(a["lane"], []).append(aid)
        return self._klanes

    def _shown_front(self, vid, k):
        """The lane position of a kerb car's front as shown."""
        fronts, ln = self.kerb_fronts(k[4])
        a = self.areas[k[4]]
        i = round((k[3] - a["start"]) / ln) - 1
        return fronts[i] if 0 <= i < len(fronts) else k[3]

    def _show_parked(self, vid, k, v):
        v.x, v.y = k[0] + k[2][0] * v.length / 2, k[1] + k[2][1] * v.length / 2
        v.angle = _angle(*k[2])
        self._mark(v)

    def _gates_on(self, dv):
        """The doors and gates on a drive's way (a building's shutter, a car park's barrier arm: entrances.json
        "gates", boxes): [(name, open from s, shut at s)], open from DOOR_OPEN_M before the car's front reaches it
        until its back is DOOR_SHUT_M past it."""
        out = []
        t_off = 0.0
        for lg, pause in zip(dv.legs, dv.pauses, strict=False):
            for name, (cx, cy, hx, hy) in self.gates.items():
                inside = [i for i, (x, y) in enumerate(lg.pts) if abs(x - cx) <= hx + 1.0 and abs(y - cy) <= hy + 1.0]
                if not inside:
                    continue
                a, b = lg.cum[inside[0]], lg.cum[inside[-1]]
                half = dv.length / 2

                def t_at(m, lg=lg):
                    m = min(max(m, 0.0), lg.length)
                    j = next((i for i, c_ in enumerate(lg.cum) if c_ >= m), len(lg.cum) - 1)
                    return lg.t[j]
                out.append((name, t_off + t_at(a - half - DOOR_OPEN_M), t_off + t_at(b + half + DOOR_SHUT_M)))
            t_off += lg.duration + pause
        return out

    def _on_road(self, dv, aid, road_c, dr):
        """The road the drive starts or ends on: its lane, and where on it (road_c: the car's middle there)."""
        lane = self.areas[aid]["lane"]
        try:
            pos = self.c.lane.getLength(lane)
            sh = self.c.lane.getShape(lane)
            best = None
            acc = 0.0
            for a, b in itertools.pairwise(sh):
                seg = math.dist(a, b) or 1e-9
                u = max(0.0, min(1.0, ((road_c[0] - a[0]) * (b[0] - a[0]) + (road_c[1] - a[1]) * (b[1] - a[1]))
                                 / seg ** 2))
                q = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
                dd = math.dist(q, road_c)
                if best is None or dd < best[0]:
                    best = (dd, acc + u * seg)
                acc += seg
            pos = best[1] if best else pos
        except Exception:  # noqa: BLE001
            pos = None
        dv.road = (road_c, dr, lane, pos)
        return dv

    def on_lane(self, dv, x, y, within=ON_LANE_M):
        """Whether a body with its middle at x, y stands on the drive's lane (within ON_LANE_M of its line), and how
        far along the lane (None: off it)."""
        road = getattr(dv, "road", None)
        if road is None or road[3] is None:
            return None
        (rx, ry), (ux, uy), _lane, pos = road
        along = (x - rx) * ux + (y - ry) * uy
        side = abs((x - rx) * uy - (y - ry) * ux)
        return pos + along if side < within else None

    def exit_now(self, vid, aid):
        """PurposeDemand: car vid is taken out of its space (area aid) for a trip now: the seconds before SUMO may
        put it on the road (the drive out), or 0 (no drive: SUMO at once)."""
        if not self.visible(aid):
            return 0.0
        if "spaces" not in self.areas[aid] and not self.areas[aid].get("hidden"):
            # (a kerb: SUMO puts it back once its way out is clear (_kerb_gate), the drive out starts then:
            # _kerb_leave)
            self._kerb_wait.setdefault(vid, self._now)
            return 0.5
        if vid in self.drives:
            return self._then_out(vid, aid)
        v = self._cur.get(vid) if getattr(self, "_cur", None) else None
        if v is None:
            return 0.0
        try:
            dv = self._out(vid, aid, self._now, v)
        except Exception:  # noqa: BLE001
            dv = None
            self.stats["failed"] += 1
        if dv is None:
            return 0.0
        dv.area, dv.why = aid, "taken out"
        self.reserve(dv, self._now)
        self.drives[vid] = dv
        self._count_out(aid)
        return dv.duration + dv.lag + EXIT_GUARD_S

    def _then_out(self, vid, aid):
        """(9 Nov, 3rd) A car taken out for a new trip while still on its drive in: it goes on into its space, stands
        a moment and drives out (SUMO keeps it in its space meanwhile: the seconds returned), rather than turning back
        on its way (it backed all the way along a big car park's aisle before)."""
        dv = self.drives[vid]
        if dv.kind != "in":
            return 0.0
        now = self._now
        x, y, face, _v, _g = dv.at_s(dv.duration)
        hh = _dir(face)
        L = dv.length
        here = type("V", (), {})()
        here.x, here.y, here.angle, here.length = x + hh[0] * L / 2, y + hh[1] * L / 2, face, L
        left = max(dv.duration - dv.progress(now), 0.0)
        try:
            od = self._out(vid, aid, now + left + PARKED_PAUSE_S, here)
        except Exception:  # noqa: BLE001
            od = None
            self.stats["failed"] += 1
        if od is None:
            return 0.0
        nd = Drive(vid, "out", dv.legs + od.legs, dv.t0, L,
                   pauses=dv.pauses[:-1] + [dv.pauses[-1] + PARKED_PAUSE_S] + od.pauses)
        nd.lag, nd.rate, nd.held_s, nd.ignored = dv.lag, dv.rate, dv.held_s, dv.ignored
        nd.area, nd.why, nd.road = aid, "taken out on its way in", od.road
        nd.n_in = len(dv.legs)
        if hasattr(dv, "gates"):
            del dv.gates
        self.drives[vid] = nd
        self.stats["in_then_out"] = self.stats.get("in_then_out", 0) + 1
        self._count_out(aid)
        return nd.duration - nd.progress(now) + EXIT_GUARD_S

    def _count_out(self, aid):
        self.stats["out" if "spaces" in self.areas[aid] else "kerb_out"] += 1

    # ------------------------------------------------------------------ every step
    def step(self, now, vehicles):
        """Moves the cars on their drives: their front bumper, heading, speed and lights in `vehicles` (this step's,
        about to be shown). Also marks every car standing in a space so the viewer keeps its heading as SUMO has it."""
        c, d = self.c, self.d
        self._cur, self._now = vehicles, now
        self._k += 1
        # (9 Nov, 4th) a car newly parked: its kerb place worked out afresh (a place kept from its stay before put it
        # where another car now stood: 0.37 m apart in CARLA)
        pk = set(d.parked)
        for vid in pk - getattr(self, "_pk_prev", set()):
            if vid not in self.drives:
                self._kerb.pop(vid, None)
        self._pk_prev = pk
        if self._k % 20 == 0:
            self._passing &= set(vehicles)
        try:
            starting = c.simulation.getParkingStartingVehiclesIDList()
            ending = set(c.simulation.getParkingEndingVehiclesIDList())
        except Exception:  # noqa: BLE001
            starting, ending = [], set()
        for vid in starting:
            aid = d.parked.get(vid)
            v = vehicles.get(vid)
            if aid is None or v is None or vid in self.drives:
                continue
            if not self.visible(aid):
                self.stats["skipped_hidden"] += 1
                continue
            try:
                dv = self._in(vid, aid, now, v)
            except Exception:  # noqa: BLE001 - a drive that cannot be made: SUMO's jump, as before
                dv = None
                self.stats["failed"] += 1
            if dv is not None:
                dv.area, dv.why = aid, "in"
                self.reserve(dv, now)
                self.drives[vid] = dv
                self._room_for(vid, aid, now, dv.duration + dv.lag)
        # the drive out of a car whose time there ends (the stop's duration): started that much earlier
        if self._k % 4 == 0:
            for vid, aid in list(d.parked.items()):
                if vid in self.drives or not self.visible(aid):
                    continue
                try:
                    # (9 Nov) the stop's duration, as SUMO gives it for a car in its space, is what is LEFT of it:
                    # its end is now + that (arrival + it was early: the drive out reached the road before SUMO put
                    # the car back, and waited there)
                    st = c.vehicle.getStops(vid, 1)
                    s0 = st[0] if st else None
                    end = now + s0.duration if s0 is not None and s0.duration < 1e6 and s0.arrival >= 0 \
                        else float("inf")
                except Exception:  # noqa: BLE001
                    end = float("inf")
                self.release[vid] = end
        for vid, end in list(self.release.items()):
            if vid not in d.parked:
                del self.release[vid]
                continue
            v = vehicles.get(vid)
            if vid in self.drives or v is None or end == float("inf"):
                continue
            if now < end - 45.0:                       # (no drive out takes that long)
                continue
            aid = d.parked[vid]
            if "spaces" not in self.areas[aid] and not self.areas[aid].get("hidden"):
                if end - now <= 1.5:                # (a kerb: SUMO puts it back once its way out is clear)
                    self._kerb_wait.setdefault(vid, now)
                    if not self._kerb_way_clear(vid, aid):
                        try:
                            c.vehicle.setParkingAreaStop(vid, aid, duration=end - now + 2.0)
                            self.release[vid] = end + 2.0
                            self.stats["kerb_stays_longer"] = self.stats.get("kerb_stays_longer", 0) + 1
                        except Exception:  # noqa: BLE001, S110
                            pass
                continue
            made = self._outs.get(vid)
            if made is not None and made[0] > now - 8.0 and made[2] == aid:
                dv = made[1]                        # (made a moment ago: the same way, from now)
                if dv is not None:
                    dv.t0 = now
            else:
                try:
                    dv = self._out(vid, aid, now, v)
                except Exception:  # noqa: BLE001
                    dv = None
                    self.stats["failed"] += 1
                self._outs[vid] = (now, dv, aid)
            if dv is None:
                self.release[vid] = float("inf")
                continue
            if now >= end - dv.duration - EXIT_GUARD_S:
                dv.area, dv.why = aid, "time up" if now <= end - dv.duration else f"time up, {now - end + dv.duration:.0f} s late"
                self._outs.pop(vid, None)
                self.reserve(dv, now)
                self.drives[vid] = dv
                self._count_out(aid)
                if dv.lag > 0:
                    self._sumo_waits(vid, dv, now)
        self._kerb_leave(now, vehicles, ending)
        self._kerb_gate(now)
        self.give_way(now, vehicles)
        self._doors_step(now)
        # move them
        for vid, dv in list(self.drives.items()):
            v = vehicles.get(vid)
            if v is None:
                self.drives.pop(vid)
                self._release_hold(vid)
                continue
            if dv.kind == "in" and vid not in d.parked and vid not in ending and abs(v.x) < 1e5 \
                    and vid in self._kerb and not getattr(dv, "kerb_after", False):
                # (9 Nov, 3rd) at a kerb: it goes on into its space, SUMO's car held where SUMO put it back; then
                # out of it as any (_kerb_leave)
                dv.kerb_after = True
                try:
                    c.vehicle.setSpeed(vid, 0.0)
                    self._lock(vid)
                except Exception:  # noqa: BLE001, S110
                    pass
                self.stats["turned_back"] = self.stats.get("turned_back", 0) + 1
            if dv.kind == "in" and vid not in d.parked and vid not in ending and abs(v.x) < 1e5 \
                    and not getattr(dv, "kerb_after", False):
                # SUMO took it out again before its drive in was over (a short stop): back from where it is shown
                # to SUMO's car, which waits for it
                # (9 Nov) the way it came, the other way, back to where it left the road (that way is clear: a
                # straight line to SUMO's car went through the cars parked at a kerb, seen in CARLA), then to SUMO's car
                h1 = _dir(v.angle)
                sumo_c = (v.x - h1[0] * v.length / 2, v.y - h1[1] * v.length / 2)
                legs = _retrace(dv, min(max(dv.progress(now), 0.0), dv.duration))
                x, y, f0, _v, _g = legs[-1].at(legs[-1].duration) if legs else dv.at(now)
                h0 = _dir(f0)                               # (its facing there)
                if math.dist((x, y), sumo_c) > 0.5:
                    # (9 Nov, 3rd) forwards if SUMO's car is ahead of its facing, else backing to it; facing as it
                    # is (before: its way's heading taken for its facing, it spun round on a loop, 5.6 m off at the
                    # end, seen at a kerb)
                    ahead = (sumo_c[0] - x) * h0[0] + (sumo_c[1] - y) * h0[1] >= 0.0
                    if ahead:
                        legs.append(Leg(hermite((x, y), h0, sumo_c, h1), 2.5))
                    else:
                        legs.append(Leg(hermite((x, y), (-h0[0], -h0[1]), sumo_c, (-h1[0], -h1[1])), V_BACK,
                                        reverse=True))
                if not legs:
                    legs = [Leg(resample([(x, y), sumo_c]), 1.0)]
                nd = Drive(vid, "out", legs, now, dv.length, pauses=[PAUSE_S] * (len(legs) - 1) + [0.0])
                nd.area, nd.why = dv.area, "left before it had parked"
                self.drives[vid] = dv = nd
                self.stats["turned_back"] = self.stats.get("turned_back", 0) + 1
                try:
                    c.vehicle.setSpeed(vid, 0.0)
                    self._lock(vid)
                except Exception:  # noqa: BLE001, S110 - it left meanwhile
                    pass
            if dv.kind == "out" and vid not in d.parked and vid not in ending and getattr(dv, "end_fixed", None) \
                    is None:
                # SUMO has put it back on the road (its place now, before this step's drawing): the drive ends there
                h = _dir(v.angle)
                actual = (v.x - h[0] * dv.length / 2, v.y - h[1] * dv.length / 2)
                last = dv.legs[-1]
                end = last.pts[-1]
                off = (actual[0] - end[0], actual[1] - end[1])
                if not dv.done(now) and abs(v.x) < 1e5:
                    if last.end_offset is None and math.hypot(*off) > 0.05:
                        self.stats["ends_moved"] = self.stats.get("ends_moved", 0) + 1
                    last.end_offset = off               # (again each step: SUMO's car may still roll to a stop)
                else:
                    dv.end_fixed = off
            if dv.kind == "out" and vid in ending:
                # (9 Nov, 4th) also when the drive has ended: SUMO's car stands there one step, it does not start
                # moving before it is shown (a 1.5 m step out of the Square)
                try:                                   # SUMO put it back first: its car waits for the drive
                    c.vehicle.setSpeed(vid, 0.0)
                    self._lock(vid)
                    self.stats["held_sumo"] += 1
                    if "spaces" in self.areas.get(dv.area, {}) and getattr(dv, "end_fixed", None) is None:
                        # (9 Nov, 4th) SUMO puts a car back from a car park 1.3 m beside its lane's line (the
                        # Square: a 1.5 m step at the hand-over): its car moved to where the drive ends
                        ex, ey, ef, _v, _g = dv.legs[-1].at(dv.legs[-1].duration)
                        eh = _dir(ef)
                        c.vehicle.moveToXY(vid, "", -1, ex + eh[0] * dv.length / 2, ey + eh[1] * dv.length / 2, ef,
                                           keepRoute=1)
                        dv.legs[-1].end_offset = None
                        dv.end_fixed = (0.0, 0.0)
                        self.stats["ends_matched"] = self.stats.get("ends_matched", 0) + 1
                except Exception:  # noqa: BLE001, S110 - it left meanwhile
                    pass
            if dv.done(now):
                if dv.kind == "out" and vid in ending:
                    pass            # (SUMO reports the car in its space at the step it puts it back: still shown)
                elif dv.kind == "out" and abs(v.x) > 1e5:
                    pass            # (SUMO has no place for it at this step yet: still shown where the drive ended)
                elif dv.kind == "in" and getattr(dv, "kerb_after", False):
                    self.drives.pop(vid)            # (in its space as shown, SUMO's car still held: _kerb_leave)
                    k_ = self._kerb.get(vid)
                    if k_ is not None:
                        self._show_parked(vid, k_, v)
                    continue
                elif dv.kind == "in" or vid not in d.parked:
                    if dv.kind == "out":
                        x, y, face, _s, _g = dv.at(now)
                        hx, hy = _dir(face)
                        seam = round(math.dist((x + hx * dv.length / 2, y + hy * dv.length / 2), (v.x, v.y)), 2)
                        self.stats["seam_max_m"] = max(self.stats["seam_max_m"], seam)
                        self.seams.append({"vid": vid, "area": dv.area, "seam_m": seam, "held": vid in self.held,
                                           "waited": bool(getattr(dv, "waited", False)), "why": dv.why,
                                           "sumo_speed": round(v.speed, 2), "late_s": round(dv.progress(now) - dv.duration, 1)})
                        del self.seams[:-200]
                    self.drives.pop(vid)
                    self._release_hold(vid)
                    if dv.kind == "in":
                        self._mark(v)
                    continue
                elif not getattr(dv, "waited", False):  # its drive is over, SUMO has not put it back yet
                    dv.waited = True
                    self.stats["waited_at_road"] += 1
            if dv.kind == "out" and vid not in d.parked and vid not in self.held and not dv.done(now):
                try:
                    c.vehicle.setSpeed(vid, 0.0)
                    self._lock(vid)
                    self.stats["held_sumo"] += 1
                except Exception:  # noqa: BLE001, S110 - it left meanwhile
                    pass
            x, y, face, speed, sig = dv.at(now)
            if dv.rate < 0.999 and not dv.done(now):        # giving way: slower, braking
                speed *= dv.rate
                sig = (sig & ~BACK) | BRAKE if dv.rate < 0.05 else sig | BRAKE
            hx, hy = _dir(face)
            v.x, v.y = x + hx * dv.length / 2, y + hy * dv.length / 2
            v.angle, v.speed = face, speed
            v.signals = (v.signals & ~OWN_BITS) | sig
            v.lane_angle = None
            v.fixed = True
        for vid, aid in d.parked.items():               # standing in a space: its heading is SUMO's
            if vid not in self.drives and self.visible(aid) and "spaces" in self.areas[aid]:
                v = vehicles.get(vid)
                if v is not None:
                    self._mark(v)
            elif vid not in self.drives and self.visible(aid) and self.areas[aid].get("roadside"):
                v = vehicles.get(vid)           # (at a kerb: where it is shown, kerb_pose)
                if v is not None and abs(v.x) < 1e5:
                    k = self.kerb_pose(vid, aid, v)
                    if k is not None:
                        v.x, v.y = k[0] + k[2][0] * v.length / 2, k[1] + k[2][1] * v.length / 2
                        v.angle = _angle(*k[2])
                        self._mark(v)
        if self._k % 20 == 0:
            for vid in [x for x, m_ in self._outs.items() if m_[0] < now - 60.0]:
                del self._outs[vid]
        self.prev = {vid: (v.x, v.y, v.angle) for vid, v in vehicles.items() if vid not in d.parked}

    # ------------------------------------------------------------------ giving way
    def reserve(self, dv, now):
        """(9 Nov) A new drive waits until its way is clear of the drives already on theirs: it starts the first
        half second (up to RESERVE_MAX_S) for which its body, all along its drive, keeps GAP_M from every other
        drive's body as that drive will be at the same moment (and from where one ends, for END_HOLD_S after: SUMO
        takes it over there). A car going out waits in its space; one coming in waits on the road where SUMO
        stopped it (the cars behind it wait too: _hold_followers). Its wait goes into its lag."""
        if self.areas.get(dv.area or "", {}).get("hidden"):
            return 0.0          # (a building's way in: one at a time anyway)
        others = []
        for oid, od in self.drives.items():
            if oid == dv.vid:
                continue
            s0 = od.progress(now)
            others.append((od, s0))
        if not others:
            return 0.0
        n = int((dv.duration + END_HOLD_S) / LOOK_DT) + 1
        mine = [circles(*dv.at_s(min(k * LOOK_DT, dv.duration))[:3], dv.length) for k in range(n)]
        start = mine[0]
        theirs = []
        horizon = int((RESERVE_MAX_S + dv.duration + END_HOLD_S) / LOOK_DT) + 1
        for od, s0 in others:
            end_at = (od.duration - s0) + END_HOLD_S        # (seconds from now: it is the road's after that)
            row = []
            for k in range(horizon):
                u = k * LOOK_DT
                if u > end_at:
                    row.append(None)
                    continue
                x, y, face, _v, _g = od.at_s(min(s0 + u, od.duration))
                row.append(circles(x, y, face, od.length))
            theirs.append(row)
        for w in range(int(RESERVE_MAX_S / LOOK_DT) + 1):
            ok = True
            for k in range(w + n):
                body = start if k < w else mine[k - w]
                for row in theirs:
                    ob = row[k] if k < len(row) else None
                    if ob is not None and _gap(body, ob) < GAP_M:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if w:
                    dv.lag += w * LOOK_DT
                    self.stats["started_later"] = self.stats.get("started_later", 0) + 1
                    self.stats["started_later_s"] = round(self.stats.get("started_later_s", 0.0) + w * LOOK_DT, 1)
                return w * LOOK_DT
        self.stats["no_clear_start"] = self.stats.get("no_clear_start", 0) + 1
        return 0.0

    def give_way(self, now, vehicles):
        """(9 Nov) The cars on drives see each other and the cars on the road beside the car park. A drive starts
        when its way is clear of the other drives (reserve); on its way, it still looks LOOK_S ahead: where its body
        would come closer than GAP_M to another car's body as it is now (a drive held up, a car SUMO drives), it
        slows (rate: its share of its own pace, standing if within a second, back up by RATE_UP a step) and its
        timetable slips (Drive.lag). Two drives each in the other's way: the one that started first goes. Held more
        than GIVE_UP_S by the same car (a car that does not move): it goes on. Cars standing in their spaces are not
        looked at (the ways keep off them)."""
        dt = max(min(now - getattr(self, "_t_prev", now - 0.5), 2.0), 0.0) if self.drives else 0.0
        self._t_prev = now
        if not self.drives:
            if self.followers:
                self._hold_followers(now, {})
            return
        cur = {}                                       # every body near a drive, as it is now: three circles
        for vid, dv in self.drives.items():
            x, y, face, _v, _g = dv.at_s(min(max(dv.progress(now), 0.0), dv.duration))
            cur[vid] = (circles(x, y, face, dv.length), True)
        mids = [c_[0][1] for c_ in cur.values()]
        for vid, v in vehicles.items():
            if vid in cur or vid in self.d.parked or abs(v.x) > 1e5:
                continue
            h = _dir(v.angle)
            m = (v.x - h[0] * v.length / 2, v.y - h[1] * v.length / 2)
            if any(abs(m[0] - p[0]) < NEAR_M and abs(m[1] - p[1]) < NEAR_M for p in mids):
                cur[vid] = (circles(m[0], m[1], v.angle, v.length), False)
        still = {vid for vid, dv in self.drives.items() if dv.rate < 0.05 or dv.done(now) or dv.progress(now) < 0}
        still |= {vid for vid, (_b, is_drive) in cur.items() if not is_drive and vehicles[vid].speed < 0.5}
        want = {}
        for vid, dv in self.drives.items():
            s = dv.progress(now)
            if s < 0.5 and dv.kind == "in" and self._kerb_place_taken(vid, dv, now):
                # (9 Nov, 4th) SUMO gives a kerb place to a car arriving as the car in it leaves: it waits on its
                # lane until that one has pulled out (both were shown in the same place: 0.37 m apart in CARLA)
                dv.lag += dt
                dv.rate = 0.0
                dv.waits_for = self._kerb_place_taken(vid, dv, now)
                self.stats["kerb_place_wait_s"] = round(self.stats.get("kerb_place_wait_s", 0.0) + dt, 1)
                continue
            if s < 0.5 and self.narrow(dv.area) and self._aisle_busy(vid, dv, now):
                # (9 Nov, 3rd) an aisle too narrow for two (the Parking du Square's, 4.3 m): it waits, in its space
                # or at the road, until the car on its drive there is through (they met in it, 62 steps, seed 103)
                dv.lag += dt
                dv.rate = 0.0
                if dv.kind == "out":
                    self._sumo_waits(vid, dv, now)
                self.stats["aisle_wait_s"] = round(self.stats.get("aisle_wait_s", 0.0) + dt, 1)
                continue
            if s >= dv.duration or s < 0.0:        # (done, or not started: waiting for its way to be clear)
                continue
            if self.areas.get(dv.area, {}).get("hidden"):
                continue        # (a building's way in, see reserve)
            hit = None
            if dv.kind == "out" and s < 0.3 and "spaces" not in self.areas.get(dv.area, {}):
                # (9 Nov, 3rd) at a kerb, before it pulls out: its lane clear beside it and behind it for the whole
                # S (SUMO's cars do not see it: one stopped beside it to park, inside its way)
                (rx, ry), (ux, uy), _lane, pos_ = dv.road
                x0_, y0_, _f, _v, _g = dv.at_s(0.0)
                lo = pos_ + (x0_ - rx) * ux + (y0_ - ry) * uy - dv.length / 2 - 2.0
                b = self._kerb_blocker(vid, dv.area, lo, pos_ + dv.length / 2) if dv.area in self.areas else None
                hit = (1, b, b in self.drives) if b is not None and b != "?" else None
            last = cur[vid][0][1]
            for k in range(1 if hit is None else LOOK_N + 1, LOOK_N + 1):
                t_ = min(s + k * LOOK_DT, dv.duration)
                x, y, face, _v, _g = dv.at_s(t_)
                body = circles(x, y, face, dv.length)
                mx, my = x - last[0], y - last[1]           # its way here (forwards or backing)
                last = (x, y)
                if abs(mx) + abs(my) < 1e-3:
                    continue
                for oid, (ob, is_drive) in cur.items():
                    if oid == vid or oid in dv.ignored:
                        continue
                    g = _gap(body, ob)
                    # (only a car it drives into: closer than now, and standing, or in front of it on its way; one
                    # coming up behind it gives way itself)
                    if g < GAP_M and _gap(cur[vid][0], ob) >= g - 1e-6 \
                            and (oid in still or (ob[1][0] - x) * mx + (ob[1][1] - y) * my > 0.0):
                        hit = (k, oid, is_drive)
                        break
                if hit is None and dv.kind == "out" and self.on_lane(dv, *cur[vid][0][1]) is None \
                        and self.on_lane(dv, x, y) is not None:
                    hit = self._merge_gap(dv, k)       # about to pull onto the road: a gap in its traffic
                if hit is not None or t_ >= dv.duration:
                    break
            want[vid] = hit
        for vid, hit in list(want.items()):           # two drives each in the other's way: the first one goes
            if hit is not None and hit[2]:
                other = want.get(hit[1])
                if other is not None and other[1] == vid and (self.drives[vid].t0, vid) < (self.drives[hit[1]].t0,
                                                                                            hit[1]):
                    want[vid] = None
        self._want = want
        # who waits on the road's traffic (a queue, a gap to merge into, SUMO to take it over), directly or behind
        # another that does: never gives up waiting (the road moves by itself)
        road = {vid for vid, dv in self.drives.items() if dv.done(now)}
        road |= {vid for vid, hit in want.items() if hit is not None and not hit[2] and hit[1] not in self.followers}
        grew = True
        while grew:
            more = {vid for vid, hit in want.items() if hit is not None and hit[2] and hit[1] in road} - road
            road |= more
            grew = bool(more)
        for vid, hit in want.items():
            dv = self.drives[vid]
            if hit is None:
                target = 1.0
                dv.blocker, dv.blocked_s = None, 0.0
            else:
                target = min(max((hit[0] - 2) / (LOOK_N - 2), 0.0), 1.0)
                if dv.blocker == hit[1]:
                    dv.blocked_s += dt
                else:
                    dv.blocker, dv.blocked_s = hit[1], 0.0
            # (a car of the road it waits for goes on by itself: a queue, the traffic to merge into. Given up on
            # only a car that this module holds itself, behind another drive: that could be waiting in a ring)
            # (9 Nov, 4th) a car SUMO drives in its way, held by this module for another drive: waited for too (given
            # up on after 20 s, the drive went through it at a kerb)
            patient = vid in road or (hit is not None and not hit[2])
            if hit is not None and not hit[2] and vid in self.held and dv.blocked_s > 15.0 \
                    and not getattr(dv, "moved_on", False):
                self._unring(vid, dv, hit[1])
            if hit is not None and not patient and dv.blocked_s > (GIVE_UP_DRIVE_S if hit[2] else GIVE_UP_S):
                dv.ignored.add(hit[1])
                self.stats["gave_up_waiting"] = self.stats.get("gave_up_waiting", 0) + 1
            rate = min(dv.rate + RATE_UP, target)
            dv.lag += (1.0 - rate) * dt
            if rate < 0.999 and dv.kind == "out":
                self._sumo_waits(vid, dv, now)
            if rate < 0.05:
                dv.held_s += dt
                self.stats["held_s"] = round(self.stats.get("held_s", 0.0) + dt, 1)
            if hit is not None and dv.rate >= 0.999:
                self.stats["gave_way"] = self.stats.get("gave_way", 0) + 1
            dv.rate = rate
        self._hold_followers(now, cur)

    def _unring(self, vid, dv, oid):
        """(9 Nov, 4th) A drive out waits for a car SUMO drives standing in its way, which waits behind this drive's
        own car held on the road (SUMO put it back early): each waits for the other (2 minutes at the Marche). SUMO's
        car is moved on 8 m (vehicle.moveTo) and the drive's end with it (the end moved over its last metres)."""
        c = self.c
        try:
            lead = c.vehicle.getLeader(oid, 30.0)
            if not lead or lead[0] != vid:
                return
            lane, pos = c.vehicle.getLaneID(vid), c.vehicle.getLanePosition(vid)
            to = min(pos + 8.0, c.lane.getLength(lane) - 1.0)
            if to - pos < 3.0:
                return
            c.vehicle.moveTo(vid, lane, to)
            c.vehicle.setSpeed(vid, 0.0)
            (x0, y0), _u0 = self._road_lane(lane, pos)
            (x1, y1), _u1 = self._road_lane(lane, to)
            last = dv.legs[-1]
            prev = last.end_offset or (0.0, 0.0)
            last.end_offset = (prev[0] + x1 - x0, prev[1] + y1 - y0)
            dv.end_fixed = (0.0, 0.0)
            dv.moved_on = True
            self.stats["moved_on"] = self.stats.get("moved_on", 0) + 1
        except Exception:  # noqa: BLE001, S110 - as SUMO has it
            pass

    def _road_lane(self, lane, pos):
        """(x, y on a lane at pos, heading unit)."""
        L = self.c.lane.getLength(lane)
        edge, idx = lane.rsplit("_", 1)
        p = min(max(pos, 0.0), L)
        x0, y0 = self.c.simulation.convert2D(edge, p, int(idx))
        x1, y1 = self.c.simulation.convert2D(edge, min(p + 1.0, L), int(idx))
        return (x0, y0), _unit(x1 - x0, y1 - y0)

    def _merge_gap(self, dv, k, horizon_s=None):
        """A drive out about to pull onto its lane: (k, car, False) for a car SUMO drives on the lane that would
        come up on it (behind it, within MERGE_S at its speed, or MERGE_M), else None."""
        c = self.c
        lane, pos = dv.road[2], dv.road[3]
        try:
            for oid in c.lane.getLastStepVehicleIDs(lane):
                if oid == dv.vid or oid in self.d.parked or oid in self.followers or oid in self.drives:
                    continue        # (a car on a drive is not where SUMO has it)
                front = c.vehicle.getLanePosition(oid)
                join = ROAD_ON_M if "spaces" in self.areas.get(dv.area, {}) else 4.0
                behind = pos - join - front                # (the drive joins about that far before its end)
                look = MERGE_S if horizon_s is None else horizon_s + MERGE_S
                if -dv.length <= behind <= max(MERGE_M, c.vehicle.getSpeed(oid) * look):
                    return (k, oid, False)
                if -(ROAD_ON_M + 2 * dv.length) <= behind < -dv.length and c.vehicle.getSpeed(oid) < 1.0:
                    return (k, oid, False)                  # (a queue standing where it would join)
        except Exception:  # noqa: BLE001, S110
            pass
        return None

    def _hold_followers(self, now, cur):
        """A car of a drive standing or crawling on the road (turning in, or about to join it): SUMO does not see it
        (for SUMO it is in its space), so the cars SUMO drives up behind it on its lane are stopped until it is off
        the lane (SUMO's setSpeed, as for a car waiting to turn into a car park)."""
        c = self.c
        standing: dict[str, list] = {}
        by_edge: dict[str, list] = {}       # (the other lanes: a car coming to park there, changing lanes onto it)
        for vid, dv in self.drives.items():
            if self.areas.get(dv.area, {}).get("hidden"):
                continue
            # (9 Nov, 3rd) at a kerb too: the car backing into its space from its lane, or pulled out onto it, is
            # not on it for SUMO, whose cars drove through it
            due = self.release.get(vid, self.d.resume_at.get(vid, getattr(dv, "done_at", float("inf"))))
            if dv.done(now) and now > due + OVERDUE_S:
                continue        # (SUMO cannot put it back: the cars behind are let go, SUMO sorts it out)
            x, y, _f, _v, _g = dv.at_s(min(max(dv.progress(now), 0.0), dv.duration))
            # (at a kerb, a car pulling out: from when it moves, beside its lane already; SUMO's cars coming up
            # behind it stopped beside it to park, inside its way)
            kerb_out = dv.kind == "out" and "spaces" not in self.areas.get(dv.area, {}) and dv.progress(now) > 0.5
            pos = self.on_lane(dv, x, y, within=4.0 if kerb_out else ON_LANE_M)
            if pos is not None:     # (its rear; and how close a car may be to be held: SUMO puts a car back in front)
                # (9 Nov, 3rd) only a car wholly behind it: one beside it, or inside it (SUMO put a car on the
                # road where a car turning in still stood, seen at a kerb), is let go on: the drive waits for it
                # (9 Nov, 4th) a car waiting at the road for SUMO to take it over: one close behind it is held too
                # (before: let through, it drove through it) and asked to pass it on the next lane (SUMO needs
                # that room to put the car back)
                standing.setdefault(dv.road[2], []).append((pos - dv.length / 2, 0.3,
                                                            dv.kind == "out" and dv.done(now), vid))
                by_edge.setdefault(dv.road[2].rsplit("_", 1)[0], []).append((pos + dv.length / 2, dv.area))
        for aid_ in {dv.area for dv in self.drives.values() if not dv.done(now) and self.narrow(dv.area)}:
            # (9 Nov, 4th) a car park that takes one car at a time: a car coming to park there waits on the road
            # before its stop while a car is on its way in it (it parked where the one before had just turned in)
            a_ = self.areas[aid_]
            by_edge.setdefault(a_["edge"], []).append((float(a_["end"]) + 0.5, aid_))
        keep = set()
        if standing:
            for oid, (_body, is_drive) in cur.items():
                if is_drive or oid in self.held:
                    continue
                try:
                    lane = c.vehicle.getLaneID(oid)
                    front = c.vehicle.getLanePosition(oid)
                    if lane in standing:
                        # (9 Nov, 4th) a car still moving up is held even with its nose beside the standing car's
                        # tail (SUMO's sublanes let it come up beside it in the lane: 0.45 m into it at the Square);
                        # one standing there already (put on the road there by SUMO) is let go
                        moving = c.vehicle.getSpeed(oid) > 0.5
                        if not any((-4.0 if moving else gap) <= rear - front <= HOLD_BEHIND_M
                                   for rear, gap, _w, _h in standing[lane]):
                            continue
                        if any(w and rear - front < HOLD_GAP_M + 3.0 for rear, _g, w, _h in standing[lane]) \
                                and oid not in self._passing:
                            self._pass_by(oid, lane)
                    else:
                        # (9 Nov, 3rd) on another lane of its road, coming to park in the same place: it changed
                        # onto the lane at the last moment and SUMO stopped it where the first still waited (seed
                        # 103, the Square)
                        # (its front up to the other's front: beside it, it would still change onto its lane and
                        # stop where it stands)
                        near = [a for fr, a in by_edge.get(lane.rsplit("_", 1)[0], ())
                                if 0.0 <= fr - front <= HOLD_BEHIND_M]
                        if not near:
                            continue
                        nxt = c.vehicle.getStops(oid, 1)
                        if not nxt or nxt[0].stoppingPlaceID not in near:
                            continue
                    if c.vehicle.getVehicleClass(oid) in ("emergency", "authority"):
                        continue
                    keep.add(oid)
                    if oid not in self.followers:
                        c.vehicle.setSpeed(oid, 0.0)
                        self._hold_lane(oid)
                        self.followers.add(oid)
                        self.stats["cars_held_behind"] = self.stats.get("cars_held_behind", 0) + 1
                except Exception:  # noqa: BLE001, S112 - it left meanwhile
                    continue
        if standing:
            # (9 Nov, 4th) a car coming fast on the lane, further back than HOLD_BEHIND_M or than the cars looked at
            # (NEAR_M): held while it can still stop (it came at 17 m/s and could not, into a car waiting at the
            # Square's way out)
            for oid, v in (getattr(self, "_cur", None) or {}).items():
                if oid in keep or oid in self.drives or oid in self.d.parked or oid in self.held or abs(v.x) > 1e5 \
                        or v.speed < 8.0:
                    continue
                try:
                    lane = c.vehicle.getLaneID(oid)
                    if lane not in standing:
                        continue
                    front = c.vehicle.getLanePosition(oid)
                    reach = v.speed * v.speed / 5.0 + 10.0
                    if not any(gap <= rear - front <= reach for rear, gap, _w, _h in standing[lane]):
                        continue
                    if c.vehicle.getVehicleClass(oid) in ("emergency", "authority"):
                        continue
                    keep.add(oid)
                    if oid not in self.followers:
                        c.vehicle.setSpeed(oid, 0.0)
                        self._hold_lane(oid)
                        self.followers.add(oid)
                        self.stats["cars_held_behind"] = self.stats.get("cars_held_behind", 0) + 1
                except Exception:  # noqa: BLE001, S112 - it left meanwhile
                    continue
        lc_near: set[str] = set()
        if not hasattr(self, "_lot_stops"):
            self._lot_stops = []
            for aid_, a_ in self.areas.items():
                if "spaces" in a_ and self.visible(aid_):
                    try:
                        p_ = self.c.simulation.convert2D(a_["edge"], float(a_["end"]), 0)
                    except Exception:  # noqa: BLE001, S112
                        continue
                    self._lot_stops.append((aid_, p_))
        # (9 Nov, 4th) a car coming to park: held by its distance (along its route) to a car standing where SUMO
        # will stop it, further back too (on the junction before the road, or too fast to stop once on it); and every
        # car coming to a car park's stop keeps to its lane's middle (SUMO's sublanes let one queue up beside the car
        # in front in the Square's 3.5 m lane, and both stopped at the same point to park)
        stand = [(e, fr, a) for e, lst in by_edge.items() for fr, a in lst]
        for oid, v in (getattr(self, "_cur", None) or {}).items():
            if oid in keep or oid in self.drives or oid in self.d.parked or oid in self.held or abs(v.x) > 1e5:
                continue
            lots = [aid_ for aid_, p_ in self._lot_stops if abs(v.x - p_[0]) < 70.0 and abs(v.y - p_[1]) < 70.0]
            if not lots:
                continue
            try:
                nxt = c.vehicle.getStops(oid, 1)
                aim = nxt[0].stoppingPlaceID if nxt else None
                if aim not in lots:
                    continue
                self._hold_lane(oid, LC_TO_STOP)
                lc_near.add(oid)
                for e, fr, a in stand:
                    if a != aim:
                        continue
                    dist = c.vehicle.getDrivingDistance(oid, e, max(fr, 0.0))
                    if 0.3 <= dist <= 70.0:
                        keep.add(oid)
                        if oid not in self.followers:
                            c.vehicle.setSpeed(oid, 0.0)
                            self.followers.add(oid)
                            self.stats["cars_held_behind"] = self.stats.get("cars_held_behind", 0) + 1
                        break
            except Exception:  # noqa: BLE001, S112 - it left meanwhile
                continue
        for oid in list(self._lc_near - lc_near - self.followers):
            self._lc_near.discard(oid)
            if oid in self._lc and oid not in self.held:
                try:
                    c.vehicle.setLaneChangeMode(oid, self._lc.pop(oid))
                except Exception:  # noqa: BLE001
                    self._lc.pop(oid, None)
        self._lc_near |= lc_near
        for oid in list(self.followers - keep):
            self.followers.discard(oid)
            if oid in self.held:
                continue            # (a car this module holds for its own drive: not let go here)
            try:
                c.vehicle.setSpeed(oid, -1.0)
                if oid in self._lc:
                    c.vehicle.setLaneChangeMode(oid, self._lc.pop(oid))
            except Exception:  # noqa: BLE001, S110 - it left meanwhile
                pass

    def _hold_lane(self, oid, mode=0):
        """(9 Nov, 4th) A car held behind or beside a drive keeps its lane while held (one held on the next lane
        changed onto the drive's lane as it braked, into the car waiting there): lane change mode 0. mode
        LC_TO_STOP: a car coming to a car park's stop changes lanes to reach it but does not squeeze past the car in
        front within its lane (SUMO's sublanes) nor overtake."""
        if oid in self._passing:
            return
        try:
            if oid not in self._lc:
                self._lc[oid] = self.c.vehicle.getLaneChangeMode(oid)
            elif mode != 0:
                return                  # (held already: stays 0)
            self.c.vehicle.setLaneChangeMode(oid, mode)
        except Exception:  # noqa: BLE001, S110 - it left meanwhile
            pass

    def _pass_by(self, oid, lane):
        """A car SUMO drives, held close behind a car waiting at the road: on to the next lane of its road (left,
        else right), to pass it."""
        c = self.c
        try:
            edge, idx = lane.rsplit("_", 1)
            n = c.edge.getLaneNumber(edge)
            to = int(idx) + 1 if int(idx) + 1 < n else int(idx) - 1
            if to < 0:
                return
            if oid in self._lc:
                c.vehicle.setLaneChangeMode(oid, self._lc.pop(oid))
            c.vehicle.changeLane(oid, to, 6.0)
            self._passing.add(oid)
            self.stats["passed_by"] = self.stats.get("passed_by", 0) + 1
        except Exception:  # noqa: BLE001, S110 - it left meanwhile
            pass

    def _sumo_waits(self, vid, dv, now):
        """A drive out held up giving way: SUMO keeps the car in its space until the drive is at the road (its stop
        made longer, or a car taken out for a trip resumed later), so SUMO's car does not wait on the road for it."""
        d = self.d
        if vid not in d.parked:
            return
        end = now + dv.duration - dv.progress(now) + EXIT_GUARD_S
        if vid in d.resume_at:
            d.resume_at[vid] = max(d.resume_at[vid], end)
            return
        if self.release.get(vid, float("inf")) >= end:
            return
        try:
            st = self.c.vehicle.getStops(vid, 1)
            s0 = st[0] if st else None
            if s0 is None or s0.duration >= 1e6 or s0.arrival < 0 or now + s0.duration >= end:
                return
            self.c.vehicle.setParkingAreaStop(vid, dv.area, duration=end - now + 1.0)     # (what is left of it)
            self.release[vid] = end + 1.0
            self.stats["stays_made_longer"] = self.stats.get("stays_made_longer", 0) + 1
        except Exception:  # noqa: BLE001, S110 - as SUMO has it
            pass

    def _room_for(self, vid, aid, now, in_s):
        """A stay shorter than the drive in and the drive out (a fast clock, a short errand): SUMO's stop is made
        that much longer, so SUMO does not put the car back on the road while it is still shown driving in."""
        try:
            st = self.c.vehicle.getStops(vid, 1)
            s0 = st[0] if st else None
            if s0 is None or s0.duration >= 1e6 or s0.arrival < 0:
                return
            left = s0.duration                      # (what is left of the stop: SUMO counts it down)
            need = in_s + 45.0
            if left < need:
                self.c.vehicle.setParkingAreaStop(vid, aid, duration=s0.duration + need - left)
                self.stats["stays_made_longer"] = self.stats.get("stays_made_longer", 0) + 1
        except Exception:  # noqa: BLE001, S110 - as SUMO has it
            pass

    @staticmethod
    def _mark(v):
        v.fixed = True
        v.lane_angle = None
        v.speed = 0.0

    def _lock(self, vid):
        """SUMO's car held where it is for a drive: no lane changes either (9 Nov, 3rd: a held car at a short kerb
        moved two lanes over towards its turn while the drive went to where it had been put: 5.3 m off)."""
        if vid in self.held:
            return
        self.held.add(vid)
        try:
            self._lc[vid] = self.c.vehicle.getLaneChangeMode(vid)
            self.c.vehicle.setLaneChangeMode(vid, 0)
        except Exception:  # noqa: BLE001, S110 - it left meanwhile
            pass

    def _release_hold(self, vid):
        if vid in self.held:
            self.held.discard(vid)
            try:
                self.c.vehicle.setSpeed(vid, -1.0)
                if vid in self._lc:
                    self.c.vehicle.setLaneChangeMode(vid, self._lc.pop(vid))
            except Exception:  # noqa: BLE001 - it left meanwhile
                self._lc.pop(vid, None)


def _retrace(dv, s):
    """Legs back along a drive's way from s seconds into it to its start, each the other way round (a leg it drove
    forwards it backs along, and the other way)."""
    out = []
    for lg, pause in zip(dv.legs, dv.pauses, strict=False):
        if s <= 0.0:
            break
        upto = min(s, lg.duration)
        k = next((i for i, t in enumerate(lg.t) if t >= upto), len(lg.t) - 1)
        x, y = lg.path_at(upto)
        pts = [*lg.pts[:k], (x, y)]
        if len(pts) > 1 and lg.cum[k] > 0.3:
            back = list(reversed(pts))
            out.append(Leg(resample(_dedupe(back)) if len(_dedupe(back)) > 1 else back,
                           V_BACK if not lg.reverse else 2.5, reverse=not lg.reverse, pivot=lg.pivot))
        s -= lg.duration + pause
    out.reverse()
    return out


def _dedupe(pts, m=0.3):
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(out[-1], p) > m:
            out.append(p)
    return out


def circles(x, y, face, length, width=WIDTH_M):
    """A car's body (its middle at x, y, facing `face`) as three circles of its width: the middle, and either end
    less half a width."""
    h = _dir(face)
    off = max(length / 2 - width / 2, 0.0)
    return [(x + h[0] * k * off, y + h[1] * k * off) for k in (-1, 0, 1)]


def _gap(c1, c2, width=WIDTH_M):
    """The room between two bodies (circles), negative where they overlap."""
    return min(math.dist(p, q) for p in c1 for q in c2) - width


def keep_right(corners, which, off=KEEP_RIGHT_M):
    """The corners `which` (indices) moved `off` to the right of the way (where it runs along them): two cars meeting
    in an aisle pass each other."""
    out = list(corners)
    for i in which:
        if i <= 0 or i >= len(corners) - 1:
            continue
        a, c, b = corners[i - 1], corners[i], corners[i + 1]
        ua, ub = _unit(c[0] - a[0], c[1] - a[1]), _unit(b[0] - c[0], b[1] - c[1])
        na, nb = (ua[1], -ua[0]), (ub[1], -ub[0])            # right of the way (x east, y north)
        m = _unit(na[0] + nb[0], na[1] + nb[1])
        k = off / max(m[0] * na[0] + m[1] * na[1], 0.5)
        out[i] = (c[0] + m[0] * k, c[1] + m[1] * k)
    return out


def load_entrances(places_path):
    """(entrances, gates, removed). The ways into the car parks inside buildings, next to the places (scenarios/<town>/
    entrances.json): area ->
    {"approach": [from the road to ...], "front": a point before the opening, "way": [the opening's line, ..., where
    the car is out of sight], "door": a CARLA door to open or null, "depart_pos": where on its lane SUMO puts a car
    leaving it (the parkingArea's departPos, past the door)}. {} when there is none (those cars are not driven:
    they appear and vanish at the road beside the building, as SUMO has them). Gates: the doors and barrier arms
    a drive may pass (CARLA environment objects): name -> (x, y, half width x, half width y), SUMO metres. Removed:
    CARLA objects the viewer takes away for good (two parked jeeps in front of Parking Central's way in)."""
    import json
    from pathlib import Path

    if places_path is None:
        from cosim import purposes
        places_path = purposes.PLACES
    p = Path(places_path).with_name("entrances.json")
    if not p.is_file():
        return {}, {}, []
    data = json.loads(p.read_text(encoding="utf-8"))
    gates = {name: tuple(g["box"]) for name, g in data.get("gates", {}).items()}
    return ({aid: e for aid, e in data.get("entrances", {}).items() if e.get("way")}, gates,
            list(data.get("removed", [])))

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
* a car parking at the kerb glides from its lane to the kerb, and back.

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

The viewer's vehicles carry their front bumper (cosim/mapping.py); the paths here are of the car's middle.
"""
from __future__ import annotations

import itertools
import math

AISLE_BACK_M = 5.6      # a space's middle to its aisle's middle: half a space (2.6 m) and half an aisle (3 m)
ROAD_ON_M = 9.0         # a car turning in goes on this far along the road first (turns of 4 m: a car's)
STRIP_GAP_M = 3.0       # the strip inside the car park: this far before its first row of spaces
TURN_R_M = 4.5          # turns (fillets) of at most this radius
KERB_PAST_M = 4.0       # at a kerb: the car pulls past its place this far, and backs into it (and out of it)
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
HOLD_GAP_M = 3.0        # ... but not one closer than this (SUMO needs that room to put a parked car back there)
MERGE_M, MERGE_S = 10.0, 4.0     # a drive out pulls onto the road when no car is this near behind (or this many
                                 # seconds at its speed)
OVERDUE_S = 3.0         # a drive out at the road this long after SUMO should have put it back: no longer held for
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


def hermite(p0, h0, p1, h1, n=None):
    """A smooth curve from p0 heading h0 (unit) to p1 heading h1 (a car gliding to the kerb)."""
    d = math.dist(p0, p1)
    n = n or max(6, int(d / STEP_M))
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

    def __init__(self, pts, v_max, reverse=False, signals=0, blink_until_m=None):
        self.pts, self.reverse, self.signals = pts, reverse, signals
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

        P = (bay_c[0] - h[0] * AISLE_BACK_M, bay_c[1] - h[1] * AISLE_BACK_M)
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
            way = keep_right([road_c, *corners, bay_c], (2, 3) if across else (2,), off=kr)
            P, before = corners[-1], corners[-2]
            a = _unit(P[0] - before[0], P[1] - before[1])            # down the aisle: on its right half
            way[-2] = (P[0] + a[1] * kr, P[1] - a[0] * kr)
            pts = fillet(way)
            side = RIGHT if (dr[0] * n[1] - dr[1] * n[0]) < 0 else LEFT
            self.stats["in"] += 1
            return self._on_road(Drive(vid, "in", [Leg(pts, V_DRIVE, signals=side, blink_until_m=ROAD_ON_M + 8.0)],
                                       now, L), aid, road_c, dr)
        # at the kerb: past its place, beside the kerb, then backing into it (as one parks along a pavement)
        kerb_c = (v.x - h[0] * L / 2, v.y - h[1] * L / 2)
        if math.dist(kerb_c, road_c) < 0.3:
            return None
        nx, ny = kerb_c[0] - road_c[0], kerb_c[1] - road_c[1]
        side = RIGHT if (dr[0] * ny - dr[1] * nx) < 0 else LEFT
        past = (kerb_c[0] + h[0] * KERB_PAST_M, kerb_c[1] + h[1] * KERB_PAST_M)
        self.stats["kerb_in"] += 1
        return self._on_road(Drive(vid, "in", [Leg(hermite(road_c, dr, past, h), 2.5, signals=side),
                                               Leg(resample([past, kerb_c]), V_BACK, reverse=True)], now, L,
                                         pauses=[PAUSE_S, 0.0]), aid, road_c, dr)

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
            P = corners[-1]
            # back out into the aisle, turning so it faces the way out
            if across:
                u_out = (-n[0], -n[1])                    # along the aisle, towards the road
            else:
                e0 = (road_c[0] - dr[0] * ROAD_ON_M, road_c[1] - dr[1] * ROAD_ON_M)
                k = (e0[0] - P[0]) * dr[0] + (e0[1] - P[1]) * dr[1]
                u_out = dr if k >= 0 else (-dr[0], -dr[1])
            # (on the right half of the aisle, as it will drive off: a car coming the other way passes)
            kr = self.keep_right_m(aid, bay_c, h)
            P = (P[0] + u_out[1] * kr, P[1] - u_out[0] * kr)
            back_end = (P[0] - u_out[0] * 3.5, P[1] - u_out[1] * 3.5)
            back = Leg(fillet([bay_c, P, back_end], r_max=3.5), V_BACK, reverse=True)
            e0 = (road_c[0] - dr[0] * ROAD_ON_M, road_c[1] - dr[1] * ROAD_ON_M)
            if across:
                g = corners[1]
                strip = (g[0] - road_c[0]) * n[0] + (g[1] - road_c[1]) * n[1]
                s_ = corners[2]
                g2 = (e0[0] + n[0] * strip, e0[1] + n[1] * strip)
                fwd = keep_right([back_end, P, s_, g2, e0, road_c], (2, 3), off=kr)
            else:
                k = (e0[0] - P[0]) * n[0] + (e0[1] - P[1]) * n[1]
                t2 = (P[0] + dr[0] * ((e0[0] - P[0]) * dr[0] + (e0[1] - P[1]) * dr[1]),
                      P[1] + dr[1] * ((e0[0] - P[0]) * dr[0] + (e0[1] - P[1]) * dr[1]))
                fwd = keep_right([back_end, t2, e0, road_c], (1,), off=kr) if math.dist(back_end, t2) > 1.0 \
                    else [back_end, e0, road_c]
            pts = fillet(_dedupe(fwd))
            side = LEFT if (dr[0] * n[1] - dr[1] * n[0]) < 0 else RIGHT   # (turning onto the road: the other way)
            go = Leg(pts, V_DRIVE, signals=side)
            return self._on_road(Drive(vid, "out", [back, go], t0, L, pauses=[PAUSE_S, 0.0]), aid, road_c, dr)
        # from the kerb: a little back, then out beside it onto the road, where SUMO puts it
        kerb_c = (v.x - h[0] * L / 2, v.y - h[1] * L / 2)
        nx, ny = road_c[0] - kerb_c[0], road_c[1] - kerb_c[1]
        side = RIGHT if (dr[0] * ny - dr[1] * nx) < 0 else LEFT
        ahead = nx * dr[0] + ny * dr[1]                      # (how far ahead of its place SUMO puts it)
        back_m = max(KERB_PAST_M - ahead, 0.0)
        start = (kerb_c[0] - h[0] * back_m, kerb_c[1] - h[1] * back_m)
        legs, pauses = [Leg(hermite(start, h, road_c, dr), 2.5, signals=side)], [0.0]
        if back_m > 0.3:
            legs.insert(0, Leg(resample([kerb_c, start]), V_BACK, reverse=True))
            pauses.insert(0, PAUSE_S)
        return self._on_road(Drive(vid, "out", legs, t0, L, pauses=pauses), aid, road_c, dr)

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

    def on_lane(self, dv, x, y):
        """Whether a body with its middle at x, y stands on the drive's lane (within ON_LANE_M of its line), and how
        far along the lane (None: off it)."""
        road = getattr(dv, "road", None)
        if road is None or road[3] is None:
            return None
        (rx, ry), (ux, uy), _lane, pos = road
        along = (x - rx) * ux + (y - ry) * uy
        side = abs((x - rx) * uy - (y - ry) * ux)
        return pos + along if side < ON_LANE_M else None

    def exit_now(self, vid, aid):
        """PurposeDemand: car vid is taken out of its space (area aid) for a trip now: the seconds before SUMO may
        put it on the road (the drive out), or 0 (no drive: SUMO at once)."""
        if not self.visible(aid) or vid in self.drives:
            return 0.0
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

    def _count_out(self, aid):
        self.stats["out" if "spaces" in self.areas[aid] else "kerb_out"] += 1

    # ------------------------------------------------------------------ every step
    def step(self, now, vehicles):
        """Moves the cars on their drives: their front bumper, heading, speed and lights in `vehicles` (this step's,
        about to be shown). Also marks every car standing in a space so the viewer keeps its heading as SUMO has it."""
        c, d = self.c, self.d
        self._cur, self._now = vehicles, now
        self._k += 1
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
            try:
                dv = self._out(vid, aid, now, v)
            except Exception:  # noqa: BLE001
                dv = None
                self.stats["failed"] += 1
            if dv is None:
                self.release[vid] = float("inf")
                continue
            if now >= end - dv.duration - EXIT_GUARD_S:
                dv.area, dv.why = aid, "time up" if now <= end - dv.duration else f"time up, {now - end + dv.duration:.0f} s late"
                self.reserve(dv, now)
                self.drives[vid] = dv
                self._count_out(aid)
                if dv.lag > 0:
                    self._sumo_waits(vid, dv, now)
        self.give_way(now, vehicles)
        self._doors_step(now)
        # move them
        for vid, dv in list(self.drives.items()):
            v = vehicles.get(vid)
            if v is None:
                self.drives.pop(vid)
                self._release_hold(vid)
                continue
            if dv.kind == "in" and vid not in d.parked and vid not in ending and abs(v.x) < 1e5:
                # SUMO took it out again before its drive in was over (a short stop): back from where it is shown
                # to SUMO's car, which waits for it
                # (9 Nov) the way it came, the other way, back to where it left the road (that way is clear: a
                # straight line to SUMO's car went through the cars parked at a kerb, seen in CARLA), then to SUMO's car
                h1 = _dir(v.angle)
                sumo_c = (v.x - h1[0] * v.length / 2, v.y - h1[1] * v.length / 2)
                legs = _retrace(dv, min(max(dv.progress(now), 0.0), dv.duration))
                x, y = legs[-1].pts[-1] if legs else dv.at(now)[:2]
                h0 = _dir(legs[-1].at(legs[-1].duration)[2]) if legs else _dir(dv.at(now)[2])
                if legs and legs[-1].reverse:
                    h0 = (-h0[0], -h0[1])                   # (its way there, not its facing)
                if math.dist((x, y), sumo_c) > 0.5:
                    ahead = (sumo_c[0] - x) * h0[0] + (sumo_c[1] - y) * h0[1] >= 0.0
                    if ahead:                               # (forwards, or backing to it: facing as it is)
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
                    self.held.add(vid)
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
            if dv.kind == "out" and vid in ending and not dv.done(now):
                try:                                   # SUMO put it back first: its car waits for the drive
                    c.vehicle.setSpeed(vid, 0.0)
                    self.held.add(vid)
                    self.stats["held_sumo"] += 1
                except Exception:  # noqa: BLE001, S110 - it left meanwhile
                    pass
            if dv.done(now):
                if dv.kind == "out" and vid in ending:
                    pass            # (SUMO reports the car in its space at the step it puts it back: still shown)
                elif dv.kind == "out" and abs(v.x) > 1e5:
                    pass            # (SUMO has no place for it at this step yet: still shown where the drive ended)
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
                    self.held.add(vid)
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
        self.prev = {vid: (v.x, v.y, v.angle) for vid, v in vehicles.items() if vid not in d.parked}

    # ------------------------------------------------------------------ giving way
    def reserve(self, dv, now):
        """(9 Nov) A new drive waits until its way is clear of the drives already on theirs: it starts the first
        half second (up to RESERVE_MAX_S) for which its body, all along its drive, keeps GAP_M from every other
        drive's body as that drive will be at the same moment (and from where one ends, for END_HOLD_S after: SUMO
        takes it over there). A car going out waits in its space; one coming in waits on the road where SUMO
        stopped it (the cars behind it wait too: _hold_followers). Its wait goes into its lag."""
        if "spaces" not in self.areas.get(dv.area or "", {}):
            return 0.0          # (at a kerb: as before; the aisles are what cars share)
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
            if s >= dv.duration or s < 0.0:        # (done, or not started: waiting for its way to be clear)
                continue
            if "spaces" not in self.areas.get(dv.area, {}):
                continue        # (at a kerb: as before, see reserve)
            hit = None
            last = cur[vid][0][1]
            for k in range(1, LOOK_N + 1):
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
            patient = vid in road
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

    def _merge_gap(self, dv, k):
        """A drive out about to pull onto its lane: (k, car, False) for a car SUMO drives on the lane that would
        come up on it (behind it, within MERGE_S at its speed, or MERGE_M), else None."""
        c = self.c
        lane, pos = dv.road[2], dv.road[3]
        try:
            for oid in c.lane.getLastStepVehicleIDs(lane):
                if oid == dv.vid or oid in self.d.parked or oid in self.followers or oid in self.drives:
                    continue        # (a car on a drive is not where SUMO has it)
                front = c.vehicle.getLanePosition(oid)
                behind = pos - ROAD_ON_M - front           # (the drive joins about ROAD_ON_M before its end)
                if -dv.length <= behind <= max(MERGE_M, c.vehicle.getSpeed(oid) * MERGE_S):
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
        for vid, dv in self.drives.items():
            if "spaces" not in self.areas.get(dv.area, {}):
                continue        # (a kerb's lane is a slow parking lane: its cars are not held, they pass by)
            due = self.release.get(vid, self.d.resume_at.get(vid, getattr(dv, "done_at", float("inf"))))
            if dv.done(now) and now > due + OVERDUE_S:
                continue        # (SUMO cannot put it back: the cars behind are let go, SUMO sorts it out)
            x, y, _f, _v, _g = dv.at_s(min(max(dv.progress(now), 0.0), dv.duration))
            pos = self.on_lane(dv, x, y)
            if pos is not None:     # (its rear; and how close a car may be to be held: SUMO puts a car back in front)
                standing.setdefault(dv.road[2], []).append((pos - dv.length / 2, HOLD_GAP_M if dv.kind == "out"
                                                            and dv.done(now) else -dv.length))
        keep = set()
        if standing:
            for oid, (_body, is_drive) in cur.items():
                if is_drive or oid in self.held:
                    continue
                try:
                    lane = c.vehicle.getLaneID(oid)
                    if lane not in standing:
                        continue
                    front = c.vehicle.getLanePosition(oid)
                    if not any(gap <= rear - front <= HOLD_BEHIND_M for rear, gap in standing[lane]):
                        continue
                    if c.vehicle.getVehicleClass(oid) in ("emergency", "authority"):
                        continue
                    keep.add(oid)
                    if oid not in self.followers:
                        c.vehicle.setSpeed(oid, 0.0)
                        self.followers.add(oid)
                        self.stats["cars_held_behind"] = self.stats.get("cars_held_behind", 0) + 1
                except Exception:  # noqa: BLE001, S112 - it left meanwhile
                    continue
        for oid in list(self.followers - keep):
            self.followers.discard(oid)
            try:
                c.vehicle.setSpeed(oid, -1.0)
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

    def _release_hold(self, vid):
        if vid in self.held:
            self.held.discard(vid)
            try:
                self.c.vehicle.setSpeed(vid, -1.0)
            except Exception:  # noqa: BLE001, S110 - it left meanwhile
                pass


def _retrace(dv, s):
    """Legs back along a drive's way from s seconds into it to its start, each the other way round (a leg it drove
    forwards it backs along, and the other way)."""
    out = []
    for lg, pause in zip(dv.legs, dv.pauses, strict=False):
        if s <= 0.0:
            break
        upto = min(s, lg.duration)
        k = next((i for i, t in enumerate(lg.t) if t >= upto), len(lg.t) - 1)
        x, y, _f, _v, _g = lg.at(upto)
        pts = [*lg.pts[:k], (x, y)]
        if len(pts) > 1 and lg.cum[k] > 0.3:
            back = list(reversed(pts))
            out.append(Leg(resample(_dedupe(back)) if len(_dedupe(back)) > 1 else back,
                           V_BACK if not lg.reverse else 2.5, reverse=not lg.reverse))
        s -= lg.duration + pause
    out.reverse()
    return out


def _dedupe(pts):
    out = [pts[0]]
    for p in pts[1:]:
        if math.dist(out[-1], p) > 0.3:
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

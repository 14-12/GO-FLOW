"""Smooth motion between SUMO steps.

SUMO advances in steps of 1 s (the RESCO scenarios' step length, kept so the traffic is the
benchmark's own); a video needs 20-30 frames per second. Between two steps each vehicle
follows a cubic Hermite curve whose end tangents point along SUMO's heading at each step and
scale with its speed, and the body's heading is taken from the curve's tangent. This is the
same scheme as the main project's browser twin (frontend playback.ts, ADR 0008), so a car
turning a corner turns continuously instead of in 1-s kinks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """One vehicle at one SUMO step: front-bumper position (m), heading (deg, SUMO), speed (m/s)."""

    x: float
    y: float
    angle: float
    speed: float


def _dir(angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return math.sin(a), math.cos(a)  # SUMO: 0 deg = north (+y), clockwise


def hermite(a: State, b: State, s: float, dt: float) -> tuple[float, float, float]:
    """Position and heading (SUMO convention) at fraction s in [0, 1] between steps a and b."""
    if s <= 0.0:
        return a.x, a.y, a.angle
    if s >= 1.0:
        return b.x, b.y, b.angle
    chord = math.hypot(b.x - a.x, b.y - a.y)
    if chord < 1e-3:  # standing still: no curve, keep the heading
        return a.x, a.y, a.angle
    # Tangent magnitudes: the distance actually covered, split by the two speeds, so a car
    # braking to a stop does not overshoot. Never longer than 1.5x the chord.
    ua, ub = _dir(a.angle), _dir(b.angle)
    ma = min(max(a.speed * dt, 0.0), 1.5 * chord) or chord
    mb = min(max(b.speed * dt, 0.0), 1.5 * chord) or chord
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    x = h00 * a.x + h10 * ma * ua[0] + h01 * b.x + h11 * mb * ub[0]
    y = h00 * a.y + h10 * ma * ua[1] + h01 * b.y + h11 * mb * ub[1]
    # Derivative for the heading.
    d00, d10, d01, d11 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1, -6 * s2 + 6 * s, 3 * s2 - 2 * s
    dx = d00 * a.x + d10 * ma * ua[0] + d01 * b.x + d11 * mb * ub[0]
    dy = d00 * a.y + d10 * ma * ua[1] + d01 * b.y + d11 * mb * ub[1]
    heading = math.degrees(math.atan2(dx, dy)) % 360.0
    return x, y, heading


def catmull_rom(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float],
                p3: tuple[float, float], t: float) -> tuple[float, float]:
    """Centripetal Catmull-Rom point at t in [0, 1] between p1 and p2 (p0, p3: the points before and
    after). Goes through the SUMO positions themselves and ignores SUMO's heading, which on internal
    lanes jumps from one segment of the lane's polyline to the next."""
    def knot(ti, a, b):
        return ti + max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-4) ** 0.5

    if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 1e-3:
        return p1
    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    u = t1 + (t2 - t1) * min(max(t, 0.0), 1.0)

    def lerp(a, b, ta, tb):
        w = (u - ta) / (tb - ta)
        return a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w

    a1, a2, a3 = lerp(p0, p1, t0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3)
    b1, b2 = lerp(a1, a2, t0, t2), lerp(a2, a3, t1, t3)
    return lerp(b1, b2, t1, t2)


class Trails:
    """Heading of each vehicle from the path its front bumper has taken: the body points from the point
    REAR_SHARE x length back along that path to the front, as a car's body follows its front wheels
    through a turn. No SUMO heading is used, so nothing can disagree with the motion.

    Measured offline on 300 s of Town05 evening traffic, 60 poses a second (~118 vehicles): turning
    direction reversed 20 times against 211 with the even turn from SUMO's step headings, yaw
    acceleration p99.9 243 against 892 deg/s2 (tools/turn_smoothness.py)."""

    REAR_SHARE = 0.7
    MAX_STEP_DEG = 2.5
    MAX_POINTS = 300
    TURN_DEG_PER_M = 15.0
    # a body across its lane at most this much: 25 deg when moving (a tight turn's body-to-lane angle is ~16 deg),
    # 12 deg under SLOW_MS (standing slanted in a queue, a body poked into the next lane: tools/check_overlaps.py)
    MAX_SLANT_DEG, MAX_SLANT_SLOW_DEG, SLOW_MS = 25.0, 12.0, 3.0

    def __init__(self):
        self.paths: dict[str, list[tuple[float, float]]] = {}
        self.heading: dict[str, float] = {}
        self.last: dict[str, tuple[float, float]] = {}

    def update(self, vid: str, x: float, y: float, length: float, sumo_angle: float,
               lane_angle: float | None = None, speed: float | None = None) -> float:
        """Front at (x, y) (SUMO coordinates) -> heading (SUMO convention, degrees). lane_angle: the direction of
        the vehicle's lane where it is (cosim/lanegeo.py); the body stays within MAX_SLANT_DEG of it (MAX_SLANT_SLOW_DEG when slower than SLOW_MS, `speed`)."""
        path = self.paths.get(vid)
        if path is None:  # first seen: a straight path behind it along SUMO's heading
            r = math.radians(sumo_angle)
            path = [(x - math.sin(r) * 2 * length, y - math.cos(r) * 2 * length), (x, y)]
            self.paths[vid] = path
            self.heading[vid] = sumo_angle % 360.0
            return self.heading[vid]
        if math.hypot(x - path[-1][0], y - path[-1][1]) > 0.02:
            path.append((x, y))
            if len(path) > self.MAX_POINTS:
                del path[: len(path) - self.MAX_POINTS // 2]
        back = self.REAR_SHARE * length
        px, py = x, y
        rx = ry = None
        for qx, qy in reversed(path[:-1]):
            d = math.hypot(px - qx, py - qy)
            if d >= back:
                f = back / d
                rx, ry = px + (qx - px) * f, py + (qy - py) * f
                break
            back -= d
            px, py = qx, qy
        if rx is None:
            rx, ry = path[0]
        # how far the front went forward (along the body) since the last pose: a car turns only as it rolls
        lx, ly = self.last.get(vid, (x, y))
        self.last[vid] = (x, y)
        h = math.radians(self.heading[vid])
        fwd = abs((x - lx) * math.sin(h) + (y - ly) * math.cos(h))
        if math.hypot(x - rx, y - ry) > 0.05 and fwd > 1e-4:
            want = math.degrees(math.atan2(x - rx, y - ry))
            d = (want - self.heading[vid] + 180.0) % 360.0 - 180.0
            # at most MAX_STEP_DEG a pose: a car creeping in a queue, nudged sideways by SUMO, swung round
            # at 400 deg/s (2 cases in 120 s); a real turn stays under 140 deg/s at 60 poses a second.
            # And at most TURN_DEG_PER_M per metre rolled forward (a car's tightest turn, ~4 m radius): a body
            # standing still never turns. SUMO's lane change moved stopped cars 3.5 m sideways at the stop line and
            # the body followed that sideways path, standing diagonally across two lanes (tools/check_overlaps.py).
            cap = min(self.MAX_STEP_DEG, self.TURN_DEG_PER_M * fwd)
            d = max(-cap, min(cap, d))
            self.heading[vid] = (self.heading[vid] + d) % 360.0
        if lane_angle is not None:
            band = self.MAX_SLANT_DEG if speed is None or speed >= self.SLOW_MS else self.MAX_SLANT_SLOW_DEG
            off = (self.heading[vid] - lane_angle + 180.0) % 360.0 - 180.0
            if abs(off) > band:  # turned back to the edge of the band, at most MAX_STEP_DEG a pose
                want = lane_angle + math.copysign(band, off)
                d = (want - self.heading[vid] + 180.0) % 360.0 - 180.0
                self.heading[vid] = (self.heading[vid] + max(-self.MAX_STEP_DEG, min(self.MAX_STEP_DEG, d))) % 360.0
        return self.heading[vid]

    def forget(self, keep: set) -> None:
        for vid in [v for v in self.paths if v not in keep]:
            self.paths.pop(vid, None)
            self.heading.pop(vid, None)
            self.last.pop(vid, None)


def ease_profile(m0: float, m1: float, s: float) -> float:
    """Share of a step's path covered at fraction s of the step: a cubic from 0 to 1 whose slopes at the
    ends are m0 and m1 (speed at each end over the step's mean speed). Slopes are scaled down where they
    would make the cubic run backwards (monotone: m0^2 + m1^2 <= 9)."""
    m0, m1 = max(m0, 0.0), max(m1, 0.0)
    r = m0 * m0 + m1 * m1
    if r > 9.0:
        k = 3.0 / math.sqrt(r)
        m0, m1 = m0 * k, m1 * k
    s = min(max(s, 0.0), 1.0)
    s2, s3 = s * s, s * s * s
    return (-2 * s3 + 3 * s2) + (s3 - 2 * s2 + s) * m0 + (s3 - s2) * m1


class Timed:
    """A Catmull-Rom step (p1 to p2, with p0 before and p3 after) driven at SUMO's speeds.

    SUMO moves a vehicle at one speed for a whole step (0.5 s), so its speed changes in jumps at every
    step. Driving the curve with its own parameter passed those jumps on, and more: measured with
    tools/turn_smoothness.py (120 s of Town05 evening traffic, 60 poses a second), 4842 speed jumps over
    0.5 m/s between two poses and acceleration p99.9 65 m/s2: cars moving forward looked as if they
    lagged. Here the speed at each end of the step is the mean of the steps on either side (so it is
    the same on both sides of a step boundary), the path covered follows ease_profile, and the curve is
    walked by arc length (a table of N points). The vehicle is still exactly at SUMO's position at every
    step."""

    N = 8

    def __init__(self, p0, p1, p2, p3):
        self.p = (p0, p1, p2, p3)
        chord = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        self.still = chord < 1e-3
        if self.still:
            return
        pts = [catmull_rom(p0, p1, p2, p3, i / self.N) for i in range(self.N + 1)]
        cum = [0.0]
        for a, b in zip(pts, pts[1:], strict=False):
            cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = cum[-1] or 1.0
        self.cum = [c / total for c in cum]
        before = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        after = math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        self.m0 = (before + chord) / 2.0 / chord
        self.m1 = (chord + after) / 2.0 / chord

    def at(self, s: float) -> tuple[float, float]:
        if self.still:
            return self.p[1]
        f = ease_profile(self.m0, self.m1, s)
        cum = self.cum
        i = 1
        while i < self.N and cum[i] < f:
            i += 1
        lo, hi = cum[i - 1], cum[i]
        t = (i - 1 + ((f - lo) / (hi - lo) if hi > lo else 0.0)) / self.N
        return catmull_rom(*self.p, t)

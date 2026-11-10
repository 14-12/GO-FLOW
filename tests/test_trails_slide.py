"""(10 Nov) interp.Trails: the body's rear rolls along the body, it does not slide sideways (tools/check_lateral.py)."""

import math

from cosim.interp import Trails


def _rear_slide(points, length=4.6, rear=0.7):
    """Largest angle (deg) between the rear point's motion and the body, the front driven through `points`."""
    tr = Trails()
    worst, prev = 0.0, None
    for i, (x, y) in enumerate(points):
        ang = tr.update("v", x, y, length, 90.0 if i == 0 else 0.0)
        r = math.radians(ang)
        rx, ry = x - math.sin(r) * rear * length, y - math.cos(r) * rear * length
        if prev is not None and i > 60:
            dx, dy = rx - prev[0], ry - prev[1]
            if math.hypot(dx, dy) > 1e-4:
                side = dx * math.cos(r) - dy * math.sin(r)
                fwd = dx * math.sin(r) + dy * math.cos(r)
                worst = max(worst, math.degrees(math.atan2(abs(side), abs(fwd))))
        prev = (rx, ry)
    return worst


def test_turn_rear_does_not_slide():
    # 2 m along x (SUMO heading 90), then a 10 m radius quarter turn at 5 m/s, 60 poses a second
    pts = [(-20.0 + 5.0 * i / 60, 0.0) for i in range(240)]
    R, n = 10.0, 190
    for i in range(1, n + 1):
        a = (math.pi / 2) * i / n
        pts.append((R * math.sin(a), R - R * math.cos(a)))
    # before 10 Nov the body was a chord of the front's path: its rear slid ~9 deg off it on this turn
    assert _rear_slide(pts) < 3.0


def test_lane_change_rear_does_not_slide():
    # 3.5 m sideways over 25 m at 8 m/s: a lane change
    pts = []
    for i in range(400):
        x = -20.0 + 8.0 * i / 60
        f = min(max((x - 0.0) / 25.0, 0.0), 1.0)
        pts.append((x, 3.5 * (3 * f * f - 2 * f ** 3)))
    assert _rear_slide(pts) < 3.0


def test_standing_body_does_not_turn():
    tr = Trails()
    tr.update("v", 0.0, 0.0, 4.6, 90.0)
    for i in range(1, 60):          # moved sideways 0.6 m on the spot (SUMO's squeeze past a wreck)
        h = tr.update("v", 0.0, 0.01 * i, 4.6, 90.0)
    assert abs(((h - 90.0) + 180.0) % 360.0 - 180.0) < 1e-6

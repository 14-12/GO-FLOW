"""The viewer drives a car into its space and out of it (cosim/parking_motion.py), on SUMO's own run: no jump on the
way in or out (at most a car's step at walking-to-driving speed between two steps), it ends in its space, facing as
SUMO has it, and it is back on the road where and when SUMO has it."""
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import parking_motion as pm  # noqa: E402
from cosim import purposes as P  # noqa: E402


def test_a_fillet_is_smooth_and_ends_where_asked():
    pts = pm.fillet([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    assert pts[0] == (0.0, 0.0) and pts[-1] == (10.0, 10.0)
    assert max(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False)) <= pm.STEP_M + 1e-6
    leg = pm.Leg(pts, 3.6)
    assert leg.at(0)[3] == 0.0 and leg.at(leg.duration)[3] == 0.0       # from standing, to standing
    assert max(leg.v) <= 3.6 + 1e-9
    back = pm.Leg(pts, 1.3, reverse=True)
    x, y, face, _v, sig = back.at(back.duration / 2)
    assert sig & pm.BACK                                                  # reversing lights


@pytest.mark.slow
def test_a_car_drives_into_its_space_and_out_without_a_jump():
    import sumolib
    import traci

    from cosim.sumo_side import Vehicle

    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a", str(P.ADD), "--step-length",
                 "0.5", "--no-step-log", "--no-warnings", "--time-to-teleport", "-1"], label="parking_motion")
    c = traci.getConnection("parking_motion")
    try:
        d = P.PurposeDemand(c, clock_speed=240.0, seed=3)      # shopping about 1 town hour: 15 s here
        d.rng = random.Random(2)
        d.purpose_at = lambda hour, vclass: "shopping"
        d.places.choose = lambda rng, what, hour, exclude=None: "marche"
        motion = pm.ParkingMotion(d, c)
        assert d.launch("car", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ())
        lot = d.places.areas["pa_lot_marche"]
        shown, events = [], []
        for _ in range(2400):
            c.simulationStep()
            t = c.simulation.getTime()
            vs = {}
            for vid in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(vid)
                vs[vid] = Vehicle(vid, "passenger", "t", c.vehicle.getLength(vid), x, y, c.vehicle.getAngle(vid),
                                  c.vehicle.getSpeed(vid), 0)
            d.step([])
            for vid, aid in d.parked.items():          # as cosim/freefly.produce: the middle on the space
                v = vs.get(vid)
                if v is not None and "spaces" in d.places.areas[aid]:
                    a = math.radians(v.angle)
                    v.x += math.sin(a) * v.length / 2
                    v.y += math.cos(a) * v.length / 2
            motion.step(t, vs)
            v = vs.get("car")
            if v is None:
                continue
            shown.append((t, v.x, v.y, v.angle, d.parked.get("car"), "car" in motion.drives))
            if "car" in motion.drives:
                events.append(motion.drives["car"].kind)
            if "in" in events and "out" in events and "car" not in motion.drives and d.parked.get("car") is None \
                    and len(shown) > 10 and shown[-10][4] is None:
                break
        assert "in" in events and "out" in events, str(motion.stats)
        def mid(s):                                          # the car's middle (the front swings in a turn)
            h = math.radians(s[3])
            return s[1] - math.sin(h) * 2.5, s[2] - math.cos(h) * 2.5
        steps = [math.dist(mid(a), mid(b)) for a, b in zip(shown, shown[1:], strict=False)]
        # no jump: never more than a road's speed limit allows in a step (14 m/s: 7 m), and on a drive in or out at
        # most its own pace (3.6 m/s: 1.8 m)
        jumps = [(round(shown[k][0], 1), round(steps[k], 1)) for k in range(len(steps))
                 if steps[k] > 7.5 or ((shown[k][5] or shown[k + 1][5]) and steps[k] > 2.0)]
        assert not jumps, jumps
        parked = [s for s in shown if s[4] == "pa_lot_marche" and not s[5]]
        assert parked                                        # standing in its space between the drives
        x, y, ang = parked[-1][1:4]
        h = math.radians(ang)
        m = (x - math.sin(h) * 2.5, y - math.cos(h) * 2.5)
        assert min(math.dist(m, s[:2]) for s in lot["spaces"]) < 0.05
        assert motion.stats["in"] == 1 and motion.stats["out"] == 1
    finally:
        c.close()


def test_keep_right_moves_the_inner_corners_to_the_right_of_the_way():
    way = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)]          # north, then east: right of north is east
    out = pm.keep_right(way, (1,), off=1.0)
    assert out[0] == way[0] and out[2] == way[2]
    assert out[1][0] > 0.0 and out[1][1] < 10.0               # the corner moved right of both legs


def test_two_bodies_side_by_side_in_neighbouring_spaces_do_not_touch():
    a = pm.circles(0.0, 0.0, 0.0, 4.6)
    b = pm.circles(2.5, 0.0, 0.0, 4.6)                        # 2.5 m apart, both facing north
    assert pm._gap(a, b) > 0.5
    c = pm.circles(0.0, 4.0, 0.0, 4.6)                        # one 4 m behind the other: overlapping
    assert pm._gap(a, c) < 0.0


@pytest.mark.slow
def test_cars_driving_in_and_out_of_the_same_car_park_at_once_keep_clear_of_each_other():
    """(9 Nov) 8 cars sent to the Marché one every 4 s, a short stay: their drives in and out cross in the aisles.
    None comes within 0.2 m of another car's body (tools/check_parking_crowd.py; before: 720 steps overlapping)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import check_parking_crowd as crowd

    res = crowd.run("marche", cars=8, every=4.0, seed=5, port_label="parking_crowd_test")
    assert res["drives_in"] == 8 and res["drives_out"] == 8, res
    assert res["overlap_steps"] == 0, res["worst"]
    assert res["stats"].get("started_later", 0) >= 1         # some waited for their way to be clear

"""tools/route_delay_table.LiveRoadTimes (10 Nov): road times read from each step."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
tc = pytest.importorskip("traci.constants")

from route_delay_table import LiveRoadTimes  # noqa: E402


class _Sim:
    def __init__(self):
        self.parking, self.stops = [], []

    def getParkingStartingVehiclesIDList(self):
        return self.parking

    def getStopStartingVehiclesIDList(self):
        return self.stops


class _Veh:
    def __init__(self):
        self.roads = {}

    def getAllSubscriptionResults(self):
        return {v: {tc.VAR_ROAD_ID: r} for v, r in self.roads.items()}


class _Conn:
    def __init__(self):
        self.simulation, self.vehicle = _Sim(), _Veh()


def _drive(rec, c, plan, vid="live1"):
    for t, road in plan:
        c.vehicle.roads = {vid: road} if road is not None else {}
        rec.step(c, t, [vid] if road is None else [])


def test_times_from_leaving_the_road_before():
    rec, c = LiveRoadTimes(0.0), _Conn()
    # first road A (part way: not filed), junction, B for 10 s, junction, C, then arrives on C (not filed)
    _drive(rec, c, [(0, "A"), (1, "A"), (2, ":j1_0"), (3, "B"), (12, "B"), (13, ":j2_0"), (14, "C"), (20, None)])
    assert dict(rec.samples) == {"B>C": [11.0]}


def test_a_road_where_the_car_parks_is_left_out():
    rec, c = LiveRoadTimes(0.0), _Conn()
    _drive(rec, c, [(0, "A"), (2, ":j_0"), (3, "B")])
    c.simulation.parking = ["live1"]
    _drive(rec, c, [(4, "B")])
    c.simulation.parking = []
    # parked on B for a long time, then pulls out and drives on: B not filed, C is (entered when it left B)
    _drive(rec, c, [(500, "B"), (501, ":k_0"), (502, "C"), (520, ":m_0"), (521, "D")])
    assert dict(rec.samples) == {"C>D": [19.0]}
    assert rec.left_out["halted"] == 1


def test_warm_up_teleports_and_other_vehicles():
    rec, c = LiveRoadTimes(10.0), _Conn()
    _drive(rec, c, [(0, "A"), (2, ":j_0"), (3, "B"), (8, ":j_0"), (9, "C"), (15, ""), (30, "D"), (31, ":x_0"),
                    (32, "E"), (40, ":y_0"), (41, "F")])
    _drive(rec, c, [(0, "A"), (2, ":j_0"), (11, "B"), (12, ":j_0"), (13, "C")], vid="hw3")
    # B entered at 2 (warm-up): out; C left by teleport: out; D first after the teleport: out; E entered at 31: in
    assert dict(rec.samples) == {"E>F": [9.0]}

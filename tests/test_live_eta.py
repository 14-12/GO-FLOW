"""cosim/live_eta.py (10-11 Nov): the navigation app's live view from its users' cars, the time left from where a car
is, and the roads ahead of a ring car's planned trip."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from cosim.live_eta import ProbeTraffic, delay_level, level, roads_ahead, time_left, uses_app


class _Sim:
    def __init__(self):
        self.parking, self.stops = [], []

    def getParkingStartingVehiclesIDList(self):
        return self.parking

    def getStopStartingVehiclesIDList(self):
        return self.stops


def _drive(pt, c, plan, vid):
    for t, road in plan:
        pt.step(c, t, [vid] if road is None else [], {vid: {0x50: road}} if road is not None else {})


def _app_car(share=0.25):
    return next(f"live{i}" for i in range(1000) if uses_app(f"live{i}", share))


def test_the_same_cars_run_the_app_at_any_share():
    ids = [f"live{i}" for i in range(2000)]
    s10 = {v for v in ids if uses_app(v, 0.10)}
    s25 = {v for v in ids if uses_app(v, 0.25)}
    assert s10 <= s25 and 0.2 < len(s25) / len(ids) < 0.3
    assert not uses_app("pt_L1_0", 1.0) and not uses_app("inc1_tow0", 1.0)   # buses, the accidents' vehicles: no


def test_a_roads_live_time_is_the_median_of_the_last_2_minutes_shrunk_to_its_usual_one():
    pt, c = ProbeTraffic(share=1.0), NS(simulation=_Sim())
    vid = _app_car(1.0)
    _drive(pt, c, [(0, "A"), (2, ":j_0"), (3, "B"), (40, ":k_0"), (41, "C")], vid)   # B: 37 s (left A at 2, B at 40)
    assert pt.live_time("B", 41.0) == 38.0
    assert pt.live_time("B", 41.0, usual=20.0, shrink=1.0) == pytest.approx((38.0 + 20.0) / 2)
    assert pt.live_time("B", 41.0 + 200.0) is None or pt.recent["B"]          # (only cleared at the next step)
    pt.step(c, 300.0, [], {})
    assert pt.live_time("B", 300.0) is None                                     # older than 2 minutes: gone
    assert pt.live_time("A", 41.0) is None                                      # its first road: not timed


def test_time_left_blends_live_times_in_by_how_soon_the_car_gets_there():
    te = NS(typical={("a", "b"): 30.0, ("b", "c"): 30.0}, len={"a": 100.0, "b": 100.0, "c": 50.0},
            speed={"a": 10.0, "b": 10.0, "c": 10.0}, free=lambda e: 10.0)
    usual = time_left(te, None, ["a", "b", "c"], 0.0, 50.0, 0.0)
    assert usual["left_s"] == usual["usual_s"] == 30.0 + 30.0 + 5.0
    slow = NS(live_time=lambda road, now, usual=None, shrink=0.0: 90.0)       # every road three times slower
    near = time_left(te, slow, ["a", "b", "c"], 0.0, 50.0, 0.0, blend_s=1e9)    # live everywhere
    assert near["left_s"] == pytest.approx(90.0 + 90.0 + 5.0, abs=0.01) and near["usual_s"] == usual["usual_s"]
    far = time_left(te, slow, ["a", "b", "c"], 0.0, 50.0, 0.0, blend_s=30.0)    # the second road far: mostly usual
    assert near["left_s"] > far["left_s"] > usual["left_s"]
    assert [lv for _, lv in near["roads"]] == [2, 2, None]                      # live 3x usual: red; the last: none


def test_colours():
    assert level(None, 30.0) is None and level(30.0, 30.0) == 0 and level(110.0, 30.0) == 3
    assert delay_level(100.0, 100.0) == 0 and delay_level(130.0, 100.0) == 1 and delay_level(150.0, 100.0) == 2
    assert delay_level(30.0, 15.0) == 0          # 15 s more: under 20 s, still green


def test_the_roads_ahead_of_a_ring_cars_trip_come_from_its_plan_not_the_nearest_lap():
    """(11 Nov) A ring car's route goes round 200 times: its last road shows up every lap. The time left took the
    nearest one, 83 s short on the dev seeds."""
    c = NS(vehicle=NS(getRoute=lambda v: ["37", "35"] * 100, getRouteIndex=lambda v: 4))
    te = NS(_end=lambda v: (300.0, 0, 3))
    leg = NS(dest_edge="37", end_v=0)
    assert roads_ahead(te, c, "hw1", leg) == ["37", "35", "37"]
    town = NS(dest_edge="c", end_v=None)
    c2 = NS(vehicle=NS(getRoute=lambda v: ["a", "b", "c", "d"], getRouteIndex=lambda v: 1))
    assert roads_ahead(NS(_end=lambda v: None), c2, "live1", town) == ["b", "c"]
    c2.vehicle.getRoute = lambda v: ["a", "b", "d"]
    assert roads_ahead(NS(_end=lambda v: None), c2, "live1", town) is None   # its end off its way

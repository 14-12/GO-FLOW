"""The hospital, the police station and the recovery yard (cosim/depots.py): where they are on Town05, their slots,
their signs, and their vehicles going out and coming back (TraCI stood in for)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cosim import depots as dp

ROOT = Path(__file__).resolve().parents[1]


def test_slots_leave_room_to_pull_out():
    s = dp.slots(63.3, 6.9, 4)
    assert s == sorted(s, reverse=True) and len(s) == 4
    assert all(a - b >= 6.9 + dp.SLOT_GAP_M - 1e-6 for a, b in zip(s, s[1:]))
    assert s[-1] - 6.9 >= 3.0
    assert len(dp.slots(25.0, 7.94, 5)) == 1  # only what fits


def test_town05_depots_are_on_parking_lanes_with_their_signs_on_the_pavement():
    sumolib = pytest.importorskip("sumolib")
    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    for plan in dp.TOWN05:
        lane = net.getEdge(plan.edge).getLanes()[plan.lane]
        assert 2.9 <= lane.getWidth() <= 3.1 and lane.getSpeed() < 2.0  # a 3 m parking lane
        s = dp.slots(lane.getLength(), plan.vehicle_len, plan.fleet, gap=plan.gap, keep_clear=plan.keep_clear)
        assert len(s) == plan.fleet
        # nobody parked at a bus stop on that lane (L2_S1 on -52_2), and its buses may use the lane
        for stop in ET.parse(ROOT / "scenarios" / "town05" / "town05.pt.add.xml").iter("busStop"):
            if stop.get("lane") == lane.getID():
                a, b = float(stop.get("startPos")), float(stop.get("endPos"))
                assert all(f <= a - 2.0 or f - plan.vehicle_len >= b + 2.0 for f in s), (plan.name, s)
                assert "bus" in plan.also
        x, y, facing = dp.sign_spot(net, plan)
        # on the pavement: closer to a pavement lane than to any lane for cars
        lanes = net.getNeighboringLanes(x, y, 6.0)
        near = min(lanes, key=lambda t: t[1])[0]
        assert near.allows("pedestrian") and not near.allows("passenger"), (plan.name, near.getID())
    img = dp.sign_texture(dp.TOWN05[0])
    assert img.size == (768, 288)


class _V:
    def __init__(self):
        self.stops, self.routes, self.types, self.stopped, self.road = {}, {}, {}, {}, {}
        self.resumed, self.targets = [], {}

    def add(self, vid, rid, typeID, **kw):
        self.types[vid] = typeID
        self.road[vid] = "E"

    def setStop(self, vid, edge, pos, laneIndex, duration):
        self.stops[vid] = (edge, pos, laneIndex)

    def setRoute(self, vid, edges):
        self.routes[vid] = list(edges)

    def isStopped(self, vid):
        return self.stopped.get(vid, False)

    def resume(self, vid):
        self.resumed.append(vid)
        self.stopped[vid] = False

    def setSpeed(self, vid, v):
        pass

    def changeTarget(self, vid, edge):
        self.targets[vid] = edge

    def getRoadID(self, vid):
        return self.road.get(vid, "X")

    def setType(self, vid, t):
        self.types[vid] = t

    def getIDList(self):
        return list(self.types)


def _depots():
    v = _V()
    conn = NS(vehicle=v, route=NS(getIDList=lambda: [], add=lambda *a: None),
              vehicletype=NS(getLength=lambda t: 5.0), lane=NS(setMaxSpeed=lambda *a: None, setAllowed=lambda *a: None),
              simulation=NS(findRoute=lambda a, b, vType="": NS(edges=(a, "M", b), length=400.0)))
    lane = NS(getLength=lambda: 60.0)
    net = NS(hasEdge=lambda e: True, getEdge=lambda e: NS(getLanes=lambda: [None, None, lane]))
    plan = dp.DepotPlan("police", "POLICE", "", "E", 2, 5.0, 3)
    return dp.Depots(conn, net, [plan]), v


def test_parked_out_and_back():
    d, v = _depots()
    for k in range(4):
        d.add_parked("police", f"p{k}", "twin_police")
    assert d.parked("police") == ["p0", "p1", "p2"]  # three slots: the fourth has no room
    vid = d.take("police", keep=1)
    assert vid == "p0"  # the front one drives straight out
    assert d.leave(vid, ["E", "M", "Z"]) and v.routes[vid] == ["E", "M", "Z"]
    assert d.take("police", ok=lambda x: x != "p1") == "p2"
    d.put_back("p2")
    assert d.take("police", keep=2) is None
    # back home: a free slot, its stop there; parked once it stands in it
    assert d.send_home(vid) and v.targets[vid] == "E" and v.stops[vid][0] == "E"
    d.step({vid: NS(), "p1": NS(), "p2": NS()})
    assert vid not in d.parked("police")
    v.stopped[vid], v.road[vid] = True, "E"
    d.step({vid: NS(), "p1": NS(), "p2": NS()})
    assert vid in d.parked("police") and d.stats["back"] == 1
    # gone from SUMO: forgotten
    del v.types["p2"]
    d.step({vid: NS(), "p1": NS()})
    assert not d.owns("p2") and "p2" not in d.parked("police")


def test_an_emergency_call_goes_out_on_a_run_and_comes_back_without_it():
    import random

    d, v = _depots()
    d.depots["hospital"] = d.depots.pop("police")
    d.depots["hospital"].plan = dp.DepotPlan("hospital", "H", "", "E", 2, 6.9, 3)
    d.add_parked("hospital", "a1", "twin_ambulance")
    ex = NS(emergency={})
    calls = dp.EmergencyCalls.__new__(dp.EmergencyCalls)
    calls.conn, calls.net, calls.depots, calls.extras, calls.rng = d.conn, None, d, ex, random.Random(1)
    calls.type_for, calls.calls, calls.spots = (lambda vid, t: t), {}, [("C", 1, 20.0)]
    calls.stats = {"calls": 0, "attended": 0, "back": 0, "no_vehicle": 0}
    calls.avoid = lambda: set()
    assert calls.spawn(None, 100.0, "ambulance", True)
    assert v.types["a1"] == "twin_ambulance_run" and ex.emergency["a1"].on_run
    assert v.stops["a1"][0] == "C"
    v.stopped["a1"], v.road["a1"] = True, "C"
    calls.step(160.0)  # there: it attends, beacons off
    assert not ex.emergency["a1"].on_run and calls.calls["a1"]["until"] is not None
    calls.step(160.0 + dp.CALL_S[1] + 1)
    assert v.types["a1"] == "twin_ambulance" and v.targets["a1"] == "E" and "a1" not in calls.calls
    assert calls.spawn(None, 400.0, "ambulance", True) and calls.stats["no_vehicle"] == 1  # none home yet


def test_no_accident_at_a_depot_on_a_tiny_road_or_beside_one_under_way():
    """A crash there is a near miss (cosim/incidents.py: its wrecks would shut the depot's road, a junction behind a
    14 m road, or another accident's scene)."""
    sumolib = pytest.importorskip("sumolib")
    from cosim import incidents as inc_mod

    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    im = inc_mod.IncidentManager.__new__(inc_mod.IncidentManager)
    lanes = {}
    im.conn = NS(vehicle=NS(getLaneID=lambda vid: lanes[vid]))
    im.net, im._pairs, im.role, im.incidents, im.distracted = net, set(), {}, [], {}
    im.stats = {"collisions": 0, "near_misses": 0, "person_hits": 0}
    im.max_active, im.cooldown_s, im.now, im.last_t = 2, 0.0, 100.0, -1e9
    im.depot_edges = {"49"}
    im.depot_junctions = set()
    halted = []
    im._halt = lambda vid, s: halted.append(vid)
    im._freeze = lambda vid: (_ for _ in ()).throw(AssertionError("an accident was made"))
    car = lambda x: NS(vclass="passenger", x=x, y=0.0, speed=8.0, type_id="t", angle=0.0, length=4.5)  # noqa: E731
    vehicles = {"a": car(0.0), "b": car(3.0)}

    def crash(lane_a, lane_b=None):
        lanes.update(a=lane_a, b=lane_b or lane_a)
        im._pairs.clear()
        im._collision(100.0, NS(collider="a", victim="b", type="collision", lane=lane_a, colliderSpeed=8.0,
                                victimSpeed=2.0), vehicles)

    crash("49_3")                   # the hospital's road
    crash("13_4")                   # 14 m into the ring junction
    im.incidents = [NS(phase="secured", x=20.0, y=0.0)]
    crash("6_4")                    # 20 m from an accident under way
    assert halted == ["a", "a", "a"] and im.stats["near_misses"] == 3

"""The ring road's own traffic (cosim/highway.py): its loops found in the network, its cars kept on the curve,
started and taken off only where the camera does not look."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cosim.highway import MIN_PER_LOOP, HighwayTraffic, find_loops

ROOT = Path(__file__).resolve().parents[1]


def test_town05_ring_is_two_loops():
    sumolib = pytest.importorskip("sumolib")
    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    loops = sorted(sorted(lp) for lp in find_loops(net))
    assert loops == [["-36", "-38"], ["35", "37"]]
    for lp in loops:  # each loop over a kilometre, three lanes for cars on every edge
        assert sum(net.getEdge(e).getLength() for e in lp) > 1300
        assert all(sum(ln.allows("passenger") for ln in net.getEdge(e).getLanes()) == 3 for e in lp)


class _Conn:
    def __init__(self):
        self.added, self.removed, self.routes, self.speeds = [], [], {}, {}
        self.road = {}
        self.vehicle = NS(add=self._add, remove=lambda v: self.removed.append(v),
                          getRoadID=lambda v: self.road.get(v, ("a", 5.0))[0],
                          getLanePosition=lambda v: self.road.get(v, ("a", 5.0))[1])
        self.route = NS(add=lambda rid, edges: self.routes.__setitem__(rid, edges))
        self.edge = NS(setMaxSpeed=lambda e, s: self.speeds.__setitem__(e, s))
        self.lane = NS(getLength=lambda ln: 500.0)

    def _add(self, vid, rid, **kw):
        self.added.append((vid, rid, kw))


def test_ring_cars_follow_the_curve_and_appear_out_of_sight():
    c = _Conn()
    share = {"v": 1.0}
    seen = {"ok": True}
    hw = HighwayTraffic(c, [["a", "b"]], lambda s: share["v"], free=lambda e, p: seen["ok"], peak=6, speed=22.0,
                        planned=False)
    hw.setup()
    assert c.speeds == {"a": 22.0, "b": 22.0}
    running = {}
    for _ in range(5):
        hw.step(running, 0.0)
        running = {vid: NS() for vid, _, _ in c.added}
    assert len(hw.cars[0]) == 6  # two at a time, up to the target
    rid = c.added[0][1]
    assert set(c.routes[rid]) == {"a", "b"} and len(c.routes[rid]) > 100  # round and round
    assert all(kw["departSpeed"] == "max" for _, _, kw in c.added)
    # at night: fewer, taken off one at a time, only where nobody looks, never an accident's
    share["v"] = 0.0
    seen["ok"] = False
    hw.step(running, 0.0)
    assert c.removed == []
    seen["ok"] = True
    hw.protected = lambda vid: vid == "hw0"
    hw.step(running, 0.0)
    assert c.removed and c.removed[0] != "hw0"
    assert hw.target(0.0) == MIN_PER_LOOP


def test_a_ring_car_waiting_to_appear_is_withdrawn():
    c = _Conn()
    hw = HighwayTraffic(c, [["a", "b"]], lambda s: 1.0, peak=2)
    hw.setup()
    hw.step({}, 0.0)
    first = [vid for vid, _, _ in c.added]
    hw.step({}, 0.0, pending=set(first))  # SUMO found no room: it would appear later, maybe in sight
    assert set(first) <= set(c.removed)


def test_the_rings_own_greens_are_kept_short():
    """At a ring junction a phase green for the ring alone is capped (the town's roads into it are short); a phase
    that also lets the town in is not; the junction agents' copy of the limits follows."""
    ph = lambda st, mx: NS(state=st, minDur=5.0, maxDur=mx, duration=42.0)  # noqa: E731
    logic = NS(programID="actuated", phases=[ph("GGr", 60.0), ph("yyr", 3.0), ph("rrG", 60.0), ph("rGG", 60.0)])
    set_logic = {}
    c = _Conn()
    c.trafficlight = NS(getIDList=lambda: ["J"], getControlledLanes=lambda t: ["a_1", "a_2", "town_0"],
                        getControlledLinks=lambda t: [[("a_1", "x", "")], [("a_2", "x", "")], [("town_0", "x", "")]],
                        getProgram=lambda t: "actuated", getAllProgramLogics=lambda t: [logic],
                        setProgramLogic=lambda t, lg: set_logic.__setitem__(t, lg))
    agent = NS(max_dur=[60.0, 3.0, 60.0, 60.0], min_dur=[5.0, 3.0, 5.0, 5.0])
    hw = HighwayTraffic(c, [["a", "b"]], lambda s: 1.0, green_max=10.0, policy=NS(junctions={"J": agent}))
    hw.setup()
    assert hw.capped == {"J": [0]}
    assert [p.maxDur for p in set_logic["J"].phases] == [10.0, 3.0, 60.0, 60.0]
    assert agent.max_dur == [10.0, 3.0, 60.0, 60.0]


def test_a_ring_car_sent_the_short_way_goes_round_again():
    """A ring car rerouted (an accident's detour) would end its trip on the ring, in sight: its route is renewed."""
    c = _Conn()
    hw = HighwayTraffic(c, [["a", "b"]], lambda s: 1.0, peak=1, planned=False)
    hw.setup()
    hw.step({}, 0.0)
    vid = c.added[0][0]
    routes = {vid: ("a", "b", "a")}
    c.vehicle.getRoute = lambda v: routes[v]
    c.vehicle.getRouteIndex = lambda v: 1
    c.vehicle.getRoadID = lambda v: "b"
    c.vehicle.setRoute = lambda v, edges: routes.__setitem__(v, tuple(edges))
    hw._k = 4
    hw.step({vid: NS()}, 0.0)
    assert routes[vid][0] == "b" and len(routes[vid]) > 100 and hw.rerouted == 1


def _planned(free=lambda e, p: True):
    c = _Conn()
    hw = HighwayTraffic(c, [["a", "b"]], lambda s: 1.0, peak=1, free=free)
    hw.setup()
    hw.step({}, 0.0)
    vid, rid, kw = c.added[0]
    routes = {vid: tuple(c.routes[rid])}
    c.vehicle.getRoute = lambda v: routes[v]
    c.vehicle.setRoute = lambda v, edges: routes.__setitem__(v, tuple(edges))
    return c, hw, vid, kw, routes


def test_a_ring_car_has_a_planned_trip_with_an_end():
    """(10 Nov) 1-3 laps from where it appears, leaving part way along its last road: a start, an end, an estimate."""
    c, hw, vid, kw, routes = _planned()
    edges = routes[vid]
    assert 3 <= len(edges) <= 7 and all(a != b for a, b in zip(edges, edges[1:]))   # along the loop
    assert 150.0 <= float(kw["arrivalPos"]) <= 440.0 and hw.end_of(vid) == (float(kw["arrivalPos"]), 0)
    assert hw.end_of("live1") is None


def test_a_ring_car_whose_end_is_in_sight_drives_one_more_lap():
    seen = {"end": False}                                         # it appears out of sight
    c, hw, vid, kw, routes = _planned(free=lambda e, p: not seen["end"])
    seen["end"] = True
    edges = routes[vid]
    c.vehicle.getRouteIndex = lambda v: len(edges) - 3          # not yet near its end: left alone
    hw.step({vid: NS()}, 0.0)
    assert routes[vid] == edges and hw.extended == 0
    c.vehicle.getRouteIndex = lambda v: len(edges) - 2          # the road before its last, its end in sight
    hw.step({vid: NS()}, 0.0)
    assert len(routes[vid]) == 2 + 2 and routes[vid][:2] == edges[-2:] and hw.extended == 1
    assert hw.end_of(vid)[1] == 1                                # its end moved: a new destination for its estimate
    seen["end"] = False                                          # out of sight: it ends there
    c.vehicle.getRouteIndex = lambda v: len(routes[v]) - 1
    hw.step({vid: NS()}, 0.0)
    assert hw.extended == 1


def test_a_planned_ring_car_given_another_way_goes_round_to_a_trip_end():
    c, hw, vid, kw, routes = _planned()
    routes[vid] = ("b", "a")                                     # an accident's detour: to the last road, the short way
    c.vehicle.getRouteIndex = lambda v: 0
    c.vehicle.getRoadID = lambda v: "b"
    hw.step({vid: NS()}, 0.0)
    assert routes[vid] == ("b", "a", "b") and hw.rerouted == 1

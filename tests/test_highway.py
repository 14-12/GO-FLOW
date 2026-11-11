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
        self.vehicle = NS(add=self._add, remove=lambda v, reason=3: self.removed.append(v),
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
    c.road[vid] = (c.routes[rid][0], 20.0)
    return c, hw, vid, kw


def _drive(c, hw, vid, roads, pos):
    """The car on each of `roads` in turn (one ring step each), then at pos on the last."""
    for r in roads:
        c.road[vid] = (r, 5.0)
        hw.step({vid: NS()}, 0.0)
    c.road[vid] = (roads[-1], pos)
    hw.step({vid: NS()}, 0.0)


def test_a_ring_car_has_a_planned_trip_with_an_end():
    """(10 Nov) 1-3 laps from where it appears, leaving part way along its last road: a start, an end, an estimate.
    (11 Nov) SUMO keeps its 200 laps: a looping route given again with an end position, SUMO 1.27 took the car off."""
    c, hw, vid, kw = _planned()
    rid = c.added[0][1]
    assert len(c.routes[rid]) > 100 and "arrivalPos" not in kw
    end, v, n = hw.end_of(vid)
    assert 150.0 <= end <= 440.0 and v == 0 and 3 <= n <= 7
    assert hw.end_of("live1") is None


def test_a_ring_car_leaves_at_its_end_out_of_sight_and_drives_on_in_sight():
    seen = {"car": False}
    c, hw, vid, kw = _planned(free=lambda e, p: not seen["car"])
    end, _, n = hw.end_of(vid)
    first = c.road[vid][0]
    roads = [("b" if first == "a" else "a") if j % 2 else first for j in range(1, n)]   # the next n-1 roads
    seen["car"] = True                                           # its end in sight: one more lap, not taken off
    _drive(c, hw, vid, roads, end + 5.0)
    assert hw.extended == 1 and vid not in c.removed and hw.end_of(vid) == (end, 1, 3)   # the road it is on + a lap
    seen["car"] = False                                          # a lap later, out of sight: it leaves there
    nxt = "b" if roads[-1] == "a" else "a"
    _drive(c, hw, vid, [nxt, roads[-1]], end + 1.0)
    assert vid in c.removed and hw.arrived == 1 and hw.end_of(vid) is None
    assert hw.take_ended() == [vid] and hw.take_ended() == []   # for the trip estimates: SUMO lists it nowhere


def test_a_ring_car_near_its_end_is_watched_every_step():
    """(11 Nov) The ring's step comes every 2 s (44 m at 80 km/h): near its end, check_ends looks every step."""
    c, hw, vid, kw = _planned()
    end, _, n = hw.end_of(vid)
    first = c.road[vid][0]
    roads = [("b" if first == "a" else "a") if j % 2 else first for j in range(1, n)]
    _drive(c, hw, vid, roads, end - 30.0)                        # on its last road, short of its end
    assert vid in hw.near_end and vid not in c.removed
    c.road[vid] = (roads[-1], end + 2.0)
    hw.check_ends()
    assert vid in c.removed


def test_a_ring_car_sent_the_short_way_goes_round_again_at_the_next_step():
    """(11 Nov) An accident on the ring re-routes the cars heading for it to the last road of their route, close by:
    check_ends sees the route id change and puts the car back on its laps at once (it ended in sight, 7 at once)."""
    c, hw, vid, kw = _planned()
    rid = c.added[0][1]
    routes = {vid: tuple(c.routes[rid])}
    ids = {vid: rid}
    c.vehicle.getRoute = lambda v: routes[v]
    c.vehicle.getRouteIndex = lambda v: 0
    c.vehicle.getRouteID = lambda v: ids[v]
    c.vehicle.setRoute = lambda v, edges: (routes.__setitem__(v, tuple(edges)), ids.__setitem__(v, "!" + v + "#2"))
    hw.check_ends({vid: {0x53: rid}})                     # its route id seen once
    routes[vid], ids[vid] = ("b", "a"), "!" + vid + "#1"  # the accident's detour: two roads left
    c.road[vid] = ("b", 30.0)
    hw.check_ends({vid: {0x53: ids[vid]}})
    assert len(routes[vid]) > 100 and routes[vid][0] == "b" and hw.rerouted == 1 and vid not in hw.round_again

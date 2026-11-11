"""(9 Nov) Cars held up behind an accident or in a knot get a new way first; they are taken off the road only as a
last resort, and every removal is counted (IncidentManager.removals)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from cosim.incidents import RELIEVE_REMOVE_S, RELIEVE_REROUTE_S, IncidentManager


class _Veh:
    def __init__(self, roads):
        self.roads, self.rerouted, self.removed = roads, [], []

    def getRoadID(self, vid):
        return self.roads[vid]

    def rerouteTraveltime(self, vid, currentTravelTimes=False):
        self.rerouted.append(vid)

    def getStopState(self, vid):
        return 1 if vid.startswith("stop") else 0

    def getLaneID(self, vid):
        return "" if vid.startswith("off") else self.roads[vid] + "_0"


def _manager(roads):
    im = IncidentManager.__new__(IncidentManager)
    veh = _Veh(roads)
    im.conn = NS(vehicle=veh)
    feeder = NS(getID=lambda: "feeder")
    im.net = NS(getEdge=lambda e: NS(getIncoming=lambda: [feeder]))
    im.role, im.standing, im.stats, im.visible = {}, {}, {}, None
    im.protected = lambda vid: False
    im._keep_lane = lambda vid, near_end=None: False
    im._remove = lambda vid: veh.removed.append(vid)
    return im, veh


def test_held_up_cars_get_a_new_way_before_any_removal():
    im, veh = _manager({"a": "crash_road", "b": "feeder"})
    inc = NS(diverted=["crash_road"])
    standing = {"a": NS(speed=0.0, x=0.0, y=0.0), "b": NS(speed=0.0, x=0.0, y=0.0)}
    im._relieve(inc, 0.0, standing)
    im._relieve(inc, 30.0, standing)
    assert not veh.rerouted and not veh.removed           # 30 s at a red light: nothing
    im._relieve(inc, RELIEVE_REROUTE_S + 1, standing)
    assert sorted(veh.rerouted) == ["a", "b"] and not veh.removed
    im._relieve(inc, RELIEVE_REMOVE_S + 1, standing)
    assert veh.removed == ["a"]                           # on the accident's own road only, never on a road into it
    assert im.removals() == {"behind_accident": 1, "knots": 0}


def test_nobody_in_sight_is_taken_off():
    im, veh = _manager({"a": "crash_road"})
    im.visible = lambda x, y: True
    inc = NS(diverted=["crash_road"])
    standing = {"a": NS(speed=0.0, x=0.0, y=0.0)}
    for t in (0.0, RELIEVE_REROUTE_S + 1, RELIEVE_REMOVE_S + 1, 2 * RELIEVE_REMOVE_S):
        im._relieve(inc, t, standing)
    assert veh.rerouted == ["a"] and not veh.removed


def test_parked_and_stopped_cars_are_never_taken_for_held_up():
    """(10 Nov) 6 of 21 cars taken off on two dev seeds were parked in a car park near an accident."""
    im, veh = _manager({"p": "crash_road", "stop1": "crash_road", "off1": "crash_road"})
    im.parked = {"p": None}
    inc = NS(diverted=["crash_road"])
    standing = {v: NS(speed=0.0, x=0.0, y=0.0) for v in ("p", "stop1", "off1")}
    for t in (0.0, RELIEVE_REROUTE_S + 1, RELIEVE_REMOVE_S + 1, 2 * RELIEVE_REMOVE_S):
        im._relieve(inc, t, standing)
    assert not veh.rerouted and not veh.removed


def test_movements_past_a_wreck_in_a_junction_are_blocked():
    """(10 Nov) A crash inside a junction blocks the movements whose way through passes the wreck, not the others."""
    from types import SimpleNamespace as NS2

    def lane(lid, shape):
        return NS2(getID=lambda: lid, getShape=lambda: shape, allows=lambda c: True)

    lanes = {":J_0_0": lane(":J_0_0", [(0.0, 0.0), (10.0, 0.0)]),   # straight through the middle
             ":J_1_0": lane(":J_1_0", [(0.0, 0.0), (0.0, -8.0)])}   # a right turn, away from it

    def conn(frm, to, via):
        return NS2(getViaLaneID=lambda: via, getFromLane=lambda: lane(frm, []), getTo=lambda: NS2(getID=lambda: to))

    node = NS2(getConnections=lambda: [conn("a_0", "b", ":J_0_0"), conn("a_0", "c", ":J_1_0")])
    im = IncidentManager.__new__(IncidentManager)
    im.net = NS2(getNode=lambda j: node, getLane=lambda lid: lanes[lid])
    wreck = NS2(removed=False, lane=":J_0_0", x=6.0, y=0.5, angle=90.0, length=4.0)
    inc = NS2(wrecks=[wreck])
    junction, blocked = im._blocked_links(inc)
    assert junction == "J" and blocked == {("a_0", "b")}


def test_ring_cars_are_not_sent_another_way():
    """(11 Nov) The ring road's own cars have no other way than the ring: sent "another way" to the end of their
    route, a ring road close by, they ended their trip at once, in view. They keep their route (and pass on the lane
    left open)."""
    im, veh = _manager({"hw3": "crash_road", "a": "crash_road"})
    inc = NS(diverted=["crash_road"])
    standing = {"hw3": NS(speed=0.0, x=0.0, y=0.0), "a": NS(speed=0.0, x=0.0, y=0.0)}
    im._relieve(inc, 0.0, standing)
    im._relieve(inc, RELIEVE_REROUTE_S + 1, standing)
    assert veh.rerouted == ["a"]

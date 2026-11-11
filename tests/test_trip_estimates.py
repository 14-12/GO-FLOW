"""cosim/trip_estimates.py: every car's trip time, estimated when it sets off."""

from __future__ import annotations

from types import SimpleNamespace as NS

from cosim.trip_estimates import TripEstimates


class _Edge:
    def __init__(self, eid, length, speed):
        self._id, self._len, self._speed = eid, length, speed

    def getID(self):
        return self._id

    def getLength(self):
        return self._len

    def getSpeed(self):
        return self._speed

    def getFunction(self):
        return ""


def _te(**kw):
    net = NS(getEdges=lambda: [_Edge("a", 100.0, 10.0), _Edge("b", 200.0, 10.0), _Edge("c", 50.0, 10.0)])
    te = TripEstimates(net, typical_file=None, **kw)
    te.typical = {("a", "b"): 25.0, ("b", "c"): 20.0}   # a: 10 s drive + 15 s at its signal; b: free (20 s)
    return te


def test_estimate_counts_the_first_road_from_where_it_is_and_the_last_up_to_its_stop():
    te = _te()
    est, free, share = te.estimate(["a", "b", "c"], 50.0, 25.0)
    # a from 50 m: 5 s drive + its 15 s wait; b: 20 s; c up to 25 m: 2.5 s
    assert round(est, 1) == 42.5 and round(free, 1) == 27.5 and 0.9 < share < 1.0
    est1, free1, _ = te.estimate(["b"], 20.0, 120.0)            # one road: start to stop at its limit
    assert est1 == free1 == 10.0
    te.typical = {}
    est2, free2, share2 = te.estimate(["a", "b"], 0.0, 200.0)   # nothing known: free flow, and said so
    assert est2 == free2 == 30.0 and share2 == 0.0


class _Conn:
    """What TripEstimates reads: the simulation's and the vehicles' subscriptions, a car's route and stops."""

    def __init__(self):
        self.sim, self.veh = {}, {}
        self.routes = {"car": (["a", "b", "c"], 0, 10.0, "r1", [NS(lane="c_0", endPos=30.0, stoppingPlaceID="pa1")]),
                       "bus": (["a", "b"], 0, 0.0, "r2", [])}
        self.simulation = NS(getSubscriptionResults=lambda: self.sim, subscribe=lambda v: None)
        self.vehicle = NS(getAllSubscriptionResults=lambda: self.veh,
                          getVehicleClass=lambda v: "bus" if v == "bus" else "passenger",
                          getRoute=lambda v: self.routes[v][0], getRouteIndex=lambda v: self.routes[v][1],
                          getLanePosition=lambda v: self.routes[v][2], getRouteID=lambda v: self.routes[v][3],
                          getStops=lambda v, n=1: self.routes[v][4][:n],
                          getStopState=lambda v: self.stop_state.get(v, 0))
        self.stop_state = {}


def test_a_trip_runs_from_its_departure_to_its_parking_and_keeps_its_first_estimate():
    te = _te(area_names={"pa1": "Marché Central"})
    c = _Conn()
    c.sim = {0x74: ("car", "bus")}
    te.step(c, 100.0)
    assert set(te.active) == {"car"} and te.stats["skipped"] == 1     # buses are not estimated
    leg = te.active["car"]
    assert leg.dest_edge == "c" and leg.dest_name == "Marché Central" and leg.est_s == 47.0  # 9 + 15, 20, 3 s
    c.sim = {}
    c.veh = {"car": {0x50: "b", 0x53: "r1"}}
    te.step(c, 120.0)
    c.routes["car"] = (["b", "x", "c"], 0, 0.0, "r1b", c.routes["car"][4])
    c.veh = {"car": {0x50: "b", 0x53: "r1b"}}                         # another way, the same destination
    te.step(c, 130.0)
    assert leg.changes == [(30.0, "new way")] and not leg.dest_changed and leg.est_s == 47.0
    c.veh = {"car": {0x50: "c", 0x53: "r1b"}}
    c.sim = {0x6d: ("car",)}
    te.step(c, 160.0)
    assert not te.active and te.done[0].how == "parked" and te.done[0].real_s == 60.0
    snap = te.snapshot("car", 170.0)
    assert snap["ended"] == "parked" and snap["elapsed_s"] == 60.0 and snap["dest"] == "Marché Central"


def test_pulling_out_of_a_space_the_trip_goes_to_the_next_stop_not_the_space_it_leaves():
    te = _te(area_names={"pa0": "Tour Horizon", "pa1": "Marché Central"})
    c = _Conn()
    # parked on road a at pa0; SUMO reports the parking's end while the car still stands there (the stop not dropped)
    c.routes["car"] = (["a", "b", "c"], 0, 10.0, "r1", [NS(lane="a_0", endPos=10.0, stoppingPlaceID="pa0"),
                                                         NS(lane="c_0", endPos=30.0, stoppingPlaceID="pa1")])
    c.stop_state["car"] = 131
    te.at_area["car"] = "pa0"
    c.sim = {0x6f: ("car",)}
    te.step(c, 100.0)
    leg = te.active["car"]
    assert leg.start_name == "Tour Horizon" and leg.dest_edge == "c" and leg.dest_name == "Marché Central"
    assert leg.est_s == 47.0


def test_a_shared_car_park_is_named_after_the_place_the_car_goes_to():
    from cosim.trip_estimates import plan_names

    demand = NS(plans={"car": NS(dest="brasserie", home="pins", origin="pins")},
                places=NS(park_of={"brasserie": ["pa1"], "hopital": ["pa1"], "pins": ["pa9"]},
                          places={"brasserie": {"name": "Brasserie du Nord"}, "hopital": {"name": "Hôpital"},
                                  "pins": {"name": "Quartier des Pins"}}))
    te = _te(area_names={"pa1": "Hôpital", "pa9": "Quartier des Pins"}, place_name=plan_names(demand))
    c = _Conn()
    c.sim = {0x74: ("car",)}
    te.step(c, 0.0)
    assert te.active["car"].dest_name == "Brasserie du Nord"
    assert te.name("other", "pa1") == "Hôpital"       # no plan: the car park's own name


def test_a_car_put_straight_into_its_space_is_not_a_trip():
    te = _te()
    c = _Conn()
    c.sim = {0x74: ("car",)}
    te.step(c, 10.0)
    c.sim = {0x6d: ("car",)}
    te.step(c, 11.0)
    assert not te.done and te.stats["dropped"] == 1 and te.snapshot("car", 12.0) is None


def test_a_new_destination_ends_the_trip_and_starts_another_with_its_own_estimate():
    te = _te()
    c = _Conn()
    c.sim = {0x74: ("car",)}
    te.step(c, 0.0)
    c.sim = {}
    c.routes["car"] = (["b", "a"], 0, 50.0, "r9", [])
    c.veh = {"car": {0x50: "b", 0x53: "r9"}}
    te.step(c, 20.0)
    assert te.done[0].how == "extended" and te.done[0].dest_changed
    nxt = te.active["car"]
    assert nxt.t0 == 20.0 and nxt.dest_edge == "a" and nxt.changes == [(0.0, "new destination given on the way")]


def test_a_ring_car_ends_at_its_planned_spot_and_one_more_lap_is_a_new_destination():
    """(10 Nov, 11 Nov) Ring-road cars (cosim/highway.py) have planned trips: the estimate runs over the roads planned
    (SUMO's route goes round 200 times) to the spot on the last where they leave; when that end moves (one more lap: it
    was in sight) the trip ends "extended" and another starts."""
    ends = {"car": (20.0, 0, 3)}
    te = _te(end_of=lambda v: ends.get(v))
    c = _Conn()
    c.routes["car"] = (["a", "b", "c", "a", "b", "c"], 0, 10.0, "r1", [])
    c.sim = {0x74: ("car",)}
    te.step(c, 0.0)
    leg = te.active["car"]
    assert leg.route == ["a", "b", "c"] and leg.dest_pos == 20.0 and leg.est_s == 9.0 + 15.0 + 20.0 + 2.0
    c.sim = {}
    ends["car"] = (20.0, 1, 3)                                   # its end one lap on: no new route in SUMO
    c.veh = {"car": {0x50: "b", 0x53: "r1"}}
    te.step(c, 30.0)
    assert te.done[0].how == "extended" and te.active["car"].t0 == 30.0

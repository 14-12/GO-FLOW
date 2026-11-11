"""Every car's trip time, estimated when it sets off (10 Nov), as a navigation app would: the estimate is made once,
at the start, and the real time is set beside it at the end.

A trip ("leg") runs from where a car sets off (put on the road, or pulling out of a parking space) to where it stops
(parks at a place) or its route ends. Its estimate: the route's typical time from past trips (the agent's table,
models/route_guidance/movement_times.json: each road's time by the road taken next, the wait at its signal
included), the road it starts on from where it is, the one it ends on up to where it stops; a road the table does not
know counts its free-flow time (length / limit), and the estimate says how much of it is from the table.

Never estimated: buses, trucks (and the recovery trucks), the incidents' stand-ins. The estimate is never made again
during the trip; when the town gives the car another way (an accident's detour, a knot undone, the route agent, a new
destination given in view) the trip keeps its estimate and notes when (`changes`), and a new destination is said so
(`dest_changed`): the accuracy is then reported apart.

Reads what SUMO already sends with every step where it can (the simulation's and the vehicles' subscriptions:
departures, parkings, road, lane position, route id); asks only for a car's route and stops when it sets off.
Used by the viewer (the followed car's panel, cosim/route_minimap.py) and tools/eval_trip_estimates.py.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPICAL = ROOT / "models" / "route_guidance" / "movement_times.json"
# (10 Nov) the ring road's movements from its own cars' trips (tools/route_delay_table.py --prefix hw, train seeds): the
# town's cars seldom drive the ring, and those that do wait at its junctions to turn off; ring cars go round
RING_TIMES = ROOT / "models" / "trip_estimates" / "ring_movement_times.json"
log = logging.getLogger("trip_estimates")

# traci.constants
_DEPARTED = 0x74          # VAR_DEPARTED_VEHICLES_IDS
_PARK_START = 0x6d        # VAR_PARKING_STARTING_VEHICLES_IDS
_PARK_END = 0x6f          # VAR_PARKING_ENDING_VEHICLES_IDS
_STOP_START = 0x69        # VAR_STOP_STARTING_VEHICLES_IDS (simulation domain)
_STOP_END = 0x6b          # VAR_STOP_ENDING_VEHICLES_IDS
_ROAD = 0x50              # VAR_ROAD_ID
_POS = 0x56               # VAR_LANEPOSITION
_ROUTE_ID = 0x53          # VAR_ROUTE_ID

SKIP_CLASSES = ("bus", "truck", "trailer", "coach")
SKIP_PREFIXES = ("pt_", "incblock")
MIN_TRIP_S = 3.0          # a "trip" that parks again within this, where it started: a car put into its space


@dataclass
class Leg:
    vid: str
    t0: float
    start_edge: str
    start_pos: float
    start_name: str | None        # the parking area's place, if it pulled out of one
    dest_edge: str
    dest_pos: float
    dest_name: str | None         # the place it parks at, if it is going to one
    est_s: float                  # the estimate made when it set off
    free_s: float                 # the free-flow time of the same route
    table_share: float            # share of the estimate from the typical-times table (the rest: free flow)
    route_id: str = ""
    route: list = field(default_factory=list)
    changes: list = field(default_factory=list)   # [(t, what)]
    dest_changed: bool = False
    t1: float | None = None
    how: str | None = None        # parked | arrived | stopped | extended (another destination) | gone (taken off)
    last_road: str = ""
    end_v: int | None = None      # a ring car's planned end, its version (cosim/highway.HighwayTraffic.end_of:
                                  # (m along its last road, version, roads left))

    @property
    def real_s(self) -> float | None:
        return None if self.t1 is None else self.t1 - self.t0


class TripEstimates:
    def __init__(self, net, typical_file: Path | None = TYPICAL, speeds: dict | None = None, area_names: dict | None = None,
                 stat: str = "median_s", place_name=None, end_of=None, ring_file: Path | None = RING_TIMES):
        """net: sumolib network (roads' lengths and limits); speeds: road -> limit set at run time (the ring road's
        80 km/h); area_names: parking area -> place name; stat: the table's "median_s" (each road's median: on dev seeds 11-20
        within 20 % 0.411 against 0.338 with the mean, the mean's sum overestimating by 23 s, docs/WORKLOG.md 10 Nov) or
        "mean_s" per road. place_name(vid, area) -> the place this car goes to or leaves at that parking area (its
        trip's plan: plan_names), else the area's own name (a car park serves several places)."""
        self.len = {}
        self.speed = {}
        for e in net.getEdges():
            if e.getFunction() == "internal":
                continue
            self.len[e.getID()] = e.getLength()
            self.speed[e.getID()] = e.getSpeed()
        self.speed.update(speeds or {})
        self.typical: dict[tuple[str, str], float] = {}
        if typical_file is not None and Path(typical_file).is_file():
            card = json.loads(Path(typical_file).read_text(encoding="utf-8"))
            for k, v in card["movements"].items():
                a, _, b = k.partition(">")
                if b and v["n"] >= 3:
                    self.typical[(a, b)] = float(v[stat])
        ring = set(speeds or {})
        if ring and ring_file is not None and Path(ring_file).is_file():   # ring road to ring road: its own cars'
            card = json.loads(Path(ring_file).read_text(encoding="utf-8"))
            for k, v in card["movements"].items():
                a, _, b = k.partition(">")
                if a in ring and b in ring and v["n"] >= 3:
                    self.typical[(a, b)] = float(v[stat])
        self.area_names = area_names or {}
        self.place_name = place_name
        self.end_of = end_of      # vid -> (m along its last road, version, roads left) for ring cars, else None
        self.live: dict[str, dict] = {}   # vid -> its time left now, the app's live traffic in (cosim/live_eta.for_car)
        self.active: dict[str, Leg] = {}
        self.done: list[Leg] = []
        self.last: dict[str, Leg] = {}     # each car's latest leg (the viewer's panel), kept after it ends
        self.vclass: dict[str, str] = {}
        self.at_area: dict[str, str] = {}       # parked car -> its parking area (named when it pulls out)
        self.stopped: set[str] = set()          # a trip ended at a stop on its way's end (police at a scene)
        self.stats = {"legs": 0, "skipped": 0, "dropped": 0}

    # ------------------------------------------------------------------ the estimate
    def free(self, e: str) -> float:
        return self.len.get(e, 0.0) / max(self.speed.get(e, 13.89), 0.1)

    def estimate(self, edges: list[str], pos0: float, end_pos: float) -> tuple[float, float, float]:
        """(typical s, free-flow s, share from the table) of driving `edges` from pos0 on the first to end_pos on the
        last. Each road but the last: its typical time by the road taken next (its signal's wait in it); the first
        from pos0 (its drive scaled, its wait whole); the last up to end_pos at its limit (no signal before a stop)."""
        if not edges:
            return 0.0, 0.0, 1.0
        if len(edges) == 1:
            s = max(0.0, end_pos - pos0) / max(self.speed.get(edges[0], 13.89), 0.1)
            return s, s, 0.0
        tot = free = known = 0.0
        for i, (a, b) in enumerate(itertools.pairwise(edges)):
            fa = self.free(a)
            t = self.typical.get((a, b))
            k = fa if t is None else t
            if i == 0:
                share = max(0.0, 1.0 - pos0 / max(self.len.get(a, 1.0), 1.0))
                drive = fa * share
                k = drive + (max(0.0, t - fa) if t is not None else 0.0)
                fa = drive
            tot += k
            free += fa
            if t is not None:
                known += k
        last = edges[-1]
        tail = min(end_pos, self.len.get(last, end_pos)) / max(self.speed.get(last, 13.89), 0.1)
        tot += tail
        free += tail
        return tot, free, (known / tot if tot > 0 else 0.0)

    # ------------------------------------------------------------------ every step
    def skip(self, c, vid: str) -> bool:
        if vid.startswith(SKIP_PREFIXES) or "tow" in vid:
            return True
        vc = self.vclass.get(vid)
        if vc is None:
            try:
                vc = self.vclass[vid] = c.vehicle.getVehicleClass(vid)
            except Exception:  # noqa: BLE001
                return True
        return vc in SKIP_CLASSES

    def setup(self, c, sumo=None) -> None:
        """The departures and parkings come with the step (the simulation's subscription, widened); the vehicles'
        route ids and lane positions with theirs (the SumoSide's own list, before the first car departs)."""
        try:
            from cosim.sumo_side import _SIM_VARS
            c.simulation.subscribe(list(dict.fromkeys(list(_SIM_VARS) + [_PARK_START, _PARK_END, _STOP_START,
                                                                          _STOP_END])))
        except Exception as e:  # noqa: BLE001
            log.info("trip estimates: simulation subscription: %s", e)
        vv = getattr(sumo, "_veh_vars", None)
        if vv is not None:
            for var in (_ROAD, _POS, _ROUTE_ID):
                if var not in vv:
                    vv.append(var)

    def name(self, vid: str, area: str) -> str | None:
        if self.place_name is not None:
            try:
                n = self.place_name(vid, area)
            except Exception:  # noqa: BLE001
                n = None
            if n:
                return n
        return self.area_names.get(area)

    def _end(self, vid: str):
        if self.end_of is None:
            return None
        try:
            return self.end_of(vid)
        except Exception:  # noqa: BLE001
            return None

    def _open(self, c, vid: str, now: float, start_name: str | None = None) -> None:
        if self.skip(c, vid):
            self.stats["skipped"] += 1
            return
        try:
            route = list(c.vehicle.getRoute(vid))
            idx = c.vehicle.getRouteIndex(vid)
            pos = c.vehicle.getLanePosition(vid)
            rid = c.vehicle.getRouteID(vid)
            stops = c.vehicle.getStops(vid, 2)
            # (10 Nov) SUMO says a car's parking ended one step before it drops that stop: while the car still stands
            # at it, the first stop is the space it is leaving, not where it goes (43 % of the trips to a place were
            # estimated 0 s, to the space they left: dev seeds 11-20)
            if stops and c.vehicle.getStopState(vid) & 1:
                stops = stops[1:]
        except Exception:  # noqa: BLE001 - gone already
            return
        if idx < 0 or idx >= len(route):
            idx = 0
        edges = route[idx:]
        dest_edge, dest_pos, dest_name = edges[-1], self.len.get(edges[-1], 0.0), None
        if stops:
            s = stops[0]
            se = s.lane.rsplit("_", 1)[0] if s.lane else ""
            if se in edges:
                j = edges.index(se)
                edges = edges[:j + 1]
                dest_edge, dest_pos = se, float(s.endPos)
                dest_name = self.name(vid, s.stoppingPlaceID) if s.stoppingPlaceID else None
        end = self._end(vid) if not stops else None
        if end is not None:       # (10 Nov) a ring car's planned trip: this many roads, it leaves part way along the last
            edges = edges[:max(1, end[2])]
            dest_edge, dest_pos = edges[-1], end[0]
        est, free, share = self.estimate(edges, pos, dest_pos)
        leg = Leg(vid, now, edges[0], pos, start_name, dest_edge, dest_pos, dest_name, round(est, 1), round(free, 1),
                  round(share, 2), rid, edges)
        leg.last_road = edges[0]
        leg.end_v = end[1] if end is not None else None
        self.active[vid] = self.last[vid] = leg
        self.stats["legs"] += 1

    def _close(self, vid: str, now: float, how: str) -> None:
        leg = self.active.pop(vid, None)
        if leg is None:
            return
        leg.t1, leg.how = now, how
        if how == "parked" and leg.real_s is not None and leg.real_s < MIN_TRIP_S and leg.last_road == leg.start_edge:
            self.stats["dropped"] += 1   # put straight into its space: not a trip
            if self.last.get(vid) is leg:
                self.last.pop(vid, None)
            return
        self.done.append(leg)

    def step(self, c, now: float, arrived=()) -> None:
        """After each SUMO step."""
        try:
            sim = c.simulation.getSubscriptionResults()
        except Exception:  # noqa: BLE001
            sim = {}
        subs = c.vehicle.getAllSubscriptionResults()
        renew = []
        for vid, leg in self.active.items():
            if leg.end_v is not None:   # (11 Nov) a ring car whose trip's end moved on (it was in sight): a new trip
                e = self._end(vid)
                if e is not None and e[1] != leg.end_v:
                    leg.changes.append((round(now - leg.t0, 1), "new destination"))
                    leg.dest_changed = True
                    renew.append(vid)
                    continue
            r = subs.get(vid)
            if r is None:
                continue
            road = r.get(_ROAD)
            if road and not road.startswith(":"):
                leg.last_road = road
            rid = r.get(_ROUTE_ID)
            if rid and leg.route_id and rid != leg.route_id:
                leg.route_id = rid
                try:
                    route = c.vehicle.getRoute(vid)
                except Exception:  # noqa: BLE001
                    route = ()
                still = leg.dest_edge in route
                if leg.end_v is not None:   # a ring car: its trip's end moved (one more lap, round again)?
                    end = self._end(vid)
                    still = still and end is not None and end[1] == leg.end_v
                leg.changes.append((round(now - leg.t0, 1), "new way" if still else "new destination"))
                if not still:   # another destination: this trip ends here, a new one starts with its own estimate
                    leg.dest_changed = True
                    renew.append(vid)
        for vid in renew:
            self._close(vid, now, "extended")
            self._open(c, vid, now, None)
            nxt = self.active.get(vid)
            if nxt is not None:
                nxt.changes.append((0.0, "new destination given on the way"))
        for vid in arrived:
            leg = self.active.get(vid)
            if leg is not None:
                self._close(vid, now, "arrived" if leg.last_road == leg.dest_edge else "gone")
            self.at_area.pop(vid, None)
            self.stopped.discard(vid)
        for vid in sim.get(_PARK_START, ()):
            try:
                st = c.vehicle.getStops(vid, 1)
                if st and st[0].stoppingPlaceID:
                    self.at_area[vid] = st[0].stoppingPlaceID
            except Exception:  # noqa: BLE001, S110 - gone meanwhile
                pass
            if vid in self.active:
                self._close(vid, now, "parked")
        for vid in sim.get(_STOP_START, ()):   # a stop on its way's end that is not a parking (police at a scene)
            leg = self.active.get(vid)
            if leg is not None and leg.last_road == leg.dest_edge:
                self._close(vid, now, "stopped")
                self.stopped.add(vid)
        for vid in sim.get(_PARK_END, ()):
            aid = self.at_area.pop(vid, None)
            self._open(c, vid, now, self.name(vid, aid) if aid else None)
        for vid in sim.get(_STOP_END, ()):
            if vid in self.stopped and vid not in self.active:
                self.stopped.discard(vid)
                self._open(c, vid, now)
        for vid in sim.get(_DEPARTED, ()):
            if vid not in self.active:
                self._open(c, vid, now)

    def finish(self, now: float) -> None:
        """At the end of a run: the trips still on their way are left out of the accuracy (no end)."""
        self.active.clear()

    def snapshot(self, vid: str, now: float) -> dict | None:
        leg = self.last.get(vid)
        if leg is None:
            return None
        return {"t0": leg.t0, "elapsed_s": (leg.t1 if leg.t1 is not None else now) - leg.t0, "est_s": leg.est_s,
                "free_s": leg.free_s, "table_share": leg.table_share, "start": leg.start_name, "dest": leg.dest_name,
                "dest_edge": leg.dest_edge, "changes": list(leg.changes), "dest_changed": leg.dest_changed,
                "ended": leg.how, "real_s": leg.real_s,
                "live": self.live.get(vid) if leg.how is None else None}


def plan_names(demand):
    """(10 Nov) place_name for TripEstimates: the place of the car's trip plan (cosim/purposes.PurposeDemand) that parks
    at that area: where it goes, its home, where it came from (a shared car park was named after its main place: a car
    to Brasserie du Nord was said to go to Hôpital)."""
    def name(vid: str, area: str) -> str | None:
        p = demand.plans.get(vid)
        if p is None:
            return None
        for pid in (p.dest, p.home, p.origin):
            if pid and area in demand.places.park_of.get(pid, []):
                return demand.places.places.get(pid, {}).get("name") or pid
        return None
    return name


def area_names(places) -> dict:
    """Parking area -> the name of the place it serves (the destination with the most jobs and visits; a home's name
    if only homes park there). places: cosim.purposes.Places."""
    best: dict[str, tuple[float, str]] = {}
    for pid, p in places.places.items():
        w = (p.get("jobs", 0) or 0) + (p.get("visits", 0) or 0) + (0.5 if p.get("homes") else 0.0)
        for aid in places.park_of.get(pid, []):
            if aid not in best or w > best[aid][0]:
                best[aid] = (w, p.get("name") or pid)
    return {aid: name for aid, (_, name) in best.items()}

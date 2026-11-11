"""Traffic on Town05's ring road (the "highway"), on top of the town's trips (24 Oct).

The ring is whole in SUMO: netconvert's --geometry.remove (cosim/town.py, netconvert_carla) joined OpenDRIVE roads
38, 12, 34 and 35 into one edge per direction, and 36 and 37 into another, so it is two loops of two long edges
(35 -> 37, 1.4 km; -38 -> -36, 1.5 km), three lanes each, meeting the town at two signalised junctions (2014, 2328).
It looked empty for two reasons:
* its limit was the town's 50 km/h (netconvert's type file gives every "driving" lane 13.89 m/s);
* the town's random trips (cosim/town.py, randomTrips) rarely use it: ~550 an hour over 2.9 km of road, a handful of
  cars on the whole ring at a time.
Here it gets HIGHWAY_SPEED, and cars of its own that drive round it (PEAK_PER_LOOP per loop at the evening peak,
following the day's curve), started and taken off where the camera does not look. They never leave the ring; at
its two junctions their greens are kept short (RING_GREEN_MAX_S, cap_greens), as the town's roads into them are
short. Measured (tools/check_highway.py, evening peak, the junction agents): ~40-65 cars on the ring (20-30 a loop)
instead of ~10, 44-55 km/h on average there (signals included), no lock-up; the town's share of cars standing
0.54-0.64 instead of 0.47-0.50.
"""

from __future__ import annotations

import logging
import random

log = logging.getLogger("highway")

HIGHWAY_SPEED = 22.22  # m/s (80 km/h): CARLA's own type for ramps; a ring with two signals, not a motorway
PEAK_PER_LOOP = 20     # cars on each loop at the evening peak (about 14 per km and direction). The town's share of
                       # cars standing hardly depends on it once the ring's greens are short (16: 0.54-0.61, 30:
                       # 0.54-0.64, 45: 0.51-0.57); 45 made queues on the ring itself (31-34 km/h on average). The
                       # viewer draws them all: on the PC 30 a loop cost 4-6 FPS (18-26 against 26-29 with 16)
MIN_PER_LOOP = 3       # at night
LAPS = 200             # a ring car's route: round and round (taken off out of sight when there are too many)
TRIP_ROADS = (2, 6)    # (10 Nov) a ring car's planned trip: this many of the ring's roads after the one it starts on
                       # (each 590-859 m: 1-3 laps), so it has a start, an end and a trip-time estimate like the town's
END_POS = (150.0, 440.0)   # where on its last road it leaves (within every ring road): taken off there out of sight,
                           # else its end one lap on
PREFIX = "hw"
RING_GREEN_MAX_S = 10.0  # a green for the ring alone at its junctions ends after this at most (below)
MIN_LEN_M = 400.0      # a ring edge: this long at least ...
MIN_LANES = 3          # ... with this many lanes for cars


def router_knows_ring(router, loops, speed: float = HIGHWAY_SPEED) -> int:
    """(10 Nov) A route search (agents/route_recommendation's Router: the town's trips, cosim/purposes.py) that knows
    the ring's limit (set in SUMO by HighwayTraffic.setup; the network file has the town's 50 km/h), so trips use the
    ring where it is quicker, as drivers would. Returns the roads changed."""
    n = 0
    for lp in loops:
        for e in lp:
            r = router.roads.get(e)
            if r is not None:
                r.speed = speed
                n += 1
    return n


def find_loops(net) -> list[list[str]]:
    """The ring's loops: chains of long multi-lane edges (MIN_LEN_M, MIN_LANES lanes for cars) that lead straight
    on into each other and back to the first. sumolib net. Town05: [['35', '37'], ['-36', '-38']]."""
    def cars(e) -> int:
        return sum(1 for ln in e.getLanes() if ln.allows("passenger"))

    big = {e.getID(): e for e in net.getEdges() if e.getFunction() != "internal" and e.getLength() >= MIN_LEN_M
           and cars(e) >= MIN_LANES}
    loops, seen = [], set()
    for eid in sorted(big):
        if eid in seen:
            continue
        chain, cur = [eid], big[eid]
        while True:
            nxt = [to.getID() for to, conns in cur.getOutgoing().items()
                   if to.getID() in big and any(c.getDirection() == "s" for c in conns)]
            if not nxt:
                break
            if nxt[0] == chain[0]:
                loops.append(chain)
                seen.update(chain)
                break
            if nxt[0] in chain:
                break
            chain.append(nxt[0])
            cur = big[nxt[0]]
    return loops


class HighwayTraffic:
    """Keeps cars driving round each loop: target(seconds_of_day) of them on each (the town's curve scaled to
    PEAK_PER_LOOP at its top). conn: TraCI; loops: find_loops(); share(seconds_of_day): the town's vehicle target
    as a share of its peak; free(edge, pos): a car may appear / vanish there (out of sight); types: [(type, weight)];
    type_for(vid, type): the type it is inserted with (cosim/vmodels.Typer)."""

    def __init__(self, conn, loops, share, free=None, types=(("passenger", 1.0),), type_for=None, seed: int = 5,
                 peak: int = PEAK_PER_LOOP, speed: float = HIGHWAY_SPEED, lengths: dict | None = None,
                 green_max: float | None = RING_GREEN_MAX_S, policy=None, planned: bool = True):
        self.conn = conn
        self.loops = [list(lp) for lp in loops]
        self.share = share
        self.free = free or (lambda edge, pos: True)
        self.types = [t for t, _ in types]
        self.weights = [w for _, w in types]
        self.type_for = type_for
        self.rng = random.Random(seed)
        self.peak = peak
        self.speed = speed
        self.lengths = lengths or {}
        self.green_max = green_max
        self.capped: dict[str, list] = {}
        self.policy = policy  # the junction agents (cosim/agents_actuated.py) keep their own copy of the greens' limits
        self.cars: list[set] = [set() for _ in self.loops]
        self.n = 0
        self._k = 0
        self.rerouted = 0
        self.added = self.removed = 0
        self.routes: dict[tuple, str] = {}
        # (10 Nov) planned trips: vid -> {"k", "edges" (its route), "end" (m along its last road)}; False: 200 laps
        self.planned = planned
        self.plan: dict[str, dict] = {}
        self.extended = 0      # one more lap: its trip's end was in sight
        self.arrived = 0       # planned trips ended (taken off at their end, out of sight)
        self.ended: list[str] = []   # those, since take_ended was last asked
        self.near_end: set[str] = set()   # on the last two roads of their trip: their end checked every step (check_ends)
        self.last_check: dict[str, tuple] = {}   # vid -> (road, pos, end, out of sight) of its last check at its end
        self.route_ids: dict[str, str] = {}      # vid -> its route id (check_ends: changed by someone else?)
        self.round_again: set[str] = set()       # re-routed short, not yet back on its laps (on a junction)
        self.protected = None  # vid -> True: never taken off (an accident's: cosim/incidents.IncidentManager.protected)

    def setup(self) -> None:
        c = self.conn
        for lp in self.loops:
            for e in lp:
                try:
                    c.edge.setMaxSpeed(e, self.speed)
                    if e not in self.lengths:
                        self.lengths[e] = c.lane.getLength(f"{e}_0")
                except Exception as e_:  # noqa: BLE001
                    log.info("highway edge %s: %s", e, e_)
        if self.green_max:
            self.cap_greens(self.green_max)
        log.info("highway: %d loops %s, %.0f km/h, its greens capped %s", len(self.loops), self.loops,
                 self.speed * 3.6, self.capped)

    def cap_greens(self, max_s: float) -> None:
        """At the junctions where the ring meets the town, a green for the ring alone ends after max_s at most. With
        the ring's own cars those greens are never empty, so the junction agents (or the actuated program) could no
        longer skip them, and the town's roads into the ring junctions (14-21 m, two or three cars) waited a whole
        cycle: their queues filled the junctions behind them (2296, then 1292). The town's share of cars standing at
        the evening peak went from 0.49-0.51 to 0.52-0.68 with greens up to 60 s, to 0.52-0.61 with 10 s, and no
        lock-up (tools/check_highway.py, seeds 11, 23, 31, 41, 53, the junction agents). SUMO's current program of
        each junction, changed in place, and the agents' copy of its limits.
        """
        c = self.conn
        ring = {e for lp in self.loops for e in lp}
        tls_ids = set()
        try:
            all_tls = list(c.trafficlight.getIDList())
        except Exception:  # noqa: BLE001 - no signals (or a test's stand-in)
            all_tls = []
        for t in all_tls:
            try:
                lanes = {ln.rsplit("_", 1)[0] for ln in c.trafficlight.getControlledLanes(t)}
            except Exception:  # noqa: BLE001
                continue
            if lanes & ring:
                tls_ids.add(t)
        for t in sorted(tls_ids):
            try:
                links = c.trafficlight.getControlledLinks(t)
                prog = c.trafficlight.getProgram(t)
                logic = next(lg for lg in c.trafficlight.getAllProgramLogics(t) if lg.programID == prog)
            except Exception as e:  # noqa: BLE001
                log.info("ring junction %s: %s", t, e)
                continue
            is_ring = [bool(lk) and lk[0][0].rsplit("_", 1)[0] in ring for lk in links]
            done = []
            for k, ph in enumerate(logic.phases):
                green = [i for i, ch in enumerate(ph.state) if ch in "Gg" and i < len(is_ring)]
                if green and all(is_ring[i] for i in green) and ph.maxDur > max_s:
                    ph.maxDur = max_s
                    ph.minDur = min(ph.minDur, max_s)
                    ph.duration = min(ph.duration, max_s)
                    done.append(k)
            if not done:
                continue
            try:
                c.trafficlight.setProgramLogic(t, logic)
                self.capped[t] = done
            except Exception as e:  # noqa: BLE001
                log.info("ring junction %s: %s", t, e)
                continue
            agent = getattr(self.policy, "junctions", {}).get(t) if self.policy is not None else None
            if agent is not None and hasattr(agent, "max_dur"):
                for k in done:
                    if k < len(agent.max_dur):
                        agent.max_dur[k] = min(agent.max_dur[k], max_s)
                        agent.min_dur[k] = min(agent.min_dur[k], max_s)

    def target(self, seconds_of_day: float) -> int:
        if self.peak <= 0:   # (10 Nov) no cars of its own: the ring carries the town's trips (router_knows_ring)
            return 0
        return max(MIN_PER_LOOP, round(self.peak * min(max(self.share(seconds_of_day), 0.0), 1.0)))

    def is_ring_car(self, vid: str) -> bool:
        return vid.startswith(PREFIX)

    def _route(self, k: int, start: int) -> str:
        key = (k, start)
        rid = self.routes.get(key)
        if rid is None:
            lp = self.loops[k]
            edges = (lp[start:] + lp[:start]) * LAPS
            rid = f"{PREFIX}_r{k}_{start}"
            self.conn.route.add(rid, edges)
            self.routes[key] = rid
        return rid

    def _add(self, k: int) -> bool:
        lp = self.loops[k]
        for _ in range(6):
            i = self.rng.randrange(len(lp))
            e = lp[i]
            pos = self.rng.uniform(10.0, max(11.0, self.lengths.get(e, 100.0) - 30.0))
            if self.free(e, pos):
                break
        else:
            return False
        vid = f"{PREFIX}{self.n}"
        self.n += 1
        t = self.rng.choices(self.types, self.weights)[0]
        tid = self.type_for(vid, t) if self.type_for else t
        try:
            self.conn.vehicle.add(vid, self._route(k, i), typeID=tid, depart="now", departLane="free",
                                  departPos=f"{pos:.1f}", departSpeed="max")
            if self.planned:   # its trip: this many roads (the one it appears on first), its end part way along the last
                self.plan[vid] = {"k": k, "n": 1 + self.rng.randint(*TRIP_ROADS), "passed": 0, "last": e,
                                  "end": round(self.rng.uniform(*END_POS), 1), "v": 0}
        except Exception as e_:  # noqa: BLE001
            log.debug("ring car %s: %s", vid, e_)
            return False
        self.cars[k].add(vid)
        self.added += 1
        return True

    def end_of(self, vid: str) -> tuple[float, int, int] | None:
        """A ring car's planned trip end: (m along its last road, how many times it moved, roads left including the one
        it is on); None for others (cosim/trip_estimates.py: a moved end is a new destination)."""
        p = self.plan.get(vid)
        return None if p is None else (p["end"], p["v"], max(1, p["n"] - p["passed"]))

    def _trips(self, k: int, mine: set) -> None:
        """(10 Nov, 11 Nov) Planned trips. SUMO's route stays the 200 laps (a route given again with a loop in it and
        an end position: SUMO 1.27 took the car off at once, in view); the trip is counted here: the roads driven
        (`passed`), and on its last road past its end, the car leaves if it is out of sight (taken off, as the ring's
        surplus always was) or its end moves one lap on (in sight). A car someone else sent another way (an accident's
        detour, a knot undone) goes round again (_keep_round)."""
        c = self.conn
        lp = self.loops[k]
        for vid in list(mine):
            p = self.plan.get(vid)
            if p is None:
                continue
            try:
                road = c.vehicle.getRoadID(vid)
            except Exception:  # noqa: BLE001 - gone
                continue
            if not road or road.startswith(":") or road not in lp:
                continue
            if road != p["last"]:
                p["passed"] += 1
                p["last"] = road
            if p["passed"] >= p["n"] - 1:
                self.near_end.add(vid)
                self._at_end(vid, p, road)

    def _at_end(self, vid: str, p: dict, road: str | None = None) -> None:
        c = self.conn
        try:
            if road is None:
                road = c.vehicle.getRoadID(vid)
                if not road or road.startswith(":"):
                    return
                if road != p["last"] and road in self.loops[p["k"]]:
                    p["passed"] += 1
                    p["last"] = road
            pos = c.vehicle.getLanePosition(vid)
        except Exception:  # noqa: BLE001 - gone
            return
        if p["passed"] < p["n"] - 1 or pos < p["end"]:
            return
        out = self.free(road, pos)
        self.last_check[vid] = (road, round(pos, 1), p["end"], out)
        if out:
            try:
                c.vehicle.remove(vid)
                self.arrived += 1
                self.ended.append(vid)   # (SUMO does not list a car taken off as arrived: take_ended)
            except Exception as e:  # noqa: BLE001
                log.debug("ring car %s at its trip's end: %s", vid, e)
                return
            self.cars[p["k"]].discard(vid)
            self.plan.pop(vid, None)
            self.near_end.discard(vid)
        else:                       # in sight: its end one lap further
            p["n"] += len(self.loops[p["k"]])
            p["v"] += 1
            self.extended += 1
            self.near_end.discard(vid)

    def take_ended(self) -> list[str]:
        """The ring cars whose trip ended (taken off at its end) since the last call: arrived, for the trip estimates
        (SUMO lists a car taken off by TraCI in no arrived list, whatever the reason given)."""
        out, self.ended = self.ended, []
        return out

    def check_ends(self, subs: dict | None = None) -> None:
        """(11 Nov) Every SUMO step in the viewer: the cars on the last road of their trip, at their end? (the ring's
        step comes every 2 s: 44 m at 80 km/h). And a ring car whose route someone changed (subs: the step's vehicle
        subscriptions, their route id) goes round again at once: an accident on the ring re-routes the cars heading
        for it to the last road of their route, a ring road close by, and they ended there (7 vanished in view within
        2 s at the evening peak; _keep_round looks every 10 s)."""
        if subs:
            for k, mine in enumerate(self.cars):
                for vid in mine:
                    rid = (subs.get(vid) or {}).get(0x53)   # VAR_ROUTE_ID
                    if rid is None:
                        continue
                    if self.route_ids.get(vid) is None:
                        self.route_ids[vid] = rid
                    elif rid != self.route_ids[vid] or vid in self.round_again:
                        self.route_ids[vid] = rid
                        self._keep_round(k, {vid})
                        try:   # still short (on a junction: setRoute refused): again at the next step
                            route, idx = self.conn.vehicle.getRoute(vid), self.conn.vehicle.getRouteIndex(vid)
                            if len(route) - idx > 2 * len(self.loops[k]) + 2:
                                self.round_again.discard(vid)
                            else:
                                self.round_again.add(vid)
                            self.route_ids[vid] = self.conn.vehicle.getRouteID(vid)
                        except Exception:  # noqa: BLE001 - gone
                            self.round_again.discard(vid)
        for vid in list(self.near_end):
            p = self.plan.get(vid)
            if p is None:
                self.near_end.discard(vid)
                continue
            self._at_end(vid, p)

    def _keep_round(self, k: int, mine: set) -> None:
        """A ring car given a new way (an accident's detour, a knot undone in sight: cosim/incidents.py routes to the
        last edge of its route) went the short way there and ended its trip on the ring, in sight (4-5 a run on the
        PC, 24 Oct): its route goes round again."""
        c = self.conn
        lp = self.loops[k]
        for vid in list(mine):
            try:
                route = c.vehicle.getRoute(vid)
                if len(route) - c.vehicle.getRouteIndex(vid) > 2 * len(lp) + 2:
                    continue
                road = c.vehicle.getRoadID(vid)
            except Exception:  # noqa: BLE001
                continue
            if road not in lp:
                continue  # in a junction (again in a moment), or sent off the ring: its own way
            i = lp.index(road)
            try:
                c.vehicle.setRoute(vid, (lp[i:] + lp[:i]) * LAPS)
                self.rerouted += 1
            except Exception as e:  # noqa: BLE001
                log.debug("ring car %s round again: %s", vid, e)

    def step(self, running: dict, seconds_of_day: float, pending: set | None = None) -> None:
        """Every couple of seconds. running: SUMO id -> Vehicle (x, y SUMO); pending: ids SUMO has not put on the
        road yet (none of ours may wait to appear: a car waiting for room would pop up later, maybe in sight)."""
        c = self.conn
        want = self.target(seconds_of_day)
        self._k += 1
        for k in range(len(self.loops)):
            mine = self.cars[k]
            for vid in list(mine):
                if vid in running:
                    continue
                if pending is not None and vid in pending:
                    try:
                        c.vehicle.remove(vid)
                    except Exception:  # noqa: BLE001
                        pass
                mine.discard(vid)
                self.plan.pop(vid, None)
                self.near_end.discard(vid)
                self.route_ids.pop(vid, None)
                self.round_again.discard(vid)
            if self.planned:
                self._trips(k, mine)
            if self._k % 5 == 0:
                self._keep_round(k, mine)
            n = len(mine)
            if n < want:
                for _ in range(min(2, want - n)):
                    self._add(k)
            elif n > want + 1:
                for vid in sorted(mine):
                    v = running.get(vid)
                    if v is None or (self.protected is not None and self.protected(vid)):
                        continue
                    try:
                        e, p = c.vehicle.getRoadID(vid), c.vehicle.getLanePosition(vid)
                    except Exception:  # noqa: BLE001
                        continue
                    if self.free(e, p):
                        try:
                            c.vehicle.remove(vid)
                            mine.discard(vid)
                            self.removed += 1
                        except Exception:  # noqa: BLE001
                            pass
                        break

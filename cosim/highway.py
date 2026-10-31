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
PREFIX = "hw"
RING_GREEN_MAX_S = 10.0  # a green for the ring alone at its junctions ends after this at most (below)
MIN_LEN_M = 400.0      # a ring edge: this long at least ...
MIN_LANES = 3          # ... with this many lanes for cars


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
                 green_max: float | None = RING_GREEN_MAX_S, policy=None):
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
        try:
            self.conn.vehicle.add(vid, self._route(k, i), typeID=self.type_for(vid, t) if self.type_for else t,
                                  depart="now", departLane="free", departPos=f"{pos:.1f}", departSpeed="max")
        except Exception as e_:  # noqa: BLE001
            log.debug("ring car %s: %s", vid, e_)
            return False
        self.cars[k].add(vid)
        self.added += 1
        return True

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

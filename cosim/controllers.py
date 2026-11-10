"""Signal controllers that act through TraCI every step, plus the interface they share.

The interface is the main project's (`utm.baselines.base.Controller`): ``name``,
``additional_files()``, ``setup(conn)``, ``step(conn)``, where ``conn.tc`` is the TraCI (or
libsumo) connection and ``step`` is called once after every simulation step. A controller
written here runs unchanged in the main project's evaluation harness (tools/eval_main_harness.py
uses exactly that), and the main project's future RL/MARL policy runs unchanged in the CARLA
renderer (cosim/sumo_side.py).

MaxPressure (Varaiya, "Max pressure control of a network of signalized intersections",
Transportation Research Part C 36, 2013): at each junction, after the minimum green, switch
to the phase whose movements have the largest pressure, the sum over its green movements
of (vehicles on the incoming lane - vehicles on the outgoing lane). Decentralised: each
junction only reads its own lanes; it needs no training and no demand estimate.

Safety does not depend on the policy (the plan's Phase 4 rule, kept here):
* only green states that the network's own program contains are ever shown, so no
  conflicting movements can be green together;
* every movement losing its green gets a yellow of the program's yellow time, then the
  program's all-red time (0 where the program has none);
* a permissive turn (g) loses its green only in transitions the program itself makes, so
  turners waiting inside the junction are always released as the program intends;
* minimum green before any switch; maximum green after which the best *other* phase is
  served; and a maximum red after which a phase with vehicles waiting is served, so no
  approach can starve.
tests/test_controllers.py replays every state a run shows and checks all three.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import traci.constants as _tc

GREEN = "Gg"


def greens(state: str) -> frozenset[int]:
    return frozenset(i for i, c in enumerate(state) if c in GREEN)


@dataclass
class _Junction:
    tls: str
    phases: list[str]                     # the program's distinct green states
    pairs: list[list[tuple[str, str]]]    # per phase: the (in lane, out lane) pairs it serves
    yellow_s: float
    all_red_s: float
    current: int = 0
    since: float = 0.0
    mode: str = "green"                   # green | yellow | red
    target: int = 0
    served: dict[int, float] = field(default_factory=dict)  # phase -> last time its green ended or began
    shown: str = ""
    switches: int = 0
    forced: int = 0                       # switches forced by max green
    log: list[tuple[float, str]] = field(default_factory=list)
    program_moves: set[tuple[int, int]] = field(default_factory=set)  # (from, to) the program itself makes
    moves: list[tuple[int, int, float, float]] = field(default_factory=list)  # (from, to, yellow at, green at)
    _yellow_at: float = 0.0


def transition_state(cur: str, nxt: str) -> str:
    """Yellow for every movement green now and not green next; green kept where it stays."""
    out = []
    for a, b in zip(cur, nxt, strict=True):
        if a in GREEN and b not in GREEN:
            out.append("y")
        elif a in GREEN:
            out.append(a)
        else:
            out.append("r")
    return "".join(out)


def all_red_state(cur: str, nxt: str) -> str:
    """Red for every movement changing; movements green in both phases keep their green."""
    return "".join(a if (a in GREEN and b in GREEN) else "r" for a, b in zip(cur, nxt, strict=True))


def program_moves(raw: list[tuple[str, float]], phases: list[str]) -> set[tuple[int, int]]:
    """(from, to) pairs of green phases that follow each other in the program's cycle."""
    seq = [phases.index(s) for s, _ in raw if s in phases]
    return {(a, b) for a, b in zip(seq, seq[1:] + seq[:1], strict=True) if a != b}


def program_timing(phases: list[tuple[str, float]]) -> tuple[float, float]:
    """(yellow, all-red) seconds as the program uses them: the longest yellow phase, and the
    longest phase that is all red (0 if none)."""
    yellow = max((d for s, d in phases if "y" in s.lower()), default=3.0)
    all_red = max((d for s, d in phases if s and set(s.lower()) <= {"r"}), default=0.0)
    return yellow, all_red


class MaxPressureController:
    name = "max_pressure"

    def __init__(self, min_green: float = 5.0, max_green: float = 60.0, max_red: float = 60.0,
                 count: str = "vehicles", reach_m: float = 75.0, record: bool = False):
        self.min_green = min_green
        self.max_green = max_green
        self.max_red = max_red
        self.count = count  # "vehicles" on a lane, or "halting" (queued: speed < 0.1 m/s)
        # Lanes are counted over this distance from the junction, crossing into the lanes
        # before (incoming) or after (outgoing) when a lane is shorter. Counting the link's
        # own lane only missed a queue standing before a 1-2 m lane at Ingolstadt7's gneJ143:
        # that approach was never served (vehicles waited 1787 s, 163 could not enter).
        self.reach_m = reach_m
        self._up: dict[str, tuple[str, ...]] = {}
        self._down: dict[str, tuple[str, ...]] = {}
        self.record = record
        self.junctions: dict[str, _Junction] = {}

    def additional_files(self) -> list:
        return []

    def setup(self, conn) -> None:
        tc = conn.tc
        now = tc.simulation.getTime()
        self._graph(tc)
        for tls in tc.trafficlight.getIDList():
            prog = tc.trafficlight.getProgram(tls)
            logic = next(lg for lg in tc.trafficlight.getAllProgramLogics(tls) if lg.programID == prog)
            raw = [(p.state, float(p.duration)) for p in logic.phases]
            phases: list[str] = []
            for s, _ in raw:
                if "y" not in s.lower() and greens(s) and s not in phases:
                    phases.append(s)
            links = tc.trafficlight.getControlledLinks(tls)
            pairs = []
            for s in phases:
                seen = []
                for i in greens(s):
                    for link in links[i] if i < len(links) else []:
                        p = (link[0], link[1])
                        if p not in seen:
                            seen.append(p)
                pairs.append(seen)
            yellow, all_red = program_timing(raw)
            j = _Junction(tls, phases, pairs, yellow, all_red, since=now, program_moves=program_moves(raw, phases))
            self.junctions[tls] = j
            self._show(tc, j, phases[0], now)

    def _graph(self, tc) -> None:
        """Lane successors/predecessors (via internal lanes) and lengths, for `_reach`."""
        self._len: dict[str, float] = {}
        self._succ: dict[str, list[str]] = {}
        self._pred: dict[str, list[str]] = {}
        for lane in tc.lane.getIDList():
            self._len[lane] = tc.lane.getLength(lane)
            for link in tc.lane.getLinks(lane):
                nxt, via = link[0], link[4] if len(link) > 4 else ""
                chain = [lane, via, nxt] if via else [lane, nxt]
                for a, b in zip(chain, chain[1:], strict=False):
                    self._succ.setdefault(a, []).append(b)
                    self._pred.setdefault(b, []).append(a)

    def _reach(self, lane: str, upstream: bool) -> tuple[str, ...]:
        """The lane plus those before it (upstream) or after it, until reach_m is covered."""
        cache = self._up if upstream else self._down
        if lane in cache:
            return cache[lane]
        nbrs = self._pred if upstream else self._succ
        out, frontier, seen = [lane], [(lane, self._len.get(lane, 0.0))], {lane}
        while frontier:
            cur, covered = frontier.pop()
            if covered >= self.reach_m:
                continue
            for n in nbrs.get(cur, []):
                if n not in seen:
                    seen.add(n)
                    out.append(n)
                    frontier.append((n, covered + self._len.get(n, 0.0)))
        cache[lane] = tuple(out)
        return cache[lane]

    def _show(self, tc, j: _Junction, state: str, now: float) -> None:
        if state != j.shown:
            tc.trafficlight.setRedYellowGreenState(j.tls, state)
            j.shown = state
            if self.record:
                j.log.append((now, state))

    def _n(self, tc, lane: str, counts: dict[str, int]) -> int:
        if lane not in counts:
            counts[lane] = (tc.lane.getLastStepHaltingNumber(lane) if self.count == "halting"
                            else tc.lane.getLastStepVehicleNumber(lane))
        return counts[lane]

    def queue_in(self, tc, lane: str, counts: dict[str, int]) -> int:
        return sum(self._n(tc, x, counts) for x in self._reach(lane, upstream=True))

    def queue_out(self, tc, lane: str, counts: dict[str, int]) -> int:
        return sum(self._n(tc, x, counts) for x in self._reach(lane, upstream=False))

    def pressure(self, tc, j: _Junction, k: int, counts: dict[str, int]) -> int:
        return sum(self.queue_in(tc, a, counts) - self.queue_out(tc, b, counts) for a, b in j.pairs[k])

    def allowed(self, j: _Junction, i: int) -> list[int]:
        """Phases reachable from phase i: any, except that a permissive movement (g: turns
        that wait inside the junction for a gap) may only lose its green in a transition the
        program itself makes.

        A fixed program follows a permissive phase ("GGGgg") with the phase that lets the
        turners waiting inside the junction leave (the protected turn). Jumping from the
        permissive phase straight to a crossing one left them trapped: measured on Cologne1
        (seed 2), vehicles stood on the junction's internal lanes for over 1600 s, a deadlock
        SUMO cannot resolve without teleporting. (An extra 3 s all-red on such jumps instead
        removed the deadlock but cost so much green that Cologne3 gridlocked.)"""
        cur = j.phases[i]
        out = []
        for k, nxt in enumerate(j.phases):
            if k == i:
                continue
            cuts = any(a == "g" and b not in GREEN for a, b in zip(cur, nxt, strict=True))
            if not cuts or (i, k) in j.program_moves:
                out.append(k)
        return out

    def demand(self, tc, j: _Junction, k: int, counts: dict[str, int]) -> int:
        return sum(self.queue_in(tc, a, counts) for a in {a for a, _ in j.pairs[k]})

    def step(self, conn) -> None:
        tc = conn.tc
        now = tc.simulation.getTime()
        counts: dict[str, int] = {}
        for j in self.junctions.values():
            self._decide(tc, j, now, counts)

    def _decide(self, tc, j: _Junction, now: float, counts: dict[str, int]) -> None:
        """One junction, one step: finish a transition, or keep or change the green."""
        held = now - j.since
        if j.mode == "yellow":
            if held >= j.yellow_s:
                if j.all_red_s > 0:
                    j.mode, j.since = "red", now
                    self._show(tc, j, all_red_state(j.phases[j.current], j.phases[j.target]), now)
                else:
                    self._start_green(tc, j, now)
            return
        if j.mode == "red":
            if held >= j.all_red_s:
                self._start_green(tc, j, now)
            return
        if held < self.min_green or len(j.phases) < 2:
            return
        p = [self.pressure(tc, j, k, counts) for k in range(len(j.phases))]
        # Fairness: plain max-pressure can leave a phase red indefinitely while its
        # outgoing lanes look fuller than the others' (measured on Town10: a car waited
        # 156 s at a red; queues then spilled back round a block into gridlock). A phase
        # with someone waiting that has been red for max_red is served next. Measured
        # on Town10 (1 h, seed 42): max_red 90 s still gridlocked (446 trips completed),
        # 60 s did not (1078); on Ingolstadt7 and Cologne3 both were gridlock-free.
        ok = self.allowed(j, j.current)
        starved = [k for k in range(len(p)) if k != j.current
                   and now - j.served.get(k, j.since) >= self.max_red and self.demand(tc, j, k, counts) > 0]
        if starved:
            most = max(starved, key=lambda k: now - j.served.get(k, j.since))
            # serve it, or if it cannot be reached directly, move on along the program
            successors = [b for a, b in sorted(j.program_moves) if a == j.current]
            best = most if most in ok else (successors[0] if successors else max(ok, key=lambda k: p[k]))
            j.forced += 1
        elif held >= self.max_green:
            # Serve the best other phase, if anyone is waiting for one.
            waiting = [k for k in ok if self.demand(tc, j, k, counts) > 0]
            if not waiting:
                return
            best = max(waiting, key=lambda k: p[k])
            j.forced += 1
        else:
            best = max(ok + [j.current], key=lambda k: (p[k], k == j.current))
            if best == j.current or p[best] <= p[j.current]:
                return
        self._switch(tc, j, best, now)

    def _switch(self, tc, j: _Junction, best: int, now: float) -> None:
        """Leave the current green for phase `best`: yellow first (then all-red if the
        program has one), as the program's own transitions do."""
        j.served[j.current] = now
        j.target, j.mode, j.since, j._yellow_at = best, "yellow", now, now
        j.switches += 1
        self._show(tc, j, transition_state(j.phases[j.current], j.phases[best]), now)

    def _start_green(self, tc, j: _Junction, now: float) -> None:
        if self.record:
            j.moves.append((j.current, j.target, j._yellow_at, now))
        j.current, j.mode, j.since = j.target, "green", now
        j.served[j.current] = now
        self._show(tc, j, j.phases[j.current], now)


class ObservedMaxPressure(MaxPressureController):
    """Max-pressure on counts given from outside (`observed`: lane -> vehicles), e.g. from
    cameras (cosim/perception.py). A lane missing from `observed` counts as empty. Everything
    else (minimum/maximum green, fairness, the safety rules) is the parent's."""

    name = "max_pressure_observed"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.observed: dict[str, int] = {}

    def counted_lanes(self) -> list[str]:
        """Every lane the pressures read: incoming lanes and those upstream of them, outgoing
        lanes and those downstream, within reach_m."""
        out = set()
        for j in self.junctions.values():
            for pairs in j.pairs:
                for a, b in pairs:
                    out.update(self._reach(a, upstream=True))
                    out.update(self._reach(b, upstream=False))
        return sorted(out)

    def _n(self, tc, lane: str, counts: dict[str, int]) -> int:
        return self.observed.get(lane, 0)


class PriorityMaxPressure(MaxPressureController):
    """Max-pressure that gives way to emergency vehicles on a run (signal pre-emption).

    `emergency()` returns the ids of the vehicles that may ask (set by the caller: in the
    simulation, the extras' ground truth of who is on a run; in a city, the vehicle's radio
    or GPS request, which is how pre-emption systems are usually triggered). A vehicle asks
    the next signal on its route once it is within `detect_m` of it. The signal then keeps
    a green that serves the vehicle's link, or changes to one as soon as the minimum green
    and the program's own transition rules allow (yellow, all-red, permissive order: the
    same safety rules as every other change). When the vehicle has passed, ordinary
    max-pressure resumes."""

    name = "max_pressure_priority"

    def __init__(self, detect_m: float = 150.0, stall_s: float = 5.0, emergency=None, **kw):
        super().__init__(**kw)
        self.detect_m = detect_m
        self.stall_s = stall_s
        self.emergency = emergency or (lambda: ())
        self.requests: dict[str, set[int]] = {}
        self._stopped_on_green: dict[str, float] = {}
        self.preemptions = 0
        self.held = 0
        self.dropped = 0

    def step(self, conn) -> None:
        tc = conn.tc
        now = tc.simulation.getTime()
        self.requests = {}
        wanted = set(self.emergency())
        for vid in (wanted & set(tc.vehicle.getIDList())) if wanted else ():
            upcoming = tc.vehicle.getNextTLS(vid)
            if not upcoming:
                continue
            tls, link, dist, state = upcoming[0]  # only the next signal
            if tls not in self.junctions or dist > self.detect_m:
                continue
            # A vehicle that stands still although its light is green is blocked further on
            # (a full exit lane): holding the green cannot help it and starves the other
            # directions. First version without this rule: an ambulance stood 8 m before a
            # green for 1087 s on Ingolstadt7 and the network jammed (seed 1, 14 Oct).
            if state in "Gg" and tc.vehicle.getSpeed(vid) < 0.5:
                since = self._stopped_on_green.setdefault(vid, now)
                if now - since >= self.stall_s:
                    self.dropped += 1
                    continue
            else:
                self._stopped_on_green.pop(vid, None)
            self.requests.setdefault(tls, set()).add(int(link))
        super().step(conn)

    def _decide(self, tc, j: _Junction, now: float, counts: dict[str, int]) -> None:
        req = self.requests.get(j.tls)
        # no request, mid-transition, or a green already held for max_green: ordinary rules
        # (which then also serve any phase starved for max_red)
        if not req or j.mode != "green" or len(j.phases) < 2 or now - j.since >= self.max_green:
            return super()._decide(tc, j, now, counts)
        cover = [len(req & greens(s)) for s in j.phases]
        if cover[j.current] == max(cover):
            self.held += 1  # already serving it: keep this green
            return
        if now - j.since < self.min_green:
            return
        ok = self.allowed(j, j.current)
        want = [k for k in ok if cover[k] == max(cover)]
        if want:
            best = want[0]
        else:  # not reachable directly: move on along the program towards it
            successors = [b for a, b in sorted(j.program_moves) if a == j.current]
            best = successors[0] if successors else max(ok, key=lambda k: cover[k])
        self.preemptions += 1
        self._switch(tc, j, best, now)


_CROSSING = re.compile(r"^:.+_c\d+$")  # a crossing edge (tools/build_town05_peds.py), e.g. ":1126_c0"


class WalkAware:
    """Mixin for max-pressure on a network with signalised crosswalks (Town05 after
    tools/build_town05_peds.py: an exclusive pedestrian phase, every car red, every crossing
    walk, then 5 s all red while the last ones finish crossing).

    Plain max-pressure counts vehicles only, so the walk phase has no pressure and nobody asks
    for it: pedestrians would wait until the fairness rule happens to serve it. Here a person
    standing on a junction's walking area whose next edge is one of its crossings counts in
    the pressure (and the demand) of the phases that give that crossing a walk, `ped_weight`
    each. The walk then lasts at least `walk_min_s` (the program's 10 s), and leaving it skips
    the car yellow and holds all red for `walk_clear_s` (the program's 5 s clearance). Car
    transitions keep the program's own yellow and all red (3 s on Town05: the 5 s clearance
    is not counted as a car all-red, which plain max-pressure would do)."""

    walk_min_s = 10.0
    walk_clear_s = 5.0
    ped_weight = 1.0

    def setup(self, conn) -> None:
        super().setup(conn)
        tc = conn.tc
        self._waiting: dict[str, int] = {}
        self._cross: dict[str, dict[int, str]] = {}   # tls -> link index -> crossing edge
        self._walks: dict[str, set[int]] = {}         # tls -> phases that are walk-only
        self._car_timing: dict[str, tuple[float, float]] = {}
        for j in self.junctions.values():
            links = tc.trafficlight.getControlledLinks(j.tls)
            cross = {}
            for i, ls in enumerate(links):
                for link in ls:
                    edge = link[1].rsplit("_", 1)[0]
                    if _CROSSING.match(edge):
                        cross[i] = edge
            self._cross[j.tls] = cross
            self._walks[j.tls] = {k for k, s in enumerate(j.phases) if greens(s) and greens(s) <= set(cross)}
            prog = tc.trafficlight.getProgram(j.tls)
            logic = next(lg for lg in tc.trafficlight.getAllProgramLogics(j.tls) if lg.programID == prog)
            states = [p.state for p in logic.phases]
            after_walk = [bool(greens(states[n - 1])) and greens(states[n - 1]) <= set(cross)
                          for n in range(len(states))]
            car_red = [float(p.duration) for n, p in enumerate(logic.phases)
                       if set(p.state.lower()) <= {"r"} and not after_walk[n]]
            j.all_red_s = max(car_red) if car_red else j.all_red_s
            self._car_timing[j.tls] = (j.yellow_s, j.all_red_s)

    def step(self, conn) -> None:
        tc = conn.tc
        subs = tc.person.getAllSubscriptionResults()  # sumo_side subscribes persons with their road
        waiting: dict[str, int] = {}
        for pid in tc.person.getIDList():
            r = subs.get(pid)
            road = r.get(_tc.VAR_ROAD_ID, "") if r else tc.person.getRoadID(pid)
            if not road.startswith(":") or "_w" not in road:
                continue
            try:
                nxt = tc.person.getNextEdge(pid)
            except Exception:  # noqa: BLE001 - gone this step
                continue
            if _CROSSING.match(nxt):
                waiting[nxt] = waiting.get(nxt, 0) + 1
        self._waiting = waiting
        super().step(conn)

    def peds(self, j: _Junction, k: int) -> int:
        cross = self._cross.get(j.tls)
        if not cross:
            return 0
        return sum(self._waiting.get(cross[i], 0) for i in greens(j.phases[k]) if i in cross)

    def pressure(self, tc, j: _Junction, k: int, counts: dict[str, int]) -> float:
        return super().pressure(tc, j, k, counts) + self.ped_weight * self.peds(j, k)

    def demand(self, tc, j: _Junction, k: int, counts: dict[str, int]) -> int:
        return super().demand(tc, j, k, counts) + self.peds(j, k)

    def _decide(self, tc, j: _Junction, now: float, counts: dict[str, int]) -> None:
        if j.mode == "green" and j.current in self._walks.get(j.tls, ()) and now - j.since < self.walk_min_s:
            return  # nobody leaves a walk early, not even for an emergency vehicle
        super()._decide(tc, j, now, counts)

    def _switch(self, tc, j: _Junction, best: int, now: float) -> None:
        if j.current in self._walks.get(j.tls, ()):
            j.yellow_s, j.all_red_s = 0.0, self.walk_clear_s
        else:
            j.yellow_s, j.all_red_s = self._car_timing.get(j.tls, (j.yellow_s, j.all_red_s))
        super()._switch(tc, j, best, now)


class WalkMaxPressure(WalkAware, MaxPressureController):
    name = "max_pressure_walk"


class WalkPriorityMaxPressure(WalkAware, PriorityMaxPressure):
    name = "max_pressure_walk_priority"


class ObservedWalkMaxPressure(WalkAware, ObservedMaxPressure):
    """Walk-aware max-pressure on vehicle counts from cameras (cosim/camera_reports.py)."""

    name = "max_pressure_walk_observed"


class _Conn:
    """What the main project's controllers expect: an object with `.tc`."""

    def __init__(self, tc):
        self.tc = tc


CONTROLLERS = {"max_pressure": MaxPressureController, "maxpressure": MaxPressureController,
               "max_pressure_priority": PriorityMaxPressure, "max_pressure_walk": WalkMaxPressure,
               "max_pressure_walk_priority": WalkPriorityMaxPressure,
               "max_pressure_walk_observed": ObservedWalkMaxPressure}


def make(name: str):
    """A per-step controller by name, or None for the program-based ones (webster, actuated,
    default), which SUMO runs from its own program files."""
    if name in ("agents_v1", "agents_v1_observed"):  # step 3, version 1 (cosim/agents.py imports this module)
        from cosim.agents import JunctionAgents, ObservedJunctionAgents

        return JunctionAgents() if name == "agents_v1" else ObservedJunctionAgents()
    if re.split(r"[-+@]", name, maxsplit=1)[0] in ("agents_goflow", "agents_observed_goflow"):  # GO-FLOW priority agent
        from cosim.goflow_priority import make as make_goflow

        return make_goflow(name)
    if re.split(r"[-+@]", name, maxsplit=1)[0] in ("agents", "agents_observed"):  # step 3, version 2: on the actuated program
        from cosim.agents_actuated import make as make_agents

        return make_agents(name)
    cls = CONTROLLERS.get(name)
    return cls() if cls else None

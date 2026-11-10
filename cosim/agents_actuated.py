"""Step 3, version 2: junction agents on top of SUMO's actuated program.

Version 1 (cosim/agents.py) replaced the actuated program with max-pressure and lost to it. Here
SUMO's own actuated logic keeps running every signal (scenarios/town05/town05.actuated.add.xml; since 25 Oct,
tools/build_town05_signals.py: two greens a cycle, the opposite approaches together with permissive left and right
turns, each 7-45 s on SUMO's induction loops, the walk on the crossings parallel to it running with it, then its
walk clearance (flashing don't walk, the cars still green), yellow and all red; until 25 Oct CARLA's cycle: four
single-approach greens, then a 10 s walk on every crossing and its 5 s clearance). One agent per junction watches
its approaches and steps in only with commands that keep the program's own states and order:

* skip   An empty phase is skipped. During the all-red before it, the agent jumps to the all-red
         that precedes the next phase with demand (every all-red is the same "all r" state, and
         SUMO keeps the remaining all-red time, so the 3 s stay). The program then starts that
         green itself, with its own 5 s minimum and its own actuation. Demand: a vehicle counted on
         the phase's approach (75 m), or one of the phase's loops passed since its green last ended
         (that vehicle is waiting at the red), or someone waiting at a crossing whose walk runs with that
         green (a pedestrian call, since 25 Oct). An exclusive walk phase (CARLA's old cycle) is skipped when nobody
         waits at its crossings. A car phase is served at least every `skip_limit`-th time it comes up.
* spill  A green whose waiting vehicles all face a full exit (halting vehicles over `full_share` of
         the exit lane: the neighbour's entry is full) moves nobody; SUMO's loops still see the
         standing queue and extend it to 60 s. The agent ends it (after the 5 s minimum, the
         condition held for `spill_hold_s`), when another phase has demand. Such a phase also counts
         as empty when the next phase is chosen (skip), up to `skip_limit`.
* trickle A green kept alive by a few moving vehicles (no queue on its approach) while another
         phase has `trickle_q` vehicles queued is ended after `trickle_min_s`.
* balance Fill = queued vehicles / storage (lane length / 7.5 m) of an approach lane, the fullest
         lane of a phase. A short link fills in one red and then blocks the junction upstream; a long
         one can store. After `balance_min_s` of green, a phase whose fill is under `balance_low`
         gives way when another phase's fill reaches `balance_high`.
* preempt An emergency vehicle on a run asks its next signal within 150 m (its own request, as in
         step 2; the camera only confirms). A green serving it is held (up to 60 s of green); any
         other car green is ended after its 5 s minimum; the following all-red jumps to its phase.
         A vehicle standing still for 5 s at a green is blocked further on and its request lapses.
         A request counts only while serving it can move the vehicle (`_can_pass`, judged by the vehicle): the
         lane it leaves the junction by is not full, and at most `ev_queue_max` vehicles stand ahead of it. In a
         jam the requests of stuck vehicles kept cutting and holding greens that moved nobody, and the town
         locked up earlier than under actuated (x1.0, dev seeds 11/12: preempt and corridor alone 1155/682
         vehicles an hour against actuated's 1580/1731, the queue features alone 2499/1513).
* pedcall Someone waiting at a crossing whose walk runs with a car green is demand for that green (a pedestrian
         call, as a push button places one): skip does not pass over it. Since 25 Oct; before, a green with no cars
         was skipped with its walk and the people waited for the skip limit (measured, seeds 21-25: pedestrian time
         loss 0.4-1.6 s per walk lower with it; out/agents_pedfix).
* corridor The request is passed on to the signal after that when the vehicle is within 300 m of it,
         moving (3 m/s or more) and due there within `corridor_eta_s` (25 s). Without the last two
         conditions, vehicles stuck in a jam kept far signals cutting and holding greens for minutes
         (dev seed 11, x1.0: 5000+ corridor requests).

* extend (optional, "+extend") At the 60 s maximum, a green whose fill is still `extend_fill` and
         `extend_margin` above every other phase's, with free exits and nobody waiting to walk, is
         extended 5 s at a time, to 90 s at most. Off by default: in the ablation it acted 0-8 times an
         hour and changed nothing measurable (x0.8 no change, x1.0 within the runs' spread).
* forecast (optional, "+forecast", since 8 Nov) The GO-FLOW forecasters' warning (cosim/goflow_prediction.py,
         `GoflowPrediction.risk`): per approach, the chance that a long queue starts there within 3 minutes
         (DSO4's p_long; 1 when the queue is long already). Read every step through `forecast` (a callable
         -> {approach: risk}, {} when it has nothing fresh: warming up, offline, a forecast older than 45 s). It acts
         only through two of the agent's own moves, as a weight, never as a command:
         - balance: a phase's fill counts `forecast_w` x its risk more. A green whose road is about to queue is not
           ended for being under `balance_low`; another phase about to queue reaches `balance_high` sooner.
         - at the green's maximum, an approach with risk `forecast_extend_p` or more, above every other phase's
           risk, its exits free and nobody waiting to walk elsewhere, gets 5 s more at a time, to 90 s at most.
         Off by default; switched on with the forecasters running (tools/eval_agents_forecast.py measures it).
* batch  (optional, "+batch") The walk is served once `batch_n` people wait at its crossings or the
         first has waited `batch_s`, instead of for the first person: fewer 15 s walks, more
         car green, at a cost in pedestrian delay.

The walk is never cut short, and nothing is ever shown that the program does not show: greens end
through their own yellow (setPhaseDuration 0), and jumps go from all red to all red only.
Where the cameras see too little (incoming coverage under 70 %: 2014 and 2328) the queue features are
off and only preempt/corridor act, which do not depend on the cameras.

Counts: "agents" reads SUMO's lanes, "agents_observed" the camera reports (cosim/camera_reports.py,
SimulatedCameras: vehicles and queued per lane, every 2 s). A name may switch features off for the
ablation: "agents-skip-trickle" is "agents" without skip and trickle; "agents+batch" adds the optional one.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import traci.constants as _tc

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "out" / "town05_cameras" / "coverage.json"
VEH_M = 7.5
FEATURES = ("skip", "spill", "trickle", "balance", "preempt", "corridor", "pedcall")
OPTIONAL = ("batch", "extend", "noflow", "forecast", "camback")  # off unless named: "agents+batch"
DEFAULT_OFF = ("trickle",)  # off unless named ("agents+trickle") since 25 Oct: with two greens a cycle
# (tools/build_town05_signals.py, opposite approaches together) ending a green kept alive by moving cars costs a
# whole switch; measured (tools/check_highway.py, 20 min at 17:30, 8 seeds): town's share of cars standing 0.437
# without it, 0.490 with it, 0.453 for SUMO's actuated program alone; without "balance" too, seed 31 locked up
QUEUE_FEATURES = ("skip", "spill", "trickle", "balance", "extend")
_CROSSING = re.compile(r"^:.+_c\d+$")


def _greens(state: str) -> frozenset[int]:
    return frozenset(i for i, c in enumerate(state) if c in "Gg")


@dataclass
class _Agent:
    tls: str
    states: list[str]
    kind: list[str]                        # per program phase: car | walk | clear | yellow | red
    min_dur: list[float]
    max_dur: list[float]
    cycle: list[int]                       # program indices of the car and walk greens, in order
    pre_red: dict[int, int]                # green -> the all-red just before it
    pairs: dict[int, list[tuple[str, str]]]  # car green -> its (in lane, out lane) pairs
    cross: dict[int, str]                  # link index -> crossing edge
    loops: dict[int, list[str]]            # car green -> its induction loops
    covered: bool = True
    link_out: dict[int, str] = field(default_factory=dict)  # link index -> the lane it leaves by
    lane_links: dict[str, set[int]] = field(default_factory=dict)  # a lane a camera may see a siren on -> the links
    #   it leads to (its own if it is an approach lane, all of the approach lanes' within reach_m upstream of them)
    stop_lines: dict[int, list[tuple[float, float, float]]] = field(default_factory=dict)  # car green -> (x, y,
    #   heading deg, SUMO's: 0 north, clockwise) where each of its approach lanes meets the junction
    held_for: str | None = None             # the emergency vehicle whose green is being held (one event per hold)
    phase: int = -1
    entered: float = 0.0
    green_since: float = 0.0
    decided: bool = False
    last_end: dict[int, float] = field(default_factory=dict)
    passed_over: dict[int, int] = field(default_factory=dict)
    blocked_since: float | None = None
    stalled_since: float | None = None
    switches: int = 0
    log: list[tuple[float, int]] = field(default_factory=list)

    def next_green(self, k: int) -> int:
        n = len(self.states)
        for d in range(1, n + 1):
            i = (k + d) % n
            if i in self.cycle:
                return i
        return k


class ActuatedAgents:
    name = "agents"
    program = "actuated"           # sumo_side loads scenarios/<net>/<net>.actuated.add.xml
    uses_cameras = False

    reach_m = 75.0
    skip_limit = 3
    full_share = 0.85
    spill_hold_s = 2.0
    trickle_min_s = 15.0
    trickle_q = 8
    trickle_cur = 2
    balance_min_s = 20.0
    balance_low = 0.3
    balance_high = 0.85
    extend_fill = 0.6
    extend_margin = 0.25
    extend_step_s = 5.0
    extend_max_s = 90.0
    detect_m = 150.0
    corridor_m = 300.0
    corridor_eta_s = 25.0
    corridor_min_speed = 3.0
    stall_s = 5.0
    ev_queue_max = 8               # vehicles standing ahead of an emergency vehicle beyond which it does not ask
    ev_hold_max_s = 60.0
    ev_starve_limit = 3            # a car phase passed over this often is served before an emergency jump
    ev_patience_s = 45.0           # an emergency vehicle still not through its signal after this long asks no more
    #   there (a stuck one's request kept the signal jumping back to its green, which moved nobody: 207, dev seed 14)
    batch_n = 2
    batch_s = 40.0
    noflow_s = 4.0                 # a green under which nobody moves for this long is ended (noflow)
    noflow_tol = 1                 # vehicles on a waiting lane that may still be moving (arriving at the back)
    min_coverage = 0.70
    forecast_w = 0.5               # +forecast: fill counted this much more per unit of risk (balance)
    forecast_extend_p = 0.8        # +forecast: risk from which a green at its maximum is extended
    # Requests that never arrive (10 Nov): the share of (emergency vehicle, junction) requests lost on the way, for
    # measuring +camback; 0 in normal runs. Which ones is fixed per pair (a hash), so paired runs lose the same ones.
    request_loss = 0.0
    loss_seed = 0
    siren_flash = 1                # agents_observed: the simulated cameras call sirens as the H window does since 10 Nov
    #   (the flash model, camera_reports.FLASH_ON / FLASH_OFF); 0: the detector's 3-of-8 vote alone, as before

    def __init__(self, off: set[str] | frozenset[str] = frozenset(), fallback: set[str] | None = None,
                 record: bool = False, emergency=None, extra: set[str] | frozenset[str] = frozenset()):
        bad = (set(off) - set(FEATURES)) | (set(extra) - set(OPTIONAL) - set(DEFAULT_OFF))
        if bad:
            raise ValueError(f"unknown agent features {sorted(bad)}; known: {FEATURES} and {OPTIONAL}")
        self.on = {f for f in FEATURES if f not in off and f not in DEFAULT_OFF} | set(extra)
        if fallback is None:
            fallback = set()
            if COVERAGE.is_file():
                cov = json.loads(COVERAGE.read_text(encoding="utf-8"))["junctions"]
                fallback = {t for t, v in cov.items() if (v["in_share"] or 0) < self.min_coverage}
        self.fallback = set(fallback)
        self.record = record
        self.emergency = emergency or (lambda: ())
        self.forecast = None            # +forecast: callable -> {approach node: risk 0-1} (GoflowPrediction.risk)
        self.forecast_nodes: list[dict] | None = None  # the approaches (nodes.json); default models/goflow/graph
        self._risk: dict[str, float] = {}
        self.junctions: dict[str, _Agent] = {}
        self.obs: dict[str, tuple[int, int]] = {}
        self.visible: dict[str, list] | None = None  # camera mode: lane -> visible stretches (Layout.visible)
        self.stats = {"skipped": 0, "walk_skipped": 0, "spill_cuts": 0, "spill_skips": 0, "trickle_cuts": 0,
                      "balance_cuts": 0, "extensions": 0, "noflow_cuts": 0,
                      "ev_holds": 0, "ev_cuts": 0, "ev_jumps": 0, "corridor_requests": 0, "ev_dropped": 0,
                      "ev_patience": 0, "ev_starved": 0,
                      "ev_exit_full": 0, "ev_queued": 0,
                      "forecast_steps": 0, "forecast_balance_cuts": 0, "forecast_balance_kept": 0,
                      "forecast_extensions": 0, "ev_lost": 0, "cam_requests": 0, "cam_patience": 0,
                      "cam_queued": 0}
        self._cam_ems: dict[str, set[str]] = {}         # junction -> lanes with a siren its cameras confirmed
        self._cam_first: dict[tuple[str, str], float] = {}
        self._lost_seen: set[tuple[str, str]] = set()
        self._stopped_on_green: dict[str, float] = {}
        self.requests: dict[str, set[int]] = {}
        self.request_by: dict[str, str] = {}   # junction -> the emergency vehicle asking (for the events)
        self._corridor_seen: set[tuple[str, str]] = set()
        self._asked: dict[tuple[str, str], float] = {}  # (junction, emergency vehicle) -> when it first asked
        # what the agents did, for the viewer (cosim/agent_hud.py): {t, tls, kind, phases, text}
        self.events: deque = deque(maxlen=400)

    def _event(self, a: _Agent, kind: str, phases: list[int], text: str, t: float | None = None) -> None:
        self.events.append({"t": self._now if t is None else t, "tls": a.tls, "kind": kind,
                            "phases": [k for k in phases if a.kind[k] == "car"], "text": text})

    # ------------------------------------------------------------------ interface
    def additional_files(self) -> list:
        return []

    def observe(self, reports: list[dict]) -> None:
        """Camera reports (cosim/camera_reports.py): lane -> (vehicles, queued), summed over cameras."""
        out: dict[str, list[int]] = {}
        for rep in reports:
            for ln, (n, q) in rep.get("lanes", {}).items():
                a = out.setdefault(ln, [0, 0])
                a[0] += n
                a[1] += q
        self.obs = {k: (v[0], v[1]) for k, v in out.items()}
        ems: dict[str, set[str]] = {}
        for rep in reports:
            for e in rep.get("ems", []):
                if e.get("confirmed") and rep.get("tls"):
                    ems.setdefault(rep["tls"], set()).add(e["lane"])
        self._cam_ems = ems

    def setup(self, conn) -> None:
        tc = conn.tc
        self._graph(tc)
        loops_by_lane: dict[str, list[str]] = {}
        for det in tc.inductionloop.getIDList():
            loops_by_lane.setdefault(tc.inductionloop.getLaneID(det), []).append(det)
        now = tc.simulation.getTime()
        for tls in tc.trafficlight.getIDList():
            logics = {lg.programID: lg for lg in tc.trafficlight.getAllProgramLogics(tls)}
            if self.program not in logics:
                continue
            tc.trafficlight.setProgram(tls, self.program)
            logic = logics[self.program]
            links = tc.trafficlight.getControlledLinks(tls)
            cross = {}
            for i, ls in enumerate(links):
                for link in ls:
                    edge = link[1].rsplit("_", 1)[0]
                    if _CROSSING.match(edge):
                        cross[i] = edge
            states, kind, mn, mx = [], [], [], []
            for p in logic.phases:
                s = p.state
                g = _greens(s)
                states.append(s)
                nm = getattr(p, "name", "")
                if "y" in s.lower() or nm == "yellow":
                    kind.append("yellow")
                elif nm == "all red":  # (27 Oct: a right turn that goes on may keep its green through it)
                    kind.append("red")
                elif nm in ("walk clearance", "walk buffer"):
                    kind.append("clear")  # (since 25 Oct: the cars' green goes on while the walkers get across;
                    #                          never jumped to, never cut by the agent: sumo_side.WalkSkip ends it
                    #                          once the crossings are empty)
                elif not g:
                    kind.append("red")
                elif g <= set(cross):
                    kind.append("walk")
                else:
                    kind.append("car")
                mn.append(float(p.minDur) if p.minDur >= 0 else float(p.duration))
                mx.append(float(p.maxDur) if p.maxDur >= 0 else float(p.duration))
            n = len(states)
            cycle = [i for i in range(n) if kind[i] in ("car", "walk")]
            # the all reds that can be jumped from and to: every link red (since 27 Oct an all red may keep a right
            # turn green that goes on into the next green; jumping from it could end that green without its yellow)
            pre_red = {i: (i - 1) % n for i in cycle if kind[(i - 1) % n] == "red" and not _greens(states[(i - 1) % n])}
            pairs, loops = {}, {}
            for i in cycle:
                if kind[i] != "car":
                    continue
                seen = []
                for li in _greens(states[i]):
                    for link in links[li] if li < len(links) else []:
                        if (link[0], link[1]) not in seen:
                            seen.append((link[0], link[1]))
                pairs[i] = seen
                loops[i] = sorted({d for a, _ in seen for d in loops_by_lane.get(a, [])})
            a = _Agent(tls, states, kind, mn, mx, cycle, pre_red, pairs, cross, loops,
                       covered=tls not in self.fallback,
                       link_out={i: ls[0][1] for i, ls in enumerate(links) if ls})
            own: dict[str, set[int]] = {}
            for i, ls in enumerate(links):
                for link in ls:
                    if not link[0].startswith(":"):
                        own.setdefault(link[0], set()).add(i)
            for ln, idx in own.items():
                for up in self._reach_up(ln):
                    a.lane_links.setdefault(up, set()).update(idx)
            for ln, idx in own.items():  # an approach lane: its own links only
                a.lane_links[ln] = set(idx)
            for k, prs in pairs.items():
                pts = []
                for i_lane in dict.fromkeys(x for x, _ in prs):
                    shp = tc.lane.getShape(i_lane)
                    if len(shp) >= 2:
                        (x0, y0), (x1, y1) = shp[-2], shp[-1]
                        pts.append((x1, y1, math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0))
                a.stop_lines[k] = pts
            a.phase = tc.trafficlight.getPhase(tls)
            a.entered = a.green_since = now
            self.junctions[tls] = a
        self._map_forecast()

    def _map_forecast(self) -> None:
        """Each car green -> the forecasters' approaches it serves (an approach's lanes among the green's in-lanes)."""
        self.served: dict[tuple[str, int], list[str]] = {}
        if "forecast" not in self.on:
            return
        nodes = self.forecast_nodes
        if nodes is None:
            f = ROOT / "models" / "goflow" / "graph" / "nodes.json"
            nodes = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
        for n in nodes:
            a = self.junctions.get(n["tls"])
            if a is None:
                continue
            for k, prs in a.pairs.items():
                if set(n["lanes"]) & {i for i, _ in prs}:
                    self.served.setdefault((a.tls, k), []).append(n["node"])
        self.stats["forecast_phases_served"] = len(self.served)

    def _risk_of(self, a: _Agent, k: int) -> float:
        return max((self._risk.get(n, 0.0) for n in self.served.get((a.tls, k), ())), default=0.0)

    # ------------------------------------------------------------------ lanes
    def _graph(self, tc) -> None:
        self._len: dict[str, float] = {}
        self._pred: dict[str, list[str]] = {}
        for lane in tc.lane.getIDList():
            self._len[lane] = tc.lane.getLength(lane)
            for link in tc.lane.getLinks(lane):
                nxt, via = link[0], link[4] if len(link) > 4 else ""
                chain = [lane, via, nxt] if via else [lane, nxt]
                for x, y in zip(chain, chain[1:], strict=False):
                    self._pred.setdefault(y, []).append(x)
        self._up: dict[str, tuple[str, ...]] = {}

    def _reach_up(self, lane: str) -> tuple[str, ...]:
        if lane in self._up:
            return self._up[lane]
        out, frontier, seen = [lane], [(lane, self._len.get(lane, 0.0))], {lane}
        while frontier:
            cur, covered = frontier.pop()
            if covered >= self.reach_m:
                continue
            for p in self._pred.get(cur, []):
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    frontier.append((p, covered + self._len.get(p, 0.0)))
        self._up[lane] = tuple(out)
        return self._up[lane]

    def _count(self, tc, lane: str) -> tuple[int, int]:
        """(vehicles, halting) on one lane, from SUMO or from the cameras."""
        c = self._cache.get(lane)
        if c is None:
            if self.uses_cameras:
                c = self.obs.get(lane, (0, 0))
            else:
                c = (tc.lane.getLastStepVehicleNumber(lane), tc.lane.getLastStepHaltingNumber(lane))
            self._cache[lane] = c
        return c

    def _approach(self, tc, a: _Agent, k: int) -> tuple[int, int]:
        """(vehicles, queued) on phase k's approach lanes within reach_m."""
        lanes = {x for i, _ in a.pairs.get(k, []) for x in self._reach_up(i)}
        n = q = 0
        for x in lanes:
            v, h = self._count(tc, x)
            n += v
            q += h
        return n, q

    EXIT_START_M = 20.0  # cameras: an exit counts only if they see it this close to its start (the junction)

    def _exit_full(self, tc, lane: str) -> bool:
        """The lane a movement leaves by is full: halting vehicles over `full_share` of it. With the cameras,
        of the part they see, and only when that part reaches the lane's start (just past the junction, where
        a full lane blocks the junction); else not known, taken as not full. Measured against the whole lane, as
        before, a camera's count (27 % of the lanes' length is seen) could never reach it: the emergency
        requests of vehicles facing a blocked exit kept their greens, and spill-back was hardly ever seen."""
        if self.uses_cameras and self.visible is not None:
            st = self.visible.get(lane, ())
            if not any(a <= self.EXIT_START_M for a, _ in st):
                return False
            cap = max(1.0, sum(b - a for a, b in st) / VEH_M)
        else:
            cap = max(1.0, self._len.get(lane, 0.0) / VEH_M)
        return self._count(tc, lane)[1] >= max(1.0, self.full_share * cap)

    def _storage(self, lane: str) -> float:
        """Vehicles the lane can hold (the part the cameras see, in camera mode)."""
        if self.uses_cameras and self.visible is not None:
            return sum(b - a for a, b in self.visible.get(lane, ())) / VEH_M
        return self._len.get(lane, 0.0) / VEH_M

    def _fill(self, tc, a: _Agent, k: int) -> float:
        """Queued / storage of phase k's fullest approach lane."""
        best = 0.0
        for i in {i for i, _ in a.pairs.get(k, [])}:
            cap = self._storage(i)
            if cap >= 1.0:
                best = max(best, self._count(tc, i)[1] / cap)
        return best

    def _blocked(self, tc, a: _Agent, k: int) -> bool:
        """Every movement of phase k with vehicles waiting faces a full exit (and some wait)."""
        waiting = blocked = 0
        for i, o in a.pairs.get(k, []):
            if self._count(tc, i)[1] > 0:
                waiting += 1
                blocked += self._exit_full(tc, o)
        return waiting > 0 and blocked == waiting

    def _stalled(self, tc, a: _Agent, k: int, now: float) -> bool:
        """Nobody moves under phase k's green. Whatever holds them (an exit full past where the cameras see, cross
        traffic stuck in the junction box, a lane blocked further on), the green moves nobody.
        With induction loops on the phase's lanes (the actuated program's own, at every junction): a vehicle
        stands on one of them and none has passed any of them for `noflow_s`. Else from the counts (cameras or
        SUMO's lanes): on every approach lane with vehicles, all of them stand (one may still be rolling up)."""
        loops = a.loops.get(k, ())
        if loops:
            standing = False
            for d in loops:
                if tc.inductionloop.getLastStepMeanSpeed(d) > 0.5:
                    return False  # one rolling over it
                if tc.inductionloop.getLastStepOccupancy(d) >= 99.0:
                    standing = True  # one standing on it (SUMO gives it no speed: -1, and 0 s since detection)
                elif tc.inductionloop.getTimeSinceDetection(d) < min(self.noflow_s, now - a.green_since):
                    return False  # one passed it a moment ago
            return standing
        lanes = 0
        for i in {i for i, _ in a.pairs.get(k, [])}:
            n, q = self._count(tc, i)
            if n == 0:
                continue
            if q == 0 or n - q > self.noflow_tol:
                return False
            lanes += 1
        return lanes > 0

    def _peds(self, a: _Agent, k: int) -> int:
        return sum(self._waiting.get(a.cross[i], 0) for i in _greens(a.states[k]) if i in a.cross)

    def _demand(self, tc, a: _Agent, k: int, now: float) -> bool:
        if a.kind[k] == "walk":
            if "batch" in self.on:
                crossings = [a.cross[i] for i in _greens(a.states[k]) if i in a.cross]
                return (self._peds(a, k) >= self.batch_n
                        or max((self._longest.get(x, 0.0) for x in crossings), default=0.0) >= self.batch_s)
            return self._peds(a, k) > 0
        if self._approach(tc, a, k)[0] > 0:
            return True
        if "pedcall" in self.on and self._peds(a, k) > 0:  # people waiting at the crossings that walk with this
            return True  # green: a pedestrian call (since 25 Oct; "agents-pedcall" without it, as before)
        since = now - a.last_end.get(k, -1e9)
        return any(tc.inductionloop.getTimeSinceDetection(d) < since for d in a.loops.get(k, ()))

    # ------------------------------------------------------------------ the step
    def step(self, conn) -> None:
        tc = conn.tc
        now = self._now = tc.simulation.getTime()
        self._cache: dict[str, tuple[int, int]] = {}
        self._pedestrians(tc)
        self._requests(tc, now)
        if "forecast" in self.on and self.forecast is not None:
            try:
                self._risk = self.forecast() or {}
            except Exception:  # noqa: BLE001 - the signals never wait for, or fail with, the forecasters
                self._risk = {}
            if self._risk:
                self.stats["forecast_steps"] += 1
        for a in self.junctions.values():
            self._decide(tc, a, now)

    def _pedestrians(self, tc) -> None:
        subs = tc.person.getAllSubscriptionResults()
        waiting: dict[str, int] = {}
        longest: dict[str, float] = {}
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
                if "batch" in self.on:
                    longest[nxt] = max(longest.get(nxt, 0.0), tc.person.getWaitingTime(pid))
        self._waiting = waiting
        self._longest = longest

    # An outside planner of the emergency requests (cosim/goflow_priority.py), None by default. Called on every step
    # as request_source(tc, now), it returns None (no plan this step: the agents ask as below, for every vehicle on a
    # run) or {"items": [...], "handles": {vehicle ids}}. Each item {"tls", "vehicle" (None: a siren a camera saw
    # with no request), "links", "rank" (0: the vehicle's next signal), "kind", "text"} goes through the same guards
    # as the agents' own requests (_external_requests); the vehicles on a run that the plan does not handle are asked
    # for as below. The safety rules stay here: the planner only says who should get which link.
    request_source = None
    external_guard = 1  # 0: a planner's requests beyond the next signal get only the exit check (as before 9 Nov)

    def _requests(self, tc, now: float) -> None:
        self.requests = {}
        self.request_by = {}
        if "preempt" not in self.on:
            return
        external = None
        if self.request_source is not None:
            try:
                external = self.request_source(tc, now)
            except Exception:  # noqa: BLE001 - a planner's fault is never the town's
                external = None
            key = "external_fallback_steps" if external is None else "external_plan_steps"
            self.stats[key] = self.stats.get(key, 0) + 1
        wanted = set(self.emergency())
        asked = {}
        if external is not None:
            self._external_requests(tc, now, external.get("items", []), asked)
            wanted -= set(external.get("handles", ()))
        if not wanted:
            self._asked = asked
            self._camera_backup(tc, now)
            return
        for vid in wanted & set(tc.vehicle.getIDList()):
            upcoming = tc.vehicle.getNextTLS(vid)
            if not upcoming:
                continue
            tls, link, dist, state = upcoming[0]
            if tls in self.junctions and dist <= self.detect_m and not self._lost(vid, tls):
                first = asked[(tls, vid)] = self._asked.get((tls, vid), now)
                if now - first > self.ev_patience_s:
                    self.stats["ev_patience"] += 1
                    continue
                if state in "Gg" and tc.vehicle.getSpeed(vid) < 0.5:
                    since = self._stopped_on_green.setdefault(vid, now)
                    if now - since >= self.stall_s:
                        self.stats["ev_dropped"] += 1
                        continue
                else:
                    self._stopped_on_green.pop(vid, None)
                if self._can_pass(tc, self.junctions[tls], vid, int(link)):
                    self.requests.setdefault(tls, set()).add(int(link))
                    self.request_by[tls] = vid
            if "corridor" in self.on and len(upcoming) > 1:
                tls2, link2, dist2, _ = upcoming[1]
                speed = tc.vehicle.getSpeed(vid)
                if tls2 in self.junctions and dist2 <= self.corridor_m and dist <= self.detect_m \
                        and not self._lost(vid, tls2) and speed >= self.corridor_min_speed and dist2 / speed <= self.corridor_eta_s \
                        and self._exit_free(tc, self.junctions[tls2], int(link2)):
                    self.requests.setdefault(tls2, set()).add(int(link2))
                    self.request_by.setdefault(tls2, vid)
                    self.stats["corridor_requests"] += 1
                    if (vid, tls2) not in self._corridor_seen:
                        self._corridor_seen.add((vid, tls2))
                        a2 = self.junctions[tls2]
                        self._event(a2, "corridor", [k for k in a2.cycle if a2.kind[k] == "car"
                                                     and int(link2) in _greens(a2.states[k])][:1],
                                    "green wave: getting ready for an ambulance")
        self._asked = asked
        self._camera_backup(tc, now)

    def _lost(self, vid: str, tls: str) -> bool:
        """Whether this vehicle's request to this junction is one of the lost ones (request_loss)."""
        if self.request_loss <= 0:
            return False
        if zlib.crc32(f"{self.loss_seed}|{vid}|{tls}".encode()) / 2 ** 32 >= self.request_loss:
            return False
        self._lost_seen.add((vid, tls))
        self.stats["ev_lost"] = len(self._lost_seen)
        return True

    def _camera_backup(self, tc, now: float) -> None:
        """+camback (10 Nov): a siren the junction's cameras confirmed on an approach, and no request at that junction
        (lost on the way, or a vehicle with no radio): the junction gives that approach its green as if asked. A
        request the junction heard and its limits refused stays refused: the cameras do not overrule it. Only
        what the cameras report (the confirmed siren, the lane it is on, the queue they count there) and the exit
        lane's own detectors; the same limits as a request: ev_patience_s per approach, at most ev_queue_max vehicles
        queued on it, an exit with room."""
        if "camback" not in self.on or not self.uses_cameras:
            return
        active = set()
        heard = {t for t, _ in self._asked}
        for tls, lanes in sorted(self._cam_ems.items()):
            a = self.junctions.get(tls)
            if a is None:
                continue
            # a vehicle asked here, granted or not (its own limits said no: a queue ahead, a full exit, patience):
            # the cameras only keep the time
            asked = bool(self.requests.get(tls)) or tls in heard
            for lane in sorted(lanes):
                links = a.lane_links.get(lane)
                if not links:
                    continue
                key = (tls, lane)
                active.add(key)
                first = self._cam_first.setdefault(key, now)
                if asked or self.requests.get(tls):
                    continue
                if now - first > self.ev_patience_s:
                    self.stats["cam_patience"] += 1
                    continue
                if self.obs.get(lane, (0, 0))[1] > self.ev_queue_max:
                    self.stats["cam_queued"] += 1
                    continue
                free = {link for link in sorted(links) if self._exit_free(tc, a, link)}
                if not free:
                    continue
                self.requests[tls] = free
                self.request_by[tls] = "camera"
                self.stats["cam_requests"] += 1
                if now == first:
                    self._event(a, "ambulance", [k for k in a.cycle if a.kind[k] == "car"
                                                 and free & _greens(a.states[k])][:1],
                                "green for flashing lights a camera saw (no request)")
                break
        for key in list(self._cam_first):
            if key not in active:
                self._cam_first.pop(key)

    def _external_requests(self, tc, now: float, items: list[dict], asked: dict) -> None:
        """An outside planner's requests (request_source), through the agents' own guards: its next signal, the
        patience limit, a vehicle stopped on its green, a link it can pass; a signal further on, the corridor's own
        conditions (the vehicle moving at `corridor_min_speed` or more and due there within `corridor_eta_s` at its
        real speed, an exit with room). Before 9 Nov a signal further on had only the exit check:
        a vehicle stuck in a queue kept the signals ahead holding green for it (dev seeds 11-20, x1.0, an emergency
        every 120 s: 976 holds a run against the agents' own 122, 1549 vehicles stuck against 824)."""
        alive = set(tc.vehicle.getIDList())
        seen = self.__dict__.setdefault("_external_seen", set())
        for it in items:
            a = self.junctions.get(it.get("tls"))
            links = [int(x) for x in it.get("links", [])]
            vid = it.get("vehicle")
            if a is None or not links:
                continue
            if vid is None or it.get("rank", 0) > 0:
                if vid is not None and vid not in alive:
                    continue
                if vid is not None and self.external_guard and not self._corridor_ok(tc, a, vid):
                    continue
                if not self._exit_free(tc, a, links[0]):
                    continue
            else:
                if vid not in alive:
                    continue
                first = asked[(a.tls, vid)] = self._asked.get((a.tls, vid), now)
                if now - first > self.ev_patience_s:
                    self.stats["ev_patience"] += 1
                    continue
                nxt = tc.vehicle.getNextTLS(vid)[:1]
                if nxt and nxt[0][3] in "Gg" and tc.vehicle.getSpeed(vid) < 0.5:
                    since = self._stopped_on_green.setdefault(vid, now)
                    if now - since >= self.stall_s:
                        self.stats["ev_dropped"] += 1
                        continue
                else:
                    self._stopped_on_green.pop(vid, None)
                if not self._can_pass(tc, a, vid, links[0]):
                    continue
            self.requests[a.tls] = set(links)
            self.request_by[a.tls] = vid or "camera"
            self.stats["external_requests"] = self.stats.get("external_requests", 0) + 1
            text = it.get("text")
            if text and (a.tls, vid, text) not in seen:
                seen.add((a.tls, vid, text))
                self._event(a, it.get("kind", "ambulance"),
                            [k for k in a.cycle if a.kind[k] == "car" and set(links) & _greens(a.states[k])][:1], text)

    def _corridor_ok(self, tc, a: _Agent, vid: str) -> bool:
        """A planner's request for a signal beyond the vehicle's next one: the agents' corridor conditions."""
        dist = next((float(s[2]) for s in tc.vehicle.getNextTLS(vid) if s[0] == a.tls), None)
        speed = tc.vehicle.getSpeed(vid)
        if dist is None or speed < self.corridor_min_speed or dist / speed > self.corridor_eta_s:
            self.stats["external_too_far"] = self.stats.get("external_too_far", 0) + 1
            return False
        return True

    def _exit_free(self, tc, a: _Agent, link: int) -> bool:
        out = a.link_out.get(link)
        if out is not None and self._exit_full(tc, out):
            self.stats["ev_exit_full"] += 1
            return False
        return True

    def _can_pass(self, tc, a: _Agent, vid: str, link: int) -> bool:
        """Whether giving the emergency vehicle its green now would move it: the lane it leaves the junction by
        is not full, and at most `ev_queue_max` vehicles stand ahead of it on its lane. Judged by the vehicle
        itself, part of its own request (it sees the vehicles ahead of it and the road past the junction), in
        both modes. Judged by the cameras, which see 27 % of the lanes' length, it let the requests of stuck
        vehicles through (x1.0, dev seeds 11-16: 1536 vehicles an hour, 3 seeds of 6 better than actuated's
        1510; judged by the vehicle, 1725, 4 of 6, ambulance time loss 407 -> 311 s)."""
        out = a.link_out.get(link)
        if out is not None:
            cap = max(1.0, self._len.get(out, 0.0) / VEH_M)
            if tc.lane.getLastStepHaltingNumber(out) >= max(1.0, self.full_share * cap):
                self.stats["ev_exit_full"] += 1
                return False
        lane = tc.vehicle.getLaneID(vid)
        if lane.startswith(":"):
            return True  # already in the junction
        pos = tc.vehicle.getLanePosition(vid)
        ahead = sum(1 for v in tc.lane.getLastStepVehicleIDs(lane)
                    if v != vid and tc.vehicle.getLanePosition(v) > pos and tc.vehicle.getSpeed(v) < 0.5)
        if ahead > self.ev_queue_max:
            self.stats["ev_queued"] += 1
            return False
        return True

    def _ev_target(self, a: _Agent, start: int) -> int | None:
        """The first green from `start` on (in program order) serving a requested link."""
        req = self.requests.get(a.tls)
        if not req:
            return None
        order = a.cycle[a.cycle.index(start):] + a.cycle[:a.cycle.index(start)]
        for k in order:
            if a.kind[k] == "car" and req & _greens(a.states[k]):
                return k
        return None

    def _choose(self, tc, a: _Agent, natural: int, now: float) -> int:
        self._spilled_now: set[int] = set()
        ev = self._ev_target(a, natural)
        if ev is not None:
            if ev != natural and a.kind[natural] == "car" and a.passed_over.get(natural, 0) >= self.ev_starve_limit \
                    and self._demand(tc, a, natural, now):
                self.stats["ev_starved"] += 1  # passed over too often for emergency vehicles: its turn first
                return natural
            return ev
        if "skip" not in self.on or not a.covered:
            return natural
        order = a.cycle[a.cycle.index(natural):] + a.cycle[:a.cycle.index(natural)]
        for k in order:
            if k not in a.pre_red:
                return k  # cannot be jumped to: take it (and everything up to it)
            if a.kind[k] == "car" and a.passed_over.get(k, 0) >= self.skip_limit - 1:
                return k
            if self._demand(tc, a, k, now):
                if a.kind[k] == "car" and "spill" in self.on and self._blocked(tc, a, k):
                    self.stats["spill_skips"] += 1
                    self._spilled_now.add(k)
                    self._event(a, "spill", [k], "skipped a green: its exit road is full")
                    continue
                return k
        return natural  # nobody anywhere: keep the program's order

    def _jump(self, tc, a: _Agent, natural: int, target: int) -> None:
        """In an all-red: jump to the all-red before `target` (SUMO keeps the remaining time)."""
        order = a.cycle[a.cycle.index(natural):]
        order += a.cycle[:a.cycle.index(natural)]
        passed = order[:order.index(target)]
        for k in passed:
            if a.kind[k] == "walk":
                self.stats["walk_skipped"] += 1
            else:
                self.stats["skipped"] += 1
                a.passed_over[k] = a.passed_over.get(k, 0) + 1
        if self.requests.get(a.tls) and self._ev_target(a, natural) == target:
            self._event(a, "ambulance", [target], "ambulance: straight to its green")
        elif empty := [k for k in passed if a.kind[k] == "car" and k not in getattr(self, "_spilled_now", ())]:
            self._event(a, "skip", empty, "skipped an empty green" if len(empty) == 1 else "skipped empty greens")
        elif any(a.kind[k] == "walk" for k in passed):
            self._event(a, "walk", [], "no one waiting to cross: walk skipped")
        r = a.pre_red[target]
        try:
            rem = tc.trafficlight.getNextSwitch(a.tls) - tc.simulation.getTime()
        except Exception:  # noqa: BLE001
            rem = 0.0
        tc.trafficlight.setPhase(a.tls, r)
        if rem > a.min_dur[r] + 1e-6:  # the all red the junction's last yellow needs, whichever all red this is
            tc.trafficlight.setPhaseDuration(a.tls, rem)  # (since 25 Oct the all reds differ: 1.5-3 s)
        a.phase = r
        if self.record:
            a.log.append((tc.simulation.getTime(), r))

    def _decide(self, tc, a: _Agent, now: float) -> None:
        ph = tc.trafficlight.getPhase(a.tls)
        if ph != a.phase:
            if a.kind[a.phase] in ("car", "walk"):
                a.last_end[a.phase] = now
            if a.kind[ph] in ("car", "walk"):
                a.green_since = now
                a.switches += 1
                a.passed_over[ph] = 0
            a.phase, a.entered, a.decided, a.blocked_since, a.stalled_since = ph, now, False, None, None
            a.held_for = None
            if self.record:
                a.log.append((now, ph))
        kind = a.kind[ph]
        if kind == "red" and _greens(a.states[ph]):
            return  # an all red some green goes on through: runs its time (never jumped from)
        if kind == "red":
            natural = a.next_green(ph)
            if not a.decided:
                a.decided = True
                target = self._choose(tc, a, natural, now)
                if target != natural and target in a.pre_red:
                    if self.requests.get(a.tls) and self._ev_target(a, natural) == target:
                        self.stats["ev_jumps"] += 1
                    self._jump(tc, a, natural, target)
            elif self.requests.get(a.tls):  # a request that came during the all-red
                target = self._ev_target(a, natural)
                if target is not None and target != natural and a.kind[natural] == "car" \
                        and a.passed_over.get(natural, 0) >= self.ev_starve_limit and self._demand(tc, a, natural, now):
                    self.stats["ev_starved"] += 1
                    target = None
                if target is not None and target != natural and target in a.pre_red:
                    self.stats["ev_jumps"] += 1
                    self._jump(tc, a, natural, target)
            return
        if kind != "car":
            return  # yellow runs its time; the walk is never cut
        self._green(tc, a, ph, now)

    def _green(self, tc, a: _Agent, ph: int, now: float) -> None:
        spent = now - a.green_since
        remaining = tc.trafficlight.getNextSwitch(a.tls) - now
        min_d, max_d = a.min_dur[ph], a.max_dur[ph]
        req = self.requests.get(a.tls)
        if req:
            if req & _greens(a.states[ph]):
                if spent < self.ev_hold_max_s and remaining < 1.0:
                    tc.trafficlight.setPhaseDuration(a.tls, 2.0)
                    self.stats["ev_holds"] += 1
                    who = self.request_by.get(a.tls)
                    if who != a.held_for:
                        a.held_for = who
                        self._event(a, "ambulance", [ph], "green held for an ambulance")
                return
            if spent >= min_d and self._ev_target(a, a.next_green(ph)) is not None:
                tc.trafficlight.setPhaseDuration(a.tls, 0)
                self.stats["ev_cuts"] += 1
                self._event(a, "ambulance", [ph], "green ended early: an ambulance is coming")
                return
        if spent < min_d:
            return
        others = [k for k in a.cycle if k != ph]
        if "noflow" in self.on:  # on every junction: where the cameras see too little, the loops tell
            if self._stalled(tc, a, ph, now):
                if a.stalled_since is None:
                    a.stalled_since = now
                if now - a.stalled_since >= self.noflow_s and any(self._demand(tc, a, k, now) for k in others):
                    tc.trafficlight.setPhaseDuration(a.tls, 0)
                    self.stats["noflow_cuts"] += 1
                    self._event(a, "noflow", [ph], "green ended: nobody could move")
                    return
            else:
                a.stalled_since = None
        if not a.covered:
            return
        if "spill" in self.on:
            if self._blocked(tc, a, ph):
                if a.blocked_since is None:
                    a.blocked_since = now
                if now - a.blocked_since >= self.spill_hold_s and any(self._demand(tc, a, k, now) for k in others):
                    tc.trafficlight.setPhaseDuration(a.tls, 0)
                    self.stats["spill_cuts"] += 1
                    self._event(a, "spill", [ph], "green ended: exit road full")
                    return
            else:
                a.blocked_since = None
        if "trickle" in self.on and spent >= self.trickle_min_s:
            n, q = self._approach(tc, a, ph)
            if q == 0 and n <= self.trickle_cur:
                best = max((self._approach(tc, a, k)[1] for k in others if a.kind[k] == "car"), default=0)
                if best >= self.trickle_q:
                    tc.trafficlight.setPhaseDuration(a.tls, 0)
                    self.stats["trickle_cuts"] += 1
                    self._event(a, "trickle", [ph], "green ended: only a trickle, a queue elsewhere")
                    return
        if "balance" in self.on and spent >= self.balance_min_s:
            w = self.forecast_w if self._risk else 0.0
            if self._fill(tc, a, ph) + w * self._risk_of(a, ph) < self.balance_low:
                full = [k for k in others if a.kind[k] == "car" and self._fill(tc, a, k) >= self.balance_high
                        and not self._blocked(tc, a, k)]
                ahead = [k for k in others if a.kind[k] == "car" and k not in full and not self._blocked(tc, a, k)
                         and self._fill(tc, a, k) + w * self._risk_of(a, k) >= self.balance_high]
                if full or ahead:
                    tc.trafficlight.setPhaseDuration(a.tls, 0)
                    self.stats["balance_cuts"] += 1
                    if full:
                        self._event(a, "balance", [ph], "green ended: a short road is filling up")
                    else:
                        self.stats["forecast_balance_cuts"] += 1
                        self._event(a, "forecast", [ph], "green ended: a long queue forecast on another road")
                    return
            elif w and self._fill(tc, a, ph) < self.balance_low and any(
                    a.kind[k] == "car" and self._fill(tc, a, k) >= self.balance_high and not self._blocked(tc, a, k)
                    for k in others):
                self.stats["forecast_balance_kept"] += 1  # (cut without the forecast: kept, its road is to queue)
        if self._risk and spent >= max_d - 1.0 and spent < self.extend_max_s - self.extend_step_s \
                and remaining < 1.0:
            r = self._risk_of(a, ph)
            self.stats["forecast_risk_at_max"] = max(self.stats.get("forecast_risk_at_max", 0.0), round(r, 3))
            why = None
            if r < self.forecast_extend_p:
                why = "low"
            elif any(self._exit_full(tc, o) for _, o in a.pairs[ph]):
                why = "exit"
            elif r <= max((self._risk_of(a, k) for k in others if a.kind[k] == "car"), default=0.0):
                why = "other"
            elif any(self._peds(a, k) for k in others):
                why = "peds"
            if why is not None and why != "low":
                self.stats[f"forecast_not_extended_{why}"] = self.stats.get(f"forecast_not_extended_{why}", 0) + 1
            if why is None:
                tc.trafficlight.setPhaseDuration(a.tls, self.extend_step_s)
                self.stats["forecast_extensions"] += 1
                if spent < max_d + 1.0:
                    self._event(a, "forecast", [ph], "green extended: a long queue forecast")
                return
        if "extend" in self.on and spent >= max_d - 1.0 and spent < self.extend_max_s - self.extend_step_s \
                and remaining < 1.0:
            f = self._fill(tc, a, ph)
            if f >= self.extend_fill and not any(self._exit_full(tc, o) for _, o in a.pairs[ph]):
                best = max((self._fill(tc, a, k) for k in others if a.kind[k] == "car"), default=0.0)
                # (nobody waiting to walk on another green: since 25 Oct the walks run with the car greens, no longer
                # in walk phases of their own, so the people waiting are those at the other greens' crossings)
                if f >= best + self.extend_margin and not any(self._peds(a, k) for k in others):
                    tc.trafficlight.setPhaseDuration(a.tls, self.extend_step_s)
                    self.stats["extensions"] += 1
                    self._event(a, "extend", [ph], "green extended: still a long queue")


class ObservedActuatedAgents(ActuatedAgents):
    """The same agents on the cameras' reports instead of SUMO's lanes."""

    name = "agents_observed"
    uses_cameras = True


def wire(pol, sumo, seed: int, more_on_run=None, cam_dir: Path | None = None) -> bool:
    """Connects the junction agents to the town as the viewer runs them (cosim/freefly.py) and the datasets record
    them (tools/goflow_prediction_dataset.py): the emergency vehicles on a run ask for green (the extras' and, through
    `more_on_run`, others such as the accidents' police cars); agents_observed reads the cameras' reports
    (camera_reports.SimulatedCameras: the 53 cameras' visible lane stretches with the detector's measured misses and
    false boxes, every 2 s), fed before the agents' step (sumo.pre_policy). False if `pol` is not a junction agent.
    (8 Nov: the dataset tool did not do this, and its agents_observed ran on no counts and no requests.)"""
    if pol is None or not hasattr(pol, "events"):
        return False

    def on_run() -> list:
        ex = sumo.extras
        out = [v for v in (ex.emergency if ex is not None else ()) if sumo.on_run(v)]
        if more_on_run is not None:
            out += list(more_on_run())
        return out

    pol.emergency = on_run
    if getattr(pol, "uses_cameras", False):
        from cosim.camera_reports import Layout, SimulatedCameras

        cam_dir = Path(cam_dir or ROOT / "out" / "town05_cameras")
        parked_file = cam_dir / "parked_static.json"
        parked = json.loads(parked_file.read_text(encoding="utf-8")) if parked_file.is_file() else None
        cams = SimulatedCameras(Layout.load(cam_dir), seed=seed, parked=parked,
                                siren_model="flash" if getattr(pol, "siren_flash", 0) else "vote")
        pol.visible = cams.layout.visible

        def feed(c) -> None:
            ex = sumo.extras
            reps = cams.step(c, c.simulation.getTime(),
                             {v: sumo.on_run(v) for v in (ex.emergency if ex is not None else ())})
            if reps is not None:
                pol.observe(reps)

        sumo.pre_policy = feed
    return True


def make(name: str):
    """ "agents", "agents_observed", optionally with features off ("agents_observed-skip-trickle"), the
    optional ones on ("agents+batch") and settings changed ("agents+noflow@noflow_s=6")."""
    base = re.split(r"[-+@]", name, maxsplit=1)[0]
    rest = name[len(base):]
    sets = re.findall(r"@(\w+)=([0-9.]+)", rest)
    rest = re.sub(r"@\w+=[0-9.]+", "", rest)
    off = set(re.findall(r"-(\w+)", rest))
    extra = set(re.findall(r"\+(\w+)", rest))
    cls = {"agents": ActuatedAgents, "agents_observed": ObservedActuatedAgents}[base]
    ctl = cls(off=off, extra=extra)
    for k, v in sets:  # a setting for a tuning run: "agents+noflow@noflow_s=6"
        if not hasattr(cls, k) or k.startswith("_"):
            raise ValueError(f"unknown agent setting {k}")
        cur = getattr(cls, k)
        setattr(ctl, k, type(cur)(float(v)) if isinstance(cur, (int, float)) else v)
    ctl.name = name
    return ctl

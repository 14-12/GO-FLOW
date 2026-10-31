"""The SUMO half of the co-simulation: runs the scenario exactly as the benchmark does and
reports, after every step, what CARLA has to draw.

SUMO is the master. Its step length stays the scenario's own (1 s for RESCO), its demand and
signal programs are the files in scenarios/, untouched; CARLA only renders. So every frame
of a CARLA video shows a state the benchmark simulation really passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import os
import traci
import traci.constants as tc

from cosim.controllers import _Conn
from cosim.controllers import make as make_controller
from cosim.extras import EXTRAS_TYPES_FILE, ExtrasConfig, TwinExtras
from cosim.sumo_tools import binary

_VEH_VARS = [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_SIGNALS, tc.VAR_ROAD_ID]
_PERSON_VARS = [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_ROAD_ID]
# what every step needs from the simulation and the signals, subscribed: sent back with the step's own answer
# instead of one round trip each (cosim/freefly.py: in the viewer every round trip cost ~5 ms, the
# thread winning Python's lock back from the others; tools/count_traci.py)
_SIM_VARS = [tc.VAR_TIME, tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS,
             tc.VAR_DEPARTED_PERSONS_IDS, tc.VAR_ARRIVED_PERSONS_IDS]


class WalkSkip:
    """The pedestrian phase of a signal (tools/build_town05_peds.py: every car red, every crossing walk,
    at the end of the cycle) is skipped when nobody is waiting to cross there: it costs the cars 15 s a
    cycle, which on Town05 at its design demand was enough to lock the town (measured, SUMO alone).
    Waiting: a person on one of the junction's walking areas whose next edge is one of its crossings."""

    def __init__(self, net_file: Path):
        import re

        self.cross: dict[str, list[int]] = {}
        pat = re.compile(r'<connection from="[^"]*" to=":([^"_]+)_c\d+"[^>]*tl="([^"]+)" linkIndex="(\d+)"')
        with open(net_file, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    self.cross.setdefault(m.group(2), []).append(int(m.group(3)))
        self.node = {}  # tls id -> junction id (the same on Town05)
        self.skipped = 0
        self.served = 0
        self.last_served: str | None = None
        self._last: dict[str, bool] = {}
        self._prev: dict[str, str] = {}
        self._clr: dict[str, bool] = {}
        self.cleared_early = 0
        # coordinated programs (cosim/coordination.py, programID "coordinated"): the time a skipped walk or an early
        # clearance frees goes to the next first green, so the cycle keeps its length and the green waves their step
        self._coord: dict[str, list[float] | None] = {}
        self.bank: dict[str, float] = {}
        self.banked = 0.0

    def _durations(self, c, tid: str) -> list[float] | None:
        """The phase durations of tid's program when it is a coordinated one, else None (asked once)."""
        if tid not in self._coord:
            durs = None
            try:
                prog = c.trafficlight.getProgram(tid)
                if prog == "coordinated":
                    for lg in c.trafficlight.getAllProgramLogics(tid):
                        if lg.programID == prog and int(lg.type) == 0:  # fixed time (coordinated-actuated: SUMO)
                            durs = [float(ph.duration) for ph in lg.phases]
            except Exception:  # noqa: BLE001
                durs = None
            self._coord[tid] = durs
        return self._coord[tid]

    def _pay_bank(self, c, tid: str) -> None:
        """At the first green of a coordinated cycle: the banked seconds added to it."""
        b = self.bank.get(tid, 0.0)
        if b <= 0.0:
            return
        try:
            if c.trafficlight.getPhase(tid) != 0:
                return
            rem = c.trafficlight.getNextSwitch(tid) - c.simulation.getTime()
            c.trafficlight.setPhaseDuration(tid, rem + b)
        except Exception:  # noqa: BLE001
            return
        self.banked += b
        self.bank[tid] = 0.0

    def is_walk(self, tid: str, state: str) -> bool:
        idx = self.cross.get(tid)
        if not idx or any(state[i] != "G" for i in idx):
            return False
        return all(ch == "r" for i, ch in enumerate(state) if i not in idx)

    def _clearance(self, c, tid: str, state: str, roads: dict[str, str], nxt) -> None:
        """Since 25 Oct (tools/build_town05_signals.py) people walk with the parallel traffic, and after the walk
        comes its clearance: the cars keep their green, nobody starts across, those on the crossing get over
        (crossing length / 1.2 m/s, up to 15 s). Once nobody is on those crossings it ends there (the cars go
        on to their yellow): a walk nobody used costs the cross street nothing. Since 27 Oct the walk has a phase
        of its own (every car red) and its clearance is all red: it ends the same way once the crossings are empty
        (nobody starts during it: PedRedHold), then the 2 s buffer runs, as a crossing with on-crossing detectors
        (UK Puffin) does."""
        idx = self.cross.get(tid) or []
        prev = self._prev.get(tid)
        self._prev[tid] = state
        if prev is None or len(prev) != len(state):
            return
        if prev != state:
            ended = [i for i in idx if prev[i] in "Gg" and state[i] in "ru"]
            named = False
            if ended:  # (27 Oct: also after the walk of its own, every car red, the all red of its clearance)
                try:
                    named = c.trafficlight.getPhaseName(tid) == "walk clearance"
                except Exception:  # noqa: BLE001
                    named = False
            self._clr[tid] = named
        if not self._clr.get(tid):
            return
        if any(road.startswith(f":{tid}_c") for road in roads.values()):
            return  # someone is still on a crossing there
        try:
            if self._durations(c, tid) is not None:
                self.bank[tid] = self.bank.get(tid, 0.0) + max(
                    0.0, c.trafficlight.getNextSwitch(tid) - c.simulation.getTime())
            c.trafficlight.setPhaseDuration(tid, 0.0)
        except Exception:  # noqa: BLE001
            return
        self._clr[tid] = False
        self.cleared_early += 1
        if nxt is not None:
            try:
                nxt[tid] = c.trafficlight.getNextSwitch(tid)
            except Exception:  # noqa: BLE001
                pass

    def step(self, c, tls: dict[str, str], roads: dict[str, str], nxt: dict[str, float] | None = None) -> None:
        """A walk phase nobody waits for is skipped; its signal's state in `tls` (and next switch in `nxt`) is
        then asked again, so what is shown never has the skipped walk (the free-fly lamps drew it green).
        A walk clearance nobody needs ends at once (_clearance)."""
        for tid, state in list(tls.items()):
            walk = self.is_walk(tid, state)
            if walk and not self._last.get(tid):
                waiting = False
                for pid, road in roads.items():
                    if road.startswith(f":{tid}_w"):
                        try:
                            if c.person.getNextEdge(pid).startswith(f":{tid}_c"):
                                waiting = True
                                break
                        except Exception:  # noqa: BLE001 - gone
                            pass
                if waiting:
                    self.served += 1
                    self.last_served = tid
                else:
                    durs = self._durations(c, tid)
                    if durs is not None:  # the walk and what follows it to the end of the cycle
                        try:
                            k = c.trafficlight.getPhase(tid)
                            left = c.trafficlight.getNextSwitch(tid) - c.simulation.getTime()
                            self.bank[tid] = self.bank.get(tid, 0.0) + left + sum(durs[k + 1:])
                        except Exception:  # noqa: BLE001
                            pass
                    c.trafficlight.setPhase(tid, 0)
                    self.skipped += 1
                    walk = False
                    tls[tid] = c.trafficlight.getRedYellowGreenState(tid)
                    if nxt is not None:
                        nxt[tid] = c.trafficlight.getNextSwitch(tid)
            self._last[tid] = walk
            self._clearance(c, tid, tls[tid], roads, nxt)  # (after a skipped walk: the state now shown)
            if self.bank.get(tid, 0.0) > 0.0:
                self._pay_bank(c, tid)
                if nxt is not None:
                    try:
                        nxt[tid] = c.trafficlight.getNextSwitch(tid)
                    except Exception:  # noqa: BLE001
                        pass


class CycleKeeper:
    """Runs the coordinated signals (programID "coordinated", fixed-time programs from cosim/coordination.py) as
    coordinated-actuated controllers do (27 Oct):

    * the cycle is kept: each signal's first green (phase 0, the green its green wave is built on) ends once a cycle
      at the second of the cycle it ended at the start. When it starts early (a green before it ended early, a walk
      nobody wanted was skipped) it is held green until then; when it starts late (an emergency vehicle held a green)
      it is shortened, never below KEEP_MIN_S. So the signals stay in step with each other whatever happens;
    * the other greens end early once they have had their minimum (the program's MIN_GREEN) and nobody is coming on
      their lanes within GAP_M of the line: their spare seconds go round to phase 0. They never run longer than
      planned."""

    KEEP_MIN_S = 10.0
    GAP_M = 35.0          # about 2.5 s at 50 km/h: the gap that ends a green with no one coming
    MIN_GREEN = (12.0, 7.0)  # (a green with straight on, one of turns only): cosim/coordination.py's minimums

    def __init__(self):
        self.ref: dict[str, float] = {}       # when phase 0 ended on time once
        self.cycle: dict[str, tuple[float, float]] = {}   # (cycle, phase 0's duration); (0, 0): not coordinated
        self.lanes: dict[str, dict[int, list[str]]] = {}  # tls -> green phase -> incoming lanes green in it
        self.mins: dict[str, dict[int, float]] = {}
        self._phase: dict[str, int] = {}
        self.corrections = 0
        self.corrected_s = 0.0
        self.gap_outs = 0

    def watch(self, c, tid: str) -> bool:
        if tid not in self.cycle:
            cyc = (0.0, 0.0)
            try:
                prog = c.trafficlight.getProgram(tid)
                if prog == "coordinated":
                    for lg in c.trafficlight.getAllProgramLogics(tid):
                        if lg.programID == prog and int(lg.type) == 0:
                            cyc = (sum(float(ph.duration) for ph in lg.phases), float(lg.phases[0].duration))
                            links = c.trafficlight.getControlledLinks(tid)
                            lanes, mins = {}, {}
                            greens = [k for k, ph in enumerate(lg.phases) if getattr(ph, "name", "") == "green"]

                            def lanes_on(state):
                                return {links[i][0][0] for i, ch in enumerate(state) if ch in "Gg" and i < len(links)
                                        and links[i] and not links[i][0][0].startswith(":")}

                            for n, k in enumerate(greens):
                                if k == 0:
                                    continue
                                ph = lg.phases[k]
                                on = [i for i, ch in enumerate(ph.state) if ch in "Gg" and i < len(links) and links[i]
                                      and not links[i][0][0].startswith(":")]
                                # the lanes whose green ends with this phase (a lane that goes on green in the next
                                # green keeps moving whatever this one does): lead-lag's single-approach greens end
                                # once that approach's left lane is empty
                                nxt = lg.phases[greens[(n + 1) % len(greens)]].state
                                ending = sorted({links[i][0][0] for i in on} - lanes_on(nxt))
                                lanes[k] = ending or sorted({links[i][0][0] for i in on})
                                straight = any(self._dir(c, links[i][0]) == "s" for i in on)
                                mins[k] = self.MIN_GREEN[0] if straight else self.MIN_GREEN[1]
                            self.lanes[tid], self.mins[tid] = lanes, mins
            except Exception:  # noqa: BLE001
                cyc = (0.0, 0.0)
            self.cycle[tid] = cyc
        return self.cycle[tid][0] > 0

    @staticmethod
    def _dir(c, link) -> str:
        frm, to, _via = link
        try:
            for ln in c.lane.getLinks(frm):
                if ln[0] == to:
                    return {"s": "s", "l": "l", "L": "l", "r": "r", "R": "r", "t": "l"}.get(ln[6], "s")
        except Exception:  # noqa: BLE001
            pass
        return "s"

    def _nobody_coming(self, c, lanes: list[str]) -> bool:
        for ln in lanes:
            try:
                length = c.lane.getLength(ln)
                for v in c.lane.getLastStepVehicleIDs(ln):
                    if length - c.vehicle.getLanePosition(v) <= self.GAP_M:
                        return False
            except Exception:  # noqa: BLE001
                return False
        return True

    def step(self, c, now: float, tls: dict[str, str], prev: dict[str, str], gap_check: bool = True) -> None:
        for tid, st in tls.items():
            if not self.watch(c, tid):
                continue
            changed = prev.get(tid) != st
            if not changed and not gap_check:
                continue
            try:
                ph = c.trafficlight.getPhase(tid)
            except Exception:  # noqa: BLE001
                continue
            was = self._phase.get(tid)
            self._phase[tid] = ph
            if ph in self.lanes.get(tid, {}) and gap_check:  # a green other than the first: gap out
                try:
                    spent = c.trafficlight.getSpentDuration(tid)
                    if spent >= self.mins[tid][ph] and c.trafficlight.getNextSwitch(tid) - now > 0.6 \
                            and self._nobody_coming(c, self.lanes[tid][ph]):
                        c.trafficlight.setPhaseDuration(tid, 0.0)
                        self.gap_outs += 1
                except Exception:  # noqa: BLE001
                    pass
                continue
            if ph != 0 or was == 0:
                continue
            C, _d0 = self.cycle[tid]
            try:
                rem = c.trafficlight.getNextSwitch(tid) - now
            except Exception:  # noqa: BLE001
                continue
            end = now + rem
            if tid not in self.ref:
                self.ref[tid] = end
                continue
            e = (end - self.ref[tid]) % C
            late = e if e <= C / 2 else e - C       # > 0: ends late, < 0: early
            if abs(late) < 0.75:
                continue
            new = max(self.KEEP_MIN_S, rem - late) if late > 0 else rem - late
            try:
                c.trafficlight.setPhaseDuration(tid, new)
            except Exception:  # noqa: BLE001
                continue
            self.corrections += 1
            self.corrected_s += abs(new - rem)


class PedRedHold:
    """People wait at the kerb while their crossing shows DON'T WALK (27 Oct). SUMO's walkers do step onto a red
    crossing now and then (Town05, x0.8, 15 min: 174 of 321 crossings started on red, both directions;
    tools/diag_junction_stops.py), and a car whose green it is then stops in the junction for them. Those standing on
    a signalised junction's corner (walking area) with a red crossing next on their way are held there (max speed
    near 0) and let go when it turns WALK; anyone already on the crossing walks on."""

    HOLD_SPEED = 0.001

    def __init__(self, net_file: Path):
        import re

        self.idx: dict[str, tuple[str, int]] = {}  # crossing edge -> (tls, link index)
        pat = re.compile(r'<connection from="[^"]*" to="(:[^"]+_c\d+)"[^>]*tl="([^"]+)" linkIndex="(\d+)"')
        with open(net_file, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    self.idx[m.group(1)] = (m.group(2), int(m.group(3)))
        self.held: dict[str, float] = {}  # person -> max speed before
        self.holds = 0

    def step(self, c, tls: dict[str, str], roads: dict[str, str]) -> None:
        for pid in list(self.held):
            if pid not in roads:
                self.held.pop(pid)
        for pid, road in roads.items():
            hold = False
            if road.startswith(":") and "_w" in road:
                try:
                    nxt = c.person.getNextEdge(pid)
                except Exception:  # noqa: BLE001 - gone
                    nxt = ""
                ti = self.idx.get(nxt)
                if ti is not None:
                    st = tls.get(ti[0], "")
                    hold = ti[1] < len(st) and st[ti[1]] not in "Gg"
            try:
                if hold and pid not in self.held:
                    self.held[pid] = c.person.getMaxSpeed(pid)
                    c.person.setSpeed(pid, self.HOLD_SPEED)
                    self.holds += 1
                elif not hold and pid in self.held:
                    c.person.setSpeed(pid, self.held.pop(pid))
            except Exception:  # noqa: BLE001
                self.held.pop(pid, None)


class EvCaution:
    """An emergency vehicle on a run (types "*_run": SUMO lets it drive into a junction against red at up to
    20 km/h, jmDriveAfterRedTime) goes through a red only when that junction is clear: nobody inside it and nobody
    coming up to it on a green (within 30 m and moving). Otherwise it stops at the line and waits there, as a driver
    with blue lights does. SUMO's red-runner does not look out for the cross traffic, nor does the cross traffic
    for it: since the opposite approaches share their green (25 Oct, tools/build_town05_signals.py) there is
    more of it, and 3-5 vehicles a run met one inside a junction (tools/check_highway.py, 20 min)."""

    LOOK_M = 30.0     # the red ahead within this: is the junction clear?
    STOP_M = 2.5      # where it waits: this far before the line

    def __init__(self):
        self.held: set[str] = set()
        self.vias: dict[str, list] = {}
        self.waits = 0

    def _clear(self, c, tls: str, state: str) -> bool:
        if tls not in self.vias:
            links = c.trafficlight.getControlledLinks(tls)
            self.vias[tls] = [(i, ls[0][0], ls[0][2]) for i, ls in enumerate(links) if ls]
        for i, frm, via in self.vias[tls]:
            if via and c.lane.getLastStepVehicleNumber(via) > 0:
                if any(not v.startswith("ped:") for v in c.lane.getLastStepVehicleIDs(via)):
                    return False  # someone in the junction
            if i < len(state) and state[i] in "Gg" and not frm.startswith(":"):
                flen = c.lane.getLength(frm)
                for v in c.lane.getLastStepVehicleIDs(frm):
                    if flen - c.vehicle.getLanePosition(v) < self.LOOK_M and c.vehicle.getSpeed(v) > 1.0:
                        return False  # someone coming on its green
        return True

    def step(self, c, vehicles: dict, tls_states: dict) -> None:
        for vid, v in vehicles.items():
            if v.vclass not in ("emergency", "authority"):
                continue
            try:
                if "_run" not in c.vehicle.getTypeID(vid):
                    if vid in self.held:
                        self.held.discard(vid)
                        c.vehicle.setSpeed(vid, -1)
                    continue
                nxt = c.vehicle.getNextTLS(vid)[:1]
            except Exception:  # noqa: BLE001
                continue
            hold = False
            if nxt:
                tls, idx, dist, st = nxt[0]
                if st in "rRyu" and dist < self.LOOK_M:
                    hold = not self._clear(c, tls, tls_states.get(tls, ""))
            try:
                if hold:
                    if vid not in self.held:
                        self.waits += 1
                    self.held.add(vid)
                    c.vehicle.slowDown(vid, 0.0 if dist < self.STOP_M + 6.0 else max(dist - self.STOP_M, 0.0) / 2.0, 0.5)
                elif vid in self.held:
                    self.held.discard(vid)
                    c.vehicle.setSpeed(vid, -1)
            except Exception:  # noqa: BLE001
                self.held.discard(vid)


class BoxNudge:
    """Cars standing inside a junction waiting for each other: a deadlock of SUMO's junction model (each yields to
    one it is in the way of). At Town05's junction 979 (between the police station and the hospital) five or six
    cars stood in its box for good and the town locked behind them (tools/eval_town05_baseline.py, x1.0 seed 15,
    x0.8 seed 12, every controller). As a driver would, one standing there BOX_STUCK_S edges through: for NUDGE_S
    it no longer waits for the cars already in the junction (SUMO speed mode bit 5), then drives as before. The
    viewer did this already (cosim/incidents.IncidentManager.unjam); here it is part of the traffic for every run
    and every controller."""

    BOX_STUCK_S = 40.0
    NUDGE_S = 6.0

    def __init__(self):
        self.since: dict[str, float] = {}
        self.nudged: dict[str, float] = {}
        self.count = 0

    def step(self, c, now: float, subs: dict) -> None:
        for vid, until in list(self.nudged.items()):
            if now >= until:
                try:
                    c.vehicle.setSpeedMode(vid, 31)
                except Exception:  # noqa: BLE001 - gone
                    pass
                del self.nudged[vid]
        seen = set()
        for vid, r in subs.items():
            road = r.get(tc.VAR_ROAD_ID, "")
            if not road.startswith(":") or r.get(tc.VAR_SPEED, 1.0) >= 0.1 or "_w" in road or "_c" in road:
                continue
            seen.add(vid)
            t0 = self.since.setdefault(vid, now)
            if now - t0 >= self.BOX_STUCK_S and vid not in self.nudged:
                try:
                    if c.vehicle.getSpeedMode(vid) != 31:
                        continue  # someone else drives it (an accident's vehicle, a held one)
                    c.vehicle.setSpeedMode(vid, 31 | 32)
                except Exception:  # noqa: BLE001
                    continue
                self.nudged[vid] = now + self.NUDGE_S
                self.since[vid] = now
                self.count += 1
        for vid in list(self.since):
            if vid not in seen:
                del self.since[vid]


@dataclass
class Vehicle:
    id: str
    vclass: str
    type_id: str
    length: float
    x: float
    y: float
    angle: float
    speed: float
    signals: int
    lane_angle: float | None = None  # direction of its lane where it is (SumoSide(lane_geometry=...))


@dataclass
class Person:
    id: str
    x: float
    y: float
    angle: float
    speed: float
    road: str = ""  # SUMO edge (":<junction>_c<n>" a crossing, ":<junction>_w<n>" a walking area)


@dataclass
class StepResult:
    time: float
    vehicles: dict[str, Vehicle]
    arrived: list[str]
    tls: dict[str, str] = field(default_factory=dict)
    persons: dict[str, Person] = field(default_factory=dict)
    persons_arrived: list[str] = field(default_factory=list)
    tls_next: dict[str, float] = field(default_factory=dict)  # sim time of each signal's next switch (when known)


class SumoSide:
    """Owns the TraCI connection. `controller`: "default" (the network's own programs),
    "webster" or "actuated" (programs from scenarios/<net>/<net>.<controller>.add.xml), or a
    per-step controller from cosim/controllers.py ("max_pressure"), called after every step."""

    def __init__(self, scenario_dir: Path, controller: str = "webster", gui: bool = False,
                 port: int = 8813, seed: int = 42, extra_args: list[str] | None = None,
                 label: str = "cosim", policy=None, extras: bool = False, emergency_every: float | None = None,
                 additional: list[str] | None = None, vehicle_vars: list[int] | None = None,
                 person_vars: list[int] | None = None, extras_types: Path | None = None,
                 lane_geometry=None):
        """extras: add the main project's pedestrians and emergency vehicles (cosim/extras.py);
        emergency_every overrides its mean interval between emergency vehicles (75 s).
        vehicle_vars / person_vars: more variables to subscribe for every vehicle / person, read from
        `vehicle_subs` / `person_subs` (this step's raw subscription results).
        extras_types: the extras' vehicle and person types from this file instead of scenarios/twin_extras.add.xml
        (cosim/vmodels.extras_types_file: sized to their CARLA models).
        lane_geometry: a cosim/lanegeo.LaneGeometry: each Vehicle then has its lane's direction (lane_angle)."""
        scenario_dir = Path(scenario_dir)
        name = scenario_dir.name
        cfg = scenario_dir / f"{name}.sumocfg"
        cmd = [binary("sumo-gui" if gui else "sumo"), "-c", str(cfg), "--seed", str(seed),
               "--no-step-log", "true", "--no-warnings", "true",
               # Teleporting off, as in every benchmark run of the main project.
               "--time-to-teleport", "-1"]
        add = scenario_dir / f"{name}.{controller}.add.xml"
        self.policy = policy or make_controller(controller)  # per-step controller, or None
        self.pre_policy = None  # called with the connection after each SUMO step, before the controller
        adds = []
        if controller != "default" and self.policy is None:
            if not add.is_file():
                raise FileNotFoundError(f"no {controller} programs for {name}: {add}")
            adds.append(str(add))
        program = getattr(self.policy, "program", None)  # a controller that runs on top of a program file
        if program:
            if not add.with_name(f"{name}.{program}.add.xml").is_file():
                raise FileNotFoundError(f"no {program} programs for {name}")
            adds.append(str(add.with_name(f"{name}.{program}.add.xml")))
        elif getattr(self.policy, "fallback", None) and add.with_name(f"{name}.actuated.add.xml").is_file():
            adds.append(str(add.with_name(f"{name}.actuated.add.xml")))  # agents hand weak junctions back to it
        if extras:
            adds.append(str(extras_types or EXTRAS_TYPES_FILE))
        adds += [str(a) for a in additional or []]  # e.g. bus stops (cosim/freefly.py)
        if adds:
            cmd += ["-a", ",".join(adds)]  # SUMO takes one -a, comma-separated
        cmd += extra_args or []
        traci.start(cmd, port=port, label=label, traceFile=os.environ.get("TRACI_TRACE") or None)
        self.conn = traci.getConnection(label)
        self.controller = controller
        self.step_length = self.conn.simulation.getDeltaT()
        self.tls_ids = list(self.conn.trafficlight.getIDList())
        self.lane_geometry = lane_geometry
        extra = list(vehicle_vars or []) + ([tc.VAR_LANE_ID, tc.VAR_LANEPOSITION] if lane_geometry is not None else [])
        self._veh_vars = _VEH_VARS + [v for v in dict.fromkeys(extra) if v not in _VEH_VARS]
        self._person_vars = _PERSON_VARS + [v for v in person_vars or [] if v not in _PERSON_VARS]
        self.vehicle_subs: dict = {}
        self.person_subs: dict = {}
        self.conn.simulation.subscribe(_SIM_VARS)
        for t in self.tls_ids:
            self.conn.trafficlight.subscribe(t, [tc.TL_RED_YELLOW_GREEN_STATE, tc.TL_NEXT_SWITCH])
        # tls the controller has a program for; the others keep the network's own.
        self.controlled: list[str] = []
        if self.policy is not None:
            self.policy.setup(_Conn(self.conn))
            self.controlled = list(self.tls_ids)
        elif controller != "default":
            for t in self.tls_ids:
                if any(lg.programID == controller for lg in self.conn.trafficlight.getAllProgramLogics(t)):
                    self.conn.trafficlight.setProgram(t, controller)
                    self.controlled.append(t)
            if not self.controlled:
                raise ValueError(f"{add.name} has no program '{controller}' for any signal of {name}")
        self._static: dict[str, tuple[str, str, float]] = {}
        self.ev_caution = EvCaution()
        self.box_nudge = BoxNudge()
        self.walk = WalkSkip(scenario_dir / f"{name}.net.xml")
        if not self.walk.cross:
            self.walk = None
        self.ped_hold = PedRedHold(scenario_dir / f"{name}.net.xml") if self.walk is not None else None
        self.keeper = CycleKeeper() if controller == "coordinated" else None
        self._tls_prev: dict[str, str] = {}
        self.extras = None
        if extras:
            cfg = ExtrasConfig(seed=seed)
            if emergency_every:
                cfg.emergency_every = emergency_every
            self.extras = TwinExtras(cfg)
            self.extras.setup(scenario_dir / f"{name}.net.xml", self.time)

    @property
    def time(self) -> float:
        return self.conn.simulation.getTime()

    def step(self, run_policy: bool = True) -> StepResult:
        """One SUMO step. run_policy=False leaves the per-step controller to the caller (to
        call after something has been measured on the new state, e.g. camera counts)."""
        c = self.conn
        c.simulationStep()
        sim = c.simulation.getSubscriptionResults()
        ran_policy = self.policy is not None and run_policy
        if ran_policy:
            if self.pre_policy is not None:  # e.g. the cameras' reports for the agents (cosim/freefly.py)
                self.pre_policy(c)
            self.policy.step(_Conn(c))
        for vid in sim[tc.VAR_DEPARTED_VEHICLES_IDS]:
            c.vehicle.subscribe(vid, self._veh_vars)
            self._static[vid] = (c.vehicle.getVehicleClass(vid), c.vehicle.getTypeID(vid),
                                 c.vehicle.getLength(vid))
        arrived = list(sim[tc.VAR_ARRIVED_VEHICLES_IDS])
        for vid in arrived:
            self._static.pop(vid, None)
        subs = self.vehicle_subs = c.vehicle.getAllSubscriptionResults()
        vehicles = {}
        for vid, r in subs.items():
            if vid not in self._static:
                continue
            vclass, type_id, length = self._static[vid]
            x, y = r[tc.VAR_POSITION]
            la = None
            if self.lane_geometry is not None and tc.VAR_LANE_ID in r:
                la = self.lane_geometry.angle(r[tc.VAR_LANE_ID], r.get(tc.VAR_LANEPOSITION, 0.0))
            vehicles[vid] = Vehicle(vid, vclass, type_id, length, x, y, r[tc.VAR_ANGLE],
                                    r[tc.VAR_SPEED], r[tc.VAR_SIGNALS], la)
        tls_next: dict[str, float] = {}
        if ran_policy:  # the controller may have just changed a signal: ask
            tls = {t: c.trafficlight.getRedYellowGreenState(t) for t in self.tls_ids}
        else:  # as they were at the end of the step, as asking would give
            tl_subs = c.trafficlight.getAllSubscriptionResults()
            tls = {t: tl_subs[t][tc.TL_RED_YELLOW_GREEN_STATE] for t in self.tls_ids}
            tls_next = {t: tl_subs[t][tc.TL_NEXT_SWITCH] for t in self.tls_ids}
        now = sim[tc.VAR_TIME]
        persons, persons_arrived, roads = {}, [], {}
        if self.extras is not None:
            self.extras.step(c, now)
            for pid in sim[tc.VAR_DEPARTED_PERSONS_IDS]:
                c.person.subscribe(pid, self._person_vars)
            persons_arrived = list(sim[tc.VAR_ARRIVED_PERSONS_IDS])
            self.person_subs = c.person.getAllSubscriptionResults()
            for pid, r in self.person_subs.items():
                if pid in persons_arrived or tc.VAR_POSITION not in r:
                    continue
                x, y = r[tc.VAR_POSITION]
                roads[pid] = r.get(tc.VAR_ROAD_ID, "")
                persons[pid] = Person(pid, x, y, r[tc.VAR_ANGLE], r[tc.VAR_SPEED], roads[pid])
        if self.walk is not None:
            self.walk.step(c, tls, roads, tls_next if tls_next else None)
        if self.ped_hold is not None and roads:
            self.ped_hold.step(c, tls, roads)
        if self.keeper is not None:
            self.keeper.step(c, now, tls, self._tls_prev, gap_check=int(now * 2) % 2 == 0)  # gaps: every second
            self._tls_prev = dict(tls)
        self.ev_caution.step(c, vehicles, tls)
        if int(now * 2) % 2 == 0:  # every second
            self.box_nudge.step(c, now, subs)
        return StepResult(now, vehicles, arrived, tls, persons, persons_arrived, tls_next)

    def on_run(self, vid: str) -> bool:
        """An emergency vehicle on a run (blue lights on): the extras' ground truth."""
        return bool(self.extras and self.extras.on_run(vid))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

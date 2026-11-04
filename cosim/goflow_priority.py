"""GO-FLOW priority agent on top of the junction agents (agents/priority_agent, integration plan of 4 Nov).

    --controller agents_goflow            the junction agents (exact counts), emergency requests planned by GO-FLOW
    --controller agents_observed_goflow   the same on the cameras' reports
    (the agents' suffixes work too: "agents_observed_goflow-trickle@horizon=...")

The junction agents (cosim/agents_actuated.py) stay in charge of every signal: skip, spill, trickle, balance,
pedcall, the walk rules, the program's own states. Only where their pre-emption requests come from changes: instead
of "the next signal within 150 m, the one after within 300 m and 25 s", a child process (the GO-FLOW coordinator, in
its own Python environment) plans every signal an emergency vehicle on a run will reach within its horizon,
arbitrates between vehicles that need conflicting greens (ambulance before police before a recovery truck, then
first come), and refuses what its safety layer refuses (siren on and confirmed, at most 3 pre-emptions per junction
in 300 s except for ambulances). The agents then execute those requests with their own guards (_can_pass,
ev_patience_s, stall_s, ev_starve_limit), as they do their own.

The child speaks JSON lines on its stdin/stdout (agents/priority_agent/town05/serve.py). Two modes:
  lockstep (default, SUMO-only evaluation): each state is answered before the step goes on (5 s timeout);
  live (the viewer: GOFLOW_PRIORITY_MODE=live): sending never blocks the SUMO thread; a plan older than 2 s of
  simulation is not used.
Whenever there is no valid plan (slow, crashed, not installed) the agents use their own requests for that step, so
the town behaves exactly as under "agents" / "agents_observed". The child is restarted after 10, 30 and 90 s, then
left off for the run.

Interpreter: GOFLOW_PRIORITY_PY, else agents/priority_agent/.venv (tools/setup_goflow_priority.bat).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from cosim.agents_actuated import ActuatedAgents, ObservedActuatedAgents, _greens

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "priority_agent"
LOG_DIR = ROOT / "tools" / "logs"
EMERGENCY_MODES = ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION")
RESTART_AFTER_S = (10.0, 30.0, 90.0)


def agent_python() -> str:
    env = os.environ.get("GOFLOW_PRIORITY_PY")
    if env:
        return env
    venv = AGENT_DIR / ".venv"
    return str(venv / "Scripts" / "python.exe") if os.name == "nt" else str(venv / "bin" / "python")


class GoflowClient:
    """The child process and its pipes. Never raises into the caller."""

    def __init__(self, mode: str = "lockstep", extra_args: list[str] | None = None, timeout_s: float = 5.0,
                 python: str | None = None):
        self.mode = mode
        self.extra_args = extra_args or []
        self.timeout_s = timeout_s
        self.python = python or agent_python()
        self.proc: subprocess.Popen | None = None
        self.hello: dict | None = None
        self.ready: dict | None = None
        self.plan: dict | None = None
        self.status = "off"
        self.restarts = 0
        self.down_since: float | None = None
        self._got = threading.Event()
        self._out: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._last_err_log = 0.0
        self.stats = {"sent": 0, "plans": 0, "errors": 0, "timeouts": 0, "dropped": 0, "restarts": 0}
        atexit.register(self.close)

    # ------------------------------------------------------------------ process
    def start(self, hello: dict) -> bool:
        self.hello = hello
        return self._spawn()

    def _spawn(self) -> bool:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [self.python, "-m", "town05.serve", *self.extra_args]
        try:
            err = (LOG_DIR / "goflow_priority.log").open("a", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, cwd=str(AGENT_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=err, text=True, encoding="utf-8", bufsize=1)
        except Exception as e:  # noqa: BLE001 - not installed: the agents run on their own
            self._down(f"cannot start {cmd[0]}: {e}")
            return False
        threading.Thread(target=self._reader, daemon=True, name="goflow-reader").start()
        if self.mode == "live":
            threading.Thread(target=self._writer, daemon=True, name="goflow-writer").start()
        self._got.clear()
        if not self._write(self.hello) or not self._got.wait(self.timeout_s * 3) or self.ready is None:
            self._down("no answer to hello")
            return False
        self.status = "ok"
        self.down_since = None
        log.info("GO-FLOW priority agent %s (%s) ready: %s", self.ready.get("version"), self.mode,
                 self.ready.get("config"))
        return True

    def _down(self, why: str) -> None:
        if self.status != "down":
            log.warning("GO-FLOW priority agent down: %s (the junction agents use their own requests)", why)
        self.status = "down"
        self.down_since = self.down_since or time.monotonic()
        self.plan = None
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def maybe_restart(self) -> None:
        if self.status != "down" or self.down_since is None or self.restarts >= len(RESTART_AFTER_S):
            return
        if time.monotonic() - self.down_since >= RESTART_AFTER_S[self.restarts]:
            self.restarts += 1
            self.stats["restarts"] += 1
            self.down_since = time.monotonic()
            log.info("GO-FLOW priority agent: restart %d", self.restarts)
            self._spawn()

    # ------------------------------------------------------------------ pipes
    def _write(self, msg: dict) -> bool:
        try:
            with self._lock:
                self.proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
                self.proc.stdin.flush()
            return True
        except Exception as e:  # noqa: BLE001 - broken pipe: the child died
            self._down(f"write failed: {e}")
            return False

    def _writer(self) -> None:
        proc = self.proc
        while proc is self.proc and proc.poll() is None:
            try:
                msg = self._out.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self._write(msg):
                return

    def _reader(self) -> None:
        proc = self.proc
        try:
            for line in proc.stdout:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                kind = m.get("type")
                if kind == "ready":
                    self.ready = m
                elif kind == "plan":
                    self.plan = m
                    self.stats["plans"] += 1
                elif kind == "error":
                    self.stats["errors"] += 1
                    now = time.monotonic()
                    if now - self._last_err_log > 10.0:
                        self._last_err_log = now
                        log.warning("GO-FLOW priority agent error in %s: %s", m.get("where"), m.get("msg"))
                elif kind == "log":
                    log.info("GO-FLOW: %s", m.get("msg"))
                self._got.set()
        except Exception:  # noqa: BLE001
            pass
        if proc is self.proc:
            self._down("process ended")

    # ------------------------------------------------------------------ use
    def exchange(self, state: dict) -> None:
        """Send a state. Lockstep: wait for its answer (plan or error). Live: queue it, never wait."""
        if self.status != "ok":
            self.maybe_restart()
            if self.status != "ok":
                return
        self.stats["sent"] += 1
        if self.mode == "live":
            try:
                self._out.put_nowait(state)
            except queue.Full:
                try:
                    self._out.get_nowait()
                    self.stats["dropped"] += 1
                except queue.Empty:
                    pass
                self._out.put_nowait(state)
            return
        self._got.clear()
        if self._write(state) and not self._got.wait(self.timeout_s):
            self.stats["timeouts"] += 1

    def valid_plan(self, now: float, max_age_s: float) -> dict | None:
        p = self.plan
        if self.status != "ok" or p is None:
            return None
        if now - float(p.get("t", -1e9)) > max_age_s:
            return None
        return p

    def close(self) -> None:
        p = self.proc
        if p is None:
            return
        self.proc = None
        try:
            if p.poll() is None:
                with self._lock:
                    p.stdin.write('{"type":"bye"}\n')
                    p.stdin.flush()
                p.wait(timeout=3)
        except Exception:  # noqa: BLE001
            pass
        try:
            if p.poll() is None:
                p.kill()
        except Exception:  # noqa: BLE001
            pass


class GoflowMixin:
    """Mixed in before ActuatedAgents: replaces _requests() by GO-FLOW's plan when there is a valid one."""

    goflow_send_every_s = 1.0
    goflow_heartbeat_s = 5.0
    goflow_max_age_s = 2.0

    def setup(self, conn) -> None:
        super().setup(conn)
        tc = conn.tc
        self.goflow_role = getattr(self, "goflow_role", None) or (lambda vid: None)
        self._ems_lanes: set[str] = set()
        self._gf_last_send = -1e9
        self._gf_last_set: frozenset = frozenset()
        self._gf_seq = 0
        self._gf_announced: set[tuple[str, str, str]] = set()
        self.goflow_stats = {"plan_steps": 0, "fallback_steps": 0, "requests": 0, "refused": 0}
        junctions = {}
        for tls, a in self.junctions.items():
            greens = [sorted(_greens(a.states[k])) for k in a.cycle if a.kind[k] == "car"]
            junctions[tls] = {"n_links": len(a.states[0]) if a.states else 0, "greens": greens}
        mode = os.environ.get("GOFLOW_PRIORITY_MODE", "lockstep")
        args = os.environ.get("GOFLOW_PRIORITY_ARGS", "").split()
        confirm = os.environ.get("GOFLOW_PRIORITY_CONFIRM", "camera" if self.uses_cameras else "v2i")
        if confirm == "camera" and not self.uses_cameras:
            confirm = "v2i"  # no camera reports without agents_observed
        run_id = os.environ.get("GOFLOW_PRIORITY_RUN") or \
            f"{time.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^A-Za-z0-9_.-]', '_', self.name)}_{os.getpid()}"
        self.goflow = GoflowClient(mode, args)
        hello = {"type": "hello", "v": 1, "run_id": run_id, "step_s": tc.simulation.getDeltaT(), "mode": mode,
                 "confirm": confirm, "junctions": junctions}
        self.goflow.start(hello)

    @property
    def goflow_status(self) -> str:
        cl = getattr(self, "goflow", None)
        if cl is None or cl.status == "off":
            return "off"
        now = getattr(self, "_now", None)
        if cl.status == "ok" and cl.plan is not None and now is not None \
                and now - float(cl.plan.get("t", -1e9)) > self.goflow_max_age_s:
            return "stale"
        return cl.status

    def observe(self, reports: list[dict]) -> None:
        super().observe(reports)
        self._ems_lanes = {e["lane"] for r in reports for e in r.get("ems", []) if e.get("confirmed")}

    def _gf_state(self, tc, now: float, wanted: list[str]) -> dict:
        vehicles = []
        alive = set(tc.vehicle.getIDList())
        for vid in wanted:
            if vid not in alive:
                continue
            try:
                lane = tc.vehicle.getLaneID(vid)
                vehicles.append({
                    "id": vid, "vclass": tc.vehicle.getVehicleClass(vid), "type_id": tc.vehicle.getTypeID(vid),
                    "role": self.goflow_role(vid), "on_run": True, "speed": round(tc.vehicle.getSpeed(vid), 2),
                    "edge": tc.vehicle.getRoadID(vid), "lane": lane,
                    "cam_confirmed": (lane in self._ems_lanes) if self.uses_cameras else None,
                    "next": [{"tls": s[0], "link": int(s[1]), "dist": round(float(s[2]), 1), "state": s[3]}
                             for s in tc.vehicle.getNextTLS(vid) if s[0] in self.junctions]})
            except Exception:  # noqa: BLE001 - left this step
                continue
        self._gf_seq += 1
        return {"type": "state", "v": 1, "seq": self._gf_seq, "t": now, "vehicles": vehicles}

    def _requests(self, tc, now: float) -> None:
        cl = getattr(self, "goflow", None)
        if cl is None or "preempt" not in self.on:
            return super()._requests(tc, now)
        try:
            wanted = [v for v in self.emergency()]
            cur = frozenset(wanted)
            due = now - self._gf_last_send >= (self.goflow_send_every_s if wanted else self.goflow_heartbeat_s)
            if due or cur != self._gf_last_set:
                self._gf_last_send, self._gf_last_set = now, cur
                cl.exchange(self._gf_state(tc, now, wanted))
            plan = cl.valid_plan(now, self.goflow_max_age_s)
        except Exception as e:  # noqa: BLE001 - never into the SUMO thread
            log.debug("GO-FLOW: %s", e)
            plan = None
        if plan is None:
            self.goflow_stats["fallback_steps"] += 1
            return super()._requests(tc, now)
        self.goflow_stats["plan_steps"] += 1
        self._gf_apply(tc, now, plan, set(wanted))

    def _gf_apply(self, tc, now: float, plan: dict, wanted: set[str]) -> None:
        self.requests, self.request_by = {}, {}
        asked = {}
        alive = set(tc.vehicle.getIDList())
        for tls, j in plan.get("junctions", {}).items():
            a = self.junctions.get(tls)
            vid = j.get("vehicle")
            if a is None or j.get("mode") not in EMERGENCY_MODES or vid not in wanted or vid not in alive:
                continue
            links = [int(x) for x in j.get("links", [])]
            if not links:
                continue
            if j.get("rank", 0) == 0:  # its next signal: the agents' own guards, as for their own requests
                first = asked[(tls, vid)] = self._asked.get((tls, vid), now)
                if now - first > self.ev_patience_s:
                    self.stats["ev_patience"] += 1
                    continue
                try:
                    nxt = tc.vehicle.getNextTLS(vid)[:1]
                except Exception:  # noqa: BLE001
                    continue
                if nxt and nxt[0][3] in "Gg" and tc.vehicle.getSpeed(vid) < 0.5:
                    since = self._stopped_on_green.setdefault(vid, now)
                    if now - since >= self.stall_s:
                        self.stats["ev_dropped"] += 1
                        continue
                else:
                    self._stopped_on_green.pop(vid, None)
                if not self._can_pass(tc, a, vid, links[0]):
                    continue
            elif not self._exit_free(tc, a, links[0]):
                continue
            self.requests[tls] = set(links)
            self.request_by[tls] = vid
            self.goflow_stats["requests"] += 1
            key = (tls, vid, j["mode"])
            if key not in self._gf_announced:
                self._gf_announced.add(key)
                phases = [k for k in a.cycle if a.kind[k] == "car" and set(links) & _greens(a.states[k])][:1]
                what = "police car" if j.get("level") == 2 else "ambulance" if j.get("level") == 3 else "recovery truck"
                if j["mode"] == "CONFLICT_RESOLUTION":
                    text = f"GO-FLOW: two emergencies here, {what} first"
                elif j.get("rank", 0) == 0:
                    text = f"GO-FLOW: green for {what}"
                else:
                    text = f"GO-FLOW: getting ready for {what} (ETA {j.get('eta_s', 0):.0f} s)"
                self._event(a, "ambulance" if j.get("rank", 0) == 0 else "corridor", phases, text)
        self.goflow_stats["refused"] += len(plan.get("refused", []))
        self._asked = asked


class GoflowAgents(GoflowMixin, ActuatedAgents):
    name = "agents_goflow"


class ObservedGoflowAgents(GoflowMixin, ObservedActuatedAgents):
    name = "agents_observed_goflow"


def make(name: str):
    """ "agents_goflow" / "agents_observed_goflow", with the agents' own suffixes ("-skip", "+batch", "@k=v")."""
    m = re.match(r"^(agents_observed_goflow|agents_goflow)(.*)$", name)
    if not m:
        raise ValueError(f"not a GO-FLOW controller: {name}")
    base, rest = m.groups()
    cls = ObservedGoflowAgents if base == "agents_observed_goflow" else GoflowAgents
    sets = re.findall(r"@(\w+)=([0-9.]+)", rest)
    rest = re.sub(r"@\w+=[0-9.]+", "", rest)
    ctl = cls(off=set(re.findall(r"-(\w+)", rest)), extra=set(re.findall(r"\+(\w+)", rest)))
    for k, v in sets:
        if not hasattr(cls, k) or k.startswith("_"):
            raise ValueError(f"unknown agent setting {k}")
        cur = getattr(cls, k)
        setattr(ctl, k, type(cur)(float(v)) if isinstance(cur, (int, float)) else v)
    ctl.name = name
    return ctl

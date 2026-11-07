"""DSO4's congestion forecaster and GO-FLOW's traffic agent in the town (integration plan, sections 4 and 9.3).

Off by default: `python -m cosim.freefly --goflow-prediction [--goflow-traffic]`. Nothing here is on the path to the
next picture or the next SUMO step:

    SUMO thread   FreeFly.produce() -> GoflowPrediction.on_step(cur, conn, sod)
                  Aggregator: one sample a simulated second from lane subscriptions (no extra round trip); every
                  30 s one `obs` (means over the window) -> queue of 4, put_nowait, the oldest dropped when full
    worker thread obs -> GO-FLOW child (if on; at most 1 s for her `goflow` answer, else "goflow": null)
                      -> DSO4 child -> forecasts and alerts, kept for the overlay and written to out/
    children      agents/goflow_prediction and agents/goflow_traffic, each with its OWN venv, CPU only, JSON lines on
                  stdin / stdout, stderr to tools/logs/; restarted after 5, 20, 60 s, then left off for the run

A child that dies, hangs or writes garbage costs the town nothing: the viewer goes on and the overlay says so.
The same Aggregator builds the training data (tools/goflow_prediction_dataset.py), so the agents see live exactly
what they were trained on.
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("goflow_prediction")

HORIZONS_S = (60, 120, 300)
RESTART_S = (5.0, 20.0, 60.0)
# the shipped models and graph (models/goflow/, in git); tools/goflow_prediction_build.bat makes new ones in out/
SHIPPED = ROOT / "models" / "goflow"
DEFAULT_GRAPH = SHIPPED / "graph" if (SHIPPED / "graph" / "nodes.json").is_file() else \
    ROOT / "out" / "goflow_prediction" / "graph"


def default_models(with_goflow: bool) -> tuple[list[str], str | None]:
    """The shipped models (models/goflow/) when present, else those tools/goflow_prediction_build.bat made in out/."""
    if (SHIPPED / "dso4" / "model_card.json").is_file():
        dso4 = [str(SHIPPED / "dso4")] + ([str(SHIPPED / "dso4_goflow")] if with_goflow else [])
        return dso4, (str(SHIPPED / "goflow_xgb") if with_goflow else None)
    out = ROOT / "out"
    dso4 = [str(out / "goflow_prediction" / "models" / "lgbm_real_v2")]
    if with_goflow:
        dso4.append(str(out / "goflow_prediction" / "models" / "lgbm_gf_v2"))
    return dso4, (str(out / "goflow_traffic" / "models" / "xgb_120s_v1") if with_goflow else None)


def trained_clock_speed(model_dir) -> float | None:
    """The clock speed (town seconds per SUMO second) of the data a model was trained on, from its card."""
    d = Path(model_dir)
    for card, key in (("model_card.json", "dataset_meta"), ("xgboost_export_metadata.json", "dataset_meta")):
        p = d / card
        if p.is_file():
            meta = json.loads(p.read_text(encoding="utf-8")).get(key) or {}
            cs = meta.get("clock_speed")
            return float(cs) if cs is not None else None
    return None
GOFLOW_WAIT_S = 1.0
ALERT_SHOW_S = 60.0          # an alert stays on the overlay this long (simulated seconds)


# ----------------------------------------------------------------------------- aggregation
class Aggregator:
    """Per approach (one per camera: tools/goflow_prediction_graph.py nodes.json), once a simulated second:
    vehicles, halting vehicles, mean speed (m/s), occupancy (SUMO's fraction) over its lanes, and the share of its
    signal links that are green. Every `dt_s`, the means of the window -> one row."""

    def __init__(self, conn, nodes: list[dict], dt_s: float = 30.0):
        import traci.constants as tc

        self.tc = tc
        self.nodes = nodes
        self.dt_s = dt_s
        self.vars = [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_MEAN_SPEED,
                     tc.LAST_STEP_OCCUPANCY]
        for n in nodes:
            for ln in n["lanes"]:
                conn.lane.subscribe(ln, self.vars)
        self.links = []
        for n in nodes:
            cl = conn.trafficlight.getControlledLinks(n["tls"])
            self.links.append((n["tls"], [i for i, g in enumerate(cl) if any(lk[0] in n["lanes"] for lk in g)]))
        self.reset()

    def reset(self, t: float | None = None) -> None:
        N = len(self.nodes)
        self.acc = [[0.0] * 5 for _ in range(N)]
        self.n = 0
        self.win_end = None if t is None else t + self.dt_s
        self.last_sample = None

    def sample(self, conn, t: float, tls_states: dict) -> dict | None:
        """Call after every SUMO step. -> a row {t, halt, veh, speed, occ, green: [N]} when a window closes."""
        if abs(t - round(t)) > 1e-6 or (self.last_sample is not None and t - self.last_sample < 1.0 - 1e-6):
            return None
        self.last_sample = t
        if self.win_end is None:
            self.win_end = t + self.dt_s
        tc = self.tc
        subs = conn.lane.getAllSubscriptionResults()
        for j, n in enumerate(self.nodes):
            veh = halt = occ = sp = 0.0
            for ln in n["lanes"]:
                r = subs.get(ln, {})
                veh += r.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)
                halt += r.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                occ += r.get(tc.LAST_STEP_OCCUPANCY, 0.0)       # already a fraction
                sp += r.get(tc.LAST_STEP_MEAN_SPEED, 0.0)
            nl = max(len(n["lanes"]), 1)
            tls, idx = self.links[j]
            st = tls_states.get(tls, "")
            green = (sum(st[i] in "Gg" for i in idx if i < len(st)) / len(idx)) if idx and st else math.nan
            a = self.acc[j]
            a[0] += halt
            a[1] += veh
            a[2] += sp / nl
            a[3] += occ / nl
            a[4] += green
        self.n += 1
        if t < self.win_end - 1e-6:
            return None
        k = max(self.n, 1)
        row = {"t": t, **{name: [self.acc[j][i] / k for j in range(len(self.nodes))]
                          for i, name in enumerate(("halt", "veh", "speed", "occ", "green"))}}
        self.reset()
        self.win_end = t + self.dt_s
        self.last_sample = t
        return row


def obs_message(row: dict, nodes: list[dict], seq: int, sod: float, clock_speed: float, scale: float = 1.0) -> dict:
    from cosim import daily

    def r(x):
        return None if x is None or not math.isfinite(x) else round(float(x), 5)

    return {"type": "obs", "v": 1, "seq": seq, "t": row["t"], "tod_s": round(sod, 1),
            "demand": round(daily.demand_factor(sod / 3600.0) * scale, 5),
            "demand_at": {str(h): round(daily.demand_factor(((sod + h * clock_speed) % 86400.0) / 3600.0) * scale, 5)
                          for h in HORIZONS_S},
            "nodes": {n["node"]: {c: r(row[c][j]) for c in ("veh", "halt", "speed", "occ", "green")}
                      for j, n in enumerate(nodes)}}


# ----------------------------------------------------------------------------- one child process
class Child:
    """One agent process: started with its own venv's Python, JSON lines both ways, restarted when it dies."""

    def __init__(self, name: str, python: Path, cwd: Path, args: list[str], log_dir: Path, on_message):
        self.name, self.python, self.cwd, self.args = name, Path(python), Path(cwd), args
        self.log_path = Path(log_dir) / f"{name}.log"
        self.on_message = on_message
        self.proc = None
        self.state = "starting"          # starting | ready | restarting | off
        self.failures = 0
        self.restart_at = 0.0
        self.error = ""
        self.lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        if not self.python.is_file():
            self.state, self.error = "off", f"no Python at {self.python} (tools\\setup_{self.name}.bat)"
            log.error("%s: %s", self.name, self.error)
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.err = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115 - kept open for the child's life
        self.err.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} start {self.args}\n")
        self.err.flush()
        try:
            self.proc = subprocess.Popen([str(self.python), *self.args], cwd=str(self.cwd), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self.err, text=True, encoding="utf-8",
                                         bufsize=1, env=env,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            self._failed(f"cannot start: {e}")
            return
        self.state = "starting"
        threading.Thread(target=self._read, args=(self.proc,), daemon=True, name=f"{self.name}-out").start()

    def _read(self, proc) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                self.err.write(f"[not JSON on stdout] {line[:200]}\n")
                continue
            if msg.get("type") == "ready":
                self.state = "ready"
            elif msg.get("type") == "error" and self.state == "starting":
                self.error = msg.get("msg", "")
            self.on_message(self, msg)
        code = proc.wait()
        if proc is self.proc and self.state != "off":
            self._failed(f"exited with code {code}" + (f": {self.error}" if self.error else ""))

    def _failed(self, why: str) -> None:
        self.failures += 1
        self.error = why
        log.error("%s %s (failure %d)", self.name, why, self.failures)
        if self.failures > len(RESTART_S):
            self.state = "off"
            log.error("%s: off for the rest of the run", self.name)
        else:
            self.state = "restarting"
            self.restart_at = time.time() + RESTART_S[self.failures - 1]

    def tick(self) -> None:
        """From the worker thread: restart when due."""
        if self.state == "restarting" and time.time() >= self.restart_at:
            self._start()

    def send(self, msg: dict) -> bool:
        if self.state != "ready" or self.proc is None or self.proc.poll() is not None:
            return False
        try:
            with self.lock:
                self.proc.stdin.write(json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n")
                self.proc.stdin.flush()
            return True
        except (OSError, ValueError) as e:
            self._failed(f"write failed: {e}")
            return False

    def close(self) -> None:
        self.state = "off"
        p = self.proc
        if p is None:
            return
        try:
            p.stdin.close()
            p.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            p.terminate()
        try:
            self.err.close()
        except Exception:  # noqa: BLE001, S110 - closing anyway
            pass


# ----------------------------------------------------------------------------- the two agents together
class GoflowPrediction:
    def __init__(self, conn, *, model_dirs: list[str], goflow_model: str | None = None,
                 nodes_file: Path | None = None, python_dso4: Path | None = None, python_goflow: Path | None = None,
                 clock_speed: float = 60.0, dt_s: float = 30.0, out_dir: Path | None = None,
                 log_dir: Path | None = None, dso4_args: list[str] | None = None):
        """dso4_args: the DSO4 child's command after its Python (tests run a fake agent through it).
        Refuses (ValueError) a clock speed the models were not trained at: their demand features and their horizons
        in town hours assume it."""
        for m in list(model_dirs) + ([goflow_model] if goflow_model else []):
            trained = trained_clock_speed(m)
            if trained is not None and abs(trained - clock_speed) > 1e-6:
                raise ValueError(f"the model in {m} was trained with the clock at {trained:g}x and the viewer runs it at "
                                 f"{clock_speed:g}x: run with --clock-speed {trained:g} (or train a model at "
                                 f"{clock_speed:g}x with tools/goflow_prediction_dataset.py --clock-speed)")
        nodes_file = Path(nodes_file or DEFAULT_GRAPH / "nodes.json")
        self.nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
        self.names = [n["node"] for n in self.nodes]
        self.tls = {n["node"]: n["tls"] for n in self.nodes}
        self.agg = Aggregator(conn, self.nodes, dt_s)
        self.clock_speed = clock_speed
        self.seq = 0
        self.q: queue.Queue = queue.Queue(maxsize=4)
        self.dropped = 0
        self.latest: dict | None = None
        self.alerts: deque = deque(maxlen=200)
        self.goflow_latest: dict | None = None
        self.lock = threading.Lock()
        self._io_lock = threading.Lock()
        self.stop = False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out = Path(out_dir or ROOT / "out" / "goflow_prediction" / f"live_{stamp}")
        self.out.mkdir(parents=True, exist_ok=True)
        self._files = {k: open(self.out / f"{k}.jsonl", "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
                       for k in ("obs", "forecasts", "alerts", "goflow")}
        log_dir = Path(log_dir or ROOT / "tools" / "logs")
        hello = {"type": "hello", "v": 1, "run": self.out.name, "dt_s": dt_s, "source": "exact",
                 "clock_speed": clock_speed, "nodes": self.names}
        self._hello = hello
        self._gf_wait: dict[int, dict] = {}
        self._gf_event = threading.Condition()
        agents = ROOT / "agents"
        args = ["-m", "goflow_prediction.serve"]
        for m in model_dirs:
            args += ["--model", str(Path(m).resolve())]
        args = dso4_args or args
        self.dso4 = Child("goflow_prediction", python_dso4 or agents / "goflow_prediction" / ".venv" / "Scripts" /
                          "python.exe", agents / "goflow_prediction", args, log_dir, self._on_dso4)
        self.goflow = None
        if goflow_model:
            self.goflow = Child("goflow_traffic", python_goflow or agents / "goflow_traffic" / ".venv" / "Scripts" /
                                "python.exe", agents / "goflow_traffic",
                                ["-m", "goflow_traffic.serve", "--model", str(Path(goflow_model).resolve())], log_dir,
                                self._on_goflow)
        self._hello_sent: set[str] = set()
        # what each child was last sent: a restarted child gets it again, so it has its 12-13 steps of history back
        # in seconds instead of waiting 6 simulated minutes; its answers about those old steps are not used
        self._hist = {"goflow_prediction": deque(maxlen=12), "goflow_traffic": deque(maxlen=13)}
        self._replay_until: dict[str, float] = {}
        self.worker = threading.Thread(target=self._work, daemon=True, name="goflow-worker")
        self.worker.start()

    # SUMO thread: cheap, never waits
    def on_step(self, cur, conn, sod: float) -> None:
        row = self.agg.sample(conn, cur.time, cur.tls)
        if row is None:
            return
        msg = obs_message(row, self.nodes, self.seq, sod, self.clock_speed)
        self.seq += 1
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(msg)
            except queue.Full:
                self.dropped += 1

    def reset(self, t: float, reason: str = "clock jump") -> None:
        """After F6 / F7 or a long pause: the windows would mix two moments."""
        self.agg.reset()
        for h in self._hist.values():
            h.clear()
        try:
            self.q.put_nowait({"type": "reset", "v": 1, "t": t, "reason": reason})
        except queue.Full:
            self.dropped += 1

    # worker thread
    def _work(self) -> None:
        last_log = time.time()
        while not self.stop:
            for ch in (self.dso4, self.goflow):
                if ch is not None:
                    ch.tick()
                    if ch.state == "ready" and ch.name not in self._hello_sent:
                        ch.send(self._hello)
                        self._hello_sent.add(ch.name)
                        # only after a crash (not at the first start, where the log already says what it was sent)
                        hist = list(self._hist[ch.name]) if ch.failures else []
                        if hist:                       # a restart: its history back, its answers on it ignored
                            self._replay_until[ch.name] = hist[-1]["t"]
                            for m in hist:
                                ch.send(m)
                            log.info("%s restarted: %d steps of history sent again", ch.name, len(hist))
                    elif ch.state != "ready":
                        self._hello_sent.discard(ch.name)
            try:
                msg = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg["type"] == "obs":
                # every step goes into the history, sent or not: a child restarted after a gap gets the steps it
                # missed too, so its window has no hole (a hole makes it start its 6-minute warm-up again)
                if self.goflow is not None:
                    self._hist["goflow_traffic"].append(msg)
                if self.goflow is not None and self.goflow.send(msg):
                    gf = self._wait_goflow(msg["seq"])
                    msg = dict(msg, goflow=gf["nodes"] if gf else None)
                self._hist["goflow_prediction"].append(msg)
            elif self.goflow is not None:
                self.goflow.send(msg)
            delivered = self.dso4.send(msg)
            # logged as DSO4 received it (GO-FLOW's answer included), so a run can be replayed offline exactly
            self._write("obs", dict(msg, delivered=delivered))
            if time.time() - last_log > 60.0 and self.dropped:
                log.warning("goflow: %d observations dropped (agents slower than the town)", self.dropped)
                last_log = time.time()

    def _wait_goflow(self, seq: int) -> dict | None:
        end = time.time() + GOFLOW_WAIT_S
        with self._gf_event:
            while seq not in self._gf_wait:
                left = end - time.time()
                if left <= 0:
                    return None
                self._gf_event.wait(left)
            res = self._gf_wait.pop(seq)
            for k in [k for k in self._gf_wait if k < seq]:
                self._gf_wait.pop(k)
            return res

    def _replayed(self, ch, msg: dict) -> bool:
        """An answer about a step re-sent after a restart: already answered before, not used again."""
        until = self._replay_until.get(ch.name)
        return until is not None and msg.get("t") is not None and msg["t"] <= until

    def _on_goflow(self, ch, msg: dict) -> None:
        if msg.get("type") == "goflow" and not self._replayed(ch, msg):
            self._write("goflow", msg)
            with self._gf_event:
                self._gf_wait[msg.get("seq")] = msg
                self._gf_event.notify_all()
            self.goflow_latest = msg
        elif msg.get("type") == "error":
            ch.err.write(f"[error] {msg.get('msg')}\n")
            ch.err.flush()

    def _on_dso4(self, ch, msg: dict) -> None:
        typ = msg.get("type")
        if typ in ("forecast", "alert") and self._replayed(ch, msg):
            return
        if typ == "forecast":
            self._write("forecasts", msg)
            with self.lock:
                self.latest = msg
        elif typ == "alert":
            self._write("alerts", msg)
            with self.lock:
                self.alerts.append(msg)
        elif typ == "error":
            ch.err.write(f"[error] {msg.get('msg')}\n")
            ch.err.flush()

    def _write(self, kind: str, msg: dict) -> None:
        with self._io_lock:
            f = self._files.get(kind)
            if f is None or f.closed:      # closing: the last answers are not written
                return
            f.write(json.dumps(msg, separators=(",", ":")) + "\n")
            f.flush()

    # main thread (overlay)
    def status(self) -> str:
        parts = []
        for ch, label in ((self.dso4, "DSO4"), (self.goflow, "GO-FLOW")):
            if ch is None:
                continue
            if ch.state == "ready":
                warm = label == "DSO4" and self.latest is None
                parts.append(f"{label} warming up" if warm else f"{label} ok")
            elif ch.state == "off":
                parts.append(f"{label} OFFLINE")
            else:
                parts.append(f"{label} {ch.state}")
        return "  ·  ".join(parts)

    def recent_alerts(self, now: float, window_s: float = ALERT_SHOW_S) -> list[dict]:
        with self.lock:
            return [a for a in self.alerts if now - a["t"] <= window_s]

    def close(self) -> None:
        self.stop = True
        self.worker.join(timeout=2.0)          # its last message written before the files close
        for ch in (self.dso4, self.goflow):
            if ch is not None:
                ch.close()
        with self._io_lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:  # noqa: BLE001, S110 - closing anyway
                    pass

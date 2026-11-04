"""GO-FLOW priority agent for Town05: a child process that speaks JSON lines (integration plan, section 4).

    python -m town05.serve [--horizon-s 30] [--recovery-s 10] [--confirm v2i|camera] [--no-rate-limit]

Run from agents/priority_agent (cosim/goflow_priority.py does that). Reads one JSON object per line
on stdin, writes one per line on stdout:
    in : hello {run_id, seed, step_s, mode, confirm, junctions: {tls: {n_links, greens}}}
         state {seq, t, vehicles: [{id, vclass, type_id, role, on_run, speed, edge, lane, cam_confirmed,
                                    next: [{tls, link, dist, state}]}]}
         bye
    out: ready {agent, version, config}
         plan  {seq, t, computed_ms, junctions: {tls: {mode, vehicle, level, links, eta_s, rank, until}}, refused}
         log / error {level, where, msg}
Never touches a signal: the town's junction agents execute the plan with their own safety rules.
Writes out/goflow_priority/<run_id>/decisions.jsonl and summary.json under the directory given by
--out (default: <project root>/out/goflow_priority).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # agents/priority_agent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from town05.conflicts import LinkMap  # noqa: E402
from town05.coordinator import EMERGENCY_MODES, TownRuleCoordinator  # noqa: E402
from town05.tracker import TownTracker  # noqa: E402

AGENT = "goflow-priority"
PROTOCOL = 1


def _version() -> str:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


class Agent:
    def __init__(self, args):
        self.args = args
        self.coord: TownRuleCoordinator | None = None
        self.tracker: TownTracker | None = None
        self.log_f = None
        self.run_dir: Path | None = None
        self.n_plans = 0
        self.ms: list[float] = []
        self.modes: dict[str, int] = {}
        self._decisions_seen = 0

    def hello(self, m: dict) -> dict:
        confirm = m.get("confirm") or self.args.confirm
        self.tracker = TownTracker(confirm)
        self.coord = TownRuleCoordinator(LinkMap(m["junctions"]), horizon_s=self.args.horizon_s,
                                         recovery_s=self.args.recovery_s, rate_limit=not self.args.no_rate_limit)
        run_id = str(m.get("run_id") or time.strftime("%Y%m%d-%H%M%S"))
        self.run_dir = Path(self.args.out) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_f = (self.run_dir / "decisions.jsonl").open("a", encoding="utf-8")
        cfg = {"horizon_s": self.args.horizon_s, "recovery_s": self.args.recovery_s,
               "max_preemptions": self.coord.mode_safety.cfg.max_preemptions,
               "window_s": self.coord.mode_safety.cfg.preemption_window_s, "confirm": confirm, "llm": False}
        return {"type": "ready", "v": PROTOCOL, "agent": AGENT, "version": _version(), "config": cfg}

    def state(self, m: dict) -> dict:
        t0 = time.perf_counter()
        t = float(m["t"])
        vehicles = self.tracker.update(t, [v for v in m.get("vehicles", []) if v.get("on_run")])
        ctx = self.coord.update(t, vehicles)
        junctions = {}
        for tls, c in ctx.items():
            if c.mode == "NORMAL":
                continue
            pv = vehicles.get(c.vehicle_id) if c.vehicle_id else None
            junctions[tls] = {"mode": c.mode, "vehicle": c.vehicle_id,
                              "level": pv.priority_level if pv else None,
                              "links": self.coord.plan_links.get(tls, []) if c.mode in EMERGENCY_MODES else [],
                              "eta_s": round(c.eta_s, 1), "rank": c.rank, "until": c.until}
            self.modes[c.mode] = self.modes.get(c.mode, 0) + 1
        new = self.coord.decisions[self._decisions_seen:]
        self._decisions_seen = len(self.coord.decisions)
        refused = [{"tls": d["tls_id"], "vehicle": d["vehicle"], "mode": d["mode"],
                    "reason": next((r["reason"] for r in reversed(self.coord.safety_log.reasons)
                                    if r["tls_id"] == d["tls_id"]), "")} for d in new if not d["accepted"]]
        ms = (time.perf_counter() - t0) * 1000
        self.ms.append(ms)
        self.n_plans += 1
        for d in new:
            self.log_f.write(json.dumps({**d, "kind": "decision"}) + "\n")
        plan = {"type": "plan", "v": PROTOCOL, "seq": m.get("seq"), "t": t, "computed_ms": round(ms, 3),
                "junctions": junctions, "refused": refused, "llm": None}
        return plan

    def summary(self) -> None:
        if self.run_dir is None:
            return
        ms = sorted(self.ms) or [0.0]
        s = {"plans": self.n_plans, "mode_steps": self.modes,
             "decisions": len(self.coord.decisions) if self.coord else 0,
             "refused": sum(1 for d in (self.coord.decisions if self.coord else []) if not d["accepted"]),
             "safety_reasons": (self.coord.safety_log.reasons[-200:] if self.coord else []),
             "plan_ms_p50": round(ms[len(ms) // 2], 3), "plan_ms_p95": round(ms[int(len(ms) * 0.95)], 3),
             "config": vars(self.args)}
        (self.run_dir / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
        if self.log_f:
            self.log_f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--recovery-s", type=float, default=10.0)
    ap.add_argument("--confirm", choices=("v2i", "camera"), default="v2i")
    ap.add_argument("--no-rate-limit", action="store_true")
    ap.add_argument("--out", default=str(HERE.parents[1] / "out" / "goflow_priority"))
    args = ap.parse_args(argv)
    agent = Agent(args)
    out = sys.stdout

    def send(obj: dict) -> None:
        out.write(json.dumps(obj, separators=(",", ":")) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
            kind = m.get("type")
            if kind == "hello":
                send(agent.hello(m))
            elif kind == "state":
                if agent.coord is None:
                    send({"type": "error", "where": "state", "msg": "state before hello"})
                    continue
                send(agent.state(m))
            elif kind == "bye":
                break
        except Exception as e:  # noqa: BLE001 - reported, never fatal: the town falls back on its own
            send({"type": "error", "where": "serve", "msg": f"{type(e).__name__}: {e}",
                  "trace": traceback.format_exc()[-800:]})
    agent.summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())

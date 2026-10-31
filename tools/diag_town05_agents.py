"""Watches one Town05 evaluation run (tools/eval_town05_baseline.run) and says where and when it jams.

    python tools/diag_town05_agents.py agents 1.0 11            # controller, demand scale, seed
    python tools/diag_town05_agents.py actuated 1.0 11 --end 3600 --json out/diag/actuated_x1_seed11.json

Every 60 s: vehicles in the network, halting, waiting to get in, and the halting vehicles per signalised
junction (on the lanes that lead into it). At the end: the vehicles standing for over 300 s, grouped by the
junction they wait for (the next signal on their way, else the node their lane ends at), and the first minute
at which more than 100 vehicles had stood for over 300 s (the lock-up).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("controller")
    ap.add_argument("scale", type=float)
    ap.add_argument("seed", type=int)
    ap.add_argument("--end", type=float, default=3600.0)
    ap.add_argument("--port", type=int, default=8960)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import eval_town05_baseline as ev

    series: list[dict] = []
    state = {"incoming": None, "locked_at": None, "stuck_by": None}

    def probe(s) -> None:
        c = s.conn
        t = c.simulation.getTime()
        if state["incoming"] is None:
            inc = {}
            for tls in c.trafficlight.getIDList():
                inc[tls] = sorted({ln for ln in c.trafficlight.getControlledLanes(tls) if not ln.startswith(":")})
            state["incoming"] = inc
        if int(t * 2) % 120 != 0:
            return
        vids = c.vehicle.getIDList()
        halting = sum(1 for v in vids if c.vehicle.getSpeed(v) < 0.1)
        long_stand = sum(1 for v in vids if c.vehicle.getWaitingTime(v) > 300.0)
        per = {tls: sum(c.lane.getLastStepHaltingNumber(ln) for ln in lanes)
               for tls, lanes in state["incoming"].items()}
        top = sorted(per.items(), key=lambda kv: -kv[1])[:6]
        series.append({"t": t, "in_net": len(vids), "halting": halting, "stood_300": long_stand,
                       "pending": len(c.simulation.getPendingVehicles()), "top": top})
        if long_stand > 100 and state["locked_at"] is None:
            state["locked_at"] = t
        if t >= a.end - 1.0:
            by = Counter()
            for v in vids:
                if c.vehicle.getWaitingTime(v) <= 300.0:
                    continue
                nxt = c.vehicle.getNextTLS(v)
                if nxt:
                    by[nxt[0][0]] += 1
                else:
                    ln = c.vehicle.getLaneID(v)
                    by["node:" + (ln.split("_")[0] if not ln.startswith(":") else ln[1:].split("_")[0])] += 1
            state["stuck_by"] = by.most_common(15)

    out = ev.run(a.controller, a.scale, a.seed, a.end, 120.0, 20.0, a.port, runs=ROOT / "out" / "diag" / "runs",
                 probe=probe)
    rep = {"controller": a.controller, "scale": a.scale, "seed": a.seed, "throughput_per_h": out["throughput_per_h"],
           "stuck": out["stuck_over_300s_at_end"], "never_entered": out["never_entered"],
           "locked_at": state["locked_at"], "stuck_by_signal": state["stuck_by"], "agent_stats": out.get("agent_stats"),
           "series": series}
    for r in series[::5]:
        print(f"{r['t']:6.0f} s  in {r['in_net']:4d}  halting {r['halting']:4d}  stood>300 {r['stood_300']:4d}  "
              f"pending {r['pending']:4d}  top {r['top'][:3]}")
    print("locked at", state["locked_at"], "stuck by signal", state["stuck_by"])
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

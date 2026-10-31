"""Where does Town05 lock up? (SUMO only)

    python tools/diag_town05_gridlock.py --controller actuated --scale 1.0 --seed 11
    python tools/diag_town05_gridlock.py --controller agents --scale 1.0 --seed 11

Runs one controller as tools/eval_town05_baseline.py does (extras on, an emergency vehicle every ~120 s)
and every minute records, per signal, the vehicles halting on its incoming lanes and inside its junction
box, and the edges with 5 or more halting vehicles. The onset is the first minute in which more than 30 % of the vehicles in the network halt. At the
onset and at the end it lists the 10 lanes with the most halting vehicles and the junction boxes with
vehicles standing in them. Writes out/agents_town05/diag_<controller>_x<scale>_seed<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim.sumo_side import SumoSide  # noqa: E402

ONSET_SHARE = 0.30


def snapshot(c) -> dict:
    halting = Counter()
    box = Counter()
    for vid in c.vehicle.getIDList():
        if c.vehicle.getSpeed(vid) >= 0.1:
            continue
        lane = c.vehicle.getLaneID(vid)
        if lane.startswith(":"):
            box[lane[1:].split("_")[0]] += 1
        else:
            halting[lane] += 1
    return {"top_lanes": halting.most_common(10), "boxes": box.most_common(10)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controller", default="actuated")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--end", type=float, default=3600.0)
    ap.add_argument("--port", type=int, default=9600)
    ap.add_argument("--out", default=str(ROOT / "out" / "agents_town05"))
    a = ap.parse_args(argv)
    s = SumoSide(ROOT / "scenarios" / "town05", a.controller, seed=a.seed, port=a.port, label="diag05",
                 extras=True, emergency_every=120.0, extra_args=["--scale", f"{a.scale:g}", "--end", f"{a.end:g}"])
    s.extras.config.pedestrian_trips = True
    if s.policy is not None and hasattr(s.policy, "emergency"):
        s.policy.emergency = lambda: [v for v, r in s.extras.emergency.items() if r.on_run]
    c = s.conn
    minutes, onset = [], None
    try:
        while s.time < a.end:
            s.step()
            if int(s.time * 2) % 120 == 0:  # every 60 s (0.5 s steps)
                ids = c.vehicle.getIDList()
                stopped = sum(1 for v in ids if c.vehicle.getSpeed(v) < 0.1)
                share = stopped / max(1, len(ids))
                per_edge = Counter(c.vehicle.getRoadID(v) for v in ids if c.vehicle.getSpeed(v) < 0.1)
                minutes.append({"t": s.time, "vehicles": len(ids), "halting_share": round(share, 3),
                                "halting_edges": {e: n for e, n in per_edge.most_common() if n >= 5}})
                if onset is None and share > ONSET_SHARE and s.time >= 300:
                    onset = {"t": s.time, **snapshot(c)}
        end = {"t": s.time, **snapshot(c)}
        stats = getattr(s.policy, "stats", None)
    finally:
        s.close()
    out = {"controller": a.controller, "scale": a.scale, "seed": a.seed, "onset": onset, "end": end,
           "minutes": minutes, "agent_stats": stats}
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"diag_{a.controller}_x{a.scale:g}_seed{a.seed}.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("onset", "end")}, indent=1))
    print("halting share per minute:", [m["halting_share"] for m in minutes])
    return 0


if __name__ == "__main__":
    sys.exit(main())

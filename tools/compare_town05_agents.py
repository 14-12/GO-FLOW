"""Step 3: the agents against actuated, seed by seed (reads the runs of tools/eval_town05_baseline.py).

    python tools/compare_town05_agents.py --out out/agents_town05

Every agents run is compared with the actuated run of the same seed and demand (the same vehicles, the
same pedestrians, the same emergency vehicles), so the differences are paired. Writes agents.md and
agents.json next to the runs.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRICS = [  # (label, path in the run's json, lower is better)
    ("waiting /trip (s)", ("vehicles", "mean_waiting_s")),
    ("time loss /trip (s)", ("vehicles", "mean_time_loss_s")),
    ("throughput /h", ("throughput_per_h",)),
    ("ambulance time loss (s)", ("emergency_on_run", "mean_time_loss_s")),
    ("pedestrian time loss (s)", ("pedestrians", "mean_time_loss_s")),
    ("stuck at end", ("stuck_over_300s_at_end",)),
    ("never entered", ("never_entered",)),
]


def get(r: dict, path) -> float | None:
    v = r
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def fmt(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    m = st.mean(xs)
    if len(xs) == 1:
        return f"{m:.1f}"
    return f"{m:.1f} ± {st.stdev(xs):.1f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "out" / "agents_town05"))
    ap.add_argument("--reference", default="actuated")
    ap.add_argument("--name", default="agents", help="write <name>.md and <name>.json")
    ap.add_argument("--title", default="Town05: junction agents (v2, on the actuated program) against actuated")
    a = ap.parse_args(argv)
    runs = Path(a.out) / "runs"
    data: dict[tuple[str, float], dict[int, dict]] = {}
    for p in sorted(runs.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault((r["controller"], float(r["scale"])), {})[int(r["seed"])] = r
    scales = sorted({sc for _, sc in data})
    order = [a.reference, "agents", "agents_observed"]
    ctls = sorted({c for c, _ in data}, key=lambda c: (order.index(c) if c in order else 9, c))
    md = [f"# {a.title}", "",
          f"Mean ± standard deviation over the seeds; Δ is the paired difference to {a.reference} (same seed, "
          "same demand), mean over the seeds, with how many seeds were better. SUMO alone, 1 h, pedestrians on "
          "the crosswalks, an emergency vehicle about every 120 s. Stuck: standing still > 300 s at the end.", ""]
    out = []
    for sc in scales:
        ref = data.get((a.reference, sc), {})
        md += [f"## Demand x{sc:g}", "", "| controller | seeds | " + " | ".join(m for m, _ in METRICS) + " |",
               "|---|---|" + "---|" * len(METRICS)]
        for c in ctls:
            rs = data.get((c, sc))
            if not rs:
                continue
            cells = []
            row = {"controller": c, "scale": sc, "seeds": sorted(rs)}
            for label, path in METRICS:
                vals = [get(rs[s], path) for s in sorted(rs)]
                vals = [v for v in vals if v is not None]
                cell = fmt(vals)
                row[label] = vals
                if c != a.reference:
                    pairs = [(get(rs[s], path), get(ref[s], path)) for s in sorted(rs) if s in ref]
                    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                    if pairs:
                        d = [x - y for x, y in pairs]
                        higher_better = label.startswith("throughput")
                        better = sum((x > y) if higher_better else (x < y) for x, y in pairs)
                        cell += f" (Δ {st.mean(d):+.1f}, {better}/{len(pairs)})"
                        row[label + " delta"] = d
                cells.append(cell)
            md.append(f"| {c} | {','.join(map(str, sorted(rs)))} | " + " | ".join(cells) + " |")
            out.append(row)
        md.append("")
    (Path(a.out) / f"{a.name}.md").write_text("\n".join(md), encoding="utf-8")
    (Path(a.out) / f"{a.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

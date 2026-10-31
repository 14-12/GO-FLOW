"""Two evaluation folders (tools/eval_town05_baseline.py runs) against each other, controller by controller, seed by
seed: what a code change did to the same runs.

    python tools/compare_runs_paired.py out/agents_final_v2 out/agents_pedfix --controllers agents,agents_observed \\
        --scales 0.5,0.8 --seeds 21,22,23,24,25 --out out/agents_pedfix/pedfix_vs_v2.md

Every row: mean of B - A over the seeds both have (same controller, demand and seed: the same vehicles, pedestrians
and emergency vehicles), and on how many seeds B was better. Writes Markdown (and prints it).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

METRICS = [("waiting /trip (s)", ("vehicles", "mean_waiting_s"), False),
           ("time loss /trip (s)", ("vehicles", "mean_time_loss_s"), False),
           ("throughput /h", ("throughput_per_h",), True),
           ("ambulance time loss (s)", ("emergency_on_run", "mean_time_loss_s"), False),
           ("pedestrian time loss (s)", ("pedestrians", "mean_time_loss_s"), False),
           ("walks completed", ("pedestrians", "completed"), True),
           ("stuck at end", ("stuck_over_300s_at_end",), False),
           ("never entered", ("never_entered",), False)]


def get(r: dict, path):
    v = r
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--controllers", default="agents,agents_observed")
    ap.add_argument("--scales", default="0.5,0.8")
    ap.add_argument("--seeds", default="21,22,23,24,25")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    A, B = Path(a.a), Path(a.b)
    seeds = [int(x) for x in a.seeds.split(",")]
    md = [f"# {a.title or f'{B} against {A}'}", "",
          f"Paired by controller, demand and seed ({', '.join(map(str, seeds))}): mean of {B.name} - {A.name}, "
          f"then mean of A -> mean of B, and on how many seeds {B.name} was better.", "",
          "| controller | demand | " + " | ".join(m for m, _, _ in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for ctl in a.controllers.split(","):
        for sc in a.scales.split(","):
            tag = f"{ctl}_x{float(sc):g}"
            cells = []
            for _label, path, higher in METRICS:
                pairs = []
                for sd in seeds:
                    pa, pb = A / "runs" / f"{tag}_seed{sd}.json", B / "runs" / f"{tag}_seed{sd}.json"
                    if pa.is_file() and pb.is_file():
                        va = get(json.loads(pa.read_text(encoding="utf-8")), path)
                        vb = get(json.loads(pb.read_text(encoding="utf-8")), path)
                        if va is not None and vb is not None:
                            pairs.append((va, vb))
                if not pairs:
                    cells.append("n/a")
                    continue
                d = [vb - va for va, vb in pairs]
                better = sum((vb > va) if higher else (vb < va) for va, vb in pairs)
                cells.append(f"{st.mean(d):+.2f} ({st.mean(x for x, _ in pairs):.1f} -> {st.mean(y for _, y in pairs):.1f}; "
                             f"{better}/{len(pairs)})")
            md.append(f"| {ctl} | x{float(sc):g} | " + " | ".join(cells) + " |")
    text = "\n".join(md) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

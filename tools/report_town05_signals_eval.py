"""Tables for the write-ups of the 26 Oct re-evaluation, from the runs themselves (no figure typed by hand).

    python tools/report_town05_signals_eval.py            # prints Markdown tables to paste/quote

Reads out/agents_final_v2 (new signals, 25 Oct), out/agents_pedfix (the pedestrian call, seeds 21-25) and
out/agents_final_v3 (corrected signals + pedestrian call, 26 Oct): per demand, actuated's means and each agent's mean
paired change against actuated on the same seed, with the seeds better.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = [("veh_wait", ("vehicles", "mean_waiting_s"), False, "s"),
     ("thru", ("throughput_per_h",), True, ""),
     ("amb", ("emergency_on_run", "mean_time_loss_s"), False, "s"),
     ("ped", ("pedestrians", "mean_time_loss_s"), False, "s"),
     ("stuck", ("stuck_over_300s_at_end",), False, "")]


def get(r, path):
    v = r
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return v


def load(folder: Path) -> dict:
    out = {}
    for p in sorted((folder / "runs").glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault((r["controller"], float(r["scale"])), {})[int(r["seed"])] = r
    return out


def table(folder: Path, scales, title: str) -> str:
    d = load(folder)
    rows = [f"**{title}** (`{folder.relative_to(ROOT).as_posix()}`)", "",
            "| demand | seeds | actuated: waiting /trip | agents (exact) vs actuated: waiting, throughput /h, ambulance "
            "time loss, pedestrian time loss | agents (cameras) vs actuated: the same | vehicles stuck > 300 s at the end (all runs) |",
            "|---|---|---|---|---|---|"]
    for sc in scales:
        ref = d.get(("actuated", sc), {})
        if not ref:
            continue
        seeds = sorted(ref)
        a_wait = st.mean(get(ref[s], M[0][1]) for s in seeds)
        a_amb = st.mean(get(ref[s], M[2][1]) for s in seeds)
        a_ped = st.mean(get(ref[s], M[3][1]) for s in seeds)
        cells = []
        stuck = sum(get(ref[s], M[4][1]) for s in seeds)
        for c in ("agents", "agents_observed"):
            rs = d.get((c, sc), {})
            parts = []
            for key, path, higher, unit in M[:4]:
                pairs = [(get(rs[s], path), get(ref[s], path)) for s in seeds if s in rs]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    parts.append("n/a")
                    continue
                dd = st.mean(x - y for x, y in pairs)
                better = sum((x > y) if higher else (x < y) for x, y in pairs)
                parts.append(f"{dd:+.1f}{unit} ({better}/{len(pairs)})")
            stuck += sum(get(rs[s], M[4][1]) for s in rs)
            cells.append(", ".join(parts))
        rows.append(f"| x{sc:g} | {seeds[0]}-{seeds[-1]} | {a_wait:.1f} s (ambulance {a_amb:.1f} s, pedestrian {a_ped:.1f} s) "
                    f"| {cells[0]} | {cells[1]} | {stuck} |")
    return "\n".join(rows) + "\n"


def main() -> int:
    print(table(ROOT / "out" / "agents_final_v2", [0.5, 0.8, 1.0, 1.2], "25 Oct, new signals (before the 26 Oct fixes)"))
    print(table(ROOT / "out" / "agents_pedfix", [0.5, 0.8], "Pedestrian call, seeds 21-25 (actuated runs copied from v2)"))
    if (ROOT / "out" / "agents_final_v3" / "runs").is_dir():
        print(table(ROOT / "out" / "agents_final_v3", [0.5, 0.8, 1.0, 1.2], "26 Oct, corrected signals + pedestrian call"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

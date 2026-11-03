"""The moment of each crash as CARLA drew it (3 Nov): from out/freefly/crash_trace.csv (cosim/pose_proc.CrashTrace,
the wrecks' actors in every picture from the server's own snapshot), for each wreck: its speed in the pictures before
the hit and after it, how long it takes to stop, and how far it moves after the hit (the push of the impact).

    python tools/check_crash_trace.py [out/freefly/crash_trace.csv] [--json out/freefly/crash_trace_check.json]

A car that stops dead at the hit goes from its speed to standing between two pictures (about 0.1 s at 9 pictures a
second); before 3 Nov it eased into the stop over about 0.3 s (up to 0.9 s) and slid back out of the car it hit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyse(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["sim_s"])
    # a vehicle far from the eye is posed every 2nd or 4th picture only (pose_proc NEAR_M, FAR_M): the pictures in
    # between repeat its pose. Speeds are taken between the pictures where it was posed anew.
    # The same pose in a third picture running: it is standing (posed every 2nd picture, a moving vehicle repeats
    # once at most; every 4th, beyond FAR_M, would read as standing: the eye is then 250 m away).
    kept = [rows[0]]
    same = 0
    for prev, r in zip(rows, rows[1:], strict=False):
        same = same + 1 if (r["x"], r["y"], r["yaw"]) == (prev["x"], prev["y"], prev["yaw"]) else 0
        if same != 1:
            kept.append(r)
    rows = kept
    t_hit, v_hit = rows[0]["t_hit"], rows[0]["v_hit"]
    sp = []
    for a, b in zip(rows, rows[1:], strict=False):
        dt = b["sim_s"] - a["sim_s"]
        if dt <= 1e-4:
            continue
        sp.append((b["sim_s"] - t_hit, math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt))
    before = [v for t, v in sp if t <= 0.0]
    # the speed it had: the median over the pictures before the hit (the first change after the trace starts may
    # span a repeat that is not in the trace, and read double)
    v_before = sorted(before)[len(before) // 2] if before else None
    out = {"vid": rows[0]["vid"], "t_hit": t_hit, "v_hit_sumo": v_hit, "pictures": len(rows),
           "pictures_before_hit": len(before), "speed_before_hit_ms": None if v_before is None else round(v_before, 2),
           "speeds_ms": [(round(t, 2), round(v, 2)) for t, v in sp]}
    if v_before is not None and v_before >= 2.0:
        last_fast = max((t for t, v in sp if v >= 0.7 * v_before and t <= 0.3), default=None)
        first_still = next((t for t, v in sp if last_fast is not None and t > last_fast and v < 0.5), None)
        out["last_picture_at_70pct_s"] = None if last_fast is None else round(last_fast, 2)
        out["first_picture_still_s"] = None if first_still is None else round(first_still, 2)
        out["stop_takes_s"] = (None if last_fast is None or first_still is None
                               else round(first_still - last_fast, 2))
    # how far it went after the hit, and back towards where it came from (a slide back out of the other car)
    at_hit = min(rows, key=lambda r: abs(r["sim_s"] - t_hit))
    end = rows[-1]
    pre = [r for r in rows if r["sim_s"] <= t_hit]
    if len(pre) >= 2:
        hx, hy = pre[-1]["x"] - pre[0]["x"], pre[-1]["y"] - pre[0]["y"]
        n = math.hypot(hx, hy)
        if n > 0.05:
            hx, hy = hx / n, hy / n
            back = 0.0
            for a, b in zip(rows, rows[1:], strict=False):
                if a["sim_s"] >= t_hit:
                    d = (b["x"] - a["x"]) * hx + (b["y"] - a["y"]) * hy
                    back += max(-d, 0.0)
            out["moved_back_after_hit_m"] = round(back, 2)
    out["moved_after_hit_m"] = round(math.hypot(end["x"] - at_hit["x"], end["y"] - at_hit["y"]), 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=str(ROOT / "out" / "freefly" / "crash_trace.csv"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    by = defaultdict(list)
    with open(a.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {k: (r[k] if k == "vid" else float(r[k])) for k in r}
            by[(row["vid"], row["t_hit"])].append(row)
    res = [analyse(v) for v in by.values() if len(v) >= 3]
    moving = [r for r in res if r.get("stop_takes_s") is not None]
    summary = {"wrecks": len(res), "moving_at_hit": len(moving),
               "stop_takes_s": sorted(r["stop_takes_s"] for r in moving),
               "moved_back_after_hit_m": sorted(r.get("moved_back_after_hit_m", 0.0) for r in res),
               "wrecks_detail": res}
    for r in res:
        print(f"{r['vid']:>12} hit {r['t_hit']:.2f}: {r['pictures_before_hit']} pictures before, "
              f"{r['speed_before_hit_ms']} m/s (SUMO {r['v_hit_sumo']}); stop takes {r.get('stop_takes_s')} s; "
              f"moved {r['moved_after_hit_m']} m after, back {r.get('moved_back_after_hit_m')} m")
    print(json.dumps({k: v for k, v in summary.items() if k != "wrecks_detail"}))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

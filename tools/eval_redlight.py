"""Grades the red-light agent (cosim/redlight.py) on a recording of tools/record_enforcement.bat against the
simulation's own record (truth.jsonl, which the agent never reads).

    python tools/eval_redlight.py out/enforcement/<recording> [--onnx out/detector_onnx/carla-yolo26n-t05.onnx]
                                  [--from S --to S] [--around-truth 8]

Runs the agent on every picture (or, with --around-truth, only on the seconds around each vehicle the record has
crossing the line), writes its cases to <recording>/agent_cases/ and the grading to <recording>/eval.json:
- crossings: each vehicle the record has crossing the line, found by the agent? (within 1.5 s, same lane); the
  agent's crossing time minus the record's (the camera's timing);
- violations: the record's entries on red (not emergency vehicles on a run), red for at least the grace time;
  caught, missed, and the agent's cases that match none (false accusations);
- plates: the agent's reading against the plate the car carries, for every crossing vehicle it read.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import enforcement as enf  # noqa: E402
from cosim import redlight  # noqa: E402


def load(rec: Path):
    site = enf.Site.from_json(json.loads((rec / "site.json").read_text(encoding="utf-8")))
    sig = [json.loads(ln) for ln in (rec / "signals.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    links = sig[0]["links"] if sig else []
    pos = {k: i for i, k in enumerate(links)}
    site.links = {ln: [pos[k] for k in ks if k in pos] for ln, ks in site.links.items()}   # (the log holds those)
    frames = [r for r in csv.DictReader((rec / "frames.csv").open(encoding="utf-8")) if r["sim_s"]]
    truth = [json.loads(ln) for ln in (rec / "truth.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    log_ = enf.SignalLog.from_changes([(r["t"], r["state"]) for r in sig],
                                      until=float(frames[-1]["sim_s"]) + 5 if frames else None)
    return site, log_, frames, truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--onnx", default=str(ROOT / "out" / "detector_onnx" / "carla-yolo26n-t05.onnx"))
    ap.add_argument("--pt", default=None, help="the .pt model through Ultralytics (GPU) instead of ONNX")
    ap.add_argument("--from", dest="t_from", type=float, default=None)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--around-truth", type=float, default=0.0)
    ap.add_argument("--offset", type=float, default=0.0, help="added to every picture's time (camera timing)")
    a = ap.parse_args(argv)
    import cv2

    rec = Path(a.recording)
    site, log_, frames, truth = load(rec)
    det = redlight.UltralyticsDetector(a.pt) if a.pt else redlight.OnnxDetector(a.onnx)
    out = rec / "agent_cases"
    agent = redlight.SiteAgent(site, log_, det, out, read_all=True)
    wanted = None
    if a.around_truth > 0:
        wanted = [(r["t"] - a.around_truth, r["t"] + a.around_truth) for r in truth]
    cap = cv2.VideoCapture(str(rec / "camera.mp4"))
    k, n_done, t_wall = 0, 0, time.time()
    rows = {int(r["frame"]): float(r["sim_s"]) for r in frames}
    # the plate camera (when the recording has it): its pictures handed to the agent as the viewer does, as JPEG
    pcap = prows = None
    if (rec / "plate.mp4").is_file() and (rec / "plate_frames.csv").is_file():
        pcap = cv2.VideoCapture(str(rec / "plate.mp4"))
        prows = [(int(r["frame"]), float(r["sim_s"]) if r["sim_s"] else None)
                 for r in csv.DictReader((rec / "plate_frames.csv").open(encoding="utf-8"))]
    pi = 0

    def plates_until(t_now: float) -> None:
        nonlocal pi
        while pcap is not None and pi < len(prows):
            kp, tp = prows[pi]
            if tp is not None and tp > t_now + 0.05:
                break
            if not pcap.grab():
                pi = len(prows)
                break
            pi += 1
            if tp is None or (wanted is not None and not any(lo <= tp <= hi for lo, hi in wanted)):
                continue
            ok_, img = pcap.retrieve()
            if ok_:
                ok_, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
                if ok_:
                    agent.plate_picture(kp, tp + a.offset, buf.tobytes())
    cases = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = rows.get(k)
        use = t is not None and (a.t_from is None or t >= a.t_from) and (a.t_to is None or t <= a.t_to) and \
            (wanted is None or any(lo <= t <= hi for lo, hi in wanted))
        if t is not None:
            plates_until(t)
        if use:
            ok, bgr = cap.retrieve()
            if ok:
                cases += agent.frame(k, t + a.offset, bgr[:, :, ::-1].copy())
                n_done += 1
                if n_done % 200 == 0:
                    print(f"{n_done} pictures, t {t:.0f} s, {time.time() - t_wall:.0f} s", flush=True)
        elif agent.agent.tracker.tracks and t is not None:
            agent.agent.tracker.update([])  # (a gap: tracks end)
        if a.t_to is not None and t is not None and t > a.t_to:
            break
        k += 1
    # last pictures: finish the tracks
    for tr in agent.agent.tracker.tracks:
        tr.done = True
    agent.frame(k + 1, (t or 0) + 30.0, bgr[:, :, ::-1].copy() if n_done else None, dets=[]) if n_done else None
    res = grade(site, truth, agent.agent.crossings, cases, a)
    res.update(pictures=n_done, wall_s=round(time.time() - t_wall))
    (rec / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in res.items() if k_ not in ("crossing_rows",)}, indent=1))
    return 0


def grade(site, truth, crossings, cases, a) -> dict:
    lo = a.t_from if a.t_from is not None else -1e9
    hi = a.t_to if a.t_to is not None else 1e9
    tr = [r for r in truth if lo + 3 <= r["t"] <= hi - 3]
    used = set()
    rows = []
    errs = []
    for r in tr:
        best = None
        for i, c in enumerate(crossings):
            if i in used or c["lane"] != r["lane"]:
                continue
            dt = c["t_cross"] - r["t"]
            if abs(dt) <= 1.5 and (best is None or abs(dt) < abs(best[1])):
                best = (i, dt)
        row = {"t": r["t"], "lane": r["lane"], "state": r["state"], "red_s": r["red_s"], "emergency": r["emergency"],
               "vclass": r["vclass"], "plate": r["plate"], "found": best is not None}
        if best is not None:
            used.add(best[0])
            c = crossings[best[0]]
            errs.append(best[1])
            row.update(dt=round(best[1], 2), agent_red_s=c["red_s"], outcome=c.get("outcome"), case=c.get("case"),
                       read=c.get("plate"), read_conf=c.get("plate_conf"))
        rows.append(row)
    viol = [r for r in rows if r["state"] in "rR" and not r["emergency"] and r["vclass"] not in ("emergency", "authority")
            and r["red_s"] >= redlight.GRACE_S]
    caught = [r for r in viol if r.get("case")]
    case_ids = {c["case"]["id"] for c in cases}
    matched_cases = {r.get("case") for r in rows if r.get("case")}
    false_cases = [cid for cid in case_ids if cid not in {r.get("case") for r in viol}]
    reads = [r for r in rows if r.get("read") is not None or r.get("found")]
    read_right = [r for r in reads if r.get("read") and r["read"] == r["plate"]]
    read_wrong = [r for r in reads if r.get("read") and r["read"] != r["plate"]]
    return {"truth_crossings": len(tr), "found": sum(1 for r in rows if r["found"]),
            "crossing_dt_s": {"median": round(statistics.median(errs), 3) if errs else None,
                              "p10": round(sorted(errs)[len(errs) // 10], 3) if errs else None,
                              "p90": round(sorted(errs)[9 * len(errs) // 10], 3) if errs else None},
            "violations": len(viol), "caught": len(caught), "missed": [r for r in viol if not r.get("case")],
            "cases": len(case_ids), "false_cases": false_cases, "unmatched_crossings": len(crossings) - len(used),
            "plates_read": len([r for r in reads if r.get("read")]), "plates_right": len(read_right),
            "plates_wrong": [(r["plate"], r["read"], r.get("read_conf")) for r in read_wrong],
            "plates_unread": len([r for r in rows if r["found"] and not r.get("read")]),
            "crossing_rows": rows, "matched_cases": sorted(c for c in matched_cases if c)}


if __name__ == "__main__":
    sys.exit(main())

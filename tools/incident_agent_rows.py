"""Recorded camera clips -> the accident agent's rows (11 Nov; docs/INCIDENT_AGENT_AUDIT.md).

For each `<camera>.mp4` of an out/incident_clips/<run>/ folder: every picture his agent would keep at 5 fps (at 10 fps
frames 1, 3, 5, ... as his process_frames and Ultralytics' vid_stride), through the live agent's own YOLO, tracker
and rows (cosim/incident_agent_proc: one model per video, his class map, confidence, image size, FP16), timed by
the clip's frames.csv: CARLA's own time of each picture (`carla_s`) put on the SUMO clock by the median gap
`sim_s - carla_s`. (Clips recorded before the evening of 6 Nov have `sim_s` in whole SUMO steps of 0.5 s: 3 of 5
pictures at 5 fps shared a time with the one before; CARLA's time keeps them apart. The gap moved < 0.1 s in 25 min.) One JSON line per picture, as the live agent
sends them to his process. agents/incident_detection/replay.py then scores them, his batch way and streamed.

    .venv\\Scripts\\python.exe tools\\incident_agent_rows.py out\\incident_clips\\<run> OUT.jsonl [--cams "a;b"]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("out")
    ap.add_argument("--cams", default="", help="camera names (spaces as in layout.json), ';' between; default all")
    ap.add_argument("--fps", type=float, default=5.0)
    a = ap.parse_args(argv)

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    from cosim.incident_agent_proc import CONFIG, WEIGHTS, class_ids, rows_of

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    device = 0 if torch.cuda.is_available() else "cpu"
    half = 16 if device != "cpu" else None
    clips = Path(a.clips)
    want = {c.strip().replace(" ", "_") for c in a.cams.split(";") if c.strip()}
    n_out = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for mp4 in sorted(clips.glob("*.mp4")):
            stem = mp4.stem
            idx = clips / f"{stem}.frames.csv"
            if not idx.is_file() or (want and stem not in want):
                continue
            cam = stem.replace("_", " ")
            with open(idx, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            sims = [float(r["sim_s"]) for r in rows]
            if rows and rows[0].get("carla_s"):
                gaps = sorted(float(r["sim_s"]) - float(r["carla_s"]) for r in rows)
                gap = gaps[len(gaps) // 2]
                sims = [float(r["carla_s"]) + gap for r in rows]
            model = YOLO(str(WEIGHTS))
            cats = class_ids(cfg["class_map"], model.names)
            opts = dict(tracker=cfg.get("tracker", "bytetrack.yaml"), persist=True, conf=float(cfg["confidence"]),
                        classes=sorted(cats), device=device, verbose=False, save=False, imgsz=int(cfg["imgsz"]))
            cap = cv2.VideoCapture(str(mp4))
            fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
            stride = max(1, round(fps / a.fps))
            t1, k, kept = time.time(), 0, 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                k += 1
                if k % stride or k - 1 >= len(sims):
                    continue
                r = model.track(bgr, quantize=half, **opts)[0]
                frame, dets = rows_of(r, float(bgr.shape[0] * bgr.shape[1]), cats)
                out.write(json.dumps({"cam": cam, "t": round(sims[k - 1], 3), "frame": frame, "dets": dets},
                                     separators=(",", ":")) + "\n")
                kept += 1
            cap.release()
            n_out += kept
            print(f"{cam}: {kept} pictures of {k} in {time.time() - t1:.0f} s", flush=True)
            del model
            if device != "cpu":
                torch.cuda.empty_cache()
    print(f"{n_out} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

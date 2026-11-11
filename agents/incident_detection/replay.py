"""Omar's agent on recorded rows, two ways: his batch path and the streaming adapter (Town05 checks, 11 Nov).

The rows come from tools/incident_agent_rows.py (our venv: the recorded MP4s through the same YOLO and tracker as
the live agent). Here, in his venv:
  batch      his aggregate_windows on the whole stream, IncidentDetector.predict_windows, window_crash_features
  stream     stream.StreamingWindows, picture by picture, as live
Both are written as windows JSONL (the live agent's fields) and compared window by window.

    agents\\incident_detection\\.venv\\Scripts\\python.exe agents\\incident_detection\\replay.py ROWS.jsonl OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stream  # noqa: E402
from agents.incident_detection.src.inference import IncidentDetector  # noqa: E402
from agents.incident_detection.src.video_processing import aggregate_windows  # noqa: E402
from crash_features import window_crash_features  # noqa: E402


def load(path):
    by_cam = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        m = json.loads(line)
        dets = [dict(zip(stream.DET_KEYS, d)) for d in m["dets"]]
        by_cam[m["cam"]].append((float(m["t"]), m["frame"], dets))
    return by_cam


def batch(cam, pictures, detector, t0):
    frames = [dict(f, source_time_sec=t - t0) for t, f, _ in pictures]
    dets = [dict(d, source_time_sec=t - t0) for t, _, ds in pictures for d in ds]
    duration = pictures[-1][0] - t0
    win = aggregate_windows(frames, dets, duration)
    out = detector.predict_windows(win, cam)
    crash = window_crash_features(pd.DataFrame(dets), duration) if dets else None
    for o in out:
        o.update(camera=cam, sim_start_s=round(t0 + o["window_start_sec"], 3),
                 sim_end_s=round(t0 + o["window_end_sec"], 3))
        imp = {k: 0.0 for k in stream.IMPACT}
        if crash is not None:
            hit = crash[(crash.window_start_sec - o["window_start_sec"]).abs() < 1e-6]
            if len(hit):
                imp = {k: float(hit.iloc[0][k]) for k in stream.IMPACT}
        o.update(impact=imp, impact_alarm=bool(imp["contact_hard_stops"] >= 1))
    return out


def main(rows_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = IncidentDetector()
    detector.model.set_params(classifier__n_jobs=1)
    report = {"cameras": {}}
    with open(out_dir / "windows_batch.jsonl", "w", encoding="utf-8") as fb, \
            open(out_dir / "windows_stream.jsonl", "w", encoding="utf-8") as fs:
        for cam, pictures in load(rows_path).items():
            kept, last = [], None  # the stream's rule (StreamingWindows.add): a picture not after the last is dropped
            for p in pictures:
                if last is None or p[0] > last:
                    kept.append(p)
                    last = p[0]
            pictures = kept
            s = stream.StreamingWindows(cam, detector)
            live = []
            for t, f, d in pictures:
                live += s.add(t, f, d)
            live += s.flush(pictures[-1][0])
            ref = batch(cam, pictures, detector, s.t0)
            diff = {"windows": [len(live), len(ref)], "score_max_abs": 0.0, "v1_disagree": 0, "impact_disagree": 0,
                    "feature_max_abs": 0.0}
            for a, b in zip(live, ref):
                diff["score_max_abs"] = max(diff["score_max_abs"], abs(a["incident_score"] - b["incident_score"]))
                diff["v1_disagree"] += a["incident_detected"] != b["incident_detected"]
                diff["impact_disagree"] += a["impact_alarm"] != b["impact_alarm"]
                diff["feature_max_abs"] = max([diff["feature_max_abs"]] + [abs(a["traffic_features"][k] -
                                                                              b["traffic_features"][k])
                                                                          for k in a["traffic_features"]])
            report["cameras"][cam] = diff
            for w in ref:
                fb.write(json.dumps(w) + "\n")
            for w in live:
                fs.write(json.dumps(w) + "\n")
            print(cam, json.dumps(diff), flush=True)
    (out_dir / "replay_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

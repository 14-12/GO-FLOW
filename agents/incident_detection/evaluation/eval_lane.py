"""Stopped-in-lane rule: threshold from Town05 training runs, then one scoring on the test runs.

Score per window = lane_spot_age_s (lane_features.py), smoothed with mean3 like v1. The
threshold is the lowest on the grid whose training false alarms are at most --max-fa per
camera-hour. The rule passes the training stage only if it also catches the near training
crash (20261031_003054, camera 2014-2). --stage test reuses the training threshold and builds
each test camera's lane prior from the training runs of the same camera, if any.

    python agents/incident_detection/evaluation/eval_lane.py --stage train
    python agents/incident_detection/evaluation/eval_lane.py --stage test
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(AGENT_DIR.parents[1]))
sys.path.insert(0, str(EVAL_DIR))
from lane_features import LANE_FEATURES, lane_prior, window_lane_features  # noqa: E402
from train_v2 import summary  # noqa: E402

GRID = np.arange(30, 905, 5.0)
NEAR_TRAINING_CRASH = "20261031_003054|2014-2 looking W"


def recordings(folder, runs):
    for path in sorted(folder.glob("*.tracks.csv.gz")):
        run, stem = path.name.split("__")[0], path.name.split("__")[1].replace(".tracks.csv.gz", "")
        if run in runs:
            yield run, stem, path


def build(folder, runs, priors=None):
    frames = []
    for run, stem, path in recordings(folder, runs):
        windows = pd.read_csv(path.with_name(path.name.replace(".tracks.csv.gz", ".csv")))
        prior = (priors or {}).get(stem)
        lane = window_lane_features(pd.read_csv(path), float(windows.window_end_sec.max()), prior)
        w = windows.merge(lane, on="window_start_sec", how="left", validate="1:1")
        w["stream"], w["prior"] = run + "|" + w.camera, prior is not None
        frames.append(w)
    return pd.concat(frames, ignore_index=True).sort_values(["stream", "window_start_sec"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "test"), required=True)
    parser.add_argument("--tracks", type=Path, default=AGENT_DIR / "data" / "town05" / "yolo11n-coco-tracks")
    parser.add_argument("--max-fa", type=float, default=1.0)
    args = parser.parse_args()
    split = json.loads((EVAL_DIR / "v2_split.json").read_text())
    train_runs, test_runs = set(split["town05"]["train_runs"]), set(split["town05"]["test_runs"])
    results = EVAL_DIR / "results"

    if args.stage == "train":
        df = build(args.tracks, train_runs)
        scores = df.lane_spot_age_s.to_numpy()
        curve = [(float(t), summary(df, scores, t)) for t in GRID]
        ok = [(t, s) for t, s in curve if s["false_alarms_per_camera_hour"] <= args.max_fa]
        threshold, s = ok[0] if ok else curve[-1]
        near = [e for e in s["per_event"] if e["stream"] == NEAR_TRAINING_CRASH]
        passed = bool(ok) and any(e["caught"] for e in near)
        for t, c in curve[::10]:
            print(f"  threshold {t:5.0f}s | FA/h {c['false_alarms_per_camera_hour']:6.2f} | "
                  f"events {c['events_caught']}/{c['events']} | normal time in alarm {c['normal_alarm_share']:.2%}")
        print(f"threshold {threshold:.0f}s | FA/h {s['false_alarms_per_camera_hour']} | events {s['events_caught']}/{s['events']}")
        for e in s["per_event"]:
            print(f"  {e['stream']:42s} {e['start_sec']:7.1f}-{e['end_sec']:7.1f}s "
                  f"{'caught after ' + str(round(e['time_to_detect_s'])) + ' s' if e['caught'] else 'missed'}")
        normal = df[df.label == 0].groupby("stream").lane_spot_age_s.max().sort_values(ascending=False)
        print("longest in-lane stop in normal traffic per camera (s):", normal.head(6).round(0).to_dict())
        print("PASS" if passed else "FAIL", "(near training crash caught at FA <= max)")
        (results / "lane_train.json").write_text(json.dumps({
            "created": datetime.now().isoformat(timespec="seconds"), "max_fa": args.max_fa, "threshold": threshold,
            "passed": passed, "train": {k: v for k, v in s.items()},
            "curve": [{"threshold": t, **{k: c[k] for k in ("false_alarms_per_camera_hour", "events_caught", "events",
                                                             "normal_alarm_share")}} for t, c in curve]}, indent=1))
        return

    trained = json.loads((results / "lane_train.json").read_text())
    if not trained["passed"]:
        raise SystemExit("The rule did not pass the training stage; not scoring the test runs.")
    threshold = trained["threshold"]
    priors = {}
    for stem in {stem for _, stem, _ in recordings(args.tracks, test_runs)}:
        history = [pd.read_csv(p) for _, s, p in recordings(args.tracks, train_runs) if s == stem]
        if history:
            priors[stem] = lane_prior(history)
    df = build(args.tracks, test_runs, priors)
    s = summary(df, df.lane_spot_age_s.to_numpy(), threshold)
    print(f"TEST threshold {threshold:.0f}s | FA/h {s['false_alarms_per_camera_hour']} | normal time in alarm "
          f"{s['normal_alarm_share']:.2%} | events {s['events_caught']}/{s['events']} | TTD median {s['time_to_detect_median_s']}")
    for e in s["per_event"]:
        print(f"  {e['stream']:42s} {e['start_sec']:7.1f}-{e['end_sec']:7.1f}s "
              f"{'caught after ' + str(round(e['time_to_detect_s'])) + ' s' if e['caught'] else 'missed'}")
    out = results / f"lane_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"threshold": threshold, "cameras_with_prior": sorted(priors), "test": s}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()

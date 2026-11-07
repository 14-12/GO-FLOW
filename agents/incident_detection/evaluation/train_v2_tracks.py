"""v2 with stopped-vehicle evidence: candidates chosen on Town05 training data, scored once on test.

Candidates (all use the mean3 rule over window scores):
  A  rule: score = max_spot_age_s (seconds a place has held a stationary vehicle)
  B  Random Forest on v1's 27 features + the track features
  C  Random Forest on the track features only
Each candidate's threshold is the lowest whose out-of-fold false alarms on Town05 training runs
(leave one run out) are at most --max-fa per camera-hour. The candidate catching the most
training accidents at that threshold is chosen (ties: fewer false alarms), before any test
scoring. AI City has no tracks yet, so this v2 is Town05-only; v1 is scored on the same test.

    python agents/incident_detection/evaluation/train_v2_tracks.py [--max-fa 1.0] [--save]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(AGENT_DIR.parents[1]))
sys.path.insert(0, str(EVAL_DIR))
from agents.incident_detection.src.feature_engineering import FEATURE_COLUMNS, add_temporal_features  # noqa: E402
from evaluate_v1 import V1_MODEL  # noqa: E402
from track_features import TRACK_FEATURES, window_track_features  # noqa: E402
from train_v2 import summary  # noqa: E402

V1_FEATURES = list(FEATURE_COLUMNS)
TRACKS = list(TRACK_FEATURES)
CANDIDATES = {
    "A_rule_spot_age": {"kind": "rule", "grid": np.arange(30, 605, 5.0)},
    "B_rf_v1_plus_tracks": {"kind": "rf", "features": V1_FEATURES + TRACKS, "grid": np.round(np.arange(0.05, 0.96, 0.025), 3)},
    "C_rf_tracks_only": {"kind": "rf", "features": TRACKS, "grid": np.round(np.arange(0.05, 0.96, 0.025), 3)},
}


def load(folder, split):
    frames = []
    for path in sorted(folder.glob("*.tracks.csv.gz")):
        windows = add_temporal_features(pd.read_csv(path.with_name(path.name.replace(".tracks.csv.gz", ".csv"))))
        tracks = window_track_features(pd.read_csv(path), float(windows.window_end_sec.max()))
        frames.append(windows.merge(tracks, on="window_start_sec", how="left", validate="1:1"))
    df = pd.concat(frames, ignore_index=True)
    train, test = set(split["town05"]["train_runs"]), set(split["town05"]["test_runs"])
    missing = (train | test) - set(df.run)
    if missing:
        raise ValueError(f"No track files for runs {sorted(missing)}")
    df["part"] = np.where(df.run.isin(train), "train", "test")
    df["stream"], df["group"] = df.run + "|" + df.camera, df.run
    return df.sort_values(["stream", "window_start_sec"]).reset_index(drop=True)


def fit_scores(candidate, fit, score):
    if candidate["kind"] == "rule":
        return None, score.max_spot_age_s.to_numpy()
    model = clone(V1_MODEL).fit(fit[candidate["features"]], fit.label)
    return model, model.predict_proba(score[candidate["features"]])[:, 1]


def oof(candidate, train):
    if candidate["kind"] == "rule":
        return train.max_spot_age_s.to_numpy()
    scores = np.zeros(len(train))
    for fit_idx, valid_idx in LeaveOneGroupOut().split(train, groups=train.group):
        _, scores[valid_idx] = fit_scores(candidate, train.iloc[fit_idx], train.iloc[valid_idx])
    return scores


def compact(s):
    keys = ("false_alarms_per_camera_hour", "normal_alarm_share", "events", "events_caught",
            "time_to_detect_median_s", "window_f2", "average_precision", "normal_hours")
    return {k: s[k] for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, default=AGENT_DIR / "data" / "town05" / "yolo11n-coco-tracks")
    parser.add_argument("--max-fa", type=float, default=1.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    split = json.loads((EVAL_DIR / "v2_split.json").read_text())
    df = load(args.tracks, split)
    train = df[df.part == "train"].reset_index(drop=True)
    test = df[df.part == "test"].reset_index(drop=True)
    seen = set(train.camera)
    print(f"train {len(train)} windows ({int(train.label.sum())} incident) | test {len(test)} ({int(test.label.sum())})", flush=True)

    # 1. Thresholds and the choice of candidate, on training data only.
    chosen_by_oof = {}
    for name, cand in CANDIDATES.items():
        scores = oof(cand, train)
        curve = [(float(t), summary(train, scores, t)) for t in cand["grid"]]
        ok = [(t, s) for t, s in curve if s["false_alarms_per_camera_hour"] <= args.max_fa]
        t, s = ok[0] if ok else curve[-1]
        chosen_by_oof[name] = {"threshold": t, "oof": compact(s),
                               "curve": [{"threshold": t_, **compact(s_)} for t_, s_ in curve]}
        print(f"OOF {name:22s} threshold {t:7.3f} | FA/h {s['false_alarms_per_camera_hour']} | "
              f"events {s['events_caught']}/{s['events']} | normal time in alarm {s['normal_alarm_share']:.2%}", flush=True)
    best = max(chosen_by_oof, key=lambda n: (chosen_by_oof[n]["oof"]["events_caught"],
                                             -chosen_by_oof[n]["oof"]["false_alarms_per_camera_hour"]))
    print("chosen on training data:", best, flush=True)

    # 2. Score once on test: every candidate (for the record) and v1.
    v1 = joblib.load(AGENT_DIR / "models" / "incident_detector.joblib")
    results = {"v1": {"threshold": float(v1["threshold"]), "scores": v1["model"].predict_proba(test[V1_FEATURES])[:, 1]}}
    models = {}
    for name, cand in CANDIDATES.items():
        models[name], scores = fit_scores(cand, train, test)
        results[name] = {"threshold": chosen_by_oof[name]["threshold"], "scores": scores}
    report_test = {}
    unseen = ~test.camera.isin(seen)
    for name, r in results.items():
        report_test[name] = {"threshold": r["threshold"], "all": summary(test, r["scores"], r["threshold"]),
                             "unseen_cameras": summary(test[unseen], r["scores"][unseen.to_numpy()], r["threshold"])}
        s = report_test[name]["all"]
        tag = " <- chosen" if name == best else ""
        print(f"TEST {name:22s} FA/h {s['false_alarms_per_camera_hour']} | normal time in alarm {s['normal_alarm_share']:.2%} | "
              f"events {s['events_caught']}/{s['events']} | TTD median {s['time_to_detect_median_s']} | "
              f"AP {s['average_precision']}{tag}", flush=True)
        for e in s["per_event"]:
            print(f"      {e['stream']:42s} {e['start_sec']:7.1f}-{e['end_sec']:7.1f}s  "
                  f"{'caught after ' + str(round(e['time_to_detect_s'])) + ' s' if e['caught'] else 'missed'}")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    out = EVAL_DIR / "results" / f"v2_tracks_{stamp}.json"
    out.write_text(json.dumps({"created": stamp, "split": "evaluation/v2_split.json", "max_fa": args.max_fa,
                               "selection": "most training accidents caught at OOF FA <= max_fa, chosen before test",
                               "chosen": best, "oof": chosen_by_oof, "test": report_test}, indent=1))
    print("saved", out)
    if args.save:
        cand = CANDIDATES[best]
        path = AGENT_DIR / "models" / "incident_detector_v2_tracks.joblib"
        joblib.dump({"candidate": best, "kind": cand["kind"], "model": models[best], "threshold": chosen_by_oof[best]["threshold"],
                     "temporal_rule": "mean3", "feature_columns": cand.get("features", ["max_spot_age_s"]),
                     "track_features": TRACKS, "window_seconds": 5.0, "split": "evaluation/v2_split.json",
                     "report": out.name}, path)
        print("saved", path)


if __name__ == "__main__":
    main()

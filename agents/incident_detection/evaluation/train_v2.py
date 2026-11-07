"""Train incident model v2 on AI City + Town05 training data and score v1 and v2 on the test sets.

v2 keeps v1's 27 features, Random Forest recipe and mean3 rule, so IncidentDetector loads it
unchanged; only the training data and the threshold change. The split is
evaluation/v2_split.json. The threshold rule is fixed in advance and uses out-of-fold scores on
training data only (grouped by AI City camera and by Town05 run):
    the lowest threshold whose Town05 false alarms are at most --max-fa per camera-hour.
Nothing in the test runs/videos is used before the final scoring.

    python agents/incident_detection/evaluation/train_v2.py [--max-fa 1.0] [--save]
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
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(AGENT_DIR.parents[1]))
sys.path.insert(0, str(EVAL_DIR))
from agents.incident_detection.src.feature_engineering import FEATURE_COLUMNS, add_temporal_features  # noqa: E402
from evaluate_v1 import V1_MODEL, camera_of  # noqa: E402

FEATURES = list(FEATURE_COLUMNS)
THRESHOLDS = np.round(np.arange(0.20, 0.96, 0.025), 3)


def load_ai_city(split):
    df = pd.read_csv(AGENT_DIR / "data" / "all_traffic_features_temporal.csv")
    df = df.sort_values(["video_id", "window_start_sec"]).reset_index(drop=True)
    test, excluded = set(split["ai_city"]["test_videos"]), set(split["ai_city"]["excluded_videos"])
    df["part"] = np.where(df.video_id.isin(test), "test", np.where(df.video_id.isin(excluded), "excluded", "train"))
    df["dataset"], df["stream"] = "ai_city", "aic_" + df.video_id.astype(str)
    cameras = camera_of(df.video_id.to_numpy(), EVAL_DIR / "camera_groups.json")
    df["group"] = "aic_cam_" + pd.Series(cameras).astype(str)
    df["camera"] = df["group"]
    return df[df.part != "excluded"]


def load_town05(split, folder):
    frames = []
    for path in sorted(folder.glob("*.csv")):
        windows = add_temporal_features(pd.read_csv(path))
        frames.append(windows)
    df = pd.concat(frames, ignore_index=True)
    train, test = set(split["town05"]["train_runs"]), set(split["town05"]["test_runs"])
    unknown = set(df.run) - train - test
    if unknown:
        raise ValueError(f"Runs not in v2_split.json: {sorted(unknown)}")
    df["part"] = np.where(df.run.isin(train), "train", "test")
    df["dataset"], df["stream"], df["group"] = "town05", df.run + "|" + df.camera, "t05_" + df.run
    return df


def mean3(df, scores):
    s = pd.Series(scores, index=df.index)
    return s.groupby(df["stream"]).transform(lambda x: x.rolling(3, min_periods=1).mean()).to_numpy()


def episodes(df, alarm):
    """Per stream: events (runs of label windows) caught or not, and false alarm episodes."""
    events, false_alarms, normal_hours, normal_alarm = [], 0, 0.0, 0
    for stream, g in df.assign(alarm=alarm).groupby("stream", sort=False):
        label, on = g.label.to_numpy().astype(bool), g.alarm.to_numpy()
        duration = (g.window_end_sec - g.window_start_sec).to_numpy()
        normal_hours += duration[~label].sum() / 3600
        normal_alarm += int((on & ~label).sum())
        for block in np.split(np.arange(len(g)), np.flatnonzero(np.diff(label.astype(int))) + 1):
            if label[block[0]]:
                hit = block[on[block]]
                events.append({"stream": stream, "start_sec": float(g.window_start_sec.iloc[block[0]]),
                               "end_sec": float(g.window_end_sec.iloc[block[-1]]), "caught": bool(len(hit)),
                               "time_to_detect_s": float(g.window_end_sec.iloc[hit[0]] - g.window_start_sec.iloc[block[0]])
                               if len(hit) else None})
        for block in np.split(np.arange(len(g)), np.flatnonzero(np.diff(on.astype(int))) + 1):
            if on[block[0]] and not label[block].any():
                false_alarms += 1
    normal_windows = int((df.label == 0).sum())
    return events, false_alarms, normal_hours, normal_alarm / max(normal_windows, 1)


def summary(df, scores, threshold):
    alarm = mean3(df, scores) >= threshold
    events, fa, hours, share = episodes(df, alarm)
    y = df.label.astype(int).to_numpy()
    ttd = [e["time_to_detect_s"] for e in events if e["caught"]]
    return {
        "windows": int(len(df)), "normal_hours": round(hours, 2),
        "false_alarm_episodes": fa, "false_alarms_per_camera_hour": round(fa / hours, 2) if hours else None,
        "normal_alarm_share": round(share, 4),
        "events": len(events), "events_caught": sum(e["caught"] for e in events),
        "time_to_detect_median_s": float(np.median(ttd)) if ttd else None,
        "window_precision": round(precision_score(y, alarm, zero_division=0), 4),
        "window_recall": round(recall_score(y, alarm, zero_division=0), 4),
        "window_f2": round(fbeta_score(y, alarm, beta=2, zero_division=0), 4),
        "average_precision": round(average_precision_score(y, scores), 4) if y.any() else None,
        "per_event": events,
    }


def oof_scores(train, seed):
    scores = np.zeros(len(train))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_idx, valid_idx in cv.split(train[FEATURES], train.label, train.group):
        model = clone(V1_MODEL).fit(train[FEATURES].iloc[fit_idx], train.label.iloc[fit_idx])
        scores[valid_idx] = model.predict_proba(train[FEATURES].iloc[valid_idx])[:, 1]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--town05", type=Path, default=AGENT_DIR / "data" / "town05" / "yolo11n-coco")
    parser.add_argument("--max-fa", type=float, default=1.0, help="false alarms per Town05 camera-hour (OOF)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="write models/incident_detector_v2.joblib")
    args = parser.parse_args()

    split = json.loads((EVAL_DIR / "v2_split.json").read_text())
    data = pd.concat([load_ai_city(split), load_town05(split, args.town05)], ignore_index=True)
    train = data[data.part == "train"].reset_index(drop=True)
    test = data[data.part == "test"].reset_index(drop=True)
    print("train windows", train.groupby("dataset").label.agg(["size", "sum"]).to_dict(),
          "| test windows", test.groupby("dataset").label.agg(["size", "sum"]).to_dict(), flush=True)

    # 1. Threshold from out-of-fold scores on training data only.
    oof = oof_scores(train, args.seed)
    curve = []
    for t in THRESHOLDS:
        row = {"threshold": float(t)}
        for name, part in train.groupby("dataset"):
            s = summary(part, oof[part.index], t)
            row[name] = {k: s[k] for k in ("false_alarms_per_camera_hour", "normal_alarm_share",
                                           "events", "events_caught", "window_f2")}
        curve.append(row)
    allowed = [r for r in curve if r["town05"]["false_alarms_per_camera_hour"] <= args.max_fa]
    threshold = allowed[0]["threshold"] if allowed else float(THRESHOLDS[-1])
    print(f"threshold {threshold} (lowest with Town05 OOF false alarms <= {args.max_fa}/h)", flush=True)

    # 2. Final v2 on all training data; 3. score v1 and v2 on the test sets.
    v2 = clone(V1_MODEL).fit(train[FEATURES], train.label)
    v1 = joblib.load(AGENT_DIR / "models" / "incident_detector.joblib")
    seen = set(train[train.dataset == "town05"].camera)
    results = {}
    for name, model, t in (("v1", v1["model"], float(v1["threshold"])), ("v2", v2, threshold)):
        scores = model.predict_proba(test[FEATURES])[:, 1]
        res = {}
        for dataset, part in test.groupby("dataset"):
            res[dataset] = summary(part, scores[part.index], t)
        t05 = test[test.dataset == "town05"]
        for label, mask in (("town05_seen_cameras", t05.camera.isin(seen)), ("town05_unseen_cameras", ~t05.camera.isin(seen))):
            part = t05[mask]
            res[label] = summary(part, scores[part.index], t) if len(part) else None
        res["threshold"] = t
        results[name] = res
        for dataset in ("town05", "ai_city", "town05_unseen_cameras"):
            s = res[dataset]
            print(f"{name} {dataset:22s} FA/h {s['false_alarms_per_camera_hour']} | normal time in alarm "
                  f"{s['normal_alarm_share']:.1%} | events {s['events_caught']}/{s['events']} | "
                  f"TTD median {s['time_to_detect_median_s']} | F2 {s['window_f2']} | AP {s['average_precision']}", flush=True)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    report = {"created": stamp, "split": "evaluation/v2_split.json", "features": len(FEATURES),
              "rule": "mean3", "threshold_rule": f"lowest threshold with Town05 OOF false alarms <= {args.max_fa} per camera-hour",
              "threshold": threshold, "seed": args.seed, "oof_curve": curve, "test": results}
    out = EVAL_DIR / "results" / f"v2_{stamp}.json"
    out.write_text(json.dumps(report, indent=1))
    print("saved", out)
    if args.save:
        path = AGENT_DIR / "models" / "incident_detector_v2.joblib"
        joblib.dump({"model": v2, "model_name": "Random Forest v2 (AI City + Town05)", "threshold": threshold,
                     "temporal_rule": "mean3", "feature_columns": FEATURES, "window_seconds": 5.0,
                     "split": "evaluation/v2_split.json", "report": out.name}, path)
        print("saved", path)


if __name__ == "__main__":
    main()

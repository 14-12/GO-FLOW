"""Stage-2 baseline: the v1 recipe under video- and camera-grouped cross-validation.

The v1 recipe is fixed (Random Forest hyperparameters, mean3 rule, threshold 0.35), so no
choice is made on the folds being scored. Reports window metrics, per-event recall, time to
detect and false alarms per camera-hour, with bootstrap 95 % intervals over events/videos.

    python agents/incident_detection/evaluation/evaluate_v1.py [--seeds 3] [--bootstrap 2000]
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

AGENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = AGENT_DIR / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import sys
sys.path.insert(0, str(AGENT_DIR.parents[1]))
from agents.incident_detection.src.feature_engineering import FEATURE_COLUMNS  # noqa: E402

THRESHOLD = 0.35
WINDOW_SECONDS = 5.0
V1_MODEL = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(n_estimators=500, max_depth=16, min_samples_leaf=2,
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
])


def camera_of(video_ids, groups_file):
    camera = {int(v): int(v) for v in video_ids}
    for group in json.loads(groups_file.read_text())["groups"]:
        for v in group:
            camera[int(v)] = int(min(group))
    return np.array([camera[int(v)] for v in video_ids])


def oof_scores(df, groups, seed):
    scores = np.zeros(len(df))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    X, y = df[list(FEATURE_COLUMNS)], df["label"].astype(int)
    for train_idx, valid_idx in cv.split(X, y, groups):
        model = clone(V1_MODEL).fit(X.iloc[train_idx], y.iloc[train_idx])
        scores[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return scores


def mean3_alarms(df, scores):
    s = pd.Series(scores, index=df.index)
    return (s.groupby(df["video_id"]).transform(lambda x: x.rolling(3, min_periods=1).mean()) >= THRESHOLD).to_numpy()


def event_table(df, alarm, intervals):
    """One row per annotated anomaly interval: detected?, seconds from anomaly start to first alarm."""
    rows = []
    for _, ev in intervals.iterrows():
        mid = (df["window_start_sec"] + df["window_end_sec"]) / 2
        inside = (df["video_id"] == ev.video_id) & (mid >= ev.start_sec) & (mid <= ev.end_sec)
        hits = df.loc[inside & alarm, "window_end_sec"]
        # An alarm on a window is known at its end; anomalies at t=0 started before the clip.
        ttd = max(0.0, float(hits.min()) - float(ev.start_sec)) if len(hits) else None
        rows.append({"video_id": int(ev.video_id), "start_sec": float(ev.start_sec),
                     "end_sec": float(ev.end_sec), "detected": bool(len(hits)), "time_to_detect_s": ttd})
    return pd.DataFrame(rows)


def false_alarms_per_video(df, alarm):
    """Alarm episodes (runs of consecutive alarm windows) containing no incident window."""
    out = {}
    for vid, g in df.assign(alarm=alarm).groupby("video_id", sort=False):
        run_id = (g["alarm"] != g["alarm"].shift()).cumsum()
        episodes = g[g["alarm"]].groupby(run_id[g["alarm"]])["label"].max()
        normal_hours = float((g["window_end_sec"] - g["window_start_sec"])[g["label"] == 0].sum()) / 3600
        out[int(vid)] = (int((episodes == 0).sum()), normal_hours)
    return out


def summarise(df, scores, intervals, rng, n_boot):
    y = df["label"].astype(int).to_numpy()
    alarm = mean3_alarms(df, scores)
    events = event_table(df, alarm, intervals)
    fa = false_alarms_per_video(df, alarm)
    videos = np.array(sorted(fa))
    fa_count = np.array([fa[v][0] for v in videos])
    fa_hours = np.array([fa[v][1] for v in videos])
    ttd = events["time_to_detect_s"].dropna().to_numpy()

    def ci(values):
        return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]

    boot_recall, boot_fa, boot_f2 = [], [], []
    by_video = {v: np.flatnonzero(df["video_id"].to_numpy() == v) for v in videos}
    for _ in range(n_boot):
        boot_recall.append(events["detected"].to_numpy()[rng.integers(0, len(events), len(events))].mean())
        pick = rng.integers(0, len(videos), len(videos))
        boot_fa.append(fa_count[pick].sum() / max(fa_hours[pick].sum(), 1e-9))
        idx = np.concatenate([by_video[videos[i]] for i in pick])
        boot_f2.append(fbeta_score(y[idx], alarm[idx], beta=2, zero_division=0))

    return {
        "window": {
            "precision": round(precision_score(y, alarm, zero_division=0), 4),
            "recall": round(recall_score(y, alarm, zero_division=0), 4),
            "f2": round(fbeta_score(y, alarm, beta=2, zero_division=0), 4), "f2_ci95": ci(boot_f2),
            "average_precision": round(average_precision_score(y, scores), 4),
            "roc_auc": round(roc_auc_score(y, scores), 4),
        },
        "events": {
            "n_events": int(len(events)), "detected": int(events["detected"].sum()),
            "event_recall": round(float(events["detected"].mean()), 4), "event_recall_ci95": ci(boot_recall),
            "time_to_detect_median_s": float(np.median(ttd)) if len(ttd) else None,
            "time_to_detect_p90_s": float(np.percentile(ttd, 90)) if len(ttd) else None,
            "per_event": events.to_dict(orient="records"),
        },
        "false_alarms": {
            "episodes": int(fa_count.sum()), "normal_hours": round(float(fa_hours.sum()), 2),
            "per_camera_hour": round(float(fa_count.sum() / fa_hours.sum()), 3), "per_camera_hour_ci95": ci(boot_fa),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="fold assignments to average over")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(DATA_DIR / "all_traffic_features_temporal.csv")
    df = df.sort_values(["video_id", "window_start_sec"]).reset_index(drop=True)
    intervals = pd.read_csv(DATA_DIR / "anomaly_intervals.csv")
    schemes = {
        "video_grouped": df["video_id"].to_numpy(),
        "camera_grouped": camera_of(df["video_id"].to_numpy(), Path(__file__).with_name("camera_groups.json")),
    }
    report = {"created": datetime.now().isoformat(timespec="seconds"), "recipe": {
        "model": "RandomForest v1 hyperparameters", "rule": "mean3", "threshold": THRESHOLD,
        "features": len(FEATURE_COLUMNS), "folds": 5, "seeds": list(range(args.seeds)),
        "data": "all 100 labelled AI City Track 4 videos (train+val+test pooled)",
        "note": "threshold 0.35 was chosen in the notebook on train+val OOF scores, so it is slightly optimistic here"},
        "n_cameras": {k: int(len(np.unique(v))) for k, v in schemes.items()}, "results": {}}

    for name, groups in schemes.items():
        report["results"][name] = []
        for seed in range(args.seeds):
            started = time.time()
            scores = oof_scores(df, groups, seed)
            summary = summarise(df, scores, intervals, np.random.default_rng(seed), args.bootstrap)
            summary["seed"] = seed
            report["results"][name].append(summary)
            w, e, f = summary["window"], summary["events"], summary["false_alarms"]
            print(f"{name:15s} seed {seed}: F2 {w['f2']:.3f} {w['f2_ci95']} | AP {w['average_precision']:.3f} | "
                  f"events {e['detected']}/{e['n_events']} | TTD median {e['time_to_detect_median_s']} s | "
                  f"FA/h {f['per_camera_hour']:.2f} {f['per_camera_hour_ci95']} | {time.time() - started:.0f}s",
                  flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"v1_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()

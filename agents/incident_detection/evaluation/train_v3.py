"""Train the Town05 accident model on the v3 training runs and, once, score it on the v3 test runs.

Every choice is made on training runs only, from out-of-fold scores grouped by run:
    the candidate that catches most training crashes within 120 s at no more than --max-fa false alarms per
    camera-hour, ties going to more crashes caught at all, then the faster median detection; its threshold is
    the lowest that keeps to --max-fa. Each candidate is also scored with one camera held out at a time, because
    all training crashes come from four cameras (a model that only knows the camera fails this check).
v1 stays as it is; the new model is a separate file with its own feature list.

    python agents/incident_detection/evaluation/train_v3.py            # training runs only, safe to repeat
    python agents/incident_detection/evaluation/train_v3.py --final    # refuses to run a second time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

EVAL_DIR = Path(__file__).resolve().parent
AGENT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(AGENT_DIR.parents[1]))
sys.path.insert(0, str(EVAL_DIR))
from agents.incident_detection.src.feature_engineering import FEATURE_COLUMNS  # noqa: E402
import v3_features  # noqa: E402
from v3_features import ALL_FEATURES, EVIDENCE_FEATURES, build  # noqa: E402

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.025), 3)
FOLDS = 5
CLIPS = AGENT_DIR.parents[2] / "Urban Traffic CARLA" / "out" / "incident_clips"
FOREST = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=2,
                                class_weight="balanced_subsample", n_jobs=6, random_state=0)
CANDIDATES = {"evidence_only": list(EVIDENCE_FEATURES), "v1_plus_evidence": list(ALL_FEATURES),
              "v1_features_only": list(FEATURE_COLUMNS)}
MAX_MEDIAN_TTD_S, MAX_TTD_S, MIN_RECALL, MAX_FA = 30.0, 120.0, 0.8, 1.0


def mean3(df, scores):
    s = pd.Series(scores, index=df.index)
    return s.groupby(df["stream"]).transform(lambda x: x.rolling(3, min_periods=1).mean()).to_numpy()


def score_alarms(df, alarm):
    """Crashes caught and false alarm episodes. An alarm touching any wreck in the picture is not false."""
    events, false_alarms, listed = [], 0, []
    hours = 0.0
    normal_on = normal_all = 0
    for stream, g in df.assign(alarm=alarm).groupby("stream", sort=False):
        target, on = g.target.to_numpy(), g.alarm.to_numpy()
        starts, ends = g.window_start_sec.to_numpy(), g.window_end_sec.to_numpy()
        hours += (ends - starts)[target == 0].sum() / 3600
        normal_on += int((on & (target == 0)).sum())
        normal_all += int((target == 0).sum())
        blocks = np.split(np.arange(len(g)), np.flatnonzero(np.diff((target == 1).astype(int))) + 1)
        for block in blocks:
            if target[block[0]] == 1:
                hit = block[on[block]]
                events.append({"stream": stream, "incident": int(g.incident.iloc[block[0]]),
                               "start_sec": float(starts[block[0]]), "caught": bool(len(hit)),
                               "time_to_detect_s": float(ends[hit[0]] - starts[block[0]]) if len(hit) else None})
        for block in np.split(np.arange(len(g)), np.flatnonzero(np.diff(on.astype(int))) + 1):
            if on[block[0]] and (target[block] == 0).all():
                false_alarms += 1
                listed.append({"stream": stream, "start_sec": float(starts[block[0]]),
                               "end_sec": float(ends[block[-1]])})
    ttd = [e["time_to_detect_s"] for e in events if e["caught"]]
    return {"normal_hours": round(hours, 2), "false_alarms": false_alarms,
            "false_alarms_per_camera_hour": round(false_alarms / hours, 2) if hours else None,
            "normal_time_in_alarm": round(normal_on / max(normal_all, 1), 4),
            "events": len(events), "caught": len(ttd), "caught_within_120s": sum(t <= MAX_TTD_S for t in ttd),
            "median_detect_s": float(np.median(ttd)) if ttd else None,
            "slowest_detect_s": float(np.max(ttd)) if ttd else None,
            "per_event": events, "false_alarm_list": listed}


def fit_rows(df):
    return df[df.target >= 0]


def oof_scores(train, features, group="run"):
    scores = np.zeros(len(train))
    folds = min(FOLDS, train[group].nunique())
    for fit_idx, valid_idx in GroupKFold(n_splits=folds).split(train, groups=train[group]):
        fold = fit_rows(train.iloc[fit_idx])
        model = clone(FOREST).fit(fold[features], fold.target)
        scores[valid_idx] = model.predict_proba(train.iloc[valid_idx][features])[:, 1]
    return scores


def choose(train, oof, max_fa):
    """The lowest threshold within the false alarm limit, with its result on training data."""
    smoothed = mean3(train, oof)
    for t in THRESHOLDS:
        result = score_alarms(train, smoothed >= t)
        if result["false_alarms_per_camera_hour"] <= max_fa:
            return float(t), result
    return float(THRESHOLDS[-1]), score_alarms(train, smoothed >= THRESHOLDS[-1])


def verdict(result):
    recall = result["caught"] / result["events"] if result["events"] else 0.0
    return {"false_alarms": bool(result["false_alarms_per_camera_hour"] <= MAX_FA), "recall": bool(recall >= MIN_RECALL),
            "median_detect": bool(result["median_detect_s"] is not None and result["median_detect_s"] <= MAX_MEDIAN_TTD_S),
            "slowest_detect": bool(result["slowest_detect_s"] is not None and result["slowest_detect_s"] <= MAX_TTD_S)}


def brief(result):
    return {k: v for k, v in result.items() if k not in ("per_event", "false_alarm_list")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-fa", type=float, default=MAX_FA)
    parser.add_argument("--final", action="store_true", help="score the test runs; allowed once")
    parser.add_argument("--refresh", action="store_true", help="rebuild the cached training features")
    parser.add_argument("--data", default="v3", choices=sorted(v3_features.LAGS), help="recorded data set")
    args = parser.parse_args()
    v3_features.configure(args.data)
    final_result = EVAL_DIR / "results" / f"{args.data}_final_test.json"
    model_path = AGENT_DIR / "models" / ("incident_detector_v2.joblib" if args.data == "v3"
                                         else f"incident_detector_town05_{args.data}.joblib")
    if args.final and final_result.exists():
        raise SystemExit(f"{final_result} exists: the test runs were already scored once.")

    train = build("train", CLIPS, refresh=args.refresh).sort_values(["stream", "window_start_sec"]).reset_index(drop=True)
    print(f"training: {train.stream.nunique()} recordings, windows per target {train.groupby('target').size().to_dict()}",
          flush=True)
    report = {"created": datetime.now().isoformat(timespec="seconds"), "max_fa": args.max_fa, "candidates": {}}
    best = None
    for name, features in CANDIDATES.items():
        oof = oof_scores(train, features)
        threshold, result = choose(train, oof, args.max_fa)
        _, other_cameras = choose(train, oof_scores(train, features, group="camera"), args.max_fa)
        report["candidates"][name] = {"threshold": threshold, "train_oof": result,
                                      "train_oof_other_cameras": other_cameras}
        print(f"{name:18s} threshold {threshold}: {result['caught']}/{result['events']} crashes, "
              f"{result['caught_within_120s']} within 120 s, {result['false_alarms_per_camera_hour']} false alarms/h, "
              f"median {result['median_detect_s']} s, slowest {result['slowest_detect_s']} s | camera held out: "
              f"{other_cameras['caught']}/{other_cameras['events']}, {other_cameras['caught_within_120s']} within 120 s, "
              f"{other_cameras['false_alarms_per_camera_hour']} false alarms/h", flush=True)
        key = (result["false_alarms_per_camera_hour"] <= args.max_fa, result["caught_within_120s"], result["caught"],
               -(result["median_detect_s"] if result["median_detect_s"] is not None else 1e9))
        if best is None or key > best[0]:
            best = (key, name, threshold)
    _, name, threshold = best
    report["chosen"], report["threshold"] = name, threshold
    print("chosen:", name, "threshold", threshold, flush=True)

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    (EVAL_DIR / "results" / f"{args.data}_train_{stamp}.json").write_text(json.dumps(report, indent=1))
    if not args.final:
        return

    features = CANDIDATES[name]
    model = clone(FOREST).fit(fit_rows(train)[features], fit_rows(train).target)
    test = build("test", CLIPS, refresh=True).sort_values(["stream", "window_start_sec"]).reset_index(drop=True)
    assert not set(test.run) & set(train.run)
    plan = json.loads((EVAL_DIR / f"{args.data}_split.json").read_text(encoding="utf-8"))
    alarm = mean3(test, model.predict_proba(test[features])[:, 1]) >= threshold
    v1 = joblib.load(AGENT_DIR / "models" / "incident_detector.joblib")
    v1_alarm = mean3(test, v1["model"].predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]) >= float(v1["threshold"])

    final = {"created": stamp, "chosen": name, "threshold": threshold, "test_runs": sorted(test.run.unique()),
             "normal_hours": round((test.target == 0).sum() * 5 / 3600, 2)}
    unseen = test.camera.isin(plan["unseen_test_cameras"]).to_numpy()
    final["v2_all"] = score_alarms(test, alarm)
    final["v2_seen_cameras"] = score_alarms(test[~unseen], alarm[~unseen])
    final["v2_unseen_cameras"] = score_alarms(test[unseen], alarm[unseen])
    final["v1_all"] = score_alarms(test, v1_alarm)
    final["criteria_met"] = verdict(final["v2_all"])
    final["passed"] = all(final["criteria_met"].values())
    final_result.write_text(json.dumps(final, indent=1))
    for key in ("v2_all", "v2_seen_cameras", "v2_unseen_cameras", "v1_all"):
        print(key, json.dumps(brief(final[key])), flush=True)
    print("criteria", final["criteria_met"], "passed", final["passed"], flush=True)
    joblib.dump({"model": model, "features": features, "threshold": threshold, "rule": "mean3",
                 "detector": "carla-yolo26n-t05", "fps": 5, "passed_acceptance": final["passed"],
                 "trained_on": sorted(train.run.unique())}, model_path)


if __name__ == "__main__":
    main()

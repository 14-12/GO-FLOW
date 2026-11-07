"""Training-only look at the accident features (crash_features.py). Never reads a test run.

For each feature: how often it fires in normal traffic (per camera-hour) and in accident time,
and for simple one-feature rules (alarm when feature >= threshold, no smoothing) the false alarm
episodes per camera-hour and the training accidents caught. Exploratory: only two training
accidents exist, so nothing here is a validated result.

    python agents/incident_detection/evaluation/eval_crash_features.py
"""
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
from crash_features import CRASH_FEATURES, window_crash_features  # noqa: E402
from train_v2 import episodes  # noqa: E402

TRACKS = AGENT_DIR / "data" / "town05" / "yolo11n-coco-tracks"
GRIDS = {
    "hard_stops": [1, 2, 3], "contact_hard_stops": [1, 2, 3], "max_speed_drop": [1, 1.5, 2, 3, 4],
    "crash_born_spot_age_s": [5, 20, 60, 120], "passing_same_direction": [1, 2, 3, 5],
    "passing_any_direction": [1, 2, 3, 5],
}


def main():
    split = json.loads((EVAL_DIR / "v2_split.json").read_text())
    train_runs = set(split["town05"]["train_runs"])
    frames = []
    for path in sorted(TRACKS.glob("*.tracks.csv.gz")):
        run = path.name.split("__")[0]
        if run not in train_runs:
            continue
        assert run not in split["town05"]["test_runs"]
        windows = pd.read_csv(path.with_name(path.name.replace(".tracks.csv.gz", ".csv")))
        feats = window_crash_features(pd.read_csv(path), float(windows.window_end_sec.max()))
        w = windows.merge(feats, on="window_start_sec", how="left", validate="1:1")
        w["stream"] = run + "|" + w.camera
        frames.append(w)
    df = pd.concat(frames, ignore_index=True).sort_values(["stream", "window_start_sec"]).reset_index(drop=True)
    normal_hours = (df.label == 0).sum() * 5 / 3600
    print(f"training runs only: {df.stream.nunique()} recordings, {normal_hours:.1f} normal hours, "
          f"{int(df.label.sum())} accident windows")

    report = {"created": datetime.now().isoformat(timespec="seconds"), "runs": sorted(train_runs), "features": {}}
    for feature in CRASH_FEATURES:
        on = df[feature] > 0
        rate_normal = on[df.label == 0].sum() / normal_hours
        share_crash = on[df.label == 1].mean()
        print(f"\n{feature}: fires in {rate_normal:.1f} normal windows per camera-hour; "
              f"in {share_crash:.0%} of accident windows")
        report["features"][feature] = {"normal_windows_per_hour": round(float(rate_normal), 2),
                                       "accident_window_share": round(float(share_crash), 3), "rules": []}
        for t in GRIDS[feature]:
            alarm = (df[feature] >= t).to_numpy()
            events, fa, hours, share = episodes(df, alarm)
            caught = [f"{e['stream'].split('|')[1]}: " + (f"after {e['time_to_detect_s']:.0f} s" if e["caught"] else "missed")
                      for e in events]
            print(f"   >= {t:<5} FA/h {fa / hours:6.2f} | normal time in alarm {share:6.2%} | {'; '.join(caught)}")
            report["features"][feature]["rules"].append({"threshold": t, "false_alarms_per_hour": round(fa / hours, 2),
                                                         "normal_alarm_share": round(share, 4), "events": events})
    out = EVAL_DIR / "results" / f"crash_features_train_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1))
    print("\nsaved", out)
    return df


if __name__ == "__main__":
    main()

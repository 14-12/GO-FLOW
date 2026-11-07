import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/incident_detection/evaluation"))
from train_v3 import score_alarms, verdict  # noqa: E402
from v3_features import emergency_features, targets  # noqa: E402


def windows(target, stream="r|c"):
    n = len(target)
    return pd.DataFrame({"stream": stream, "target": target, "incident": [1 if t else -1 for t in target],
                         "window_start_sec": np.arange(n) * 5.0, "window_end_sec": np.arange(n) * 5.0 + 5})


def test_alarm_on_a_far_wreck_is_neither_right_nor_wrong():
    df = windows([0, 0, -1, -1, 0, 0])
    r = score_alarms(df, np.array([0, 0, 1, 1, 0, 0], dtype=bool))
    assert r["false_alarms"] == 0 and r["events"] == 0
    assert r["normal_hours"] == round(4 * 5 / 3600, 2)


def test_alarm_in_normal_time_is_false_and_a_crash_is_timed_from_its_first_window():
    df = windows([0, 0, 0, 1, 1, 1, 0, 0])
    r = score_alarms(df, np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=bool))
    assert r["false_alarms"] == 1 and r["caught"] == 1
    assert r["median_detect_s"] == 10.0     # window 4 ends 10 s after window 3 starts


def test_results_and_verdict_can_be_saved_as_json():
    df = windows([0, 0, 0, 1, 1, 1, 0, 0])
    r = score_alarms(df, np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=bool))
    assert json.loads(json.dumps({"result": r, "verdict": verdict(r)}))["verdict"]["recall"] is True


def test_targets_split_counted_crashes_from_other_wrecks():
    w = pd.DataFrame({"sim_start_s": [0.0, 100.0, 200.0, 300.0], "sim_end_s": [5.0, 105.0, 205.0, 305.0]})
    visible = [{"camera": "c", "incident": 1, "from_sim_s": 90, "to_sim_s": 110},
               {"camera": "c", "incident": 2, "from_sim_s": 190, "to_sim_s": 210},
               {"camera": "other", "incident": 3, "from_sim_s": 290, "to_sim_s": 310}]
    target, incident = targets(w, "c", {1}, visible)
    assert target.tolist() == [0, 1, -1, 0]
    assert incident.tolist() == [-1, 1, 2, -1]


def test_only_lights_on_vehicles_count_and_standing_still_is_told_apart():
    rows = []
    for i in range(50):
        t = i / 5
        rows.append({"source_time_sec": t, "track_id": 1, "class_id": 7, "confidence": 0.9, "center_x": 100.0,
                     "center_y": 100.0, "bbox_height": 40.0})      # police lights, standing
        rows.append({"source_time_sec": t, "track_id": 2, "class_id": 5, "confidence": 0.9, "center_x": 10.0 * i,
                     "center_y": 200.0, "bbox_height": 40.0})      # ambulance lights, driving
        rows.append({"source_time_sec": t, "track_id": 3, "class_id": 0, "confidence": 0.9, "center_x": 300.0,
                     "center_y": 300.0, "bbox_height": 40.0})      # a parked car
    f = emergency_features(pd.DataFrame(rows), 10.0).set_index("window_start_sec")
    assert f.loc[5.0, "emergency_lights_frames"] == 25
    assert f.loc[5.0, "emergency_lights_stopped_frames"] == 25     # only the police car

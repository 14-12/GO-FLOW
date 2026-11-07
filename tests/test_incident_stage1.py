import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_stage1", REPO / "agents/incident_detection/evaluation/run_stage1.py")
run_stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_stage1)


def windows(alarms, scores=None):
    return [{"window_start_sec": 5.0 * i, "window_end_sec": 5.0 * (i + 1), "incident_detected": bool(a),
             "incident_score": (scores or [0.5] * len(alarms))[i]} for i, a in enumerate(alarms)]


def clip(incidents, duration=60.0):
    return {"duration_s": duration, "incidents_in_picture": incidents}


def incident(number, on_screen, wreck_end, before=False):
    return {"incident": number, "kind": "rear-end", "severity_label": "severe", "crash_before_clip": before,
            "crash_on_screen_clip_s": on_screen, "wreck_in_picture_clip_s": [max(0.0, on_screen - 2.2), wreck_end]}


def test_accident_caught_with_time_to_detect_from_on_screen_second():
    # 12 windows; the crash shows at 21.7 s and the wrecks stay to 40 s (+2.2 s lag).
    summary, labels = run_stage1.score_clip(
        windows([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]), clip([incident(3, 21.7, 40.0)]), lag=2.2)
    (accident,) = summary["accidents"]
    assert accident["view_clip_s"] == [21.7, 42.2]
    assert labels == [False] * 4 + [True] * 4 + [False] * 4
    assert accident["caught"] and accident["first_alarm_clip_s"] == 30.0
    assert abs(accident["time_to_detect_s"] - 8.3) < 1e-9
    assert summary["false_alarm_episodes"] == 0


def test_false_alarm_episodes_per_normal_hour_and_crash_before_clip():
    summary, _ = run_stage1.score_clip(
        windows([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]), clip([incident(1, -2.8, 7.0, before=True)]), lag=2.2)
    (accident,) = summary["accidents"]
    assert accident["view_clip_s"] == [0.0, 9.2] and accident["time_to_detect_s"] == 5.0
    # Windows 3-4 and 9 are two false episodes over 50 s of normal windows.
    assert summary["false_alarm_episodes"] == 2
    assert summary["false_alarm_clip_s"] == [[15.0, 25.0], [45.0, 50.0]]
    assert summary["false_alarms_per_camera_hour"] == round(2 / (50 / 3600), 2)


def test_missed_accident_and_normal_clip():
    summary, _ = run_stage1.score_clip(windows([0] * 6), clip([incident(4, 11.0, 29.5)], duration=30.0), lag=2.2)
    assert summary["accidents"][0]["caught"] is False and summary["accidents"][0]["time_to_detect_s"] is None
    summary, labels = run_stage1.score_clip(windows([0, 1, 0]), clip([], duration=15.0), lag=2.2)
    assert not any(labels) and summary["accidents"] == [] and summary["false_alarm_episodes"] == 1

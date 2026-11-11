"""The streaming adapter against Omar's own functions run once on the whole stream (his batch path)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import stream  # noqa: E402
from stream import IMPACT, StreamingWindows, window_of  # noqa: E402
from agents.incident_detection.src.feature_engineering import CURRENT_COLUMNS  # noqa: E402
from agents.incident_detection.src.inference import IncidentDetector  # noqa: E402
from agents.incident_detection.src.video_processing import aggregate_windows  # noqa: E402
from crash_features import window_crash_features  # noqa: E402

DETECTOR = IncidentDetector()


def synthetic(seed, seconds=120.0, fps=5.0, crash_at=None, gaps=(), start=1003.3, drop=0.0):
    """Pictures (t, frame row, detection rows) of a road: cars drive down the picture; a crash makes two stop dead,
    touching. gaps: (from, to) sim seconds with no picture. drop: share of pictures lost at random."""
    rng = random.Random(seed)
    cars, nid, out = [], 1, []
    t = start
    while t < start + seconds:
        t = round(t + 1.0 / fps, 6)
        if any(a <= t < b for a, b in gaps) or rng.random() < drop:
            for c in cars:
                c["y"] += c["v"] / fps
            continue
        if rng.random() < 0.25:
            cars.append({"id": nid, "x": rng.uniform(200, 760), "y": 60.0, "v": rng.uniform(60, 140), "h": 40.0,
                         "cls": rng.choice([0, 0, 0, 1, 2, 3]), "stop": False})
            nid += 1
        if crash_at is not None and abs(t - crash_at) < 0.5 / fps and len(cars) >= 2:
            a, b = cars[-1], cars[-2]
            a["x"], a["y"], a["stop"], b["stop"] = b["x"] + 10, b["y"] + 20, True, True
        dets, cov, confs = [], 0.0, []
        for c in cars:
            if not c["stop"]:
                c["y"] += c["v"] / fps
            if c["y"] > 540:
                continue
            conf = rng.uniform(0.3, 0.95)
            w = c["h"] * 1.6
            cat = ("car", "car", "bus", "truck")[c["cls"]]
            dets.append({"track_id": c["id"], "class_id": c["cls"], "category": cat,
                         "confidence": conf, "center_x": c["x"] + rng.uniform(-1, 1), "center_y": c["y"],
                         "bbox_height": c["h"], "bbox_area": c["h"] * w})
            cov += c["h"] * w
            confs.append(conf)
        cars = [c for c in cars if c["y"] <= 540]
        out.append((t, {"vehicle_count": len(dets), "bbox_coverage": min(cov / (960 * 540), 1.0),
                        "mean_detection_confidence": float(np.mean(confs)) if confs else 0.0}, dets))
    return out


def batch(pictures, t0, end):
    """His path: every row of the whole stream at once."""
    frames = [dict(f, source_time_sec=t - t0) for t, f, _ in pictures]
    dets = [dict(d, source_time_sec=t - t0) for t, _, ds in pictures for d in ds]
    duration = end - t0
    win = aggregate_windows(frames, dets, duration)
    verdicts = DETECTOR.predict_windows(win, "cam")
    crash = window_crash_features(pd.DataFrame(dets), duration) if dets else None
    return win, verdicts, crash


def streamed(pictures, end):
    s = StreamingWindows("cam", DETECTOR)
    out = []
    for t, f, d in pictures:
        out += s.add(t, f, d)
    out += s.flush(end)
    return s, out


@pytest.mark.parametrize("case", [
    dict(seed=1),
    dict(seed=2, crash_at=1050.0),
    dict(seed=3, crash_at=1070.0, gaps=((1030.0, 1046.0),)),
    dict(seed=4, drop=0.3, crash_at=1040.0),
    dict(seed=5, fps=10.0, seconds=60.0),
])
def test_streaming_equals_his_batch_path(case):
    pictures = synthetic(**case)
    end = pictures[-1][0]
    s, out = streamed(pictures, end)
    win, verdicts, crash = batch(pictures, s.t0, end)
    assert len(out) == len(verdicts) == len(win)
    for o, v, (_, w) in zip(out, verdicts, win.iterrows()):
        assert o["window_start_sec"] == pytest.approx(v["window_start_sec"])
        assert o["window_end_sec"] == pytest.approx(v["window_end_sec"])
        for k in CURRENT_COLUMNS:
            assert o["traffic_features"][k] == pytest.approx(v["traffic_features"][k], abs=1e-9), k
        assert o["incident_score"] == pytest.approx(v["incident_score"], abs=1e-12)
        assert o["recent_scores"] == pytest.approx(v["recent_scores"], abs=1e-12)
        assert o["incident_detected"] == v["incident_detected"]
        assert o["sim_start_s"] == pytest.approx(s.t0 + v["window_start_sec"])
        if crash is not None:
            c = crash[np.isclose(crash.window_start_sec, v["window_start_sec"])].iloc[0]
            for k in IMPACT:
                assert o["impact"][k] == pytest.approx(float(c[k]), abs=1e-9), k


def test_a_crash_shows_as_a_contact_hard_stop():
    pictures = synthetic(seed=2, crash_at=1050.0)
    _, out = streamed(pictures, pictures[-1][0])
    hit = [o for o in out if o["impact_alarm"]]
    assert hit and hit[0]["sim_start_s"] <= 1055.0 + 1e-9


def test_windows_follow_the_simulation_clock():
    assert window_of(1003.3) == 1000.0 and window_of(1004.999999) == 1000.0 and window_of(1005.0) == 1005.0
    pictures = synthetic(seed=1, seconds=30.0)
    s, out = streamed(pictures, pictures[-1][0])
    assert s.t0 == 1000.0
    assert [o["sim_start_s"] for o in out][:3] == [1000.0, 1005.0, 1010.0]


def test_late_and_repeated_pictures_are_dropped():
    s = StreamingWindows("cam", DETECTOR)
    f = {"vehicle_count": 0, "bbox_coverage": 0.0, "mean_detection_confidence": 0.0}
    s.add(10.0, f, [])
    assert s.add(10.0, f, []) == [] and s.add(9.0, f, []) == []
    assert s.stats["late"] == 2 and s.stats["pictures"] == 1


def test_window_with_no_picture_is_a_zero_window():
    s = StreamingWindows("cam", DETECTOR)
    f = {"vehicle_count": 1, "bbox_coverage": 0.1, "mean_detection_confidence": 0.5}
    s.add(1.0, f, [])
    out = s.add(17.0, f, [])
    assert [o["sim_start_s"] for o in out] == [0.0, 5.0, 10.0]
    assert out[1]["pictures"] == 0 and out[1]["traffic_features"]["mean_vehicle_count_per_frame"] == 0.0
    assert s.stats["empty_windows"] == 2


def test_serve_protocol(tmp_path):
    pictures = synthetic(seed=2, crash_at=1050.0, seconds=40.0)
    lines = [json.dumps({"op": "hello"})]
    for t, f, d in pictures:
        lines.append(json.dumps({"op": "frame", "cam": "A", "t": t, "frame": f,
                                 "dets": [[x[k] for k in stream.DET_KEYS] for x in d]}))
    lines += [json.dumps({"op": "frame", "cam": "A", "t": "bad"}), json.dumps({"op": "flush"}), "not json",
              json.dumps({"op": "stop"})]
    p = subprocess.run([sys.executable, str(HERE / "serve.py")], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, timeout=120)
    out = [json.loads(x) for x in p.stdout.splitlines()]
    assert out[0]["op"] == "ready" and out[0]["model"]["sha256_12"] == stream.model_sha(
        HERE / "models" / "incident_detector.joblib")
    windows = [o for o in out if o["op"] == "window"]
    _, ref = streamed(pictures, pictures[-1][0])
    assert [w["sim_start_s"] for w in windows] == [r["sim_start_s"] for r in ref]
    assert [w["incident_score"] for w in windows] == pytest.approx([r["incident_score"] for r in ref])
    assert sum(o["op"] == "error" for o in out) == 2
    assert out[-1]["op"] == "bye" and p.returncode == 0

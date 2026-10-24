import pandas as pd
import numpy as np
import pytest

from agents.incident_detection.src import IncidentDetector
from agents.incident_detection.src.feature_engineering import CURRENT_COLUMNS, FEATURE_COLUMNS, add_temporal_features
from agents.incident_detection.src.video_processing import aggregate_windows, process_video
from src.incident_pipeline import run_traffic_pipeline


def windows():
    rows = []
    for i in range(4):
        row = {name: float(i) for name in CURRENT_COLUMNS}
        row.update(window_start_sec=i * 5.0, window_end_sec=(i + 1) * 5.0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_artifact_schema_and_causal_temporal_features():
    detector = IncidentDetector()
    assert detector.feature_columns == FEATURE_COLUMNS
    assert detector.temporal_rule == "mean3"
    assert detector.threshold == pytest.approx(0.35)
    frame = add_temporal_features(windows())
    assert frame.delta_unique_vehicle_count.tolist() == [0, 1, 1, 1]
    assert frame.roll3_unique_vehicle_count.tolist() == [0, 0.5, 1, 2]


def test_mean3_only_uses_past_scores():
    class Model:
        def predict_proba(self, frame):
            return np.array([[1 - x, x] for x in [0.9, 0.1, 0.1, 0.9]])
    detector = IncidentDetector.__new__(IncidentDetector)
    detector.model = Model()
    detector.threshold = 0.35
    detector.temporal_rule = "mean3"
    detector.feature_columns = FEATURE_COLUMNS
    results = detector.predict_windows(windows(), "camera")
    assert [r["incident_detected"] for r in results] == [True, True, True, True]
    assert results[0]["recent_scores"] == [0.9]
    assert results[2]["recent_scores"] == [0.9, 0.1, 0.1]


def test_empty_detections_and_bad_video():
    frame_rows = [{"source_time_sec": 0., "vehicle_count": 0,
                   "bbox_coverage": 0., "mean_detection_confidence": 0.}]
    result = aggregate_windows(frame_rows, [], 5)
    assert result.iloc[0].unique_vehicle_count == 0
    assert result.iloc[0].stopped_ratio == 0
    with pytest.raises(ValueError, match="MP4"):
        process_video("missing.mp4")


def test_handoff_order_and_real_agent_invocation():
    events = []
    class Detector:
        def predict_windows(self, frame, source):
            events.append("incident")
            return [{"source_id": source, "window_start_sec": 0., "window_end_sec": 5.,
                     "incident_detected": True}]
    class Agent:
        def handle_message(self, message):
            events.append("prediction")
            assert message["incident_context"]["incident_detected"] is True
            assert message["traffic_data"]["current_flow"] == 3.0
            return {"status": "warning", "forecast": {"predicted_occupancy": 0.2}}
    result = run_traffic_pipeline("x.mp4", incident_detector=Detector(),
        video_processor=lambda _: None, prediction_agent=Agent(),
        observation_provider=lambda source, incident: {
            "sensor_id": 1, "timestep": 20,
            "traffic_data": {"current_flow": 3.0, "current_speed": 4.0,
                             "current_occupancy": 0.1}},
        history_provider=lambda sensor, timestep: None)
    assert events == ["incident", "prediction"]
    assert result["windows"][0]["prediction"]["forecast"]["predicted_occupancy"] == 0.2


def test_missing_measurements_do_not_invoke_prediction():
    class Detector:
        def predict_windows(self, frame, source):
            return [{"window_start_sec": 0., "incident_detected": False}]
    class Agent:
        def handle_message(self, message):
            pytest.fail("Prediction agent should not run")
    result = run_traffic_pipeline("x.mp4", incident_detector=Detector(),
        video_processor=lambda _: None, prediction_agent=Agent(),
        observation_provider=lambda source, incident: {"sensor_id": 1},
        history_provider=lambda sensor, timestep: None)
    assert "Missing measured prediction fields" in result["windows"][0]["prediction_error"]

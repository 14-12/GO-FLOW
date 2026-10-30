import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.incident_detection.src import DetectorConfig, load_detector_config, process_frames, process_video
from agents.incident_detection.src.video_processing import aggregate_windows, result_rows

COCO_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REPO = Path(__file__).resolve().parents[1]


class Value:
    def __init__(self, *values):
        self.values = np.array([values], dtype=float)

    def __getitem__(self, index):
        return self

    def item(self):
        return float(self.values[0, 0])

    def cpu(self):
        return self

    def numpy(self):
        return self.values[0]


class Box:
    def __init__(self, track_id, cls, x):
        self.id, self.cls, self.conf = Value(track_id), Value(cls), Value(0.9)
        self.xyxy = Value(x, 10, x + 20, 30)


class FakeModel:
    """Reports which input frame it saw (encoded in pixel [0, 0, 0]) and one moving car."""
    names = COCO_NAMES

    def __init__(self):
        self.seen, self.options = [], None

    def track(self, frame, **options):
        self.seen.append(int(frame[0, 0, 0]))
        self.options = options
        result = type("Result", (), {"boxes": [Box(1, 2, x=len(self.seen))]})()
        return [result]


def frames(n):
    for i in range(n):
        yield np.full((40, 60, 3), i, dtype=np.uint8)


def test_frame_selection_matches_ultralytics_vid_stride():
    model = FakeModel()
    windows = process_frames(frames(91), 15, (60, 40), "cam", yolo_factory=lambda: model, device="cpu")
    # Ultralytics grabs `stride` frames and keeps the last one: indices 2, 5, 8, ...
    assert model.seen == list(range(2, 91, 3))
    assert model.options["classes"] == [2, 3, 5, 7] and model.options["conf"] == 0.25
    assert "imgsz" not in model.options
    assert windows.window_start_sec.tolist() == [0.0, 5.0]
    assert windows.window_end_sec.tolist() == [5.0, 91 / 15]
    assert windows.car_unique.tolist() == [1, 1]
    assert windows.attrs == {"source_id": "cam", "detector": "yolo11n-coco"}


def test_process_video_wraps_process_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 15, (60, 40))
    for frame in frames(91):
        writer.write(frame)
    writer.release()
    video_model, frame_model = FakeModel(), FakeModel()
    from_video = process_video(clip, yolo_factory=lambda: video_model, device="cpu")
    capture = cv2.VideoCapture(str(clip))
    decoded = [capture.read()[1] for _ in range(91)]
    capture.release()
    from_frames = process_frames(decoded, 15, (60, 40), "clip", duration_sec=91 / 15,
                                 yolo_factory=lambda: frame_model, device="cpu")
    assert video_model.seen == frame_model.seen
    pd.testing.assert_frame_equal(from_video, from_frames)


def test_config_resolves_class_names_and_rejects_bad_maps():
    config = load_detector_config()
    assert config.class_ids(COCO_NAMES) == {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    carla = DetectorConfig("carla", "w.pt", {"Car": "car", "van": "car", "ambulance": "truck"})
    assert carla.class_ids({0: "car", 1: "van", 2: "ambulance", 3: "pedestrian"}) == {0: "car", 1: "car", 2: "truck"}
    with pytest.raises(ValueError, match="no classes"):
        carla.class_ids(COCO_NAMES)
    with pytest.raises(ValueError, match="class_map"):
        DetectorConfig("bad", "w.pt", {"car": "pedestrian"})


def test_config_file_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.delenv("GOFLOW_YOLO_WEIGHTS", raising=False)
    (tmp_path / "w.pt").write_bytes(b"")
    path = tmp_path / "detector.json"
    path.write_text(json.dumps({"name": "carla", "weights": "w.pt", "class_map": {"car": "car"}, "imgsz": 960}))
    config = load_detector_config(path)
    assert config.weights == str(tmp_path / "w.pt") and config.imgsz == 960
    monkeypatch.setenv("GOFLOW_DETECTOR_CONFIG", str(path))
    monkeypatch.setenv("GOFLOW_YOLO_WEIGHTS", "other.pt")
    assert load_detector_config().weights == "other.pt"


def local_weights():
    for candidate in (os.environ.get("GOFLOW_YOLO_WEIGHTS"), REPO / "yolo11n.pt", REPO.parent / "yolo11n.pt"):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("no local yolo11n.pt (set GOFLOW_YOLO_WEIGHTS); not downloading in tests")


def test_real_yolo_windows_identical_to_v1_video_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    ultralytics = pytest.importorskip("ultralytics")
    weights = local_weights()
    from ultralytics.utils import ASSETS
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    clip, fps, size = tmp_path / "pan.mp4", 15, (640, 480)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(150):  # 10 s panning down the street, so tracks move
        writer.write(image[i * 2:i * 2 + 480, 20 + i:20 + i + 640])
    writer.release()

    # The v1 implementation: Ultralytics reads the file itself with vid_stride.
    capture = cv2.VideoCapture(str(clip))
    frame_count, real_fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    stride = max(1, round(real_fps / 5))
    categories = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    reference_rows, reference_detections = [], []
    results = ultralytics.YOLO(weights).track(
        source=str(clip), tracker="bytetrack.yaml", persist=True, stream=True, vid_stride=stride,
        conf=0.25, classes=[2, 3, 5, 7], device="cpu", verbose=False, save=False)
    for index, result in enumerate(results):
        row, detections = result_rows(result, index / (real_fps / stride), size[0] * size[1], categories)
        reference_rows.append(row)
        reference_detections.extend(detections)
    reference = aggregate_windows(reference_rows, reference_detections, frame_count / real_fps)
    assert reference.unique_vehicle_count.sum() > 0

    config = DetectorConfig("yolo11n-coco", weights, {name: name for name in categories.values()})
    from_video = process_video(clip, config=config, device="cpu")
    capture = cv2.VideoCapture(str(clip))
    decoded = [capture.read()[1] for _ in range(int(frame_count))]
    capture.release()
    from_frames = process_frames(decoded, real_fps, size, "pan", duration_sec=frame_count / real_fps,
                                 config=config, device="cpu")
    pd.testing.assert_frame_equal(from_video, reference)
    pd.testing.assert_frame_equal(from_frames, reference)

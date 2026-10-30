"""YOLO11/ByteTrack extraction with the training notebook's window statistics."""
from pathlib import Path

import numpy as np
import pandas as pd

from .detector_config import CATEGORIES, load_detector_config

WINDOW_SECONDS = 5.0
TARGET_FPS = 5.0
VEHICLE_CLASSES = (2, 3, 5, 7)
COCO_CATEGORIES = dict(zip(VEHICLE_CLASSES, CATEGORIES))


def aggregate_windows(frame_rows, detection_rows, duration):
    """Aggregate tracked detections; empty detections still yield zero-valued windows."""
    frames = pd.DataFrame(frame_rows)
    if frames.empty:
        raise ValueError("No video frames were processed")
    detections = pd.DataFrame(detection_rows)
    frames["window_start_sec"] = (np.floor(frames.source_time_sec / WINDOW_SECONDS) * WINDOW_SECONDS).round(3)
    if not detections.empty:
        if "category" not in detections:
            detections["category"] = detections.class_id.map(COCO_CATEGORIES)
        detections["window_start_sec"] = (np.floor(detections.source_time_sec / WINDOW_SECONDS) * WINDOW_SECONDS).round(3)
        detections = detections.sort_values(["track_id", "source_time_sec"]).reset_index(drop=True)
        for current, previous in (("center_x", "prev_x"), ("center_y", "prev_y"), ("source_time_sec", "prev_time")):
            detections[previous] = detections.groupby("track_id")[current].shift(1)
        distance = np.sqrt((detections.center_x - detections.prev_x) ** 2 + (detections.center_y - detections.prev_y) ** 2)
        dt = detections.source_time_sec - detections.prev_time
        detections["normalized_speed"] = distance.div(dt.where(dt > 0)).div(detections.bbox_height.clip(lower=1))
        detections.replace([np.inf, -np.inf], np.nan, inplace=True)
        tracks = detections.groupby(["window_start_sec", "track_id"], as_index=False).agg(
            observations=("track_id", "size"), median_speed=("normalized_speed", "median"))
        tracks["stopped"] = (tracks.observations >= 5) & (tracks.median_speed < 0.08)
    else:
        tracks = pd.DataFrame(columns=["window_start_sec", "stopped"])
    rows = []
    for value in np.arange(0, duration, WINDOW_SECONDS):
        start = round(float(value), 3)
        fg = frames[frames.window_start_sec == start]
        dg = detections[detections.window_start_sec == start] if not detections.empty else detections
        count = int(dg.track_id.nunique()) if not dg.empty else 0
        speeds = dg.normalized_speed.dropna() if not dg.empty else []
        stopped = int(tracks.loc[tracks.window_start_sec == start, "stopped"].sum())
        rows.append({
            "window_start_sec": start, "window_end_sec": float(min(start + WINDOW_SECONDS, duration)),
            "unique_vehicle_count": count,
            "mean_vehicle_count_per_frame": float(fg.vehicle_count.mean()) if not fg.empty else 0.0,
            "mean_normalized_speed": float(np.mean(speeds)) if len(speeds) else 0.0,
            "median_normalized_speed": float(np.median(speeds)) if len(speeds) else 0.0,
            "stopped_vehicle_count": stopped, "stopped_ratio": stopped / count if count else 0.0,
            "bbox_coverage_proxy": float(fg.bbox_coverage.mean()) if not fg.empty else 0.0,
            "max_bbox_coverage_proxy": float(fg.bbox_coverage.max()) if not fg.empty else 0.0,
            "mean_detection_confidence": float(fg.mean_detection_confidence.mean()) if not fg.empty else 0.0,
            **{f"{category}_unique": int(dg.loc[dg.category == category, "track_id"].nunique()) if not dg.empty else 0
               for category in CATEGORIES},
        })
    return pd.DataFrame(rows)


def result_rows(result, time_sec, frame_area, categories):
    """One frame row and its tracked detection rows from an Ultralytics tracking result."""
    confidences, area, detections = [], 0.0, []
    for box in result.boxes or []:
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        confidences.append(confidence)
        area += (x2 - x1) * (y2 - y1)
        detections.append({"source_time_sec": time_sec, "track_id": int(box.id[0].item()),
                           "class_id": class_id, "category": categories[class_id], "confidence": confidence,
                           "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
                           "bbox_height": y2 - y1, "bbox_area": (x2 - x1) * (y2 - y1)})
    frame = {"source_time_sec": time_sec, "vehicle_count": len(confidences),
             "bbox_coverage": min(area / frame_area, 1.0),
             "mean_detection_confidence": float(np.mean(confidences)) if confidences else 0.0}
    return frame, detections


def default_device():
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def build_detector(config, yolo_factory=None):
    """Fresh model per stream so tracker IDs never leak across videos or cameras."""
    if yolo_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install ultralytics to run YOLO and ByteTrack") from error
        yolo_factory = lambda: YOLO(config.weights)
    model = yolo_factory()
    return model, config.class_ids(model.names)


def process_frames(frames, fps, frame_size, source_id=None, *, duration_sec=None, config=None,
                   yolo_factory=None, device=None):
    """Window features from an iterable of BGR frames captured at `fps`.

    Keeps input frames stride-1, 2*stride-1, ... (0-based), the same frames Ultralytics
    selects with vid_stride, so an MP4 gives identical windows through either entry point.
    frame_size is (width, height). duration_sec defaults to frames received / fps.
    """
    fps = float(fps)
    width, height = (float(x) for x in frame_size)
    if not all(np.isfinite(x) and x > 0 for x in (fps, width, height)):
        raise ValueError("fps and frame_size must be positive")
    config = config or load_detector_config()
    device = default_device() if device is None else device
    model, categories = build_detector(config, yolo_factory)
    stride = max(1, round(fps / TARGET_FPS))
    processed_fps = fps / stride
    options = dict(tracker=config.tracker, persist=True, conf=config.confidence,
                   classes=sorted(categories), device=device, verbose=False, save=False)
    if config.imgsz:
        options["imgsz"] = config.imgsz
    frame_rows, detection_rows, received = [], [], 0
    for received, frame in enumerate(frames, start=1):
        if received % stride:
            continue
        result = model.track(frame, **options)[0]
        row, detections = result_rows(result, len(frame_rows) / processed_fps, width * height, categories)
        frame_rows.append(row)
        detection_rows.extend(detections)
    windows = aggregate_windows(frame_rows, detection_rows, duration_sec or received / fps)
    windows.attrs.update(source_id=source_id, detector=config.name)
    return windows


def process_video(video_path, *, yolo_factory=None, device=None, config=None, source_id=None):
    """Process one MP4 with a fresh tracker; inject a factory for lightweight tests."""
    path = Path(video_path)
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise ValueError(f"Readable MP4 path required: {path}")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install opencv-python-headless to process video") from error
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open MP4: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not all(np.isfinite(x) and x > 0 for x in (fps, frame_count, width, height)):
        cap.release()
        raise ValueError("Invalid MP4 FPS, frame count, or frame dimensions")

    def read_frames():
        try:
            for _ in range(int(frame_count)):
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame
        finally:
            cap.release()

    return process_frames(read_frames(), fps, (width, height), source_id or path.stem,
                          duration_sec=frame_count / fps, config=config,
                          yolo_factory=yolo_factory, device=device)

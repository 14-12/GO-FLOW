# Incident Detection Agent — integration handoff

This report describes **only the incident detection agent** and the interface by which another component can consume its results. It reflects the code in this repository as of 30 October 2026. Give this file and the referenced code to the integrating developer or Claude.

## 1. Purpose and boundary

The agent analyzes a highway MP4 and reports a possible traffic incident for each five-second video window. It uses YOLO11n vehicle detections, ByteTrack identities, a saved Random Forest classifier, and a causal three-window confirmation rule. It does **not** identify an accident type, establish physical speed, issue an emergency alert, or control traffic signals. `severity` is always `"UNKNOWN"` in its own output.

The reusable incident agent is `agents/incident_detection/src/`. `src/incident_pipeline.py` is an optional adapter that calls the separate Traffic Prediction agent afterward. Consumers who need only incident detection should call `process_video` and `IncidentDetector` directly.

## 2. Files to carry over

| File | Purpose |
| --- | --- |
| `agents/incident_detection/src/video_processing.py` | MP4 validation, YOLO11n/ByteTrack processing, five-second aggregation |
| `agents/incident_detection/src/feature_engineering.py` | Feature names/order and causal temporal features |
| `agents/incident_detection/src/inference.py` | Loads model artifact, scores windows, applies verdict rule |
| `agents/incident_detection/models/incident_detector.joblib` | Required trained Random Forest and saved configuration |
| `agents/incident_detection/requirements.txt` | Incident directory's dependency list |
| `tests/test_incident_pipeline.py` | Contract and behavior checks |

Keep the module layout intact or update imports accordingly. The artifact is loaded with `joblib`; use the committed trusted file, not an arbitrary uploaded model file. `yolo11n.pt` is a separate YOLO weight file; the workspace currently has one at the parent project root. Set `GOFLOW_YOLO_WEIGHTS` to an explicit local path for repeatable offline execution.

## 3. Direct input and execution

Input is a readable `.mp4` file path and a caller-chosen `source_id` string, normally the camera ID. A valid video must expose positive FPS, frame count, width, and height. There is no JSON request server or HTTP endpoint in the current implementation. The detector does not accept PEMS08 measurements and does not require an LLM or API key.

From the `GO-FLOW` repository root:

```python
from agents.incident_detection.src import IncidentDetector, process_video

windows = process_video("highway.mp4")
incidents = IncidentDetector().predict_windows(windows, source_id="camera_01")
```

`process_video` returns a pandas DataFrame with one row per five-second window. `predict_windows` returns a list of JSON-serializable dictionaries in time order. A zero-frame/unreadable video raises an exception. A valid video with no tracked vehicles still produces windows with zero-valued traffic features and model scores; zero detections do not mean a proven absence of incidents.

The bundled CLI includes the *optional prediction adapter*:

```powershell
python -m pip install -r agents/incident_detection/requirements.txt
python -m scripts.run_incident_pipeline C:\path\to\highway.mp4 --source-id camera_01
```

The CLI prints JSON containing `source_id` and `windows`. Each window contains `incident`, `handoff`, `prediction`, and sometimes `prediction_error`. For an incident-only integration, consume **`windows[*].incident`**; the sibling fields belong to the separate prediction adapter. Running the CLI with only a video normally produces a controlled prediction error because no measured traffic feed was supplied. That error does not invalidate the incident result.

## 4. Processing and model decision

1. YOLO11n processes the MP4 with a fresh ByteTrack instance for each video. It keeps COCO classes 2 car, 3 motorcycle, 5 bus, and 7 truck, with detection confidence at least 0.25. Frames are sampled at approximately 5 FPS using `stride = max(1, round(source_fps / 5))`; actual sampled FPS can differ.
2. Tracked detections are grouped into windows `[0,5)`, `[5,10)`, and so on, with the last end time clipped to the video duration. All times are **seconds relative to the video start**, not wall-clock timestamps.
3. The code calculates 13 current traffic features per window. Normalized speed is bounding-box center motion per second divided by box height. It is an image-motion proxy, **not km/h**. A track is marked stopped when it has at least five observations in the window and median normalized speed below 0.08.
4. Seven current features receive a delta from the previous window and a trailing mean over up to three windows. The first delta is zero. These 14 features use current and past data only.
5. The saved Random Forest checks for the exact 27-feature order and returns `predict_proba(... )[:,1]` as `incident_score`. The final Boolean is true when the mean of the current and up to two preceding scores is at least the artifact's threshold, approximately 0.35. The first two verdicts use fewer than three scores. This is a smoothing rule, not a requirement for three consecutive positive windows.

The score is a classifier score, **not a calibrated probability that an accident occurred**. The detector has no incident category classifier. Even a positive verdict has `severity: "UNKNOWN"`.

## 5. Incident output contract

Each direct result, or each CLI `windows[*].incident`, has these fields:

| Field | Type and meaning |
| --- | --- |
| `source_id` | string; caller-supplied camera/video identifier |
| `window_start_sec`, `window_end_sec` | numbers; relative video seconds |
| `incident_detected` | Boolean; smoothed threshold verdict |
| `incident_score` | number; raw current-window Random Forest positive-class score |
| `recent_scores` | array of 1–3 numbers; scores used for this verdict, oldest first |
| `temporal_rule` | string; currently `"mean3"` |
| `decision_threshold` | number; saved threshold, approximately `0.35` |
| `severity` | string; currently always `"UNKNOWN"` |
| `traffic_features` | object containing the 13 current-window features listed below |

`traffic_features` contains `unique_vehicle_count`, `mean_vehicle_count_per_frame`, `mean_normalized_speed`, `median_normalized_speed`, `stopped_vehicle_count`, `stopped_ratio`, `bbox_coverage_proxy`, `max_bbox_coverage_proxy`, `mean_detection_confidence`, `car_unique`, `motorcycle_unique`, `bus_unique`, and `truck_unique`. All values are serialized as numbers, including counts. Unique counts are based on tracked IDs; class counts are unique tracked IDs of each class. Bounding-box coverage is summed detection-box area divided by frame area, capped at 1 per frame. These are video-derived features, not physical road sensor measurements.

Example shape (illustrative values, not a benchmark or live event):

```json
{
  "source_id": "camera_01",
  "window_start_sec": 0.0,
  "window_end_sec": 5.0,
  "incident_detected": false,
  "incident_score": 0.07,
  "recent_scores": [0.07],
  "temporal_rule": "mean3",
  "decision_threshold": 0.35,
  "severity": "UNKNOWN",
  "traffic_features": {
    "unique_vehicle_count": 0.0,
    "mean_vehicle_count_per_frame": 0.0,
    "mean_normalized_speed": 0.0,
    "median_normalized_speed": 0.0,
    "stopped_vehicle_count": 0.0,
    "stopped_ratio": 0.0,
    "bbox_coverage_proxy": 0.0,
    "max_bbox_coverage_proxy": 0.0,
    "mean_detection_confidence": 0.0,
    "car_unique": 0.0,
    "motorcycle_unique": 0.0,
    "bus_unique": 0.0,
    "truck_unique": 0.0
  }
}
```

The video-relative time must be mapped to camera capture time by the integrating system if wall-clock time is needed. The agent does not output coordinates, road segment ID, frame images, bounding boxes, track IDs, or a recommended response.

## 6. Integration recommendation

The simplest adapter passes the whole incident object to the orchestrator, preserves its `source_id` and window times, and uses `incident_detected` as the Boolean signal. Preserve `incident_score`, `recent_scores`, `decision_threshold`, and `traffic_features` as evidence for review. Treat `severity` as unknown. Downstream code should not interpret `incident_score` as an accident probability or `mean_normalized_speed` as a road speed sensor.

If integrating with this repository's Traffic Prediction agent, `src.incident_pipeline.run_traffic_pipeline(...)` runs detection first, then builds a separate handoff message. With genuine independent measurements, `observation_provider(source_id, incident_result)` must return `sensor_id`, five-minute `timestep`, and `traffic_data` with measured `current_flow`, `current_speed`, and `current_occupancy`. `history_provider(sensor_id, timestep)` must return chronological PEMS08 measurements for the sensor and required graph neighbors, including `location`, `timestep`, `flow`, `speed`, and `occupy`, with at least 13 contiguous five-minute observations through the target timestep. The resulting handoff includes `request_id`, `sensor_id`, `timestep`, `traffic_data`, and `incident_context` with `incident_detected` and null type/severity. An optional `timestamp` can be supplied by the observation provider.

There is currently **no validated camera-to-PEMS08 sensor map** or live physical measurement feed in this repository. Do not derive flow, physical speed, occupancy, or sensor identity from the video features. With missing providers, the adapter sends an incomplete message to the prediction agent, which returns a controlled error and null forecast. Its `prediction_error` records that separate failure; the incident object remains usable.

## 7. Dependencies, validation, and limits

The incident requirements file lists Ultralytics, OpenCV headless, PyTorch, NumPy, pandas, scikit-learn 1.6.1, joblib, XGBoost, LangChain, langchain-openai, and `lap>=0.5.12`. The reusable deterministic inference path directly uses Ultralytics/ByteTrack, OpenCV, PyTorch for device selection, NumPy, pandas, joblib, and the scikit-learn artifact. The notebook's optional LangChain experiment is outside this path; no APInex key or external LLM is needed for incident inference. YOLO weights may be downloaded on first use unless `GOFLOW_YOLO_WEIGHTS` points to a local file.

The agent README reports accuracy **83.43%**, precision **24.30%**, recall **54.11%**, F1 **33.53%**, and F2 **43.44%** for the saved evaluation. Those values describe the academic evaluation, not deployment performance on a new camera. Precision is especially limited, so positive windows should be reviewed before operational action. The current code does not provide confidence calibration, scene-specific tuning, a monitoring service, or a production alert workflow.

Run the focused checks from the repository root with an environment containing pytest and the model dependencies:

```powershell
python -m pytest tests/test_incident_pipeline.py -q
```

The tests cover artifact feature order and threshold, causal temporal features, mean-three behavior, empty detections, invalid video, and the optional prediction handoff. They do not establish accuracy on the integrating team's cameras. For a deployment evaluation, collect labeled representative video, measure false positives and missed incidents by camera and conditions, and validate any proposed alert threshold on held-out scenes.

## 8. Handoff checklist for the other developer

1. Copy the six files/groups in section 2 and provide local YOLO weights; install dependencies in the target Python environment.
2. Call the direct two-line API in section 3 for incident-only use, or call the CLI and read `windows[*].incident`.
3. Provide stable `source_id` values and attach trustworthy camera capture timestamps outside this agent if needed.
4. Keep the exact output field meanings and unknown severity; handle processing exceptions separately from negative verdicts.
5. Test with labeled video from the intended camera before connecting any operational response.

For exact implementation details, read `agents/incident_detection/src/{video_processing,feature_engineering,inference}.py`. For the optional prediction bridge, read `src/incident_pipeline.py` and `docs/incident_prediction_integration.md`.

# Incident Detection Agent

Traffic anomaly detection using YOLO11, ByteTrack, Random Forest, temporal features and LangChain.

## Pipeline

Video -> YOLO11 -> ByteTrack -> 5-second features -> Random Forest -> temporal confirmation -> LangChain agent.

## Model

The trained detector is stored in `models/incident_detector.joblib`.

## Evaluation

- Accuracy: 83.43%
- Precision: 24.30%
- Recall: 54.11%
- F1: 33.53%
- F2: 43.44%

This is an academic prototype and is not yet a production incident alert system.

## Reusable inference

`src/video_processing.py` accepts an MP4, runs YOLO11n with a fresh ByteTrack
instance, and produces five-second traffic windows. `src/feature_engineering.py`
adds the 14 causal temporal features. `src/inference.py` loads the committed
Random Forest artifact, checks its 27-feature order, scores each window, and
applies the saved `mean3` threshold (currently 0.35). No ground-truth labels,
Colab globals, or LLM service are used in inference. Normalized speed is image
motion divided by bounding-box height, not km/h. Scores are classifier scores,
not calibrated accident probabilities. Severity remains `UNKNOWN`.

Install the requirements from this directory, then run from the repository root:

```powershell
python -m pip install -r agents/incident_detection/requirements.txt
python -m scripts.run_incident_pipeline C:\path\to\highway.mp4 --source-id camera_01
python -m pytest tests/test_incident_pipeline.py -q
```

Ultralytics may need to acquire `yolo11n.pt` on first use. For offline use,
set `GOFLOW_YOLO_WEIGHTS` to an existing local weight path. ByteTrack requires
`lap>=0.5.12`, included in `requirements.txt`. The optional LangChain
experiment remains in the notebook and reads `APINEX_API_KEY` from Colab
Secrets; the integrated pipeline never calls APInex.

## Frames input and detector config

`process_frames(frames, fps, frame_size, source_id)` takes any iterable of BGR
images (for example live CARLA camera frames) and returns the same five-second
windows; `process_video` is now a thin wrapper that reads the MP4 and calls it.
Frames are subsampled exactly like Ultralytics' `vid_stride`, so an MP4 gives
identical windows through either function (checked against the v1 video path
with real YOLO in `tests/test_incident_frames.py`). The whole stream is consumed
before windows are returned, so live use passes a finite chunk of frames.

Detector weights, class map, confidence, image size and tracker come from
`config/detector_v1.json` (the v1 settings: `yolo11n.pt`, COCO car, motorcycle,
bus, truck, confidence 0.25). Another detector is used by pointing
`GOFLOW_DETECTOR_CONFIG` at its file or passing `config=` directly; see
`config/detector_carla_yolo26n_t05.json`. Class names are resolved against
the model's own `names`, so a wrong map fails loudly. The window table carries
`attrs["detector"]` with the config name. The saved Random Forest was trained on
COCO YOLO11n features; other detectors change the feature distribution, so
their windows must be evaluated before any result is trusted.

`config/detector_carla_yolo26n_t05.json` is the Urban Traffic CARLA detector
(9 classes, no motorcycle) with the class map agreed for stage 1: van, police
and police_lights count as car; ambulance and ambulance_lights as truck;
pedestrians are left out. Its weights are on the shared PC; elsewhere point
`GOFLOW_YOLO_WEIGHTS` at a local copy.

## Evaluation by camera

`evaluation/evaluate_v1.py` re-scores the v1 recipe on all 100 AI City videos
with 5-fold cross-validation grouped by video and by camera
(`evaluation/camera_groups.json`), and reports per-event recall, time to detect,
false alarms per camera-hour and bootstrap 95 % intervals. It reads the feature
CSVs from `data/` (not committed) and writes a new JSON under
`evaluation/results/` on every run.

## Town05 stage 1

`evaluation/run_stage1.py <delivery folder> --detector <config> [--detector ...]`
runs v1 unchanged on each clip of a Urban Traffic CARLA delivery (`labels.json`
plus `clips/`) and writes one JSON per clip and detector, named
`<clip>__<detector>__v1.json`, plus a `stage1_summary_*.json`. Each window keeps
the v1 fields and gains `run`, `window_start_time_s` and `window_end_time_s`
(SUMO seconds of the clip's own run). An accident counts as in view from its
on-screen second (`crash_on_screen_clip_s`, clamped to 0) until its wrecks leave
the picture plus the render lag; false alarms are runs of alarm windows with no
in-view window, per hour of normal windows. The clips stay outside git.

## Handoff to Traffic Prediction

`src.incident_pipeline.run_traffic_pipeline` runs this detector first, then
calls `TrafficPredictionControlAgent.handle_message` in `live` mode when the
caller supplies genuine PEMS08 measurements. Each returned window contains
`incident`, `handoff`, `prediction`, and, when needed, `prediction_error`.
The incident object includes `source_id`, five-second start/end, confirmed
verdict, raw score, recent scores, rule, threshold, `severity: UNKNOWN`, and
all 13 current traffic features. The handoff message uses the prediction
agent's real contract: `request_id`, `sensor_id`, five-minute `timestep`,
`traffic_data` (`current_flow`, `current_speed`, `current_occupancy`), and
`incident_context` (`incident_detected`, null type, null severity).

An MP4 alone cannot produce a PEMS08 forecast. The prediction model requires
13 contiguous five-minute measured observations for the mapped sensor and
the required graph neighbors. A caller must provide both
`observation_provider(source_id, incident_result)` with an independently known
camera-to-sensor/timestep mapping and measured traffic values, and
`history_provider(sensor_id, timestep)` returning the causal history DataFrame.
No video motion proxy is converted into flow, physical speed, or occupancy.
Without these providers, each window includes the partial handoff actually
sent to `TrafficPredictionControlAgent.handle_message` and its controlled
missing-fields response. The incident result remains available. A standalone
genuine PEMS08 live inference, unrelated to any camera, can be checked with
`python -m scripts.run_pems08_live_smoke`.

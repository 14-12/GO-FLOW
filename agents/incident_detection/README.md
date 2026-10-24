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
provide that weight through its normal model cache. The optional LangChain
experiment remains in the notebook and reads `APINEX_API_KEY` from Colab
Secrets; the integrated pipeline never calls APInex.

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
Without these providers, incident results remain available and each window
reports the exact missing prediction inputs.

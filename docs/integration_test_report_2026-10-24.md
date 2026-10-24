# GO-FLOW integration validation — 2026-10-24

## Environment

| Check | Status | Evidence |
|---|---|---|
| Isolated environment | PASS | `.venv-integration`, Python 3.11.13 on Windows |
| Runtime packages | PASS | Ultralytics 8.4.188, OpenCV 5.0.0, PyTorch 2.14.1+cpu, scikit-learn 1.6.1, XGBoost 3.2.0, NumPy 2.1.3, pandas 3.0.0, joblib 1.6.0, `lap` 0.5.13, LangChain and project dependencies |
| GPU | BLOCKED | `torch.cuda.is_available()` returned `False`; CPU was used |
| Incident artifact | PASS | Committed joblib loaded, 27 ordered features, threshold `0.35000000000000003`, `mean3`; no retraining |
| YOLO weight | PASS | Official `yolo11n.pt` downloaded for inference, stored outside the repository at `C:\Users\abdel\Desktop\projetml\yolo11n.pt` |

The initial installation resolved scikit-learn 1.9.1 and emitted model-version
warnings. The artifact was serialized with 1.6.1, so the incident requirement
was pinned to that version and tests were rerun. Ultralytics also required
`lap>=0.5.12` for ByteTrack; this requirement is now explicit.

## Automated tests

| Suite | Passed | Failed | Skipped | Status |
|---|---:|---:|---:|---|
| Requested incident and prediction-contract tests before fixes | 35 | 0 | 0 | PASS |
| Full repository suite before environment fixes | 143 | 1 | 0 | FAIL; 42 setup errors |
| Full repository suite after fixes | 187 | 0 | 0 | PASS |

The original full-suite setup errors came from `data/raw/traffic_dataset.csv`
being checked out with CRLF bytes under `core.autocrlf=true`. The teammate's
runtime checks its original LF SHA-256 (`7bcc4a2161229a4b5187bd727c1e63f989aa193301d6773137bd4a59e1680293`).
The added `.gitattributes` preserves those bytes on Windows. The remaining
legacy test needed IPython in the isolated environment; with it installed,
the full suite passed. The original test run with the right CSV bytes but
without IPython had 185 passed and 1 failed.

## Synthetic MP4 smoke test

**PASS.** A 320×192 black MP4 was generated with OpenCV at 5 FPS for 65
frames (13 seconds), outside the repository. OpenCV reopened it and reported
the expected FPS and frame count. `python -m scripts.run_incident_pipeline`
processed it through YOLO11n, ByteTrack, window aggregation, 27-feature
engineering, the saved Random Forest, and `mean3`. No tracked vehicles were
expected or found. This clip is not evidence of real traffic detection.

| Window (s) | Unique tracked vehicles | Raw score | Recent scores used | Confirmed incident |
|---|---:|---:|---|---|
| 0–5 | 0 | 0.06961404786511331 | `[0.06961404786511331]` | false |
| 5–10 | 0 | 0.06961404786511331 | two past/current scores | false |
| 10–13 | 0 | 0.06961404786511331 | three past/current scores | false |

Each result contained `source_id: camera_01`, its start/end time, all 13
current traffic features, a saved threshold, a classifier score, recent
scores, `temporal_rule: mean3`, and `severity: UNKNOWN`.

The first window's actual partial handoff to Agent 2 was:

```json
{"request_id":"camera_01-0","incident_context":{"incident_detected":false,"incident_type":null,"severity":null}}
```

The real `TrafficPredictionControlAgent.handle_message()` response was:

```json
{
  "agent": "traffic_prediction_control",
  "request_id": "camera_01-0",
  "sensor_id": null,
  "forecast": {"horizon_minutes": 15, "predicted_occupancy": null},
  "traffic_state": null,
  "risk_level": null,
  "recommended_action": "manual_review",
  "reason": "ValueError: request_id, sensor_id, timestep et traffic_data sont obligatoires.",
  "confidence": 0.0,
  "decision_source": "rule_based_fallback",
  "status": "error",
  "metadata": {"integration_test_fixture": false, "error_controlled": true}
}
```

This is a genuine Agent 2 validation response, not a forecast. The same
controlled error was captured for all three windows. A new regression test
asserts that the no-provider path actually calls Agent 2. The CLI was fixed
to keep Ultralytics progress output off stdout so its output remains JSON.

## Genuine PEMS08 prediction, separate from video

**PASS.** `python -m scripts.run_pems08_live_smoke` read real committed PEMS08
rows for sensor 33, timesteps 15215–15227, plus corresponding graph-neighbor
rows. It called the existing agent in `prediction_mode="live"`; no model was
trained or replaced.

Input message:

```json
{"request_id":"pems08-live-33-15227","sensor_id":33,"timestep":15227,"traffic_data":{"current_flow":0.0,"current_speed":65.0,"current_occupancy":0.0}}
```

Agent 2 returned `status: warning`, `metadata.prediction_mode: live`, and a
genuine XGBoost forecast of **0.0027397831436246634 occupancy at +15
minutes**. It reported `traffic_state: free_flow`, `risk_level: low`, and
`decision_source: rule_based_fallback`; its optional LLM was not called. This
sensor result is not linked to the synthetic camera clip.

## Real highway MP4 and full combined forecast

**BLOCKED.** No real highway MP4 was present in the workspace. Vehicle
detections, ByteTrack IDs on visible vehicles, camera-domain behavior, and
incident classification on real footage remain unverified. The exact command
for the supplied clip is:

```powershell
$env:GOFLOW_YOLO_WEIGHTS='C:\Users\abdel\Desktop\projetml\yolo11n.pt'
python -m scripts.run_incident_pipeline <VIDEO_PATH> --source-id camera_01
```

**BLOCKED.** The repository has no validated mapping from camera `camera_01`
and video time to a PEMS08 sensor and five-minute timestep. For a genuine
two-agent forecast, `observation_provider(source_id, incident_result)` must
provide `sensor_id`, `timestep`, and measured `traffic_data.current_flow`,
`current_speed`, and `current_occupancy`. A causal
`history_provider(sensor_id, timestep)` must return at least 13 contiguous
five-minute rows from t−12 through t, with `location`, `timestep`, `flow`,
`speed`, and `occupy`, for that sensor and its required graph neighbors.
Missing or inconsistent inputs produce controlled validation errors. Video
motion proxies cannot substitute for those measurements.

## Demonstration readiness

**PASS** for an MP4-to-incident-output demonstration and the explicit Agent 2
missing-data handoff. **PASS** for a separate genuine PEMS08 forecast.
**BLOCKED** for one video producing a scientifically valid combined incident
result and forecast until a real video and validated camera/sensor measurement
provider are supplied.

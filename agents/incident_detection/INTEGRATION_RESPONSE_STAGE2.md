# GO-FLOW Incident Agent: Response to the Town05 Integration Report

Oct 30, 2026 · reply to "GO-FLOW Incident Agent × Urban Traffic CARLA: Integration Report"

## 1. Summary

The four first tasks of the report's section 11 are done:
1. **Code compared with the notebooks.** The code in `src/` reproduces the notebook exactly.
2. **Honest baseline measured.** v1 has been re-scored with cross-validation grouped by camera, and per-event metrics are written down.
3. **`process_frames` added.** `process_video` is now a wrapper around it.
4. **Detector settings moved to a config file.**

v1 is unchanged: same 27 features, same model file, same output fields. All 192 repository tests pass.

The main finding is that v1 catches most incidents eventually, but raises about **8 false alarms per camera-hour**. That is far above the proposed target of under 1, so v1 cannot act in Town05 as it is.

**Where the code is:**

| | |
|---|---|
| Repository | `14-12/GO-FLOW` |
| Branch | `feature/incident-carla-integration`, built from `integration/incident-to-traffic-prediction` |
| Commit | `b19dc1f` |
| Pushed? | Not yet. It will be pushed once agreed. |

## 2. Task 1: the code against the notebooks

**Same results.** `IncidentDetector` from `src/` gives exactly the notebook's held-out test result on the original split:
- confusion matrix `[[2124, 349], [95, 112]]`
- precision 0.243, recall 0.541, F2 0.434, ROC-AUC 0.757

The temporal features rebuilt by `src/feature_engineering.py` match the notebook's feature CSV, with a largest difference of 7e-15.

**Differences that matter for the integration:**

| Point | Notebook | `src/` |
|---|---|---|
| Tracker | One YOLO model for the whole batch with `persist=True`, so track ids carry across videos | A fresh model per video. This only changes the first frames of each video. |
| Detector | `yolo11n.pt` hard-coded | Now set from `config/detector_v1.json` (task 4) |
| Input | MP4 only | MP4 or frames (task 3) |
| Temporal rule | 4 candidate rules | `mean3` only; any other rule is rejected |
| Time | Seconds from the start of the video | Same. Absolute time (change 3 of section 6) is not done yet. |
| LLM | LangChain + APInex | None in the detection path, which matches the report's rules |

**The real v1 output fields.** Section 9 of the report says "every v1 field". These are the fields `IncidentDetector.predict_windows` returns:
- `source_id`
- `window_start_sec`, `window_end_sec`
- `incident_detected` (bool)
- `incident_score`
- `recent_scores`
- `temporal_rule`
- `decision_threshold`
- `severity` (always `"UNKNOWN"`)
- `traffic_features` (the 13 current features)

**Not mentioned in the report: a second agent downstream.** `src/incident_pipeline.run_traffic_pipeline` passes every window to GO-FLOW's Traffic Prediction agent, which works on PEMS08 data. Any change to the output contract has to keep that handoff working, and its tests cover it.

## 3. Task 2: the honest v1 baseline

### The cameras repeat across splits
AI City gives no camera ids, so I grouped the videos by camera myself:
- extracted frames at 5 s and 300 s from each video;
- compared the edge maps of the frames;
- linked any two videos scoring at least 0.6;
- checked the groups by eye on contact sheets.

The 100 videos come from about **64 cameras**, and 19 cameras have more than one video. **13 of those groups mix train with val or test.** For example:
- test 82 has the same view as train 12, 35, 38 and 40;
- test 85 matches train 43;
- test 30 matches train 23, 57 and 84;
- test 1 matches train 79 and 96.

So the report's point 2 in section 4 is confirmed. The groups are in `evaluation/camera_groups.json`. Video 2 had no extracted frame and is kept as its own group.

### Results
The recipe is fixed: v1's Random Forest hyperparameters, `mean3`, threshold 0.35. It is scored with 5-fold cross-validation over all 100 labelled videos (the original train, val and test pooled), repeated with 3 fold seeds. The 95 % intervals come from a bootstrap over events (for recall) and over videos (for F2 and false alarms).

| Metric | Grouped by video | **Grouped by camera** |
|---|---|---|
| Window F2 | 0.43 – 0.53 (mean 0.48) | **0.37 – 0.45 (mean 0.41)**; 95 % CI about 0.23 – 0.63 |
| Average precision | 0.33 – 0.43 | **0.24 – 0.34** |
| Events caught (out of 18) | 14 – 15 | **15 in every seed (83 %)** |
| Median time to detect | 20 – 29 s | **25 – 35 s** |
| False alarms per camera-hour | 6.4 – 8.2 | **7.4 – 8.6**; 95 % CI about 4.6 – 11.9 |

**What the numbers say:**
- **Recall is helped by long events.** Most anomalies last 5 to 15 minutes, so one alarm at any point counts as a catch.
- **Some catches are very late.** Video 11 is caught after 400 to 650 s, and video 49 after 168 s.
- **Some events are always or nearly always missed.** Videos 9 and 63 are missed in every seed, and video 2 in 2 of the 3 seeds. This fits point 4 of the report: the stalled vehicle is not detected.
- **False alarms are the main problem.** There are about 8 per camera-hour.
- **The numbers lean optimistic.** The 0.35 threshold was chosen in the notebook on train+val scores, which overlap these folds.
- **The intervals are wide.** There are only 18 events. Town05 data is what can narrow them.

**Files:**
- full results with every event: `evaluation/results/v1_baseline_20261030_172949.json`
- command to rerun: `python agents/incident_detection/evaluation/evaluate_v1.py --seeds 3`

## 4. Tasks 3 and 4: frames input and detector config

### `process_frames`
```python
from agents.incident_detection.src import process_frames, IncidentDetector
windows = process_frames(frames, fps=10, frame_size=(960, 540), source_id="1427-1 looking N")
results = IncidentDetector().predict_windows(windows, "1427-1 looking N")
```
- **Input:** `frames` is any iterable of BGR images at `fps`.
- **Frame selection:** frames are subsampled to 5 fps with the same frame choice as Ultralytics' `vid_stride`. At 10 fps it keeps frames 1, 3, 5, … (0-based).
- **Proof of identical output:** `tests/test_incident_frames.py` runs real YOLO11n on a generated clip. `process_video`, `process_frames` and the old v1 video path give identical windows.
- **Limit for live use:** the stream is consumed before windows are returned. In the viewer, send finite chunks, for example 15 s per camera. True streaming output is not done yet.

### Detector config
- **Default:** `config/detector_v1.json` holds the v1 settings (`yolo11n.pt`, COCO car, motorcycle, bus and truck, confidence 0.25).
- **Choosing another file:** set `GOFLOW_DETECTOR_CONFIG` to its path, or pass `config=` directly.
- **Fields:** weights, `class_map` (detector class name → v1 category), confidence, `imgsz`, tracker.
- **Name checking:** class names are checked against the model's own `names`, so a wrong map fails with a clear error.
- **Recording the detector:** each window table carries `attrs["detector"]` with the config name.
- **CARLA example:** `config/detector_carla_yolo26n.example.json` is a draft for your detector. Its class map is an assumption (see section 5).
- **Warning:** the saved Random Forest was trained on COCO YOLO11n features. Windows from another detector have a different feature distribution, so they must be evaluated before any result is trusted.

## 5. What we need from you

1. **Your detector's class names.** Send the output of `print(YOLO("carla-yolo26n.pt").names)` and the weights file, or its path on the shared PC.
2. **Agreement on the class mapping.** v1 counts only car, motorcycle, bus and truck. The draft maps van and police to car and ambulance to truck, and leaves pedestrians out. Is that acceptable?
3. **The stage-1 clips** (section 5, step 1 of your report):
   - 4 ring-road cameras and 4 junction cameras, 5 minutes each;
   - some with a crash in view, some with red-light queues;
   - MP4 at 960x540 and 10 fps;
   - for each clip, the camera name from `layout.json`, the sim second of frame 0, and the `incidents.jsonl` rows for that run.
4. **The time base.** Should `window_start_time_s` be SUMO sim seconds? We will add it next to the video-relative seconds.
5. **How data moves.** A shared folder path for clips and labels; videos stay out of git.
6. **Decisions from section 12**, especially:
   - May v2 be trained with Town05 data?
   - Should a detection ever act in the town, or only be shown and logged?

## 6. Proposed next steps

| Step | Who | Depends on |
|---|---|---|
| Push `feature/incident-carla-integration` | GO-FLOW | Agreement |
| Agree the class map and the time base (section 5, items 2 and 4) | Both | — |
| Stage 1: run v1 unchanged on the 8 clips, with both detectors, and report false alarms at red lights and crash catches | GO-FLOW | The clips and detector |
| Add absolute time and an optional stopped-track evidence output (changes 3 and 4 of section 6), with no change to the existing v1 fields | GO-FLOW | Interface agreement |
| Stage 3 dataset, then v2 | Both | Section 7 of your report |

## 7. Message for your Claude session

```markdown
GO-FLOW update on the Town05 integration (attached: INTEGRATION_RESPONSE_STAGE2.md).
Tasks 1-4 of section 11 of your report are done, on branch feature/incident-carla-integration
(commit b19dc1f) of 14-12/GO-FLOW:
- src/ reproduces the notebook exactly; v1 is unchanged (27 features, mean3, threshold 0.35).
- Camera-grouped CV baseline for v1: window F2 about 0.41, 15/18 events caught, median time
  to detect 25-35 s, about 8 false alarms per camera-hour. 13 AI City camera groups crossed the
  original splits.
- process_frames(frames, fps, frame_size, source_id) added; process_video wraps it, with
  identical windows tested against real YOLO.
- Detector config in agents/incident_detection/config/ (detector_v1.json; a draft
  detector_carla_yolo26n.example.json).
The real v1 output fields are listed in section 2 of the attached file; use those names in the
interface of section 9.
Please read the attached file, then help prepare what section 5 of it asks for: the detector's
class names, the class mapping, the 8 stage-1 clips with camera names, start sim seconds and
incident rows, the time base, and a shared folder for data. Do not edit GO-FLOW files.
```

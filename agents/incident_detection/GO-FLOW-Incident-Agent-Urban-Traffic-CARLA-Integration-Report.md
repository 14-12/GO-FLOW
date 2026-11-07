# GO-FLOW Incident Agent × Urban Traffic CARLA: Integration Report

Oct 30, 2026 · @Hela

## 1. How to use this report

The GO-FLOW incident detection agent can become a working part of our Town05 traffic system: our town can generate the labelled video it lacks, and act on what it detects. This report is written for the agent's developer and for his Claude session. It explains our project, reviews the current model, and sets out the integration plan, the code changes, the data, the evaluation and the interface between the two systems.

- **Read first:** sections 2 (our project), 3 (what our town gives) and 5 (the plan).
- **For his Claude session:** paste section 11 as the first message, with this whole report attached. It states the goals, the rules and the first tasks.
- **Sources reviewed:** the handoff report `incident_agent_handoff_report.md` and the two Colab notebooks (`incident_agent_training.ipynb`, `incident_agent_training_clean.ipynb`), plus our own code. The GO-FLOW source files themselves were not read yet; where this report describes them, it relies on the handoff report.
- **Nothing is changed yet** on either side. Every change below is a proposal to agree on first.

## 2. Our project in brief

Urban Traffic CARLA is a city traffic co-simulation: SUMO drives every vehicle and pedestrian, and CARLA renders the same town in 3D, on Town05 (CARLA 0.9.16, SUMO 1.27.1, Python 3.11, Windows PC with a 4 GB GPU). SUMO is the master; CARLA mirrors its state about 45 times a second. Everything below runs today.

| Part | Where in the code | What it does |
| --- | --- | --- |
| Town05 network and signals | `scenarios/town05/`, `tools/build_town05_signals.py`, `cosim/signal_plan.py` | 15 signalised junctions, protected greens (no turning across oncoming traffic), 51 signalised crosswalks, a ring road with 3 lanes each way |
| Coordinated signals | `cosim/coordination.py`, `cosim/sumo_side.py` (`CycleKeeper`) | One 115 s cycle for the whole town, offsets for green waves, greens that end early when nobody comes |
| Junction AI agents | `cosim/agents_actuated.py` | 15 agents, one per junction, adapting the greens to queues and emergency vehicles |
| Traffic cameras | `cosim/perception.py`, `cosim/inspector.py`, `out/town05_cameras/` | 55 cameras, one per road at each signalised junction: 9 m up, pitch -22 deg, 960x540, 75 deg field of view |
| Camera detector | `models/carla-yolo26n.pt`, `tools/train_detector.py` | YOLO26n fine-tuned on CARLA images: car, van, bus, truck, ambulance, police (with and without lights), pedestrian |
| Camera reports | `cosim/camera_reports.py` | Per camera every 2 s: vehicles and queues per lane, parked cars off the road, vehicles still over 90 s (`stopped_long`), sirens seen |
| Accidents | `cosim/incidents.py`, `cosim/incident_scene.py`, `cosim/towing.py` | Careless drivers (10 % of cars) look away and crash for real; police come, close the lane, a recovery truck tows the wrecks |
| Viewer | `cosim/freefly.py`, `town05_freefly.bat` | Fly through the town; accident markers, camera inspection (H key), clock and lighting by time of day |
| LLM brief | `cosim/llm.py` | One language-model call that writes a plain-language summary of the town's state |

The accidents matter most for this integration. Nothing is staged: a distracted driver rear-ends a queue or runs a red light, and SUMO detects the collision. Each incident records its start second, type (rear-end or side impact), severity (impact speed, labelled minor / moderate / severe), lane, position and every later event (police on scene, lane closed, towing, reopening).

## 3. What our town can give the incident agent

Our town supplies the three things the agent misses most: labelled incidents in volume, the look-alike situations that cause false alarms, and context a real city also has.

| What | Where it comes from | Why it matters to the agent |
| --- | --- | --- |
| Camera video from 55 fixed positions | CARLA camera sensors at the poses in `out/town05_cameras/layout.json` | Input in the agent's own format (MP4 or frames), from junction and ring-road views |
| Exact incident labels | `IncidentManager.snapshot()` in `cosim/incidents.py`: id, `t0`, kind, severity, lane, x/y, wrecks, phase, events | Ground truth per second, not coarse intervals; also type and severity, which the agent now reports as UNKNOWN |
| Whether a wreck is in a camera's picture | The camera's pinhole model (`cosim/perception.project`) applied to each wreck's position | Labels "incident visible in this camera", the label the agent actually needs |
| Look-alike negatives | Red-light queues, rush-hour jams on the ring road, buses at stops, delivery vans at depots, parking lots by the junctions, emergency vehicles taking over signals | The situations that make a stopped-vehicle model fire in a city; rare in highway training data |
| Signal state every second | SUMO's state string per junction; which lane each link serves | "Vehicles stopped while their light is green" is the strongest sign of an incident; real cities can get the same from their controllers |
| Lane masks per camera | `out/town05_cameras/masks.npz` and `layout.json` | Which pixels are which lane: stopped cars are counted per lane, parked cars off the road are ignored |
| Real speed | Camera height, pitch and field of view are known | Speed in m/s on the road plane instead of box-height speed |
| A detector trained on CARLA | `models/carla-yolo26n.pt` | Sees CARLA vehicles, including small far-away ones, better than COCO YOLO11n; also recognises police and ambulances |
| A response chain | Police dispatch, lane closure, signal changes, accident markers, the LLM brief | A confirmed detection can trigger something real in the town |

One limit: CARLA images are not real video. Town05 data should be added to the real AI City data, never replace it, and the final test of a real-world claim must stay on real video.

## 4. Review of the current incident model

The method is sound, but the final numbers rest on 3 test incidents, and the detector often never sees the stalled vehicle. Data: AI City Challenge 2021 Track 4, 100 labelled highway videos, 18 anomaly intervals in 17 videos, 17,704 windows of 5 s (1,933 incident windows).

**What is done well**

- Splits by whole video, model chosen with 5-fold cross-validation grouped by video, test used once at the end.
- Time features use only the past (change from the previous window, mean of the last 3), so live behaviour matches the evaluation.
- The first notebook picked a threshold on 2 validation videos (F2 0.96), saw the test fall to 0.42, and moved to grouped cross-validation. The right reaction.
- No secrets in the notebooks: the API key and GitHub token are read from Colab secrets.

**Results, same Random Forest, two evaluations**

| Evaluation | Incident videos | Precision | Recall | F2 |
| --- | --- | --- | --- | --- |
| Grouped cross-validation, mean3 rule | 14 | 0.617 | 0.655 | 0.647 |
| Held-out test, mean3 rule | 3 | 0.243 | 0.541 | 0.434 |

**Problems that limit it**

1. **The test is 3 incidents.** 207 test windows come from 3 events, so the test numbers are mostly chance; the cross-validation numbers are the better estimate, and neither has a confidence interval.
2. **Same cameras in training and test, probably.** Track 4 videos come from a limited set of highway cameras. Grouping by video does not keep a camera out of both sides; the model can learn the scene rather than the event. Group by camera location instead.
3. **A few events dominate.** Video 11 is incident in all 178 windows, about 13 % of the training positives. Weight by event, not by window.
4. **The stalled vehicle is often not detected.** In the notebook's own missed case (video 2, from 585 s) the windows show about 0.2 vehicles per frame: YOLO11n at confidence 0.25 does not see the small, distant vehicle, so no feature can catch it. Strong Track 4 entries used background modelling over time to find vehicles that stay still.
5. **Counts depend on the camera.** 20 vehicles is heavy for one view and light for another; features should be relative to each camera's normal level.
6. **The score is not a probability.** Random Forest scores are uncalibrated, so the 0.35 threshold has no stable meaning across cameras.
7. **Per-window metrics only.** Operators need incidents caught, time to detect, and false alarms per camera-hour.

## 5. Integration plan

Five stages, each with a gate that must pass before the next starts. The agent stays his code in his repository; our side adds an adapter and the data tools.

&#91;embedded content: integration architecture · live path and learning path\]

Live, his agent turns camera frames into alarms and our adapter confirms them before anything acts; offline, our crash log labels the video that trains version 2.

1. **Run it unchanged on Town05 clips.** Record 5-minute MP4s from 4 ring-road and 4 junction cameras, with and without a crash in view. Run `process_video` and `IncidentDetector` as they are.
   - Gate: it runs end to end on our clips; we know its false-alarm rate at red lights and whether it fires on a crash at all.
2. **Fix the evaluation on his real data.** Regroup the AI City splits by camera location, report per-event recall, time to detect and false alarms per camera-hour, with bootstrap confidence intervals.
   - Gate: honest real-video baseline numbers, written down before anything changes.
3. **Build the Town05 dataset.** Long recordings from many cameras, times of day and traffic levels, labelled per window from our incident log (section 7).
   - Gate: at least 200 crashes seen by a camera, and the look-alike negatives in every split.
4. **Train version 2.** New features and model (section 6), trained on AI City plus Town05, tested on each separately.
   - Gate: v2 beats v1 on the real AI City test, not only on Town05. If it only wins on Town05, it is a Town05 model and is labelled as such.
5. **Run live and act.** The agent watches 2 to 4 cameras in the viewer. A detection confirmed by our own check (a vehicle still in a moving lane while its light is green) sends police, closes the lane and tells the signals.
   - Gate: on held-out Town05 runs, most crashes are caught within 30 s, and false alarms stay under 1 per camera-hour.

The 30 s and 1 per camera-hour targets are proposals to agree on, not measured values.

## 6. Code changes needed in GO-FLOW

Ten changes make the agent usable live and reliable on our town; v1 and its 27-feature contract stay working throughout.

1. **Frames as input, not only files.** Add `process_frames(frames, fps, frame_size, source_id)` that takes an iterator of images with timestamps; `process_video` becomes a thin wrapper around it. Live use in the viewer needs this.
2. **A configurable detector.** The COCO class ids 2, 3, 5, 7 and `yolo11n.pt` are fixed today. Take the weights path, a class map, confidence and image size from a config, so our CARLA detector (car, van, bus, truck, police, ambulance) can be used. Report which detector produced each result.
3. **Absolute time.** Accept a start time (sim seconds or wall clock) and output `window_start` and `window_end` in it as well as video-relative seconds.
4. **Optional evidence output.** Behind a flag, return per window the tracks judged stopped: track id, box, class, seconds stopped. Our side needs it to place an alarm on a lane and draw it.
5. **Version-2 features** (next to the 27, never replacing them):
   - how long each track has been still, carried across windows (not reset every 5 s);
   - share of still tracks among moving ones in the same lane or image region;
   - lane changes around one spot (vehicles swerving past an obstacle);
   - each feature relative to the camera's own normal level (a running baseline per `source_id`);
   - speed in m/s on the road plane when a camera calibration is given (height, pitch, field of view, or a homography).
6. **Optional context inputs.** Lane masks and signal state per lane, when available, to give "still while the light is green" and to drop parked cars. The model must still run on video alone (context missing = its own value, not zero).
7. **A stationary-object detector.** A slow background model per camera (for example a median over the last 2 to 5 minutes) that finds vehicles which have become part of the scene, to catch stalled cars YOLO misses.
8. **Model and decision rule.** Gradient boosting with class weights, calibrated with isotonic regression inside grouped cross-validation; a hysteresis rule (alarm on above a high score for N windows, off below a low score) instead of mean3. Save as `incident_detector_v2.joblib` with feature list, thresholds, library versions and a hash of the training data.
9. **Evaluation code.** Splits grouped by camera location, per-event metrics, time to detect, false alarms per camera-hour, bootstrap confidence intervals (section 8).
10. **Lean dependencies.** A separate inference requirements file without LangChain, langchain-openai or XGBoost; pin scikit-learn to the version that saved the model; load only the committed model file (joblib runs code when it loads).

## 7. Town05 dataset: how it is generated and labelled

Our side records the video and writes the labels; his side trains and evaluates on them. One recording run = one SUMO seed, one traffic level, one start time of day, a set of cameras, 15 to 20 minutes.

**Recording**

- Video: one MP4 per camera, 960x540 (the cameras' own size), 10 frames a second, H.264. The agent samples 5 frames a second, so 10 is enough and halves the disk use.
- Variation across runs: traffic x0.5, x0.8 and x1.0; day, dusk and night; all 55 cameras over the whole dataset, a few per run (the 4 GB GPU renders only a few cameras at once).
- More crashes per hour: data runs raise the share of careless drivers above the normal 10 %. The crashes stay real collisions, only more frequent. The share to use is set after a first run measures the crash rate.

**Label files** (one folder per run)

| File | One row per | Main fields |
| --- | --- | --- |
| `run.json` | run | seed, traffic level, clock start, careless share, cameras, software versions |
| `incidents.jsonl` | incident | id, start second, kind, severity (m/s and label), lane, x/y, wrecks, every event with its second |
| `windows_<camera>.csv` | 5 s window of one camera | sim and video seconds, `incident_visible`, incident id, seconds since the crash, phase (crash, police, recovery), wreck distance (m) and box in the picture |
| `context_<camera>.csv` | 5 s window of one camera | share of green per lane, queued vehicles per lane, buses at stops, emergency vehicles on a run |
| `negatives.csv` | look-alike episode | type (red-light queue, jam, bus stop, depot, emergency run), camera, start and end |

**Rules for the labels**

- `incident_visible` = 1 when a wreck is inside the picture by the camera's pinhole model, within 90 m, from the crash until it is towed away. Occlusion by buildings is not checked at first; a later version can test it with rays.
- A window counts as "onset" for its first 60 s after the crash; time to detect is measured from the crash second.
- Splits are fixed before training: whole cameras and whole seeds are held out, never single windows.

## 8. Evaluation protocol

Every model, v1 included, is judged the same way, on real and synthetic video separately, per event and per hour of watching.

| Metric | Definition | Why |
| --- | --- | --- |
| Event recall | Share of incidents with at least one alarm while the wreck is visible | Did we catch the crash at all |
| Time to detect | Seconds from the crash (or anomaly start) to the first alarm; median and 90th percentile | A late alarm is worth little |
| False alarms per camera-hour | Alarm episodes with no incident in view, divided by hours watched | What an operator feels |
| False alarms by cause | The same, split by look-alike type (red-light queue, jam, bus stop, depot, emergency run) | Shows what to fix next |
| Window precision, recall, F2, average precision | As in the notebooks | Kept for comparison with v1 |

- **Splits:** AI City grouped by camera location; Town05 by camera and by seed. Real and synthetic results are reported apart, never pooled.
- **Uncertainty:** 95 % confidence intervals by bootstrap over events (not windows), since events are the independent units.
- **Breakdowns:** ring-road vs junction cameras, day vs night, traffic level, crash severity.
- **Fixed before looking:** thresholds are chosen on development data only; each test set is used once per model version, and every result is kept, good or bad.

## 9. Interface between the incident agent and our system

The agent keeps its v1 output exactly and adds optional fields; our adapter supplies the camera, the time and, when available, the context. Field names below are a proposal to agree on.

**Camera identity.** `source_id` = our camera name from `out/town05_cameras/layout.json`, for example `"1427-1 looking N"`. Each camera there also gives its junction (`tls`), the road it looks along (`road_to`), position, yaw (deg), pitch -22 deg, 960x540 and 75 deg field of view.

**What our adapter sends** (per call, or per stream in live mode):

```json
{
  "source_id": "1427-1 looking N",
  "start_time_s": 3725.0,
  "fps": 10,
  "frames": "<iterator of 960x540 BGR images, or an MP4 path>",
  "camera": {"height_m": 9.0, "pitch_deg": -22.0, "fov_deg": 75.0, "width": 960, "height": 540},
  "context": {
    "lane_mask": "<960x540 array of lane ids, optional>",
    "signal": [{"t": 3725.0, "lanes": {"-6_0": "G", "-6_1": "r"}}]
  }
}
```

**What the agent returns** per 5 s window: every v1 field, unchanged in meaning, plus:

```json
{
  "model_version": "v2",
  "detector": "carla-yolo26n",
  "window_start_time_s": 3725.0,
  "window_end_time_s": 3730.0,
  "alarm_state": "on",
  "evidence": [
    {"track_id": 17, "cls": "car", "box": [412, 230, 470, 268], "still_s": 41.5, "lane": "-6_1", "light": "G"}
  ],
  "context_used": true
}
```

**What our side does with it**

- Joins the window to our incident log by camera and time, for evaluation.
- Live: shows the alarm on the camera's overlay and the accident marker; a confirmed alarm (agent on, and our camera report shows a vehicle still over 30 s in a lane whose light is green) goes to the response chain and to the LLM brief.
- Errors stay errors: a processing failure is reported as `error`, never as `incident_detected: false`.
- Severity stays `"UNKNOWN"` until a severity model is trained and evaluated; it is never filled from our simulator at run time.

## 10. Environment and working rules on the shared PC

The two projects share one Windows PC but keep separate code and separate Python environments, and talk through files or a subprocess.

| Item | Our project | GO-FLOW |
| --- | --- | --- |
| Folder | `D:\Projects\Urban Traffic CARLA` | `D:\Projects\Urban Traffic CARLA\GO-FLOW` (its own git repository, branch `feature/incident-detection-agent`) |
| Python | `.venv` in the project root, Python 3.11 | its own `GO-FLOW\.venv`, with the scikit-learn version that saved the model |
| Main libraries | CARLA 0.9.16 client, SUMO 1.27.1, PyTorch 2.6.0 with CUDA 12.4, Ultralytics 8.4.100 or later | Ultralytics, OpenCV, PyTorch, pandas, scikit-learn, joblib |
| Tests | `tools\test.bat` (pytest on `tests\` only, so GO-FLOW's tests are not collected) | `python -m pytest tests/test_incident_pipeline.py -q` |
| Outputs | `out\...`, logs in `tools\logs\` | new results under `out\incident_agent\` in our project, never over existing files |

**Rules**

- Never install GO-FLOW's requirements into our `.venv`: its pinned scikit-learn, LangChain and XGBoost would change a working setup. Our adapter starts GO-FLOW's Python as a separate process.
- The GPU has 4 GB. CARLA rendering uses most of it, so heavy processing runs on recorded clips with CARLA closed; live, at most 2 to 4 cameras.
- Our files are changed only through agreed changes, with a backup of anything replaced. Results are never deleted or overwritten; each run writes a new folder.
- Do not touch the free-fly camera controls: `cosim\fly_proc.py` and `DefaultInput.ini`.
- Paths contain spaces: always quote them.
- Our `.bat` files are started by double-click and write their logs to `tools\logs\`.
- No numbers are claimed without a run that produced them; every result names its run, seed and model version.

## 11. Brief for his Claude session

Paste the block below as the first message of his Claude session, with this report attached (export it as Markdown or PDF from the document menu).

```markdown
You are working on the GO-FLOW incident detection agent (repository 14-12/GO-FLOW, branch
feature/incident-detection-agent, folder agents/incident_detection). The attached report,
"GO-FLOW Incident Agent x Urban Traffic CARLA: Integration Report", explains the CARLA/SUMO
Town05 co-simulation we are integrating with, reviews the current model, and sets out the plan.
Read the whole report before doing anything.

Goal: make the agent reliable on Town05 and better on real video, in the five stages of
section 5, without breaking v1.

Rules:
- Keep v1 working: same 27 features, same output fields and meanings, same model file.
  New work goes into v2 files beside it (for example incident_detector_v2.joblib).
- Never use the held-out test set to choose anything. Report every result with its data,
  split, seed and model version; keep bad results too.
- Real (AI City) and synthetic (Town05) results are always reported separately.
- Severity stays "UNKNOWN" until a severity model is trained and evaluated.
- No LLM in the detection path. No API keys in code or notebooks.
- Work in GO-FLOW's own .venv. Never install packages into the Urban Traffic CARLA .venv and
  never edit files of that project; its outputs come to you as files (section 7).
- Ask before any change to the output contract (section 9).

First tasks, in order:
1. Read agents/incident_detection/src/{video_processing,feature_engineering,inference}.py and
   tests/test_incident_pipeline.py; tell me how they differ from the notebooks.
2. Stage 2: regroup the AI City splits by camera location (find which videos share a camera),
   re-run grouped cross-validation on data/processed/all_traffic_features_temporal.csv, and add
   per-event recall, time to detect, false alarms per camera-hour and bootstrap 95 % intervals.
   Report v1's real baseline before changing anything.
3. Change 1 of section 6: add process_frames(frames, fps, frame_size, source_id) with
   process_video as a wrapper, plus tests that both give identical windows on the same clip.
4. Change 2: detector weights, class map, confidence and image size from a config file.
Then stop and report; stages 3 to 5 start once the Town05 dataset (section 7) is delivered.
```

**His checklist**

- [ ] Share the AI City feature file (`all_traffic_features_temporal.csv`) and `video_splits.csv`
- [ ] Find which AI City videos share a camera and regroup the splits
- [ ] Write down v1's camera-grouped, per-event baseline
- [ ] Add `process_frames` and the detector config
- [ ] Agree the interface fields in section 9
- [ ] Receive the first Town05 clips and run v1 on them unchanged (stage 1)

## 12. Open questions to decide together

- [ ] Goal priority: improve his model as a deliverable (video only), or the best detection for Town05 (signal state and lane masks allowed as inputs)?
- [ ] May v2 be retrained with Town05 data, or must his submitted model stay trained on AI City only?
- [ ] Should a detection ever act in the town (police, lane closure, signals), or only be shown and logged?
- [ ] Targets for stage 5: most crashes caught within 30 s and under 1 false alarm per camera-hour, or other numbers?
- [ ] Live detector: his YOLO11n with ByteTrack, or our CARLA-trained YOLO26n through the config?
- [ ] Which cameras for the first clips: the ring road's (closest to his training data) plus a few junctions?
- [ ] Exchange of data: a shared folder on the PC, or the GO-FLOW repository (video files are large; keep them out of git)?

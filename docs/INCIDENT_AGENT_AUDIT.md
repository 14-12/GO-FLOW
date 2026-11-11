# Omar's accident agent: audit, plan and integration (11 Nov 2026)

Repository `14-12/GO-FLOW`, branch `feature/incident-carla-integration`, HEAD `03a7010` (7 Nov, 3moreeeee), read in full
from the source, not from the reports. Agent code: `agents/incident_detection/` (commits f4dc8df, 8dc5956, b19dc1f,
03a7010). The rest of that branch (`src/agents`, `src/decision`, `src/llm`, `src/backends`, `src/prediction`) is
Yesmine's traffic-prediction agent (392fbae), read only where Omar's code calls it.

Part 1 is the audit, part 2 the plan, part 3 what was built and measured.

## Summary (11 Nov, morning)

* **What it is.** A camera-only accident detector: YOLO + ByteTrack, 5 s windows, 27 features, his v1 Random Forest,
  "mean of 3 >= 0.35". Plus, in his research code, a fast "crash moment" signal (a car stopping dead while touching
  another). A library, not a service; no model passes the agreed criteria on Town05 (his own reports).
* **What was integrated.** His code, unchanged, behind a streaming adapter and two processes (YOLO in ours on the GPU,
  his model in its own venv), in the viewer with `--incident-agent "<cameras>"`, **in shadow**: shown and logged,
  never a report, never a command. Off by default; the town runs as before without it.
* **Checked.** Streaming = his batch path exactly (2 164 windows on real CARLA video, 0 differences); 30 + 10 new tests
  pass, the project's 353 too; his process killed live: back in 10 s, the viewer unaffected.
* **Measured in CARLA.** Viewer 47.4 -> 41.6 fps with 2 cameras (38-39 with 4). The GTX 1650 keeps 2 cameras at
  5 fps beside CARLA, not 4 (half the pictures lost). Graded 40-min run, 6 crashes in view: v1 caught all of them but
  is in alarm 56 % of normal time (48 false alarms per camera-hour); the crash moment saw 3 of 6 within 6 s with no
  real false alarm, on only 0.37 normal camera-hours, and flips between two near-identical cameras.
* **Decision.** Keep it in shadow. Next: crash moment + a hazard-lights class (labels from the simulator), a held-out
  test on new seeds, and only then let a confirmed alarm become an earlier *report*.


---

## Part 1. Audit

### 1.1 What the agent is

One job: say, for each 5-second window of one fixed camera's video, whether an accident is in the picture. It does
not classify, rate severity (`"UNKNOWN"`), locate, or act. It never reads SUMO or CARLA state.

```
frames (BGR) ──> YOLO + ByteTrack ──> per frame: count, box coverage, mean confidence
   (5 fps)        (persist=True)      per detection: track id, class, centre, box height, area
                                              │
                         aggregate_windows (5 s windows): 13 current features
                                              │
                         add_temporal_features: delta and roll3 of 7 of them  (27 features, causal)
                                              │
                         Random Forest (sklearn Pipeline: median imputer + RF 500 trees, depth 16)
                                              │
                         mean of the last 3 window scores >= 0.35  ->  incident_detected
```

| Part | File, function | What it does (checked in the code) |
| --- | --- | --- |
| Detector settings | `src/detector_config.py` `DetectorConfig`, `load_detector_config` | weights, class map (detector name -> car/motorcycle/bus/truck), confidence, imgsz, tracker; names checked against the model's own; env `GOFLOW_DETECTOR_CONFIG`, `GOFLOW_YOLO_WEIGHTS` |
| Frames to rows | `src/video_processing.py` `process_frames`, `result_rows`, `build_detector` | a fresh YOLO per call (tracker ids never shared), keeps every `stride`-th frame (vid_stride), one `model.track(persist=True)` per kept frame |
| Windows | `aggregate_windows` | 5 s windows from 0; speed = centre shift / dt / box height; a track "stopped" if >= 5 observations and median speed < 0.08 |
| Features | `src/feature_engineering.py` `add_temporal_features` | `diff().fillna(0)` and `rolling(3)` over the windows given; raises on duplicate window times |
| Verdict | `src/inference.py` `IncidentDetector.predict_windows` | loads the joblib, checks the feature order and rule; `incident_detected = mean(last 3 scores) >= threshold (0.35)` |
| Hand-off | `src/incident_pipeline.py` `run_traffic_pipeline` | every window to Yesmine's `TrafficPredictionControlAgent` (PEMS08 sensors) with `incident_type`/`severity` null |
| Research code (not in `src/`) | `evaluation/crash_features.py` `window_crash_features` | "crash moment": a track at under 0.1 box heights/s that was over 0.8 one second before, with its box touching another (`contact_hard_stops`); "avoidance": moving tracks passing a spot stood on for 20 s |
| | `evaluation/track_features.py`, `v3_features.py`, `train_v3.py` | stopped "spots", emergency-light frames, 60/180 s memories; the v2/v3/v4 training and scoring (`score_alarms`) |

Lifecycle: none. It is a library: one call per finished clip. `process_frames` consumes the whole iterable before it
returns (the stage-2 response says so: "the stream is consumed before windows are returned"). There is no process, no
server, no state kept between calls, no timeout, no error channel other than exceptions.

Trace from observation to decision (one window W of camera C): pictures of W -> `model.track` -> `result_rows` ->
`aggregate_windows` row W -> `add_temporal_features` uses rows W-1 and W-2 of the same call -> RF score of W ->
mean with the scores of W-1, W-2 -> `incident_detected`. A decision for W exists only after the call ends, so in
`process_frames` the first alarm of a clip comes when the whole clip is done.

### 1.2 What it has been measured to do (his results files, checked against `evaluation/results/*.json`)

| Version | Data | Crashes caught | False alarms per camera-hour | Median / slowest time to detect | Status |
| --- | --- | --- | --- | --- | --- |
| v1 (the committed model) | AI City, camera-grouped CV | 15/18 | 7.4-8.6 | 25-35 s | trained on real highway video, COCO YOLO11n |
| v1 | Town05 stage 1, 8 clips | 5/5 | 37 (40 % of normal time in alarm) | 3-5 s | in alarm almost all the time |
| v1 | Town05 v3 test runs | 10/10 | 18.8 (59 % in alarm) | 5 / 15 s | |
| v2 (Town05 v3 data) | v3 test runs | 10/10 | 0.60 | 70 / 140 s | **not accepted**; reacts to police lights; joblib not in the repo |
| v4 data (after our 3-6 Nov fixes) | training runs only, out-of-fold | 18/19 at best | 0.82 | 290 s | no model passes even on training; test runs unopened |
| impact signal alone (`contact_hard_stops` >= 1) | v4 training runs | 9/19, all within 5 s | about 1 | 5 s | not packaged; exploration only |

The acceptance criteria he proposed and we agreed (his FINAL_REPORT section 5): at most 1 false alarm per
camera-hour, at least 80 % of clearly visible crashes, median within 30 s, all within 120 s, and **"until v2 meets all
of this, nothing acts in the town"**. Nothing meets them today.

### 1.3 Strengths (with the evidence)

1. **Honest method.** Splits by whole run and by camera (`evaluation/camera_groups.json`: he found 13 AI City camera
   groups crossing the original splits), test runs opened once (`train_v3.py --final` refuses a second run), every bad
   result kept and reported, rule changes disclosed (V3 report "Limits").
2. **Causal features.** `add_temporal_features` uses only past windows; `crash_features` and `track_features` are
   causal by construction. Live behaviour can therefore match the evaluation.
3. **Clean, small, testable library.** 5 files, 290 lines; `yolo_factory` injection makes YOLO-free tests possible
   (`tests/test_incident_frames.py` `FakeModel`); 21 incident tests pass here (2 skipped: real YOLO).
4. **Defensive loading.** `IncidentDetector` rejects a feature order or temporal rule it does not know; the detector
   config rejects class names the model does not have.
5. **The crash-moment signal** (`crash_features.py`) is the only accident-specific, fast evidence anyone has measured
   on Town05: half the crashes within 5 s at about one false alarm per camera-hour.
6. **Detector-agnostic.** The CARLA detector plugs in by config (`config/detector_carla_yolo26n_t05.json`).

### 1.4 Weaknesses (evidence), and what is simply not built yet

**Weaknesses (built, and wrong or limiting):**

| # | Weakness | Evidence |
| --- | --- | --- |
| W1 | The only shipped model does not tell a crash from a jam on Town05 (19-37 false alarms per camera-hour) | STAGE1_RESULTS_v1, V3_FINAL_TEST_RESULTS |
| W2 | Its features are detector-specific: `mean_detection_confidence`, counts and box coverage were learnt on COCO YOLO11n; with the CARLA detector their distribution differs | his own warning, INTEGRATION_RESPONSE_STAGE2 section 4 |
| W3 | "Live" is batch: `process_frames` returns only after the iterable ends, rebuilds the tracker on every call, and restarts the clock at 0 | `video_processing.process_frames` (`build_detector` per call, `len(frame_rows) / processed_fps`) |
| W4 | Picture times are counted, not measured: a dropped frame shifts every later time and every speed | `result_rows(result, len(frame_rows) / processed_fps, ...)` |
| W5 | Temporal features and `mean3` are only right from the third window of a call; called on chunks, each chunk starts cold (`diff().fillna(0)` -> 0, `rolling(min_periods=1)`) | `feature_engineering.add_temporal_features`, `inference.predict_windows` |
| W6 | The speed is in box heights per second over one frame (0.2 s): noisy, and depends on the camera's distance and angle | `aggregate_windows` |
| W7 | The crash-moment rule assumes exactly 5 frames a second (`FPS = 5`, `shift(5)`) | `crash_features.add_motion` |
| W8 | The joblib is a pickle: it runs code when it loads; the scikit-learn version must match (pinned 1.6.1) | `inference.IncidentDetector`, requirements |
| W9 | The requirements file pulls LangChain, langchain-openai, XGBoost and unpinned torch/ultralytics, none needed for inference | `agents/incident_detection/requirements.txt` |
| W10 | The hand-off sends every window, alarm or not, to an agent built on PEMS08 sensors 0-169, which Town05 does not have | `src/incident_pipeline.run_traffic_pipeline` |

**Not implemented yet (missing, not wrong):** a streaming/process interface; absolute (simulation) time in the
output; timeouts and an error status; per-window evidence (which tracks, where); severity and incident type; a model
that passes the criteria; a hazard-lights class (his main recommendation, V4 report); any action.

### 1.5 Our side today (what it overlaps with)

| Our component | File | Relation to his agent |
| --- | --- | --- |
| The accidents themselves (simulation) | `cosim/incidents.py` `IncidentManager` | the ground truth; grades the agent, is never read by it |
| How an accident becomes "known" | `IncidentManager._advance`: police sent after a log-normal delay (median 240 s) from the crash, standing for "someone called it in" (`police_record["sent"]`) | **the gap his agent can fill**: a camera alarm is a second, earlier way of reporting |
| Who acts on a known accident | the police, closure and towing in `IncidentManager`; `route_guidance.incident_alerts` (only after `sent`) | the response chain; must stay the only owner of actions |
| Camera reports to the junction agents | `cosim/camera_reports.py` (`stopped_long`, sirens) | junction control; independent; unchanged |
| Detector on a camera picture | `cosim/inspector.py` (H key), `cosim/redlight_proc.py` (red-light agent) | same detector, same "own process, short queue, drop when busy" pattern |
| Camera recording with timed labels | `cosim/clip_recorder.py`, `freefly.demo_record_clips` (`--record-clips`, `--crash-focus`) | his training data; also our truth for grading a live run |
| Team agents in their own venv | `cosim/goflow_prediction.py`, `agents/goflow_traffic` (JSON lines, restarts 5/20/60 s) | the pattern to follow |

One thing found on our side: `IncidentManager._divert` sends drivers around a crash from the moment it happens, before
anyone has reported it ("as a navigation app would tell them"). A navigation app cannot know that early. It is
traffic behaviour, not an agent decision, and changing it changes the town's traffic, so it is left as it is and
listed under remaining issues.

### 1.6 Compatibility

| Question | Finding |
| --- | --- |
| Duplicate decisions | None if the agent only reports. Police, closure, towing, route alerts and signals each have one owner today; the agent must not become a second one |
| Conflicting commands | The agent issues no command; junction agents, signals, cameras' reports are untouched |
| Circular communication | None: frames go in, windows come out; nothing it outputs feeds back into what it sees |
| Data structures | His window dict is kept; we add `run`-free fields: `camera`, `sim_start_s`, `sim_end_s`, `received_sim_s`, `impact` (the crash-moment counts), `model` (name + sha256) |
| Time | His clip-relative seconds -> our SUMO seconds of the picture shown (`FreeFly.depicted_at`, as the clip recorder) |
| Latency | A window closes 5 s after it starts; YOLO about 20-40 ms a picture on the GTX 1650 (planned; measured in part 3: 31 ms alone, 93-137 ms beside CARLA); RF about 80 ms a call (measured here: 79.5 ms, `n_jobs=-1`) |
| Lifecycle | Viewer start -> sensors + detection process + his process; viewer end -> stop, join 5 s, kill. Nothing in the SUMO step or the picture path waits for it |
| Python environments | Ours has torch/ultralytics, no pandas/scikit-learn. His needs pandas + scikit-learn 1.6.1. Rule (integration report section 10): never install his requirements in our venv. So: YOLO in our venv (GPU), his model in its own CPU venv |
| Licence | The GO-FLOW repository has no LICENSE file; the code is our own team's (same repository, same organisation). Recorded in the package README; no third-party code is copied |
| Interference with junction agents / lights / cameras | 4 extra CARLA camera sensors at 5 fps and one more YOLO on the GPU (4 GB). Measured in part 3 (fps, SUMO thread) |

**Ownership.** Seeing (pictures -> alarm): his agent. Grading: SUMO's record, offline, never read by the agent.
Deciding and acting (police, closure, route alerts): `IncidentManager` and `route_guidance`, as today. A camera alarm
may become a *report* (an earlier `sent`) only after the acceptance criteria are met on held-out runs. Until then it is
shown and logged (decision 4 of his FINAL_REPORT section 6.6, which we confirmed).

### 1.7 Component decisions

| Component | Decision | Why |
| --- | --- | --- |
| `src/feature_engineering.py` | **Keep** unchanged | causal, tested, the model's contract |
| `src/inference.py`, `models/incident_detector.joblib` (v1) | **Keep** unchanged, shadow only | his only shipped model; its Town05 numbers say it must not act; running it live gives its first live measurement |
| `src/video_processing.aggregate_windows` | **Keep** unchanged, called on a rolling buffer | the window statistics the model was trained on |
| `src/video_processing.process_frames` | **Adapt**: a streaming wrapper | W3-W5: one tracker per camera for the whole run, picture times from the simulation, a window out as soon as it closes, temporal features and mean3 over the camera's whole history |
| `src/video_processing.result_rows` | **Adapt**: the same rows built in the detection process (pandas is not in our venv); equivalence tested | the YOLO runs where the GPU and torch are |
| `src/detector_config.py`, `config/detector_carla_yolo26n_t05.json` | **Keep** unchanged; weights from our `models/` through `GOFLOW_YOLO_WEIGHTS` | the config's path is absolute to one PC |
| `evaluation/crash_features.py`, `track_features.py` | **Keep** unchanged, used live as a second channel ("impact") | the fastest accident-specific evidence measured |
| `evaluation/v3_features.py`, `train_v3.py`, v2 model | **Reject for now** | v2 not accepted, not in the repo; it is mostly a police-light detector |
| `src/incident_pipeline.py` + `TrafficPredictionControlAgent` (PEMS08) | **Reject** | PEMS08 sensors are not Town05's; the town already has `agents/goflow_traffic`; a second consumer would duplicate decisions |
| LangChain explanation (notebook) | **Reject** | no LLM in the detection path (his own rule); the town has `cosim/llm.py` |
| `requirements.txt` | **Replace** by a pinned inference list | W9 (his own report, change 10) |
| `evaluation/train_v3.score_alarms` | **Adapt**: the same scoring rules on our live logs | so live and offline numbers mean the same |
| Severity `"UNKNOWN"` | **Keep** | no severity model exists |
| Acting in the town | **Not now** | criteria not met |

---

## Part 2. Plan

### 2.1 Architecture

```
viewer process (cosim/freefly.py)                       detection process (our venv, GPU)          his process (own venv, CPU)
───────────────────────────────                         ─────────────────────────────────          ───────────────────────────
CARLA sensors, 4 cameras, 5 fps  ──JPEG, short queue──>  one YOLO + ByteTrack per camera  ──JSON──>  agents/incident_detection/serve.py
picture time = depicted SUMO s      (full: dropped,      (persist), his class map            lines   StreamingWindows per camera:
                                     counted)            frame row + detection rows                    aggregate_windows (his)
                                                         (= his result_rows)                           add_temporal_features (his)
                                                                                                       IncidentDetector (his v1)
IncidentAgent.poll()  <──────────── windows ─────────────  restarts his process 5/20/60 s  <──JSON──   window_crash_features (his)
  latest per camera -> overlay line                       then leaves it off                lines
  windows.jsonl, truth.jsonl (grading only)
```

* **Position.** Beside the town, like the red-light agent: in no loop that makes a picture or a SUMO step.
* **Inputs.** Camera pictures only (what a city has). **Outputs.** One record per camera and window: his v1 fields,
  plus `camera`, `sim_start_s`, `sim_end_s`, `received_sim_s`, `impact` and `model`.
* **Decisions.** None leave the agent. The viewer shows them and logs them. Grading reads SUMO's record afterwards.
* **Conflicts.** Impossible by construction: no command path exists.
* **Failure.** Detection process down: the viewer says so and goes on. His process down: restarted after 5, 20, 60 s,
  then off. A full queue drops pictures (counted). No CUDA: the agent stays off (CPU YOLO cannot keep up).
* **Off by default.** `--incident-agent "<cameras>"` turns it on. Without it nothing new is imported or started.

### 2.2 Files

| File | New / changed | Content |
| --- | --- | --- |
| `agents/incident_detection/src/*`, `config/*`, `models/incident_detector.joblib`, `evaluation/crash_features.py`, `evaluation/track_features.py` | new, **byte-identical** to 03a7010 (md5 checked) | his code |
| `agents/incident_detection/stream.py` | new | `StreamingWindows`: his functions on a rolling buffer, per camera |
| `agents/incident_detection/serve.py` | new | JSON lines in/out, one `StreamingWindows` per camera |
| `agents/incident_detection/requirements.txt`, `README.md`, `tests/` | new | pinned inference deps; what is his, what is ours; tests |
| `cosim/incident_agent_proc.py` | new | the detection process (YOLO, tracker, rows, his process's life) |
| `cosim/incident_agent.py` | new | sensors, queue, status, logs, overlay text |
| `cosim/freefly.py` | changed | `--incident-agent`, `--incident-agent-fps`; start, poll, close; one overlay line; summary JSON |
| `tools/setup_incident_agent.bat` | new | his venv + its tests |
| `tools/eval_incident_agent.py` | new | grading (his rules) and the offline comparison |
| `tests/test_incident_agent.py` | new | viewer side with fakes |

### 2.3 Dependencies and configuration

His venv (`agents/incident_detection/.venv`, Python 3.11): numpy, pandas, scikit-learn==1.6.1, joblib, pytest. Our
venv: unchanged (torch 2.6 + CUDA, ultralytics 8.4.177, opencv). Detector: `models/carla-yolo26n-t05.pt` with his
`detector_carla_yolo26n_t05.json` class map, confidence 0.25, imgsz 960, `bytetrack.yaml`, FP16 (`quantize=16`, as
the inspector).

### 2.4 Risks

| Risk | Mitigation |
| --- | --- |
| GPU memory (4 GB) and frame rate | 5 fps, 4 cameras max (part 3: the PC keeps 2); measured with and without; off by default |
| A pickle from another repository | only the committed file, md5 recorded; never a path from outside |
| Streaming differs from his batch path | test: same rows -> identical windows and scores |
| Live drops change his features | logged per window; live vs offline compared on the same scene |
| Someone reads a shadow alarm as a report | the overlay says "shadow, not reported"; nothing is wired to the police or route alerts |

### 2.5 Tests

1. **Functional (his venv):** streaming == batch (`aggregate_windows` on the whole stream, then
   `add_temporal_features` and `predict_windows`), field by field, on synthetic streams with crashes, gaps, empty
   windows; serve protocol; his own tests still pass.
2. **Functional (our venv):** the rows built from a fake Ultralytics result equal his `result_rows`; queue drops;
   restart policy; status; the overlay text; the grading rules on hand-made cases.
3. **Integration on the PC:** the detection process on a recorded clip (`out/incident_clips/20261106_141418`) through
   his process, against his own batch path on the same frames.
4. **Real CARLA:** the viewer with the agent on 4 ring-road cameras, crash focus 0.5 (option A), dev seed, 30 min;
   graded against SUMO's record. A run with his process killed halfway.
5. **Before / after:** the same 10-minute run with the agent off and on: viewer fps, SUMO thread time, GPU memory.

---

## Part 3. What was built and measured

### 3.1 Built

| File | What |
| --- | --- |
| `agents/incident_detection/src/*`, `config/*`, `models/incident_detector.joblib`, `evaluation/crash_features.py`, `evaluation/track_features.py` | his, byte-identical to `03a7010` (git blob hashes equal; the joblib md5 `02ec5f47…`, sha256 `7ce31bb81739…`) |
| `agents/incident_detection/stream.py` | `StreamingWindows`: a window out as soon as it closes; his `aggregate_windows`, `add_temporal_features` (inside `predict_windows`), `IncidentDetector` and `window_crash_features`, unchanged, on what each window needs (each track's last 10 detections, the last 6 windows) |
| `agents/incident_detection/serve.py` | his process: JSON lines; `n_jobs=1` on his forest (same votes; 89 instead of 147 ms a window) |
| `agents/incident_detection/replay.py` | recorded rows his batch way and streamed, compared window by window |
| `agents/incident_detection/requirements.txt`, `README.md`, `tests/` | pinned inference deps; provenance; 30 tests |
| `cosim/incident_agent_proc.py` | detection process (our venv, GPU): `Detector` (one `track(persist=True)` per camera, or batched), `rows_of` (= his `result_rows`), `AgentChild` (his process: short queue, restarts 5/20/60 s, then off) |
| `cosim/incident_agent.py` | viewer side: CARLA cameras at 5 fps, picture time = the SUMO second shown, JPEG queue that drops when busy, verdicts logged (`out/incident_agent/<time>/windows.jsonl`), a note on a crash moment, status, summary |
| `cosim/freefly.py` | `--incident-agent "<cameras>"` (off by default): start, poll, close, summary (`out/town05_freefly_incident_agent.json`, and in the demo JSON) |
| `tools/setup_incident_agent.bat` | his venv (Python 3.11) + `lap` in ours (ByteTrack) + his tests |
| `tools/setup_ml.bat` | `lap` added (ByteTrack needs it; it was missing from our venv) |
| `tools/incident_agent_rows.py`, `tools/incident_agent_detcheck.py`, `tools/eval_incident_agent.py` | rows from recorded clips; batched vs one-by-one detector; grading by his rules |
| `tests/test_incident_agent.py` | 10 tests, no CARLA / GPU / his venv |

Nothing in `cosim/incidents.py`, `route_guidance.py`, the junction agents, the signals or the camera reports was
changed. No path exists from a verdict to any of them.

### 3.2 Tests and results

Everything below was run on the project PC (GTX 1650 Max-Q 4 GB, CARLA 0.9.16, SUMO 1.27.1) unless said otherwise.
Logs: `Claude outputs\jobs\done\{d_ia_setup,d2_ia_offline,e1_ia_replay,e_ia_perf_off,f_ia_perf_on,g1_ia_perf_two,
g3_ia_perf_batch,g2/g4_ia_detcheck,h_ia_live_two,i_ia_fault,j_ia_live_offline}.bat.log`; results in
`out\incident_agent\`.

**Functional (passed).**

| Test | Result |
| --- | --- |
| His own incident tests on his branch (VM, Python 3.13, scikit-learn 1.6.1) | 21 passed, 2 skipped (real YOLO) |
| `agents/incident_detection/tests` in his venv on the PC: streaming = his batch path on synthetic streams (crashes, a 16 s gap, 30 % pictures lost, 10 fps); our rows = his `result_rows` (20 random cases); the process protocol (bad lines answered, never fatal) | 30 passed |
| `tests/test_incident_agent.py` (our venv): rows, class map against the CARLA detector's names, his process restarted 3 times then left off, drops instead of waits, verdict counting and notes, his scoring rules, the viewer option | 10 passed |
| The whole project suite after the change | 353 passed (before the batched detector; re-run at the end: see 3.4) |

**Streaming against his batch path on real pictures (passed, exact).** The 25-minute ring-road trial of 6 Nov
(`out\incident_clips\20261106_141418`, 4 cameras, 29 942 pictures at 5 fps) through the live detector, then through
`replay.py`: 1 202 windows, **0 differences** (largest score difference 0.0, features 0.0, v1 and crash-moment
verdicts the same in every window).

One thing found on the way: in clips recorded before the evening of 6 Nov, `sim_s` is in whole SUMO steps (0.5 s), so
3 of 5 pictures at 5 fps share the time of the one before. Fed as they are, his speeds divide by zero (NaN) and the
two paths differed in 136 of 1 202 windows. `incident_agent_rows.py` now times pictures by CARLA's own clock put on
SUMO's (gap steady within 0.1 s over 25 min); live pictures were never affected (`depicted_at` is continuous).

**Batched detector against one `track()` per camera (passed, within FP16).** 2 000 pictures of 4 cameras:
raw boxes 0 unmatched of 815, centres within 0.56 px; tracked rows 6 778 against 6 773, 9 unmatched, 1 class
different, the same track identity for 99.97 % of rows. Offline it is faster (25.8 against 31.3 ms a picture); beside
CARLA it was not (below), so the default stays his way.

**Performance, before / after (measured).** Seed 11, 09:00, the 300 s demo flight, 1280x720:

| Run | Viewer fps (mean / 1 % low) | Detector, ms a picture | Pictures dropped | Verdict delay, s (median / p95 / max) |
| --- | --- | --- | --- | --- |
| agent off | 47.4 / 33.9 | | | |
| 4 cameras, one by one | 38.2 / 25.2 | 93-99 | 53 % | (on SUMO's clock, not comparable) |
| 4 cameras, batched | 39.1 / 26.9 | 108 | 54 % | 1.46 / 3.02 / 4.72 |
| 2 cameras, one by one | 41.6 / 27.8 | 97 | 10 % | 0.73 / 2.28 / 3.08 |

The SUMO thread was not slowed (median 1 900 against 2 085 ms busy in 10 s with 4 cameras; its peaks are the
town's). The GPU is the limit: offline the detector takes 31 ms a picture, beside CARLA 93-137 ms. **This PC keeps
2 cameras at his 5 fps**; with 4, half the pictures are lost and his per-picture features (speeds, the crash moment's
"one second before") no longer mean what they were trained on. The verdict delay is the time from a window's end to
the viewer having it, on the pictures' own clock; windows are 5 s, so an alarm comes 5-6 s after the first picture of
the window that shows it.

**Failure (passed).** Seed 13, 2 cameras, his process killed after 90 s (`GOFLOW_INCIDENT_KILL_AFTER_S=90`): the viewer
went on (37.6 fps), the process was back and ready 10 s later (5 s wait + 4 s load), one window of each camera was
lost, nothing else. In the other runs: 0 errors, 0 restarts.

**Real CARLA: shadow run with crashes in view (executed and graded).** Seed 12, 09:00, 40 min, careless share 0.1,
crash focus 0.5 (option A) on the two cameras 2328-2 looking W and 2014-3 looking E, which also recorded the clips
for grading. 9 crashes in town, 6 counted (impact on screen, wreck within 35 m, after the first minute). Because the
focus puts crashes before these cameras, a wreck was in view 72 % of the time: only 0.37 camera-hours of normal
traffic. 22 709 pictures processed, 5.5 % dropped, verdict delay 0.44 / 1.64 / 2.46 s.

| Channel | Crashes caught | Time to detect (window end), median / slowest | Delay as lived (verdict at the viewer - crash) | False alarms | Normal time in alarm |
| --- | --- | --- | --- | --- | --- |
| v1 (his model) | 6 / 6 | 10 / 15 s | 7.5-20 s | 18 in 0.37 h = **48 / h** | **56 %** |
| crash moment | 2 / 6 (one at 5 s, one 115 s late) | | 6 s for the fast one | 1 in 0.37 h (2.7 / h) | 0.4 % |
| criteria | >= 80 % | <= 30 / 120 s | | <= 1 / h | |

* v1 "catches" everything because it is in alarm more than half the time with no crash in view; it fails the
  false-alarm criterion by a factor of 48, as on his recordings (19-41 / h).
* The crash moment's one false alarm is the window 265-270 s on 2328-2, which ends 2 s after crash 2's impact
  (t0 = 268 s): the window's middle (267.5 s) falls before the wreck is in view, so the rule counts it as false. Read
  by hand, it is crash 2 seen within 2 s. On that reading the crash moment saw 3 of 6 crashes within 6 s and raised no
  false alarm in 0.37 h. Too few hours to claim a rate; it matches his offline 9 of 19 at about 1 / h.
* Offline, his batch path on the same run's recorded clips (`j_ia_live_offline`): see 3.3.

**Offline on the 6 Nov trial (CARLA closed), his batch path:** 1.22 normal camera-hours, 2 counted crashes: v1 2 / 2
(5 and 100 s) with 36 false alarms (29.6 / h, 33 % in alarm); crash moment 1 / 2 within 5 s, 0 false alarms.

**Not executed.** A held-out test in his sense (new seeds, opened once, at least 30 crashes): the night's runs are
development runs (dev seeds 11-13; seed 12 for the graded run). Junction cameras live. Four cameras at 5 fps live
(the PC cannot keep them). Acting on a verdict (by design).

### 3.3 Live against offline on the same run

The graded run's own recordings (the clip recorder's cameras at the same poses, 10 fps, H.264) through his batch path
(`j_ia_live_offline`): the streaming adapter again matched his batch path exactly (962 windows, 0 differences). Live
against offline, window by window: v1's verdict the same in 90.5 % of windows (score difference median 0.05, 90th
percentile 0.18); both 25 pictures in the median window. The scores are close but not equal because the two are
different cameras at the same pose (their own picture instants, JPEG against H.264) and 5.5 % of live pictures were
dropped.

The crash moment is fragile to that: live and offline both saw crash 2 (window 265-270 s); live saw crash 1 at 5 s,
offline crash 4 at 5 s instead, and offline raised one more false alarm (2328-2, 900-905 s). Offline grading: v1 6 / 6
with 48 / h false alarms (51 % of normal time in alarm); crash moment 1 / 6 counted (+ crash 2 in the window before,
as above), 2 false alarms in 0.37 h. A signal that flips on which of two near-identical cameras looks needs many more
crashes before any rate is quoted.

### 3.4 Dependencies and configuration

* His venv `agents/incident_detection/.venv` (Python 3.11): numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1 (the
  version that saved his model), joblib 1.4.2, pytest 8.4.2. Not in it: torch, ultralytics, OpenCV, LangChain,
  XGBoost.
* Our venv: `lap` 0.5.14 added (ByteTrack needs it; `tools/setup_ml.bat` and `tools/setup_incident_agent.bat` install
  it). Nothing else changed.
* Detector: `models/carla-yolo26n-t05.pt` with his class map, confidence 0.25, imgsz 960, `bytetrack.yaml`, FP16.
* Switches: `--incident-agent "<cameras>"` (viewer); `GOFLOW_INCIDENT_BATCH=1` (batched detector);
  `GOFLOW_YOLO_WEIGHTS` (other weights); `GOFLOW_INCIDENT_KILL_AFTER_S` (fault test only).

### 3.5 Remaining issues

1. **No model meets the criteria on Town05.** v1 is at 30-48 false alarms per camera-hour live and offline. It stays in
   shadow; nothing can act on it.
2. **The crash moment is fast but sees about half the crashes** (his 9 / 19; here 3 / 6 and 1 / 2). Its false-alarm
   rate rests on 1.6 normal camera-hours tonight and his 13.4; not enough to accept it alone.
3. **GPU budget:** 2 cameras at 5 fps beside CARLA on the GTX 1650. More cameras need another GPU, a lighter
   detector setting (to be measured, since his features depend on it), or recorded video.
4. **Window labels by the window's middle** can call the impact window "false" when the impact is in its second half
   (crash 2 tonight). His rule, kept so numbers compare with his; worth a margin of one window in the next version.
5. **Ultralytics numbers all trackers' tracks from one counter and resets it when a tracker is made.** With one
   `track()` model per camera in one process, a camera's second tracker made later can hand out ids still alive on the
   first camera. Here the trackers are all made within the first second; the batched path makes them together.
6. **Our side, found during the audit:** `IncidentManager._divert` sends drivers around a crash from the moment it
   happens, before it is reported. Not changed (it changes the town's traffic and is not part of this task).
7. Clips recorded before the evening of 6 Nov carry SUMO-step times; use CARLA's time for them (done in the tool).

### 3.6 Recommendations

1. **Keep v1 in shadow; do not wire it to anything.** Its numbers on Town05 are clear.
2. **Next model: the crash moment plus hazard lights**, as Omar proposed. Our side can produce the labels without
   hand work: the simulator knows which cars have their hazards on (`IncidentManager.light_bits`, 2 s after a hit), so
   a `hazard_lights` class can be labelled automatically for the CARLA detector (`tools/train_detector.py`).
3. **When a model passes on a held-out set** (new seeds 91+, at least 30 crashes, opened once): let a confirmed alarm
   become a *report* only: an earlier `police_record["sent"]`, exactly where the simulated call puts it today, so
   the police, the closure and the route alerts keep one owner each. Then measure the gain: time from crash to
   police on scene and to the route alerts, paired dev seeds, before / after.
4. **Do not send his windows to Yesmine's PEMS08 agent.** If an accident context is wanted in the town's traffic agent
   (`agents/goflow_traffic`), pass the *report*, not raw windows, after point 3.
5. For his next delivery: score with a one-window margin around the impact, report verdict delay as lived, and keep
   using the streaming adapter's replay to check that the live path and his training path agree.

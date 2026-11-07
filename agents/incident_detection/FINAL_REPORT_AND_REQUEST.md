# GO-FLOW Accident Agent × Urban Traffic CARLA: Final Report and Data Request

Nov 1, 2026 · reply to the stage-1 delivery and the full recordings of 31 Oct

## 1. Summary

- **The agent's job.** It detects **accidents only**. Traffic jams, red-light queues and congestion, including congestion caused by a crash out of view, must never raise an alarm. Your labels already follow this: crash in the picture = 1, everything else = 0.
- **v1 does not work on Town05.** It catches every crash, but it is in alarm about 40 % of normal time (31–37 false alarms per camera-hour).
- **Three v2 approaches failed** at under 1 false alarm per camera-hour. All of them looked for "stopped vehicles", and the normal runs are full of long jams and blocked junctions.
- **The approach that works on jams.** Two new accident-specific signals stay quiet in jams, at about 0.2 false alarms per camera-hour on 9 hours of normal traffic:
  - a crash moment;
  - cars driving around a stopped car.
- **Why we cannot finish yet.** The training runs contain only **one clearly visible crash**, which is not enough to build or validate a model. Your detector, which sees police and ambulance lights, is also needed.

**This is our last request.** Section 6 lists everything we need to finish v2 without another round of questions. Please answer all of it in one delivery.

## 2. What we did with your data

| Step | Data | Result |
|---|---|---|
| Stage 1: v1 unchanged | 8 clips, COCO YOLO11n | 5/5 accidents caught, but in alarm 40 % of normal time; 37 false alarms per camera-hour |
| Feature extraction | all 39 recordings (18 h), YOLO11n + ByteTrack, one row per 5 s window plus every vehicle track | done |
| v1 on the full held-out runs | 4 runs `20261030_*`, 10 crashes in view | 10/10 caught; 31 false alarms per camera-hour; 39 % of normal time in alarm |
| v2a: v1 features retrained on AI City + Town05 | train `20261031_*` | 0.5 false alarms per hour, but 1/10 crashes |
| v2b: "spot stopped for a long time" | same | 0–2/10 crashes |
| v2c: "stopped inside a driving lane" | training runs only | failed on training data, so the test was not run |
| Accident-specific signals | training runs only | quiet in jams (0.2 false alarms per hour); see section 3 |

**Labels and split.**
- **Split:** we used yours. The 4 runs `20261031_*` were used for training. The 4 runs `20261030_*` were held out.
- **Labels:** 1 while a wreck is in the picture (`incidents.json` "visible", plus the 2.2 s render lag), otherwise 0.

**The held-out runs are now used up.** The `20261030_*` runs were scored several times during development, so they are no longer a clean test set. A new held-out set is needed (section 6).

## 3. What we learned

1. **The "normal" runs contain big jams.** At 10 % careless drivers there are still about 5 crashes per run somewhere in town. The lane closures back traffic up across the city. For example:
   - ring-road camera 2014-3 shows every lane stopped for about 30 minutes;
   - camera 1722-4 shows a police car and a blocked junction queue.

   These periods are correctly labelled normal, but any rule based on stopped cars alarms on them.
2. **Parked cars and false detections.** Most cameras have places held by a still "vehicle" for 50–100 % of the time.
3. **Signs that separate an accident from a jam:**
   - **Crash moment:** a car goes from driving to standing within about 1 s while touching another car. On 9.1 h of normal traffic this gives 0.22 false alarms per camera-hour.
   - **Avoidance:** moving cars pass a car that has stood for at least 20 s, going the way it was going. Nobody passes in a jam.
   - **Emergency vehicles:** police cars or ambulances with lights, and tow trucks, stopped at the scene. This needs your detector.
4. **The impact looks soft on screen.** In the one clear training crash (run 20261031_003054, camera 2014-2, rear-end at 81 km/h), the cars slow from driving to standing over **2–3 s**, like hard braking.
   - With a 1 s rule the crash is missed.
   - With a 3 s rule it is caught after 15 s, but false alarms rise to 7.6 per camera-hour.

   We think the viewer's position interpolation smooths the impact.
5. **Far crashes are not realistic targets.** The crashes 83–87 m from the camera are a few pixels wide.

## 4. Deliveries done on our side

All code is in `agents/incident_detection/` of 14-12/GO-FLOW. Branch `feature/incident-carla-integration` holds the stage-2 work (commit b19dc1f). The files below are on our PC and will be pushed to the same branch.

| File | What it does |
|---|---|
| `src/` | `process_frames()` for live frames, and an optional `detections_out` for vehicle tracks. v1 is unchanged; 199/199 tests pass. |
| `config/detector_carla_yolo26n_t05.json` | Your detector with the agreed class map |
| `evaluation/run_stage1.py` | Runs and scores a delivery folder (`labels.json` plus clips) |
| `evaluation/extract_town05.py` | Windows and tracks from your `incident_clips` layout |
| `evaluation/crash_features.py` | The crash-moment and avoidance signals |
| `evaluation/v2_split.json` | The split used so far |
| `evaluation/results/` | Every result in this report, as JSON |

## 5. What v2 must reach

We propose these acceptance criteria. If you or Firas want other numbers, please state them in the delivery; otherwise we use these.

1. **False alarms:** at most 1 per camera-hour on held-out normal runs, with jams and red lights included.
2. **Recall:** at least 80 % of clearly visible crashes caught.
3. **Speed:** median detection within 30 s of the impact on screen, and every catch within 120 s.
4. **Scope:** results reported separately for cameras seen and not seen in training. Far crashes (over 50 m) are reported but not counted in the criteria.
5. **Until v2 meets all of this, nothing acts in the town.** Detections are shown in the viewer and logged only.

## 6. What we need from you (complete list)

### 6.1 The detector (required)

- `carla-yolo26n-t05.pt`, its settings file `carla-yolo26n-t05.json`, and the Ultralytics version used.
- Its validation results per class, especially for `police_lights`, `ambulance_lights`, `truck` and `car`: precision, recall and mAP50.

### 6.2 New crash recordings (required)

**Settings:**
- normal setting (10 % careless drivers);
- **new seeds**, not any run delivered so far.

**How many crashes:**
- at least **30 crashes clearly visible** to a fixed camera;
- 30 is the minimum: about 20 for training and 10 held out.

**What counts as clearly visible:**
- the wreck is about 30 m or less from the camera;
- the wreck is fully inside the picture, not at the frame edge;
- the impact happens on screen, not before the recording starts.

**Recording length:**
- at least **2 min before** the impact;
- at least **5 min after** it, so police arrival and recovery are included.

**Variety:**
- ring-road and junction cameras, roughly half each;
- different times of day;
- different crash kinds (rear-end, side, others the simulator produces);
- at least 3 held-out crashes on cameras that never appear in the training runs.

**Normal traffic:** keep the full recording of every camera in these runs, so we also get normal traffic, jams included, from the same runs.

**Split, decided by you before delivery:**
- folder `train/`: about two thirds of the runs;
- folder `test/`: about one third of the runs.

We will open `test/` only once, for the final scoring.

### 6.3 Labels for every crash (required)

Please use the same layout as `out/incident_clips/<run>/`: `run.json`, `incidents.json` with "visible", and `<camera>.mp4` with `<camera>.frames.csv`. Also add one `crashes.json` per delivery, with one entry per crash in view:

```json
{"run": "...", "camera": "2014-2 looking W", "incident": 2,
 "kind": "rear-end", "impact_speed_kmh": 81,
 "crash_sim_s": 204.0, "impact_on_screen_video_s": 172.3,
 "wreck_vehicles": ["live73", "hw19"], "lane": "37_3",
 "distance_to_camera_m": 13, "fully_in_frame": true,
 "wreck_in_picture_video_s": [172.3, 335.0],
 "police_on_scene_video_s": 210.0, "tow_arrival_video_s": 260.0}
```

### 6.4 Recording settings (required)

- Same cameras as before: 960x540, 10 fps, H.264, and `frames.csv` with `sim_s` for every frame.
- **The impact smoothing:** is it the viewer's position interpolation? If it can be switched off, please record the new runs with real simulation positions. If not, tell us the interpolation settings so we can allow for them.

### 6.5 Optional, but very useful

1. **2D boxes of the wreck vehicles and emergency vehicles,** per frame, in each camera's image. CARLA can project actors to the camera. This lets us check that the detector sees the wrecks.
2. **5–10 "look-alike" events with no crash,** in the same format: a vehicle broken down or stopped in a lane, a police car stopped without a crash, a long red-light queue. They test false alarms directly.

### 6.6 Decisions (please confirm in one line each)

1. The acceptance criteria in section 5: accepted, or changed to what?
2. v2 may be trained on Town05 data, as a separate model, with v1 kept unchanged.
3. Results go to `out\goflow_exchange\results\`, one JSON per clip, naming the detector config and model version.
4. Detections are shown and logged only until section 5 is met; acting in the town is a later, separate decision.

## 7. What we will deliver after receiving it

1. **v2.** It will use:
   - the crash-moment signal;
   - the avoidance signal;
   - stopped emergency vehicles from your detector;
   - the v1 features.

   Thresholds are tuned on `train/` only, with jams counted as false alarms.
2. **One scoring on `test/`**, against section 5, with every crash and false alarm listed. The results go to `out\goflow_exchange\results\`.
3. **The v2 model and config** in GO-FLOW, plugged into `process_frames()` for the viewer. The output keeps v1's fields and adds `run`, `window_start_time_s` and `window_end_time_s`.

## 8. Message for your Claude session

```markdown
GO-FLOW final request (attached: FINAL_REPORT_AND_REQUEST.md). The GO-FLOW agent detects accidents ONLY,
never jams. v1 alarms ~40 % of normal Town05 time; 3 v2 approaches failed because the "normal" runs are full
of jams. New accident-specific signals (crash moment, cars driving around a stopped car) stay quiet in jams
(~0.2 false alarms per camera-hour), but only one clearly visible training crash exists.
Please prepare ONE delivery with everything in section 6:
1. carla-yolo26n-t05.pt + its .json settings + per-class validation results (police_lights, ambulance_lights).
2. New runs, new seeds, 10 % careless: at least 30 crashes clearly visible (wreck within ~30 m, fully in
   frame, impact on screen, >=2 min before and >=5 min after), mixed ring road/junction, times of day and
   crash kinds, full recordings kept; split by you into train/ (~2/3) and test/ (~1/3), with >=3 test
   crashes on cameras not in train/.
3. Same layout as out/incident_clips plus crashes.json with the fields in section 6.3.
4. Answer whether the 2-3 s soft impact on screen is the viewer's interpolation, and record without it if
   possible.
5. Optional: per-frame 2D boxes of wreck and emergency vehicles; 5-10 look-alike events with no crash.
6. Confirm the four decisions in section 6.6.
Write to out\goflow_exchange\stage2_<date>\ with a README. Do not edit GO-FLOW files.
```

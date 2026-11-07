# GO-FLOW Incident Agent: Stage 1 Results on Town05

Oct 30, 2026 · reply to "Urban Traffic CARLA → GO-FLOW: Stage 1 Delivery"

## 1. Summary

v1 was run unchanged on all 8 stage-1 clips with the COCO YOLO11n detector (v1 as trained). The run with `carla-yolo26n-t05` is ready but has not been done yet: the weights file is not in the delivery (see section 6).

**v1 does not transfer to Town05.** It catches all 5 accidents in view, but it is in alarm for **40 % of normal time**, so the catches mean little. On the AI City baseline the figure was about 8 false alarm episodes per camera-hour; here it is **37 per camera-hour**, and that count understates the problem, because one long alarm counts as one episode.

This confirms the delivery's proposal: v1 must not trigger anything in the town.

**Agreements taken over from the delivery:**
- the class map;
- the time base: SUMO seconds of each clip's own run, with `run` carried along;
- the 2.2 s render lag;
- one result file per clip and detector, naming the clip, the detector config and the model version;
- the section 12 proposals, pending Firas.

## 2. What was added on the GO-FLOW side

| File | What it does |
|---|---|
| `config/detector_carla_yolo26n_t05.json` | Your detector with the agreed map: car, van, police, police_lights → car; bus → bus; truck, ambulance, ambulance_lights → truck; pedestrian left out; `imgsz` 960. It replaces the draft example. |
| `evaluation/run_stage1.py` | Reads `labels.json` and `clips/`, runs v1 unchanged per clip and detector, scores each clip, and writes `<clip>__<detector>__v1.json` plus a `stage1_summary_*.json`. `--rescore` rescores the saved windows without running YOLO again. |
| `tests/test_incident_stage1.py` | Tests the scoring rules (catch, time to detect, false alarm episodes, crash before frame 0). |

**Each window in the result files:**
- keeps every v1 field unchanged;
- gains `run`, `window_start_time_s` and `window_end_time_s` (`sim_s_frame0` + video seconds);
- gains `label_incident_in_view`, used for scoring.

**Model version.** Each file carries `model.version` = `v1` and the first 12 hex digits of the SHA-256 of `incident_detector.joblib`.

**Scoring rules.** Times are clip seconds as seen on screen.
- **In view:** an accident is in view from `max(0, crash_on_screen_clip_s)` to the end of `wreck_in_picture_clip_s` plus 2.2 s.
- **Incident window:** a 5 s window whose midpoint is in view.
- **Caught:** at least one alarm on an incident window.
- **Time to detect:** the end of the first such window minus the on-screen crash second. It is 0 when the crash happened before frame 0.
- **False alarm episode:** a run of alarm windows with no incident window.
- **Share of normal time in alarm:** the number that matters most here. It counts every alarm window outside the accident time.

## 3. Results: v1 with COCO YOLO11n

| Clip | Normal time in alarm | False alarm episodes | Accident | Caught after | Alarms in the minute before the crash showed |
|---|---|---|---|---|---|
| ring1 crash aftermath | 56 % | 2 | #1 (before frame 0) | 5 s | (in view from frame 0) |
| ring2 normal | **69 %** | 3 | — | — | — |
| ring3 crash | **100 %** | 0 | #1 (before frame 0), #3 | 5 s, 3 s | #3: 11 of 11 windows |
| ring4 normal | 17 % | 2 | — | — | — |
| junction1 far crash | 18 % | 0 | #2 (85 m away) | 5 s | 2 of 11 windows |
| junction2 far crash | 50 % | 0 | #4 (87 m away) | 3 s | 6 of 11 windows |
| junction3 normal | 30 % | 4 | — | — | — |
| junction4 normal | 35 % | 4 | — | — | — |
| **All 8 clips** | **40 %** | **15 in 24 min of normal time (37 per hour)** | | **5 of 5** | |

The pre-crash alarm counts are in each result file as `alarm_windows_in_60s_before_view`.

**Reading the table:**
- **The alarm is on almost regardless of the accident.** In ring3, v1 was in alarm throughout the minute before crash #3 appeared. In ring2, a clip with no accident, it is in alarm 69 % of the time.
- **Scores barely separate.** The mean score is 0.28 on normal windows and 0.48 on incident windows. Many normal windows score above the 0.35 threshold.
- **junction1 is the only possible genuine signal.** There, alarms jump from 2 of 11 windows before the crash to all windows after it. The crash is 85 m away, so this may be the queue it causes rather than the crash itself. It should be checked on the video.
- **ring3's "normal" gap is not really normal.** The 23 s between the end of #1's recovery and crash #3 is disturbed traffic, so its 100 % is harsh. ring2 and the junctions have no such excuse.

## 4. Why: the features are out of v1's range

v1 learned on AI City highway cameras: high, far, free-flowing traffic. In Town05 the cameras are 9 m up at −22°, and junction queues are normal. Mean features per window:

| Feature | AI City normal | AI City incident | Town05 clips (range of clip means) |
|---|---|---|---|
| `stopped_ratio` | 0.04 | 0.16 | **0.15 – 0.84** |
| `median_normalized_speed` | 0.38 | 0.50 | **0.02 – 0.21** |
| `bbox_coverage_proxy` | 0.015 | 0.029 | **0.011 – 0.15** |
| `mean_detection_confidence` | 0.15 | 0.45 | **0.18 – 0.61** |
| `mean_vehicle_count_per_frame` | 1.8 | 5.0 | **1.0 – 6.1** |

In AI City, stopped vehicles, large boxes and confident detections went with incidents. In Town05 they are everyday traffic: red-light queues, slow city speeds, close cameras. v1 reads ordinary city traffic as an incident.

No threshold change fixes this: normal and incident scores overlap. The fix is a v2 that sees Town05 normal traffic. The detector change alone will not do it either, since these features come from geometry and traffic, not only from detection quality. The second detector run will show how much it helps.

## 5. What this means for the plan

- **v1 must not act in the town.** At 40 % of normal time in alarm, it should not even be shown in the viewer as an alarm without a "v1, not valid on Town05" label.
- **Stage 1 is enough to start v2 design.** Its first requirement is Town05 normal traffic, at junctions and on the ring road, in quantity. The false alarms are the main problem, and they come from normal traffic.
- **Ideas for v2:**
  - features relative to each camera's own normal level;
  - the stopped-track evidence agreed as an interface addition: a vehicle stopped in a moving lane while its light is green;
  - your proposed `emergency_stopped_count`.

  All of these need held-out Town05 runs to be measured fairly.

## 6. What we need from you

1. **`carla-yolo26n-t05.pt`** (5.5 MB), copied to us, to run the second detector. The delivery folder has only the clips. Once it arrives, the run is one command:
   ```
   set GOFLOW_YOLO_WEIGHTS=<local copy of carla-yolo26n-t05.pt>
   python agents/incident_detection/evaluation/run_stage1.py <stage1 folder> --detector agents/incident_detection/config/detector_carla_yolo26n_t05.json
   ```
   Alternatively, run it yourselves on the shared PC with the same command. In that case, leave `GOFLOW_YOLO_WEIGHTS` unset; the config already points at the weights.
2. **Placing the results.** The 8 result files and the summary are in `agents/incident_detection/evaluation/results/stage1/` (on the `feature/incident-carla-integration` branch once pushed; we can also send them directly). Please copy them to `out\goflow_exchange\results\`. We have no access to the shared PC from here.
3. **Confirm the scoring rule.** The "in view" interval runs from the on-screen crash second to the wrecks leaving the picture, plus the 2.2 s lag. Should the queue that remains after the wrecks are cleared also count as incident time, or as normal time?
4. **For v2, longer normal recordings.** About 30 minutes per camera of ordinary traffic would help, with red-light queues at junctions and dense ring-road traffic, from runs not used for testing. Please say which runs in `out\incident_clips\` may be used for training, and which stay held out.
5. **The close city crash** you are already recording, in a new dated folder.

## 7. Message for your Claude session

```markdown
GO-FLOW stage-1 results (attached: STAGE1_RESULTS_v1.md), branch feature/incident-carla-integration
of 14-12/GO-FLOW.
- v1 was run unchanged on all 8 stage1_20261030 clips with COCO YOLO11n. It catches 5/5 accidents
  but is in alarm 40 % of normal time (ring2 normal: 69 %; ring3: alarm for the whole minute before
  crash #3 showed); 37 false alarm episodes per camera-hour. v1 must not act in the town.
- Cause: feature shift. Town05 normal traffic has stopped_ratio 0.15-0.84 (AI City normal 0.04),
  low speeds, large boxes and confident detections, which v1 learned as incident signs.
- Added: config/detector_carla_yolo26n_t05.json (agreed class map), evaluation/run_stage1.py
  (one JSON per clip and detector, v1 fields plus run, window_start_time_s, window_end_time_s).
Please: (1) send carla-yolo26n-t05.pt, or run the command in section 6 of the attached file on the
shared PC; (2) copy evaluation/results/stage1/ into out\goflow_exchange\results\; (3) answer the
scoring question in section 6; (4) say which runs in out\incident_clips\ may be used to train v2
and which stay held out, and plan ~30 min of normal traffic per camera.
Do not edit GO-FLOW files.
```

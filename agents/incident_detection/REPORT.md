# GO-FLOW Accident Agent: full report (7 Nov 2026)

This report covers the accident agent's work with the Urban Traffic CARLA simulator (Town05) from 24 Oct to 7 Nov 2026. Each stage has its own detailed report, linked below.

## 1. Summary

- **The agent's job:** detect **accidents only**. Jams, red-light queues and congestion, even congestion caused by a crash out of view, must never raise an alarm.
- **v1 (current, unchanged)** catches every crash, but it is in alarm 40–60% of normal time on Town05, with 19–40 false alarms per camera-hour. It cannot be used in the town.
- **v2 (on the 1–2 Nov data)** passed 2 of the 4 criteria on a one-time final test, and failed on speed:

  | Result | Value |
  |---|---|
  | Crashes caught | 10 of 10 |
  | False alarms per camera-hour | 0.60 |
  | Median time to detect | 70 s |
  | Slowest detection | 140 s |

  It mostly reacted to police lights, because the simulator's police arrived 15–45 s after a crash.
- **The simulator was fixed (3–6 Nov):**
  - police are sent after about 4 minutes;
  - the crash is drawn at real contact;
  - the cameras move smoothly;
  - frame times are exact;
  - option A makes crashes likelier in front of the cameras.
- **On the new data (6–7 Nov), no model meets the criteria even on the training runs.** About half the crashes show a sharp impact the detector sees within 5 s. The other half show nothing until minutes later. **The final test was not run, and the new test runs are still unopened.**
- **Next step (team decision):** teach the CARLA detector a separate "hazard lights" class, then retrain on the same recordings and run the one-time final test.

Nothing acts in the town until the acceptance criteria are met.

## 2. Acceptance criteria (agreed before testing)

| Criterion | Needed |
|---|---|
| False alarms on normal traffic (jams included) | 1 or fewer per camera-hour |
| Clearly visible crashes caught | 80% or more |
| Median time to detect | 30 s or less |
| Slowest detection | 120 s or less |

- Cameras seen and unseen in training are reported apart.
- Far crashes are reported, but not counted either way.
- A "counted" crash is within 35 m, its impact is on screen within 10 s, and it happened after the run's first minute.

## 3. Timeline and results

| Date | Data | What was done | Result | Details |
|---|---|---|---|---|
| 24–30 Oct | AI City videos | v1: YOLO11n + ByteTrack, 27 window features, Random Forest, 3-window mean ≥ 0.35 | Baseline | `README.md` |
| 30 Oct–1 Nov | Simulator stage 1 + full recordings of 30–31 Oct | v1 unchanged on Town05 | 10/10 crashes, 31 false alarms/h, 39% of normal time in alarm | `STAGE1_RESULTS_v1.md`, `FINAL_REPORT_AND_REQUEST.md` |
| 31 Oct–1 Nov | Same | Three v2 attempts based on "stopped vehicles" | Failed: jams look the same | `FINAL_REPORT_AND_REQUEST.md` |
| 1–2 Nov | **v3 data:** 12 runs recorded on this PC, 10 cameras, 5 fps (22 train / 10 test crashes) | v2 model with crash, track and emergency-light evidence; one final test | Not accepted: 10/10, 0.60 false alarms/h, median 70 s, slowest 140 s | `V3_FINAL_TEST_RESULTS.md` |
| 3–6 Nov | Simulator fixes + option A (Firas) | Pulled; recorder and training updated | Ready | Simulator `docs/WORKLOG.md` |
| 6–7 Nov | **v4 data:** 12 runs, 4 cameras, 10 fps, option A (19 train / 9 test crashes) | Training only | No model passes on training; test not opened | `V4_TRAINING_RESULTS.md` |

### v3 final test, once, on held-out runs

| | v1 | v2 |
|---|---|---|
| Crashes caught | 10/10 | 10/10 |
| Median / slowest time to detect | 5 s / 15 s | 70 s / 140 s |
| False alarms per camera-hour | 18.81 | 0.60 |
| Normal time in alarm | 59.4% | 0.56% |

v1 is "fast" only because it is in alarm most of the time. All 12 of v2's false alarms had stopped emergency lights in view.

### v4 training runs (out-of-fold by run, at most 1 false alarm per camera-hour)

| Approach | Caught | Within 120 s | Median |
|---|---|---|---|
| Best model (v1 features + evidence) | 18/19 | 2 | 290 s |
| Impact signal alone (hard stop with contact) | 9/19 | 9 | 5 s |
| Impact signal OR model | 16/19 | 9 | 57 s |

## 4. What we learned

1. **Jams and accidents look alike to "stopped vehicle" features.** Accident-specific signals are needed: the impact, cars steering around a wreck, and hazard or emergency lights.
2. **A model takes whatever shortcut the simulator offers.** With police on scene within 45 s, v2 became a police-light detector. With realistic police, that shortcut is gone.
3. **The impact is the only fast signal, and the detector sees it for only about half the crashes.** For the rest, the first visible signs come minutes later.
4. **The detector probably labels hazard lights as police or ambulance lights.** Many v4 crashes show "emergency lights" 0–35 s after impact, long before the police come (they are sent after about 4 minutes). This is not yet checked on the video. If hazard lights get their own class, they could become a fast and clean crash signal.
5. **Crash yield:** option A gave about 2.5 counted crashes per 50-minute run in CARLA. The SUMO-only estimate was about 6.

## 5. Limitations

- **Simulator only:** the models are trained on Town05 with the CARLA detector and are not expected to work on real video.
- **Few crashes:**
  - v3: 22 train / 10 test;
  - v4: 19 train / 9 test.
  - With 10 test crashes, "10 of 10" still allows a true catch rate as low as about 69%.
- **Mostly ring-road crashes:** junction crashes are under-represented in both data sets.
- **Rule change:** in v3, the model selection rule was changed after seeing training results, before the test. This is disclosed in `V3_FINAL_TEST_RESULTS.md`.
- **v3 test ran twice:** the first run crashed while saving, before any score was shown.
- **Busier normal time in v4:** option A adds queues around the extra crashes, so v4's normal traffic is busier than real traffic.

## 6. How to reproduce

From the GO-FLOW root, with `.venv-integration`. Recording runs from the simulator folder with its own `.venv`.

```
# record a plan (resumable)
.venv\Scripts\python.exe <GO-FLOW>\agents\incident_detection\evaluation\record_v3_runs.py --plan v4_split.json
# run the detector (GOFLOW_YOLO_WEIGHTS = the CARLA model)
python agents/incident_detection/evaluation/extract_town05.py <incident_clips> --detector agents/incident_detection/config/detector_carla_yolo26n_t05.json --device 0 --tracks --out <dir> --list <order.txt>
# train on training runs only / one-time final test
python agents/incident_detection/evaluation/train_v3.py --data v4
python agents/incident_detection/evaluation/train_v3.py --data v4 --final
```

`--final` refuses to run a second time. The fixed splits are in `evaluation/v3_split.json` and `evaluation/v4_split.json`. The results are in `evaluation/results/`.

**Not in git:**
- the recordings and track files (too large);
- the v2 model file (it did not pass);
- the training notebooks.

## 7. Decision needed

1. **Hazard-lights class in the CARLA detector (recommended).** Then retrain on the v4 recordings and run the one-time final test on the unopened v4 test runs.
2. **Or accept two alarm levels:** a fast "impact seen" alarm and a slow "wreck standing" alarm, with new agreed time limits. This must be agreed before any new test.

## Files

| File | Content |
|---|---|
| `REPORT.md` | This summary |
| `V4_TRAINING_RESULTS.md` | v4 data and training results |
| `V3_FINAL_TEST_RESULTS.md` | v3 final test |
| `FINAL_REPORT_AND_REQUEST.md` | Results on the 30–31 Oct data and the data request |
| `STAGE1_RESULTS_v1.md` | v1 on the stage-1 clips |
| `evaluation/` | Scripts, splits and result JSONs |
| `tests/test_incident_v3.py` | Tests for the v3/v4 scoring and features |

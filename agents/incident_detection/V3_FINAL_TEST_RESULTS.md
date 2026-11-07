# Accident agent v2: final test results (2 Nov 2026)

## Result: not accepted

The new model catches every test crash and has few false alarms, but it is too slow. It failed 2 of the 4 criteria, so nothing should act in the town based on it.

| Criterion | Needed | Result | Pass? |
|---|---|---|---|
| False alarms per camera-hour (normal time, jams included) | 1 or less | 0.60 (12 in 19.93 h) | Yes |
| Clearly visible crashes caught | 80% or more | 100% (10 of 10) | Yes |
| Median time to detect | 30 s or less | 70 s | **No** |
| Slowest detection | 120 s or less | 140 s (2 crashes above 120 s) | **No** |

## Old model and new model on the same test runs


| | v1 (current) | v2 (new) |
|---|---|---|
| Crashes caught | 10 of 10 | 10 of 10 |
| Median time to detect | 5 s | 70 s |
| Slowest detection | 15 s | 140 s |
| False alarms per camera-hour | 18.81 | 0.60 |
| Normal time spent in alarm | 59.4% | 0.56% |

v1 looks fast only because it is in alarm most of the time, so it does not really tell crashes from normal traffic. v1 was not changed.

## Seen and unseen cameras

| Cameras | Normal hours | False alarms per hour | Crashes caught |
|---|---|---|---|
| Seen in training (7) | 13.33 | 0.82 | 10 of 10 |
| Never seen (3) | 6.60 | 0.15 | No crashes happened in view |

**We do not know if v2 catches crashes on new cameras.** The low false alarm rate there does not answer that.

## Each test crash

| Run | Camera | Time to detect |
|---|---|---|
| 20261101_224226 | 2014-3 looking E | 35 s |
| 20261101_224226 | 2014-2 looking W | 50 s |
| 20261102_033846 | 2014-2 looking W | 60 s |
| 20261102_014958 | 2014-3 looking E | 65 s |
| 20261102_003143 | 2014-2 looking W | 70 s |
| 20261102_033846 | 2328-2 looking W | 70 s |
| 20261102_014958 | 2328-2 looking W | 80 s |
| 20261102_014958 | 2014-2 looking W | 95 s |
| 20261101_224226 | 2328-2 looking W | 125 s |
| 20261102_033846 | 838-2 looking N | 140 s |

The slowest crash, on 838-2, is the only one on a camera that had no crashes in training.

## Why it is slow: it mostly reacts to police lights

- Stopped emergency lights appeared in 91% of test crash windows and 6.4% of normal windows.
- In training, these lights usually appeared 15–45 s after the impact. The crash moment itself gives a very weak signal.
- All 12 false alarms had stopped emergency lights in view and no hard stops. I did not watch those clips, so I do not know why the vehicles had stopped.

So v2 is close to a "police car stopped here" detector. This has two risks:

- It would probably miss or delay crashes when no police arrive.
- Police in the simulator arrive unrealistically fast, so real delays could be longer.

## Limits of this test

- **Few crashes:** only 10. With 10 of 10 caught, the true catch rate could still be as low as about 69% (95% range).
- **Few camera places:** 9 of the 10 test crashes and all 23 training crashes are on the ring-road cameras (2014-2, 2014-3, 2328-1, 2328-2). Junction crashes are not really tested.
- **Simulator only:** the model is trained on Town05 with the CARLA detector and is not expected to work on real video.
- **Training data:**
  - 41 recordings were used. To save time, only 26 of the 66 training recordings without a crash were processed.
  - One training run was cut short at 268 s, cause unknown. It is included.
  - Video is 5 fps, not 10, and all runs start in the daytime.
  - Some crashes have only about 70 s of normal traffic before them.
- **Rule change:** after seeing training results, but before the test, I changed the model selection rule to count catches within 120 s. This helped choose v2, so it is disclosed here.
- **Test run twice:** the first run crashed while saving, before any score was printed or saved. After fixing the save bug, I ran it again with the same settings. Nothing was chosen or tuned from the test.

The test runs have now been used, so any new model needs new test runs.

## What is needed next

The team needs to decide one of these:

1. **Keep the 30 s and 120 s limits.** v2 stays unused. The agent needs a stronger signal from the crash moment, then a new test on new runs. This is my recommendation.
2. **Accept alerts after about 1–2.5 minutes.** This would need a new agreed limit and a new test, because this test cannot be used to pass v2 under changed rules. It also accepts the dependence on police lights.

## Files

All paths are relative to this folder.

- Results: `evaluation/results/v3_final_test.json`
- Model, marked as not passed: `models/incident_detector_v2.joblib`
- Training and test script: `evaluation/train_v3.py`

The full test suite passes (204 tests). The v1 model file is unchanged. Nothing was committed or pushed.

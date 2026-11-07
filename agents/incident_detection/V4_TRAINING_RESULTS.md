# Accident agent: results on the new simulator data (7 Nov 2026)

## Result: no model passes, even on training data, so the final test was not run

The new recordings use the fixed simulator: police come after about 4 minutes, the impact is at real contact, the cameras move smoothly, and option A puts more crashes in front of the cameras. On these recordings, no model I tried meets the criteria on the training runs.

Running the one-time final test would only confirm a failure and use up the test runs. **The 5 test runs are still unopened**, so they can test a better model later.

## Data recorded

- 12 runs of 50 minutes, 4 cameras each, 10 fps, recorded 6–7 Nov on this PC.
- The plan follows the simulator's `tools/record_stage2.bat`. The split was fixed before recording.
- Runs 7–12 were added before they were recorded, because runs 1–6 gave too few crashes. That decision used only the simulator's crash log.

| | Runs | Crashes counted (≤35 m, impact on screen) | Normal time |
|---|---|---|---|
| Training | 7 | 19 | 13.45 camera-hours |
| Test, unopened | 5 | 9 (4 on cameras never used before) | not looked at |

The crash yield was about 2.5 counted crashes per 50-minute run. The SUMO-only estimate was about 6.

## Training results

These scores are out-of-fold, grouped by run, with at most 1 false alarm per camera-hour.

| Approach | Crashes caught | Caught within 120 s | Median time to detect |
|---|---|---|---|
| Best model (v1 features + evidence) | 18 of 19 | 2 | 290 s |
| Impact signal alone (hard stop with contact) | 9 of 19 | 9 | 5 s |
| Impact signal OR model | 16 of 19 | 9 | 57 s |
| Needed | 80% (16 of 19) | all | 30 s |

I also tried giving more weight to the first minute of each crash, and training only on the first minute. Neither helped. The models learn mostly from the long time a wreck stands in view, so they are slow.

## What the data shows

- **Half the crashes have a clear impact signal, and it is fast.** For 9 of 19 crashes, the detector saw a hard stop with contact in the first 5 seconds. This happens only about once per camera-hour in normal traffic.
- **The other half show no impact the detector can see.** Their first signs come minutes later: lights, cars steering around the wreck, the police.
- **The detector probably reports hazard lights as police or ambulance lights.** Many crashes show "emergency lights" 0–35 s after impact, but the police are sent after about 4 minutes. Hazard lights come on 2 s after a hit. I have not confirmed this on the video.

## Ideas for the next step (team decision)

1. **Detector (Firas/Hela):** add a separate "hazard lights" class to the CARLA detector. If hazard lights are detected reliably, they would be a fast and clean crash signal. This is my main recommendation.
2. **Look at the 10 crashes with no impact signal** to see why the detector misses them: cars too small, hidden, or a gentle hit.
3. **Accept two kinds of alarm:** a fast "impact seen" alarm (about half of crashes, within seconds) and a slow "wreck standing" alarm (most crashes, within minutes). This would need new agreed criteria.

## Files

- Recording plan and split: `agents/incident_detection/evaluation/v4_split.json`
- Crashes per run: `evaluation/results/v4_recording_progress.json`
- Training results: `evaluation/results/v4_train_*.json` and `v4_train_explore.json`
- Commands: `train_v3.py --data v4 [--final]`, `record_v3_runs.py --plan v4_split.json`

v1 is unchanged and all 204 tests pass. Nothing was committed or pushed.

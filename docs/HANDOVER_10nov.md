# Handover: Urban Traffic CARLA (10 Nov 2026)

Read this first, then `docs/HANDOVER_8nov.md` (rules, the forecasters) and `docs/HANDOVER_2nov.md` (the base project),
`README.md`, and the end of `docs/WORKLOG.md` (sections of 9 and 10 Nov: every number below comes from there).

## 1. Working rules (unchanged, keep them)

- Commits only when the user asks; author and committer the user, no mention of Claude anywhere:
  `git -c core.autocrlf=true -c user.name=firas191 -c user.email=mohamedfiras.benkhalifa@esprit.tn commit ...`;
  check `git show --stat` before saying it is done. The user pushes himself. Git needs delete permission in the
  folder (its lock files): ask once.
- Work on the PC folder `D:\Projects\Urban Traffic CARLA` through the computer link. When work is done in the cloud:
  check the PC file is the version you started from (md5), write it, check the md5 of every file written.
- Agents decide only from what a real system would have; SUMO's truth only grades them.
- Every claim measured; train seeds 101-130, dev 11-20; a test is opened once, on seeds never used. **Test seeds
  used: 21-30, 31-40, 41-50, 51-60 (forecasters, agents A/B), 61-70 (route agent), 71-80 (GO-FLOW priority agent
  test, 9 Nov). 81-90 are reserved for the forecasters' v7 test. Anything else new: 91 and above.**
- Short, plain answers; no AI-sounding wording; the user decides content.
- In the cloud a session's background jobs die when the container restarts (it happened on 10 Nov): long jobs go to
  the PC (`.bat`), or are checked often.

## 2. State of the PC folder (nothing committed since `c90f93e`)

`git status` shows many changes, all deployed and md5-checked, none committed: the GO-FLOW priority agent merge
(9 Nov), the audit fixes (9-10 Nov), `tools\goflow_v7_build.bat` and this file. **Commit them (the user asked,
10 Nov), never push:**
* Yosr's 5 commits should stay in history: fetch `integration/priority-agent-town05` (`8e12ee8`), then make the
  commit a merge with the working tree as its content, e.g. `git add -A` (not `out/`, not `Claude outputs/`),
  `tree=$(git write-tree)`, `git commit-tree $tree -p HEAD -p 8e12ee8 -m "..."` with the user's identity, then
  `git update-ref` / `git reset --soft` to it; check `git show --stat` and `git log --graph -3`.
* or several commits if the user prefers (priority agent / audit / demand / v7).

## 3. What was done (9-10 Nov), in short

| What | Result (dev seeds 11-20, SUMO alone, a day at clock 30 from 07:00) |
|---|---|
| GO-FLOW priority agent (Yosr) | merged, kept **off by default**: failed its test (seeds 71-80, 1 row of 8); its hook `request_source` and guard kept |
| `WalkSkip` bug: a skipped walk skipped every car green after it in 12 of 15 coordinated programs | fixed (`sumo_side.WalkSkip.after_walk`); coordinated 934 -> 283 s/km waiting |
| Viewer default controller | now `agents_observed` (was `coordinated`, worse on 8-9 of 10 seeds); HUD strip says who runs the signals and the traffic level |
| Cars taken off the road | counted in every run (`removed`); reroute before removing; parked cars never; junction crashes reroute drivers round the wreck: 27 -> 18 a run |
| Demand | trips at the day's rate, nobody taken off to follow the curve (`live.Population`, flow); the town's own traffic is **0.45** of the old curve (`daily.DEMAND_BASE`, the user's choice): what Town05 carries at the rush |
| Signal design (step 4) | permissive / longer or shorter greens tried: nothing better than today's |
| Spillback (step 5) | max-pressure worse than the agents; the agents' `spill` rule worth a little; no change |
| Lane changes (step 6) | real lane changes already ~4 s; the fixes tried for drifts cost traffic: not kept. **Needs a clip from the user** |
| Forecast launcher | normal traffic by default, `busy` = x1.15 (x2 would now lock the town) |
| Dataset tool | `--trips purpose` (default; v6 was trained on random trips) |

With the junction agents now: about 10-11 km/h over the day, 260 s waiting a km, 64 % of cars standing at any moment,
18 cars a run taken off as a last resort (behind accidents, knots).

## 4. Left to do, in order

1. **Forecasters v7.** `tools\goflow_v7_build.bat` (the user starts it) makes the data (4 datasets x 40 seeds, then
   the test data on seeds 81-90) and trains into `out\goflow_prediction\v7\models` (dso4, goflow_xgb, dso4_goflow,
   congestion). Log: `tools\logs\goflow_v7_build.log` (lines `EXIT 0`, `ALL DONE`). Then:
   - dev scores: v7 against v6 (`models\goflow`) on the new town: `goflow_prediction.evaluate --model ... --dataset
     out\goflow_prediction\v7\D7_all.npz --seeds 11-20 --out ...` per model, and the classifier's dev scores (in the log);
   - write the bar in the WORKLOG **before** the test (proposal: queue-fill MAE below "stays as now" at every horizon
     under the agents; below v6 on most rows; long-queue alerts at 0.7 right at least 40 % and catching at least
     half under x1.15 and x1.3);
   - test once: `tools\goflow_prediction_test_score.py --seeds 81-90 --models out\goflow_prediction\v7\models
     --reason "v7" --out out\goflow_prediction\test4_v7 out\goflow_prediction\v7\T7_*` (needs the models folder to
     hold dso4, dso4_goflow, goflow_xgb), and v6 on the same runs for comparison;
   - if it passes: copy into `models\goflow` (model cards, `models\goflow\model_card.json`, `.gitattributes` keeps
     them byte for byte), update `agents\goflow_prediction\README.md`, WORKLOG; tests; check in the viewer with
     `town05_freefly_forecast.bat` (and `busy`).
   - the agents with the forecasts (`agents_observed+forecast`) can be measured again afterwards
     (`tools\eval_agents_forecast.py`, new seeds 91+).
2. **Siren detection (step 8).** The H window counts one picture at a time (no tracking, no vote): add the tracking
   and voting `cosim\camera_reports.py` already has; a model that sees the lights flash over several frames; per-class
   results (police / police with lights / ambulance / ambulance with lights); judged by how much emergency vehicles
   are delayed, not by the detector's score. The junction agents' pre-emption uses the vehicle's own request (as real
   systems mostly do); the camera is a confirmation.
3. **Lane changes (step 6).** Find the glitch yourself: run the viewer in CARLA on the PC (computer use: start
   `town05_freefly.bat`, watch lane changes near queues, accidents and bus stops, take screenshots or record), or
   replay SUMO's lateral motion through the viewer's pose code (`tools\check_overlaps.py`, `out\audit_10nov\lat2.py`);
   name the cause, fix it, and measure it without slowing traffic (dev seeds, paired).
4. **Remaining cars taken off** (~18 a run): cars behind a bus standing at its stop cannot pass (no sideways move while
   standing); junction-crash knots at the rush.
5. **Not yet seen in CARLA** (SUMO-only so far): the new default controller and HUD strip, the 0.45 traffic, the
   rerouting round junction crashes, the forecast launcher. One look with `town05_freefly.bat`.
6. The forecasters' data and the old evaluations were made with cars taken off: older numbers in the WORKLOG are
   not comparable with the new ones.

## 5. Where things are

| What | Where |
|---|---|
| Audit scripts (probe a run, classify standing cars, lateral motion, removals, tables) | `out\audit_10nov\` (`probe_run.py`, `devtab2.py`, `devrun.sh`, `lat2.py`, `remdiag.py`, `ovcrab.py`, `README.txt`) |
| v7 build | `tools\goflow_v7_build.bat` -> `out\goflow_prediction\v7\` |
| Priority agent | `agents\priority_agent\`, `cosim\goflow_priority.py`, off by default |

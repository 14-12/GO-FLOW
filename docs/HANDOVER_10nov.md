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

## 6. Night of 10 Nov: done, and what is left (details and numbers in docs/WORKLOG.md)

* Committed (never pushed): the merge with Yosr's 5 commits, then one commit per step below.
* Forecasters v7: built, dev and test (seeds 81-90, once) passed the bar; shipped in `models/goflow`. Test seeds now
  used: 21-90. Anything new: 91 and above.
* Fixed on the way: a "crash" between two cars put on the road overlapping (follower standing) broke an incident
  every 2 s; the v7 bat left the dev seeds out (comma in a batch argument; dataset tool `--resume`).
* Lane changes (step 6): the glitch was the viewer's body heading (a chord of the front's path: the rear slid
  sideways in turns, lane changes and drifts). Now a tractrix (`interp.Trails`), measured with
  `tools/check_lateral.py`: sliding 57-61 -> 12-14 s per vehicle-hour. Left: sideways moves while standing that the
  town allows on purpose (squeezes past wrecks, buses into their stop).
* Siren (step 8): the H window votes over each vehicle's track. Left: a model that sees the lights flash needs clips
  recorded in CARLA; and the user's decision on what the camera should do, since the agents' pre-emption does not
  read it (so it cannot change emergency-vehicle delay as wired).
* Cars taken off (step 4): looked at one by one; buses held in knots, not at stops; the two seeds that lock start
  with an accident on road -19. Buses off kerb lanes tried, not kept. Next: the -19 knot.
* Not yet seen in CARLA: the forecast launcher with v7 (computer use timed out at night; it needs approving again).

## 7. Morning of 10 Nov

* Seen in CARLA: the forecast launcher with v7, normal and `busy` (HUD strip "traffic x1.15", panel "DSO4 ok ·
  GO-FLOW ok", long-queue alerts); the HUD strip says "junction agents".
* Siren: `cosim/siren_flash.py` + `models/siren_flash.json` decide SIREN ON in the H window from the light bar's
  flashing (clips: `tools/record_ev_clips.bat`, `out/ev_clips`); on the test junctions' clips, lights-off vehicles
  called on 24 % -> 6 %. Still the user's decision: a role for the camera in pre-emption.
* Cars taken off: holding kerb departures at a standing queue tried on 30 dev runs, not kept; runs too chaotic for a
  small change to show (0-88 cars a run).
* A "Job runner" window (`Claude outputs\job_runner.bat`, approved by the user) runs the .bat files put in
  `Claude outputs\jobs`; close it to stop it.

## 8. Afternoon of 10 Nov

* Camera back-up for lost emergency requests (the user's choice): `agents_observed+camback`. A siren the cameras
  confirm with no request heard at that junction gets the approach's green, with a request's limits. Dev seeds 11-20:
  with 10 % of requests lost, ambulance time loss 71.8 -> 53.9 s and waiting 267 -> 188 s/km (7/10 seeds); with none
  lost no direction at 07:00, 12:00 or 17:00. On by default (the user's decision); `agents_observed-camback` without it.
* The simulated cameras call sirens with the flash model's measured rates (`camera_reports.FLASH_ON / FLASH_OFF`,
  `@siren_flash=0` for the old vote); the counts are unchanged draw for draw.
* Getting past accidents: drivers already move over from the crash on (`_merge`, `_squeeze`, a lane left open at
  30 km/h); nothing added. A zipper merge could help the 7 % stopped right behind a wreck.
* Cars taken off: half come from accidents (drivers held RELIEVE_REMOVE_S, then taken off, by design), a quarter
  from red-light queues that did not clear in 5 min. The -19 closure was not tried (`_divert` already sends drivers
  around). Fewer removals needs a design choice: traffic let past an accident.
* The H window ran 2 h with the flash model, no emergency vehicle passed the cameras watched. At the evening peak
  (491 vehicles, 485 pedestrians) the viewer's SUMO thread fell below real time (12 steps in 10 s) and the picture
  stood still; the clock's x30 is not the cause (it only sets the time of day).
* The viewer's SUMO thread at the evening peak: busy 81 % -> 45 % (the simulated cameras read SUMO's subscriptions,
  the ground of standing vehicles kept; the same trips, checked); the "'half'" warning gone (`quantize=16`).

## 9. Evening of 10 Nov

* Every car's trip time estimated when it sets off (the user's request; buses and trucks left out):
  `cosim/trip_estimates.py`, the follow panel's ITS TRIP rows (driving / done in, estimated at start, time left /
  off by). Dev seeds 11-20: median error 27 s on trips of 112 s (median), 41 % within 20 %, 80 % within 60 s, no bias.
  Kept: the old table (`models/route_guidance/movement_times.json`) with each road's median; the table from the
  viewer's own traffic (`models/trip_estimates/movement_times.json`, `route_delay_table.py --road-times live`) is
  a little worse on dev and kept for later.
* Fixed on the way: a trip opened as a car pulled out of its space took that space as its destination (43 % of the
  trips to a place estimated 0 s); a shared car park named the wrong place on the panel; the panel said "0 m left"
  after a detour (the way left is now SUMO's route).
* keep_driving: new destinations only in view (710 -> 652 in 15 min at the evening peak; vanished in view 3 -> 1).
* The ring road carrying the town's trips: tried, not kept. Still the user's decision: ring cars with a planned trip
  along the ring (so they have an estimate too), or none.
* The user asked why a followed car estimated 2:51 drove over 4 min: a detour and the demo's staged breakdown on its
  way. Offered: a second, live "time left" beside the start estimate (the user's decision).
* Job runner: a job's `set` stays for the next jobs (the runner calls them in one window): start each job with
  `setlocal`.

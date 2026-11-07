# Integrating the GO-FLOW forecasters into Urban Traffic CARLA: the complete guide

This guide is for the project owner, who integrates the work on the PC that has CARLA. It lists everything to do,
in order, from cloning the branch to merging it into `main`.

| | |
| --- | --- |
| **Branch** | `integration/goflow-prediction-town05` (GO-FLOW), from `main` at a301ff6 |
| **What it adds** | DSO4's congestion forecaster (Rayan Zikoukout) and GO-FLOW's traffic prediction agent, running in the viewer behind an option that is **off by default** |
| **Already done and checked** | the code; the tests; the end-to-end check with SUMO; the build script, run in a clean copy of the branch |
| **Left for you** | building the models on your PC; **seeing it in CARLA**; reviewing; merging |
| **Time** | about 1 h 30: 1 h of unattended building, 20 min of checks in CARLA, then the review |

Background, if you want it:
* `docs/GOFLOW_PREDICTION_HANDOFF.md`: a summary of what was checked;
* `agents/goflow_prediction/README.md` and `agents/goflow_traffic/README.md`: each agent, and what changed from its
  PEMS08 version;
* the shared doc "DSO4 → Town05 integration plan": the full reasoning.

---

## Contents

0. [In one minute: what you will see](#0-in-one-minute-what-you-will-see)
1. [Get the branch](#1-get-the-branch)
2. [Install (once)](#2-install-once)
3. [Build the models (once, about 1 h, CARLA closed)](#3-build-the-models-once-about-1-h-carla-closed)
4. [Run the tests](#4-run-the-tests)
5. [Fly with the forecasters](#5-fly-with-the-forecasters)
6. [What to check in CARLA (the checklist)](#6-what-to-check-in-carla-the-checklist)
7. [Measure the frame rate, on and off](#7-measure-the-frame-rate-on-and-off)
8. [Troubleshooting](#8-troubleshooting)
9. [Switching it off or removing it](#9-switching-it-off-or-removing-it)
10. [Review: what changed in the project](#10-review-what-changed-in-the-project)
11. [Merging into main](#11-merging-into-main)
12. [Reference: options, files, messages, logs](#12-reference-options-files-messages-logs)
13. [Results so far, and known limits](#13-results-so-far-and-known-limits)
14. [What comes next](#14-what-comes-next)
15. [Decisions we need from you](#15-decisions-we-need-from-you)

---

## 0. In one minute: what you will see

* With `--goflow-prediction`, every 30 simulated seconds the viewer sends the state of the town's **55 approaches**
  (one per traffic camera) to DSO4's forecaster.
  * It forecasts each approach's queue at **+60, +120 and +300 s**.
  * When an approach that still flows is forecast to fill up, a pill appears over its junction:
    * "QUEUE FILLING UP IN 120 S" (`spillback_soon`);
    * or "QUEUE SPILLING BACK IN 120 S" (`propagation`), when the approach after it is already full.
* With `--goflow-traffic` as well, GO-FLOW's agent also runs.
  * Its +120 s occupancy forecast and its traffic state go into DSO4's input.
  * DSO4 then raises an alert only if GO-FLOW also says the approach is congested or severe.
* **A status line, bottom right:** `FORECAST · DSO4 ok · GO-FLOW ok · last minute: N alerts`. You may also see
  `warming up` (the first 6 simulated minutes), `restarting` or `OFFLINE`.
* **The agents run in their own processes**, with their own Python environments, on the CPU only. The viewer never
  waits for them: they cost the SUMO thread about 0.1 ms per step.
* **Expect few alerts.** At the viewer's normal traffic, queues rarely fill up (section 13). The status line and the
  forecasts are there all the time; the pills are rare.

---

## 1. Get the branch

```bat
cd "Urban Traffic CARLA"
git fetch origin
git switch integration/goflow-prediction-town05
```

If your clone uses another remote name, put it instead of `origin`. Use `git switch -c ... origin/...` if the branch
is not known locally yet.

---

## 2. Install (once)

Your usual setup stays as it is:
* `tools\setup_venv.bat` for the project's `.venv`;
* CARLA in `carla\CARLA_0.9.16\` (`tools\download_carla.bat`);
* `tools\setup_ml.bat` if you use the detector.

**Nothing of the agents goes into the project's `.venv`.** Each agent has its own environment. Double-click:

| Script | Creates | Installs | Then runs |
| --- | --- | --- | --- |
| `tools\setup_goflow_prediction.bat` | `agents\goflow_prediction\.venv` (Python 3.11) | numpy, pandas, lightgbm, pytest | its tests: 18 should pass |
| `tools\setup_goflow_traffic.bat` | `agents\goflow_traffic\.venv` (Python 3.11) | numpy 2.1.3, pandas 3.0.0, xgboost 3.2.0, scikit-learn, pytest (GO-FLOW's own versions) | its tests: 15 should pass |

* Logs: `tools\logs\setup_goflow_prediction.log`, `tools\logs\setup_goflow_traffic.log`.
* Each needs `py -3.11`, as `setup_venv.bat` does.
* If pip times out (it happened once on a slow connection), run the script again.

---

## 3. Build the models (once, about 1 h, CARLA closed)

The trained models are **not in git**: they live in `out\`, which git ignores. Build them on your PC with one
double-click:

```bat
tools\goflow_prediction_build.bat
```

It runs SUMO only (CARLA can stay closed) and does six steps. A step whose output already exists is skipped:
nothing is ever overwritten.

| Step | What | Output (under `out\`) | Time |
| --- | --- | --- | --- |
| 1 | The 55 approaches and the road graph between them | `goflow_prediction\graph\` (`nodes.json`, `distance.csv`) | seconds |
| 2 | The dataset: 60 runs × 30 simulated min (+5 warm-up), the viewer's traffic (day curve, buses, ring road), coordinated signals, clock at 60×, accidents off, one row every 30 s | `goflow_prediction\datasets\D2_x1_coord_v1\` | about 25 min with 4 jobs |
| 3 | DSO4's model (LightGBM, one per horizon); train seeds 101-140, dev seeds 11-20 | `goflow_prediction\models\lgbm_real_v2\` | a few min |
| 4 | GO-FLOW's model (XGBoost, +120 s) | `goflow_traffic\models\xgb_120s_v1\` | a few min |
| 5 | GO-FLOW's forecasts on the whole dataset (out-of-fold on the train seeds), then DSO4 + GO-FLOW | `...\dataset_goflow.npz`, `goflow_prediction\models\lgbm_gf_v2\` | 10-15 min |
| 6 | The live check (section 4) | `goflow_prediction\live_check_<time>\report.json` | about 5 min |

* At the end it prints every `EXIT` code (all must be `0`) and the `[ok]` lines.
* The full log is `tools\logs\goflow_prediction_build.log`.
* **For a quick try first** (about 10 min), with smaller data that is not for real use:

  ```bat
  set GFP_SEEDS=101-104,11-12& set GFP_TRAIN=101-104& set GFP_DEV=11-12& set GFP_TAG=try
  tools\goflow_prediction_build.bat
  ```

  Steps 3 to 5 then write the model folders with this small data. Delete `out\goflow_prediction\models` and
  `out\goflow_traffic\models` before the full build, or keep the quick ones for a first look.
* **Other settings:** `GFP_JOBS` (default 4). The test seeds 21-30 are simulated but never used for training or
  tuning.

The three model folders together take about 10 MB.

---

## 4. Run the tests

| What | Command | Expected |
| --- | --- | --- |
| The project's suite | `tools\test.bat` | **235 passed, 2 failed.** The 2 failures (`tests/test_redlight.py`, plate reader) need `cv2`, which only `setup_ml.bat` installs; they fail the same way on `main` (228 passed there). With `setup_ml.bat` done, all should pass. |
| The connector alone | `.venv\Scripts\python -m pytest tests\test_goflow_prediction.py` | 7 passed, about 6 s |
| DSO4's agent | `cd agents\goflow_prediction` then `.venv\Scripts\python -m pytest tests` | 18 passed |
| GO-FLOW's agent | `cd agents\goflow_traffic` then `.venv\Scripts\python -m pytest tests` | 15 passed |
| End to end with SUMO (no CARLA) | `.venv\Scripts\python tools\goflow_prediction_live_check.py --goflow` | see below |

The live check runs the town twice (seed 11, 12 min by default; the build script uses 20 min with the kill at
minute 10). It plugs in the real connector and both agents, exactly as the viewer does. In `report.json`:
* **"normal":**
  * `replay.identical` equals `replay.compared` (the forecasts written live equal an offline replay of the same
    messages);
  * `obs_dropped` is 0;
  * `connector_ms_per_step.median` is well under 1 ms;
  * both children are `ready`.
* **"kill":** DSO4's process is killed mid-run.
  * `restarted.after_wall_s` is a few seconds;
  * forecasts continue after the kill;
  * `dso4_failures` is 1.

Measured here: replay 30/30, 0 dropped, 0.09 ms median, restart in 6.4-6.7 s (`live_check_20261107_130926`, and the
build script's run in a clean copy of the branch).

---

## 5. Fly with the forecasters

```bat
set FREEFLY_ARGS=--goflow-prediction --goflow-traffic
town05_freefly.bat
```

* DSO4 alone (no GO-FLOW): `set FREEFLY_ARGS=--goflow-prediction`.
* You can add your usual options, e.g. `--set-time 17:30`. The forecasters work with any `--controller`. They read
  SUMO's lanes, not the camera reports.
* **The first 6 simulated minutes the status line says `warming up`:** DSO4 needs 12 steps of 30 s and GO-FLOW 13.
  At the default clock speed that is 6 minutes on the viewer's clock, which runs with the town.
* **F6 / F7 (clock ± 1 h)** resets the agents' windows (their history would mix two hours), so `warming up` comes
  back for 6 minutes. That is expected.
* **Every message of the run** is written to `out\goflow_prediction\live_<date>_<time>\`: `obs.jsonl` (what DSO4
  received), `forecasts.jsonl`, `alerts.jsonl`, `goflow.jsonl`.
* **Models from other folders:** `--goflow-prediction-model DIR` (repeatable; the default is `lgbm_real_v2`, plus
  `lgbm_gf_v2` with `--goflow-traffic`) and `--goflow-traffic-model DIR`.

---

## 6. What to check in CARLA (the checklist)

Only these points are **not** checked yet. Please tick them, and note anything odd (a screenshot helps).

- [ ] **Off is unchanged.** `town05_freefly.bat` with no `FREEFLY_ARGS`: the viewer looks and runs as on `main`. No
      new line on the overlay, and no Python process for the agents in Task Manager.
- [ ] **It starts.**
  * With `--goflow-prediction --goflow-traffic`, the loading screen and the first view are as usual.
  * `tools\logs\freefly_town05.log` has a line `goflow forecast: [...] -> ...\live_...`.
  * Task Manager shows more `python.exe` processes: the agents. A venv's `python.exe` starts the real one, so each
    agent may show as two.
- [ ] **The status line**, bottom right, reads `FORECAST · DSO4 warming up · GO-FLOW ok · last minute: 0 alerts`,
      then `DSO4 ok · GO-FLOW ok` after about 6 simulated minutes.
  * Does it sit well, without covering anything?
  * Is it readable over a bright sky and at night?
- [ ] **The alert pills.** When `alerts.jsonl` in the run's `live_...` folder gets a line, fly to that junction (the
      line has `tls`, the junction id; F8 cycles through the junctions).
  * Within 60 simulated seconds of the alert, a pill should be over the junction: about 14 m up, above the junction
    agents' pills when `--agent-hud` is on.
  * Check that it is placed over the right junction, that its text is readable, and that it disappears after 60 s.
  * Alerts are rare at normal traffic (section 13). To see one sooner, try the evening rush (`--set-time 17:00`) or
    more incidents (`--careless-share 0.3`). It may still take a while.
- [ ] **The frame rate** with the options on and off (section 7).
- [ ] **An agent killed.**
  * In Task Manager, end the `python.exe` whose command line has `goflow_prediction.serve` (add the "Command line"
    column).
  * The viewer must go on.
  * The status line shows `DSO4 restarting`, then `DSO4 ok` within about 10 s, and forecasts continue.
  * Killing it 4 times leaves it `OFFLINE` for the rest of the run, by design.
- [ ] **F6 / F7.** The clock jumps and the status line goes back to `warming up`; nothing else changes.
- [ ] **Esc.**
  * The viewer closes as usual.
  * The agents' processes end with it (none left in Task Manager after a few seconds).
  * `tools\logs\goflow_prediction.log` ends without a traceback.

---

## 7. Measure the frame rate, on and off

The scripted 70 s flight writes the frame rate and smoothness to `out\town05_freefly_demo.json`. It keeps your
`FREEFLY_ARGS` and adds its own options.

```bat
set FREEFLY_ARGS=
tools\town05_freefly_demo.bat
copy out\town05_freefly_demo.json out\town05_freefly_demo_off.json

set FREEFLY_ARGS=--goflow-prediction --goflow-traffic
tools\town05_freefly_demo.bat
copy out\town05_freefly_demo.json out\town05_freefly_demo_on.json
```

* Compare the two files. Expect no measurable difference: the connector costs about 0.1 ms per 500 ms SUMO step,
  and the agents run in other processes on the CPU.
* Note that 70 s of flight is shorter than the 6-minute warm-up, so this measures the cost of running, not of the
  pills.

---

## 8. Troubleshooting

| What you see | Why | What to do |
| --- | --- | --- |
| Status `DSO4 OFFLINE` from the start | Its venv or its model is missing | `tools\logs\freefly_town05.log` says which: "no Python at ..." means run `tools\setup_goflow_prediction.bat`; a "cannot load model" error in `tools\logs\goflow_prediction.log` means run `tools\goflow_prediction_build.bat` |
| Status `GO-FLOW OFFLINE` | The same for GO-FLOW | `tools\setup_goflow_traffic.bat`, then step 4 of the build script; log `tools\logs\goflow_traffic.log` |
| No status line at all with the option on | The connector did not start | `freefly_town05.log`: "goflow forecast not started" with the traceback; most often `out\goflow_prediction\graph\nodes.json` is missing (build step 1) |
| `warming up` for good | No observations reach DSO4 | The SUMO thread runs (cars move)? Look for errors in `goflow_prediction.log`. `out\goflow_prediction\live_...\obs.jsonl` should get a line every 30 simulated seconds with `"delivered":true` |
| `DSO4 restarting` over and over, then `OFFLINE` | The agent crashes at once | The end of `tools\logs\goflow_prediction.log` has its traceback |
| Many observations dropped (a warning in `freefly_town05.log`) | An agent is slower than the town (it should take ~10 ms per step) | Is the CPU saturated? Is the GPU in use by the agents (it must not be)? They run with `CUDA_VISIBLE_DEVICES=""` |
| Never an alert | Normal at this traffic (section 13) | Check that `forecasts.jsonl` grows. Alerts need a queue forecast to exceed that approach's own Q99 of the training data |
| `setup_goflow_*.bat`: "No Python 3.11" / pip timeout | | Install Python 3.11; run the script again |
| Build step 2 slow | 60 runs | `GFP_JOBS=6` on more cores; it is SUMO only |

---

## 9. Switching it off or removing it

* **Off:** it is off unless `--goflow-prediction` is given. Clear `FREEFLY_ARGS` (`set FREEFLY_ARGS=`).
* **Removing it from a run without restarting:** not possible. End the agents' processes and the viewer goes on;
  their status shows `OFFLINE`.
* **Removing it from the project:**
  * revert the 53 lines in `cosim/freefly.py`;
  * delete `agents/`, `cosim/goflow_prediction.py`, `cosim/goflow_prediction_hud.py`, `tools/goflow_prediction_*`,
    `tools/setup_goflow_*.bat`, `tests/test_goflow_prediction.py`;
  * remove the two README rows.

  Nothing else depends on them.

---

## 10. Review: what changed in the project

**Existing files** (all changes are inactive with the options off):

| File | Change |
| --- | --- |
| `cosim/freefly.py` (+53 / −3) | 1. Four options in `parse_args` (`--goflow-prediction`, `--goflow-prediction-model`, `--goflow-traffic`, `--goflow-traffic-model`). 2. `self.forecast = None` in `FreeFly.__init__`. 3. `self._wire_forecast(a)` after `self._wire_agents(a)` in `_load`; `_wire_forecast` starts the connector, and on any failure logs it and goes on. 4. In `produce()`, right after `cur = self.sumo.step()`, one call `self.forecast.on_step(cur, self.sumo.conn, self.clock.seconds_of_day)`. 5. In `markers_draw`, the forecast HUD's paint. 6. `ff.forecast.reset(...)` on F6 / F7, and `self.forecast.close()` in `FreeFly.close()`. |
| `README.md` | One row in the viewer options table, one in "Other things you can run" |

Why `produce()` and not `SumoSide.pre_policy`: `pre_policy` runs only when a per-step controller exists, and under
the default `coordinated` controller `controllers.make()` returns `None`, so it would never be called.

**New files:**

| Path | What |
| --- | --- |
| `agents/goflow_prediction/` | DSO4's forecaster: package `goflow_prediction` (data, features, LightGBM training, agent, evaluation, JSON-lines server), tests, README, requirements |
| `agents/goflow_traffic/` | GO-FLOW's agent on Town05: her rules unchanged, her feature recipe, her XGBoost settings, training, out-of-fold export, server; tests, README (with the decisions made), requirements |
| `cosim/goflow_prediction.py` | The connector: `Aggregator` (30 s means per approach from lane subscriptions), `Child` (an agent process: start, JSON lines, restart after 5 / 20 / 60 s, then off), `GoflowPrediction` (queue of 4, never blocks; GO-FLOW first, at most 1 s, then DSO4; after a crash the last 12 steps, missed ones included, are sent again) |
| `cosim/goflow_prediction_hud.py` | The pills and the status line (reuses `agent_hud.AgentHud`'s drawing) |
| `tools/goflow_prediction_graph.py` | The approaches and the graph |
| `tools/goflow_prediction_dataset.py` | The SUMO-only datasets (the viewer's traffic) |
| `tools/goflow_prediction_live_check.py` | The end-to-end check without CARLA |
| `tools/goflow_prediction_build.bat` | Everything of section 3 in one double-click |
| `tools/setup_goflow_prediction.bat`, `tools/setup_goflow_traffic.bat` | The agents' venvs |
| `tests/test_goflow_prediction.py` | The connector's tests, in the project's `.venv` with none of the agents' packages |
| `docs/GOFLOW_INTEGRATION_GUIDE.md` (this file), `docs/GOFLOW_PREDICTION_HANDOFF.md` | |

**What to look at in the review:**
* the `freefly.py` diff (`git diff main -- cosim/freefly.py`);
* `cosim/goflow_prediction.py`: threads, the queue, restarts;
* that nothing heavy is in git: the largest new file is 20 KB, and the models are built, not committed.

Project rules followed:
* separate venvs;
* off by default;
* nothing slow on the SUMO thread;
* results never overwritten (every tool refuses an existing output folder);
* no secrets;
* no large files;
* `fly_proc.py` and `DefaultInput.ini` untouched;
* the test seeds 21-30 never used.

---

## 11. Merging into main

1. Run sections 3 to 7 and tick the checklist (section 6).
2. **A problem?** Tell Rayan what you saw (a screenshot, the logs) and the fix comes on this branch. Or fix it there
   yourself: it is a normal branch.
3. **A pull request** from `integration/goflow-prediction-town05` into `main`, which you review. Never push to
   `main` directly (the project's rule).

   GitHub's link: `https://github.com/14-12/GO-FLOW/pull/new/integration/goflow-prediction-town05`.
4. **After the merge:**
   * on every PC that should run it: section 2 (the two setup scripts) and section 3 (the build script, or copy the
     three model folders, about 10 MB);
   * the shared plan doc and `docs/WORKLOG.md` can get a dated entry; the numbers to quote are in section 13 with
     their runs.

---

## 12. Reference: options, files, messages, logs

**Viewer options**

| Option | Default | What it does |
| --- | --- | --- |
| `--goflow-prediction` / `--no-goflow-prediction` | off | DSO4's forecaster, its alerts on the overlay |
| `--goflow-prediction-model DIR` | `out\goflow_prediction\models\lgbm_real_v2` (+ `lgbm_gf_v2` with `--goflow-traffic`) | Its model folder(s); repeat the option for two |
| `--goflow-traffic` / `--no-goflow-traffic` | off | GO-FLOW's agent too, read by DSO4 |
| `--goflow-traffic-model DIR` | `out\goflow_traffic\models\xgb_120s_v1` | GO-FLOW's model folder |

**Messages** (JSON lines on the agents' stdin and stdout, version 1; times in SUMO seconds)

| From → to | Type | Main fields |
| --- | --- | --- |
| town → agents | `hello` | `nodes` (55 camera names, in order), `dt_s` 30, `source` "exact", `clock_speed` |
| town → agents | `obs` | `seq`, `t`, `tod_s`, `demand`, `demand_at` {"60", "120", "300"}, `nodes` {name: `veh`, `halt`, `speed` m/s, `occ` (fraction), `green` (share of its links green)}; to DSO4 also `goflow` {name: `pred_occ`, `traffic_state`, `risk_level`} or `null` |
| town → agents | `reset` | `t`, `reason` (after F6 / F7) |
| GO-FLOW → town | `goflow` | `seq`, `t`, `horizon_s` 120, `nodes` {name: `pred_occ`, `traffic_state`, `risk_level`, or `null` while its 13 steps are not complete} |
| DSO4 → town | `forecast` | `seq`, `t`, `chosen` {horizon: the model its self-evaluation picked}, `nodes` {name: `fill` [3], `halt` [3]}, `cpu_ms` |
| DSO4 → town | `alert` | `t`, `kind` (`spillback_soon` / `propagation`), `node`, `tls`, `horizon_s`, `fill_now`, `pred_fill`, `source`, `model`, `action`, `goflow_pred_occ`, `confirmed_by_goflow` |
| any → town | `ready`, `error` | `error` has `msg`; an agent never stops on a bad message |

`fill` = halting vehicles / storage of the approach (lane length / 7.5 m), as the junction agents' `_fill`.

**Logs and outputs**

| Path | What |
| --- | --- |
| `tools\logs\freefly_town05.log` | the viewer; the connector's start or its failure |
| `tools\logs\goflow_prediction.log`, `goflow_traffic.log` | each agent's stderr: start lines, errors, tracebacks |
| `tools\logs\goflow_prediction_build.log` | the build script |
| `out\goflow_prediction\live_<time>\` | every message of a viewer run |
| `out\goflow_prediction\live_check_<time>\report.json` | the end-to-end check |
| `out\goflow_prediction\models\*\model_card.json`, `thresholds.json`, `dev_metrics.csv` | what each model is, its alert thresholds, its dev scores |

---

## 13. Results so far, and known limits

All of these are on the **dev seeds** (11-20) of `D2_x1_coord_v1`. The **test seeds (21-30) are locked** and have not
been opened.

| Queue fill MAE | +60 s | +120 s | +300 s |
| --- | --- | --- | --- |
| **DSO4 LightGBM** | **0.0136** | **0.0182** | **0.0222** |
| Last value | 0.0177 | 0.0261 | 0.0397 |
| Value one signal cycle ago (115 s) | 0.0266 | 0.0261 | 0.0427 |
| Average by demand | 0.0223 | 0.0219 | 0.0228 |

* **GO-FLOW XGBoost at +120 s:** occupancy MAE 0.0151, against 0.0201 for the last value.
* **DSO4 + GO-FLOW:** the same forecast errors as DSO4 alone (within 0.0001).
* **Real graph, random graph, no graph:** within 0.0002. At this traffic there is little propagation between
  junctions for the graph to carry.

### Alerting on the jam and crash runs

The repo currently contains the committed nominal-traffic evaluation and the crash/jam dataset generation, while the dense jam/crash alert table is still being finalized on the incident-heavy runs.

| Dataset / run set | Scale | Accidents | Careless share | Seeds | Model | Alerts | Precision | Onsets caught | Median lead time |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `D2_x1_coord_v1` (current committed dev eval) | 1.0x | off | n/a | 11-20 | DSO4 alone | 0 | 0.0 | 0/0 | n/a |
| `D2_x1_coord_v1` (current committed dev eval) | 1.0x | off | n/a | 11-20 | DSO4 + GO-FLOW | 0 | 0.0 | 0/0 | n/a |
| `D3_x1_acc_v1` + `DJ_x4_acc_v1` (crash/jam set) | 1.0x / 4.0x | on | default 0.10 (higher values in queued runs) | 101-140 + dev seeds | DSO4 alone | pending in repo | pending | pending | pending |
| `D3_x1_acc_v1` + `DJ_x4_acc_v1` (crash/jam set) | 1.0x / 4.0x | on | default 0.10 (higher values in queued runs) | 101-140 + dev seeds | DSO4 + GO-FLOW | pending in repo | pending | pending | pending |

The key point is that the nominal viewer traffic is not the situation that produces meaningful alerts: the alerting signal only appears when queues are forced by higher demand or accident-induced spillback. The crash/jam dataset generation is present in the branch, but the final alert summary for those runs still needs to be committed alongside the shipped model bundle before the review checklist is fully closed.

**Known limits:**
1. **Not seen in CARLA yet** (section 6).
2. **The alerts are not validated on the crash/jam runs yet.** The normal traffic case produces almost no queue-fill events, which is why the useful alerting signal only appears under higher-demand or accident-heavy conditions.
   * In the current committed dev eval, `D2_x1_coord_v1`, the alert count is 0 for both DSO4 alone and DSO4 + GO-FLOW.
   * The crash/jam run set is already generated (`D3_x1_acc_v1`, `DJ_x4_acc_v1`), but the final alert summary is still pending in the repo.
3. **Exact counts only.** The agents read SUMO's lanes, not the camera reports: GO-FLOW needs speed and occupancy,
   which the reports do not carry.
4. **GO-FLOW's "flow"** is the number of vehicles on the approach, since the data has no loop counts per step.
   GO-FLOW agreed to the decisions in `agents/goflow_traffic/README.md`.
5. **Trained on SUMO-only runs.** They have the viewer's traffic parts (day curve, buses, ring road, the incident
   manager's `unjam`), but without the viewer's "remove only out of sight". The live check matches what the agents
   receive live exactly, but the viewer's own traffic may differ slightly from the training runs.

---

## 14. What comes next

1. **Your checks in CARLA** (section 6), then the merge (section 11).
2. **Measure the alerts on traffic that really congests:**
   * the eval harness's trip demand (`tools/eval_town05_baseline.py`, x0.8 / x1.0, `--ignore-junction-blocker 60`,
     as `agents_final_jb`);
   * add a dataset mode for it in `tools/goflow_prediction_dataset.py`;
   * then precision, onsets caught and lead time on the dev seeds; GO-FLOW's confirmation compared with DSO4 alone.
3. **Open the test seeds 21-30 once**, with everything frozen:
   `python -m goflow_prediction.evaluate ... --seeds 21-30 --open-test "<reason>"`. A lock file prevents a second
   opening.
4. **Optional, later:**
   * camera-report inputs for DSO4;
   * incident inputs (closed lanes, never severity), with the `integration/incident-to-traffic-prediction` branch;
   * an agents feature `+forecast` (longer greens before a forecast queue), evaluated with the project's protocol.

---

## 15. Decisions we need from you

1. **Models:** ship the three folders (about 10 MB) with the project, or rebuild on each PC with
   `tools\goflow_prediction_build.bat`?
2. **Alert evaluation:** on the eval harness's congested demand (section 14.2)? Or should the viewer's traffic
   itself congest more?
3. **Overlay:** keep the pills on with `--goflow-prediction`, or behind a separate option, as the agents' HUD
   (`--agent-hud`)?
4. **Merge:** after your checks pass, may it go into `main` as it is, off by default?

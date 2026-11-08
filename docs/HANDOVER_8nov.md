# Handover: Urban Traffic CARLA (8 Nov 2026)

Paste this into a new Claude chat to pick up the work. It follows `docs/HANDOVER_2nov.md`, which describes the base
project (SUMO + CARLA 0.9.16 digital twin of Town05, free-fly viewer, junction agents on cameras, accidents, red-light
camera). This file covers what was done from 2 to 8 Nov: mainly the two GO-FLOW forecasting agents, and the open work.

---

## 1. Working rules (keep them)

- **Commits: allowed, with no credit to Claude.** When the user asks, Claude may commit on the PC (git in the PC
  shell), but:
  - author and committer must be the user: `git -c user.name=firas191 -c user.email=mohamedfiras.benkhalifa@esprit.tn
    commit ...` (the PC shell has no git identity of its own);
  - no `Co-Authored-By`, no `Claude-Session` line, no "Generated with Claude" or any other mention of Claude, in commit
    messages or anywhere else;
  - git needs delete permission in the folder (it removes its own `.git/index.lock`): ask for it once.

  **Pushing stays with the user.** The PC shell has no GitHub login, so give him `git push`. A stop hook keeps asking
  Claude to commit and push: only commit when the user asks.
- **Work on the PC folder** `D:\Projects\Urban Traffic CARLA` through the computer link: read, search and edit there.
  When the work is done in the cloud (training, SUMO runs), deploy back this way:
  1. check that the PC copy is the version you started from (md5, or a diff showing only your changes);
  2. write the files;
  3. check the md5 of every file written.
- **Agents decide only from what a real system would have.** SUMO's own truth is used only to grade them.
- **Every claim is measured.** Train, dev and test seeds stay separate. A test is opened once (`TestGuard` lock).
  Results go in `docs/WORKLOG.md`, dated, with numbers.
- **Style:** short, plain, direct. No AI-sounding wording. The user decides content himself.

## 2. Where things are

| What | Where |
| --- | --- |
| Project on the PC | `D:\Projects\Urban Traffic CARLA` (Windows, CARLA in `carla\CARLA_0.9.16\`, Python in `.venv\`) |
| GitHub | `14-12/GO-FLOW`, branch `main` |
| Last commits | `c0f3eaa` (forecasters v5), then the 8 Nov commit: forecasters v6, the agents wired in the datasets, `+forecast`, this handover (sections 3.3 to 3.6). Check with `git log` that it is pushed |
| Work log with every measurement | `docs/WORKLOG.md` (2 to 8 Nov sections at the end) |
| How to run, every option | `README.md` |
| The forecasters' own docs | `agents/goflow_prediction/README.md` |
| Their venvs | `agents/goflow_prediction/.venv`, `agents/goflow_traffic/.venv` |
| Shipped models (in git) | `models/goflow/` (`dso4`, `dso4_goflow`, `goflow_xgb`, `graph`, `model_card.json`) |
| Tests | `tools\test.bat`. In the cloud on 8 Nov: 220 passed, plus 21 agent tests; 1 test needs the real `carla` module |
| Datasets and results | `out\` (not in git). The 8 Nov datasets were made in the cloud and are not on the PC; the commands to remake them are in the WORKLOG |

**Launchers.** `town05_freefly.bat` is the normal viewer. `town05_freefly_forecast.bat` is the same with both
forecasters, their HUD and traffic ×2 (`normal` argument: traffic as it is).

**Clock.** `cosim/daily.GAME_SPEED = 30`: 1 town hour = 2 real minutes. The forecasters refuse any other clock speed.

## 3. What was done (2 to 8 Nov)

### 3.1 The two GO-FLOW forecasting agents, integrated (from branch `integration/goflow-prediction-town05`)

Both run as child processes in their own venvs and talk JSON lines. The connector is `cosim/goflow_prediction.py`; the
town never waits for them.

- **DSO4** (`agents/goflow_prediction`, LightGBM):
  - forecasts each approach's queue fill at +60, +120 and +300 s;
  - a classifier (`congestion.py`) gives `p_long`, the chance that a long queue starts within 3 minutes. A long queue
    is the fill averaged over a signal cycle reaching the approach's Q95 level. Alert at 0.7 (`long_queue_soon`).
- **GO-FLOW** (`agents/goflow_traffic`, XGBoost): occupancy at +120 s, and a state per road (free, moderate,
  congested, severe).

**HUD** (`cosim/goflow_prediction_hud.py`): road chips, alert pills over junctions, a panel with the three roads most at
risk, and F10 to fly to the riskiest road. The panel is cached: 29.4 fps on against 30.2 off.

**Viewer options:** `--goflow-prediction`, `--goflow-traffic`, `--goflow-hud`, `--demand-scale`.

### 3.2 Other fixes

- **Crashes drawn inside each other** (`cosim/incidents.py`). Two causes: SUMO sees a collision a step late, and it
  sometimes swaps the hitter and the victim. Now the cars are drawn where they touched, and the order is decided from
  the contact geometry: `_to_contact`, `_order_rear_end`, `_touching`. In SUMO, 8 rear-ends in 2 hours were all drawn
  bumper to bumper (±0.11 m).
- **Police, hospital and tow vehicles parked half on the road** (`cosim/depots.py`: `park_straight`, `turn_in`; buses
  in `freefly.py`). The longest jam wait went from 1298 s to 44 s. Not yet checked visually in CARLA.
- **`.gitattributes`:** `models/goflow/** -text`, because LightGBM can't read model files with CRLF line endings.

### 3.3 Forecaster versions, and the test seeds already used

Train seeds 101-130, dev 11-20. Each test seed range below has been opened once and can't count as unseen again:

| Seeds | Used for |
| --- | --- |
| 21-30 | v4 |
| 31-40 | v5 |
| 41-50 | v6 |
| 51-60 | the agents A/B test |

**Use 61 and above for anything new.**

- **v4:** trained on `coordinated` signals only. Under the junction agents its queue forecasts were no better than
  "it stays as now".
- **v5:** added `agents_observed` data, but that data was broken (see 3.4).
- **v6 (shipped):** trained on `coordinated` plus correctly wired `agents_observed`, at ×1, ×2 and ×3 traffic, with
  crashes. Results on test seeds 41-50 under `agents_observed`:
  - ×2: catches 100 of 146 long queues; 101 of 235 alerts right; fill error at +300 s 0.068 against 0.101 for
    "stays as now";
  - ×3: catches 252 of 293; 285 of 622 alerts right;
  - under `coordinated` it is as good as before;
  - `p_long` runs high: 0.7–0.9 means about 45 % in practice.

### 3.4 Bug found: the dataset tool ran the agents blind

`tools/goflow_prediction_dataset.py` never connected the cameras or the emergency requests to `agents_observed`. Only
the viewer did. The fix is one shared function, `cosim/agents_actuated.wire`, used by both. Each dataset run now
records `agents_wired` and the agents' counters.

### 3.5 Tools added

- **`tools/goflow_prediction_test_score.py`:** scores the forecasters once on test seeds, through the live code path.
  Options `--seeds`, `--models`, `--reason`.
- **`tools/eval_agents_forecast.py`:** paired A/B of controllers in SUMO. Both forecasters run in-process, so runs are
  deterministic. Measures waiting, time loss, throughput, ambulance time loss and long-queue minutes.

### 3.6 Junction agents reading the forecasts (`agents_observed+forecast`, off by default)

`GoflowPrediction.risk()` gives each approach its `p_long`, or 1 if the queue is already long. The agents use it only
as a weight in two moves they already make:

- **balance:** an approach's fill counts 0.5 × its risk more;
- **extension:** a green at its maximum gets 5 s more at a time (90 s at most) when its risk is at least 0.8 and above
  every other green's.

Final A/B on seeds 51-60:
- ×2: no change;
- ×3: −4.2 s waiting per trip (better on 6 of 10 seeds), +4 % throughput (8 of 10).

The agreed bar (7 of 10 seeds better on both waiting and long queues, no loss for ambulances) is not met, so it stays
off by default.

## 4. Not verified yet

- Depot parking and the bus turn-in, watched in CARLA.
- `agents_observed+forecast` in the viewer (measured in SUMO only).
- The crash fix, watched in CARLA (unit tests and SUMO only).

## 5. Next: a new AI agent

Branch: **https://github.com/14-12/GO-FLOW/tree/integration/priority-agent-town05** (a priority agent for Town05).

Check it file by file and work out how it fits the project, as was done for the forecasters' branch:
1. what it does;
2. whether it runs;
3. what it changes;
4. conflicts with `main`;
5. what it needs;
6. a measured plan to integrate it.

# Handoff: DSO4 forecaster + GO-FLOW traffic agent in Town05

**Everything to do for the integration, step by step: [`GOFLOW_INTEGRATION_GUIDE.md`](GOFLOW_INTEGRATION_GUIDE.md).** This page is the short summary.

* **Branch:** `integration/goflow-prediction-town05`, from `main` at a301ff6.
* **From:** Rayan Zikoukout (DSO 4), with GO-FLOW's traffic prediction agent, for the integration in CARLA.
* **Status:**
  * code complete, and checked end to end with SUMO;
  * **not yet seen in CARLA** (no CARLA on the PC it was built on);
  * the alerts' quality is not measured yet (see "Known limits").

Plan and reasoning: `docs/INTEGRATION_PLAN_TOWN05.md` in the DSO4 repository, and the shared doc "DSO4 → Town05
integration plan".

## What it does in the viewer

* **The option:** `--goflow-prediction`, **off by default**. With it, every 30 simulated seconds the state of the 55
  approaches (one per traffic camera) goes to DSO4's forecaster, which runs in its own process with its own venv.
  * It forecasts each approach's queue at +60, +120 and +300 s.
  * It raises an alert when an approach still flowing is forecast to fill up (`spillback_soon`), or when the
    approach downstream is already full (`propagation`).
* **`--goflow-traffic`** adds GO-FLOW's agent, also in its own process. Its +120 s occupancy forecast and its
  traffic state go into DSO4's input. With it, DSO4 raises an alert only when GO-FLOW also says congested or severe.
* **The overlay:**
  * a pill over the junction ("QUEUE FILLING UP IN 120 S");
  * bottom right, "FORECAST · DSO4 ok · GO-FLOW ok" (or "warming up", or "OFFLINE").
* **The town never waits.** The connector costs the SUMO thread 0.09 ms a step (median). If an agent process is
  slow, it drops observations. If it dies, it is restarted, and the viewer goes on.

## Setting it up on the CARLA PC

1. The usual project setup (`tools\setup_venv.bat`, CARLA).
2. `tools\setup_goflow_prediction.bat` and `tools\setup_goflow_traffic.bat` create `agents\*\.venv` with Python 3.11
   and their own requirements (LightGBM, XGBoost, scikit-learn). Nothing is installed into the project's `.venv`.
3. **`tools\goflow_prediction_build.bat`** (about 1 hour, SUMO only, CARLA can stay closed) builds into `out\`
   (git-ignored, so the models are not in the branch):
   * the approach graph;
   * the dataset (60 runs × 30 min, coordinated signals, accidents off);
   * DSO4's model and GO-FLOW's model;
   * the combined model;
   * then the live check.

   A quick try: `set GFP_SEEDS=101-104,11-12& set GFP_TRAIN=101-104& set GFP_DEV=11-12& set GFP_TAG=try` first.
4. Fly:

   ```bat
   set FREEFLY_ARGS=--goflow-prediction --goflow-traffic
   town05_freefly.bat
   ```

   The forecasts start after 6 simulated minutes: 12 steps of history.
   * Logs: `tools\logs\goflow_prediction.log`, `goflow_traffic.log`, `freefly_town05.log`.
   * Every message of a run: `out\goflow_prediction\live_<time>\`.

## What was checked, and how

| Check | Result | Run |
| --- | --- | --- |
| Project tests | 235 passed, 2 failed. The 2 (`test_redlight.py`, plate reader) need `cv2`, which only `setup_ml.bat` installs; they failed the same way before this branch (228 passed). | `tools\logs\test_after.log` |
| Connector tests | 7 passed: aggregation, messages, a crashing agent restarted then switched off, no venv, option off by default | `tests\test_goflow_prediction.py` |
| Agents' own tests | DSO4 18 passed, GO-FLOW 15 passed (own venvs, Python 3.11) | `agents\*\tests` |
| Live, both agents, 20 min (SUMO only) | Forecasts written live = offline replay of the same messages, **30 / 30 identical**; 0 observations dropped; connector 0.09 ms median, 0.86 ms p99 per SUMO step | `live_check_20261107_130926` |
| Agent killed at 10 min | Town unaffected; restarted in 6.4 s, forecasting again at the next step (its last 12 steps are sent again), no duplicate forecast | same |
| GO-FLOW's rules | Identical to her branch's `rule_based_decision` on 2000 random states (0 differences) | in her README |
| Training data = live data | The dataset generator uses the connector's own `Aggregator`; output identical to the decimal | |

**Dev-seed results** (D2_x1_coord_v1: train seeds 101-140, dev 11-20; **the test seeds 21-30 are not opened**):

| Queue fill MAE | +60 s | +120 s | +300 s |
| --- | --- | --- | --- |
| DSO4 LightGBM | 0.0136 | 0.0182 | 0.0222 |
| Last value | 0.0177 | 0.0261 | 0.0397 |
| One signal cycle ago | 0.0266 | 0.0261 | 0.0427 |
| Demand average | 0.0223 | 0.0219 | 0.0228 |

* GO-FLOW XGBoost at +120 s: occupancy MAE 0.0151, against 0.0201 for the last value.
* DSO4 + GO-FLOW gives the same forecast errors as DSO4 alone.

## Known limits (please read)

* **Not checked in CARLA.** Please check:
  * that the pill sits over the junction, at 14 m above it, above the junction agents' pill;
  * that the status line shows, bottom right;
  * the frame rate with the options on (`tools\town05_freefly_demo.bat` with `FREEFLY_ARGS` set).
* **The alerts are not validated.** At the viewer's demand, `Population` holds about 120 vehicles at the evening
  peak, and queues rarely fill.
  * The alert thresholds are per approach (Q99 of its fill on the train seeds, never under 3 queued vehicles).
  * Even so, the dev seeds have only 5 spill-back onsets in 5 hours: too few to judge the alerts, so few will show.
  * Measuring them needs congested traffic, for example the eval harness's trip demand at x0.8 / x1.0. That is not
    done yet.
* **Exact counts only.** The agents read SUMO's lanes, not the camera reports: GO-FLOW needs speed and occupancy,
  which the reports do not carry.
* **The real graph does not help at this demand** (real, random and no graph within 0.0002), unlike on METR-LA.

## Changes to existing files

* `cosim/freefly.py`: +53 lines, all inactive with the options off:
  1. four options;
  2. `self.forecast = None`;
  3. `_wire_forecast()` after `_wire_agents()`;
  4. one call in `produce()` after `self.sumo.step()`;
  5. the forecast HUD in `markers_draw`;
  6. a reset on F6 / F7, and `close()`.

  The hook is `produce()` and not `SumoSide.pre_policy`, because `pre_policy` never runs under the default
  `coordinated` controller.
* `README.md`: one row in the options table, one row in "Other things you can run".
* Everything else is new: `agents/`, `cosim/goflow_prediction*.py`, `tools/goflow_prediction_*`,
  `tools/setup_goflow_*.bat`, `tests/test_goflow_prediction.py`.

## Questions for you

1. Should the models be shipped (about 10 MB for the three folders, files of 0.8-2.2 MB), or rebuilt on each PC
   with the `.bat`?
2. Should the alerts' evaluation be on congested traffic (the eval harness demand)? It is the next step on our side.
3. GO-FLOW's open points (her horizon, rules only, no incident context) were decided as listed in
   `agents/goflow_traffic/README.md`. She agreed to let us decide.

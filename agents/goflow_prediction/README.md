# goflow_prediction: the DSO4 congestion forecaster on Town05

Rayan Zikoukout's DSO 4, prepared for Urban Traffic CARLA (Town05) following the integration plan
(`docs/INTEGRATION_PLAN_TOWN05.md` in the DSO4 repository, and the shared doc of the same name).

**Status (7 Nov):** trained on Town05 (crashes on, traffic x1 / x2 / x3; the shipped models are in `models/goflow/`)
and checked in CARLA; results in `docs/WORKLOG.md`. The tests run on a toy corridor (`tests/conftest.py`), a test
fixture, not Town05.

## What it does

Every 30 SUMO seconds the town sends one observation per approach (the 55 approaches are the 55 cameras). The agent:

1. **perceives:** a ring buffer of 12 steps (6 min);
2. **self-evaluates:** each candidate's error on its forecasts whose target has now been seen, over the last 12;
   the best candidate is used for each horizon;
3. **decides:** forecasts each approach's queue at +60, +120 and +300 s. The queue is `fill` = halting vehicles /
   storage (lane length / 7.5 m), the junction agents' own `_fill`;
4. **acts:** with the long-queue classifier next to its model (`congestion.json`, `congestion_model.txt`; since
   7 Nov), a `long_queue_soon` alert when an approach is likely (0.7) to start a queue that stays over its level
   within 3 minutes, and each forecast carries every approach's chance (`p_long`). Without it, the first rule: for an
   approach under its `low` now and forecast at its `full` or more, a `propagation` alert when a downstream approach
   is already full, or a `spillback_soon` alert otherwise. The cooldown is 120 s per approach.

### Long queues (7 Nov)

The first rule fired on queues refilling at the next red and caught no build-up on the dev seeds, at any traffic.
A queue sampled every 30 s swings with the signal (cycle 115 s), so `congestion.py` looks at the fill averaged over
4 steps (about one cycle): an approach's *level* is the Q95 of that mean on the train seeds (at least 3 vehicles); a
long queue *starts* when the mean reaches the level after a whole cycle under it, from under 0.6 x the level. A
LightGBM classifier (the forecast's features plus the approach against its level) gives the chance of a start within
6 steps (3 minutes); trained with `python -m goflow_prediction.congestion --dataset ... --model DIR --train-seeds ...
--dev-seeds ...`. Dev seeds 11-20 (5 h per traffic level), alert at 0.7:

| Traffic | Long queues | Caught before they start | Alerts right | Median warning |
| --- | --- | --- | --- | --- |
| x1 | 6 | 3 | 3 of 5 | 30 s |
| x2 | 40 | 27 | 27 of 50 | 60 s |
| x3 | 136 | 101 | 116 of 227 | 60 s |

The candidates are LightGBM (and LightGBM + GO-FLOW when her real forecast is in the message), last value, and the
value one signal cycle (115 s) ago.

## What changed from the METR-LA / PEMS08 version, and why

The original stays unchanged in the DSO4 repository (`src/`, `agent/`), with its results.

| Before (METR-LA / PEMS08) | Now (Town05) | Why (plan section 2) |
| --- | --- | --- |
| speed in mph | queue `fill` per approach | stopping at a red is normal; the cameras report no speed |
| 5-min steps, +15 / 30 / 60 min | 30 s steps, +60 / 120 / 300 s (`config.TimeBase`) | one SUMO minute is one town hour |
| slot of the day, weekday | scheduled demand now and at the target (`demand`, `demand_at`) | the town's hour runs 60×; the demand curve is known in advance |
| 207 / 170 sensors, sensor id | 55 approaches (camera names), id optional (`--no-node-id`) | for the held-out-junction test |
| split by date | split by SUMO seed (train 101-140, dev 11-20, test 21-30) | held-out seeds, as the project does |
| REST API (FastAPI) | JSON lines on stdin / stdout (`serve.py`, `schema.py`) | the town talks to its helper processes this way |
| package `src` | package `goflow_prediction` | `src` clashed with GO-FLOW's package |
| 50 % / 75 % of free-flow speed | fill ≥ 0.85 (the agents' `full_share`), < 0.6 "still flowing" | the town's own thresholds |
| GO-FLOW from `export_goflow.py` (PEMS08) | `goflow` field in the observations, only from her real Town05 model | no stand-in for her model |

Kept as they were: the no-leakage checks (now per seed and run), the locked test (`TestGuard`), the masked
metrics, the real / random / no-graph ablation (`graphs.py`), Graph WaveNet (`gwn.py`, not trained on Town05 yet),
the Gaussian-kernel graph, and the self-evaluation on matured forecasts only.

## Files

| File | Role |
| --- | --- |
| `config.py` | time base, thresholds, model card |
| `data.py` | dataset format (documented at the top), graph, splits by seed, windows inside one run |
| `features.py` | features for training and live use (one function for both) |
| `baselines.py` | last value, one-cycle persistence, demand-HA |
| `train_lgbm.py` | one LightGBM per horizon, early stopping on the dev seeds, model card |
| `agent.py` | the four-stage agent, and `replay()` (offline evaluation through the live code path) |
| `evaluate.py` | dev scores, or the test opened once (`--open-test`) |
| `schema.py`, `serve.py` | the JSON-lines messages and the live process |
| `leakage.py`, `metrics.py`, `graphs.py`, `gwn.py` | checks, errors and events, graph ablation, Graph WaveNet |

## Use

```bat
tools\setup_goflow_prediction.bat
agents\goflow_prediction\.venv\Scripts\python -m pytest agents\goflow_prediction\tests
agents\goflow_prediction\.venv\Scripts\python -m goflow_prediction.train_lgbm --dataset D2.npz --graph distance.csv --out out\goflow_prediction\models\v1
agents\goflow_prediction\.venv\Scripts\python -m goflow_prediction.evaluate --model out\goflow_prediction\models\v1 --dataset D2.npz --seeds 11-20 --out out\goflow_prediction\eval\v1_dev
agents\goflow_prediction\.venv\Scripts\python -m goflow_prediction.serve --model out\goflow_prediction\models\v1
```

Run the commands from `agents\goflow_prediction\`, or with that folder on `PYTHONPATH`.

## Not done yet (plan, section 7)

* The SUMO-only dataset generator (`tools/goflow_prediction_dataset.py`, S2) and the graph tool
  (`tools/goflow_prediction_graph.py`). They need SUMO, which is not installed on this PC.
* Stage S1 (the original code unchanged on Town05 data), Graph WaveNet on Town05, the adapter `cosim/goflow_prediction.py`
  and the HUD (S6).
* Every result, Town05 or toy, has to come from a named run; none is claimed here.

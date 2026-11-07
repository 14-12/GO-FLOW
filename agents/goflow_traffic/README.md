# goflow_traffic: GO-FLOW's traffic prediction agent on Town05 (proposal for its author's review)

This is the Town05 version of GO-FLOW's `TrafficPredictionControlAgent`, written from her branch
`feature/traffic-prediction-agent` (commit 392fbae) for the joint integration plan (section 9, step J3).

**It is her work and her decision.** Her PEMS08 code, model, contract and results stay on her branch, untouched.
This package is a proposal: nothing it outputs stands for her agent until she has reviewed it and trained it on
Town05 data herself. There is **no Town05 result** yet. The tests run on a toy corridor, which is a test fixture.

## What is kept exactly

* **Her decision rules** (`policy.py` = her `src/decision/policy.py` `indicators` and `rule_based_decision`):
  * the same conditions, the same order, the same actions, confidence 0.5;
  * checked against her original code on 2000 random states, with 0 differences
    (`traffic_state`, `risk_level`, `recommended_action`, `confidence`).
* **Her feature recipe** (`features.py`): flow, speed and occupancy; lags 1, 2, 3, 6, 12; rolling means 3, 6, 12 of the
  values before t (`shift(1).rolling`, checked against pandas in the tests); `location` categorical; float32; 13
  observations.
* **Her model:** XGBoost with her selected trial-3 settings (n_estimators 300, max_depth 6, learning_rate 0.1,
  min_child_weight 10), predicting from the best iteration only.
* **Her thresholds:** the same quantiles (occupancy Q50 / Q90 / Q97, speed Q25, change Q90), fitted on the train
  seeds only.
* **Her checks on load:** feature order, target, horizon, best iteration. No training at prediction time.
* **Rules-only live mode:** `decision_source: rule_based_fallback`, CPU only.

## Decisions (she left them to us: "do what works best", 5 Nov 2026)

| Question | Decision | Why |
| --- | --- | --- |
| Q-J1: retrain on Town05? | **Yes**, with `train.py` | Her PEMS08 model knows only sensors 0-169 |
| Q-J2: which horizon? | **One forecast at +120 s** (`--horizon-s`, steps of 30 s) | On PEMS08 a single +15 min forecast was what improved DSO4's alerts (precision 0.471 → 0.561); 120 s is the middle of DSO4's horizons, about one signal cycle. Others only if the dev seeds show a gain. |
| Q-J3: LLM live? | **No: rules only**, CPU | It would compete with CARLA for the GPU and takes seconds; her rules are her deterministic fallback anyway |
| Q-J4: incident / emergency contexts? | **Not in v1**: they stay unknown, as in her reason text | The town must never pass its own ground truth (severity) at run time |
| Camera mode? | **Exact source only**; `train.py` and `serve.py` refuse cameras | The camera reports carry no speed and no occupancy, which her model needs |

## What changed from her PEMS08 code, and why

| Change | Why |
| --- | --- |
| `location` = Town05 approach index (0..N-1, names in the metadata) | PEMS08's 0..169 do not exist in Town05 |
| `day_index` dropped | a Town05 run is not a calendar day (the clock runs 60×) |
| `slot_in_day`, `hour`, `minute` from the town clock | the same meaning as hers, on the town's day |
| horizon 120 s, steps of 30 s | +15 min would be 15 town hours |
| `flow` = vehicles on the approach (`veh`) | the town's data has no loop counts per step; to revisit if the generator later counts crossings |
| her thresholds refitted on the Town05 train seeds | PEMS08 quantiles do not apply to a city |
| JSON-lines messages (`hello` / `obs` / `reset` in, `ready` / `goflow` / `error` out) instead of her request / response contract | the town feeds every approach every 30 s |
| no local LLM | decision Q-J3 |

Kept from her contract: values that are booleans, strings, NaN or infinite are refused (an `error` message, that
approach left unknown), never converted; an approach with an incomplete 13-step history gets no forecast.

## Use

```bat
tools\setup_goflow_traffic.bat
agents\goflow_traffic\.venv\Scripts\python -m pytest agents\goflow_traffic\tests
agents\goflow_traffic\.venv\Scripts\python -m goflow_traffic.train --dataset D2.npz --graph distance.csv --out out\goflow_traffic\models\v1
agents\goflow_traffic\.venv\Scripts\python -m goflow_traffic.serve --model out\goflow_traffic\models\v1
```

Its `goflow` messages are what the adapter passes on to goflow_prediction in each observation (plan section 9.3).

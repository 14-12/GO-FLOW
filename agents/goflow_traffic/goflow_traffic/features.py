"""Her XGBoost features (src/prediction/xgboost_predictor.py), on Town05 steps.

Kept from her recipe: flow, speed and occupancy now; lags 1, 2, 3, 6, 12; rolling means over 3, 6, 12 steps of the
values BEFORE t (shift(1).rolling); `location` categorical; float32; 13 observations needed (t-12 .. t).

Changed (README.md, "What changed and why"):
* `location` = index of the Town05 approach (0..N-1, names in the metadata), not the PEMS08 sensor 0..169;
* `day_index` is dropped: a Town05 run is not a calendar day (the clock runs 60x);
* `slot_in_day`, `hour`, `minute` come from the town clock (seconds of the day), with 5-minute slots as hers;
* `flow` = vehicles on the approach (`veh`): the town's data has no loop counts per step. A question for her.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)
ROLLING = (3, 6, 12)
HISTORY = 13                      # t-12 .. t
VARIABLES = ("flow", "speed", "occupy")
FEATURE_COLUMNS = (("location", "slot_in_day", "hour", "minute") + VARIABLES
                   + tuple(f"{v}_lag_{k}" for v in VARIABLES for k in LAGS)
                   + tuple(f"{v}_rolling_mean_{w}" for v in VARIABLES for w in ROLLING))


def build(hist: dict, tod_s: np.ndarray, n_nodes: int) -> pd.DataFrame:
    """hist: variable -> [S, 13, N] (oldest first, last = t). tod_s [S] town clock at t.
    -> one row per (sample, approach), columns FEATURE_COLUMNS, `location` categorical."""
    S, L, N = hist["occupy"].shape
    assert L == HISTORY and N == n_nodes
    slot = (np.asarray(tod_s) // 300).astype(int) % 288
    cols = {"location": np.tile(np.arange(N), S),
            "slot_in_day": np.repeat(slot, N), "hour": np.repeat(slot * 5 // 60, N),
            "minute": np.repeat(slot * 5 % 60, N)}
    for v in VARIABLES:
        x = hist[v]
        cols[v] = x[:, -1].reshape(-1)
        for k in LAGS:
            cols[f"{v}_lag_{k}"] = x[:, -1 - k].reshape(-1)
        for w in ROLLING:
            cols[f"{v}_rolling_mean_{w}"] = x[:, -1 - w:-1].mean(axis=1).reshape(-1)   # t-w .. t-1, as shift(1)
    F = pd.DataFrame(cols)[list(FEATURE_COLUMNS)]
    F["location"] = F["location"].astype(pd.CategoricalDtype(categories=list(range(N))))
    num = list(FEATURE_COLUMNS[1:])
    F[num] = F[num].astype("float32")
    return F

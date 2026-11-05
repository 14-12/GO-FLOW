"""Baselines the models must beat (plan, section 6.3). Everything learned comes from the train seeds only."""
from __future__ import annotations

import numpy as np

from .config import TimeBase


def last_value(y: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """The value at t0, repeated for every horizon. y [T, N] -> [S, N]."""
    return y[t0]


def cycle_persistence(y: np.ndarray, t0: np.ndarray, k: int, tb: TimeBase, run: np.ndarray) -> np.ndarray:
    """The value one signal cycle before the target (t0 + k - cycle), when that is observed (<= t0) and in the same
    run; else the last value. Strong under the coordinated controller, where the queue repeats every cycle."""
    c = tb.cycle_steps
    src = t0 + k - c
    while True:
        late = src > t0
        if not late.any():
            break
        src = np.where(late, src - c, src)
    ok = (src >= 0) & (run[np.clip(src, 0, None)] == run[t0])
    return np.where(ok[:, None], y[np.clip(src, 0, None)], y[t0])


class DemandHA:
    """Mean per (approach, demand bin) over the train seeds: Town05's "time-of-day average", since the town's
    hour is fast (60x) and what really sets the traffic is the scheduled demand."""

    def __init__(self, bins: int = 10):
        self.edges = np.linspace(0.0, 1.0, bins + 1)

    def _bin(self, d):
        return np.clip(np.digitize(d, self.edges[1:-1]), 0, len(self.edges) - 2)

    def fit(self, y: np.ndarray, demand: np.ndarray, rows: np.ndarray):
        b = self._bin(demand[rows])
        nb, N = len(self.edges) - 1, y.shape[1]
        s, c = np.zeros((nb, N)), np.zeros((nb, N))
        v = y[rows]
        ok = np.isfinite(v)
        np.add.at(s, b, np.where(ok, v, 0.0))
        np.add.at(c, b, ok)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.table = np.where(c > 0, s / c, np.nan)
            node_mean = np.nanmean(np.where(ok, v, np.nan), axis=0)
        self.table = np.where(np.isnan(self.table), node_mean[None, :], self.table)
        return self

    def predict(self, demand_target: np.ndarray) -> np.ndarray:
        """demand at the target time [S] -> [S, N]."""
        return self.table[self._bin(demand_target)]

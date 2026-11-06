"""Town05 datasets: loading, the approach graph, splits by seed and windows that never cross two runs.

Dataset file (`.npz`, written by tools/goflow_prediction_dataset.py, one or several SUMO runs concatenated):

    nodes      [N]      approach ids = camera names ("1427-1")
    storage    [N]      vehicles the approach can hold (lane lengths / 7.5 m, summed over its lanes)
    t          [T]      SUMO second at the end of each step
    run        [T]      run index (a window never crosses two runs)
    seed       [T]      SUMO seed of that run (the splits are by seed)
    tod_s      [T]      town clock, seconds of the day
    demand     [T]      cosim/daily.demand_factor at t
    demand_at  [T, H]   demand_factor at t + h, for the horizons in meta["horizons_s"] (a known schedule)
    halt, veh, speed, occ, green   [T, N]   means over the step; NaN = not measured (camera mode: speed, occ)
    meta       json string: dt_s, horizons_s, source, controller, clock_speed, scale, git_commit, ...
    goflow_pred_occ [T, N]  optional: GO-FLOW's occupancy forecast made at t (only from her real model)
    goflow_state    [T, N]  optional: her traffic_state at t (goflow_traffic.export)

Graph file: `distance.csv` with columns from,to,cost (road metres, i -> j = traffic goes from i to j), node ids as in
`nodes`. Same kernel as the PEMS08 loader of DSO4: w = exp(-(d / sigma)^2), sigma = std of the distances.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TimeBase

CHANNELS = ("halt", "veh", "speed", "occ", "green")


@dataclass
class Dataset:
    nodes: list[str]
    storage: np.ndarray      # [N]
    t: np.ndarray            # [T]
    run: np.ndarray          # [T]
    seed: np.ndarray         # [T]
    tod_s: np.ndarray        # [T]
    demand: np.ndarray       # [T]
    demand_at: np.ndarray    # [T, H]
    x: dict                  # channel -> [T, N]
    meta: dict
    goflow: np.ndarray | None = None   # [T, N]
    goflow_state: np.ndarray | None = None   # [T, N] her traffic_state ("" = none yet)

    @property
    def fill(self) -> np.ndarray:
        """Queued / storage, the target. NaN where halt is unknown or storage < 1 vehicle."""
        st = np.where(self.storage >= 1.0, self.storage, np.nan)
        return self.x["halt"] / st[None, :]

    @property
    def T(self) -> int:
        return len(self.t)

    @property
    def N(self) -> int:
        return len(self.nodes)

    def timebase(self) -> TimeBase:
        return TimeBase(dt_s=float(self.meta["dt_s"]), horizons_s=tuple(int(h) for h in self.meta["horizons_s"]))


def load_dataset(path) -> Dataset:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    x = {c: z[c].astype(np.float64) for c in CHANNELS}
    ds = Dataset(nodes=[str(n) for n in z["nodes"]], storage=z["storage"].astype(np.float64),
                 t=z["t"].astype(np.float64), run=z["run"].astype(np.int64), seed=z["seed"].astype(np.int64),
                 tod_s=z["tod_s"].astype(np.float64), demand=z["demand"].astype(np.float64),
                 demand_at=z["demand_at"].astype(np.float64), x=x, meta=meta,
                 goflow=z["goflow_pred_occ"].astype(np.float64) if "goflow_pred_occ" in z.files else None,
                 goflow_state=z["goflow_state"].astype(str) if "goflow_state" in z.files else None)
    check_dataset(ds)
    return ds


def save_dataset(ds: Dataset, path) -> None:
    extra = {} if ds.goflow is None else {"goflow_pred_occ": ds.goflow.astype(np.float32)}
    np.savez_compressed(path, nodes=np.array(ds.nodes), storage=ds.storage, t=ds.t, run=ds.run, seed=ds.seed,
                        tod_s=ds.tod_s, demand=ds.demand, demand_at=ds.demand_at, meta=json.dumps(ds.meta),
                        **{c: ds.x[c].astype(np.float32) for c in CHANNELS}, **extra)


def check_dataset(ds: Dataset) -> None:
    T, N = ds.T, ds.N
    assert len(set(ds.nodes)) == N, "duplicate node ids"
    assert ds.storage.shape == (N,)
    for name in ("run", "seed", "tod_s", "demand"):
        assert getattr(ds, name).shape == (T,), name
    assert ds.demand_at.shape == (T, len(ds.meta["horizons_s"])), "demand_at must have one column per horizon"
    for c in CHANNELS:
        assert ds.x[c].shape == (T, N), c
    tb = ds.timebase()
    for r in np.unique(ds.run):
        tt = ds.t[ds.run == r]
        assert np.allclose(np.diff(tt), tb.dt_s), f"run {r}: steps are not {tb.dt_s} s apart"
        assert len(np.unique(ds.seed[ds.run == r])) == 1, f"run {r} has several seeds"


def load_graph(dist_csv, nodes: list[str]) -> np.ndarray:
    """Directed Gaussian-kernel adjacency [N, N] (diagonal 1), as src/data.load_pems08 of DSO4."""
    e = pd.read_csv(dist_csv, dtype={"from": str, "to": str})
    idx = {n: i for i, n in enumerate(nodes)}
    unknown = sorted(set(e["from"]).union(e["to"]) - set(idx))
    if unknown:
        raise ValueError(f"graph names nodes not in the dataset: {unknown[:5]}")
    e = e.groupby(["from", "to"], as_index=False)["cost"].min()
    N = len(nodes)
    dist = np.full((N, N), np.inf)
    dist[e["from"].map(idx).to_numpy(), e["to"].map(idx).to_numpy()] = e["cost"].to_numpy(dtype=float)
    sigma = float(e["cost"].std()) if len(e) > 1 else 0.0
    if not np.isfinite(sigma) or sigma <= 0:   # all roads the same length: the spread is 0, use the mean
        sigma = max(float(e["cost"].mean()), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.exp(-np.square(dist / sigma))
    adj[~np.isfinite(dist)] = 0.0
    np.fill_diagonal(adj, 1.0)
    return adj.astype(np.float32)


def seed_mask(ds: Dataset, seeds) -> np.ndarray:
    return np.isin(ds.seed, np.asarray(list(seeds)))


def anchors(ds: Dataset, seeds, tb: TimeBase, stride: int = 1) -> np.ndarray:
    """t0 indices (last observed step) whose input [t0-n_in+1, t0] and every target t0+h lie in ONE run of the
    given seeds. Windows never cross two runs, nor two splits (splits are whole seeds)."""
    ok = seed_mask(ds, seeds)
    out = []
    for r in np.unique(ds.run[ok]):
        idx = np.flatnonzero(ds.run == r)
        first, last = idx[0] + tb.n_in - 1, idx[-1] - tb.max_steps
        if last >= first:
            out.append(np.arange(first, last + 1, stride))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


class MaskedScaler:
    """z-score from the train rows only, NaN ignored."""

    def fit(self, v):
        v = v[np.isfinite(v)]
        self.mean, self.std = float(v.mean()), float(v.std() or 1.0)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

"""Tabular features for LightGBM, one row = (window, approach). Built from the input window only (steps up to t0)
and from known schedules (the demand curve at the target time): nothing reads the target.

The same function serves training (windows gathered from a dataset) and live use (the agent's ring buffer), so the
features seen live are exactly those seen in training.

From DSO4's src/features.py: own lags, weighted means of the upstream and downstream neighbours (graph i -> j =
traffic from i to j), plus what Town05 needs: the downstream approaches' queues (a full exit is what makes a queue
spill back), the green share, the scheduled demand, and optionally GO-FLOW's own occupancy forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Dataset


def neighbor_mats(adj: np.ndarray):
    """adj[i, j] > 0: edge i -> j. upstream of k = {i: adj[i, k] > 0}, downstream = {j: adj[k, j] > 0}."""
    A = adj.astype(np.float64).copy()
    np.fill_diagonal(A, 0.0)
    return A, A.T.copy()   # (V @ A_up)[k] sums k's upstream, (V @ A_down)[k] its downstream


def nan_neighbor_mean(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """V [..., N] (NaN = unknown) -> weighted mean over neighbours [..., N], NaN if none known."""
    valid = np.isfinite(V).astype(np.float64)
    num, den = np.nan_to_num(V) @ A, valid @ A
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def neighbor_max(V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Max over the neighbours (A > 0) of V [S, N] -> [S, N]; NaN if none known."""
    S, N = V.shape
    out = np.full((S, N), np.nan)
    for k in range(N):
        nb = np.flatnonzero(A[:, k] > 0)
        if len(nb):
            vals = V[:, nb]
            has = np.isfinite(vals).any(1)
            out[has, k] = np.nanmax(vals[has], axis=1)
    return out


@dataclass
class Window:
    """What one or more forecasts see. Arrays [S, n_in, N] (oldest -> t0); demand [S]; demand_at [S, H]."""
    fill: np.ndarray
    veh: np.ndarray
    speed: np.ndarray
    occ: np.ndarray
    green: np.ndarray
    demand: np.ndarray
    demand_at: np.ndarray
    goflow: np.ndarray | None = None    # [S, N] GO-FLOW's occupancy forecast made at t0, or None


def gather(ds: Dataset, t0: np.ndarray, n_in: int) -> Window:
    idx = t0[:, None] + np.arange(-n_in + 1, 1)[None, :]
    st = np.where(ds.storage >= 1.0, ds.storage, np.nan)
    return Window(fill=ds.x["halt"][idx] / st, veh=ds.x["veh"][idx], speed=ds.x["speed"][idx],
                  occ=ds.x["occ"][idx], green=ds.x["green"][idx], demand=ds.demand[t0], demand_at=ds.demand_at[t0],
                  goflow=None if ds.goflow is None else ds.goflow[t0])


def build(w: Window, h_index: int, storage: np.ndarray, A_up, A_down, use_node_id: bool = True,
          use_goflow: bool = False):
    """-> (F [S*N, d] float32, names). h_index = column of the horizon in demand_at."""
    S, L, N = w.fill.shape
    cols, names = [], []

    def add(v, name):
        cols.append(np.broadcast_to(v, (S, N)).astype(np.float32))
        names.append(name)

    with np.errstate(invalid="ignore"), _quiet():
        for lag in range(L):
            add(w.fill[:, L - 1 - lag], f"fill_lag{lag}")
        add(np.nanmean(w.fill, 1), "fill_mean")
        add(w.fill[:, -1] - w.fill[:, max(0, L - 4)], "fill_rise3")
        add(w.veh[:, -1], "veh_lag0")
        add(np.nanmean(w.veh, 1), "veh_mean")
        add(w.speed[:, -1], "speed_lag0")
        add(w.occ[:, -1], "occ_lag0")
        add(w.green[:, -1], "green_lag0")
        add(np.nanmean(w.green, 1), "green_mean")
        for lag in range(3):
            add(nan_neighbor_mean(w.fill[:, L - 1 - lag], A_up), f"up_fill_lag{lag}")
            add(nan_neighbor_mean(w.fill[:, L - 1 - lag], A_down), f"down_fill_lag{lag}")
        add(nan_neighbor_mean(np.nanmean(w.fill, 1), A_up), "up_fill_mean")
        add(nan_neighbor_mean(np.nanmean(w.fill, 1), A_down), "down_fill_mean")
        add(neighbor_max(w.fill[:, -1], A_down), "down_fill_max_lag0")
        add(w.demand[:, None], "demand_now")
        add(w.demand_at[:, h_index][:, None], "demand_target")
        add((w.demand_at[:, h_index] - w.demand)[:, None], "demand_change")
        add(storage[None, :], "storage")
        if use_goflow:
            g = w.goflow if w.goflow is not None else np.full((S, N), np.nan)
            add(g, "gf_pred_occ")
            add(nan_neighbor_mean(g, A_up), "gf_pred_occ_up")
            add(nan_neighbor_mean(g, A_down), "gf_pred_occ_down")
        if use_node_id:
            add(np.arange(N)[None, :], "node")
    F = np.stack(cols, axis=-1).reshape(S * N, -1)
    return F, names


class _quiet:
    """nanmean over an all-NaN window gives NaN on purpose; keep the warning out of the logs."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)

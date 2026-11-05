"""Masked errors (NaN target = unknown, left out) per horizon, and the spill-back event metrics."""
from __future__ import annotations

import numpy as np


def masked_metrics(pred, true) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = np.isfinite(true) & np.isfinite(pred)
    if not m.any():
        return dict(MAE=np.nan, RMSE=np.nan, n=0)
    e = pred[m] - true[m]
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), n=int(m.sum()))


def horizon_table(preds: dict, trues: dict, model: str, split: str) -> list[dict]:
    """preds / trues: horizon key ('60') -> array. One row per horizon."""
    return [dict(model=model, split=split, horizon_s=int(h), **masked_metrics(preds[h], trues[h])) for h in preds]


def onsets(fill: np.ndarray, run: np.ndarray, full: float, low: float, hold: int = 2, before: int = 4) -> np.ndarray:
    """[T, N] bool: the step where an approach becomes full (fill >= full for `hold` steps) after `before` steps
    under `low`, inside one run. The plan's onset rule (DSO4's rule rescaled to Town05 steps)."""
    T, N = fill.shape
    out = np.zeros((T, N), bool)
    f = np.nan_to_num(fill, nan=-1.0)
    for t in range(before, T - hold + 1):
        if run[t - before] != run[t + hold - 1]:
            continue
        out[t] = (f[t:t + hold] >= full).all(0) & (f[t - before:t] < low).all(0) & (f[t - before:t] >= 0).all(0)
    return out


def event_metrics(alerts: list[dict], onset_mask: np.ndarray, t: np.ndarray, nodes: list[str], fill: np.ndarray,
                  full: float, dt_s: float) -> dict:
    """alerts: {t, node, horizon_s} (SUMO s). An alert is right if its approach reaches `full` within its horizon
    plus one step. An onset is caught if an alert on its approach came within the largest horizon before it."""
    idx_t = {float(v): i for i, v in enumerate(t)}
    col = {n: j for j, n in enumerate(nodes)}
    right = 0
    for a in alerts:
        i, j = idx_t.get(float(a["t"])), col[a["node"]]
        k = int(round(a["horizon_s"] / dt_s)) + 1
        seg = fill[i + 1:i + 1 + k, j] if i is not None else np.array([])
        right += bool(np.nanmax(seg, initial=-1) >= full)
    hmax = max([a["horizon_s"] for a in alerts], default=0)
    by_node: dict[int, list[float]] = {}
    for a in alerts:
        by_node.setdefault(col[a["node"]], []).append(float(a["t"]))
    leads, caught = [], 0
    ti, tj = np.nonzero(onset_mask)
    for i, j in zip(ti, tj):
        ts = [x for x in by_node.get(j, []) if t[i] - hmax - dt_s <= x < t[i]]
        if ts:
            caught += 1
            leads.append(t[i] - min(ts))
    hours = (len(t) * dt_s) / 3600.0
    n_junctions = len({n.split("-")[0] for n in nodes}) or 1
    return dict(alerts=len(alerts), precision=right / len(alerts) if alerts else np.nan,
                onsets=int(onset_mask.sum()), caught=caught,
                recall=caught / onset_mask.sum() if onset_mask.sum() else np.nan,
                median_lead_s=float(np.median(leads)) if leads else np.nan,
                false_per_junction_hour=(len(alerts) - right) / (hours * n_junctions) if hours else np.nan)

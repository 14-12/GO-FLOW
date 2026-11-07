"""Her XGBoost, retrained on Town05 (step J3 of the joint plan: HER step; this script is the proposal she reviews).

    python -m goflow_traffic.train --dataset D2.npz --graph distance.csv --out OUT \
        --train-seeds 101-140 --dev-seeds 11-20 [--horizon-s 120]

Same dataset and graph files as goflow_prediction (agents/goflow_prediction/goflow_prediction/data.py documents them).
Hyper-parameters: her selected trial 3 (results/model_benchmark/xgboost_validation_results.csv: n_estimators 300,
max_depth 6, learning_rate 0.1, min_child_weight 10), early stopping on the DEV seeds. Her thresholds are refitted on
the TRAIN seeds with her quantiles. Writes OUT/xgboost_model.json, OUT/xgboost_export_metadata.json (features,
target, horizon, nodes, neighbours, best_iteration, thresholds, seeds, git commit) and OUT/dev_metrics.csv. OUT must not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, HISTORY, build
from .policy import fit_thresholds

PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "min_child_weight": 10}


def seeds_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def windows(run, seed, seeds, k):
    """t indices with t-12 .. t and t+k inside one run of the given seeds."""
    out = []
    ok = np.isin(seed, list(seeds))
    for r in np.unique(run[ok]):
        idx = np.flatnonzero(run == r)
        first, last = idx[0] + HISTORY - 1, idx[-1] - k
        if last >= first:
            out.append(np.arange(first, last + 1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def history(z, t):
    idx = t[:, None] + np.arange(-HISTORY + 1, 1)[None, :]
    return {"flow": z["veh"][idx].astype(np.float64), "speed": z["speed"][idx].astype(np.float64),
            "occupy": z["occ"][idx].astype(np.float64)}


def neighbours(graph, nodes):
    """Her direct neighbours: an edge either way (edges | edges.T in her decision agent), never itself."""
    e = pd.read_csv(graph, dtype={"from": str, "to": str})
    unknown = sorted(set(e["from"]).union(e["to"]) - set(nodes))
    if unknown:
        raise ValueError(f"graph names nodes not in the dataset: {unknown[:5]}")
    nb = {n: set() for n in nodes}
    for a, b in zip(e["from"], e["to"]):
        if a != b:
            nb[a].add(b)
            nb[b].add(a)
    return {n: sorted(v) for n, v in nb.items()}


def train(dataset, graph, out, train_seeds, dev_seeds, horizon_s=120, version=None):
    from xgboost import XGBRegressor

    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: results are never overwritten")
    if set(train_seeds) & set(dev_seeds):
        raise ValueError("a seed is both train and dev")
    z, meta = load(dataset)
    if meta.get("source", "exact") != "exact":
        raise ValueError("her model needs speed and occupancy, which only the exact source gives (decision 4)")
    dt = float(meta["dt_s"])
    k = round(horizon_s / dt)
    if abs(k * dt - horizon_s) > 1e-9 or k < 1:
        raise ValueError(f"horizon {horizon_s} s is not a whole number of {dt} s steps")
    nodes = [str(n) for n in z["nodes"]]
    N = len(nodes)
    nb = neighbours(graph, nodes)
    occ, run, seed, tod = z["occ"].astype(np.float64), z["run"], z["seed"], z["tod_s"]
    t_tr, t_dev = windows(run, seed, train_seeds, k), windows(run, seed, dev_seeds, k)
    if not len(t_tr) or not len(t_dev):
        raise ValueError("no train or dev window")
    F_tr, F_dev = build(history(z, t_tr), tod[t_tr], N), build(history(z, t_dev), tod[t_dev], N)
    y_tr, y_dev = occ[t_tr + k].reshape(-1), occ[t_dev + k].reshape(-1)
    m_tr, m_dev = np.isfinite(y_tr), np.isfinite(y_dev)
    model = XGBRegressor(**PARAMS, enable_categorical=True, tree_method="hist", device="cpu", n_jobs=4,
                         early_stopping_rounds=50, eval_metric="mae", random_state=42)
    model.fit(F_tr[m_tr], y_tr[m_tr], eval_set=[(F_dev[m_dev], y_dev[m_dev])], verbose=False)
    it = (0, model.best_iteration + 1)
    pred = model.predict(F_dev, iteration_range=it)
    last = occ[t_dev].reshape(-1)
    mae = lambda p: float(np.abs(p[m_dev] - y_dev[m_dev]).mean())
    rows_tr = np.flatnonzero(np.isin(seed, list(train_seeds)))
    th = fit_thresholds(occ[rows_tr].reshape(-1), z["speed"][rows_tr].astype(np.float64).reshape(-1),
                        np.abs(occ[t_tr + k] - occ[t_tr]).reshape(-1))
    os.makedirs(out)
    model.save_model(os.path.join(out, "xgboost_model.json"))
    pd.DataFrame([{"model": "xgboost", "split": "dev", "horizon_s": horizon_s, "MAE": mae(pred), "n": int(m_dev.sum())},
                  {"model": "last_value", "split": "dev", "horizon_s": horizon_s, "MAE": mae(last), "n": int(m_dev.sum())}]
                 ).to_csv(os.path.join(out, "dev_metrics.csv"), index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    metadata = {"version": version or os.path.basename(os.path.normpath(out)), "features": list(FEATURE_COLUMNS),
                "target": f"occupy_t+{horizon_s}s", "horizon_s": horizon_s, "dt_s": dt, "nodes": nodes,
                "best_iteration": int(model.best_iteration), "neighbours": nb, "params": PARAMS, "thresholds_train_only": th,
                "train_seeds": list(train_seeds), "dev_seeds": list(dev_seeds), "dataset": os.path.abspath(dataset),
                "dataset_meta": meta, "git_commit": commit,
                "based_on": "feature/traffic-prediction-agent 392fbae (src/prediction/xgboost_predictor.py)"}
    with open(os.path.join(out, "xgboost_export_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1)
    print(f"xgboost +{horizon_s}s dev MAE={mae(pred):.5f}  last value {mae(last):.5f}  "
          f"(best_iteration {model.best_iteration}); thresholds {th}")
    return metadata


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seeds", default="101-140")
    ap.add_argument("--dev-seeds", default="11-20")
    ap.add_argument("--horizon-s", type=int, default=120, help="her horizon on SUMO seconds (open question Q-J2)")
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)
    train(a.dataset, a.graph, a.out, seeds_arg(a.train_seeds), seeds_arg(a.dev_seeds), a.horizon_s, a.version)


if __name__ == "__main__":
    main()

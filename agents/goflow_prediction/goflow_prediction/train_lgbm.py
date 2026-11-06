"""LightGBM on the queue target, one model per horizon, early stopping on the DEV seeds (the test seeds are not read).

    python -m goflow_prediction.train_lgbm --dataset D2.npz --graph distance.csv --out out/goflow_prediction/models/v1 \
        --train-seeds 101-140 --dev-seeds 11-20 [--graph-kind real|random|none] [--no-node-id] [--goflow]

Writes into --out (which must not exist yet: results are never overwritten):
    lgbm_<h>s.txt        one model per horizon
    adjacency.npy        the graph the features used (so live use needs no SUMO files)
    storage.npy, nodes.json
    thresholds.json      per-approach alert thresholds (train seeds only)
    model_card.json      version, time base, seeds, features, dataset, git commit, dev metrics
    dev_metrics.csv      model and baselines per horizon on the dev seeds
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict

import numpy as np
import pandas as pd

from .baselines import DemandHA, cycle_persistence, last_value
from .config import ModelCard, approach_thresholds
from .data import anchors, load_dataset, load_graph
from .features import build, gather, neighbor_mats
from .graphs import get_graph
from .leakage import check_seed_splits, check_windows
from .metrics import masked_metrics

SEED = 42
PARAMS = dict(objective="regression", metric="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              seed=SEED, deterministic=True, force_row_wise=True, verbose=-1)


def seeds_arg(s: str) -> list[int]:
    """'101-140' or '11,12,15' or '101-110,120'."""
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def targets(ds, t0, k):
    return ds.fill[t0 + k]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def train(dataset, graph, out, train_seeds, dev_seeds, graph_kind="real", use_node_id=True, use_goflow=False,
          train_stride=1, rounds=3000, version=None, test_seeds=()):
    import lightgbm as lgb

    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: results are never overwritten, pick a new folder")
    check_seed_splits(train_seeds, dev_seeds, test_seeds)
    ds = load_dataset(dataset)
    tb = ds.timebase()
    if use_goflow and ds.goflow is None:
        raise ValueError("--goflow needs goflow_pred_occ in the dataset (GO-FLOW's real Town05 outputs)")
    adj = get_graph(load_graph(graph, ds.nodes), graph_kind, SEED)
    A_up, A_down = neighbor_mats(adj)
    a_tr, a_dev = anchors(ds, train_seeds, tb, train_stride), anchors(ds, dev_seeds, tb)
    if not len(a_tr) or not len(a_dev):
        raise ValueError("no train or dev window: check the seeds and the run lengths")
    check_windows(ds, a_tr, tb, train_seeds)
    check_windows(ds, a_dev, tb, dev_seeds)
    w_tr, w_dev = gather(ds, a_tr, tb.n_in), gather(ds, a_dev, tb.n_in)
    fill = ds.fill
    rows_tr = np.flatnonzero(np.isin(ds.seed, train_seeds))
    ha = DemandHA().fit(fill, ds.demand, rows_tr)

    os.makedirs(out)
    rows, names = [], None
    for hi, (h, k) in enumerate(tb.steps.items()):
        F_tr, names = build(w_tr, hi, ds.storage, A_up, A_down, use_node_id, use_goflow)
        F_dev, _ = build(w_dev, hi, ds.storage, A_up, A_down, use_node_id, use_goflow)
        y_tr, y_dev = targets(ds, a_tr, k).reshape(-1), targets(ds, a_dev, k).reshape(-1)
        m_tr, m_dev = np.isfinite(y_tr), np.isfinite(y_dev)
        cat = [names.index("node")] if use_node_id else []
        d_tr = lgb.Dataset(F_tr[m_tr], y_tr[m_tr], feature_name=names, categorical_feature=cat, free_raw_data=True)
        d_dev = lgb.Dataset(F_dev[m_dev], y_dev[m_dev], reference=d_tr)
        model = lgb.train(PARAMS, d_tr, num_boost_round=rounds, valid_sets=[d_dev],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        model.save_model(os.path.join(out, f"lgbm_{h}s.txt"), num_iteration=model.best_iteration)
        S, N = len(a_dev), ds.N
        truth = y_dev.reshape(S, N)
        candidates = {
            "lgbm" + ("_gf" if use_goflow else ""): model.predict(F_dev, num_iteration=model.best_iteration).reshape(S, N),
            "last_value": last_value(fill, a_dev),
            "cycle_persistence": cycle_persistence(fill, a_dev, k, tb, ds.run),
            "demand_ha": ha.predict(ds.demand_at[a_dev, hi]),
        }
        for name, p in candidates.items():
            r = dict(model=name, split="dev", horizon_s=int(h), **masked_metrics(p, truth))
            rows.append(r)
            print(f"{name:18s} {h:>4s} s  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  n={r['n']}")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "dev_metrics.csv"), index=False)
    np.save(os.path.join(out, "adjacency.npy"), adj)
    np.save(os.path.join(out, "storage.npy"), ds.storage)
    full, low = approach_thresholds(fill, ds.storage, rows_tr)
    with open(os.path.join(out, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({"rule": "per approach: full = Q99 of fill on the train seeds (>= 3 vehicles), "
                   "low = Q90 (< 0.7 full)", "train_seeds": list(train_seeds),
                   "full": dict(zip(ds.nodes, map(float, full))), "low": dict(zip(ds.nodes, map(float, low)))},
                  f, indent=1)
    with open(os.path.join(out, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump(ds.nodes, f)
    card = ModelCard(version=version or os.path.basename(os.path.normpath(out)), timebase=asdict(tb),
                     source=ds.meta.get("source", "exact"), nodes=ds.nodes, train_seeds=list(train_seeds),
                     dev_seeds=list(dev_seeds), features=names, use_node_id=use_node_id, use_goflow=use_goflow,
                     dataset=os.path.abspath(dataset), git_commit=git_commit(),
                     metrics_dev={f"{r['model']}@{r['horizon_s']}": r["MAE"] for r in rows})
    card_d = asdict(card)
    card_d["graph_kind"] = graph_kind
    card_d["dataset_meta"] = ds.meta
    with open(os.path.join(out, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card_d, f, indent=1)
    return table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-seeds", default="101-140")
    ap.add_argument("--dev-seeds", default="11-20")
    ap.add_argument("--test-seeds", default="21-30", help="only checked to be disjoint; never read")
    ap.add_argument("--graph-kind", choices=["real", "random", "none"], default="real")
    ap.add_argument("--no-node-id", dest="node_id", action="store_false",
                    help="no approach id feature (the model to test on held-out junctions)")
    ap.add_argument("--goflow", action="store_true", help="GO-FLOW's occupancy forecast as features")
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--version", default=None)
    a = ap.parse_args(argv)
    train(a.dataset, a.graph, a.out, seeds_arg(a.train_seeds), seeds_arg(a.dev_seeds), a.graph_kind, a.node_id,
          a.goflow, a.train_stride, version=a.version, test_seeds=seeds_arg(a.test_seeds))


if __name__ == "__main__":
    main()

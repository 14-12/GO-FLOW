"""Her Town05 forecasts for a whole dataset, as DSO4 reads them (`goflow_pred_occ`, and her traffic_state per row).

    python -m goflow_traffic.export --dataset D2.npz --graph distance.csv --model OUT_MODEL \
        --train-seeds 101-140 --dev-seeds 11-20 --out D2_goflow.npz [--folds 4]

* On the dev and test seeds: her model trained on all train seeds (--model, from train.py), step by step through
  agent.GoFlowTown05, the live code path.
* On the train seeds: OUT-OF-FOLD. Her model is retrained --folds times, each time without a quarter of the train
  seeds, and forecasts only the seeds it did not see. Otherwise DSO4's LightGBM + GO-FLOW would learn from forecasts
  that are better on train than they can ever be live (in-sample).

Writes a NEW dataset file: the input dataset plus `goflow_pred_occ` [T, N] and `goflow_state` [T, N] (her
traffic_state, "" while warming up), and meta["goflow"] (model versions, folds).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np

from .agent import GoFlowTown05
from .train import seeds_arg, train


def run_model(model_dir, z, seeds):
    """Her agent over every run of `seeds`, one observation at a time. -> {row: (pred [N], states [N])}."""
    agent = GoFlowTown05(model_dir)
    nodes = [str(n) for n in z["nodes"]]
    if nodes != agent.nodes:
        raise ValueError("dataset nodes differ from the model's")
    out = {}
    rows = np.flatnonzero(np.isin(z["seed"], list(seeds)))
    prev = None
    for i in rows:
        if z["run"][i] != prev:
            agent.reset()
            prev = z["run"][i]
        obs = {"t": float(z["t"][i]), "tod_s": float(z["tod_s"][i]),
               "nodes": {n: {"veh": float(z["veh"][i, j]), "speed": float(z["speed"][i, j]),
                             "occ": float(z["occ"][i, j])} for j, n in enumerate(nodes)}}
        msg, _ = agent.step(obs)
        if msg is not None:
            pred = np.array([np.nan if v is None else v["pred_occ"] for v in msg["nodes"].values()])
            st = np.array(["" if v is None else v["traffic_state"] for v in msg["nodes"].values()], dtype=object)
            out[int(i)] = (pred, st)
    return out


def export(dataset, graph, model_dir, train_seeds, dev_seeds, out, folds=4, horizon_s=None):
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: results are never overwritten")
    z = dict(np.load(dataset, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    with open(os.path.join(model_dir, "xgboost_export_metadata.json"), encoding="utf-8") as f:
        full_meta = json.load(f)
    horizon_s = horizon_s or full_meta["horizon_s"]
    T, N = z["occ"].shape
    pred = np.full((T, N), np.nan, np.float32)
    state = np.full((T, N), "", dtype="<U10")
    other = sorted(set(np.unique(z["seed"]).tolist()) - set(train_seeds))
    for i, (p, s) in run_model(model_dir, z, other).items():
        pred[i], state[i] = p, s
    rng = np.random.default_rng(0)
    shuffled = list(rng.permutation(train_seeds))
    parts = [sorted(int(x) for x in shuffled[k::folds]) for k in range(folds)]
    versions = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, held in enumerate(parts):
            fit = [s for s in train_seeds if s not in held]
            mdir = os.path.join(tmp, f"fold{k}")
            train(dataset, graph, mdir, fit, dev_seeds, horizon_s, version=f"oof-fold{k}")
            for i, (p, s) in run_model(mdir, z, held).items():
                pred[i], state[i] = p, s
            versions.append({"fold": k, "held_out_seeds": held})
    meta["goflow"] = {"model": full_meta["version"], "horizon_s": horizon_s, "train_rows": "out-of-fold",
                      "folds": versions, "dev_test_rows": f"model {full_meta['version']} trained on all train seeds"}
    z["goflow_pred_occ"] = pred
    z["goflow_state"] = state
    z["meta"] = json.dumps(meta)
    np.savez_compressed(out, **z)
    known = np.isfinite(pred).mean()
    print(f"[ok] {out}: her forecast on {known:.1%} of the rows (the others warming up: 13 steps per run)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-seeds", default="101-140")
    ap.add_argument("--dev-seeds", default="11-20")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    export(a.dataset, a.graph, a.model, seeds_arg(a.train_seeds), seeds_arg(a.dev_seeds), a.out, a.folds)


if __name__ == "__main__":
    main()

"""Scores a trained agent by replaying a dataset through it (agent.replay: the live code path).

    python -m goflow_prediction.evaluate --model DIR [--model DIR_GF] --dataset D2.npz --seeds 11-20 --out OUT
    python -m goflow_prediction.evaluate ... --seeds 21-30 --open-test "final v1"     # once: TestGuard lock

Writes OUT/forecast_metrics.csv (per horizon: the agent's chosen forecast and each baseline), OUT/events.json
(alert precision, onsets caught, lead time, false alerts per junction-hour), OUT/alerts.jsonl, OUT/run.json
(models, seeds, dataset, git commit). OUT must not exist yet.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .agent import ForecastAgent, ModelBundle, replay
from .baselines import cycle_persistence, last_value
from .data import load_dataset
from .leakage import TestGuard
from .metrics import event_metrics, masked_metrics, onsets
from .train_lgbm import git_commit, seeds_arg


def evaluate(model_dirs, dataset, seeds, out, open_test=None, lock_path=None):
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists: results are never overwritten")
    bundles = [ModelBundle(d) for d in model_dirs]
    used = set(seeds) & set(bundles[0].card["train_seeds"] + bundles[0].card["dev_seeds"])
    is_test = not (set(seeds) <= set(bundles[0].card["dev_seeds"]))
    if used and is_test:
        raise ValueError(f"seeds {sorted(used)} were used for training or tuning: not a test")
    if is_test:
        if not open_test:
            raise ValueError("these are not the dev seeds: pass --open-test REASON to open the test (once)")
        TestGuard(lock_path or os.path.join(os.path.dirname(os.path.abspath(out)), "TEST_OPENED.lock")).open(open_test)
    ds = load_dataset(dataset)
    agent = ForecastAgent(bundles)
    if ds.nodes != agent.nodes:
        raise ValueError("dataset nodes differ from the model's")
    tb = agent.tb
    alerts, fc = replay(agent, ds, seeds)
    fill = ds.fill
    rows = sorted(fc)
    t0 = np.array(rows, dtype=np.int64)
    table = []
    for hi, (h, k) in enumerate(tb.steps.items()):
        ok = (t0 + k < ds.T) & (ds.run[np.clip(t0 + k, 0, ds.T - 1)] == ds.run[t0])
        tt = t0[ok]
        truth = fill[tt + k]
        agent_pred = np.array([[np.nan if v["fill"][hi] is None else v["fill"][hi]
                                for v in fc[i]["nodes"].values()] for i in tt])
        for name, p in (("agent", agent_pred), ("last_value", last_value(fill, tt)),
                        ("cycle_persistence", cycle_persistence(fill, tt, k, tb, ds.run))):
            table.append(dict(model=name, horizon_s=int(h), **masked_metrics(p, truth)))
    sel = np.isin(ds.seed, list(seeds))
    full, low = agent.full, agent.low      # the thresholds the agent alerts with
    om = onsets(fill, ds.run, full, low) & sel[:, None]
    ev = event_metrics(alerts, om, ds.t, ds.nodes, fill, full, tb.dt_s, rows=np.flatnonzero(sel), run=ds.run)
    ev["thresholds"] = "per approach (thresholds.json)" if bundles[0].thresholds is not None else "fixed 0.85 / 0.6"
    os.makedirs(out)
    pd.DataFrame(table).to_csv(os.path.join(out, "forecast_metrics.csv"), index=False)
    with open(os.path.join(out, "events.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=1, default=float)
    with open(os.path.join(out, "alerts.jsonl"), "w", encoding="utf-8") as f:
        for a in alerts:
            f.write(json.dumps(a) + "\n")
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
        json.dump(dict(models=[b.version for b in bundles], seeds=list(seeds), dataset=os.path.abspath(dataset),
                       dataset_meta=ds.meta, test=is_test, git_commit=git_commit()), f, indent=1)
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps(ev, indent=1, default=float))
    return pd.DataFrame(table), ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--open-test", default=None, metavar="REASON")
    a = ap.parse_args(argv)
    evaluate(a.model, a.dataset, seeds_arg(a.seeds), a.out, a.open_test)


if __name__ == "__main__":
    main()

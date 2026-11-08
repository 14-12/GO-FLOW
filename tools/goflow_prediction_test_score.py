"""Scores the shipped GO-FLOW forecasters on the closed TEST seeds (21-30), once, through the live code path.

    python tools/goflow_prediction_test_score.py --out out/goflow_prediction/test_v4 --reason "v4 under agents_observed" \
        out/goflow_prediction/datasets/T_agents_observed_x1 out/goflow_prediction/datasets/T_agents_observed_x2 ...

Each dataset folder (tools/goflow_prediction_dataset.py, --seeds 21-30) is scored as it is, one result per folder:

* GO-FLOW (goflow_traffic.agent.GoFlowTown05, step by step): occupancy at +120 s, MAE against "it stays as now".
* DSO4 (goflow_prediction.agent.ForecastAgent with dso4 + dso4_goflow, step by step, GO-FLOW's forecasts fed in as
  the viewer feeds them): queue fill at +60/+120/+300 s, MAE against "it stays as now" and "as one cycle ago".
* The long-queue alerts: the agent's own p_long, scored as congestion.train scores the dev seeds (alerts with a
  cooldown, right if a long queue starts within k steps, starts caught), at 0.5 / the shipped threshold / 0.8; and
  how often a long queue really started for each band of p_long (is "70 %" really about 70 %?).

The test seeds open once (leakage.TestGuard, OUT/TEST_OPENED.lock): every dataset goes in the one call.
Writes OUT/<dataset name>.json and OUT/summary.json. OUT must not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agents" / "goflow_prediction"), str(ROOT / "agents" / "goflow_traffic")]

from goflow_prediction.agent import ForecastAgent, ModelBundle, replay  # noqa: E402
from goflow_prediction.baselines import cycle_persistence, last_value  # noqa: E402
from goflow_prediction.congestion import CYCLE_STEPS, cycle_means, score_alerts, starts  # noqa: E402
from goflow_prediction.data import load_dataset  # noqa: E402
from goflow_prediction.leakage import TestGuard  # noqa: E402
from goflow_prediction.metrics import masked_metrics  # noqa: E402

MODELS = ROOT / "models" / "goflow"
TEST_SEEDS = list(range(21, 31))
BANDS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]


def with_goflow(src: Path, dst: Path) -> None:
    """src dataset + GO-FLOW's step-by-step forecasts (goflow_pred_occ, goflow_state) -> dst."""
    from goflow_traffic.export import run_model

    z = np.load(src, allow_pickle=False)
    out = run_model(str(MODELS / "goflow_xgb"), z, TEST_SEEDS)
    T, N = z["occ"].shape
    occ, st = np.full((T, N), np.nan), np.full((T, N), "", dtype="<U16")
    for i, (p, s) in out.items():
        occ[i], st[i] = p, s.astype(str)
    np.savez_compressed(dst, **{k: z[k] for k in z.files}, goflow_pred_occ=occ, goflow_state=st)


def score(folder: Path, tmp: Path) -> dict:
    src = folder / "dataset.npz"
    gf = tmp / f"{folder.name}_goflow.npz"
    with_goflow(src, gf)
    ds = load_dataset(str(gf))
    if sorted(set(ds.seed.tolist())) != TEST_SEEDS:
        raise ValueError(f"{folder}: not the test seeds 21-30")
    res = {"dataset": str(folder), "meta": {k: ds.meta.get(k) for k in ("controller", "scale", "accidents", "clock_speed",
                                                                     "minutes", "warmup", "git_commit")}}

    # GO-FLOW: occupancy at +120 s
    k = 4
    occ = ds.x["occ"]
    rows = np.array([i for i in range(ds.T - k) if ds.run[i + k] == ds.run[i] and np.isfinite(ds.goflow[i]).any()])
    res["goflow_occ_120s"] = {"goflow": masked_metrics(ds.goflow[rows], occ[rows + k]),
                              "last_value": masked_metrics(occ[rows], occ[rows + k])}

    # DSO4: queue fill, the agent as it runs live
    agent = ForecastAgent([ModelBundle(str(MODELS / "dso4")), ModelBundle(str(MODELS / "dso4_goflow"))])
    if ds.nodes != agent.nodes:
        raise ValueError("dataset nodes differ from the model's")
    alerts, fc = replay(agent, ds, TEST_SEEDS)
    fill, tb = ds.fill, agent.tb
    t0 = np.array(sorted(fc), dtype=np.int64)
    table = []
    for hi, (h, kk) in enumerate(tb.steps.items()):
        ok = (t0 + kk < ds.T) & (ds.run[np.clip(t0 + kk, 0, ds.T - 1)] == ds.run[t0])
        tt = t0[ok]
        pred = np.array([[np.nan if v["fill"][hi] is None else v["fill"][hi] for v in fc[i]["nodes"].values()]
                         for i in tt])
        truth = fill[tt + kk]
        for name, p in (("agent", pred), ("last_value", last_value(fill, tt)),
                        ("cycle_persistence", cycle_persistence(fill, tt, kk, tb, ds.run))):
            table.append(dict(model=name, horizon_s=int(h), **masked_metrics(p, truth)))
    res["dso4_fill"] = table

    # Long-queue alerts: the agent's own p_long
    cg = agent.congestion
    st = np.where(ds.storage >= 1.0, ds.storage, np.nan)
    c = cycle_means(ds.x["halt"] / st, ds.run)
    st_mask = starts(c, ds.run, cg.level)
    K = cg.k
    rows = np.array([i for i in t0 if i + K < ds.T and ds.run[i + K] == ds.run[i]
                     and all(fc[i]["nodes"][n].get("p_long") is not None for n in agent.nodes)], dtype=np.int64)
    prob = np.array([[fc[i]["nodes"][n]["p_long"] for n in agent.nodes] for i in rows], dtype=float)
    ev = score_alerts(prob, rows, ds.run, st_mask, K)
    res["long_queue"] = {"threshold_shipped": cg.threshold, "k_steps": K,
                         "hours": float(len(rows) * tb.dt_s / 3600),
                         "by_threshold": [ev(x) for x in sorted({0.5, cg.threshold, 0.8})]}
    fut = np.zeros(prob.shape, bool)
    for d in range(1, K + 1):
        fut |= st_mask[rows + d]
    elig = prob > 0
    cal = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = elig & (prob >= lo) & (prob < hi)
        cal.append({"p_long": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                    "long_queue_started": round(float(fut[m].mean()), 3) if m.any() else None})
    res["long_queue"]["calibration"] = cal
    res["long_queue"]["agent_alerts"] = sum(a.get("kind") == "long_queue_soon" for a in alerts)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="+", help="dataset folders (each with dataset.npz) on seeds 21-30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reason", required=True, help="written in the lock: why the test is opened")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"{out} exists: results are never overwritten")
    for d in a.datasets:
        if not (Path(d) / "dataset.npz").is_file():
            raise FileNotFoundError(f"{d}/dataset.npz")
    out.mkdir(parents=True)
    TestGuard(str(out / "TEST_OPENED.lock")).open(a.reason)
    tmp = out / "tmp"
    tmp.mkdir()
    summary = []
    for d in a.datasets:
        r = score(Path(d), tmp)
        name = Path(d).name
        (out / f"{name}.json").write_text(json.dumps(r, indent=1, default=float), encoding="utf-8")
        thr = next(x for x in r["long_queue"]["by_threshold"] if x["threshold"] == r["long_queue"]["threshold_shipped"])
        fill = {f"{x['model']}@{x['horizon_s']}": round(x["MAE"], 4) for x in r["dso4_fill"]}
        line = {"dataset": name, **{k: r["meta"][k] for k in ("controller", "scale")},
                "goflow_occ120_mae": round(r["goflow_occ_120s"]["goflow"]["MAE"], 4),
                "goflow_last_value_mae": round(r["goflow_occ_120s"]["last_value"]["MAE"], 4), **fill,
                "long_alerts": thr["alerts"], "long_right": thr["right"], "long_starts": thr["starts"],
                "long_caught": thr["caught"], "long_lead_s": thr["median_lead_s"]}
        summary.append(line)
        print(json.dumps(line))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

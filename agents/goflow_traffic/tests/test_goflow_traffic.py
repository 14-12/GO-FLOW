"""Tests on a toy corridor (a test fixture, NOT Town05 data and not her results)."""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from goflow_traffic.agent import GoFlowTown05
from goflow_traffic.features import FEATURE_COLUMNS, build
from goflow_traffic.policy import rule_based_decision
from goflow_traffic.train import train

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = ["a-1", "b-1", "c-1", "d-1"]
TH = {"occupancy_median": 0.1, "occupancy_high": 0.3, "occupancy_severe": 0.5, "speed_low": 3.0,
      "speed_median": 8.0, "strong_change": 0.05}


def toy(path, seeds, steps=150):
    N, cols = len(NODES), {k: [] for k in ("t", "run", "seed", "tod", "occ", "veh", "speed")}
    for r, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        tt = np.arange(steps) * 30.0 + 30
        wave = 0.2 + 0.15 * np.sin(tt[:, None] / 300.0 + np.arange(N)[None, :] * 0.7 + s)
        occ = np.clip(wave + rng.normal(0, 0.01, (steps, N)), 0, 1)
        for k, v in (("t", tt), ("run", np.full(steps, r)), ("seed", np.full(steps, s)),
                     ("tod", (61200 + tt * 60) % 86400), ("occ", occ), ("veh", occ * 30), ("speed", 12 * (1 - occ))):
            cols[k].append(v)
    c = {k: np.concatenate(v) for k, v in cols.items()}
    T = len(c["t"])
    meta = {"dt_s": 30.0, "horizons_s": [60, 120, 300], "source": "exact", "note": "test fixture"}
    np.savez_compressed(path, nodes=np.array(NODES), storage=np.full(N, 20.0), t=c["t"], run=c["run"],
                        seed=c["seed"], tod_s=c["tod"], demand=np.full(T, 0.5), demand_at=np.full((T, 3), 0.5),
                        halt=c["occ"] * 20, veh=c["veh"], speed=c["speed"], occ=c["occ"],
                        green=np.full((T, N), 0.5), meta=json.dumps(meta))
    g = os.path.join(os.path.dirname(path), "distance.csv")
    pd.DataFrame([{"from": a, "to": b, "cost": 100.0} for a, b in itertools.pairwise(NODES)]).to_csv(g, index=False)
    return path, g


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    d = tmp_path_factory.mktemp("gf")
    ds, g = toy(str(d / "toy.npz"), [101, 102, 103, 104, 11])
    out = d / "xgb-test"
    train(ds, g, str(out), [101, 102, 103, 104], [11], horizon_s=120, version="xgb-test")
    return out, ds


def test_rolling_means_match_her_pandas_recipe():
    rng = np.random.default_rng(0)
    x = rng.random((13, 1))
    hist = {"flow": x, "speed": x, "occupy": x}
    F = build({k: v[None] for k, v in hist.items()}, np.array([3600.0]), 1)
    s = pd.Series(x[:, 0])
    for w in (3, 6, 12):
        assert np.isclose(F[f"occupy_rolling_mean_{w}"].iloc[0], s.shift(1).rolling(w).mean().iloc[-1], atol=1e-6)
    for k in (1, 2, 3, 6, 12):
        assert np.isclose(F[f"occupy_lag_{k}"].iloc[0], s.shift(k).iloc[-1], atol=1e-6)
    assert tuple(F.columns) == FEATURE_COLUMNS and str(F["location"].dtype) == "category"
    assert F["hour"].iloc[0] == 1 and F["slot_in_day"].iloc[0] == 12


def state(cur=0.05, pred=0.05, speed=10.0, nb=(), occ_trend=0.0, speed_trend=0.0):
    return {"current_occupancy": cur, "predicted_occupancy": pred, "occupancy_change": pred - cur,
            "current_speed": speed, "recent_occupancy_trend": occ_trend, "recent_speed_trend": speed_trend,
            "neighbors": [{"current_occupancy": o} for o in nb]}


@pytest.mark.parametrize("st,traffic,action", [
    (state(pred=0.6), "severe", "manual_review"),
    (state(cur=0.1, pred=0.16), "congested", "manual_review"),            # strong rise (>= 0.05)
    (state(pred=0.35, nb=(0.4,)), "congested", "coordinate_with_neighbor"),
    (state(nb=(0.31,)), "moderate", "monitor"),                          # a congested neighbour
    (state(speed=2.0), "moderate", "monitor"),                           # low speed
    (state(occ_trend=0.01, speed_trend=-0.1), "moderate", "monitor"),
    (state(), "free_flow", "keep_current_signal_plan"),
])
def test_her_rules_unchanged(st, traffic, action):
    d = rule_based_decision(st, TH, 120)
    assert (d["traffic_state"], d["recommended_action"], d["confidence"]) == (traffic, action, 0.5)
    assert "+120 s" in d["reason"]


def test_train_writes_metadata_and_beats_last_value(model):
    out, _ = model
    meta = json.loads((out / "xgboost_export_metadata.json").read_text(encoding="utf-8"))
    assert meta["nodes"] == NODES and meta["horizon_s"] == 120 and meta["neighbours"]["b-1"] == ["a-1", "c-1"]
    assert set(meta["thresholds_train_only"]) >= {"occupancy_high", "strong_change"}
    t = pd.read_csv(out / "dev_metrics.csv").set_index("model").MAE
    assert t["xgboost"] < t["last_value"]
    with pytest.raises(FileExistsError):
        train(model[1], os.path.join(os.path.dirname(model[1]), "distance.csv"), str(out), [101], [11])


def obs(z, i, seq, drop=None):
    nodes = {n: {"veh": float(z["veh"][i, j]), "speed": float(z["speed"][i, j]), "occ": float(z["occ"][i, j]),
                 "halt": None, "green": None} for j, n in enumerate(NODES)}
    if drop:
        nodes[drop]["occ"] = None
    return {"type": "obs", "v": 1, "seq": seq, "t": float(z["t"][i]), "tod_s": float(z["tod_s"][i]), "nodes": nodes}


def test_agent_warmup_and_incomplete_history(model):
    out, ds = model
    z = np.load(ds)
    a = GoFlowTown05(out)
    rows = np.flatnonzero(z["seed"] == 11)
    res = [a.step(obs(z, i, s, drop="c-1" if s == 5 else None))[0] for s, i in enumerate(rows[:20])]
    assert all(r is None for r in res[:12]) and res[12] is not None
    assert res[12]["nodes"]["c-1"] is None                 # a missing value in its 13 steps: no forecast for it
    assert res[18]["nodes"]["c-1"]["traffic_state"] in {"free_flow", "moderate", "congested", "severe"}
    assert res[18]["decision_source"] == "rule_based_fallback"


def test_serve_process(model):
    out, ds = model
    z = np.load(ds)
    rows = np.flatnonzero(z["seed"] == 11)[:14]
    lines = ["garbage", json.dumps({"type": "hello", "v": 1, "nodes": NODES, "dt_s": 30})]
    lines += [json.dumps(obs(z, i, s)) for s, i in enumerate(rows)]
    p = subprocess.run([sys.executable, "-m", "goflow_traffic.serve", "--model", str(out)],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=PKG, timeout=120, check=False)
    assert p.returncode == 0, p.stderr
    msgs = [json.loads(x) for x in p.stdout.splitlines()]
    assert msgs[0]["type"] == "ready" and msgs[0]["horizons_s"] == [120]
    assert msgs[1]["type"] == "error"
    gf = [m for m in msgs if m["type"] == "goflow"]
    assert [m["seq"] for m in gf] == [12, 13]


def test_missing_model_is_an_error(tmp_path):
    p = subprocess.run([sys.executable, "-m", "goflow_traffic.serve", "--model", str(tmp_path)],
                       input="", capture_output=True, text=True, cwd=PKG, timeout=60, check=False)
    assert p.returncode == 2 and json.loads(p.stdout.splitlines()[0])["type"] == "error"


def test_strict_inputs_as_her_contract(model):
    out, ds = model
    z = np.load(ds)
    a = GoFlowTown05(out)
    m = obs(z, 0, 0)
    m["nodes"]["a-1"]["occ"] = True
    m["nodes"]["b-1"]["speed"] = "fast"
    _, problems = a.step(m)
    assert any("a-1.occ" in p for p in problems) and any("b-1.speed" in p for p in problems)


def test_camera_source_refused(tmp_path):
    p, g = toy(str(tmp_path / "cam.npz"), [101, 11])
    z = dict(np.load(p))
    meta = json.loads(str(z["meta"]))
    meta["source"] = "cameras"
    z["meta"] = json.dumps(meta)
    np.savez_compressed(p, **z)
    with pytest.raises(ValueError, match="exact"):
        train(p, g, str(tmp_path / "m"), [101], [11])


def test_joint_her_messages_feed_dso4(model, tmp_path):
    """Her messages, as this package writes them, pass DSO4's input checks when the town forwards them."""
    pred_pkg = os.path.join(os.path.dirname(PKG), "goflow_prediction")
    if not os.path.isdir(pred_pkg):
        pytest.skip("goflow_prediction not next to this package")
    sys.path.insert(0, pred_pkg)
    from goflow_prediction.schema import check_in

    out, ds = model
    z = np.load(ds)
    a = GoFlowTown05(out)
    rows = np.flatnonzero(z["seed"] == 11)[:14]
    msg = None
    for s, i in enumerate(rows):
        msg, _ = a.step(obs(z, i, s))
    assert msg is not None
    town_obs = obs(z, rows[-1], 13)
    town_obs.update(demand=0.5, demand_at={"60": 0.5, "120": 0.5, "300": 0.5}, goflow=msg["nodes"])
    assert check_in(town_obs) == []

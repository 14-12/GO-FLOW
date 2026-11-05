import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from goflow_prediction.agent import ForecastAgent, ModelBundle, obs_from_dataset, replay
from goflow_prediction.config import TimeBase
from goflow_prediction.data import anchors, load_dataset, load_graph
from goflow_prediction.evaluate import evaluate
from goflow_prediction.leakage import TestGuard, check_seed_splits
from goflow_prediction.metrics import onsets
from goflow_prediction.schema import check_in
from goflow_prediction.train_lgbm import seeds_arg, train

from conftest import NODES

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_timebase_steps():
    tb = TimeBase()
    assert tb.steps == {"60": 2, "120": 4, "300": 10}
    assert tb.cycle_steps == 4
    with pytest.raises(ValueError):
        TimeBase(dt_s=30, horizons_s=(45,))


def test_seeds_arg():
    assert seeds_arg("101-103,11") == [101, 102, 103, 11]


def test_seed_splits_disjoint():
    with pytest.raises(AssertionError):
        check_seed_splits([1, 2], [2], [3])


def test_dataset_roundtrip_and_windows_stay_in_one_run(toy):
    ds = load_dataset(toy / "toy.npz")
    assert ds.nodes == NODES and ds.fill.shape == (ds.T, 6)
    tb = ds.timebase()
    a = anchors(ds, [101, 102], tb)
    assert len(a)
    assert (ds.run[a - tb.n_in + 1] == ds.run[a]).all() and (ds.run[a + tb.max_steps] == ds.run[a]).all()
    assert np.isin(ds.seed[a], [101, 102]).all()


def test_graph_direction_and_unknown_node(toy, tmp_path):
    adj = load_graph(toy / "distance.csv", NODES)
    assert adj[0, 1] > 0 and adj[1, 0] == 0 and np.allclose(np.diag(adj), 1)
    pd.DataFrame([{"from": "a-1", "to": "zz-9", "cost": 1.0}]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        load_graph(tmp_path / "bad.csv", NODES)


def test_onset_rule():
    f = np.array([[0.1]] * 4 + [[0.9], [0.9], [0.9]])
    om = onsets(f, np.zeros(7, int), full=0.85, low=0.6)
    assert om[:, 0].tolist() == [False] * 4 + [True, False, False]
    om2 = onsets(f, np.array([0, 0, 0, 1, 1, 1, 1]), full=0.85, low=0.6)
    assert not om2.any()   # the four steps before do not lie in one run


def test_training_beats_last_value_on_toy(trained):
    t = pd.read_csv(trained / "dev_metrics.csv")
    card = json.loads((trained / "model_card.json").read_text(encoding="utf-8"))
    assert card["train_seeds"] == list(range(101, 107)) and card["dev_seeds"] == [11, 12]
    for h in (120, 300):
        m = t[t.horizon_s == h].set_index("model").MAE
        assert m["lgbm"] < m["last_value"], f"toy corridor, {h} s"


def test_never_overwrite(toy, trained):
    with pytest.raises(FileExistsError):
        train(str(toy / "toy.npz"), str(toy / "distance.csv"), str(trained), [101], [11])


def test_agent_warmup_forecast_and_propagation_alert(toy, trained):
    ds = load_dataset(toy / "toy.npz")
    agent = ForecastAgent([ModelBundle(trained)])
    rows = np.flatnonzero(ds.seed == 11)
    outs = [agent.step(obs_from_dataset(ds, i, s)) for s, i in enumerate(rows)]
    assert all(o["forecast"] is None for o in outs[:11]) and outs[11]["forecast"] is not None
    fc = outs[20]["forecast"]
    assert list(fc["nodes"]) == NODES and len(fc["nodes"]["a-1"]["fill"]) == 3
    alerts = [a for o in outs for a in o["alerts"]]
    assert alerts, "the toy jams should raise alerts"
    assert {a["kind"] for a in alerts} <= {"spillback_soon", "propagation"}
    assert any(a["kind"] == "propagation" and a["source"] for a in alerts)
    by_node = {}
    for a in alerts:   # cooldown: two alerts on one approach are at least 120 s apart
        if a["node"] in by_node:
            assert a["t"] - by_node[a["node"]] >= 120
        by_node[a["node"]] = a["t"]


def test_time_gap_resets_window(toy, trained):
    ds = load_dataset(toy / "toy.npz")
    agent = ForecastAgent([ModelBundle(trained)])
    rows = np.flatnonzero(ds.seed == 11)[:13]
    for s, i in enumerate(rows):
        out = agent.step(obs_from_dataset(ds, i, s))
    assert out["forecast"] is not None
    jump = obs_from_dataset(ds, rows[-1], 99)
    jump["t"] += 3600
    assert agent.step(jump)["warming_up"]


def test_replay_and_dev_evaluation(toy, trained, tmp_path):
    table, ev = evaluate([str(trained)], str(toy / "toy.npz"), [11, 12], str(tmp_path / "ev"))
    assert set(table.model) == {"agent", "last_value", "cycle_persistence"}
    assert ev["alerts"] > 0 and ev["onsets"] > 0
    assert (tmp_path / "ev" / "alerts.jsonl").is_file()


def test_test_seeds_need_open_test_and_open_once(toy, trained, tmp_path):
    with pytest.raises(ValueError):
        evaluate([str(trained)], str(toy / "toy.npz"), [21], str(tmp_path / "t1"))
    lock = tmp_path / "TEST_OPENED.lock"
    evaluate([str(trained)], str(toy / "toy.npz"), [21], str(tmp_path / "t2"), open_test="unit test",
             lock_path=str(lock))
    with pytest.raises(RuntimeError):
        TestGuard(str(lock)).open("again")
    with pytest.raises(ValueError):   # dev seeds are not a test
        evaluate([str(trained)], str(toy / "toy.npz"), [101], str(tmp_path / "t3"), open_test="x")


def test_schema_rejects_bad_messages():
    ok = {"type": "obs", "v": 1, "t": 30.0, "demand": 0.5, "demand_at": {"60": 0.5},
          "nodes": {"a-1": {"halt": 3.0, "veh": None}}}
    assert check_in(ok) == []
    assert check_in({**ok, "v": 2})
    assert check_in({**ok, "t": True})
    assert check_in({**ok, "nodes": {"a-1": {"halt": float("nan")}}})
    assert check_in({**ok, "nodes": {"a-1": {"queue": 1}}})
    assert check_in({**ok, "goflow": {"a-1": {"pred_occ": "high"}}})
    assert check_in([1, 2])


def test_serve_process_json_lines(toy, trained):
    """The live process end to end: ready, an error for garbage (and it keeps going), forecasts after warm-up."""
    ds = load_dataset(toy / "toy.npz")
    rows = np.flatnonzero(ds.seed == 12)[:14]
    lines = [json.dumps({"type": "hello", "v": 1, "dt_s": 30, "nodes": NODES, "source": "exact"}), "not json",
             json.dumps({"type": "obs", "v": 1, "t": "x"})]
    lines += [json.dumps(obs_from_dataset(ds, i, s)) for s, i in enumerate(rows)]
    p = subprocess.run([sys.executable, "-m", "goflow_prediction.serve", "--model", str(trained)],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=PKG, timeout=120)
    assert p.returncode == 0, p.stderr
    msgs = [json.loads(x) for x in p.stdout.splitlines()]
    assert msgs[0]["type"] == "ready" and msgs[0]["horizons_s"] == [60, 120, 300]
    assert sum(m["type"] == "error" for m in msgs) == 2
    fcs = [m for m in msgs if m["type"] == "forecast"]
    assert len(fcs) == 3 and fcs[0]["seq"] == 11


def test_serve_reports_missing_model(tmp_path):
    p = subprocess.run([sys.executable, "-m", "goflow_prediction.serve", "--model", str(tmp_path / "none")],
                       input="", capture_output=True, text=True, cwd=PKG, timeout=60)
    assert p.returncode == 2
    assert json.loads(p.stdout.splitlines()[0])["type"] == "error"


def test_goflow_model_and_confirmation(toy, tmp_path):
    """With GO-FLOW's forecast in the data, the GO-FLOW model trains; without her answer it is not used."""
    from conftest import toy_dataset
    from goflow_prediction.data import save_dataset

    ds = toy_dataset(list(range(101, 105)) + [11], goflow=True)
    save_dataset(ds, tmp_path / "gf.npz")
    out = tmp_path / "gf-model"
    train(str(tmp_path / "gf.npz"), str(toy / "distance.csv"), str(out), list(range(101, 105)), [11],
          use_goflow=True, rounds=100, version="gf-test")
    b = ModelBundle(out)
    assert b.name == "lgbm_gf" and "gf_pred_occ" in b.card["features"]
    agent = ForecastAgent([b])
    rows = np.flatnonzero(ds.seed == 11)[:14]
    for s, i in enumerate(rows):   # no goflow field in the messages: her model's absence is handled
        msg = obs_from_dataset(ds, i, s)
        msg.pop("goflow")
        out_ = agent.step(msg)
    assert out_["forecast"]["chosen"]["60"] in ("last_value", "cycle_persistence", "lgbm_gf")
    assert "lgbm_gf" not in agent._available()

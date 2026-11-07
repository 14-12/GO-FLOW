import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import NODES
from goflow_prediction.agent import ForecastAgent, ModelBundle, obs_from_dataset
from goflow_prediction.config import TimeBase
from goflow_prediction.data import anchors, load_dataset, load_graph
from goflow_prediction.evaluate import evaluate
from goflow_prediction.leakage import TestGuard, check_seed_splits
from goflow_prediction.metrics import onsets
from goflow_prediction.schema import check_in
from goflow_prediction.train_lgbm import seeds_arg, train

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
                       input="\n".join(lines) + "\n", capture_output=True, text=True, cwd=PKG, timeout=120, check=False)
    assert p.returncode == 0, p.stderr
    msgs = [json.loads(x) for x in p.stdout.splitlines()]
    assert msgs[0]["type"] == "ready" and msgs[0]["horizons_s"] == [60, 120, 300]
    assert sum(m["type"] == "error" for m in msgs) == 2
    fcs = [m for m in msgs if m["type"] == "forecast"]
    assert len(fcs) == 3 and fcs[0]["seq"] == 11


def test_serve_reports_missing_model(tmp_path):
    p = subprocess.run([sys.executable, "-m", "goflow_prediction.serve", "--model", str(tmp_path / "none")],
                       input="", capture_output=True, text=True, cwd=PKG, timeout=60, check=False)
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


def test_per_approach_thresholds_from_train(trained):
    th = json.loads((trained / "thresholds.json").read_text(encoding="utf-8"))
    assert th["train_seeds"] == list(range(101, 107))
    for n in NODES:
        assert th["full"][n] >= 3 / 20.0 - 1e-9          # never under 3 queued vehicles (storage 20)
        assert th["low"][n] < th["full"][n]
    agent = ForecastAgent([ModelBundle(trained)])
    assert np.allclose(agent.full, [th["full"][n] for n in NODES])


def test_event_metrics_scores_each_alert_in_its_own_run():
    from goflow_prediction.metrics import event_metrics

    t = np.array([30.0, 60, 90, 30, 60, 90])            # two runs with the same SUMO times
    run = np.array([0, 0, 0, 1, 1, 1])
    fill = np.array([[0.1], [0.1], [0.1], [0.1], [0.9], [0.9]])
    om = np.zeros((6, 1), bool)
    alert_run0 = {"t": 30.0, "node": "a-1", "horizon_s": 60, "row": 0}
    alert_run1 = {"t": 30.0, "node": "a-1", "horizon_s": 60, "row": 3}
    assert event_metrics([alert_run0], om, t, ["a-1"], fill, 0.85, 30.0, run=run)["precision"] == 0.0
    assert event_metrics([alert_run1], om, t, ["a-1"], fill, 0.85, 30.0, run=run)["precision"] == 1.0


def test_merge_keeps_runs_apart_and_parts_selectable(tmp_path):
    from conftest import toy_dataset
    from goflow_prediction.data import merge, save_dataset

    a, b = toy_dataset([101, 11]), toy_dataset([102, 12])
    a.meta["clock_speed"] = b.meta["clock_speed"] = 60.0
    save_dataset(a, tmp_path / "a.npz")
    save_dataset(b, tmp_path / "b.npz")
    merge([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "m.npz")
    m = load_dataset(tmp_path / "m.npz")
    assert m.T == a.T + b.T and sorted(set(m.run)) == [0, 1, 2, 3]
    assert m.meta["clock_speed"] == 60.0 and len(m.meta["parts"]) == 2
    one = m.subset(np.flatnonzero(m.part == 1))
    assert sorted(set(one.seed)) == [12, 102] and np.allclose(one.x["halt"], b.x["halt"])
    with pytest.raises(FileExistsError):
        merge([tmp_path / "a.npz"], tmp_path / "m.npz")


def test_long_queue_start_needs_a_cycle_under_then_over():
    """congestion.starts: over its level after a whole cycle under it, from under 0.6 x the level 4 steps before."""
    from goflow_prediction import congestion as cg

    fill = np.array([[0.1]] * 10 + [[0.9]] * 6 + [[0.1]] * 6 + [[0.55]] * 6)   # one approach, one run
    run = np.zeros(len(fill), int)
    c = cg.cycle_means(fill, run)
    assert np.isnan(c[2, 0]) and c[3, 0] == pytest.approx(0.1)
    s = cg.starts(c, run, np.array([0.5]))
    # c(11) = (0.1 + 0.1 + 0.9 + 0.9) / 4 = 0.5 reaches the level; the slow rise to 0.55 at the end is a start too
    # (under the level for a cycle, at 0.1 four steps before); the first step of each jump is not (c still under)
    assert list(np.flatnonzero(s[:, 0])) == [11, 25]
    assert not cg.starts(c, run, np.array([0.6]))[25, 0]    # 0.55 never reaches 0.6


def test_long_queue_classifier_makes_the_alerts(toy, trained, tmp_path):
    """7 Nov: with congestion.json next to the model, the alerts come from the classifier ("long_queue_soon", with its
    probability) and every forecast carries each approach's chance of a long queue; on the toy jams (one starts at
    f-1 and spills upstream every 2 steps) it catches most of them on the dev seeds."""
    import shutil

    from goflow_prediction import congestion as cg
    from goflow_prediction.agent import ForecastAgent, ModelBundle, replay
    from goflow_prediction.data import load_dataset

    model = tmp_path / "with_congestion"
    shutil.copytree(trained, model)
    card = cg.train(str(toy / "toy.npz"), str(model), list(range(101, 107)), [11, 12])
    assert set(card["level"]) == set(NODES) and card["k_steps"] == cg.K and card["train_positives"] > 0
    with pytest.raises(ValueError):
        cg.train(str(toy / "toy.npz"), str(model), [101, 11], [11])
    agent = ForecastAgent([ModelBundle(model)])
    assert agent.congestion is not None
    ds = load_dataset(toy / "toy.npz")
    alerts, fc = replay(agent, ds, [11, 12])
    assert alerts and {a["kind"] for a in alerts} == {"long_queue_soon"}
    a = alerts[0]
    assert a["horizon_s"] == cg.K * 30 and a["p"] >= cg.THRESHOLD and "longer green" in a["action"]
    f = next(iter(fc.values()))
    assert f["long_queue_s"] == cg.K * 30 and all(0.0 <= v["p_long"] <= 1.0 for v in f["nodes"].values())
    dev = card["dev_scores_by_part"]["0"]["by_threshold"]
    at = next(e for e in dev if e["threshold"] == cg.THRESHOLD)
    assert at["starts"] > 0 and at["recall"] >= 0.5
    # the plain model folder (no congestion.json): the first rule, as before
    assert ForecastAgent([ModelBundle(trained)]).congestion is None

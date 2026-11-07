"""cosim/goflow_prediction.py (the DSO4 / GO-FLOW connector), with no SUMO, no CARLA and none of the agents' packages:
a fake TraCI connection, and fake agent processes written here."""
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from cosim import goflow_prediction as gp
from cosim.sumo_side import StepResult

ROOT = Path(__file__).resolve().parents[1]
NODES = [{"node": "1-1", "tls": "1", "edge": "e1", "lanes": ["e1_0", "e1_1"], "storage": 10.0},
         {"node": "2-1", "tls": "2", "edge": "e2", "lanes": ["e2_0"], "storage": 5.0}]


class FakeConn:
    """Just what Aggregator calls: lane subscriptions and the signals' controlled links."""

    def __init__(self):
        import traci.constants as tc

        self.tc = tc
        self.values = {}
        self.lane = self
        self.trafficlight = self

    def subscribe(self, lane, vars_):
        self.values.setdefault(lane, {v: 0 for v in vars_})

    def getAllSubscriptionResults(self):
        return self.values

    def getControlledLinks(self, tls):
        return [[("e1_0", "x", "")], [("e1_1", "x", "")], [("other", "x", "")]] if tls == "1" else [[("e2_0", "y", "")]]

    def set(self, lane, veh, halt, speed, occ):
        tc = self.tc
        self.values[lane] = {tc.LAST_STEP_VEHICLE_NUMBER: veh, tc.LAST_STEP_VEHICLE_HALTING_NUMBER: halt,
                             tc.LAST_STEP_MEAN_SPEED: speed, tc.LAST_STEP_OCCUPANCY: occ}


def test_aggregator_means_over_the_window():
    c = FakeConn()
    agg = gp.Aggregator(c, NODES, dt_s=3.0)
    c.set("e1_0", 2, 1, 4.0, 0.2)
    c.set("e1_1", 4, 3, 6.0, 0.4)
    c.set("e2_0", 1, 0, 10.0, 0.1)
    tls = {"1": "Grr", "2": "r"}
    rows = [agg.sample(c, t, tls) for t in (1.0, 1.5, 2.0, 3.0, 4.0)]
    assert rows[1] is None                       # half seconds are not sampled
    row = rows[-1]                               # window 1..4 s: samples at 1, 2, 3, 4
    assert row["t"] == 4.0
    assert row["halt"] == [4.0, 0.0] and row["veh"] == [6.0, 1.0]
    assert row["speed"] == [5.0, 10.0] and row["occ"] == pytest.approx([0.3, 0.1])
    assert row["green"] == [0.5, 0.0]            # 1 of the 2 links of e1's lanes is green


def test_obs_message_passes_dso4_checks():
    sys.path.insert(0, str(ROOT / "agents" / "goflow_prediction"))
    from goflow_prediction.schema import check_in

    row = {"t": 30.0, "halt": [1.0, float("nan")], "veh": [2.0, 0.0], "speed": [3.0, 0.0], "occ": [0.1, 0.0],
           "green": [0.5, float("nan")]}
    msg = gp.obs_message(row, NODES, 0, 17 * 3600.0, 60.0)
    assert check_in(msg) == []
    assert msg["nodes"]["2-1"]["halt"] is None   # NaN becomes null, never a non-JSON number
    assert set(msg["demand_at"]) == {"60", "120", "300"} and msg["demand"] > 0.9


FAKE_OK = ("import sys, json\n"
           "print(json.dumps({'type':'ready','v':1}), flush=True)\n"
           "for line in sys.stdin:\n"
           "    m = json.loads(line)\n"
           "    if m.get('type') == 'obs':\n"
           "        print(json.dumps({'type':'forecast','v':1,'seq':m['seq'],'t':m['t'],'nodes':{}}), flush=True)\n")
FAKE_CRASH = "import json\nprint(json.dumps({'type':'ready','v':1}), flush=True)\nraise SystemExit(3)\n"


def child(tmp_path, code, got):
    script = tmp_path / "fake_agent.py"
    script.write_text(code, encoding="utf-8")
    return gp.Child("fake", Path(sys.executable), tmp_path, [str(script)], tmp_path, lambda ch, m: got.append(m))


def wait(cond, s=10.0):
    end = time.time() + s
    while time.time() < end and not cond():
        time.sleep(0.05)
    return cond()


def test_child_answers(tmp_path):
    got = []
    ch = child(tmp_path, FAKE_OK, got)
    assert wait(lambda: ch.state == "ready")
    assert ch.send({"type": "obs", "v": 1, "seq": 7, "t": 30.0})
    assert wait(lambda: any(m.get("seq") == 7 for m in got))
    ch.close()


def test_child_crash_restart_then_off(tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "RESTART_S", (0.1, 0.1, 0.1))
    ch = child(tmp_path, FAKE_CRASH, [])
    for _ in range(40):
        ch.tick()
        if ch.state == "off":
            break
        time.sleep(0.25)
    assert ch.state == "off" and ch.failures == 4
    assert "exited with code 3" in ch.error
    assert ch.send({"type": "obs", "v": 1}) is False   # nothing is sent to a child that is off


def test_no_agent_venv_the_town_goes_on(tmp_path):
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps(NODES), encoding="utf-8")
    conn = FakeConn()
    fp = gp.GoflowPrediction(conn, model_dirs=[str(tmp_path)], goflow_model=str(tmp_path), nodes_file=nodes,
                             python_dso4=tmp_path / "none.exe", python_goflow=tmp_path / "none.exe",
                             out_dir=tmp_path / "live", log_dir=tmp_path, dt_s=1.0)
    assert fp.dso4.state == "off" and "setup_goflow_prediction" in fp.dso4.error
    t0 = time.perf_counter()
    for k in range(200):                       # 100 simulated seconds: never waits, never raises
        fp.on_step(StepResult(time=k * 0.5, vehicles={}, arrived=[], tls={"1": "Grr", "2": "r"}), conn, 17 * 3600.0)
    assert time.perf_counter() - t0 < 2.0
    assert "DSO4 OFFLINE" in fp.status() and "GO-FLOW OFFLINE" in fp.status()
    assert fp.recent_alerts(100.0) == []
    fp.close()


def test_flag_off_by_default():
    from cosim.freefly import parse_args

    a = parse_args([])
    assert a.goflow_prediction is False and a.goflow_traffic is False
    b = parse_args(["--goflow-prediction", "--goflow-traffic"])
    assert b.goflow_prediction and b.goflow_traffic


FAKE_COUNTER = ("import sys, json, os\n"
                "print(json.dumps({'type':'ready','v':1}), flush=True)\n"
                "seen = 0\n"
                "for line in sys.stdin:\n"
                "    m = json.loads(line)\n"
                "    if m.get('type') != 'obs':\n"
                "        continue\n"
                "    seen += 1\n"
                "    if seen >= 3:\n"
                "        print(json.dumps({'type':'forecast','v':1,'seq':m['seq'],'t':m['t'],'nodes':{},'seen':seen}),"
                " flush=True)\n"
                "    if m['seq'] == 4 and not os.path.exists(MARK):\n"   # crashes once (the resent history has step 4)
                "        open(MARK, 'w').close()\n"
                "        raise SystemExit(1)\n")


def test_restarted_child_gets_its_history_back(tmp_path, monkeypatch):
    """After a crash the child is sent its last steps again: it forecasts at once, and its answers on those old
    steps are not used twice."""
    monkeypatch.setattr(gp, "RESTART_S", (0.1, 0.1, 0.1))
    script = tmp_path / "fake.py"
    script.write_text(f"MARK = {str(tmp_path / 'crashed')!r}\n" + FAKE_COUNTER, encoding="utf-8")
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps(NODES), encoding="utf-8")
    conn = FakeConn()
    fp = gp.GoflowPrediction(conn, model_dirs=[], nodes_file=nodes, python_dso4=Path(sys.executable),
                             out_dir=tmp_path / "live", log_dir=tmp_path, dt_s=1.0, dso4_args=[str(script)])
    assert wait(lambda: fp.dso4.state == "ready")
    for k in range(20):                    # one obs per simulated second
        fp.on_step(StepResult(time=float(k), vehicles={}, arrived=[], tls={"1": "Grr", "2": "r"}), conn, 3600.0)
        time.sleep(0.15)
    assert wait(lambda: fp.dso4.failures == 1 and fp.dso4.state == "ready")
    assert wait(lambda: fp.latest is not None and fp.latest["seq"] >= 15)
    with open(tmp_path / "live" / "forecasts.jsonl", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    seqs = [m["seq"] for m in lines]
    assert len(seqs) == len(set(seqs)), "an old step was answered twice after the restart"
    after = [m for m in lines if m["seq"] > 4]
    assert after and after[0]["seen"] >= 3   # the restarted child had history at once (no 3-step warm-up)
    fp.close()


def test_refuses_a_clock_speed_the_models_were_not_trained_at(tmp_path):
    m = tmp_path / "model"
    m.mkdir()
    (m / "model_card.json").write_text(json.dumps({"dataset_meta": {"clock_speed": 60.0}}), encoding="utf-8")
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps(NODES), encoding="utf-8")
    assert gp.trained_clock_speed(m) == 60.0
    with pytest.raises(ValueError, match="--clock-speed 60"):
        gp.GoflowPrediction(FakeConn(), model_dirs=[str(m)], nodes_file=nodes, clock_speed=1.0,
                            out_dir=tmp_path / "live", log_dir=tmp_path)


def test_clean_checkout_uses_shipped_model_bundle(monkeypatch, tmp_path):
    shipped = ROOT / "models" / "goflow"
    dst = tmp_path / "models" / "goflow"
    shutil.copytree(shipped, dst)
    assert not (tmp_path / "out").exists()
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    monkeypatch.setattr(gp, "SHIPPED", dst)
    monkeypatch.setattr(gp, "DEFAULT_GRAPH", dst / "graph")
    dso4, goflow = gp.default_models(with_goflow=True)
    assert dso4 == [str(dst / "dso4"), str(dst / "dso4_goflow")]
    assert goflow == str(dst / "goflow_xgb")
    nodes = dst / "graph" / "nodes.json"
    assert nodes.is_file()
    fp = gp.GoflowPrediction(FakeConn(), model_dirs=[str(dst / "dso4")], goflow_model=str(dst / "goflow_xgb"),
                             nodes_file=nodes, python_dso4=tmp_path / "none.exe", python_goflow=tmp_path / "none.exe",
                             out_dir=tmp_path / "live", log_dir=tmp_path)
    assert fp.nodes and all("lanes" in n for n in fp.nodes)
    assert fp.nodes[0]["node"] == "1427-1"
    fp.close()


def test_pills_have_their_own_option():
    from cosim.freefly import parse_args

    assert parse_args(["--goflow-prediction"]).goflow_hud is False
    assert parse_args(["--goflow-prediction", "--goflow-hud"]).goflow_hud is True


def _offline(tmp_path, demand_scale=1.0):
    nodes = tmp_path / "nodes.json"
    nodes.write_text(json.dumps(NODES), encoding="utf-8")
    (tmp_path / "thresholds.json").write_text(json.dumps({"full": {"1-1": 0.5, "2-1": 0.8}}), encoding="utf-8")
    conn = FakeConn()
    fp = gp.GoflowPrediction(conn, model_dirs=[str(tmp_path)], goflow_model=str(tmp_path), nodes_file=nodes,
                             python_dso4=tmp_path / "none.exe", python_goflow=tmp_path / "none.exe",
                             out_dir=tmp_path / "live", log_dir=tmp_path, dt_s=1.0, demand_scale=demand_scale)
    return fp, conn


def test_at_risk_ranks_by_forecast_against_full_with_goflow_state(tmp_path):
    """7 Nov, the overlay's panel: the approaches closest to full in 2 minutes, queue now and then, GO-FLOW's state."""
    fp, conn = _offline(tmp_path)
    assert fp.full == {"1-1": 0.5, "2-1": 0.8}
    assert fp.at_risk() == [] and fp.warm_up_left_s() == fp.n_in * 1.0
    conn.set("e1_0", 3, 2, 1.0, 0.3)
    for k in range(4):
        fp.on_step(StepResult(time=k * 0.5, vehicles={}, arrived=[], tls={"1": "Grr", "2": "r"}), conn, 17 * 3600.0)
    assert fp.steps == 1 and fp.obs_latest["nodes"]["1-1"]["halt"] == 2.0
    fp.latest = {"nodes": {"1-1": {"fill": [0.2, 0.3, 0.4], "halt": [2.0, 3.0, 4.0]},
                           "2-1": {"fill": [0.6, 0.6, 0.6], "halt": [3.0, 3.0, 3.0]}}}
    fp.goflow_latest = {"nodes": {"2-1": {"traffic_state": "congested", "risk_level": "high"}}}
    rows = fp.at_risk()
    assert [r["node"] for r in rows] == ["2-1", "1-1"]          # 0.6 / 0.8 = 0.75 before 0.3 / 0.5 = 0.6
    assert rows[0]["risk"] == pytest.approx(0.75) and rows[0]["state"] == "congested"
    assert rows[1]["queue_now"] == 2.0 and rows[1]["queue_then"] == 3.0 and rows[1]["horizon_s"] == 120
    assert fp.states() == {"2-1": "congested"} and fp.warm_up_left_s() is None
    fp.close()


def test_a_clock_jump_says_warming_up_again(tmp_path):
    """Rayan's F6 / F7 reset left the overlay on "DSO4 ok" while the agents warmed up again (seen in CARLA, 7 Nov)."""
    fp, _ = _offline(tmp_path)
    fp.latest, fp.goflow_latest, fp.steps = {"nodes": {}}, {"nodes": {}}, 20
    fp.dso4.state = fp.goflow.state = "ready"
    assert fp.status() == "DSO4 ok  ·  GO-FLOW ok"
    fp.reset(100.0)
    assert fp.status() == "DSO4 warming up  ·  GO-FLOW warming up"
    assert fp.warm_up_left_s() == fp.n_in * 1.0 and fp.at_risk() == []
    fp.dso4.state = fp.goflow.state = "off"
    fp.close()


def test_demand_scale_reaches_the_agents(tmp_path):
    """--demand-scale: the agents are told the demand they were trained with (tools/goflow_prediction_dataset --scale)."""
    from cosim.freefly import parse_args

    assert parse_args([]).demand_scale == 1.0 and parse_args(["--demand-scale", "3"]).demand_scale == 3.0
    fp, conn = _offline(tmp_path, demand_scale=3.0)
    for k in range(4):
        fp.on_step(StepResult(time=k * 0.5, vehicles={}, arrived=[], tls={"1": "Grr", "2": "r"}), conn, 17 * 3600.0)
    one = gp.obs_message({"t": 1.0, **{c: [0.0, 0.0] for c in ("halt", "veh", "speed", "occ", "green")}}, NODES, 0,
                         17 * 3600.0, 60.0)
    assert fp.obs_latest["demand"] == pytest.approx(3.0 * one["demand"])
    fp.close()


def test_panel_and_chips_draw(tmp_path, monkeypatch):
    """The panel (always) and the chips (--goflow-hud) render, headless."""
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    from cosim.goflow_prediction_hud import ForecastHud

    fp, _ = _offline(tmp_path)
    fp.latest = {"nodes": {"1-1": {"fill": [0.2, 0.4, 0.4], "halt": [2.0, 4.0, 4.0]}}}
    fp.goflow_latest = {"nodes": {"1-1": {"traffic_state": "severe", "risk_level": "high"}}}
    fp.alerts.append({"type": "alert", "kind": "spillback_soon", "node": "1-1", "tls": "1", "t": 90.0,
                      "horizon_s": 120, "fill_now": 0.2, "pred_fill": 0.6})

    class V:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class TF:                                   # looking along +x from 30 m back, level
        location = V(-30.0, 0.0, 5.0)

        def get_forward_vector(self):
            return V(1.0, 0.0, 0.0)

        def get_right_vector(self):
            return V(0.0, 1.0, 0.0)

        def get_up_vector(self):
            return V(0.0, 0.0, 1.0)

    screen = pygame.Surface((1280, 720))
    hud = ForecastHud()
    rects = hud.draw(screen, (1280, 720), TF(), {"1": (0.0, 0.0, 0.0)}, {"1-1": (0.0, 0.0, 0.0)}, fp, 100.0, True)
    assert len(rects) == 3                      # a chip, the alert's pill, the panel
    panel = rects[-1]
    assert panel.right == 1280 - 16 and panel.bottom == 720 - 16 and panel.width >= 330
    assert len(hud.draw(screen, (1280, 720), TF(), {"1": (0.0, 0.0, 0.0)}, {"1-1": (0.0, 0.0, 0.0)}, fp, 100.0,
                        False)) == 1           # without --goflow-hud: the panel only
    fp.close()
    pygame.quit()


def test_chips_are_few_and_on_screen(monkeypatch):
    """CARLA, 7 Nov: seven chips round one junction, several for queues of 1-3 cars, one cut by the screen's edge.
    Now the 4 most at risk, a queue of 3 vehicles or more (or severe), inside the picture."""
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    from cosim.goflow_prediction_hud import MAX_CHIPS, ForecastHud, _full

    class V:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class TF:
        location = V(-30.0, 0.0, 5.0)

        def get_forward_vector(self):
            return V(1.0, 0.0, 0.0)

        def get_right_vector(self):
            return V(0.0, 1.0, 0.0)

        def get_up_vector(self):
            return V(0.0, 0.0, 1.0)

    rows = [{"node": f"n{i}", "risk": 0.9 - i * 0.05, "queue_now": 6.0, "queue_then": 7.0, "horizon_s": 120,
             "state": "congested"} for i in range(7)]
    rows.append({"node": "small", "risk": 0.95, "queue_now": 1.0, "queue_then": 2.0, "horizon_s": 120, "state": "congested"})
    anchors = {r["node"]: (0.0, -24.0 + i * 1.0, 0.0) for i, r in enumerate(rows)}   # n0 near the left edge
    screen = pygame.Surface((1280, 720))
    rects = ForecastHud().draw_chips(screen, (1280, 720), TF(), anchors, rows)
    assert len(rects) == MAX_CHIPS == 4
    assert all(r.left >= 8 and r.right <= 1280 - 8 for r in rects)
    assert _full(1.4) == "full" and _full(0.68) == "68 % full"
    pygame.quit()


def test_long_queue_chance_ranks_the_roads_and_titles_the_alert(tmp_path, monkeypatch):
    """7 Nov: with DSO4's long-queue classifier, the panel ranks by its chance; the alert says it plainly."""
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    from cosim.goflow_prediction_hud import ForecastHud

    fp, _ = _offline(tmp_path)
    fp.latest = {"long_queue_s": 180,
                 "nodes": {"1-1": {"fill": [0.2, 0.45, 0.4], "halt": [2.0, 4.0, 4.0], "p_long": 0.2},
                           "2-1": {"fill": [0.1, 0.1, 0.1], "halt": [1.0, 1.0, 1.0], "p_long": 0.8}}}
    rows = fp.at_risk()
    assert [r["node"] for r in rows] == ["2-1", "1-1"]       # by chance, though 1-1 is nearer full
    assert rows[0]["p_long"] == 0.8 and rows[0]["long_s"] == 180
    a = {"kind": "long_queue_soon", "horizon_s": 180, "node": "2-1", "p": 0.81, "queue_now": 4.0, "t": 90.0,
         "goflow_state": "congested"}
    hud = ForecastHud()
    assert hud.title(a) == "LONG QUEUE LIKELY WITHIN 3 MIN"
    assert "81 % chance" in hud.subtitle(a, 100.0) and "congested" in hud.subtitle(a, 100.0)
    p1 = hud.panel("DSO4 ok", None, rows, a, 100.0, 1)
    assert p1.get_width() >= 330
    fp.close()
    pygame.quit()


def test_the_panel_is_rendered_once_while_it_says_the_same(tmp_path, monkeypatch):
    import os

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.init()
    from cosim.goflow_prediction_hud import ForecastHud

    fp, _ = _offline(tmp_path)
    fp.latest = {"nodes": {"1-1": {"fill": [0.2, 0.4, 0.4], "halt": [2.0, 4.0, 4.0]}}}
    hud = ForecastHud()
    made = []
    real = hud.panel
    monkeypatch.setattr(hud, "panel", lambda *a: made.append(1) or real(*a))

    class TF:
        location = type("L", (), {"x": 0.0, "y": 0.0, "z": 0.0})()

    screen = pygame.Surface((1280, 720))
    for _ in range(5):
        hud.draw(screen, (1280, 720), TF(), {}, {}, fp, 100.0, False)
    assert len(made) == 1
    fp.latest = {"nodes": {"1-1": {"fill": [0.2, 0.6, 0.4], "halt": [2.0, 6.0, 4.0]}}}
    hud.draw(screen, (1280, 720), TF(), {}, {}, fp, 100.0, False)
    assert len(made) == 2
    fp.close()
    pygame.quit()


def test_restart_errors_do_not_nest(tmp_path, monkeypatch):
    """CARLA, 7 Nov: "exited with code 3221226505: exited with code 3221226505: ..." after each restart."""
    monkeypatch.setattr(gp, "RESTART_S", (0.1, 0.1, 0.1))
    ch = child(tmp_path, FAKE_CRASH, [])
    for _ in range(40):
        ch.tick()
        if ch.state == "off":
            break
        time.sleep(0.25)
    assert ch.error.count("exited with code") == 1

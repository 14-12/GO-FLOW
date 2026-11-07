"""cosim/goflow_prediction.py (the DSO4 / GO-FLOW connector), with no SUMO, no CARLA and none of the agents' packages:
a fake TraCI connection, and fake agent processes written here."""
import json
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

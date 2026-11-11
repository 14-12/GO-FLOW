"""Omar's accident agent beside the viewer (cosim/incident_agent.py, cosim/incident_agent_proc.py) and its grading
(tools/eval_incident_agent.py), without CARLA, the GPU or his venv."""
import json
import queue
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import incident_agent_proc as iap  # noqa: E402
from cosim.incident_agent import IncidentAgent  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import eval_incident_agent as ev  # noqa: E402


class V:
    def __init__(self, *values):
        self.v = np.array([values], dtype=float)

    def __getitem__(self, i):
        return self

    def item(self):
        return float(self.v[0, 0])

    def cpu(self):
        return self

    def numpy(self):
        return self.v[0]


class Box:
    def __init__(self, tid, cls, x1, y1, x2, y2, conf=0.5):
        self.id = None if tid is None else V(tid)
        self.cls, self.conf, self.xyxy = V(cls), V(conf), V(x1, y1, x2, y2)


def test_rows_are_his_result_rows():
    boxes = [Box(3, 0, 10, 20, 30, 60, 0.8), Box(None, 2, 0, 0, 50, 50), Box(4, 7, 100, 100, 140, 130, 0.4)]
    r = type("R", (), {"boxes": boxes})
    frame, dets = iap.rows_of(r, 960 * 540, {0: "car", 7: "car"})
    assert frame == {"vehicle_count": 2, "bbox_coverage": pytest.approx((20 * 40 + 40 * 30) / (960 * 540)),
                     "mean_detection_confidence": pytest.approx(0.6)}
    assert dets[0] == [3, 0, "car", pytest.approx(0.8), 20.0, 40.0, 40.0, 800.0]
    assert dets[1][:3] == [4, 7, "car"] and len(dets) == 2  # the box without a track id is skipped
    empty = type("R", (), {"boxes": None})
    assert iap.rows_of(empty, 100.0, {}) == ({"vehicle_count": 0, "bbox_coverage": 0.0,
                                               "mean_detection_confidence": 0.0}, [])


def test_class_map_is_checked_against_the_detector():
    names = {0: "car", 1: "van", 5: "ambulance_lights", 8: "pedestrian"}
    assert iap.class_ids({"car": "car", "Van": "car", "ambulance_lights": "truck"}, names) == {0: "car", 1: "car",
                                                                                             5: "truck"}
    with pytest.raises(ValueError):
        iap.class_ids({"motorcycle": "motorcycle"}, names)


def test_the_shipped_class_map_fits_the_carla_detector():
    cfg = json.loads(iap.CONFIG.read_text(encoding="utf-8"))
    names = json.loads((ROOT / "models" / "carla-yolo26n-t05.json").read_text(encoding="utf-8"))["classes"]
    ids = iap.class_ids(cfg["class_map"], dict(enumerate(names)))
    assert set(ids.values()) <= {"car", "bus", "truck"} and 8 not in ids  # pedestrians are not counted


FAKE = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if m["op"] == "hello":
            print(json.dumps({"op": "ready", "model": {"version": "v1"}}), flush=True)
        elif m["op"] == "frame":
            if m["t"] < 0:
                sys.exit(3)
            print(json.dumps({"op": "window", "camera": m["cam"], "sim_start_s": m["t"], "sim_end_s": m["t"] + 5}),
                  flush=True)
        elif m["op"] == "stop":
            print(json.dumps({"op": "bye", "stats": {}}), flush=True)
            break
""")


def drain(q, until, timeout=10.0):
    got, t_end = [], time.time() + timeout
    while time.time() < t_end:
        try:
            got.append(q.get(timeout=0.1))
        except queue.Empty:
            pass
        if until(got):
            break
    return got


def test_child_is_fed_restarted_then_left_off(tmp_path, monkeypatch):
    script = tmp_path / "fake.py"
    script.write_text(FAKE, encoding="utf-8")
    monkeypatch.setattr(iap, "RESTART_S", (0.1, 0.1, 0.1))
    q = queue.Queue()
    c = iap.AgentChild(q, "det", tmp_path / "err.log", cmd=[sys.executable, "-u", str(script)])
    c.start()
    c.send(json.dumps({"op": "frame", "cam": "A", "t": 10.0}))
    got = drain(q, lambda g: any(k == "window" for k, _ in g))
    assert ("ready", {"model": {"version": "v1"}}) in got
    assert [b for k, b in got if k == "window"][0]["camera"] == "A"
    for _ in range(4):  # it dies on a negative time: restarted 3 times, then off
        c.send(json.dumps({"op": "frame", "cam": "A", "t": -1.0}))
        t_end = time.time() + 5
        while c.alive() and time.time() < t_end:
            time.sleep(0.05)
        t_end = time.time() + 5
        while not c.alive() and c.off is None and time.time() < t_end:
            c.check(time.time())
            time.sleep(0.05)
    assert c.off is not None and c.stats["restarts"] == 3
    c.send("x")
    assert c.stats["sent"] == 5  # nothing is queued once it is off


def test_child_queue_drops_rather_than_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(iap, "TO_AGENT_MAX", 3)
    c = iap.AgentChild(queue.Queue(), "det", tmp_path / "err.log", cmd=[sys.executable, "-c", "pass"])
    c.inbox = queue.Queue(maxsize=3)  # never started: nothing reads it
    for _ in range(5):
        c.send("{}")
    assert c.stats["sent"] == 3 and c.stats["dropped"] == 2


def agent(tmp_path):
    cams = {"A": SimpleNamespace(x=1.0), "B": SimpleNamespace(x=2.0)}
    a = IncidentAgent(None, cams, 960, 540, 75.0, lambda: 100.0, out_dir=tmp_path,
                      process=False)
    a.q_out, a.q_frames = queue.Queue(), queue.Queue(maxsize=1)
    return a


def test_verdicts_counted_and_noted_on_the_crash_moment_only(tmp_path):
    a = agent(tmp_path)
    for i, (v1, imp) in enumerate([(True, False), (True, False), (False, True), (False, True), (True, True)]):
        a.q_out.put(("window", {"camera": "A", "sim_start_s": 80.0 + 5 * i, "sim_end_s": 85.0 + 5 * i,
                                "incident_detected": v1, "impact_alarm": imp}))
    a.q_out.put(("ready", {"model": {"version": "v1", "sha256_12": "x"}, "detector": "d"}))
    notes = a.poll()
    assert len(notes) == 1 and "shadow: not reported" in notes[0]
    assert a.counts["v1_episodes"] == 2 and a.counts["impact_episodes"] == 1
    assert a.counts["v1_alarm_windows"] == 3 and a.counts["windows"] == 5 and a.ready
    rows = [json.loads(x) for x in (tmp_path / "windows.jsonl").read_text().splitlines()]
    assert rows[0]["received_sim_s"] == 100.0
    s = a.summary()
    assert s["verdict_lag_s"]["max"] == pytest.approx(15.0)
    assert "v1 alarm on 1" in a.status() and "crash moment 1" in a.status()


def test_pictures_dropped_when_the_detection_process_is_busy(tmp_path):
    a = agent(tmp_path)
    assert a.send("A", 1.0, b"x") and not a.send("A", 1.2, b"y")
    assert a.counts["sent"] == 1 and a.counts["dropped"] == 1 and a.summary()["dropped_share"] == 0.5
    a.q_out.put(("off", "test"))
    a.poll()
    assert a.off == "test" and not a.send("A", 1.4, b"z") and a.status().startswith("accident agent off")


def test_close_writes_the_summary(tmp_path):
    a = agent(tmp_path)
    out = a.close()
    assert json.loads((tmp_path / "summary.json").read_text())["counts"] == out["counts"]


def windows(cam, start, n, alarms=(), impact=()):
    return [{"camera": cam, "sim_start_s": start + 5 * i, "sim_end_s": start + 5 * i + 5,
             "incident_detected": i in alarms, "impact_alarm": i in impact,
             "received_sim_s": start + 5 * i + 6.0} for i in range(n)]


def test_grading_follows_his_rules():
    ws = windows("A", 0.0, 40, alarms={3, 4, 30}, impact={21}) + windows("B", 0.0, 40, alarms={10})
    visible = [{"camera": "A", "incident": 1, "from_sim_s": 101.0, "to_sim_s": 160.0},
               {"camera": "B", "incident": 2, "from_sim_s": 45.0, "to_sim_s": 60.0}]  # B: not counted (far)
    counted = [{"camera": "A", "incident": 1, "t0": 100.0}]
    ev.label(ws, visible, counted)
    assert [w["target"] for w in ws[19:33]] == [0] + [1] * 12 + [0]
    assert ws[40 + 10]["target"] == -1
    r = ev.score(ws, ev.CHANNELS["either"], {1: 100.0})
    assert r["events"] == 1 and r["caught"] == 1
    assert r["per_event"][0]["time_to_detect_s"] == 10.0 and r["per_event"][0]["lived_delay_s"] == 11.0
    # one false alarm (A 15-25 s); A's alarm at 150 s is on the crash; B's is on another wreck: neither right nor wrong
    assert r["false_alarms"] == 1 and r["normal_hours"] == pytest.approx((80 - 12 - 3) * 5 / 3600, abs=1e-3)
    v1 = ev.score(ws, ev.CHANNELS["v1"], {1: 100.0})
    assert v1["caught"] == 1 and v1["median_detect_s"] == 55.0 and v1["criteria_met"]["median_detect"] is False
    imp = ev.score(ws, ev.CHANNELS["impact"], {1: 100.0})
    assert imp["false_alarms"] == 0 and imp["median_detect_s"] == 10.0


def test_viewer_option_and_a_bad_camera(monkeypatch):
    from cosim import freefly

    a = freefly.parse_args(["--incident-agent", "2014-2 looking W;nowhere"])
    assert a.incident_agent == "2014-2 looking W;nowhere"
    assert freefly.parse_args([]).incident_agent == ""
    ff = type("FF", (), {"world": None, "sim_now": 0.0, "depicted_at": None})()
    assert freefly.start_incident_agent(ff, a) is None  # logged, the viewer goes on
    assert freefly.start_incident_agent(ff, freefly.parse_args([])) is None

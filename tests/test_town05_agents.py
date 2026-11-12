"""Steps 1 and 2 of the multi-agent plan: walk-aware max-pressure on Town05, the camera reports
(cosim/camera_reports.py) and the LLM client's failover (cosim/llm.py)."""

from __future__ import annotations

import math
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from cosim.camera_reports import (
    CameraReporter,
    Layout,
    ReportRules,
    SimulatedCameras,
    Tracker,
    _poisson,
    junction_summary,
    observed,
)
from cosim.controllers import WalkMaxPressure, greens
from cosim.perception import CamSpec, LaneCamera

ROOT = Path(__file__).resolve().parents[1]
TOWN05 = ROOT / "scenarios" / "town05"


# ----------------------------------------------------------------------------- camera reports


def _layout() -> Layout:
    """One 100x60 camera: lane A (index 1) in the lower left, lane B (index 2) lower right, a road
    strip the camera does not count in the middle, and off-road (a parking lot) at the top."""
    mask = np.zeros((60, 100), dtype=np.int32)
    mask[40:60, 0:40] = 1
    mask[40:60, 60:100] = 2
    road = mask > 0
    road[40:60, 40:60] = True
    cam = LaneCamera(CamSpec("J", "J", "K", 0, 0, 9, 0), 100, 60, 75.0, mask=mask, lanes=["A_0", "B_0"])
    return Layout([cam], ["J-1 looking E"], ["A_0", "B_0"], {"A_0": 0, "B_0": 0},
                  {"A_0": [("J", "in")], "B_0": [("J", "out")]}, {"A_0": [(0, 50)], "B_0": [(0, 50)]},
                  {"A_0": 50.0, "B_0": 50.0}, [road])


def test_tracker_keeps_ids_and_measures_standing_still():
    tr = Tracker()
    tr.update(0.0, [(10, 30, 20, 45)], ["car"])
    tr.update(0.5, [(10.5, 30, 20.5, 45)], ["car"])  # half a pixel: still
    tr.update(1.0, [(30, 30, 40, 45)], ["car"])      # 20 px away: no overlap, a new track
    ids = {t.id for t in tr.tracks}
    assert len(tr.tracks) == 2 and len(ids) == 2
    first = min(tr.tracks, key=lambda t: t.id)
    assert first.still_for(0.5) == 0.5


def test_parked_cars_never_count_as_a_queue():
    rep = CameraReporter(_layout(), 0, ReportRules(park_s=10.0, queue_s=1.0))
    for k in range(30):  # 15 s at 2 pictures a second
        t = k * 0.5
        rep.observe(t, [(5, 10, 15, 20),      # top: off the road, standing: parked
                        (10, 42, 20, 55),     # lane A (incoming), standing: queued
                        (70, 42, 80, 55 if k % 2 else 52),  # lane B (outgoing), moving
                        (45, 42, 55, 55)],    # the road strip nobody counts
                    ["car", "car", "van", "truck"])
    r = rep.report(15.0)
    assert r["parked"] == 1 and r["other_road"] == 1
    assert r["lanes"]["A_0"] == [1, 1] and r["in"] == 1 and r["q_in"] == 1
    assert r["lanes"]["B_0"][0] == 1 and r["out"] == 1
    assert observed([r]) == {"A_0": 1, "B_0": 1}
    assert junction_summary([r])["J"]["parked"] == 1


def test_a_report_counts_only_what_the_latest_picture_shows():
    rep = CameraReporter(_layout(), 0)
    rep.observe(0.0, [(10, 42, 20, 55)], ["car"])
    rep.observe(0.5, [(25, 42, 35, 55)], ["car"])  # jumped: the tracker starts a new track, the old one lingers
    r = rep.report(0.5)
    assert r["lanes"]["A_0"][0] == 1 and len(rep.tracker.tracks) == 2


def test_a_car_stopping_off_the_road_is_not_parked_at_once():
    rep = CameraReporter(_layout(), 0, ReportRules(park_s=10.0))
    for k in range(6):
        rep.observe(k * 0.5, [(5, 10, 15, 20)], ["car"])
    assert rep.report(2.5)["parked"] == 0


def test_siren_needs_votes():
    rep = CameraReporter(_layout(), 0, ReportRules(vote_frames=8, vote_need=3))
    names = ["ambulance", "ambulance_lights", "ambulance", "ambulance", "ambulance_lights"]
    for k, n in enumerate(names):
        rep.observe(k * 0.5, [(10, 42, 20, 55)], [n])
    e = rep.report(2.0)["ems"]
    assert e == [{"lane": "A_0", "votes": 2, "of": 5, "confirmed": False}]
    rep.observe(2.5, [(10, 42, 20, 55)], ["police_lights"])
    assert rep.report(2.5)["ems"][0]["confirmed"]
    for k in range(8):  # siren no longer seen in 8 pictures: still the same confirmed vehicle
        rep.observe(3.0 + 0.5 * k, [(10, 42, 20, 55)], ["ambulance"])
    assert rep.report(6.5)["ems"] == [{"lane": "A_0", "votes": 0, "of": 8, "confirmed": True}]


def test_layout_round_trip(tmp_path):
    lay = _layout()
    lay.save(tmp_path)
    back = Layout.load(tmp_path)
    assert back.names == lay.names and back.owner == lay.owner and back.visible == lay.visible
    assert (back.cams[0].mask == lay.cams[0].mask).all() and (back.road[0] == lay.road[0]).all()


def test_poisson_mean():
    import random

    rng = random.Random(3)
    xs = [_poisson(rng, 0.234) for _ in range(20000)]
    assert abs(sum(xs) / len(xs) - 0.234) < 0.02


class _FakeTC:
    """The few TraCI calls SimulatedCameras makes."""

    class lane:  # noqa: N801
        @staticmethod
        def getLastStepVehicleIDs(ln):  # noqa: N802
            return {"A_0": ["v1", "v2", "amb"], "B_0": ["v3", "m1"]}[ln]

    class vehicle:  # noqa: N801
        pos = {"v1": 10.0, "v2": 80.0, "amb": 20.0, "v3": 30.0, "m1": 5.0}

        @staticmethod
        def getVehicleClass(v):  # noqa: N802
            return "motorcycle" if v == "m1" else "passenger"

        @classmethod
        def getLanePosition(cls, v):  # noqa: N802
            return cls.pos[v]

        @staticmethod
        def getLength(v):  # noqa: N802
            return 4.5

        @staticmethod
        def getSpeed(v):  # noqa: N802
            return 0.0 if v in ("v1", "amb") else 8.0


def test_simulated_cameras_see_only_the_visible_stretch():
    sim = SimulatedCameras(_layout(), recall=1.0, false_per_frame=0.0, siren_recall=1.0, siren_false=0.0,
                           parked={"J-1": 4})
    reps = None
    t = 0.0
    while reps is None:
        reps = sim.step(_FakeTC, t, {"amb": True})
        t += 0.5
    r = reps[0]
    # v2 stands beyond the visible 0-50 m; the motorcycle is not a detector class
    assert r["lanes"] == {"A_0": [2, 2], "B_0": [1, 0]}
    assert r["parked"] == 4 and r["cam"] == "J-1"
    assert r["ems"] and r["ems"][0]["lane"] == "A_0"


def test_simulated_cameras_call_sirens_as_the_flash_model():
    """siren_model "flash": after FLASH_S in view, one call per vehicle at the flash model's rates, which then holds;
    the counts' random stream is the same draw for draw as with the vote."""
    from cosim import camera_reports as cr

    def run(model, on, seed):
        sim = SimulatedCameras(_layout(), seed=seed, siren_recall=1.0, siren_false=1.0, siren_model=model)
        out = []
        for k in range(12):
            reps = sim.step(_FakeTC, k * 0.5, {"amb": on})
            if reps is not None:
                out.append((reps[0]["lanes"], [e["confirmed"] for e in reps[0]["ems"]]))
        return out

    vote, flash = run("vote", False, 3), run("flash", False, 3)
    assert [x[0] for x in vote] == [x[0] for x in flash]          # the counts are untouched
    assert vote[-1][1] == [True]                                  # every picture a siren: the vote confirms
    calls = [run("flash", False, s)[-1][1] == [True] for s in range(200)]
    assert abs(sum(calls) / 200 - cr.FLASH_OFF["ambulance"]) < 0.06   # the flash model mostly says no
    calls = [run("flash", True, s)[-1][1] == [True] for s in range(200)]
    assert abs(sum(calls) / 200 - cr.FLASH_ON["ambulance"]) < 0.08


# ----------------------------------------------------------------------------- walk-aware max-pressure


@pytest.mark.slow
def test_walk_max_pressure_serves_pedestrians_safely():
    from cosim.sumo_side import SumoSide

    import xml.etree.ElementTree as ET

    def exclusive_walk(st: str) -> bool:  # every car red, the crossings (indexed last) walking
        return st.endswith("G") and set(st.rstrip("G")) == {"r"}

    if not any(exclusive_walk(p.get("state", "")) for p in ET.parse(TOWN05 / "town05.net.xml").iter("phase")):
        pytest.skip("no exclusive walk phase since 25 Oct (tools/build_town05_signals.py: walk with the traffic)")
    ctl = WalkMaxPressure(record=True)
    s = SumoSide(TOWN05, "max_pressure_walk", seed=3, port=8911, label="walk-test", policy=ctl, extras=True,
                 emergency_every=1e9, extra_args=["--scale", "0.4"])
    s.extras.config.pedestrian_trips = True
    try:
        while s.time < 900:
            s.step()
    finally:
        s.close()
    walks_shown = 0
    for j in ctl.junctions.values():
        walk = ctl._walks[j.tls]
        assert walk, j.tls  # every Town05 signal has its pedestrian phase
        allowed = [greens(p) for p in j.phases]
        log = j.log
        for _, state in log:
            assert any(greens(state) <= a for a in allowed), (j.tls, state)
        for (t0, s0), (t1, s1) in zip(log, log[1:], strict=False):
            for a, b in zip(s0, s1, strict=True):
                assert not (a in "Gg" and b == "r"), (j.tls, t1, s0, s1)  # green left through yellow
            if s0 in j.phases and greens(s0) <= set(ctl._cross[j.tls]) and s1 != s0:
                walks_shown += 1
                assert t1 - t0 >= ctl.walk_min_s - 1e-6, (j.tls, t0, t1)  # the walk lasts its 10 s
        for a, b, y, g in j.moves:
            if a in walk:
                assert g - y >= ctl.walk_clear_s - 1e-6, (j.tls, y, g)  # 5 s clearance after a walk
            else:
                assert g - y >= j.yellow_s - 1e-6
    assert walks_shown > 0


# ----------------------------------------------------------------------------- LLM failover


class _Handler(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.calls.append((self.path, body["model"], self.headers.get("Authorization", "")))
        if body["model"] == "busy":
            self.send_response(429)
            self.send_header("retry-after", "7")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit reached for requests per minute"}}')
            return
        if body["model"] == "strict" and body.get("response_format", {}).get("type") == "json_schema":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Failed to validate JSON", "code": "json_validate_failed"}}')
            return
        if body["model"] == "short" and body["max_tokens"] < 100:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "max completion tokens reached before generating a valid document"}}')
            return
        if body["model"] == "offschema":
            data = json.dumps({"choices": [{"message": {"content": '{"hotspots": ["2014"]}'}}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if body["model"] == "badkey":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "invalid api key gsk_abcdefghijklmnopqrstuvwxyz"}')
            return
        out = {"choices": [{"message": {"content": '```json\n{"ok": true}\n```'}}],
               "usage": {"total_tokens": 12}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_api(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    cfg = {"providers": [
        {"name": "a-nokey", "base_url": url, "model": "ok", "key_env": "TEST_KEY_MISSING"},
        {"name": "b-badkey", "base_url": url, "model": "badkey", "key_env": "TEST_KEY_B"},
        {"name": "c-busy", "base_url": url, "model": "busy", "key_env": "TEST_KEY_C"},
        {"name": "d-ok", "base_url": url, "model": "ok", "key_env": "TEST_KEY_D", "rpm": 2}]}
    (tmp_path / "llm.json").write_text(json.dumps(cfg))
    monkeypatch.delenv("TEST_KEY_MISSING", raising=False)
    for k in ("B", "C", "D"):
        monkeypatch.setenv(f"TEST_KEY_{k}", f"secret-{k}")
    _Handler.calls = []
    yield tmp_path
    srv.shutdown()


def test_llm_fails_over_to_the_next_provider(fake_api, caplog):
    from cosim.llm import LLM, NoProvider

    llm = LLM(fake_api / "llm.json", fake_api / "usage.json", env_file=fake_api / "none.env")
    r = llm.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert r.provider == "d-ok" and r.json() == {"ok": True}
    assert [c[1] for c in _Handler.calls] == ["badkey", "busy", "ok"]
    assert _Handler.calls[-1][2] == "Bearer secret-D"
    st = {s["provider"]: s for s in llm.status()}
    assert st["c-busy"]["cooling_down_s"] in (6, 7) and st["b-badkey"]["off"].startswith("HTTP 401")
    assert "gsk_abcdefghijkl" not in st["b-badkey"]["off"]  # key-like text is masked
    assert json.loads((fake_api / "usage.json").read_text())["calls"]["d-ok"] == 1
    # the busy and bad providers are not asked again; d-ok's 2 requests a minute are then used up
    llm.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert [c[1] for c in _Handler.calls] == ["badkey", "busy", "ok", "ok"]
    with pytest.raises(NoProvider):
        llm.chat([{"role": "user", "content": "hi"}], max_tokens=8, wait_s=0)
    assert all("secret-" not in rec.getMessage() for rec in caplog.records)


def test_llm_retries_plain_json_when_the_schema_is_refused(fake_api):
    from cosim.llm import LLM

    cfg = json.loads((fake_api / "llm.json").read_text())
    cfg["providers"] = [dict(cfg["providers"][3], name="e-strict", model="strict")]
    (fake_api / "strict.json").write_text(json.dumps(cfg))
    llm = LLM(fake_api / "strict.json", fake_api / "usage2.json", env_file=fake_api / "none.env")
    r = llm.chat([{"role": "user", "content": "hi"}], max_tokens=8,
                 json_schema={"name": "x", "schema": {"type": "object"}})
    assert r.provider == "e-strict" and r.json() == {"ok": True}
    assert [c[1] for c in _Handler.calls] == ["strict", "strict"]


def test_llm_gives_room_and_rejects_off_schema_replies(fake_api):
    from cosim.llm import LLM

    cfg = json.loads((fake_api / "llm.json").read_text())
    base = cfg["providers"][3]
    cfg["providers"] = [dict(base, name="f-off", model="offschema"), dict(base, name="g-short", model="short")]
    (fake_api / "two.json").write_text(json.dumps(cfg))
    llm = LLM(fake_api / "two.json", fake_api / "usage3.json", env_file=fake_api / "none.env")
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    r = llm.chat([{"role": "user", "content": "hi"}], max_tokens=50, json_schema={"name": "x", "schema": schema})
    # f-off answers JSON of the wrong shape: skipped; g-short refuses 50 tokens, then answers with 200
    assert r.provider == "g-short" and r.json() == {"ok": True}
    assert [c[1] for c in _Handler.calls] == ["offschema", "short", "short"]
    assert any("off schema" in e for e in llm.last_skipped)


# ----------------------------------------------------------------------------- step 3, version 2


def test_agent_names_switch_features_off():
    from cosim.agents_actuated import FEATURES, make

    a = make("agents_observed-skip-trickle")
    assert a.uses_cameras and a.name == "agents_observed-skip-trickle"
    assert a.on == set(FEATURES) - {"skip", "trickle"}
    assert "extend" not in a.on and "extend" in make("agents+extend").on  # optional since the ablation
    b = make("agents+batch-corridor")
    assert not b.uses_cameras and b.on == (set(FEATURES) - {"corridor", "trickle"}) | {"batch"}
    assert "trickle" not in make("agents").on and "trickle" in make("agents+trickle").on  # (off since 25 Oct)
    with pytest.raises(ValueError):
        make("agents-nonsense")


@pytest.mark.slow
def test_actuated_agents_keep_the_program_safe():
    """Replays every state the agents show (exact counts, x0.8, an emergency vehicle every 60 s) and checks
    the program's rules: only its own states; a car green ends through yellow; its all red after a car
    yellow; a car green lasts its minimum; the walk (a phase of its own again since 27 Oct) lasts its 7 s, then its
    clearance and at least the 2 s buffer before a car green. The clearance's length against each crossing is
    checked against SUMO's junction logic in tests/test_signal_programs.py."""
    from cosim.agents_actuated import ActuatedAgents
    from cosim.sumo_side import SumoSide

    ctl = ActuatedAgents(record=True)
    s = SumoSide(TOWN05, "agents", seed=2, port=8913, label="agents-v2-test", policy=ctl, extras=True,
                 emergency_every=60.0, extra_args=["--scale", "0.8"])
    s.extras.config.pedestrian_trips = True
    ex = s.extras
    ctl.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    shown: dict[str, list[tuple[float, str]]] = {t: [] for t in ctl.junctions}
    try:
        while s.time < 900:
            r = s.step()
            for t, st in r.tls.items():
                if t in shown and (not shown[t] or shown[t][-1][1] != st):
                    shown[t].append((r.time, st))
    finally:
        s.close()
    acted = sum(ctl.stats[k] for k in ("skipped", "walk_skipped", "ev_jumps", "ev_cuts"))
    assert acted > 0
    for t, log in shown.items():
        a = ctl.junctions[t]
        program = set(a.states)
        car = [i for i in range(len(a.states[0])) if i not in a.cross]
        for _, st in log:
            assert st in program, (t, st)
        red_since = None  # when every car link last became red after a car yellow
        walk_end = None   # when a walk that was shown ended
        for (t0, s0), (t1, s1) in zip(log, log[1:], strict=False):
            for i in car:
                assert not (s0[i] in "Gg" and s1[i] == "r"), (t, t1, s0, s1)
            k0 = a.states.index(s0)
            if a.kind[k0] == "car":
                assert t1 - t0 >= a.min_dur[k0] - 1e-6, (t, t0, t1, s0)
            if a.kind[k0] == "walk" and t1 - t0 > 0.5 + 1e-6:  # not skipped at its first step (WalkSkip)
                assert t1 - t0 >= 7.0 - 1e-6, (t, t0, t1)  # MUTCD 4I.06: 7 s
            k1 = a.states.index(s1)
            if a.kind[k0] == "yellow" and a.kind[k1] == "red":
                red_since, red_k = t1, k1
            if a.kind[k0] == "walk":
                walk_end = t1 if t1 - t0 > 0.5 + 1e-6 else None
            if a.kind[k1] == "car" and walk_end is not None:
                # its clearance (ended early once the crossings are empty: sumo_side.WalkSkip) and the 2 s buffer
                assert t1 - walk_end >= 2.0 - 1e-6, (t, walk_end, t1)
                walk_end = None
            if a.kind[k1] == "car" and a.kind[k0] == "red" and red_since is not None:
                # its all red (since 25 Oct from the junction's width and speed: 1.5-3 s)
                need = min(a.min_dur[i] for i, st in enumerate(a.states) if st == a.states[red_k])  # (identical
                assert t1 - red_since >= need - 1e-6, (t, red_since, t1)                       # all reds)


def test_an_emergency_request_counts_only_when_the_vehicle_can_move():
    """cosim.agents_actuated._can_pass: a full exit or a long queue ahead of the vehicle drops its request."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ActuatedAgents, _Agent

    lanes = {"in_0": ["ev", "a", "b"], "out_0": []}
    pos = {"ev": 10.0, "a": 40.0, "b": 50.0}
    halting = {"out_0": 0}
    tc = NS(vehicle=NS(getLaneID=lambda v: "in_0", getLanePosition=lambda v: pos[v], getSpeed=lambda v: 0.0),
            lane=NS(getLastStepVehicleIDs=lambda ln: lanes[ln], getLastStepHaltingNumber=lambda ln: halting[ln]))
    ag = ActuatedAgents(fallback=set())
    ag._len = {"out_0": 75.0}
    a = _Agent("J", ["G"], ["car"], [5.0], [60.0], [0], {}, {}, {}, {}, link_out={0: "out_0"})
    assert ag._can_pass(tc, a, "ev", 0)                      # two ahead, exit free
    halting["out_0"] = 9                                      # 75 m holds 10: 85 % full
    assert not ag._can_pass(tc, a, "ev", 0) and ag.stats["ev_exit_full"] == 1
    halting["out_0"] = 0
    lanes["in_0"] = ["ev"] + [f"q{i}" for i in range(9)]
    pos.update({f"q{i}": 20.0 + 7.5 * i for i in range(9)})
    assert not ag._can_pass(tc, a, "ev", 0) and ag.stats["ev_queued"] == 1  # 9 standing ahead


def test_agent_names_carry_settings():
    """make(): "@name=value" changes a setting for a tuning run; unknown settings are refused."""
    from cosim.agents_actuated import ActuatedAgents, make

    a = make("agents_observed+noflow@noflow_s=6@full_share=0.7-trickle")
    assert a.uses_cameras and "noflow" in a.on and "trickle" not in a.on
    assert a.noflow_s == 6.0 and a.full_share == 0.7 and ActuatedAgents.full_share == 0.85  # the class keeps its own
    with pytest.raises(ValueError):
        make("agents@nonsense=1")


def test_noflow_sees_a_green_under_which_nobody_moves():
    """cosim.agents_actuated._stalled: from the loops (a vehicle standing on one, none passing any), else from the
    counts (every vehicle on the waiting lanes standing, one may still roll up)."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ActuatedAgents, _Agent

    loop = {"d1": {"speed": -1.0, "occ": 100.0, "since": 0.0}, "d2": {"speed": -1.0, "occ": 0.0, "since": 30.0}}
    tc = NS(inductionloop=NS(getLastStepMeanSpeed=lambda d: loop[d]["speed"],
                             getLastStepOccupancy=lambda d: loop[d]["occ"],
                             getTimeSinceDetection=lambda d: loop[d]["since"]))
    ag = ActuatedAgents(fallback=set())
    a = _Agent("J", ["G", "r"], ["car", "red"], [5.0, 3.0], [60.0, 3.0], [0], {}, {0: [("in_0", "out_0")]}, {},
               {0: ["d1", "d2"]})
    a.green_since = 0.0
    assert ag._stalled(tc, a, 0, now=20.0)                     # one stands on d1, nobody passed d2 for 30 s
    loop["d2"]["since"] = 1.0
    assert not ag._stalled(tc, a, 0, now=20.0)                 # one passed d2 a second ago
    loop["d2"].update(since=30.0, speed=6.0)
    assert not ag._stalled(tc, a, 0, now=20.0)                 # one rolling over d2
    loop["d2"]["speed"] = -1.0
    loop["d1"].update(occ=0.0, since=30.0)
    assert not ag._stalled(tc, a, 0, now=20.0)                 # nobody on any loop: nothing to say
    a.loops = {}                                               # no loops: the counts (here SUMO's lanes)
    ag._cache = {"in_0": (5, 5)}
    assert ag._stalled(tc, a, 0, now=20.0)
    ag._cache = {"in_0": (5, 4)}
    assert ag._stalled(tc, a, 0, now=20.0)                     # one still rolling up at the back
    ag._cache = {"in_0": (5, 2)}
    assert not ag._stalled(tc, a, 0, now=20.0)


def test_agents_record_what_they_did():
    """ActuatedAgents.events: what the viewer draws (cosim/agent_hud.py): a skipped green names the skipped phases,
    a walk skipped alone has none."""
    from cosim.agents_actuated import ActuatedAgents, _Agent

    calls = []
    tc = __import__("types").SimpleNamespace(trafficlight=__import__("types").SimpleNamespace(
        setPhase=lambda t, p: calls.append((t, p)), getTime=lambda: 0.0),
        simulation=__import__("types").SimpleNamespace(getTime=lambda: 12.0))
    ag = ActuatedAgents(fallback=set())
    ag._now = 12.0
    states = ["GGrr", "yyrr", "rrrr", "rrGG", "rryy", "rrrr"]
    kinds = ["car", "yellow", "red", "car", "yellow", "red"]
    a = _Agent("J", states, kinds, [5.0] * 6, [60.0] * 6, [0, 3], {0: 5, 3: 2}, {0: [], 3: []}, {}, {})
    ag._spilled_now = set()
    ag._jump(tc, a, 3, 0)                                      # in the all red before 3: straight to 0
    e = ag.events[-1]
    assert calls == [("J", 5)] and e["kind"] == "skip" and e["phases"] == [3] and e["t"] == 12.0


def test_agent_hud_draws_decisions_where_they_happen(monkeypatch):
    """cosim/agent_hud.py: a pill over the junction, chevrons at the stop lines of the green it was about, the list top
    right; nothing for a decision older than SHOW_S or far away."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from types import SimpleNamespace as NS

    pygame.font.init()
    from cosim.agent_hud import AgentHud

    v = lambda x, y, z: NS(x=x, y=y, z=z)  # noqa: E731
    pitch = math.atan2(30.0, 50.0)  # 50 m back, 30 m up, looking at (0, 0, 0) along +x
    tf = NS(location=v(-50.0, 0.0, 30.0), get_forward_vector=lambda: v(math.cos(pitch), 0.0, -math.sin(pitch)),
            get_right_vector=lambda: v(0.0, 1.0, 0.0), get_up_vector=lambda: v(math.sin(pitch), 0.0, math.cos(pitch)))
    scr = pygame.Surface((1280, 720))
    hud = AgentHud()
    centres = {"1292": (0.0, 0.0, 0.0), "979": (2000.0, 0.0, 0.0)}
    stops = {"1292": {3: [(-12.0, -2.0, 0.0, 0.0)]}}
    evs = [{"t": 99.0, "tls": "1292", "kind": "noflow", "phases": [3], "text": "green ended: nobody could move"}]
    rects = hud.draw(scr, (1280, 720), tf, centres, stops, evs, 100.0)
    assert len(rects) == 3                                      # chevron, pill, list
    assert scr.get_at((rects[0].centerx, rects[0].centery))[:3] in ((249, 115, 22), (16, 17, 20))
    old = [dict(evs[0], t=50.0)]
    assert len(hud.draw(scr, (1280, 720), tf, centres, stops, old, 100.0)) == 1   # only the list
    far = [dict(evs[0], tls="979")]
    assert len(hud.draw(scr, (1280, 720), tf, centres, stops, far, 100.0)) == 1   # too far for a pill


def test_forecast_weighs_balance_and_extends_a_green_about_to_queue():
    """+forecast (cosim/agents_actuated.py): the forecasters' risk counts in balance (a green whose road is about to
    queue is kept; another about to queue ends the green sooner) and extends a green at its maximum; nothing without
    a fresh forecast."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ActuatedAgents, _Agent
    from cosim.goflow_prediction import long_queue_risk

    calls = []
    now = {"t": 100.0, "next": 130.0}
    tc = NS(trafficlight=NS(getNextSwitch=lambda tls: now["next"],
                            setPhaseDuration=lambda tls, d: calls.append(d)))
    ag = ActuatedAgents(fallback=set(), extra={"forecast"})
    assert "forecast" in ag.on and "forecast" not in ActuatedAgents(fallback=set()).on
    ag._len = {"in0": 75.0, "in2": 75.0, "out0": 75.0, "out2": 75.0}
    ag._waiting = {}
    a = _Agent("J", ["GGrr", "yyrr", "rrrr", "rrGG", "rryy", "rrrr"], ["car", "yellow", "red", "car", "yellow", "red"],
               [5.0, 3.0, 2.0, 5.0, 3.0, 2.0], [45.0, 3.0, 2.0, 45.0, 3.0, 2.0], [0, 3], {0: 5, 3: 2},
               {0: [("in0", "out0")], 3: [("in2", "out2")]}, {}, {})
    ag.served = {("J", 0): ["J-1"], ("J", 3): ["J-2"]}
    a.green_since, ag._now = 75.0, 100.0                        # 25 s of green: balance may act

    def go(counts, risk):
        calls.clear()
        ag._cache = {"in0": counts[0], "in2": counts[1], "out0": (0, 0), "out2": (0, 0)}
        ag._risk = risk
        ag._green(tc, a, 0, now["t"])
        return list(calls)

    # 1 of 10 queued here, 9 of 10 on the other road: balance ends this green; a long queue forecast here keeps it
    assert go([(1, 1), (9, 9)], {}) == [0]
    assert go([(1, 1), (9, 9)], {"J-1": 1.0}) == [] and ag.stats["forecast_balance_kept"] == 1
    # the other road 6 of 10: not enough alone; with a 0.6 risk there (0.6 + 0.5 x 0.6 >= 0.85) the green ends
    assert go([(1, 1), (6, 6)], {}) == []
    assert go([(1, 1), (6, 6)], {"J-2": 0.6}) == [0] and ag.stats["forecast_balance_cuts"] == 1
    # at its 45 s maximum: extended 5 s while its risk is high and above the other's, not otherwise
    a.green_since, now["next"] = 55.0, 100.5
    assert go([(5, 5), (2, 2)], {"J-1": 0.9, "J-2": 0.2}) == [5.0] and ag.stats["forecast_extensions"] == 1
    assert go([(5, 5), (2, 2)], {"J-1": 0.9, "J-2": 0.9}) == []
    assert go([(5, 5), (2, 2)], {"J-1": 0.5}) == []
    assert go([(5, 5), (2, 2)], {}) == []
    # what the agents read: p_long, 1 where long already; nothing from a stale forecast or one without the classifier
    fc = {"t": 90.0, "long_queue_s": 180, "nodes": {"J-1": {"p_long": 0.4}, "J-2": {"p_long": 0.0, "long_now": True}}}
    assert long_queue_risk(fc, 100.0) == {"J-1": 0.4, "J-2": 1.0}
    assert long_queue_risk(fc, 140.0) == {} and long_queue_risk(dict(fc, long_queue_s=None), 100.0) == {}


def test_gate_shortens_the_green_that_feeds_a_road_forecast_to_queue():
    """+gate (cosim/agents_actuated.py): a green whose movements lead to an approach with a long queue forecast is
    ended after gate_min_s, when someone else waits, its own road has room and is not itself forecast to queue."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ActuatedAgents, _Agent

    calls = []
    tc = NS(trafficlight=NS(getNextSwitch=lambda tls: 130.0, setPhaseDuration=lambda tls, d: calls.append(d)),
            inductionloop=NS(getTimeSinceDetection=lambda d: 1e9))
    ag = ActuatedAgents(fallback=set(), extra={"gate"})
    assert "gate" in ag.on and "gate" not in ActuatedAgents(fallback=set()).on
    ag._len = {"in0": 75.0, "in2": 75.0, "out0": 40.0, "out2": 75.0, "mid": 60.0, "k1": 50.0, "far": 500.0, "k2": 50.0}
    ag._pred, ag._up, ag._waiting = {}, {}, {}
    # the lanes going on: out0 -> mid -> k1 (approach K-1, 100 m on); out2 -> far (500 m) -> k2: too far to count
    ag._succ = {"out0": ["mid"], "mid": ["k1"], "out2": ["far"], "far": ["k2"]}
    node_of = {"k1": "K-1", "k2": "K-2", "in0": "J-1"}
    assert ag._reach_down("out0", node_of) == {"K-1"} and ag._reach_down("out2", node_of) == set()
    a = _Agent("J", ["GGrr", "yyrr", "rrrr", "rrGG", "rryy", "rrrr"], ["car", "yellow", "red", "car", "yellow", "red"],
               [5.0, 3.0, 2.0, 5.0, 3.0, 2.0], [45.0, 3.0, 2.0, 45.0, 3.0, 2.0], [0, 3], {0: 5, 3: 2},
               {0: [("in0", "out0")], 3: [("in2", "out2")]}, {}, {})
    ag.served = {("J", 0): ["J-1"], ("J", 3): ["J-2"]}
    ag.fed = {("J", 0): {"K-1": 1.0}}

    def go(counts, risk, spent=12.0):
        calls.clear()
        a.green_since, ag._now = 100.0 - spent, 100.0
        ag._cache = {"in0": counts[0], "in2": counts[1], "out0": (0, 0), "out2": (0, 0)}
        ag._risk = ag._gate_risk = risk
        ag._green(tc, a, 0, 100.0)
        return list(calls)

    assert go([(3, 3), (2, 2)], {"K-1": 0.8}) == [0] and ag.stats["gate_cuts"] == 1   # the road ahead is to queue
    assert go([(3, 3), (2, 2)], {"K-1": 0.5}) == []                 # under gate_p
    assert go([(3, 3), (2, 2)], {}) == []                           # no fresh forecast
    assert go([(3, 3), (2, 2)], {"K-1": 0.8}, spent=7.0) == []      # its 10 s first
    assert go([(3, 3), (0, 0)], {"K-1": 0.8}) == []                 # nobody else waiting: nothing gained
    assert go([(8, 8), (2, 2)], {"K-1": 0.8}) == []                 # its own road 8 of 10 full: the queue would move back
    assert go([(3, 3), (2, 2)], {"K-1": 0.8, "J-1": 0.9}) == []     # its own road forecast to queue too
    assert ag.stats["gate_cuts"] == 1 and ag.events[-1]["kind"] == "gate"
    # +accgate: an accident on the road ahead counts as that risk, with no forecast at all
    ag3 = ActuatedAgents(fallback=set(), extra={"accgate"})
    ag3._len, ag3._pred, ag3._up, ag3._waiting, ag3.served, ag3.fed = ag._len, {}, {}, {}, ag.served, ag.fed
    ag3._succ, ag3._node_of, ag3._nodes_on = ag._succ, node_of, {"kroad": ["K-1"]}
    assert ag3._nodes_of_road("kroad") == ("K-1",) and ag3._nodes_of_road("mid") == ("K-1",)
    ag3._now, ag3._gate_risk = 100.0, {"K-1": 1.0}
    ag3._cache = {"in0": (3, 3), "in2": (2, 2), "out0": (0, 0), "out2": (0, 0)}
    calls.clear()
    a.green_since = 88.0
    ag3._green(tc, a, 0, 100.0)
    assert calls == [0] and ag3.stats["accgate_cuts"] == 1 and "accident" in ag3.events[-1]["text"]
    # without +gate the same forecast changes nothing
    ag2 = ActuatedAgents(fallback=set())
    ag2._len, ag2._pred, ag2._up, ag2._waiting, ag2.served, ag2.fed = ag._len, {}, {}, {}, ag.served, ag.fed
    ag2._now, ag2._risk = 100.0, {"K-1": 0.8}
    ag2._cache = {"in0": (3, 3), "in2": (2, 2), "out0": (0, 0), "out2": (0, 0)}
    calls.clear()
    ag2._green(tc, a, 0, 100.0)
    assert calls == []


def test_bus_priority_holds_its_green_and_may_shorten_the_others():
    """+bus (cosim/agents_actuated.py): a line bus near its signal asks; its green is held past the maximum by at
    most bus_extend_s; with bus_cut the other greens end after bus_cut_min_s unless their road is full. Not at its
    stop, not with a stop still before the line, not behind a long standing queue, not for ever."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ActuatedAgents, _Agent

    calls = []
    bus = {"dist": 80.0, "stopped": False, "stops": [], "pos": 10.0, "ahead": 0}
    lane_ids = lambda lane: ["bus1"] + [f"c{i}" for i in range(bus["ahead"])]  # noqa: E731
    tc = NS(trafficlight=NS(getNextSwitch=lambda tls: 100.5, setPhaseDuration=lambda tls, d: calls.append(d)),
            vehicle=NS(getNextTLS=lambda v: [("J", 0, bus["dist"], "r")], isStopped=lambda v: bus["stopped"],
                       getStops=lambda v, n: bus["stops"], getRoadID=lambda v: "in0", getLaneID=lambda v: "in0_0",
                       getLanePosition=lambda v: bus["pos"] if v == "bus1" else 50.0, getSpeed=lambda v: 0.0),
            lane=NS(getLastStepHaltingNumber=lambda ln: 0, getLastStepVehicleIDs=lane_ids),
            inductionloop=NS(getTimeSinceDetection=lambda d: 1e9))
    ag = ActuatedAgents(fallback=set(), extra={"bus"})
    assert "bus" in ag.on and "bus" not in ActuatedAgents(fallback=set()).on
    ag._len = {"in0": 75.0, "in2": 75.0, "out0": 75.0, "out2": 75.0}
    ag._pred, ag._up, ag._waiting = {}, {}, {}
    a = _Agent("J", ["GGrr", "yyrr", "rrrr", "rrGG", "rryy", "rrrr"], ["car", "yellow", "red", "car", "yellow", "red"],
               [5.0, 3.0, 2.0, 5.0, 3.0, 2.0], [45.0, 3.0, 2.0, 45.0, 3.0, 2.0], [0, 3], {0: 5, 3: 2},
               {0: [("in0", "out0")], 3: [("in2", "out2")]}, {}, {}, link_out={0: "out0", 1: "out0", 2: "out2", 3: "out2"})
    ag.junctions, ag.served, ag.fed = {"J": a}, {}, {}
    ag.buses = lambda: ["bus1"]

    def ask(now=100.0):
        ag._bus_requests(tc, now)
        return dict(ag.bus_requests)

    assert ask() == {"J": {0}}
    bus["dist"] = 200.0
    assert ask() == {}                                   # too far
    bus.update(dist=80.0, stopped=True)
    assert ask() == {}                                   # at its stop
    bus.update(stopped=False, stops=[NS(lane="in0_0", endPos=40.0)])
    assert ask() == {}                                   # a stop still before the line
    bus.update(stops=[NS(lane="other_0", endPos=40.0)], ahead=7)
    assert ask() == {}                                   # 7 standing ahead of it
    bus.update(ahead=4)
    assert ask() == {"J": {0}} and ag._bus_near == set()   # 4 standing ahead: asks, but no hold is worth it yet
    bus.update(ahead=0)
    assert ask(100.0) == {"J": {0}} and ag._bus_near == {"J"} and ask(150.0) == {"J": {0}} and ask(170.0) == {}   # 60 s of asking, no more

    def green(ph, spent, counts=((3, 3), (2, 2))):
        calls.clear()
        a.green_since, ag._now = 100.0 - spent, 100.0
        ag._cache = {"in0": counts[0], "in2": counts[1], "out0": (0, 0), "out2": (0, 0)}
        ag._green(tc, a, ph, 100.0)
        return list(calls)

    ag.bus_requests, ag.bus_by, ag._bus_near = {"J": {0}}, {"J": "bus1"}, {"J"}
    assert green(0, 45.0) == [2.0] and green(0, 54.0) == [2.0]     # its green held at and past the 45 s maximum
    assert green(0, 56.0) == []                                    # but not past 45 + 10 s
    ag._bus_near = set()
    assert green(0, 45.0) == []                                    # nor for a bus deep in the queue
    ag._bus_near = {"J"}
    assert green(3, 12.0) == []                                    # the other green: left alone without bus_cut
    ag.bus_cut = 1
    assert green(3, 12.0) == [0] and green(3, 8.0) == []           # ended after its 10 s
    assert green(3, 12.0, ((3, 3), (8, 8))) == []                  # not when its own road is 8 of 10 full
    assert ag.stats["bus_cuts"] == 1 and ag.stats["bus_greens_held"] == 1
    ag.bus_requests = {}
    assert green(0, 45.0) == [] and green(3, 12.0) == []           # no bus: nothing


def test_camera_backup_gives_green_to_a_siren_without_a_request():
    """camback (10 Nov, on by default): a siren the cameras confirm on an approach with no request at the junction gets that
    approach's links, within the same limits as a request; request_loss drops a fixed share of requests."""
    from types import SimpleNamespace as NS

    from cosim.agents_actuated import ObservedActuatedAgents, _Agent, make

    tc = NS(lane=NS(getLastStepHaltingNumber=lambda ln: 0))
    ag = make("agents_observed")
    assert isinstance(ag, ObservedActuatedAgents) and "camback" in ag.on
    ag._len = {"out_0": 75.0, "out_1": 75.0}
    a = _Agent("J", ["Gr", "rG"], ["car", "car"], [5.0, 5.0], [60.0, 60.0], [0, 1], {}, {}, {}, {},
               link_out={0: "out_0", 1: "out_1"}, lane_links={"in_1": {1}, "up_1": {1}})
    ag.junctions = {"J": a}
    ag._now, ag._cache = 0.0, {}
    ag.observe([{"tls": "J", "lanes": {}, "ems": [{"lane": "up_1", "votes": 4, "of": 8, "confirmed": True}]}])
    ag.requests, ag.request_by = {}, {}
    ag._camera_backup(tc, 0.0)
    assert ag.requests == {"J": {1}} and ag.request_by["J"] == "camera" and ag.events[-1]["phases"] == [1]
    ag.requests = {"J": {0}}                                       # a vehicle asked: the camera stays out of it
    ag._camera_backup(tc, 1.0)
    assert ag.requests == {"J": {0}}
    ag.requests, ag._asked = {}, {("J", "ev"): 0.0}               # one asked and its own limits refused it
    ag._camera_backup(tc, 1.0)
    assert ag.requests == {}
    ag._asked = {}
    ag._camera_backup(tc, 1.0 + ag.ev_patience_s)                 # seen there too long: stuck, no more green
    assert ag.requests == {} and ag.stats["cam_patience"] == 1
    ag.observe([{"tls": "J", "lanes": {}, "ems": [{"lane": "in_1", "votes": 2, "of": 8, "confirmed": False}]}])
    ag._camera_backup(tc, 100.0)                                   # not confirmed: nothing
    assert ag.requests == {}
    assert "camback" not in make("agents_observed-camback").on and not make("agents_observed")._lost("ev", "J")
    lossy = make("agents_observed@request_loss=0.5")
    lost = sum(lossy._lost(f"ev{i}", "J") for i in range(400))
    assert 150 < lost < 250 and lossy._lost("ev0", "J") == lossy._lost("ev0", "J")

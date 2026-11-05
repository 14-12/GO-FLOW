"""Tests of the Town05 adapter (no SUMO needed). Run from agents/priority_agent:  python -m pytest -q town05/tests"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from town05.conflicts import LinkMap, TownJunction  # noqa: E402
from town05.coordinator import TownRuleCoordinator  # noqa: E402
from town05.tracker import TownTracker  # noqa: E402

# A 4-way junction "J": links 0-1 from the north, 2-3 east, 4-5 south, 6-7 west; one green per approach
# (Town05's "approach" set), plus "K" where north and south go together (the "permissive" set).
HELLO = {"J": {"n_links": 8, "greens": [[0, 1], [2, 3], [4, 5], [6, 7]]},
         "K": {"n_links": 8, "greens": [[0, 1, 4, 5], [2, 3, 6, 7]]}}


def veh(vid, vclass="emergency", nxt=(("J", 0, 80.0),), speed=10.0, role=None, cam=None):
    return {"id": vid, "vclass": vclass, "type_id": "twin_ambulance_run", "role": role, "on_run": True,
            "speed": speed, "edge": "e", "lane": "e_0", "cam_confirmed": cam,
            "next": [{"tls": t, "link": link, "dist": d, "state": "r"} for t, link, d in nxt]}


def test_linkmap():
    lm = LinkMap(HELLO)
    assert lm.serving("J", 3) == [1]
    assert not lm.compatible("J", 0, 4)
    assert lm.compatible("K", 0, 4)
    assert not lm.known("J", 9)
    assert set(TownJunction("J", lm).incoming) == {str(i) for i in range(8)}


def test_tracker_classes_and_passed():
    tr = TownTracker("v2i")
    out = tr.update(0.0, [veh("a"), veh("p", "authority"), veh("t", "passenger", role="tow"), veh("c", "passenger")])
    assert out["a"].priority_level == 3 and out["p"].priority_level == 2
    assert "c" not in out and "t" not in out  # recovery trucks get no signal priority
    assert out["a"].is_active and out["a"].upcoming[0].approach == "0"
    out = tr.update(1.0, [veh("a", nxt=(("K", 2, 200.0),))])
    assert out["a"].passed == ["J"]


def test_camera_confirmation_is_sticky():
    tr = TownTracker("camera")
    assert not tr.update(0.0, [veh("a", cam=False)])["a"].is_active
    assert tr.update(1.0, [veh("a", cam=True)])["a"].is_active
    assert tr.update(2.0, [veh("a", cam=False)])["a"].is_active


def plan(coord, tr, t, vs):
    return coord.update(t, tr.update(t, vs))


def test_ambulance_before_police_on_conflicting_links():
    tr, co = TownTracker(), TownRuleCoordinator(LinkMap(HELLO))
    ctx = plan(co, tr, 0.0, [veh("p", "authority", (("J", 2, 50.0),)), veh("a", "emergency", (("J", 0, 90.0),))])
    assert ctx["J"].vehicle_id == "a" and ctx["J"].mode == "CONFLICT_RESOLUTION"
    assert co.plan_links["J"] == [0]


def test_first_come_and_compatible_links_merged():
    tr, co = TownTracker(), TownRuleCoordinator(LinkMap(HELLO))
    plan(co, tr, 0.0, [veh("a1", nxt=(("K", 0, 150.0),))])
    ctx = plan(co, tr, 1.0, [veh("a1", nxt=(("K", 0, 140.0),)), veh("a2", nxt=(("K", 4, 60.0),))])
    assert ctx["K"].vehicle_id == "a1" and ctx["K"].mode == "EMERGENCY_PREEMPTION"
    assert co.plan_links["K"] == [0, 4]


def test_horizon_and_recovery():
    tr, co = TownTracker(), TownRuleCoordinator(LinkMap(HELLO), horizon_s=30.0, recovery_s=10.0)
    ctx = plan(co, tr, 0.0, [veh("a", nxt=(("J", 0, 100.0), ("K", 2, 500.0)))])
    assert ctx["J"].mode == "EMERGENCY_PREEMPTION" and ctx["K"].mode == "NORMAL"  # K: ETA 50 s > 30 s
    ctx = plan(co, tr, 5.0, [veh("a", nxt=(("K", 2, 200.0),))])
    assert ctx["J"].mode == "RECOVERY" and ctx["K"].mode == "EMERGENCY_PREEMPTION"
    ctx = plan(co, tr, 16.0, [veh("a", nxt=(("K", 2, 120.0),))])
    assert ctx["J"].mode == "NORMAL"


def test_rate_limit_spares_ambulances_not_police():
    tr, co = TownTracker(), TownRuleCoordinator(LinkMap(HELLO))
    for i in range(4):
        ctx = plan(co, tr, 10.0 * i, [veh(f"p{i}", "authority", (("J", 2, 50.0),))])
    assert ctx["J"].mode != "EMERGENCY_PREEMPTION"  # 4th police car in 300 s refused
    ctx = plan(co, tr, 41.0, [veh("a", "emergency", (("J", 0, 50.0),))])
    assert ctx["J"].mode == "EMERGENCY_PREEMPTION" and ctx["J"].vehicle_id == "a"


def test_serve_roundtrip(tmp_path):
    p = subprocess.Popen([sys.executable, "-m", "town05.serve", "--out", str(tmp_path)], cwd=str(HERE),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    msgs = [{"type": "hello", "v": 1, "run_id": "t", "confirm": "v2i", "junctions": HELLO},
            {"type": "state", "v": 1, "seq": 1, "t": 0.0, "vehicles": [veh("a")]},
            {"type": "state", "v": 1, "seq": 2, "t": 1.0, "vehicles": "bad"},
            {"type": "bye"}]
    out, _ = p.communicate("\n".join(json.dumps(m) for m in msgs) + "\n", timeout=60)
    got = [json.loads(x) for x in out.splitlines()]
    assert got[0]["type"] == "ready" and got[0]["config"]["horizon_s"] == 30.0
    assert got[1]["type"] == "plan" and got[1]["seq"] == 1
    assert got[1]["junctions"]["J"]["links"] == [0] and got[1]["junctions"]["J"]["vehicle"] == "a"
    assert got[2]["type"] == "error"
    assert (tmp_path / "t" / "summary.json").is_file()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

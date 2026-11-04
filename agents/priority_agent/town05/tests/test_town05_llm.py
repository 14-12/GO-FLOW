"""The background LLM coordinator on Town05 (town05/llm_town.py), with a scripted fake model: no Ollama needed.
Needs langchain-core (requirements-llm.txt)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import AIMessage  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from town05.conflicts import LinkMap  # noqa: E402
from town05.coordinator import TownRuleCoordinator  # noqa: E402
from town05.llm_town import TownLLMAdvisor  # noqa: E402
from town05.tracker import TownTracker  # noqa: E402

HELLO = {"J": {"n_links": 8, "greens": [[0, 1], [2, 3], [4, 5], [6, 7]]},
         "K": {"n_links": 8, "greens": [[0, 1, 4, 5], [2, 3, 6, 7]]}}
LANE_LINKS = {"J": {"n_0": [0], "n_1": [1], "e_0": [2], "e_1": [3], "s_0": [4], "s_1": [5], "w_0": [6], "w_1": [7]}}


class FakeLLM:
    """Answers with the scripted tool calls (one AIMessage per invoke), then a final sentence."""

    def __init__(self, script, delay_s=0.0, fail=False):
        self.script, self.delay_s, self.fail, self.calls = list(script), delay_s, fail, 0

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        self.calls += 1
        time.sleep(self.delay_s)
        if self.fail:
            raise ConnectionError("Ollama is not running")
        if self.script:
            name, args = self.script.pop(0)
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{self.calls}"}])
        return AIMessage(content="Done.")


def veh(vid, vclass="emergency", nxt=(("J", 0, 80.0),), speed=10.0, lane="n_0"):
    return {"id": vid, "vclass": vclass, "type_id": "x", "role": None, "on_run": True, "speed": speed, "edge": "e",
            "lane": lane, "cam_confirmed": None,
            "next": [{"tls": t, "link": link, "dist": d, "state": "r"} for t, link, d in nxt]}


def setup(llm, timeout_s=5.0):
    lm = LinkMap(HELLO)
    co = TownRuleCoordinator(lm)
    log = []
    adv = TownLLMAdvisor(llm, co, lm, LANE_LINKS, timeout_s=timeout_s, log=log.append)
    return TownTracker(), co, adv, log


def step(tr, co, adv, t, vehicles, extra=None):
    state = {"t": t, "vehicles": vehicles, **(extra or {})}
    vs = tr.update(t, vehicles)
    ctx = co.update(t, vs)
    junctions = {tls: {"mode": c.mode, "vehicle": c.vehicle_id, "links": co.plan_links.get(tls, [])}
                 for tls, c in ctx.items() if c.mode != "NORMAL"}
    adv.observe(t, vs, state)
    return adv.apply(t, junctions)


def wait_idle(adv, s=5.0):
    end = time.time() + s
    while (adv.busy or not adv._q.empty()) and time.time() < end:
        time.sleep(0.02)
    time.sleep(0.05)


def test_rules_answer_at_once_and_llm_overrides_later():
    # police asked first, ambulance arrives clearly first: the rules keep first-seen (police), the LLM may switch
    llm = FakeLLM([("set_intersection_mode", {"tls_id": "J", "mode": "CONFLICT_RESOLUTION", "vehicle_id": "a"})],
                  delay_s=0.3)
    tr, co, adv, log = setup(llm)
    step(tr, co, adv, 0.0, [veh("p", "authority", (("J", 2, 280.0),), lane="e_0")])
    plan = step(tr, co, adv, 1.0, [veh("p", "authority", (("J", 2, 270.0),), lane="e_0"),
                                   veh("a", "authority", (("J", 0, 40.0),))])
    assert plan["J"]["vehicle"] == "p"          # the rules' answer, immediately (the LLM is still thinking)
    wait_idle(adv)
    plan = step(tr, co, adv, 2.0, [veh("p", "authority", (("J", 2, 260.0),), lane="e_0"),
                                   veh("a", "authority", (("J", 0, 30.0),))])
    assert plan["J"]["vehicle"] == "a" and plan["J"]["source"] == "llm" and plan["J"]["links"] == [0]


def test_llm_cannot_put_police_before_ambulance():
    llm = FakeLLM([("set_intersection_mode", {"tls_id": "J", "mode": "CONFLICT_RESOLUTION", "vehicle_id": "p"})])
    tr, co, adv, log = setup(llm)
    step(tr, co, adv, 0.0, [veh("a", nxt=(("J", 0, 100.0),)), veh("p", "authority", (("J", 2, 40.0),), lane="e_0")])
    wait_idle(adv)
    plan = step(tr, co, adv, 1.0, [veh("a", nxt=(("J", 0, 90.0),)), veh("p", "authority", (("J", 2, 30.0),), lane="e_0")])
    assert plan["J"]["vehicle"] == "a"
    assert any(r["kind"] == "llm_refused" and "priority over" in r["reason"] for r in log)


def test_llm_down_means_rules_only():
    tr, co, adv, log = setup(FakeLLM([], fail=True))
    step(tr, co, adv, 0.0, [veh("a")])
    wait_idle(adv)
    plan = step(tr, co, adv, 1.0, [veh("a", nxt=(("J", 0, 70.0),))])
    assert plan["J"]["vehicle"] == "a" and plan["J"].get("source") != "llm"
    assert adv.stats.failures == 1 and adv.status()["failures"] == 1


def test_late_answer_refused():
    llm = FakeLLM([("set_intersection_mode", {"tls_id": "J", "mode": "EMERGENCY_PREEMPTION", "vehicle_id": "a"})],
                  delay_s=0.6)
    tr, co, adv, log = setup(llm, timeout_s=0.3)
    step(tr, co, adv, 0.0, [veh("a")])
    wait_idle(adv)
    assert adv.stats.late >= 1 or adv.stats.failures >= 1
    assert "J" not in adv.overrides


def test_camera_only_green_with_limits():
    llm = FakeLLM([("request_green_for_lane", {"tls_id": "J", "lane": "s_0", "reason": "lights over 5 frames"}),
                   ("request_green_for_lane", {"tls_id": "J", "lane": "s_0", "reason": "again"})])
    tr, co, adv, log = setup(llm)
    sight = {"sightings": [{"cam": "J-2", "tls": "J", "lane": "s_0", "votes": 5, "of": 8}]}
    step(tr, co, adv, 0.0, [], sight)
    wait_idle(adv)
    plan = step(tr, co, adv, 1.0, [])
    assert plan["J"]["mode"] == "CAMERA_PRIORITY" and plan["J"]["links"] == [4]
    assert any(r["kind"] == "llm_refused" and "at most one" in r["reason"] for r in log)  # the second one
    assert "J" not in step(tr, co, adv, 22.0, [])                                         # held 20 s at most


def test_camera_green_needs_a_real_sighting():
    llm = FakeLLM([("request_green_for_lane", {"tls_id": "J", "lane": "w_0", "reason": "?"})])
    tr, co, adv, log = setup(llm)
    step(tr, co, adv, 0.0, [], {"sightings": [{"cam": "J-2", "tls": "J", "lane": "s_0", "votes": 5, "of": 8}]})
    wait_idle(adv)
    assert "J" not in adv.overrides
    assert any("no confirmed camera sighting" in r.get("reason", "") for r in log)


def test_escalation_needs_a_dispatch_message():
    llm = FakeLLM([("escalate_priority", {"vehicle_id": "a2", "reason": "cardiac arrest"})])
    tr, co, adv, log = setup(llm)
    step(tr, co, adv, 0.0, [veh("a1"), veh("a2", nxt=(("J", 2, 90.0),), lane="e_0")])
    wait_idle(adv)
    assert "a2" not in co.bonus
    llm.script = [("escalate_priority", {"vehicle_id": "a2", "reason": "cardiac arrest"})]
    step(tr, co, adv, 1.0, [veh("a1"), veh("a2", nxt=(("J", 2, 80.0),), lane="e_0")],
         {"dispatch": ["a2 carries a patient in cardiac arrest"]})
    wait_idle(adv)
    assert co.bonus.get("a2") == 0.5
    plan = step(tr, co, adv, 2.0, [veh("a1"), veh("a2", nxt=(("J", 2, 70.0),), lane="e_0")])
    assert plan["J"]["vehicle"] == "a2"   # the rules re-plan with the escalation


def test_serve_with_llm_but_no_ollama(tmp_path):
    """--llm without langchain-ollama or Ollama: the agent still answers, with the rules."""
    import subprocess

    p = subprocess.Popen([sys.executable, "-m", "town05.serve", "--llm", "--llm-url", "http://127.0.0.1:9",
                          "--llm-timeout", "2", "--out", str(tmp_path)], cwd=str(HERE),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    msgs = [{"type": "hello", "run_id": "t", "junctions": {k: {**v, "lane_links": LANE_LINKS.get(k, {})}
                                                           for k, v in HELLO.items()}},
            {"type": "state", "seq": 1, "t": 0.0, "vehicles": [veh("a")]}, {"type": "bye"}]
    out, _ = p.communicate("\n".join(json.dumps(m) for m in msgs) + "\n", timeout=60)
    got = [json.loads(x) for x in out.splitlines()]
    assert got[0]["type"] == "ready" and got[1]["type"] == "plan"
    assert got[1]["junctions"]["J"]["vehicle"] == "a" and got[1]["llm"] is not None

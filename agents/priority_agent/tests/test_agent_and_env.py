"""Tests de l'environnement RL, des coordinateurs et de la plomberie de l'agent LLM.

`ScriptedChatModel` est un DOUBLE DE TEST : un faux modèle de chat qui émet des appels d'outils
selon un script fixe. Il vérifie la boucle d'outils, la validation de sécurité et le repli, sans LLM.
Il n'est jamais utilisé pour produire des résultats d'évaluation.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from coordination.llm_agent import LLMCoordinator
from environment.runtime import OBS_DIM, Runtime
from environment.traffic_env import TrafficSignalEnv
from simulation.scenarios import eval_scenario, write_scenario


class ScriptedChatModel(BaseChatModel):
    mode: str = "plan"   # plan | crash | bad_tool | fake_preempt
    n: int = 0

    @property
    def _llm_type(self):
        return "scripted-test-double"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if self.mode == "crash":
            raise ConnectionError("ollama down")
        last = messages[-1]
        calls = []
        if isinstance(last, HumanMessage):
            calls = [{"name": "get_priority_requests" if self.mode != "bad_tool" else "open_all_lights",
                      "args": {}, "id": "c0"}]
        elif isinstance(last, ToolMessage) and last.content.startswith("["):
            for pv in json.loads(last.content):
                for st in pv["upcoming"]:
                    if self.mode == "fake_preempt" or (pv["siren_on"] and pv["confirmed"]):
                        calls.append({"name": "set_intersection_mode", "id": f"c{len(calls) + 1}",
                                      "args": {"tls_id": st["tls_id"], "mode": "EMERGENCY_PREEMPTION",
                                               "vehicle_id": pv["vehicle_id"], "recovery_s": 20}})
        msg = AIMessage(content="" if calls else "done", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _rt(name, seed=1):
    spec = eval_scenario(name, seed)
    return spec, Runtime(write_scenario(spec, Path(tempfile.mkdtemp())), seed=seed, events=spec["events"]).start()


def test_env_observation_and_mask():
    env = TrafficSignalEnv(seed_offset=77)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.min() >= 0 and obs.max() <= 1
    for _ in range(30):
        m = env.action_masks()
        obs, r, done, _, _ = env.step(int(np.flatnonzero(m)[0]))
        assert np.isfinite(r)
    assert env.rt.safety_log.rejected_actions == 0
    assert env.rt.safety_log.state_violations == 0
    env.close()


def test_llm_plans_full_corridor():
    spec, rt = _rt("S2_urgence")
    co = LLMCoordinator(ScriptedChatModel(mode="plan"), rt)
    seen = set()
    while rt.t < 330:
        rt.step()
        ctx = co.update(rt.t, rt.tracker.vehicles)
        seen |= {(tls, c.mode, c.vehicle_id) for tls, c in ctx.items() if c.vehicle_id}
    rt.close()
    assert {("J1", "EMERGENCY_PREEMPTION", "amb_0"), ("J2", "EMERGENCY_PREEMPTION", "amb_0"),
            ("J3", "EMERGENCY_PREEMPTION", "amb_0")} <= seen
    assert co.stats.calls >= 1 and co.stats.failures == 0


def test_llm_failure_falls_back_to_rules():
    spec, rt = _rt("S2_urgence")
    co = LLMCoordinator(ScriptedChatModel(mode="crash"), rt)
    modes = set()
    while rt.t < 330:
        rt.step()
        modes |= {c.mode for c in co.update(rt.t, rt.tracker.vehicles).values()}
    rt.close()
    assert co.stats.failures >= 1
    assert "EMERGENCY_PREEMPTION" in modes   # le watchdog a appliqué les règles
    assert any(d["source"] == "watchdog_rules" for d in co.decisions)


def test_unknown_tool_is_counted_invalid():
    spec, rt = _rt("S2_urgence")
    co = LLMCoordinator(ScriptedChatModel(mode="bad_tool"), rt)
    while rt.t < 260:
        rt.step()
        co.update(rt.t, rt.tracker.vehicles)
    rt.close()
    assert co.stats.invalid_tool_calls >= 1


def test_safety_blocks_llm_preempting_fake_request():
    spec, rt = _rt("S5_degrade")
    co = LLMCoordinator(ScriptedChatModel(mode="fake_preempt"), rt)
    fake_ctx = False
    while rt.t < 260:
        rt.step()
        ctx = co.update(rt.t, rt.tracker.vehicles)
        fake_ctx |= any(c.vehicle_id == "fake_0" for c in ctx.values())
    rt.close()
    assert not fake_ctx
    assert co.stats.rejected_by_safety >= 1


def test_guard_keeps_intersection_for_arriving_emergency():
    """Le LLM ne peut pas retirer un carrefour à une ambulance qui arrive (erreur observée avec Qwen 3B)."""
    spec, rt = _rt("S4_concurrentes")
    co = LLMCoordinator(None, rt)
    while rt.t < 300:
        rt.step()
        co._t, co._vehicles = rt.t, rt.tracker.vehicles
    co._apply_mode("J3", "EMERGENCY_PREEMPTION", "amb_0", 10, "test")
    assert co.assign["J3"].vehicle_id == "amb_0"
    res = json.loads(co._apply_mode("J3", "NORMAL", None, 10, "llm"))
    assert not res["accepted"] and "reserved for amb_0" in res["reason"]
    assert co.assign["J3"].vehicle_id == "amb_0"
    rt.close()


class Replay7BModel(BaseChatModel):
    """Rejoue les appels d'outils réellement émis par qwen2.5:7b sur S4 graine 1 (run du 14/10)."""
    i: int = 0

    @property
    def _llm_type(self):
        return "replay-test-double"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        E, C = "EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION"
        script = [[("J1", E, "amb_0"), ("J2", "NORMAL", "amb_0"), ("J3", "NORMAL", "amb_0")],
                  [("J2", C, "amb_0"), ("J2", E, "fire_0")],
                  [("J3", E, "police_0"), ("J2", C, "police_0"), ("J1", "RECOVERY", "police_0")],
                  [("J2", E, "police_0")]]
        calls = []
        if isinstance(messages[-1], HumanMessage) and self.i < len(script):
            calls = [{"name": "set_intersection_mode", "id": f"c{k}",
                      "args": {"tls_id": a, "mode": b, "vehicle_id": v, "recovery_s": 30}}
                     for k, (a, b, v) in enumerate(script[self.i])]
            self.i += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="" if calls else "ok", tool_calls=calls))])


def test_reflex_and_guard_repair_observed_llm_mistakes():
    """Erreurs réelles du 7B : corridor incomplet (J2/J3 en NORMAL) et J2 donné à la police alors que
    les pompiers arrivaient. Le réflexe complète le corridor, la garde refuse le mauvais arbitrage."""
    spec, rt = _rt("S4_concurrentes")
    co = LLMCoordinator(Replay7BModel(), rt, horizon_s=30.0, verbose=False)
    got = set()
    while rt.t < 345:
        rt.step()
        ctx = co.update(rt.t, rt.tracker.vehicles)
        got |= {(tls, c.vehicle_id) for tls, c in ctx.items() if c.vehicle_id}
    rt.close()
    assert {("J2", "amb_0"), ("J3", "amb_0"), ("J2", "fire_0")} <= got
    bad = [d for d in co.decisions if d["source"] == "llm" and d["tls_id"] == "J2" and d["vehicle"] == "police_0"]
    assert bad and not any(d["accepted"] for d in bad)


class DispatchReaderModel(BaseChatModel):
    """Double de test : relève la priorité du véhicule cité par un message du SAMU (ou d'un véhicule inventé)."""
    target: str = "amb_1"

    @property
    def _llm_type(self):
        return "dispatch-test-double"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        last, calls = messages[-1], []
        if isinstance(last, HumanMessage) and "arrêt cardiaque" in last.content:
            calls = [{"name": "escalate_priority", "id": "e", "args": {"vehicle_id": self.target, "reason": "cardiac"}}]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="" if calls else "ok", tool_calls=calls))])


def _run_dispatch(target):
    spec, rt = _rt("S7_dispatch")
    co = LLMCoordinator(DispatchReaderModel(target=target), rt, horizon_s=30.0, verbose=False,
                        dispatch=spec["dispatch"])
    got = set()
    while rt.t < 345:
        rt.step()
        got |= {(tls, c.vehicle_id) for tls, c in co.update(rt.t, rt.tracker.vehicles).items() if c.vehicle_id}
    rt.close()
    return co, got


def test_dispatch_escalation_gives_j2_to_critical_patient():
    co, got = _run_dispatch("amb_1")
    assert co.bonus == {"amb_1": 0.5} and ("J2", "amb_1") in got
    j2 = [d for d in co.decisions if d["tls_id"] == "J2" and d["accepted"]]
    assert j2[-1]["vehicle"] == "amb_1"


def test_escalation_without_dispatch_backing_is_rejected():
    """Escalade « inventée » (véhicule cité par aucun message reçu) : refusée par le garde-fou."""
    co, _ = _run_dispatch("amb_9")
    assert co.bonus == {} and co.stats.rejected_by_safety == 1
    assert any(d["mode"] == "ESCALATE" and not d["accepted"] for d in co.decisions)

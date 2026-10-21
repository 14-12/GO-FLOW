"""Tests ciblés de la séparation moteur/garde-fous/politique, sans appel de modèle."""
import copy
from functools import partial
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.agents import TrafficPredictionControlAgent
from src.agents.decision_agent import create_runtime
from src.decision import decision_engine, guardrails, policy


ROOT = Path(__file__).resolve().parents[1]


class DecisionLayerTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads((ROOT / "tests/fixtures/traffic_policy_before.json").read_text(encoding="utf-8"))
        self.thresholds = fixture["thresholds"]
        self.rules = create_runtime()
        self.rules.THRESHOLDS = copy.deepcopy(self.thresholds)
        # Le contexte est une fixture explicite ; aucun chargement historique ou XGBoost.
        self.state = copy.deepcopy(fixture["agent_cases"][0]["state"])
        self.agent = TrafficPredictionControlAgent.__new__(TrafficPredictionControlAgent)
        self.agent.rules = self.rules
        self.agent.policy = dict(policy.DEFAULT_PRIORITY_POLICY)
        self.agent.llm_backend = None
        self.agent._cached = {}
        self.agent.last_trace = {}

    def qualitative_output(self, **changes):
        decision = decision_engine.rule_based_decision(self.state, self.thresholds)
        # Justification conforme aux contrôles qualitatifs historiques du backend 09.
        decision["reason"] = (
            f"Occupation actuelle={self.state['current_occupancy']:.6f}; "
            f"prévue={self.state['predicted_occupancy_t15']:.6f}. "
            "Incidents inconnus; urgences inconnues; plan de feux inconnu.")
        return {**decision, **changes}

    def assert_fallback(self, output):
        self.agent.llm_backend = Mock(return_value=output)
        result = self.agent.decide(copy.deepcopy(self.state))
        self.assertEqual(result["decision_source"], "rule_based_fallback")
        self.assertEqual(result["decision"], self.agent._rule_decision(self.state))
        return result

    def test_invalid_json(self):
        baseline = self.agent._rule_decision(self.state)
        for output in ["{", json.dumps(baseline)[:-1] + ',"confidence":0.5}',
                       json.dumps(baseline).replace('"confidence": 0.5', '"confidence": NaN')]:
            with self.subTest(output=output):
                self.assertFalse(guardrails.validate_decision(output))
                self.assert_fallback(output)

    def test_forbidden_action(self):
        for action in ["invented_action", "extend_green_phase", "reduce_green_phase"]:
            with self.subTest(action=action):
                output = {**self.agent._rule_decision(self.state), "recommended_action": action}
                self.assertFalse(guardrails.validate_contextual_decision(
                    output, self.state, validator=self.agent.validate, rule_decision=self.agent._rule_decision))
                self.assert_fallback(output)
        self.assertFalse(guardrails.validate_decision(
            {**self.agent._rule_decision(self.state), "recommended_action": "invented_action"}))

    def test_invented_incident(self):
        output = self.qualitative_output()
        output["reason"] += " Accident détecté sur la route."
        checks = guardrails.qualitative_checks(output, self.state, rules=self.rules)
        self.assertTrue(checks["hallucination_detected"])
        self.assertFalse(checks["reason_grounded"])
        self.assert_fallback(output)

    def test_invented_emergency(self):
        for claim in [" Véhicule d'urgence détecté.", " Emergency vehicle detected."]:
            with self.subTest(claim=claim):
                output = self.qualitative_output()
                output["reason"] += claim
                checks = guardrails.qualitative_checks(output, self.state, rules=self.rules)
                self.assertTrue(checks["hallucination_detected"])
                self.assertFalse(checks["reason_grounded"])
                self.assert_fallback(output)

    def test_inconsistent_risk_action(self):
        output = self.qualitative_output(risk_level="high", recommended_action="keep_current_signal_plan")
        self.assertTrue(guardrails.validate_decision(output))
        self.assertFalse(guardrails.qualitative_checks(output, self.state, rules=self.rules)["risk_action_coherent"])
        self.assert_fallback(output)

    def test_fallback_for_absent_or_failed_backend(self):
        absent = self.agent.decide(copy.deepcopy(self.state))
        self.assertEqual(absent["decision_source"], "rule_based_fallback")
        self.assertEqual(absent["trace"]["fallback_reason"], "llm_unavailable_for_this_context")
        self.assertEqual(self.assert_fallback(None)["trace"]["fallback_reason"], "llm_returned_no_output")
        self.agent.llm_backend = Mock(side_effect=TimeoutError("Fixture de test"))
        failed = self.agent.decide(copy.deepcopy(self.state))
        self.assertEqual(failed["trace"]["fallback_reason"], "llm_backend_error")
        self.assertIn("TimeoutError", failed["trace"]["llm_error"])

    def test_emergency_priority_before_and_after_llm(self):
        state = copy.deepcopy(self.state)
        state["emergency_vehicle_present"] = True
        state["emergency_context"] = {"emergency_vehicle_present": True, "priority_level": "high"}
        state["incident_detected"] = True
        state["incident_context"] = {"incident_detected": True, "incident_type": "fixture", "severity": "high"}
        self.agent.llm_backend = Mock(side_effect=AssertionError("Le LLM ne doit pas être appelé"))
        result = self.agent.decide(state)
        self.agent.llm_backend.assert_not_called()
        self.assertEqual(result["trace"]["priority"], "emergency")
        self.assertEqual(result["decision"]["recommended_action"], "manual_review")
        self.assertEqual(result["decision"]["risk_level"], "high")
        self.assertTrue(result["trace"]["critical_guard_before"])
        self.assertTrue(result["trace"]["critical_guard_after"])
        hostile = {**result["decision"], "recommended_action": "reduce_green_phase", "risk_level": "low"}
        corrected, locked = guardrails.protect_critical_priority(
            hostile, state, critical_check=guardrails.is_critical_priority,
            rule_decision=self.agent._rule_decision)
        self.assertTrue(locked)
        self.assertEqual(corrected, result["decision"])

    def test_incident_priority(self):
        for severity in ["low", "medium", "high", None]:
            with self.subTest(severity=severity):
                state = copy.deepcopy(self.state)
                state["incident_detected"] = True
                state["incident_context"] = {"incident_detected": True, "incident_type": "fixture", "severity": severity}
                self.agent.llm_backend = Mock(side_effect=AssertionError("Le LLM ne doit pas être appelé"))
                result = self.agent.decide(state)
                self.agent.llm_backend.assert_not_called()
                self.assertEqual(result["trace"]["priority"], "incident")
                self.assertEqual(result["decision"]["recommended_action"], "manual_review")
                self.assertEqual(result["decision"]["risk_level"], "high")
                self.assertEqual(result["trace"]["critical_guard_before"], severity in {"high", None})

    def test_core_final_decision_and_accepted_backend(self):
        rule_decision = partial(decision_engine.rule_based_decision, thresholds=self.thresholds)
        contract = partial(decision_engine.decision_contract, thresholds=self.thresholds)
        grounded = partial(guardrails.is_grounded_decision, validator=guardrails.validate_decision,
                           contract_builder=contract)
        for output, source in [(None, "rule_based"), ("{", "rule_based"),
                               (json.dumps(rule_decision(self.state)), "llm")]:
            with self.subTest(source=source, output=output):
                backend = Mock(return_value=output)
                result = decision_engine.final_decision(
                    self.state, build_traffic_state=lambda sensor, timestep: copy.deepcopy(self.state),
                    llm_decision=backend, is_grounded_decision=grounded, rule_based_decision=rule_decision)
                self.assertEqual(result["source"], source)
                self.assertEqual(result["decision"], rule_decision(self.state))
        self.agent.llm_backend = lambda state: self.agent._rule_decision(state)
        self.assertEqual(self.agent.decide(copy.deepcopy(self.state))["decision_source"], "llm")

    def test_qualitative_and_strict_contracts_remain_distinct(self):
        output = self.qualitative_output()
        self.assertTrue(guardrails.qualitative_checks(output, self.state, rules=self.rules)["reason_grounded"])
        # Le contrat enrichi exige toujours sa justification exacte, comme auparavant.
        self.assertFalse(self.agent._guarded_valid(output, self.state))
        integer_confidence = {**output, "confidence": 1}
        self.assertTrue(guardrails.validate_decision(integer_confidence))
        self.assertFalse(guardrails.validate_llm_decision(integer_confidence))

    def test_legacy_policy_import_is_same_module(self):
        from src.config import traffic_policy
        self.assertIs(traffic_policy, policy)

    def test_saved_09_decisions_and_validation_are_unchanged(self):
        rows = json.loads((ROOT / "results/llm_agent/llm_scenario_outputs.json").read_text(encoding="utf-8"))
        inputs = [{"state": row["state"]} for row in rows]
        checker = partial(guardrails.qualitative_checks, rules=self.rules)
        for row in rows:
            with self.subTest(scenario=row["scenario"]):
                result = decision_engine.final_saved_scenario_decision(
                    row["state"], inputs=inputs, llm_backend=lambda state: copy.deepcopy(row["llm_output"]),
                    qualitative_checker=checker, rule_decision=self.rules.rule_based_decision)
                self.assertEqual(result, {key: row[key] for key in
                                         ["decision", "source", "fallback_reason", "llm_output", "validation"]})


if __name__ == "__main__":
    unittest.main()

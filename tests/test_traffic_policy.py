"""Non-régression face aux sorties figées avant la centralisation de la politique."""
import copy
import json
from pathlib import Path
import unittest

from src.agents.decision_agent import create_runtime
from src.agents.traffic_prediction_agent import (
    INPUT_SCHEMA, OUTPUT_SCHEMA, PRIORITY_POLICY, TrafficPredictionControlAgent)
from src.config import traffic_policy as policy


ROOT = Path(__file__).resolve().parents[1]


class TrafficPolicyRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ces sorties ont été calculées par le code original avant toute modification.
        cls.before = json.loads((ROOT / "tests/fixtures/traffic_policy_before.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.rules = create_runtime()
        self.rules.THRESHOLDS = copy.deepcopy(self.before["thresholds"])

    def test_34_rule_cases_match_before_refactor_exactly(self):
        for case in self.before["rule_cases"]:
            with self.subTest(name=case["name"]):
                state = copy.deepcopy(case["state"])
                self.assertEqual(self.rules.indicators(state), case["indicators"])
                self.assertEqual(self.rules.rule_based_decision(state), case["decision"])
                self.assertEqual(self.rules.decision_contract(state), case["contract"])
                self.assertTrue(self.rules.is_grounded_decision(case["decision"], state))
                self.assertEqual(state, case["state"])

    def test_30_agent_priority_cases_match_before_refactor_exactly(self):
        # Les méthodes de décision seules n'ont besoin d'aucun modèle ni jeu de données.
        agent = TrafficPredictionControlAgent.__new__(TrafficPredictionControlAgent)
        agent.rules = self.rules
        agent.policy = dict(PRIORITY_POLICY)
        agent.llm_backend = None
        agent._cached = {}
        agent.last_trace = {}
        for case in self.before["agent_cases"]:
            with self.subTest(priority=case["priority"], state=case["state"]["current_occupancy"]):
                state = copy.deepcopy(case["state"])
                self.assertEqual(agent._critical(state), case["critical"])
                self.assertEqual(agent._priority(state), case["priority"])
                self.assertEqual(agent._rule_decision(state), case["decision"])
                self.assertEqual(agent.decide(state), case["result"])
                corrected, locked = agent._post_guard(case["decision"], state)
                self.assertEqual(corrected, case["decision"])
                self.assertEqual(locked, case["critical"])
                self.assertEqual(state, case["state"])

    def test_24_validation_cases_match_before_refactor_exactly(self):
        for case in self.before["validation_cases"]:
            with self.subTest(output=case["output"]):
                self.assertEqual(self.rules.validate_decision(case["output"]), case["valid"])

    def test_schemas_and_default_configuration_are_unchanged(self):
        self.assertEqual(INPUT_SCHEMA, self.before["input_schema"])
        self.assertEqual(OUTPUT_SCHEMA, self.before["output_schema"])
        record = json.loads((ROOT / "results/decision_agent/decision_agent_methodology.md").read_text(
            encoding="utf-8").split("```json")[1].split("```")[0])
        self.assertEqual(self.rules.CONFIG, record["config"])
        for case in self.before["agent_cases"]:
            self.assertEqual(PRIORITY_POLICY[case["priority"]], case["result"]["trace"]["priority_rank"])

    def test_thresholds_remain_dynamic_and_neighbor_threshold_is_shared(self):
        thresholds = policy.TrafficThresholds.from_mapping(self.rules.THRESHOLDS)
        self.assertEqual(thresholds.occupancy_high_threshold, self.rules.THRESHOLDS["occupancy_high"])
        self.assertEqual(thresholds.neighbor_occupancy_threshold, thresholds.occupancy_high_threshold)
        # Modifier seulement une fixture en mémoire : aucune valeur train n'est réécrite.
        state = copy.deepcopy(self.before["rule_cases"][0]["state"])
        self.rules.THRESHOLDS["speed_low"] = state["current_speed"]
        self.assertTrue(self.rules.indicators(state)["low_speed"])
        self.assertEqual(thresholds.speed_low_threshold, state["current_speed"])
        self.rules.THRESHOLDS["occupancy_high"] = 0.
        self.assertEqual(thresholds.neighbor_occupancy_threshold, 0.)

    def test_runtime_configuration_is_not_shared(self):
        other = create_runtime()
        self.rules.CONFIG["occupancy_quantiles"].append(1.)
        self.rules.ACTIONS.add("fixture")
        self.assertEqual(other.CONFIG, {**other.CONFIG, **policy.default_rule_config()})
        self.assertNotIn("fixture", other.ACTIONS)
        self.assertNotIn("fixture", policy.ACTIONS)
        modified_priorities = dict(PRIORITY_POLICY)
        modified_priorities["emergency"] = -1
        self.assertEqual(PRIORITY_POLICY, dict(policy.DEFAULT_PRIORITY_POLICY))

    def test_threshold_access_preserves_old_lazy_branches(self):
        state = copy.deepcopy(self.before["rule_cases"][0]["state"])
        state["current_occupancy"] = 1.
        # L'ancien code ne lisait pas la médiane lorsque le niveau était déjà severe.
        del self.rules.THRESHOLDS["occupancy_median"]
        self.assertEqual(self.rules.rule_based_decision(state)["traffic_state"], "severe")


if __name__ == "__main__":
    unittest.main()

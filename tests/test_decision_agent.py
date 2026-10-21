"""Tests hors ligne des règles partagées ; aucune inférence ni écriture de résultats."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.agents import decision_agent


ROOT = Path(__file__).resolve().parents[1]


class DecisionRulesTests(unittest.TestCase):
    def setUp(self):
        self.rules = decision_agent.create_runtime()
        # Seuils synthétiques explicites : ne jamais modifier les seuils sauvegardés.
        self.rules.THRESHOLDS = {
            "occupancy_median": 10., "occupancy_high": 20., "occupancy_severe": 30.,
            "speed_low": 40., "speed_median": 60., "strong_change": 5.}
        self.state = {
            "sensor_id": 0, "timestep": 8, "current_flow": 100., "current_speed": 60.,
            "current_occupancy": 5., "predicted_occupancy_t15": 5., "occupancy_change": 0.,
            "recent_occupancy_trend": 0., "recent_speed_trend": 0., "neighbors": [],
            "incident_detected": None, "emergency_vehicle_present": None}

    def test_runtimes_are_isolated(self):
        other = decision_agent.create_runtime()
        self.rules.CONFIG["trend_history"] = 99
        self.rules.ACTIONS.add("fixture")
        self.assertEqual(other.CONFIG["trend_history"], 6)
        self.assertNotIn("fixture", other.ACTIONS)
        self.assertFalse(hasattr(other, "LOOKUP"))
        self.assertIsNot(self.rules.final_decision.__globals__, other.final_decision.__globals__)

    def test_legacy_import_shares_assignments(self):
        from scripts import decision_agent as legacy
        self.assertIs(legacy, decision_agent)
        with patch.object(legacy, "THRESHOLDS", self.rules.THRESHOLDS, create=True):
            self.assertEqual(legacy.rule_based_decision(self.state),
                             self.rules.rule_based_decision(self.state))

    def test_threshold_boundaries_and_actions(self):
        cases = [
            ({"current_occupancy": 10.}, "free_flow", "low", "keep_current_signal_plan"),
            ({"current_occupancy": np.nextafter(10., 20.)}, "moderate", "medium", "monitor"),
            ({"current_occupancy": 20.}, "congested", "high", "manual_review"),
            ({"predicted_occupancy_t15": 30.}, "severe", "high", "manual_review"),
            ({"occupancy_change": 5.}, "congested", "high", "manual_review"),
            ({"current_speed": 40.}, "moderate", "medium", "monitor"),
            ({"recent_occupancy_trend": 1., "recent_speed_trend": -1.},
             "moderate", "medium", "monitor"),
            ({"current_occupancy": 30., "occupancy_change": -10.},
             "severe", "high", "manual_review"),
            ({"neighbors": [{"current_occupancy": 20.}]}, "moderate", "medium", "monitor"),
            ({"current_occupancy": 30., "neighbors": [{"current_occupancy": 20.}]},
             "severe", "high", "coordinate_with_neighbor")]
        for changes, traffic, risk, action in cases:
            with self.subTest(changes=changes):
                result = self.rules.rule_based_decision({**self.state, **changes})
                self.assertEqual((result["traffic_state"], result["risk_level"],
                                  result["recommended_action"]), (traffic, risk, action))
                self.assertEqual(result["confidence"], .5)
                self.assertIn("Incidents et urgences inconnus; plan de feux indisponible.", result["reason"])

    def test_indicators(self):
        state = {**self.state, "occupancy_change": -5.,
                 "neighbors": [{"current_occupancy": 20.}, {"current_occupancy": 2.}]}
        flags = self.rules.indicators(state)
        self.assertTrue(flags["improving"])
        self.assertFalse(flags["strong_increase"])
        self.assertEqual(flags["congested_neighbors"], 1)
        self.assertEqual(flags["neighbor_occupancy_range"], 18.)
        self.assertIsNone(self.rules.indicators(self.state)["neighbor_occupancy_range"])

    def test_strict_json_and_validation(self):
        baseline = self.rules.rule_based_decision(self.state)
        self.assertTrue(self.rules.validate_decision(json.dumps(baseline)))
        for value in [True, float("nan"), float("inf"), -0.1, 1.1, "0.5", None]:
            with self.subTest(confidence=value):
                self.assertFalse(self.rules.validate_decision({**baseline, "confidence": value}))
        for key, value in [("traffic_state", []), ("risk_level", "invented"),
                           ("recommended_action", "invented"), ("reason", " ")]:
            self.assertFalse(self.rules.validate_decision({**baseline, key: value}))
        for output in [None, [], "{", {**baseline, "extra": 1}, {"reason": "incomplet"}]:
            self.assertFalse(self.rules.validate_decision(output))
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError):
                self.rules.strict_json_loads(text)
        duplicate = json.dumps(baseline)[:-1] + ', "confidence": 0.5}'
        self.assertFalse(self.rules.validate_decision(duplicate))

    def test_contract_and_grounding(self):
        for occupancy in [5., 15., 30.]:
            state = {**self.state, "current_occupancy": occupancy}
            baseline = self.rules.rule_based_decision(state)
            contract = self.rules.decision_contract(state)
            self.assertTrue(self.rules.is_grounded_decision(baseline, state))
            for action in contract["allowed_actions"]:
                for reason in contract["allowed_reasons"]:
                    self.assertTrue(self.rules.is_grounded_decision(
                        json.dumps({**baseline, "recommended_action": action, "reason": reason}), state))
            for key, value in [("reason", "Accident détecté."), ("confidence", .9),
                               ("recommended_action", "extend_green_phase"),
                               ("recommended_action", "reduce_green_phase")]:
                self.assertFalse(self.rules.is_grounded_decision({**baseline, key: value}, state))
            self.assertFalse(self.rules.is_grounded_decision("{", state))

    def test_causal_state_and_spatial_context(self):
        # Une arête de poids très faible reste un voisin ; ignorer seulement la diagonale.
        observations = np.arange(3 * 10 * 3, dtype=float).reshape(3, 10, 3)
        self.rules.OBSERVATIONS = observations
        self.rules.ADJACENCY = np.array([[1., 1e-100, 0.], [1e-100, 0., 0.], [0., 0., 0.]])
        rows = [{"location": sensor, "timestep": t, "current_flow": 100. + sensor,
                 "current_speed": 60. + sensor, "current_occupancy": 5. + sensor,
                 "predicted_occupancy_t15": 7. + sensor, "target_occupy_t15": 999.}
                for sensor in range(3) for t in [8, 9]]
        self.rules.LOOKUP = pd.DataFrame(rows).set_index(["location", "timestep"])
        state = self.rules.build_traffic_state(np.int64(0), np.int64(8))
        self.assertEqual(state["recent_occupancy_trend"], 3.)
        self.assertEqual(state["recent_speed_trend"], 3.)
        self.assertEqual(state["occupancy_change"], 2.)
        self.assertEqual(state["neighbors"], [{"sensor_id": 1, "timestep": 8,
                         "current_speed": 61., "current_occupancy": 6., "predicted_occupancy_t15": 8.}])
        self.assertNotIn("target_occupy_t15", state)
        self.assertIsNone(state["incident_detected"])
        self.assertIsNone(state["emergency_vehicle_present"])
        observations[:, 7:, :] = -999.
        self.assertEqual(self.rules.build_traffic_state(0, 8), state)
        self.assertEqual(self.rules.build_traffic_state(2, 8)["neighbors"], [])
        for sensor, timestep in [(True, 8), (0, False), (0.0, 8), (0, "8")]:
            with self.assertRaises(ValueError):
                self.rules.build_traffic_state(sensor, timestep)
        with self.assertRaises(KeyError):
            self.rules.build_traffic_state(0, 100)

    def test_final_decision_fallback_and_input_guard(self):
        baseline = self.rules.rule_based_decision(self.state)
        with patch.object(self.rules, "build_traffic_state", return_value=copy.deepcopy(self.state)):
            for output, reason in [(None, "llm_unavailable"), ("{", "invalid_or_ungrounded_llm_output"),
                                  ({**baseline, "reason": "Accident détecté."},
                                   "invalid_or_ungrounded_llm_output")]:
                with patch.object(self.rules, "llm_decision", return_value=output):
                    self.assertEqual(self.rules.final_decision(self.state), {
                        "decision": baseline, "source": "rule_based", "fallback_reason": reason})
            for error in [OSError(), TimeoutError(), ValueError(), TypeError(), KeyError()]:
                with patch.object(self.rules, "llm_decision", side_effect=error):
                    result = self.rules.final_decision(self.state)
                    self.assertEqual(result["fallback_reason"], "llm_error:" + type(error).__name__)
            for output in [baseline, json.dumps(baseline)]:
                with patch.object(self.rules, "llm_decision", return_value=output):
                    self.assertEqual(self.rules.final_decision(self.state)["source"], "llm")
            def mutate(state):
                state["current_flow"] = -1.
                return None
            with patch.object(self.rules, "llm_decision", side_effect=mutate):
                self.rules.final_decision(self.state)
                self.assertEqual(self.state["current_flow"], 100.)
            with patch.object(self.rules, "llm_decision") as backend:
                with self.assertRaises(ValueError):
                    self.rules.final_decision({**self.state, "predicted_occupancy_t15": 999.})
                backend.assert_not_called()

    def test_llm_interface_without_network(self):
        self.assertIsNone(self.rules.llm_decision(self.state))
        self.rules.CONFIG["llm_model"] = "fixture-instruct"
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = io.StringIO(json.dumps({"message": {"content": "{}"}}))
        opener = unittest.mock.MagicMock()
        opener.open.return_value = response
        with patch.object(self.rules.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(self.rules.llm_decision(self.state), "{}")
        request = opener.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(json.loads(payload["messages"][1]["content"])["contract"],
                         self.rules.decision_contract(self.state))
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 30)

    def test_environment_inspection_offline(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(self.rules.importlib.util, "find_spec", return_value=None), \
                patch.object(self.rules.shutil, "which", return_value=None), \
                patch.object(self.rules.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = OSError("Fixture hors ligne")
            result = self.rules.inspect_environment(Path(folder))
        self.assertEqual(result["llm_mode"], "placeholder_rule_based")
        self.assertIsNone(result["selected_llm"])


class SavedDataRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = decision_agent.create_runtime()
        # Charger les données réelles en lecture seule, sans sonder ni appeler un LLM.
        with patch.object(cls.rules, "inspect_environment", return_value={"selected_llm": None}), \
                redirect_stdout(io.StringIO()):
            cls.preview = cls.rules.initialize(ROOT)
        cls.inputs = json.loads((ROOT / "results/decision_agent/agent_inputs.json").read_text(encoding="utf-8"))
        cls.outputs = json.loads((ROOT / "results/decision_agent/agent_outputs.json").read_text(encoding="utf-8"))

    def test_initialization_thresholds_states_and_saved_decisions_unchanged(self):
        methodology = (ROOT / "results/decision_agent/decision_agent_methodology.md").read_text(encoding="utf-8")
        record = json.loads(methodology.split("```json")[-1].split("```")[0])
        self.assertEqual(self.rules.THRESHOLDS, record["thresholds_train_only"])
        self.assertEqual(len(self.preview), 5)
        for item, output in zip(self.inputs, self.outputs):
            state = item["state"]
            self.assertEqual(self.rules.build_traffic_state(state["sensor_id"], state["timestep"]), state)
            self.assertEqual(self.rules.rule_based_decision(state), output["decision"])

    def test_selection_and_evaluation_match_saved_experiment(self):
        scenarios = self.rules.select_scenarios()
        saved = pd.read_csv(ROOT / "results/decision_agent/selected_scenarios.csv")
        pd.testing.assert_frame_equal(scenarios, saved, check_dtype=False)
        inputs, outputs, validations = self.rules.evaluate_scenarios(scenarios)
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(outputs, self.outputs)
        pd.testing.assert_frame_equal(validations, pd.read_csv(ROOT / "results/decision_agent/validation_results.csv"))

    def test_saving_is_explicit_and_limited_to_temporary_directory(self):
        scenarios = self.rules.select_scenarios()
        inputs, outputs, validations = self.rules.evaluate_scenarios(scenarios)
        with tempfile.TemporaryDirectory() as folder, patch.object(self.rules, "ROOT", Path(folder)), \
                patch.object(self.rules, "file_hash", return_value="fixture"), redirect_stdout(io.StringIO()):
            self.rules.save_results(scenarios, inputs, outputs, validations, "Méthode de test")
            target = Path(folder) / "results/decision_agent"
            self.assertEqual(json.loads((target / "agent_outputs.json").read_text(encoding="utf-8")), outputs)
            self.assertEqual(len(list(target.iterdir())), 5)

    def test_existing_orchestrator_integration(self):
        from src.agents.traffic_prediction_agent import TrafficPredictionControlAgent
        from tests.multi_agent_integration_cases import run_integration_cases
        # Réutiliser le chargement en lecture seule ; les backends des cas sont simulés.
        with patch("src.agents.traffic_prediction_agent.create_runtime", return_value=self.rules), \
                patch.object(self.rules, "initialize", return_value=self.preview):
            agent = TrafficPredictionControlAgent(ROOT)
        self.assertIsNone(agent._init_error)
        _, results = run_integration_cases(agent, ROOT)
        self.assertTrue(all(item["passed"] for item in results))

    def test_notebook_imports_and_embedded_guards(self):
        notebook = json.loads((ROOT / "notebooks/08_llm_decision_agent.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook08", "exec")
        for directory in [ROOT, ROOT / "notebooks"]:
            namespace = {}
            with patch.object(Path, "cwd", return_value=directory):
                exec("".join(notebook["cells"][2]["source"]), namespace)
            namespace["_rules"] = self.rules
            for index in [4, 7, 9, 11, 13, 16]:
                exec("".join(notebook["cells"][index]["source"]), namespace)
            with patch.object(self.rules, "initialize", return_value=self.preview):
                namespace["initialize"](ROOT)
            namespace["inputs"] = self.inputs
            with redirect_stdout(io.StringIO()):
                exec("".join(notebook["cells"][19]["source"]), namespace)


if __name__ == "__main__":
    unittest.main()

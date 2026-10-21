"""Contrat public et exemples : tests indépendants du forecasting et du LLM."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

from jsonschema import Draft202012Validator

from src.contracts import (
    ACTIONS, Action, INPUT_SCHEMA, OUTPUT_SCHEMA, get_input_schema,
    load_example, validate_input, validate_output)
from src.contracts.enums import DecisionSource, ResponseStatus, RiskLevel, TrafficState
from src.contracts.examples import EXAMPLE_NAMES
from src.decision import guardrails, policy


ROOT = Path(__file__).resolve().parents[1]


class AgentContractTests(unittest.TestCase):
    def setUp(self):
        self.message = load_example("input")
        self.warning = load_example("output_warning")
        self.error = load_example("output_error")

    def test_schemas_are_same_objects_as_legacy_exports_and_values_unchanged(self):
        from src.agents.traffic_prediction_agent import INPUT_SCHEMA as old_input, OUTPUT_SCHEMA as old_output
        self.assertIs(INPUT_SCHEMA, old_input)
        self.assertIs(OUTPUT_SCHEMA, old_output)
        fixture = json.loads((ROOT / "tests/fixtures/traffic_policy_before.json").read_text(encoding="utf-8"))
        self.assertEqual(INPUT_SCHEMA, fixture["input_schema"])
        self.assertEqual(OUTPUT_SCHEMA, fixture["output_schema"])
        for schema in [INPUT_SCHEMA, OUTPUT_SCHEMA, get_input_schema("live")]:
            Draft202012Validator.check_schema(schema)

    def test_enums_are_single_source_of_values(self):
        self.assertIs(ACTIONS, policy.ACTIONS)
        self.assertIs(ACTIONS, guardrails.ACTIONS)
        properties = OUTPUT_SCHEMA["properties"]
        for name, enum in [("recommended_action", Action), ("decision_source", DecisionSource),
                           ("status", ResponseStatus)]:
            self.assertEqual(properties[name]["enum"], [item.value for item in enum])
        for name, enum in [("traffic_state", TrafficState), ("risk_level", RiskLevel)]:
            self.assertEqual(properties[name]["enum"], [item.value for item in enum] + [None])
        self.assertEqual(json.loads(json.dumps(Action.MONITOR)), "monitor")

    def test_shipped_examples_are_valid_and_match_saved_exchanges(self):
        self.assertTrue(validate_input(self.message))
        self.assertTrue(validate_input(self.message, prediction_mode="live"))
        self.assertTrue(validate_output(self.warning))
        self.assertTrue(validate_output(self.error))
        messages = json.loads((ROOT / "results/multi_agent_integration/orchestrator_messages.json").read_text(encoding="utf-8"))
        responses = json.loads((ROOT / "results/multi_agent_integration/agent_responses.json").read_text(encoding="utf-8"))
        self.assertIn(self.message, messages)
        self.assertIn(self.warning, responses)
        self.assertIn(self.error, responses)
        self.assertEqual(self.warning["request_id"], self.message["request_id"])
        self.assertEqual(self.warning["sensor_id"], self.message["sensor_id"])

    def test_document_json_examples_match_canonical_files(self):
        text = (ROOT / "docs/traffic_prediction_agent_contract.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)
        self.assertEqual([json.loads(block) for block in blocks],
                         [load_example(name) for name in EXAMPLE_NAMES])

    def test_minimal_input_and_optional_contexts(self):
        minimal = {key: self.message[key] for key in INPUT_SCHEMA["required"]}
        self.assertTrue(validate_input(minimal))
        for value in [None, {}]:
            message = {**minimal, **{name: value for name in [
                "incident_context", "emergency_context", "public_transport_context", "cooperative_context"]}}
            self.assertTrue(validate_input(message))
        self.assertTrue(validate_input({**minimal, "timestamp": "2026-10-20T12:00:00+01:00"}))

    def test_invalid_required_fields_and_unknown_fields(self):
        for name in INPUT_SCHEMA["required"]:
            invalid = copy.deepcopy(self.message)
            del invalid[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                validate_input(invalid)
        for changes in [{"request_id": " "}, {"sensor_id": True}, {"sensor_id": 170},
                        {"timestep": "15227"}, {"timestamp": "2026-10-20T12:00:00"},
                        {"prediction_mode": "live"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_input({**self.message, **changes})
        with self.assertRaisesRegex(ValueError, "prediction_mode"):
            validate_input(self.message, prediction_mode="invalid")

    def test_measures_are_finite_numbers_and_booleans_rejected(self):
        for value in [True, "1", float("nan"), float("inf"), None]:
            invalid = copy.deepcopy(self.message)
            invalid["traffic_data"]["current_speed"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_input(invalid)

    def test_internal_context_consistency(self):
        for changes in [
            {"incident_context": {"incident_detected": False, "severity": "high"}},
            {"incident_context": {"incident_type": "incident"}},
            {"emergency_context": {"priority_level": "high"}},
            {"public_transport_context": {"priority_requested": "true"}},
            {"cooperative_context": {"neighboring_sensors": "invalid"}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_input({**self.message, **changes})
        self.assertTrue(validate_input({**self.message, "incident_context": {
            "incident_detected": True, "incident_type": "fixture", "severity": "high"},
            "integration_test_fixture": True}))

    def test_neighbor_schema_without_claiming_graph_validation(self):
        neighbor = {"sensor_id": 1, "timestep": self.message["timestep"],
                    "current_speed": 60., "current_occupancy": 0.05}
        message = {**self.message, "cooperative_context": {"neighboring_sensors": [neighbor]}}
        # Seule la forme est validée ici ; la présence d'une arête exige l'état de l'agent.
        self.assertTrue(validate_input(message))
        for invalid in [{}, {**neighbor, "extra": 1}, {**neighbor, "sensor_id": True},
                        {**neighbor, "current_speed": float("nan")}]:
            message["cooperative_context"]["neighboring_sensors"] = [invalid]
            with self.subTest(neighbor=invalid), self.assertRaises(ValueError):
                validate_input(message)
        # L'ancien adaptateur garde son premier contrôle ; build_state rejettera ensuite {}.
        message["cooperative_context"]["neighboring_sensors"] = [{}]
        self.assertTrue(validate_input(message, check_schema=False))

    def test_mode_schemas_are_independent(self):
        live = get_input_schema("live")
        self.assertEqual(live["properties"]["timestep"], {"type": "integer", "minimum": 13})
        self.assertNotIn("maximum", live["properties"]["timestep"])
        self.assertTrue(validate_input({**self.message, "timestep": 18013}, prediction_mode="live"))
        with self.assertRaises(ValueError):
            validate_input({**self.message, "timestep": 18013})
        live["properties"]["sensor_id"]["maximum"] = -1
        self.assertEqual(INPUT_SCHEMA["properties"]["sensor_id"]["maximum"], 169)

    def test_response_status_and_error_invariants(self):
        # ok est contractuellement valide, sans être présenté comme sortie réelle de l'agent.
        self.assertTrue(validate_output({**self.warning, "status": "ok"}))
        for changes in [{"forecast": {"horizon_minutes": 15, "predicted_occupancy": 0.1}},
                        {"confidence": 0.5}, {"recommended_action": "monitor"},
                        {"traffic_state": "free_flow"}]:
            self.assertFalse(validate_output({**self.error, **changes}))
        for changes in [{"status": "invalid"}, {"confidence": True}, {"sensor_id": 170},
                        {"recommended_action": "invented"}, {"extra": 1}]:
            self.assertFalse(validate_output({**self.warning, **changes}))

    def test_schema_validation_does_not_replace_decision_guardrails(self):
        # Ces valeurs existent dans le contrat général mais restent interdites par le grounding.
        for action in [Action.EXTEND_GREEN_PHASE, Action.REDUCE_GREEN_PHASE]:
            self.assertTrue(validate_output({**self.warning, "recommended_action": action.value}))

    def test_examples_are_independent_and_unknown_name_rejected(self):
        example = load_example("input")
        example["traffic_data"]["current_speed"] = -1
        self.assertEqual(load_example("input"), self.message)
        with self.assertRaises(ValueError):
            load_example("../input")

    def test_public_contract_import_loads_no_ml_library(self):
        code = (
            "import sys; from src.contracts import INPUT_SCHEMA, OUTPUT_SCHEMA, validate_input, validate_output; "
            "assert not {'xgboost','torch','pandas','numpy'} & set(sys.modules)"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

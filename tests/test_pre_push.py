"""Validation avant push : aucun entraînement, serveur LLM ou export scientifique."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jsonschema import Draft202012Validator

from src.agents import TrafficPredictionControlAgent
from src.agents.decision_agent import create_runtime
from src.contracts import OUTPUT_SCHEMA, load_example, validate_input, validate_output
from tests.multi_agent_integration_cases import run_integration_cases


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "results/multi_agent_integration/integration_test_results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def agent():
    # Charger les exports une seule fois ; neutraliser uniquement le diagnostic matériel.
    rules = create_runtime()
    with patch.object(rules, "inspect_environment", return_value={"selected_llm": None}), redirect_stdout(io.StringIO()):
        rules.initialize(ROOT)
    with patch("src.agents.traffic_prediction_agent.create_runtime", return_value=rules), patch.object(rules, "initialize"):
        instance = TrafficPredictionControlAgent(ROOT)
    assert instance._init_error is None
    return instance


@pytest.fixture(scope="module")
def integration_results(agent):
    orchestrator, results = run_integration_cases(agent, ROOT)
    assert len(results) == len(EXPECTED) == 24
    for response in orchestrator.responses:
        Draft202012Validator(OUTPUT_SCHEMA).validate(response)
        assert validate_output(response)
    return {row["test"]: row for row in results}


@pytest.mark.parametrize("expected", EXPECTED, ids=[row["test"] for row in EXPECTED])
def test_24_scenarios_sans_regression(integration_results, expected):
    # Comparer tous les diagnostics sauvegardés, pas seulement l'action finale.
    assert integration_results[expected["test"]] == expected
    assert expected["passed"]


def test_import_et_validation_input(agent):
    assert isinstance(agent, TrafficPredictionControlAgent)
    assert validate_input(load_example("input"))
    with pytest.raises(ValueError):
        validate_input({"request_id": "incomplet"})


@pytest.mark.parametrize("scenario,changes,action,priority", [
    ("trafic_fluide", {}, "keep_current_signal_plan", "normal"),
    ("forte_hausse_predite", {}, "coordinate_with_neighbor", "severe_congestion"),
    ("trafic_fluide", {"incident_context": {"incident_detected": True, "severity": "high"}}, "manual_review", "incident"),
    ("trafic_fluide", {"emergency_context": {"emergency_vehicle_present": True, "priority_level": "high"}}, "manual_review", "emergency"),
    ("trafic_fluide", {"public_transport_context": {"priority_requested": True}}, "manual_review", "public_transport"),
], ids=["normal", "congestion", "incident_critique", "urgence_critique", "transport_public"])
def test_recommandation_et_schema(agent, scenario, changes, action, priority):
    states = json.loads((ROOT / "results/decision_agent/agent_inputs.json").read_text(encoding="utf-8"))
    state = next(row["state"] for row in states if row["scenario"] == scenario)
    message = {"request_id": scenario, "sensor_id": state["sensor_id"], "timestep": state["timestep"],
               "traffic_data": {key: state[key] for key in ("current_flow", "current_speed", "current_occupancy")},
               "integration_test_fixture": True, **changes}
    backend = Mock(side_effect=AssertionError("Priorité externe : aucun appel LLM attendu"))
    with patch.object(agent, "llm_backend", backend if changes else None):
        response = agent.handle_message(message)
    assert response["status"] == "warning"
    assert response["recommended_action"] == action
    assert response["metadata"]["priority"] == priority
    assert validate_output(response)
    if changes:
        backend.assert_not_called()


@pytest.mark.parametrize("backend,reason", [
    (lambda state: None, "llm_returned_no_output"),
    (lambda state: "{", "llm_backend_error"),
    (Mock(side_effect=ConnectionError("Backend indisponible simulé")), "llm_backend_error"),
], ids=["fallback_sans_sortie", "fallback_json_invalide", "backend_indisponible"])
def test_fallback_llm(agent, backend, reason):
    with patch.object(agent, "llm_backend", backend):
        response = agent.handle_message(load_example("input"))
    assert validate_output(response)
    assert response["decision_source"] == "rule_based_fallback"
    assert response["recommended_action"] == "keep_current_signal_plan"
    assert response["metadata"]["fallback_reason"] == reason


@pytest.mark.parametrize("case", ["voisin_invalide", "donnee_incoherente"])
def test_entree_incoherente_erreur_controlee(agent, case):
    message = copy.deepcopy(load_example("input"))
    if case == "voisin_invalide":
        message["cooperative_context"] = {"neighboring_sensors": [
            {"sensor_id": 999, "timestep": message["timestep"], "current_speed": 65.0, "current_occupancy": 0.0}]}
    else:
        message["traffic_data"]["current_occupancy"] += 1.0
    response = agent.handle_message(message)
    assert response["status"] == "error"
    assert response["metadata"]["error_controlled"]
    assert response["forecast"]["predicted_occupancy"] is None
    assert validate_output(response)

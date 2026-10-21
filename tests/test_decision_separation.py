"""Comparer les règles et les décisions complètes aux témoins antérieurs."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agents import TrafficPredictionControlAgent
from src.agents.decision_agent import create_runtime
from src.decision import decision_engine, guardrails, policy

ROOT = Path(__file__).resolve().parents[1]
BEFORE = json.loads((ROOT / "tests/fixtures/decision_separation_before.json").read_text(encoding="utf-8"))
RULES = json.loads((ROOT / "tests/fixtures/traffic_policy_before.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", BEFORE["cases"], ids=[case["name"] for case in BEFORE["cases"]])
def test_decision_et_trace_strictement_identiques(case):
    rules = create_runtime()
    rules.THRESHOLDS = copy.deepcopy(BEFORE["thresholds"])
    # Isoler la décision des lectures CSV, du forecasting et des backends réels.
    agent = TrafficPredictionControlAgent.__new__(TrafficPredictionControlAgent)
    agent.rules = rules
    agent.policy = dict(policy.DEFAULT_PRIORITY_POLICY)
    agent._cached = {}
    agent.last_trace = {}
    backend = Mock(side_effect=lambda state: copy.deepcopy(case["candidate"]))
    agent.llm_backend = backend if case["backend_present"] else None
    result = agent.decide(copy.deepcopy(case["state"]))
    assert result == case["expected"]
    assert agent.last_trace == case["expected"]["trace"]
    if case["name"] in {"priorite_emergency", "priorite_incident", "transport_public"}:
        backend.assert_not_called()
    if case["name"] in {"hallucination_incident", "hallucination_emergency"}:
        assert not guardrails.validate_contextual_decision(
            case["candidate"], case["state"], validator=agent.validate, rule_decision=agent._rule_decision)


@pytest.mark.parametrize("case", RULES["rule_cases"], ids=[case["name"] for case in RULES["rule_cases"]])
def test_regles_seuils_et_adaptateurs_identiques(case):
    # Inclure les égalités et les flottants immédiatement avant/après chaque seuil.
    thresholds = RULES["thresholds"]
    state = copy.deepcopy(case["state"])
    assert policy.indicators(state, thresholds) == case["indicators"]
    assert decision_engine.indicators(state, thresholds) == case["indicators"]
    assert policy.rule_based_decision(state, thresholds) == case["decision"]
    assert decision_engine.rule_based_decision(state, thresholds) == case["decision"]
    assert decision_engine.decision_contract(state, thresholds) == case["contract"]


def test_reponses_completes_des_24_scenarios_historiques():
    from tests.test_llama_cpp_backend import IntegrationRegressionTests
    from tests.multi_agent_integration_cases import run_integration_cases

    # Réutiliser le chargement hors ligne existant, sans exécuter de notebook.
    IntegrationRegressionTests.setUpClass()
    orchestrator, results = run_integration_cases(IntegrationRegressionTests.agent, ROOT)
    expected_results = json.loads((ROOT / "results/multi_agent_integration/integration_test_results.json").read_text(encoding="utf-8"))
    expected_responses = json.loads((ROOT / "results/multi_agent_integration/agent_responses.json").read_text(encoding="utf-8"))
    assert len(results) == 24
    assert results == expected_results
    # Deux cas sont des contrôles internes sans message : 22 réponses pour 24 cas.
    assert len(orchestrator.responses) == len(expected_responses) == 22
    assert orchestrator.responses == expected_responses

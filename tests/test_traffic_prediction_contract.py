"""Contrat d'équipe : identité des exports et absence de nouveau comportement."""
import copy
import json
from pathlib import Path

import pytest

from src import contracts
from src.contracts import enums, schemas, validation
from src.contracts import traffic_prediction_contract as contract
from src.agents import traffic_prediction_agent as agent


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["INPUT_SCHEMA", "OUTPUT_SCHEMA", "get_input_schema"])
def test_exports_identiques_pour_tous_les_consommateurs(name):
    assert getattr(contract, name) is getattr(schemas, name)
    assert getattr(agent, name) is getattr(schemas, name)
    assert getattr(contracts, name) is getattr(schemas, name)


@pytest.mark.parametrize("name", [
    "ACTIONS", "TRAFFIC_STATES", "RISK_LEVELS", "RESPONSE_STATUSES", "DECISION_SOURCES",
])
def test_valeurs_autorisees_sans_copie(name):
    assert getattr(contract, name) is getattr(enums, name)
    assert getattr(contracts, name) is getattr(enums, name)


def test_champs_derives_et_extensions_historiques_preservees():
    assert contract.INPUT_FIELDS == set(schemas.INPUT_SCHEMA["properties"])
    assert contract.REQUIRED_INPUT_FIELDS == {"request_id", "sensor_id", "timestep", "traffic_data"}
    assert contract.OPTIONAL_INPUT_FIELDS == contract.INPUT_FIELDS - contract.REQUIRED_INPUT_FIELDS
    assert contract.OUTPUT_FIELDS == contract.REQUIRED_OUTPUT_FIELDS == contract.RESPONSE_KEYS
    assert "metadata" in contract.OUTPUT_FIELDS
    assert "integration_test_fixture" in contract.OPTIONAL_INPUT_FIELDS
    for name, fields in contract.CONTEXT_FIELDS.items():
        assert fields == set(contract.INPUT_SCHEMA["properties"][name]["properties"])


def test_contrat_identique_au_temoin_avant_refactorisation():
    before = json.loads((ROOT / "tests/fixtures/traffic_policy_before.json").read_text(encoding="utf-8"))
    assert contract.INPUT_SCHEMA == before["input_schema"]
    assert contract.OUTPUT_SCHEMA == before["output_schema"]
    assert contract.validate_input is validation.validate_input
    assert contract.validate_output is validation.validate_output


@pytest.mark.parametrize("context", list(contract.CONTEXT_FIELDS))
def test_contextes_absents_null_et_partiels(context):
    for value in (None, {}):
        message = contract.load_example("input")
        message[context] = value
        assert contract.validate_input(message)
    message.pop(context)
    assert contract.validate_input(message)


def test_exemples_et_copies_du_schema_live():
    assert contract.validate_input(contract.load_example("input"))
    assert contract.validate_output(contract.load_example("output_warning"))
    assert contract.validate_output(contract.load_example("output_error"))
    original = copy.deepcopy(contract.INPUT_SCHEMA)
    live = contract.get_input_schema("live")
    live["properties"]["sensor_id"]["maximum"] = -1
    assert contract.INPUT_SCHEMA == original


def test_responsabilites_documentees():
    document = (ROOT / "docs/traffic_prediction_agent_contract.md").read_text(encoding="utf-8")
    assert "ne prédit pas les bus" in document
    assert "ne calcule pas les routes" in document
    for name in contract.CONTEXT_FIELDS:
        assert name in document

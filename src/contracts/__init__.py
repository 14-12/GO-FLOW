"""Contrat d'intégration officiel, importable sans charger de modèle ou de données."""
from importlib import import_module

from .enums import (
    ACTIONS, AGENT_NAME, Action, DecisionSource, PredictionMode, ResponseStatus,
    RiskLevel, TrafficState)


# Les imports différés évitent un cycle entre la politique de décision et les schémas.
_EXPORTS = {
    "INPUT_SCHEMA": ("schemas", "INPUT_SCHEMA"),
    "OUTPUT_SCHEMA": ("schemas", "OUTPUT_SCHEMA"),
    "get_input_schema": ("schemas", "get_input_schema"),
    "validate_input": ("validation", "validate_input"),
    "validate_output": ("validation", "validate_output"),
    "load_example": ("examples", "load_example"),
}
# Exposer les vues du point d'entrée d'intégration sans changer les anciens exports.
for _name in (
    "INPUT_FIELDS", "OUTPUT_FIELDS", "REQUIRED_INPUT_FIELDS",
    "OPTIONAL_INPUT_FIELDS", "REQUIRED_OUTPUT_FIELDS", "CONTEXT_FIELDS",
    "RESPONSE_KEYS", "TRAFFIC_STATES", "RISK_LEVELS", "RESPONSE_STATUSES",
    "DECISION_SOURCES",
):
    _EXPORTS[_name] = ("traffic_prediction_contract", _name)
__all__ = ["ACTIONS", "AGENT_NAME", "Action", "DecisionSource", "PredictionMode",
           "ResponseStatus", "RiskLevel", "TrafficState", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value

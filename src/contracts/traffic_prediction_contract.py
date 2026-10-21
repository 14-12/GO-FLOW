"""Point d'entrée public du contrat, sans duplication des définitions existantes.

Les schémas restent définis dans schemas, les valeurs dans enums et les contrôles
dans validation. Les anciens imports désignent les mêmes objets pour compatibilité.
Importer ce module ne charge aucun modèle, donnée de trafic ou backend LLM.
"""
from .enums import (
    ACTIONS, AGENT_NAME, DECISION_SOURCES, RESPONSE_STATUSES, RISK_LEVELS,
    TRAFFIC_STATES, Action, DecisionSource, PredictionMode, ResponseStatus,
    RiskLevel, TrafficState,
)
from .schemas import (
    CONTEXT_FIELDS, INPUT_SCHEMA, OUTPUT_SCHEMA, RESPONSE_KEYS, get_input_schema,
)
from .validation import validate_input, validate_output
from .examples import load_example


# Ces vues sont dérivées des schémas : aucune seconde liste de champs à maintenir.
INPUT_FIELDS = frozenset(INPUT_SCHEMA["properties"])
OUTPUT_FIELDS = frozenset(OUTPUT_SCHEMA["properties"])
REQUIRED_INPUT_FIELDS = frozenset(INPUT_SCHEMA["required"])
OPTIONAL_INPUT_FIELDS = INPUT_FIELDS - REQUIRED_INPUT_FIELDS
REQUIRED_OUTPUT_FIELDS = frozenset(OUTPUT_SCHEMA["required"])

__all__ = [
    "AGENT_NAME", "INPUT_SCHEMA", "OUTPUT_SCHEMA", "get_input_schema",
    "INPUT_FIELDS", "OUTPUT_FIELDS", "REQUIRED_INPUT_FIELDS",
    "OPTIONAL_INPUT_FIELDS", "REQUIRED_OUTPUT_FIELDS", "CONTEXT_FIELDS",
    "RESPONSE_KEYS", "ACTIONS", "TRAFFIC_STATES", "RISK_LEVELS",
    "RESPONSE_STATUSES", "DECISION_SOURCES", "Action", "TrafficState",
    "RiskLevel", "ResponseStatus", "DecisionSource", "PredictionMode",
    "validate_input", "validate_output", "load_example",
]

"""Valeurs du contrat public existant, indépendantes du forecasting et du LLM."""
from enum import StrEnum


AGENT_NAME = "traffic_prediction_control"


class Action(StrEnum):
    KEEP_CURRENT_SIGNAL_PLAN = "keep_current_signal_plan"
    MONITOR = "monitor"
    EXTEND_GREEN_PHASE = "extend_green_phase"
    REDUCE_GREEN_PHASE = "reduce_green_phase"
    COORDINATE_WITH_NEIGHBOR = "coordinate_with_neighbor"
    MANUAL_REVIEW = "manual_review"


class TrafficState(StrEnum):
    FREE_FLOW = "free_flow"
    MODERATE = "moderate"
    CONGESTED = "congested"
    SEVERE = "severe"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionSource(StrEnum):
    LLM = "llm"
    RULE_BASED_FALLBACK = "rule_based_fallback"


class ResponseStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


class PredictionMode(StrEnum):
    HISTORICAL_REPLAY = "historical_replay"
    LIVE = "live"


# Les anciennes constantes restent des ensembles de chaînes, sans changement de valeurs.
ACTIONS = frozenset(item.value for item in Action)
TRAFFIC_STATES = frozenset(item.value for item in TrafficState)
RISK_LEVELS = frozenset(item.value for item in RiskLevel)
DECISION_SOURCES = frozenset(item.value for item in DecisionSource)
RESPONSE_STATUSES = frozenset(item.value for item in ResponseStatus)
DECISION_KEYS = frozenset({"traffic_state", "risk_level", "recommended_action", "reason", "confidence"})

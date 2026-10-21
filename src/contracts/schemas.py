"""Schémas JSON publics existants et variante explicite du mode live."""
import copy

from src.decision import policy
from .enums import AGENT_NAME, Action, DecisionSource, PredictionMode, ResponseStatus, RiskLevel, TrafficState

CONTEXT_FIELDS = {"incident_context": {"incident_detected", "incident_type", "severity"},
                  "emergency_context": {"emergency_vehicle_present", "priority_level"},
                  "cooperative_context": {"neighboring_sensors"},
                  "public_transport_context": {"priority_requested"}}
RESPONSE_KEYS = {"agent", "request_id", "sensor_id", "forecast", "traffic_state", "risk_level",
                 "recommended_action", "reason", "confidence", "decision_source", "status", "metadata"}

# Les schémas publiés restent des dictionnaires sérialisables pour les autres agents.
INPUT_SCHEMA = {
    "type": "object", "required": ["request_id", "sensor_id", "timestep", "traffic_data"],
    "additionalProperties": False, "properties": {
        "request_id": {"type": "string", "minLength": policy.NONEMPTY_TEXT_MIN_LENGTH, "maxLength": policy.REQUEST_ID_MAX_LENGTH},
        "sensor_id": {"type": "integer", "minimum": policy.SENSOR_ID_MINIMUM, "maximum": policy.SENSOR_ID_MAXIMUM},
        "timestep": {"type": "integer", "minimum": policy.HISTORICAL_FIRST_TIMESTEP, "maximum": policy.HISTORICAL_LAST_TIMESTEP},
        "timestamp": {"type": ["string", "null"]},
        "traffic_data": {"type": "object", "required": ["current_flow", "current_speed", "current_occupancy"],
                         "additionalProperties": False, "properties": {
                             k: {"type": "number"} for k in
                             ["current_flow", "current_speed", "current_occupancy"]}},
        "integration_test_fixture": {"type": "boolean"}}}
for context, fields in CONTEXT_FIELDS.items():
    properties = {}
    for field in fields:
        if field in {"incident_detected", "emergency_vehicle_present", "priority_requested"}:
            properties[field] = {"type": ["boolean", "null"]}
        elif field in {"severity", "priority_level"}:
            properties[field] = {"enum": [item.value for item in RiskLevel] + [None]}
        elif field == "neighboring_sensors":
            properties[field] = {"type": ["array", "null"], "items": {
                "type": "object", "additionalProperties": False,
                "required": ["sensor_id", "timestep", "current_speed", "current_occupancy"],
                "properties": {"sensor_id": {"type": "integer", "minimum": policy.SENSOR_ID_MINIMUM, "maximum": policy.SENSOR_ID_MAXIMUM},
                               "timestep": {"type": "integer"}, "current_speed": {"type": "number"},
                               "current_occupancy": {"type": "number"},
                               "predicted_occupancy_t15": {"type": "number"}}}}
        else:
            properties[field] = {"type": ["string", "null"], "minLength": policy.NONEMPTY_TEXT_MIN_LENGTH}
    INPUT_SCHEMA["properties"][context] = {"type": ["object", "null"],
                                           "properties": properties, "additionalProperties": False}
OUTPUT_SCHEMA = {"type": "object", "required": sorted(RESPONSE_KEYS), "additionalProperties": False,
                 "properties": {
                     "agent": {"const": AGENT_NAME},
                     "request_id": {"type": ["string", "null"]},
                     "sensor_id": {"type": ["integer", "null"]},
                     "forecast": {"type": "object", "required": ["horizon_minutes", "predicted_occupancy"],
                                  "additionalProperties": False, "properties": {
                                      "horizon_minutes": {"const": policy.FORECAST_HORIZON_MINUTES},
                                      "predicted_occupancy": {"type": ["number", "null"]}}},
                     "traffic_state": {"enum": [item.value for item in TrafficState] + [None]},
                     "risk_level": {"enum": [item.value for item in RiskLevel] + [None]},
                     "recommended_action": {"enum": [item.value for item in Action]},
                     "reason": {"type": "string", "minLength": policy.NONEMPTY_TEXT_MIN_LENGTH},
                     "confidence": {"type": "number", "minimum": policy.CONFIDENCE_MINIMUM, "maximum": policy.CONFIDENCE_MAXIMUM},
                     "decision_source": {"enum": [item.value for item in DecisionSource]},
                     "status": {"enum": [item.value for item in ResponseStatus]},
                     "metadata": {"type": "object"}}}


def get_input_schema(prediction_mode="historical_replay"):
    """Renvoyer une copie indépendante du schéma propre au mode demandé."""
    if prediction_mode not in {PredictionMode.LIVE, PredictionMode.HISTORICAL_REPLAY}:
        raise ValueError("prediction_mode doit être live ou historical_replay.")
    schema = copy.deepcopy(INPUT_SCHEMA)
    if prediction_mode == PredictionMode.LIVE:
        schema["properties"]["timestep"] = {"type": "integer", "minimum": policy.LIVE_FIRST_TIMESTEP}
    return schema

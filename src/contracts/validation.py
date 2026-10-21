"""Validation Python du contrat public ; aucun modèle ni état spatial n'est chargé."""
import json
import math
import numbers
from datetime import datetime

from src.decision import policy
from .enums import (
    ACTIONS, AGENT_NAME, DECISION_KEYS, DECISION_SOURCES, RESPONSE_STATUSES,
    RISK_LEVELS, TRAFFIC_STATES)
from .schemas import CONTEXT_FIELDS, INPUT_SCHEMA, OUTPUT_SCHEMA, RESPONSE_KEYS, get_input_schema


def _validate_wire_schema(value, schema):
    """Appliquer le schéma publié à une valeur sérialisable en JSON strict."""
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import ValidationError

    try:
        wire_value = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Valeur non sérialisable en JSON strict : {error}") from error
    try:
        Draft202012Validator(schema).validate(wire_value)
    except ValidationError as error:
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        raise ValueError(f"Contrat JSON invalide ({path}) : {error.message}") from error



def is_finite_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def is_integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def strict_json_loads(text):
    def reject_constant(value):
        raise ValueError(f"Constante JSON invalide : {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Clé JSON dupliquée : {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def decision_field_checks(data, *, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS, actions=ACTIONS):
    """Décrire les mêmes contrôles de champs pour le validateur et la démonstration."""
    data = data if isinstance(data, dict) else {}
    confidence = data.get("confidence")
    return {
        "traffic_state": isinstance(data.get("traffic_state"), str) and data["traffic_state"] in traffic_states,
        "risk_level": isinstance(data.get("risk_level"), str) and data["risk_level"] in risk_levels,
        "recommended_action": isinstance(data.get("recommended_action"), str) and data["recommended_action"] in actions,
        "reason": isinstance(data.get("reason"), str) and bool(data["reason"].strip()),
        "confidence": (not isinstance(confidence, bool) and isinstance(confidence, numbers.Real)
                       and math.isfinite(confidence)
                       and policy.CONFIDENCE_MINIMUM <= confidence <= policy.CONFIDENCE_MAXIMUM)}


def validate_decision(output, *, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS,
                      actions=ACTIONS, decision_keys=DECISION_KEYS):
    # Refuser les champs supplémentaires, NaN, booléens comme confiance et JSON dupliqué.
    try:
        data = strict_json_loads(output) if isinstance(output, str) else output
        if not isinstance(data, dict) or set(data) != decision_keys:
            return False
        return all(decision_field_checks(data, traffic_states=traffic_states, risk_levels=risk_levels, actions=actions).values())
    except (ValueError, TypeError):
        return False


def validate_input(message, *, prediction_mode="historical_replay", input_schema=None, check_schema=True):
    """Lever ValueError si le message est invalide ; renvoyer True sinon.

    check_schema=False conserve le contrôle historique de l'adaptateur agent.
    Les contraintes du graphe et de concordance avec les observations exigent
    toujours build_state : ce validateur indépendant ne les invente pas.
    """
    if prediction_mode not in {"historical_replay", "live"}:
        raise ValueError("prediction_mode doit être live ou historical_replay.")
    schema = get_input_schema(prediction_mode) if input_schema is None else input_schema
    if not isinstance(message, dict):
        raise ValueError("Message entrant : objet JSON requis.")
    if set(message) - set(schema["properties"]):
        raise ValueError("Champs entrants inconnus.")
    if not set(schema["required"]).issubset(message):
        raise ValueError("request_id, sensor_id, timestep et traffic_data sont obligatoires.")
    if not isinstance(message["request_id"], str) or not message["request_id"].strip() or len(message["request_id"]) > policy.REQUEST_ID_MAX_LENGTH:
        raise ValueError("request_id : chaîne non vide de 200 caractères maximum.")
    if not is_integer(message["sensor_id"]) or not policy.SENSOR_ID_MINIMUM <= message["sensor_id"] < policy.SENSOR_COUNT:
        raise ValueError("sensor_id : entier entre 0 et 169 requis.")
    if prediction_mode == "historical_replay":
        if not is_integer(message["timestep"]) or not policy.HISTORICAL_FIRST_TIMESTEP <= message["timestep"] <= policy.HISTORICAL_LAST_TIMESTEP:
            raise ValueError("timestep hors de la période test exportée 15177–17853.")
    elif not is_integer(message["timestep"]) or message["timestep"] < policy.LIVE_FIRST_TIMESTEP:
        raise ValueError("Le mode live exige un timestep entier >= 13, dans le repère original.")
    if "integration_test_fixture" in message and not isinstance(message["integration_test_fixture"], bool):
        raise ValueError("integration_test_fixture doit être booléen.")
    stamp = message.get("timestamp")
    if stamp is not None:
        if not isinstance(stamp, str):
            raise ValueError("timestamp doit être ISO 8601 avec fuseau ou null.")
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp sans fuseau : refusé.")
    traffic = message["traffic_data"]
    expected = set(INPUT_SCHEMA["properties"]["traffic_data"]["properties"])
    if not isinstance(traffic, dict) or set(traffic) != expected or not all(is_finite_number(x) for x in traffic.values()):
        raise ValueError("traffic_data : trois mesures numériques finies requises.")
    for name, fields in CONTEXT_FIELDS.items():
        context = message.get(name)
        if context is None:
            continue
        if not isinstance(context, dict) or set(context) - fields:
            raise ValueError(f"{name} : objet avec champs autorisés ou null requis.")
        for field, value in context.items():
            if field in {"incident_detected", "emergency_vehicle_present", "priority_requested"}:
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"{field} : true, false ou null requis.")
            elif field in {"severity", "priority_level"}:
                if value is not None and (not isinstance(value, str) or value not in policy.CONTEXT_PRIORITY_LEVELS):
                    raise ValueError(f"{field} : low, medium, high ou null requis.")
            elif field == "incident_type":
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError("incident_type : texte non vide ou null requis.")
            elif field == "neighboring_sensors":
                if value is not None and not isinstance(value, list):
                    raise ValueError("neighboring_sensors : liste ou null requise.")
        if name == "incident_context" and context.get("incident_detected") is not True:
            if context.get("incident_type") is not None or context.get("severity") is not None:
                raise ValueError("Incident non confirmé : type et gravité doivent rester null.")
        if name == "emergency_context" and context.get("emergency_vehicle_present") is not True:
            if context.get("priority_level") is not None:
                raise ValueError("Urgence non confirmée : priorité doit rester null.")
    if check_schema:
        _validate_wire_schema(message, schema)
    return True


def validate_output(response, *, validator=validate_decision, decision_keys=DECISION_KEYS,
                    response_keys=RESPONSE_KEYS, check_schema=True):
    """Renvoyer un booléen ; contrôler aussi les invariants warning/ok/error existants."""
    try:
        json.dumps(response, allow_nan=False)
        if not isinstance(response, dict) or set(response) != response_keys:
            return False
        if response["agent"] != AGENT_NAME or response["status"] not in RESPONSE_STATUSES:
            return False
        if response["decision_source"] not in DECISION_SOURCES:
            return False
        if response["request_id"] is not None and not isinstance(response["request_id"], str):
            return False
        if response["sensor_id"] is not None and (not is_integer(response["sensor_id"]) or not policy.SENSOR_ID_MINIMUM <= response["sensor_id"] < policy.SENSOR_COUNT):
            return False
        if not isinstance(response["metadata"], dict):
            return False
        forecast = response["forecast"]
        if not isinstance(forecast, dict) or set(forecast) != {"horizon_minutes", "predicted_occupancy"} or forecast["horizon_minutes"] != policy.FORECAST_HORIZON_MINUTES:
            return False
        if response["status"] == "error":
            valid = (forecast["predicted_occupancy"] is None and response["traffic_state"] is None
                    and response["risk_level"] is None and response["recommended_action"] == "manual_review"
                    and isinstance(response["reason"], str) and bool(response["reason"])
                    and is_finite_number(response["confidence"]) and response["confidence"] == policy.ERROR_CONFIDENCE)
        else:
            keys = decision_keys() if callable(decision_keys) else decision_keys
            decision = {key: response[key] for key in keys}
            valid = is_finite_number(forecast["predicted_occupancy"]) and validator(decision)
        if valid and check_schema:
            _validate_wire_schema(response, OUTPUT_SCHEMA)
        return valid
    except Exception:
        return False

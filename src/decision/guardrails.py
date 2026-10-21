"""Validateurs et garde-fous existants, sans nouvelle règle de décision."""
import json
import math
import numbers
import re

from . import policy
from src.contracts import validation as contract_validation

TRAFFIC_STATES = policy.TRAFFIC_STATES
RISK_LEVELS = policy.RISK_LEVELS
ACTIONS = policy.ACTIONS
DECISION_KEYS = policy.DECISION_KEYS


def is_finite_number(value):
    return contract_validation.is_finite_number(value)


def is_integer(value):
    return contract_validation.is_integer(value)


def strict_json_loads(text):
    return contract_validation.strict_json_loads(text)


def decision_field_checks(data, *, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS, actions=ACTIONS):
    return contract_validation.decision_field_checks(
        data, traffic_states=traffic_states, risk_levels=risk_levels, actions=actions)


def validate_decision(output, *, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS,
                      actions=ACTIONS, decision_keys=DECISION_KEYS):
    return contract_validation.validate_decision(
        output, traffic_states=traffic_states, risk_levels=risk_levels,
        actions=actions, decision_keys=decision_keys)


def is_grounded_decision(output, state, *, validator, contract_builder, parser=strict_json_loads):
    # Le chemin strict impose les raisons exactes et la confiance du contrat existant.
    if not validator(output):
        return False
    data = parser(output) if isinstance(output, str) else output
    contract = contract_builder(state)
    return (data["traffic_state"] == contract["traffic_state"]
            and data["risk_level"] == contract["risk_level"]
            and data["recommended_action"] in contract["allowed_actions"]
            and data["reason"] in contract["allowed_reasons"]
            and data["confidence"] == contract["confidence"])


def extract_json(text):
    # Extraire un seul objet ; aucune correction de valeur, de clé ou de ponctuation.
    start = text.find("{")
    if start < 0:
        raise ValueError("Aucun objet JSON dans la réponse brute.")
    decoder = json.JSONDecoder(parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                               object_pairs_hook=_unique_pairs)
    result, end = decoder.raw_decode(text[start:])
    if "{" in text[start + end:] or "}" in text[start + end:]:
        raise ValueError("Plusieurs blocs ou accolades résiduelles : extraction ambiguë.")
    if not isinstance(result, dict):
        raise ValueError("La réponse JSON n'est pas un objet.")
    return result


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Clé dupliquée : {key}")
        result[key] = value
    return result


def validate_llm_decision(output, *, validator=validate_decision):
    # Réutiliser le validateur strict de 08 ; le LLM doit fournir une confiance float.
    return isinstance(output, dict) and validator(output) and isinstance(output["confidence"], float)


def qualitative_checks(output, state, *, rules, validator=None):
    valid = (validate_llm_decision(output, validator=rules.validate_decision)
             if validator is None else validator(output))
    reason = output.get("reason", "") if isinstance(output, dict) else ""
    reason = reason if isinstance(reason, str) else ""
    missing_sentence = "Incidents inconnus; urgences inconnues; plan de feux inconnu."
    missing_respected = missing_sentence in reason
    remaining = reason.replace(missing_sentence, "").lower()
    # Un contrôle lexical conservateur : aucune affirmation incident/urgence/voisin en texte libre.
    event_claim = bool(re.search(r"incident|accident|urgence|emergency|crash", remaining))
    neighbor_claim = bool(re.search(r"voisin|neighbor", remaining))
    hallucination = event_claim or neighbor_claim
    numbers_in_reason = re.findall(r"[-+]?\d+(?:[.,]\d+)?", reason)
    supported = [state["current_occupancy"], state["predicted_occupancy_t15"], policy.FORECAST_HORIZON_MINUTES]
    numeric_supported = all(any(abs(float(token.replace(",", ".")) - value) <= policy.NARRATIVE_NUMERIC_TOLERANCE
                               for value in supported) for token in numbers_in_reason)
    current = f"{state['current_occupancy']:.{policy.REASON_DECIMAL_PLACES}f}"
    forecast = f"{state['predicted_occupancy_t15']:.{policy.REASON_DECIMAL_PLACES}f}"
    grounded = current in reason and forecast in reason and numeric_supported and not hallucination
    action = output.get("recommended_action") if isinstance(output, dict) else None
    risk = output.get("risk_level") if isinstance(output, dict) else None
    traffic = output.get("traffic_state") if isinstance(output, dict) else None
    allowed = policy.LLM_ACTIONS_BY_RISK
    neighbor_support = rules.indicators(state)["congested_neighbors"] > policy.CONGESTED_NEIGHBOR_COUNT_THRESHOLD
    coherent = (isinstance(risk, str) and isinstance(action, str) and isinstance(traffic, str)
                and risk in allowed and action in allowed[risk]
                and (action != "coordinate_with_neighbor" or neighbor_support)
                and policy.TRAFFIC_RISK_LEVELS.get(traffic) == risk)
    # Les règles définissent un plancher de risque expérimental, jamais abaissé par le LLM.
    rank = policy.RISK_RANK
    coherent = coherent and rank.get(risk, policy.UNKNOWN_RISK_RANK) >= rank[rules.rule_based_decision(state)["risk_level"]]
    return {"json_valid": valid, "authorized_action": valid and action in rules.ACTIONS,
            "no_unsupported_claim_detected": not hallucination and numeric_supported,
            "reason_grounded": grounded, "risk_action_coherent": coherent,
            "confidence_valid": valid, "missing_information_respected": missing_respected,
            "hallucination_detected": hallucination or not numeric_supported,
            "hallucination_check_applicable": isinstance(output, dict)}


def is_critical_priority(state):
    # Toute urgence confirmée verrouille la décision ; un incident le fait si high ou inconnu.
    return (state["emergency_vehicle_present"] is True or
            (state["incident_detected"] is True
             and state["incident_context"]["severity"] in policy.CRITICAL_INCIDENT_SEVERITIES))


def protect_critical_priority(decision, state, *, critical_check, rule_decision):
    # Recontrôler après le LLM, même si une priorité critique aurait dû empêcher son appel.
    if critical_check(state):
        return rule_decision(state), True
    return decision, False


def validate_agent_decision(decision, *, rules):
    try:
        json.dumps(decision, allow_nan=False)
        return rules is not None and rules.validate_decision(decision)
    except Exception:
        return False


def validate_contextual_decision(decision, state, *, validator, rule_decision):
    # Le contrat enrichi conserve exactement état, risque, action et raison déterministes.
    if not validator(decision):
        return False
    baseline = rule_decision(state)
    # Réutiliser le grounding strict de 08, avec des raisons adaptées au contexte externe.
    return (decision["traffic_state"] == baseline["traffic_state"]
            and decision["risk_level"] == baseline["risk_level"]
            and decision["recommended_action"] == baseline["recommended_action"]
            and decision["reason"] == baseline["reason"])


def validate_response(response, *, agent, response_keys):
    return contract_validation.validate_output(
        response, validator=agent.validate,
        decision_keys=lambda: agent.rules.DECISION_KEYS,
        response_keys=response_keys, check_schema=False)


def validation_report(output, state, *, agent, rules):
    """Présenter les diagnostics existants ; la décision reste validée par l'agent."""
    quality = qualitative_checks(output, state, rules=rules)
    fields = agent.rules.decision_field_checks(output)
    candidate = output if isinstance(output, dict) else {}
    reason = candidate.get("reason", "")
    reason = reason if isinstance(reason, str) else ""
    remaining = reason.replace(
        "Incidents inconnus; urgences inconnues; plan de feux inconnu.", "").lower()
    valid = agent.validate(output)
    return {
        "JSON valide": valid,
        "traffic_state valide": fields["traffic_state"],
        "risk_level valide": fields["risk_level"],
        "action autorisée": fields["recommended_action"],
        "raison suffisamment fondée": quality["reason_grounded"],
        "aucune hallucination incident détectée": valid and not bool(re.search(r"incident|accident|crash", remaining)),
        "aucune hallucination emergency détectée": valid and not bool(re.search(r"urgence|emergency", remaining)),
        "cohérence risque/action": quality["risk_action_coherent"],
        "informations manquantes signalées": quality["missing_information_respected"],
        "contrat enrichi de l’agent respecté": agent._guarded_valid(output, state)}


def validate_priority_policy(values):
    if set(values) != set(policy.DEFAULT_PRIORITY_POLICY) or not all(is_finite_number(x) for x in values.values()):
        raise ValueError("Politique expérimentale : cinq priorités numériques finies requises.")
    # Les priorités critiques doivent rester au-dessus des niveaux non critiques.
    if not (values["emergency"] > values["incident"] >
            max(values[k] for k in ["severe_congestion", "public_transport", "normal"])):
        raise ValueError("La politique doit préserver emergency > incident > niveaux non critiques.")

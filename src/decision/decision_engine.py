"""Règles déterministes et orchestration LLM/fallback de GoFlow."""
import copy

from . import guardrails, policy


def indicators(state, thresholds):
    """Adaptateur historique vers les indicateurs de la politique metier."""
    return policy.indicators(state, thresholds)


def rule_based_decision(state, thresholds=None, *, indicator_fn=None):
    """Conserver les points d injection historiques des indicateurs."""
    callback = indicator_fn if indicator_fn is not None else lambda value: indicators(value, thresholds)
    return policy.rule_based_decision(state, indicator_fn=callback)


def decision_contract(state, thresholds=None, *, rule_decision_fn=None, indicator_fn=None):
    baseline = (rule_decision_fn(state) if rule_decision_fn is not None
                else rule_based_decision(state, thresholds))
    flags = indicator_fn(state) if indicator_fn is not None else indicators(state, thresholds)
    extra = (f"Tendance passée de l'occupation={state['recent_occupancy_trend']:+.{policy.REASON_DECIMAL_PLACES}f} "
             f"et de la vitesse={state['recent_speed_trend']:+.{policy.REASON_DECIMAL_PLACES}f} par timestep; "
             f"{len(state['neighbors'])} voisins directs. "
             "Incidents et urgences inconnus; plan de feux indisponible.")
    actions = {"low": list(policy.LOW_RISK_ALLOWED_ACTIONS), "medium": list(policy.MEDIUM_RISK_ALLOWED_ACTIONS),
               "high": [baseline["recommended_action"], "manual_review"]}[baseline["risk_level"]]
    return {"traffic_state": baseline["traffic_state"], "risk_level": baseline["risk_level"],
            "allowed_actions": sorted(set(actions)), "allowed_reasons": [baseline["reason"], extra],
            "confidence": baseline["confidence"], "indicators": flags}


def final_decision(state, *, build_traffic_state, llm_decision, is_grounded_decision,
                   rule_based_decision, strict_json_loads=guardrails.strict_json_loads):
    """Conserver le chemin canonique du notebook 08 avec ses dépendances explicites."""
    # Reconstruire l'état depuis les exports pour refuser une entrée modifiée ou fictive.
    canonical = build_traffic_state(state["sensor_id"], state["timestep"])
    if state != canonical:
        raise ValueError("État fourni différent des observations et prévisions sauvegardées.")
    reason = "llm_unavailable"
    try:
        result = llm_decision(copy.deepcopy(canonical))
        if result is not None:
            if is_grounded_decision(result, canonical):
                output = strict_json_loads(result) if isinstance(result, str) else result
                return {"decision": output, "source": "llm", "fallback_reason": None}
            reason = "invalid_or_ungrounded_llm_output"
    except (OSError, ValueError, TypeError, KeyError) as error:
        reason = f"llm_error:{type(error).__name__}"
    output = rule_based_decision(canonical)
    assert is_grounded_decision(output, canonical)
    return {"decision": output, "source": "rule_based", "fallback_reason": reason}


def priority_for_state(state, *, rule_decision, priority_policy):
    """Deleguer la hierarchie metier sans modifier la politique injectee."""
    return policy.priority_for_state(state, rule_decision=rule_decision, priority_policy=priority_policy)


def context_reason(state):
    """Conserver le texte exact du contexte externe."""
    return policy.context_reason(state)


def contextual_rule_based_decision(state, *, rule_decision, priority_for, context_reason):
    """Deleguer les regles contextuelles avec les callbacks existants."""
    return policy.contextual_rule_based_decision(
        state, rule_decision=rule_decision, priority_for=priority_for, context_reason=context_reason)


def final_agent_decision(context, state):
    """Orchestrer le contrat enrichi de l'agent, son backend et son replay éventuel.

    Le contexte fournit les adaptateurs de règles/garde-fous, la politique, le
    backend et le cache existants. Aucun modèle ni donnée n'est chargé ici.
    Ses points d'injection et sa trace restent ceux de TrafficPredictionControlAgent.
    """
    baseline = context._rule_decision(state)
    trace = {"priority": context._priority(state), "priority_rank": context.policy[context._priority(state)],
             "llm_mode": "not_called", "fallback_reason": None, "llm_error": None,
             "critical_guard_before": context._critical(state), "critical_guard_after": False}
    candidate = None
    if context._critical(state):
        trace["fallback_reason"] = "critical_priority_guard_before_llm"
    elif trace["priority"] in policy.LLM_BYPASS_PRIORITIES:
        trace["fallback_reason"] = "external_priority_requires_orchestrator_review"
    else:
        try:
            if context.llm_backend is not None:
                trace["llm_mode"] = "injected_backend"
                candidate = context.llm_backend(copy.deepcopy(state))
            else:
                cached = context._cached.get((state["sensor_id"], state["timestep"]))
                # Les réponses sauvegardées ne décrivent aucun contexte externe connu.
                external_known = any(state[field] is not None for field in
                                     ["incident_detected", "emergency_vehicle_present"])
                if cached and not external_known:
                    trace["llm_mode"] = "replay_saved_09_no_new_inference"
                    candidate = copy.deepcopy(cached.get("llm_output"))
                    trace["cached_original_fallback"] = cached.get("fallback_reason")
                else:
                    trace["fallback_reason"] = "llm_unavailable_for_this_context"
            if candidate is None and trace["fallback_reason"] is None:
                trace["fallback_reason"] = "llm_returned_no_output"
            if isinstance(candidate, str):
                candidate = context.rules.strict_json_loads(candidate)
            if candidate is not None and not context._guarded_valid(candidate, state):
                trace["fallback_reason"] = "llm_output_invalid_or_not_grounded"
                candidate = None
        except Exception as error:
            trace["llm_error"] = f"{type(error).__name__}: {error}"
            trace["fallback_reason"] = "llm_backend_error"
            candidate = None
    decision, locked = context._post_guard(candidate or baseline, state)
    trace["critical_guard_after"] = locked
    source = "llm" if candidate is not None and not locked else "rule_based_fallback"
    assert context.validate(decision)
    context.last_trace = trace
    return {"decision": decision, "decision_source": source, "trace": copy.deepcopy(trace)}


def final_saved_scenario_decision(state, *, inputs, llm_backend, qualitative_checker, rule_decision):
    """Conserver le contrat qualitatif des cinq scénarios de 09, sans le renforcer."""
    canonical = next((item["state"] for item in inputs
                      if item["state"]["sensor_id"] == state["sensor_id"]
                      and item["state"]["timestep"] == state["timestep"]), None)
    if canonical is None or canonical != state:
        raise ValueError("Seuls les cinq états sauvegardés, inchangés, sont autorisés.")
    output, error = None, None
    try:
        output = llm_backend(copy.deepcopy(state))
        checks = qualitative_checker(output, state)
        if all(value for key, value in checks.items() if key != "hallucination_detected"):
            return {"decision": output, "source": "llm", "fallback_reason": None,
                    "llm_output": output, "validation": checks}
        error = "Critères refusés : " + ", ".join(key for key, value in checks.items()
                                                  if key != "hallucination_detected" and not value)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    checks = qualitative_checker(output, state)
    return {"decision": rule_decision(state), "source": "rule_based_fallback",
            "fallback_reason": error, "llm_output": output, "validation": checks}

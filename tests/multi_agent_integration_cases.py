"""Scénarios d'intégration : mesures réelles de 08, contextes externes simulés marqués."""
from pathlib import Path
import copy
import json


class MockOrchestrator:
    """Orchestrateur simulé de test, pas l'orchestrateur final du groupe."""

    def __init__(self, agent):
        self.agent = agent
        self.messages = []
        self.responses = []

    def send(self, message):
        self.messages.append(copy.deepcopy(message))
        response = self.agent.handle_message(copy.deepcopy(message))
        assert self.agent.validate_response(response), "Réponse non conforme au contrat JSON"
        assert json.loads(json.dumps(response, allow_nan=False)) == response
        self.responses.append(response)
        return response


def run_integration_cases(agent, root):
    inputs = json.loads((Path(root) / "results/decision_agent/agent_inputs.json").read_text(encoding="utf-8"))
    states = {row["scenario"]: row["state"] for row in inputs}
    orchestrator = MockOrchestrator(agent)
    results = []

    def message(name, state_name="trafic_fluide", fixture=False):
        state = states[state_name]
        return {"request_id": name, "timestamp": None, "sensor_id": state["sensor_id"],
                "timestep": state["timestep"],
                "traffic_data": {k: state[k] for k in ["current_flow", "current_speed", "current_occupancy"]},
                "incident_context": {"incident_detected": None, "incident_type": None, "severity": None},
                "emergency_context": {"emergency_vehicle_present": None, "priority_level": None},
                "public_transport_context": {"priority_requested": None},
                "cooperative_context": {"neighboring_sensors": None},
                "integration_test_fixture": fixture}

    def check(name, msg, expected_status="warning", expected_action=None, expected_priority=None):
        try:
            schema_valid = agent.validate_message(msg)
        except Exception:
            schema_valid = False
        response = orchestrator.send(msg)
        passed = response["status"] == expected_status
        if expected_action:
            passed &= response["recommended_action"] == expected_action
        if expected_priority:
            passed &= response["metadata"].get("priority") == expected_priority
        if expected_status != "error":
            passed &= response["forecast"]["predicted_occupancy"] == agent.predict(msg["sensor_id"], msg["timestep"])
        results.append({"test": name, "passed": bool(passed), "input_valid_expected": expected_status != "error",
                        "input_schema_valid": schema_valid,
                        "response_contract_valid": agent.validate_response(response), "status": response["status"],
                        "decision_source": response["decision_source"],
                        "error_controlled": response["status"] == "error" and response["metadata"].get("error_controlled", False),
                        "integration_test_fixture": msg.get("integration_test_fixture", False),
                        "llm_mode": response["metadata"].get("llm_mode"),
                        "critical_guard_before": response["metadata"].get("critical_guard_before", False),
                        "critical_guard_after": response["metadata"].get("critical_guard_after", False)})
        return response

    check("trafic_normal", message("trafic_normal"), expected_action="keep_current_signal_plan", expected_priority="normal")
    check("forte_congestion_prevue", message("forte_congestion_prevue", "forte_hausse_predite"),
          expected_action="coordinate_with_neighbor", expected_priority="severe_congestion")
    incident = message("incident_externe", fixture=True)
    incident["incident_context"] = {"incident_detected": True, "incident_type": "integration_test_fixture", "severity": "high"}
    check("incident_externe", incident, expected_action="manual_review", expected_priority="incident")
    emergency = message("urgence_externe", fixture=True)
    emergency["emergency_context"] = {"emergency_vehicle_present": True, "priority_level": "high"}
    check("urgence_externe", emergency, expected_action="manual_review", expected_priority="emergency")
    cooperative = message("voisins_congestionnes", "voisins_contrastes")
    cooperative["cooperative_context"]["neighboring_sensors"] = copy.deepcopy(states["voisins_contrastes"]["neighbors"])
    check("voisins_congestionnes", cooperative, expected_action="monitor")
    missing = message("contextes_absents")
    for field in ["incident_context", "emergency_context", "public_transport_context", "cooperative_context"]:
        missing.pop(field)
    check("contextes_absents", missing, expected_action="keep_current_signal_plan")
    check("message_invalide", {"integration_test_fixture": True}, expected_status="error")

    transit = message("priorite_transport_public", fixture=True)
    transit["public_transport_context"]["priority_requested"] = True
    check("priorite_transport_public", transit, expected_action="manual_review", expected_priority="public_transport")
    both = copy.deepcopy(incident)
    both["request_id"] = "urgence_prioritaire_sur_incident"
    both["emergency_context"] = copy.deepcopy(emergency["emergency_context"])
    check("urgence_prioritaire_sur_incident", both, expected_action="manual_review", expected_priority="emergency")
    for name, mutate in [
        ("trafic_modifie", lambda m: m["traffic_data"].update(current_occupancy=m["traffic_data"]["current_occupancy"] + 1)),
        ("bool_capteur", lambda m: m.update(sensor_id=True)),
        ("horizon_hors_export", lambda m: m.update(timestep=20000)),
        ("boolean_externe_invalide", lambda m: m["emergency_context"].update(emergency_vehicle_present="true")),
        ("contexte_contradictoire", lambda m: m["incident_context"].update(incident_detected=False, severity="high")),
        ("voisin_arbitraire", lambda m: m["cooperative_context"].update(neighboring_sensors=[
            {"sensor_id": m["sensor_id"], "timestep": m["timestep"], "current_speed": 0.0, "current_occupancy": 0.0}])),
        ("timestamp_sans_fuseau", lambda m: m.update(timestamp="2026-10-13T12:00:00"))]:
        invalid = message(name, fixture=True)
        mutate(invalid)
        check(name, invalid, expected_status="error")

    # Un backend simulé essaie de casser une priorité critique : il ne doit jamais être appelé.
    original_backend = agent.llm_backend
    calls = []
    try:
        def hostile(state):
            calls.append(state)
            return {"traffic_state": "free_flow", "risk_level": "low", "recommended_action": "reduce_green_phase",
                    "reason": "Sortie hostile integration_test_fixture", "confidence": 0.99}
        agent.llm_backend = hostile
        guarded = copy.deepcopy(emergency)
        guarded["request_id"] = "garde_fou_avant_llm"
        response = check("garde_fou_avant_llm", guarded, expected_action="manual_review", expected_priority="emergency")
        results[-1]["passed"] &= not calls and response["metadata"]["critical_guard_before"] and response["metadata"]["critical_guard_after"]
        # Tester séparément le garde-fou après LLM pour une sortie qui aurait échappé au contrôle avant.
        state = agent.build_state(guarded)
        corrected, locked = agent._post_guard(hostile(state), state)
        results.append({"test": "garde_fou_apres_llm", "passed": locked and corrected["recommended_action"] == "manual_review"
                        and corrected["risk_level"] == "high", "input_valid_expected": None,
                        "response_contract_valid": agent.validate(corrected), "status": "guard_unit_test",
                        "decision_source": "rule_based_fallback", "error_controlled": False,
                        "integration_test_fixture": True, "llm_mode": "synthetic_hostile_fixture"})
        for name, error in [("timeout_llm", TimeoutError("integration_test_fixture")),
                            ("erreur_llm", RuntimeError("integration_test_fixture"))]:
            def failed(state, exception=error):
                raise exception
            agent.llm_backend = failed
            response = check(name, message(name, fixture=True), expected_action="keep_current_signal_plan")
            results[-1]["passed"] &= response["decision_source"] == "rule_based_fallback" and type(error).__name__ in response["metadata"]["llm_error"]
        agent.llm_backend = lambda state: {"risk_level": "low"}
        check("json_llm_invalide", message("json_llm_invalide", fixture=True), expected_action="keep_current_signal_plan")
        agent.llm_backend = lambda state: None
        response = check("llm_sans_sortie", message("llm_sans_sortie", fixture=True), expected_action="keep_current_signal_plan")
        results[-1]["passed"] &= response["metadata"]["fallback_reason"] == "llm_returned_no_output"
        # Chemin accepté simulé : test du contrat seulement, pas une nouvelle inférence Qwen.
        agent.llm_backend = lambda state: agent._rule_decision(state)
        response = check("backend_valide_simule", message("backend_valide_simule", fixture=True), expected_action="keep_current_signal_plan")
        results[-1]["passed"] &= response["decision_source"] == "llm"
    finally:
        agent.llm_backend = original_backend
    # Une réponse modifiée par un consommateur ne doit pas modifier la politique interne.
    response_policy = orchestrator.responses[0]["metadata"]["priority_policy_experimental"]
    saved_policy = dict(response_policy)
    response_policy["emergency"] = -1
    isolated = agent.policy == saved_policy
    orchestrator.responses[0]["metadata"]["priority_policy_experimental"] = saved_policy
    results.append({"test": "isolement_politique_reponse", "passed": isolated,
                    "input_valid_expected": None, "response_contract_valid": True,
                    "status": "guard_unit_test", "decision_source": "rule_based_fallback",
                    "error_controlled": False, "integration_test_fixture": True,
                    "llm_mode": "response_mutation_fixture"})
    assert all(row["passed"] for row in results), [r["test"] for r in results if not r["passed"]]
    return orchestrator, results

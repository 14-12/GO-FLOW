"""Contrat orchestrateur et garde-fous de l'agent Traffic Prediction & Control."""
from pathlib import Path
import copy
import io
import json
import logging
import math
import numbers
from contextlib import redirect_stdout

from .decision_agent import create_runtime
from src.decision import decision_engine, guardrails, policy
from src.contracts import validation as contract_validation
from src.contracts.enums import AGENT_NAME
from src.contracts.traffic_prediction_contract import CONTEXT_FIELDS, INPUT_SCHEMA, OUTPUT_SCHEMA, RESPONSE_KEYS, get_input_schema

PRIORITY_POLICY = dict(policy.DEFAULT_PRIORITY_POLICY)



def _finite(value):
    return guardrails.is_finite_number(value)


def _integer(value):
    return guardrails.is_integer(value)


class TrafficPredictionControlAgent:
    """Prévision et décision PEMS08 vers l'orchestrateur, sans commande de feux.

    llm_backend : callable optionnel recevant un état enrichi, renvoyant un dict/JSON.
    Sans backend, les cinq réponses réelles de 09 sont rejouées et validées ; ailleurs,
    le fallback s'applique. Un replay n'est jamais annoncé comme nouvelle inférence.
    prediction_mode : historical_replay par défaut pour la compatibilité ; live
    utilise XGBoost et history_provider pour chaque capteur et ses voisins directs.
    En live, aucune ancienne sortie LLM n'est rejouée implicitement.
    """

    def __init__(self, project_root=None, llm_backend=None, priority_policy=None, logger=None,
                 *, prediction_mode="historical_replay", predictor=None, history_provider=None):
        self.root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        if prediction_mode not in {"live", "historical_replay"}:
            raise ValueError("prediction_mode doit être live ou historical_replay.")
        if prediction_mode == "live" and not callable(history_provider):
            raise ValueError("Le mode live exige history_provider(sensor_id, timestep), avec t−12 à t.")
        self.prediction_mode = prediction_mode
        self.predictor = predictor
        self.history_provider = history_provider
        # Le schéma historique reste inchangé pour les consommateurs et tests existants.
        self.input_schema = get_input_schema(prediction_mode)
        self.policy = dict(PRIORITY_POLICY if priority_policy is None else priority_policy)
        guardrails.validate_priority_policy(self.policy)
        self.llm_backend = llm_backend
        self.logger = logger or logging.getLogger(__name__)
        self.last_trace = {}
        self._init_error = None
        self._cached = {}
        try:
            # Isoler les règles et les données de chaque instance, depuis le paquet src.
            self.rules = create_runtime()
            if self.prediction_mode == "live":
                from src.prediction import XGBoostTrafficPredictor
                self.predictor = predictor if predictor is not None else XGBoostTrafficPredictor(self.root)
                self.rules.initialize_live(self.root)
                # Les anciennes réponses LLM ne décrivent pas les nouvelles observations.
                return
            with redirect_stdout(io.StringIO()):
                self.rules.initialize(self.root)
            path = self.root / "results/llm_agent/llm_scenario_outputs.json"
            if path.exists():
                for row in json.loads(path.read_text(encoding="utf-8")):
                    state = row["state"]
                    canonical = self.rules.build_traffic_state(state["sensor_id"], state["timestep"])
                    if state != canonical:
                        raise ValueError("Les états LLM sauvegardés ne correspondent plus aux données.")
                    self._cached[(state["sensor_id"], state["timestep"])] = row
        except Exception as error:
            self._init_error = f"{type(error).__name__}: {error}"

    def validate_message(self, message):
        # Conserver le contrôle historique ; build_state valide ensuite le contexte spatial.
        return contract_validation.validate_input(
            message, prediction_mode=self.prediction_mode,
            input_schema=self.input_schema, check_schema=False)

    def predict(self, sensor_id, timestep, *, history=None):
        # Choisir explicitement l'inférence ou la compatibilité historique, sans fit.
        if self._init_error:
            raise RuntimeError(self._init_error)
        if not _integer(sensor_id) or not _integer(timestep):
            raise ValueError("sensor_id et timestep doivent être entiers.")
        if self.prediction_mode == "live":
            if history is None:
                history = self.history_provider(int(sensor_id), int(timestep))
            return self.predictor.predict(history, sensor_id=sensor_id, timestep=timestep)
        row = self.rules.LOOKUP.loc[(int(sensor_id), int(timestep))]
        return float(row.predicted_occupancy_t15)

    def _live_observation(self, sensor_id, timestep):
        history = self.history_provider(sensor_id, timestep)
        # Le prédicteur valide capteur, grille temporelle, causalité et complétude.
        forecast = self.predict(sensor_id, timestep, history=history)
        current = history.iloc[-1]
        past = history.iloc[-(policy.TREND_HISTORY_LENGTH + 1):-1]
        return {"sensor_id": sensor_id, "timestep": timestep,
                "current_flow": float(current.flow), "current_speed": float(current.speed),
                "current_occupancy": float(current.occupy), "predicted_occupancy_t15": forecast,
                "occupancy_change": forecast - float(current.occupy),
                "recent_occupancy_trend": (float(past.occupy.iloc[-1]) - float(past.occupy.iloc[0])) / policy.TREND_INTERVAL_COUNT,
                "recent_speed_trend": (float(past.speed.iloc[-1]) - float(past.speed.iloc[0])) / policy.TREND_INTERVAL_COUNT}

    def _build_live_state(self, sensor_id, timestep):
        state = self._live_observation(sensor_id, timestep)
        neighbors = []
        for neighbor in (self.rules.ADJACENCY[sensor_id] > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD).nonzero()[0]:
            if neighbor == sensor_id:
                continue
            observation = self._live_observation(int(neighbor), timestep)
            neighbors.append({key: observation[key] for key in
                              ("sensor_id", "timestep", "current_speed", "current_occupancy",
                               "predicted_occupancy_t15")})
        state.update(neighbors=neighbors, incident_detected=None, emergency_vehicle_present=None)
        return state

    def build_state(self, message):
        self.validate_message(message)
        if self._init_error:
            raise RuntimeError(self._init_error)
        builder = self._build_live_state if self.prediction_mode == "live" else self.rules.build_traffic_state
        state = builder(int(message["sensor_id"]), int(message["timestep"]))
        for field, value in message["traffic_data"].items():
            if not math.isclose(float(value), state[field], rel_tol=policy.STATE_RELATIVE_TOLERANCE, abs_tol=policy.STATE_ABSOLUTE_TOLERANCE):
                origin = "l'historique causal fourni" if self.prediction_mode == "live" else "le snapshot de la prévision sauvegardée"
                raise ValueError(f"{field} incompatible avec {origin}.")
        contexts = {name: {field: (message.get(name) or {}).get(field) for field in fields}
                    for name, fields in CONTEXT_FIELDS.items()}
        external_neighbors = contexts["cooperative_context"]["neighboring_sensors"]
        if external_neighbors is not None:
            actual = {n["sensor_id"]: n for n in state["neighbors"]}
            seen = set()
            required = {"sensor_id", "timestep", "current_speed", "current_occupancy"}
            for neighbor in external_neighbors:
                if not isinstance(neighbor, dict) or not required.issubset(neighbor) or set(neighbor) - required - {"predicted_occupancy_t15"}:
                    raise ValueError("Voisin externe : objet sensor_id/timestep/speed/occupancy requis.")
                identifier = neighbor["sensor_id"]
                if not _integer(identifier) or identifier not in actual or identifier in seen:
                    raise ValueError("Voisin inconnu du graphe, identifiant invalide ou dupliqué.")
                seen.add(identifier)
                if not _integer(neighbor["timestep"]) or neighbor["timestep"] != state["timestep"]:
                    raise ValueError("Voisin externe à un autre timestep : refusé.")
                for field in ["current_speed", "current_occupancy", "predicted_occupancy_t15"]:
                    if field in neighbor and (not _finite(neighbor[field]) or not math.isclose(
                            float(neighbor[field]), actual[identifier][field], rel_tol=policy.STATE_RELATIVE_TOLERANCE, abs_tol=policy.STATE_ABSOLUTE_TOLERANCE)):
                        raise ValueError("Mesure voisin externe incohérente avec le snapshot PEMS08.")
        # Ne jamais remplacer les voisins réels par une liste externe arbitraire.
        state.update(contexts)
        state["incident_detected"] = contexts["incident_context"]["incident_detected"]
        state["emergency_vehicle_present"] = contexts["emergency_context"]["emergency_vehicle_present"]
        state["integration_test_fixture"] = message.get("integration_test_fixture", False)
        return state

    def _priority(self, state):
        return decision_engine.priority_for_state(
            state, rule_decision=self.rules.rule_based_decision, priority_policy=self.policy)

    def _context_reason(self, state):
        return decision_engine.context_reason(state)

    def _rule_decision(self, state):
        return decision_engine.contextual_rule_based_decision(
            state, rule_decision=self.rules.rule_based_decision,
            priority_for=self._priority, context_reason=self._context_reason)

    def _critical(self, state):
        return guardrails.is_critical_priority(state)

    def _post_guard(self, decision, state):
        return guardrails.protect_critical_priority(
            decision, state, critical_check=self._critical, rule_decision=self._rule_decision)

    def validate(self, decision):
        return guardrails.validate_agent_decision(decision, rules=getattr(self, "rules", None))

    def _guarded_valid(self, decision, state):
        return guardrails.validate_contextual_decision(
            decision, state, validator=self.validate, rule_decision=self._rule_decision)

    def decide(self, state):
        return decision_engine.final_agent_decision(self, state)

    def validate_response(self, response):
        return guardrails.validate_response(response, agent=self, response_keys=RESPONSE_KEYS)

    def handle_message(self, message):
        self.last_trace = {}
        try:
            state = self.build_state(message)
            result = self.decide(state)
            warnings = [f"Mapping location=index NPZ sous hypothèse pour {policy.AMBIGUOUS_SENSOR_COUNT} capteurs indiscernables.",
                        "Aucun plan de feux disponible ; recommandation sans commande exécutée."]
            if result["decision_source"] == "rule_based_fallback":
                warnings.append("Décision déterministe : " + str(result["trace"]["fallback_reason"]))
            response = {"agent": AGENT_NAME, "request_id": message["request_id"],
                        "sensor_id": int(message["sensor_id"]),
                        "forecast": {"horizon_minutes": policy.FORECAST_HORIZON_MINUTES, "predicted_occupancy": state["predicted_occupancy_t15"]},
                        **result["decision"], "decision_source": result["decision_source"], "status": "warning",
                        "metadata": {"timestep": state["timestep"], "priority_policy_experimental": dict(self.policy),
                                     **result["trace"], "warnings": warnings,
                                     "integration_test_fixture": state["integration_test_fixture"]}}
            if self.prediction_mode == "live":
                response["metadata"]["prediction_mode"] = "live"
            if not self.validate_response(response):
                raise RuntimeError("Contrat de réponse non respecté.")
        except Exception as error:
            safe = message if isinstance(message, dict) else {}
            response = {"agent": AGENT_NAME,
                        "request_id": safe.get("request_id") if isinstance(safe.get("request_id"), str) else None,
                        "sensor_id": int(safe["sensor_id"]) if _integer(safe.get("sensor_id")) and policy.SENSOR_ID_MINIMUM <= safe["sensor_id"] < policy.SENSOR_COUNT else None,
                        "forecast": {"horizon_minutes": policy.FORECAST_HORIZON_MINUTES, "predicted_occupancy": None},
                        "traffic_state": None, "risk_level": None, "recommended_action": "manual_review",
                        "reason": f"{type(error).__name__}: {error}", "confidence": policy.ERROR_CONFIDENCE,
                        "decision_source": "rule_based_fallback", "status": "error",
                        "metadata": {"integration_test_fixture": safe.get("integration_test_fixture") is True,
                                     "error_controlled": True}}
        # La journalisation ne doit jamais casser le retour à l'orchestrateur.
        try:
            self.logger.info(json.dumps({"request_id": response["request_id"], "sensor_id": response["sensor_id"],
                                         "forecast": response["forecast"], "decision_source": response["decision_source"],
                                         "action": response["recommended_action"], "validation": self.validate_response(response),
                                         "fallback": response["metadata"].get("fallback_reason"),
                                         "llm_mode": response["metadata"].get("llm_mode"),
                                         "integration_test_fixture": response["metadata"]["integration_test_fixture"],
                                         "error": response["reason"] if response["status"] == "error" else None},
                                        ensure_ascii=False, allow_nan=False))
        except Exception:
            pass
        return response

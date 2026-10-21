"""Valeurs existantes de la politique déterministe, sans nouveau seuil.

Les seuils numériques de trafic restent calculés sur train ou relus dans les
artifacts existants. TrafficThresholds leur donne des noms explicites sans les
figer ici et sans changer les dictionnaires CONFIG/THRESHOLDS des anciens appels.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.contracts.enums import ACTIONS, DECISION_KEYS, RISK_LEVELS, TRAFFIC_STATES, Action


# Quantiles expérimentaux déjà utilisés sur l'occupation de train.
OCCUPANCY_MEDIAN_QUANTILE = 0.50
OCCUPANCY_HIGH_QUANTILE = 0.90
OCCUPANCY_SEVERE_QUANTILE = 0.97
# La vitesse basse est le quartile inférieur de train, dans les unités source.
SPEED_LOW_QUANTILE = 0.25
# La hausse forte utilise Q90 des variations absolues d'occupation à trois pas.
OCCUPANCY_CHANGE_HIGH_QUANTILE = 0.90
# Période train purgée existante ; aucun seuil n'est ajusté sur validation/test.
TRAIN_FIRST_TIMESTEP = 13
TRAIN_LAST_TIMESTEP = 12497
VALIDATION_FIRST_TIMESTEP = 12501
# Les tendances excluent t et comparent t−6 à t−1, soit cinq intervalles.
TREND_HISTORY_LENGTH = 6
TREND_INTERVAL_COUNT = 5
# Le signe strict de la pente suffit : aucun seuil d'amplitude supplémentaire.
TREND_ZERO_THRESHOLD = 0
# Le test > 0 signifie au moins un voisin à occupation élevée.
CONGESTED_NEIGHBOR_COUNT_THRESHOLD = 0
# Toute arête de poids strictement positif est conservée, même très petite.
DIRECT_NEIGHBOR_WEIGHT_THRESHOLD = 0

# Confiance expérimentale constante, non calibrée ; ce n'est pas une probabilité.
RULE_BASED_CONFIDENCE = 0.5
# Bornes inclusives du contrat JSON ; NaN, infinis et booléens restent refusés.
CONFIDENCE_MINIMUM = 0
CONFIDENCE_MAXIMUM = 1
# Une erreur contrôlée ne produit aucune décision exploitable.
ERROR_CONFIDENCE = 0.0
# Les justifications existantes affichent les mesures à six décimales.
REASON_DECIMAL_PLACES = 6

# Rangs expérimentaux existants ; emergency > incident > autres niveaux.
EMERGENCY_PRIORITY = 100
INCIDENT_PRIORITY = 80
SEVERE_CONGESTION_PRIORITY = 60
PUBLIC_TRANSPORT_PRIORITY = 40
NORMAL_PRIORITY = 20
DEFAULT_PRIORITY_POLICY = MappingProxyType({
    "emergency": EMERGENCY_PRIORITY, "incident": INCIDENT_PRIORITY,
    "severe_congestion": SEVERE_CONGESTION_PRIORITY,
    "public_transport": PUBLIC_TRANSPORT_PRIORITY, "normal": NORMAL_PRIORITY})
# Ces priorités imposent une revue manuelle ; les deux premières imposent high.
ORCHESTRATOR_REVIEW_PRIORITIES = frozenset({"emergency", "incident", "public_transport"})
HIGH_RISK_OVERRIDE_PRIORITIES = frozenset({"emergency", "incident"})
# Les urgences/incidents critiques sont traités avant ; ces autres contextes évitent aussi le LLM.
LLM_BYPASS_PRIORITIES = frozenset({"incident", "public_transport"})
# Un incident confirmé high ou de gravité inconnue verrouille le garde-fou critique.
CRITICAL_INCIDENT_SEVERITIES = frozenset({"high", None})
CONTEXT_PRIORITY_LEVELS = RISK_LEVELS

# Les valeurs du contrat viennent de src.contracts ; les sous-ensembles sont la politique existante.
LOW_RISK_ALLOWED_ACTIONS = (Action.KEEP_CURRENT_SIGNAL_PLAN.value, Action.MONITOR.value)
MEDIUM_RISK_ALLOWED_ACTIONS = (Action.MONITOR.value,)

# Contrat qualitatif existant du backend 09, distinct du grounding strict de 08/10.
LLM_ACTIONS_BY_RISK = MappingProxyType({
    "low": frozenset({Action.KEEP_CURRENT_SIGNAL_PLAN.value, Action.MONITOR.value}),
    "medium": frozenset({Action.MONITOR.value, Action.MANUAL_REVIEW.value}),
    "high": frozenset({Action.MANUAL_REVIEW.value, Action.COORDINATE_WITH_NEIGHBOR.value})})
# Ordre déjà fourni au schéma de génération du backend local, sans action de phase.
LLM_GENERATION_ALLOWED_ACTIONS = (
    Action.KEEP_CURRENT_SIGNAL_PLAN.value, Action.MONITOR.value,
    Action.COORDINATE_WITH_NEIGHBOR.value, Action.MANUAL_REVIEW.value)
# Longueur maximale existante de la justification dans ce schéma de génération.
LLM_REASON_MAX_LENGTH = 220
TRAFFIC_RISK_LEVELS = MappingProxyType({
    "free_flow": "low", "moderate": "medium", "congested": "high", "severe": "high"})
# Le niveau LLM ne peut pas descendre sous celui de la règle déterministe.
RISK_RANK = MappingProxyType({"low": 0, "medium": 1, "high": 2})
UNKNOWN_RISK_RANK = -1
# Tolérance déjà appliquée aux nombres cités dans les justifications à six décimales.
NARRATIVE_NUMERIC_TOLERANCE = 0.000001

# Contrat PEMS08 existant : identifiants, horizon et plage du replay historique.
SENSOR_ID_MINIMUM = 0
SENSOR_ID_MAXIMUM = 169
SENSOR_COUNT = 170
FORECAST_HORIZON_MINUTES = 15
FORECAST_HORIZON_STEPS = 3
HISTORICAL_FIRST_TIMESTEP = 15177
HISTORICAL_LAST_TIMESTEP = 17853
HISTORICAL_TIMESTEP_COUNT = 2677
HISTORICAL_PREDICTION_COUNT = 455090
SOURCE_TIMESTEP_ORIGIN = 1
SOURCE_TIMESTEP_COUNT = 17856
TRAFFIC_FEATURE_COUNT = 3
SPEED_FEATURE_INDEX = 1
OCCUPANCY_FEATURE_INDEX = 2
SPATIAL_EDGE_COUNT = 274
AMBIGUOUS_SENSOR_COUNT = 25
# Le mode live exige le lag maximal 12 et l'observation courante, soit t >= 13.
LIVE_FIRST_TIMESTEP = 13
# Contraintes de texte déjà appliquées au message et aux raisons.
NONEMPTY_TEXT_MIN_LENGTH = 1
REQUEST_ID_MAX_LENGTH = 200
# Tolérances d'alignement des messages/voisins avec l'état disponible.
STATE_RELATIVE_TOLERANCE = 1e-9
STATE_ABSOLUTE_TOLERANCE = 1e-9
# Tolérances plus strictes déjà utilisées pour auditer l'export contre le CSV brut.
SOURCE_RELATIVE_TOLERANCE = 1e-12
SOURCE_ABSOLUTE_TOLERANCE = 1e-12


def default_rule_config():
    """Créer les valeurs déterministes historiques, sans partager la liste mutable."""
    return {
        "trend_history": TREND_HISTORY_LENGTH,
        "train_first_timestep": TRAIN_FIRST_TIMESTEP,
        "train_last_timestep": TRAIN_LAST_TIMESTEP,
        "occupancy_quantiles": [OCCUPANCY_MEDIAN_QUANTILE, OCCUPANCY_HIGH_QUANTILE,
                                OCCUPANCY_SEVERE_QUANTILE],
        "low_speed_quantile": SPEED_LOW_QUANTILE,
        "change_quantile": OCCUPANCY_CHANGE_HIGH_QUANTILE}


@dataclass(frozen=True)
class TrafficThresholds:
    """Noms des seuils réels, résolus à chaque appel depuis THRESHOLDS."""

    values: Mapping[str, float]

    @property
    def occupancy_median_threshold(self):
        """Médiane train ; le passage à moderate utilise strictement >."""
        return self.values["occupancy_median"]

    @property
    def occupancy_high_threshold(self):
        """Q90 train ; comparaison inclusive >= pour le capteur et les voisins."""
        return self.values["occupancy_high"]

    @property
    def occupancy_severe_threshold(self):
        """Q97 train ; comparaison inclusive, prioritaire sur les autres conditions."""
        return self.values["occupancy_severe"]

    @property
    def speed_low_threshold(self):
        """Q25 de vitesse train ; la faible vitesse utilise <=, unités source."""
        return self.values["speed_low"]

    @property
    def occupancy_change_high_threshold(self):
        """Q90 des changements absolus ; >= pour la hausse, <= à l'opposé pour l'amélioration."""
        return self.values["strong_change"]

    @property
    def neighbor_occupancy_threshold(self):
        """Le voisin utilise exactement le même Q90 que le capteur, sans seuil distinct."""
        return self.occupancy_high_threshold

    @classmethod
    def from_mapping(cls, thresholds: Mapping[str, float]):
        # Garder les lectures paresseuses des anciens branchements, sans nouvelle validation.
        return cls(values=thresholds)


# Regles metier historiques : extraction sans modification des conditions.
def indicators(state, thresholds):
    # Ne convertir ni occupancy en pourcentage ni speed en unité non documentée.
    thresholds = TrafficThresholds.from_mapping(thresholds)
    peak = max(state["current_occupancy"], state["predicted_occupancy_t15"])
    neighbors = state["neighbors"]
    congested = sum(n["current_occupancy"] >= thresholds.neighbor_occupancy_threshold for n in neighbors)
    values = [n["current_occupancy"] for n in neighbors]
    return {"relative_occupancy": "severe" if peak >= thresholds.occupancy_severe_threshold else
            "high" if peak >= thresholds.occupancy_high_threshold else
            "moderate" if peak > thresholds.occupancy_median_threshold else "low",
            "low_speed": state["current_speed"] <= thresholds.speed_low_threshold,
            "strong_increase": state["occupancy_change"] >= thresholds.occupancy_change_high_threshold,
            "improving": state["occupancy_change"] <= -thresholds.occupancy_change_high_threshold,
            "occupancy_rising_in_past": state["recent_occupancy_trend"] > TREND_ZERO_THRESHOLD,
            "speed_falling_in_past": state["recent_speed_trend"] < TREND_ZERO_THRESHOLD,
            "congested_neighbors": congested,
            "neighbor_occupancy_range": float(max(values) - min(values)) if values else None}


def rule_based_decision(state, thresholds=None, *, indicator_fn=None):
    """Appliquer la règle existante ; l'adaptateur historique peut injecter ses indicateurs."""
    flags = indicator_fn(state) if indicator_fn is not None else indicators(state, thresholds)
    if flags["relative_occupancy"] == "severe":
        # Q97 est prioritaire, même si une baisse future est également indiquée.
        traffic, risk = "severe", "high"
    elif flags["relative_occupancy"] == "high" or flags["strong_increase"]:
        # Q90 d'occupation ou hausse forte suffisent pour le risque élevé.
        traffic, risk = "congested", "high"
    elif (flags["relative_occupancy"] == "moderate" or flags["low_speed"]
          or flags["congested_neighbors"] > CONGESTED_NEIGHBOR_COUNT_THRESHOLD
          or (flags["occupancy_rising_in_past"] and flags["speed_falling_in_past"])):
        traffic, risk = "moderate", "medium"
    else:
        # Sans indicateur défavorable, conserver le plan courant comme recommandation.
        traffic, risk = "free_flow", "low"
    if risk == "high":
        # Une coordination exige au moins un voisin à occupation élevée.
        action = "coordinate_with_neighbor" if flags["congested_neighbors"] else "manual_review"
    elif risk == "medium":
        action = "monitor"
    else:
        action = "keep_current_signal_plan"
    reason = (f"Occupation actuelle={state['current_occupancy']:.{REASON_DECIMAL_PLACES}f}, "
              f"prévue à +{FORECAST_HORIZON_MINUTES} min="
              f"{state['predicted_occupancy_t15']:.{REASON_DECIMAL_PLACES}f}, "
              f"variation={state['occupancy_change']:+.{REASON_DECIMAL_PLACES}f}; "
              f"vitesse={state['current_speed']:.{REASON_DECIMAL_PLACES}f}; voisins à occupation élevée="
              f"{flags['congested_neighbors']}/{len(state['neighbors'])}. "
              "Incidents et urgences inconnus; plan de feux indisponible.")
    # Score de complétude expérimental, constant : aucune probabilité calibrée revendiquée.
    return {"traffic_state": traffic, "risk_level": risk, "recommended_action": action,
            "reason": reason, "confidence": RULE_BASED_CONFIDENCE}


def priority_for_state(state, *, rule_decision, priority_policy):
    # Conserver le maximum des rangs injectés et l'ordre existant en cas d'égalité.
    active = ["normal"]
    if state["emergency_vehicle_present"] is True:
        active.append("emergency")
    if state["incident_detected"] is True:
        active.append("incident")
    if rule_decision(state)["traffic_state"] == "severe":
        active.append("severe_congestion")
    if state["public_transport_context"]["priority_requested"] is True:
        active.append("public_transport")
    return max(active, key=lambda name: priority_policy[name])


def context_reason(state):
    incident = state["incident_detected"]
    emergency = state["emergency_vehicle_present"]
    transit = state["public_transport_context"]["priority_requested"]
    return ("Incident " + ("signalé par le contexte externe" if incident is True else
                           "non signalé par le contexte externe" if incident is False else "inconnu")
            + "; urgence " + ("signalée par le contexte externe" if emergency is True else
                               "non signalée par le contexte externe" if emergency is False else "inconnue")
            + "; priorité transport public " + ("demandée" if transit is True else
                                                "non demandée" if transit is False else "inconnue")
            + "; plan de feux indisponible.")


def contextual_rule_based_decision(state, *, rule_decision, priority_for, context_reason):
    decision = rule_decision(state)
    # Remplacer seulement la clause d'informations absentes, selon les contextes reçus.
    decision["reason"] = decision["reason"].split("Incidents et urgences inconnus;")[0] + context_reason(state)
    priority = priority_for(state)
    if priority in ORCHESTRATOR_REVIEW_PRIORITIES:
        decision["recommended_action"] = "manual_review"
        if priority in HIGH_RISK_OVERRIDE_PRIORITIES:
            decision["risk_level"] = "high"
        elif decision["risk_level"] == "low":
            decision["risk_level"] = "medium"
        decision["reason"] += f" Priorité expérimentale {priority} : arbitrage orchestrateur requis."
    return decision


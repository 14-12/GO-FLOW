"""Her deterministic decision rules, copied from src/decision/policy.py (feature/traffic-prediction-agent, 392fbae).

`indicators` and `rule_based_decision` keep her conditions exactly (Q97 severe first, Q90 or strong rise congested,
then moderate on median / low speed / a congested neighbour / rising occupancy with falling speed). Changed: the
horizon in the reason is in SUMO seconds, and the thresholds are recomputed on the Town05 train seeds with her
quantiles (`fit_thresholds`); nothing is tuned on dev or test.
"""
from __future__ import annotations

import numpy as np

# Her quantiles (src/decision/policy.py), applied to the Town05 train seeds.
OCCUPANCY_QUANTILES = (0.50, 0.90, 0.97)
SPEED_LOW_QUANTILE = 0.25
OCCUPANCY_CHANGE_HIGH_QUANTILE = 0.90
TREND_HISTORY_LENGTH = 6
TREND_ZERO_THRESHOLD = 0
CONGESTED_NEIGHBOR_COUNT_THRESHOLD = 0
RULE_BASED_CONFIDENCE = 0.5
REASON_DECIMAL_PLACES = 6
TRAFFIC_RISK_LEVELS = {"free_flow": "low", "moderate": "medium", "congested": "high", "severe": "high"}


def fit_thresholds(occ: np.ndarray, speed: np.ndarray, change: np.ndarray) -> dict:
    """occ, speed: train values; change: |occ(t+k) - occ(t)| on train windows (her strong_change at her horizon)."""
    o, s, c = occ[np.isfinite(occ)], speed[np.isfinite(speed)], change[np.isfinite(change)]
    q50, q90, q97 = np.quantile(o, OCCUPANCY_QUANTILES)
    return {"occupancy_median": float(q50), "occupancy_high": float(q90), "occupancy_severe": float(q97),
            "speed_low": float(np.quantile(s, SPEED_LOW_QUANTILE)) if len(s) else float("nan"),
            "speed_median": float(np.median(s)) if len(s) else float("nan"),
            "strong_change": float(np.quantile(c, OCCUPANCY_CHANGE_HIGH_QUANTILE))}


def indicators(state, th):
    peak = max(state["current_occupancy"], state["predicted_occupancy"])
    neighbors = state["neighbors"]
    congested = sum(n["current_occupancy"] >= th["occupancy_high"] for n in neighbors)
    values = [n["current_occupancy"] for n in neighbors]
    speed = state["current_speed"]
    return {"relative_occupancy": "severe" if peak >= th["occupancy_severe"] else
            "high" if peak >= th["occupancy_high"] else
            "moderate" if peak > th["occupancy_median"] else "low",
            "low_speed": speed is not None and np.isfinite(speed) and speed <= th["speed_low"],
            "strong_increase": state["occupancy_change"] >= th["strong_change"],
            "improving": state["occupancy_change"] <= -th["strong_change"],
            "occupancy_rising_in_past": state["recent_occupancy_trend"] > TREND_ZERO_THRESHOLD,
            "speed_falling_in_past": state["recent_speed_trend"] < TREND_ZERO_THRESHOLD,
            "congested_neighbors": congested,
            "neighbor_occupancy_range": float(max(values) - min(values)) if values else None}


def rule_based_decision(state, th, horizon_s: int) -> dict:
    flags = indicators(state, th)
    if flags["relative_occupancy"] == "severe":
        traffic, risk = "severe", "high"
    elif flags["relative_occupancy"] == "high" or flags["strong_increase"]:
        traffic, risk = "congested", "high"
    elif (flags["relative_occupancy"] == "moderate" or flags["low_speed"]
          or flags["congested_neighbors"] > CONGESTED_NEIGHBOR_COUNT_THRESHOLD
          or (flags["occupancy_rising_in_past"] and flags["speed_falling_in_past"])):
        traffic, risk = "moderate", "medium"
    else:
        traffic, risk = "free_flow", "low"
    if risk == "high":
        action = "coordinate_with_neighbor" if flags["congested_neighbors"] else "manual_review"
    elif risk == "medium":
        action = "monitor"
    else:
        action = "keep_current_signal_plan"
    d = REASON_DECIMAL_PLACES
    speed = state["current_speed"]
    reason = (f"Occupation actuelle={state['current_occupancy']:.{d}f}, "
              f"prévue à +{horizon_s} s={state['predicted_occupancy']:.{d}f}, "
              f"variation={state['occupancy_change']:+.{d}f}; "
              f"vitesse={'inconnue' if speed is None or not np.isfinite(speed) else f'{speed:.{d}f}'}; "
              f"voisins à occupation élevée={flags['congested_neighbors']}/{len(state['neighbors'])}. "
              "Incidents et urgences inconnus; plan de feux indisponible.")
    return {"traffic_state": traffic, "risk_level": risk, "recommended_action": action,
            "reason": reason, "confidence": RULE_BASED_CONFIDENCE}

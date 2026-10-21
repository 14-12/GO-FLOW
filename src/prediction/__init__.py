"""Inférence sur les modèles GoFlow existants, sans entraînement."""
from .xgboost_predictor import XGBoostTrafficPredictor

__all__ = ["XGBoostTrafficPredictor"]

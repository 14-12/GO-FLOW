"""Saved Random Forest inference without labels, Colab, or LLM calls."""
from pathlib import Path

import joblib
import numpy as np

from .feature_engineering import CURRENT_COLUMNS, FEATURE_COLUMNS, add_temporal_features


class IncidentDetector:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models/incident_detector.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Incident model not found: {path}")
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.temporal_rule = artifact["temporal_rule"]
        self.feature_columns = tuple(artifact["feature_columns"])
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError("Saved model feature order differs from the notebook contract")
        if self.temporal_rule != "mean3":
            raise ValueError(f"Unsupported saved temporal rule: {self.temporal_rule}")

    def predict_windows(self, windows, source_id):
        frame = add_temporal_features(windows)
        if frame.empty:
            return []
        scores = self.model.predict_proba(frame.loc[:, list(self.feature_columns)])[:, 1]
        results = []
        for i, (_, row) in enumerate(frame.iterrows()):
            recent = [float(x) for x in scores[max(0, i - 2):i + 1]]
            results.append({
                "source_id": str(source_id),
                "window_start_sec": float(row.window_start_sec),
                "window_end_sec": float(row.window_end_sec),
                "incident_detected": bool(np.mean(recent) >= self.threshold),
                "incident_score": float(scores[i]),
                "recent_scores": recent,
                "temporal_rule": self.temporal_rule,
                "decision_threshold": self.threshold,
                "severity": "UNKNOWN",
                "traffic_features": {key: float(row[key]) for key in CURRENT_COLUMNS},
            })
        return results

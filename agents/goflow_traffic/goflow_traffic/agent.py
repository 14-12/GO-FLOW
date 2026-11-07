"""Her agent on Town05, step by step: 13 observations of history per approach, her XGBoost forecast, her rules.

Like her XGBoostTrafficPredictor, the model folder is checked on load (feature order, target, horizon, categories,
best iteration); nothing is trained here. The state each decision reads is her build_traffic_state: current flow,
speed, occupancy, the forecast, its change, trends over the 6 steps t-6 .. t-1, and the direct neighbours.
"""
from __future__ import annotations

import json
import os
from collections import deque

import numpy as np

from .features import FEATURE_COLUMNS, HISTORY, build
from .policy import TRAFFIC_RISK_LEVELS, TREND_HISTORY_LENGTH, rule_based_decision


class GoFlowTown05:
    def __init__(self, folder):
        from xgboost import XGBRegressor

        meta_p = os.path.join(folder, "xgboost_export_metadata.json")
        model_p = os.path.join(folder, "xgboost_model.json")
        if not os.path.isfile(model_p) or not os.path.isfile(meta_p):
            raise FileNotFoundError(f"no trained Town05 model in {folder} (train.py); nothing is trained here")
        with open(meta_p, encoding="utf-8") as f:
            self.meta = json.load(f)
        if tuple(self.meta["features"]) != FEATURE_COLUMNS:
            raise ValueError("features or their order differ from goflow_traffic.features")
        self.nodes = [str(n) for n in self.meta["nodes"]]
        self.N = len(self.nodes)
        self.col = {n: j for j, n in enumerate(self.nodes)}
        self.dt_s = float(self.meta["dt_s"])
        self.horizon_s = int(self.meta["horizon_s"])
        self.th = self.meta["thresholds_train_only"]
        self.nb = [[self.col[m] for m in self.meta["neighbours"].get(n, [])] for n in self.nodes]
        self.model = XGBRegressor(enable_categorical=True, device="cpu", n_jobs=2)
        self.model.load_model(model_p)
        booster = self.model.get_booster()
        if tuple(booster.feature_names or ()) != FEATURE_COLUMNS:
            raise ValueError("the saved model's feature order differs from the metadata")
        # her checks (src/prediction/xgboost_predictor.py): one categorical then floats, same best iteration
        if booster.feature_types != ["c"] + ["float"] * (len(FEATURE_COLUMNS) - 1):
            raise ValueError("feature types differ from the audited export (location categorical, then floats)")
        best = getattr(self.model, "best_iteration", None)
        if best is not None and int(best) != int(self.meta["best_iteration"]):
            raise ValueError("best iteration differs between the model and its metadata")
        self.iteration_range = (0, int(self.meta["best_iteration"]) + 1)
        self.version = self.meta["version"]
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=HISTORY)
        self.last_t = None

    def step(self, obs: dict):
        """-> (goflow message or None while warming up, problems)."""
        t = float(obs["t"])
        if self.last_t is not None and abs(t - self.last_t - self.dt_s) > 1e-6:
            self.reset()
        problems = []
        row = {v: np.full(self.N, np.nan) for v in ("flow", "speed", "occupy")}
        for n, v in (obs.get("nodes") or {}).items():
            j = self.col.get(n)
            if j is None:
                problems.append(f"unknown node {n}")
                continue
            if not isinstance(v, dict):
                problems.append(f"node {n}: not an object")
                continue
            for name, key in (("flow", "veh"), ("speed", "speed"), ("occupy", "occ")):
                x = v.get(key)
                if x is None:
                    continue                       # unknown: stays NaN, that approach gets no forecast
                # as her contract: booleans, strings, NaN and infinities are refused, never converted
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x):
                    problems.append(f"node {n}.{key}: not a finite number")
                    continue
                row[name][j] = float(x)
        self.buf.append((t, float(obs.get("tod_s", 0.0)), row))
        self.last_t = t
        if len(self.buf) < HISTORY:
            return None, problems
        hist = {v: np.stack([b[2][v] for b in self.buf])[None] for v in ("flow", "speed", "occupy")}
        complete = np.isfinite(hist["occupy"][0]).all(0) & np.isfinite(hist["flow"][0]).all(0)
        F = build(hist, np.array([self.buf[-1][1]]), self.N)
        pred = self.model.predict(F, iteration_range=self.iteration_range)
        occ = hist["occupy"][0]          # [13, N]
        spd = hist["speed"][0]
        out = {}
        for j, n in enumerate(self.nodes):
            if not complete[j]:
                out[n] = None             # her predictor refuses incomplete history: no forecast, no decision
                continue
            past_o = occ[-1 - TREND_HISTORY_LENGTH:-1, j]
            past_s = spd[-1 - TREND_HISTORY_LENGTH:-1, j]
            state = {"current_occupancy": float(occ[-1, j]), "predicted_occupancy": float(pred[j]),
                     "occupancy_change": float(pred[j] - occ[-1, j]), "current_speed": float(spd[-1, j]),
                     "recent_occupancy_trend": float((past_o[-1] - past_o[0]) / (TREND_HISTORY_LENGTH - 1)),
                     "recent_speed_trend": float(np.nan_to_num((past_s[-1] - past_s[0]) / (TREND_HISTORY_LENGTH - 1))),
                     "neighbors": [{"current_occupancy": float(occ[-1, k])} for k in self.nb[j]
                                   if np.isfinite(occ[-1, k])]}
            d = rule_based_decision(state, self.th, self.horizon_s)
            assert TRAFFIC_RISK_LEVELS[d["traffic_state"]] == d["risk_level"]
            out[n] = {"pred_occ": round(float(pred[j]), 6), "traffic_state": d["traffic_state"],
                      "risk_level": d["risk_level"]}
        msg = {"type": "goflow", "v": 1, "seq": obs.get("seq"), "t": t, "model": self.version,
               "horizon_s": self.horizon_s, "decision_source": "rule_based_fallback", "nodes": out}
        return msg, problems

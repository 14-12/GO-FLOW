"""Inférence XGBoost conforme aux notebooks 02/03 et à l'export sauvegardé."""
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd


# Ordre audité dans le modèle JSON, ses métadonnées et le notebook 03, cellule 5.
FEATURE_COLUMNS = (
    "location", "day_index", "slot_in_day", "hour", "minute", "flow", "speed", "occupy",
    "flow_lag_1", "flow_lag_2", "flow_lag_3", "flow_lag_6", "flow_lag_12",
    "speed_lag_1", "speed_lag_2", "speed_lag_3", "speed_lag_6", "speed_lag_12",
    "occupy_lag_1", "occupy_lag_2", "occupy_lag_3", "occupy_lag_6", "occupy_lag_12",
    "flow_rolling_mean_3", "flow_rolling_mean_6", "flow_rolling_mean_12",
    "speed_rolling_mean_3", "speed_rolling_mean_6", "speed_rolling_mean_12",
    "occupy_rolling_mean_3", "occupy_rolling_mean_6", "occupy_rolling_mean_12")
SOURCE_COLUMNS = ("location", "timestep", "flow", "speed", "occupy")
LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)


class XGBoostTrafficPredictor:
    """Charger une fois le modèle ; prédire à partir de t−12…t d'un seul capteur.

    L'historique est un DataFrame chronologique aux colonnes SOURCE_COLUMNS.
    Les colonnes supplémentaires, notamment les cibles, sont entièrement ignorées.
    Les timesteps suivent le repère PEMS08 original : origine globale 1, pas de
    cinq minutes. Aucune origine calendaire ou remise à zéro locale n'est inférée.
    """

    minimum_history = 13
    timestep_origin = 1
    feature_columns = FEATURE_COLUMNS

    def __init__(self, project_root=None, *, model_path=None, metadata_path=None):
        root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        folder = root / "results/model_benchmark/xgboost"
        self.model_path = Path(model_path) if model_path is not None else folder / "xgboost_model.json"
        self.metadata_path = (Path(metadata_path) if metadata_path is not None else
                              self.model_path.parent / "xgboost_export_metadata.json")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Modèle XGBoost sérialisé absent : {self.model_path}. Aucun entraînement autorisé.")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Métadonnées XGBoost absentes : {self.metadata_path}.")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if tuple(metadata.get("features", ())) != FEATURE_COLUMNS:
            raise ValueError("Les features ou leur ordre diffèrent du contrat XGBoost audité.")
        if metadata.get("target") != "target_occupy_t15" or metadata.get("horizon_minutes") != 15:
            raise ValueError("Cible ou horizon incompatibles avec le modèle XGBoost audité.")
        categories = metadata.get("sensor_categories")
        if categories != list(range(170)):
            raise ValueError("Catégories de capteurs incompatibles avec l'export audité (0 à 169).")
        self.sensor_categories = tuple(categories)
        self._sensor_dtype = pd.CategoricalDtype(categories=categories)
        # Importer XGBoost uniquement pour le mode d'inférence, jamais pour le replay.
        from xgboost import XGBRegressor
        self._model = XGBRegressor(enable_categorical=True, n_jobs=4, device="cpu")
        self._model.load_model(self.model_path)
        booster = self._model.get_booster()
        if tuple(booster.feature_names or ()) != FEATURE_COLUMNS:
            raise ValueError("L'ordre des features du modèle diffère des métadonnées.")
        if booster.feature_types != ["c"] + ["float"] * 31:
            raise ValueError("Types de features incompatibles avec l'export audité.")
        if self._model.best_iteration != metadata.get("best_iteration"):
            raise ValueError("Meilleure itération incohérente entre modèle et métadonnées.")
        # L'export utilise predict après early stopping : ne pas utiliser les arbres suivants.
        self._iteration_range = (0, self._model.best_iteration + 1)

    def _validate_history(self, history, sensor_id, timestep):
        if not isinstance(history, pd.DataFrame):
            raise ValueError("Historique attendu : DataFrame avec location, timestep, flow, speed et occupy.")
        if not history.columns.is_unique or not set(SOURCE_COLUMNS).issubset(history.columns):
            raise ValueError("Colonnes de l'historique absentes ou dupliquées.")
        if len(history) < self.minimum_history:
            raise ValueError(f"Historique insuffisant : {self.minimum_history} observations consécutives requises "
                             f"(t−12 à t), {len(history)} reçues.")
        frame = history.loc[:, list(SOURCE_COLUMNS)].copy()
        for name in SOURCE_COLUMNS:
            if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
                raise ValueError(f"{name} doit contenir des valeurs numériques finies, sans booléens.")
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(values).all():
            raise ValueError("L'historique contient des valeurs manquantes ou non finies.")
        for name in ("location", "timestep"):
            if not pd.api.types.is_integer_dtype(frame[name]):
                raise ValueError(f"{name} doit contenir des entiers.")
        if frame.location.nunique() != 1:
            raise ValueError("L'historique doit appartenir à un seul capteur.")
        actual_sensor = int(frame.location.iloc[-1])
        if actual_sensor not in self.sensor_categories:
            raise ValueError("Capteur inconnu du modèle XGBoost sauvegardé.")
        if sensor_id is not None and (isinstance(sensor_id, bool) or
                not isinstance(sensor_id, numbers.Integral) or sensor_id != actual_sensor):
            raise ValueError("sensor_id incompatible avec le capteur de l'historique.")
        times = frame.timestep.to_numpy()
        if times[0] < self.timestep_origin or not np.all(np.diff(times) == 1):
            raise ValueError("Historique non contigu ou non chronologique : un pas de cinq minutes est requis.")
        if timestep is not None:
            if isinstance(timestep, bool) or not isinstance(timestep, numbers.Integral):
                raise ValueError("timestep doit être un entier.")
            if times[-1] > timestep:
                raise ValueError("Historique non causal : des observations futures dépassent le timestep demandé.")
            if times[-1] != timestep:
                raise ValueError("L'historique doit inclure l'observation courante au timestep demandé.")
        # Les moyennes sont calculées en float64 avant la conversion finale de 03.
        frame[["flow", "speed", "occupy"]] = frame[["flow", "speed", "occupy"]].astype("float64")
        return frame

    def build_features(self, history, *, sensor_id=None, timestep=None):
        """Construire la ligne courante dans l'ordre exact, sans consulter de cible."""
        frame = self._validate_history(history, sensor_id, timestep)
        relative_time = frame.timestep - self.timestep_origin
        frame["day_index"] = relative_time // 288
        frame["slot_in_day"] = relative_time % 288
        minutes = frame.slot_in_day * 5
        frame["hour"] = minutes // 60
        frame["minute"] = minutes % 60
        for variable in ("flow", "speed", "occupy"):
            for lag in LAGS:
                frame[f"{variable}_lag_{lag}"] = frame[variable].shift(lag)
            for window in ROLLING_WINDOWS:
                frame[f"{variable}_rolling_mean_{window}"] = (
                    frame[variable].shift(1).rolling(window=window, min_periods=window).mean())
        features = frame.loc[:, list(FEATURE_COLUMNS)].iloc[[-1]].copy()
        features["location"] = features.location.astype(self._sensor_dtype)
        numeric = list(FEATURE_COLUMNS[1:])
        features[numeric] = features[numeric].astype("float32")
        if features.isna().any().any() or not np.isfinite(features[numeric].to_numpy()).all():
            raise ValueError("Features XGBoost manquantes ou non finies après conversion float32.")
        return features

    def predict(self, history, *, sensor_id=None, timestep=None):
        """Renvoyer uniquement predicted_occupancy_t15, sous forme de float non écrêté."""
        features = self.build_features(history, sensor_id=sensor_id, timestep=timestep)
        value = float(self._model.predict(features, iteration_range=self._iteration_range)[0])
        if not np.isfinite(value):
            raise ValueError("Le modèle XGBoost a produit une prévision non finie.")
        return value

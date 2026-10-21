"""Features originales, inférence réelle sans fit et compatibilité des deux modes."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.agents import TrafficPredictionControlAgent
from src.agents.decision_agent import create_runtime
from src.prediction import XGBoostTrafficPredictor


ROOT = Path(__file__).resolve().parents[1]


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(XGBRegressor, "fit", side_effect=AssertionError("Entraînement interdit")):
            cls.predictor = XGBoostTrafficPredictor(ROOT)
        # Ne garder en mémoire que trois capteurs pour comparer les calculs originaux.
        parts = []
        for chunk in pd.read_csv(ROOT / "data/raw/traffic_dataset.csv", chunksize=250_000):
            parts.append(chunk.loc[chunk.location.isin([0, 134, 169])])
        data = pd.concat(parts, ignore_index=True)
        cls.histories = {int(s): frame.reset_index(drop=True) for s, frame in data.groupby("location")}
        cls.saved = pd.read_csv(
            ROOT / "results/model_benchmark/xgboost/xgboost_test_predictions.csv",
            usecols=["location", "timestep", "predicted_occupancy_t15"]).set_index(["location", "timestep"])
        cls.metadata = json.loads((ROOT / "results/model_benchmark/xgboost/xgboost_export_metadata.json").read_text(encoding="utf-8"))
        notebook02 = json.loads((ROOT / "notebooks/02_feature_engineering.ipynb").read_text(encoding="utf-8"))
        notebook03 = json.loads((ROOT / "notebooks/03_model_benchmark.ipynb").read_text(encoding="utf-8"))
        # Exécuter seulement les cellules de calcul des features et la fonction de conversion.
        # Aucun import du notebook complet, appel fit, cible, split ou export n'est exécuté.
        tree = ast.parse("".join(notebook03["cells"][5]["source"]))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preparer_features")
        cls.reference_features = {}
        for sensor, history in cls.histories.items():
            namespace = {"df": history.copy(), "feature_columns": cls.metadata["features"],
                         "type_capteur": pd.CategoricalDtype(cls.metadata["sensor_categories"])}
            for index in [5, 7, 9]:
                exec("".join(notebook02["cells"][index]["source"]), namespace)
            exec(compile(ast.Module(body=[function], type_ignores=[]), "preparer_features_original", "exec"), namespace)
            cls.reference_features[sensor] = namespace["preparer_features"](namespace["df"].iloc[12:])

    def test_exact_features_order_types_and_original_calculations(self):
        self.assertEqual(list(self.predictor.feature_columns), self.metadata["features"])
        for sensor in self.histories:
            for timestep in [13, 288, 289, 15177, 16252, 17853, 17856]:
                with self.subTest(sensor=sensor, timestep=timestep):
                    history = self.histories[sensor].iloc[timestep - 13:timestep].copy()
                    before = history.copy(deep=True)
                    actual = self.predictor.build_features(history, sensor_id=sensor, timestep=timestep)
                    expected = self.reference_features[sensor].loc[[timestep - 1]]
                    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
                    pd.testing.assert_frame_equal(history, before)
                    self.assertNotIn("target_occupy_t15", actual.columns)

    def test_inference_matches_saved_predictions(self):
        # Le CSV conserve la représentation décimale courte des float32 XGBoost.
        for sensor in self.histories:
            for timestep in range(15177, 17854, 137):
                with self.subTest(sensor=sensor, timestep=timestep):
                    history = self.histories[sensor].iloc[timestep - 13:timestep]
                    actual = self.predictor.predict(history, sensor_id=sensor, timestep=timestep)
                    expected = self.saved.loc[(sensor, timestep), "predicted_occupancy_t15"]
                    self.assertEqual(np.float32(actual), np.float32(expected))
                    self.assertIs(type(actual), float)

    def test_one_load_for_multiple_predictions(self):
        original = XGBRegressor.load_model
        with patch.object(XGBRegressor, "load_model", autospec=True, side_effect=original) as load, \
                patch.object(XGBRegressor, "fit", side_effect=AssertionError("Entraînement interdit")):
            predictor = XGBoostTrafficPredictor(ROOT)
            history = self.histories[0].iloc[:13]
            first = predictor.predict(history)
            self.assertEqual(first, predictor.predict(history))
            self.assertEqual(load.call_count, 1)
        self.assertEqual(predictor._iteration_range, (0, 133))

    def test_insufficient_history_and_invalid_inputs(self):
        history = self.histories[0].iloc[:13].copy()
        for size in [0, 1, 12]:
            with self.assertRaisesRegex(ValueError, "Historique insuffisant.*13"):
                self.predictor.predict(history.iloc[:size])
        fixtures = [None, history.drop(columns="occupy"), history.iloc[::-1],
                    pd.concat([history.iloc[:12], history.iloc[[11]]]),
                    pd.concat([history.iloc[:6], history.iloc[7:14], history.iloc[[-1]]]),
                    history.assign(location=170), history.assign(flow=True),
                    history.assign(speed=float("inf")), history.assign(occupy=float("nan")),
                    history.assign(timestep=history.timestep.astype(float)),
                    history.assign(location=[0] * 12 + [1]),
                    history.assign(flow="100")]
        for fixture in fixtures:
            with self.subTest(fixture=type(fixture).__name__), self.assertRaises(ValueError):
                self.predictor.predict(fixture)
        for kwargs in [{"sensor_id": 1}, {"sensor_id": True}, {"timestep": True},
                       {"timestep": 12}, {"timestep": 14}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.predictor.predict(history, **kwargs)

    def test_no_target_or_future_leakage(self):
        history = self.histories[0].iloc[:13].copy()
        actual = self.predictor.predict(history, timestep=13)
        changed = history.assign(target_occupy_t15=-999., target_flow_t15=999.)
        self.assertEqual(actual, self.predictor.predict(changed, timestep=13))
        with self.assertRaisesRegex(ValueError, "non causal"):
            self.predictor.predict(self.histories[0].iloc[:14], timestep=13)
        features = self.predictor.build_features(history)
        changed = history.copy()
        changed.loc[changed.index[-1], ["flow", "speed", "occupy"]] = 999.
        other = self.predictor.build_features(changed)
        lag_and_rolling = [c for c in features if "_lag_" in c or "_rolling_mean_" in c]
        pd.testing.assert_frame_equal(features[lag_and_rolling], other[lag_and_rolling])

    def test_temporal_origin_is_global_and_live_extends_export_period(self):
        history = self.histories[0].iloc[:13].copy()
        history["timestep"] += 17856
        actual = self.predictor.build_features(history)
        relative = int(history.timestep.iloc[-1]) - 1
        self.assertEqual(actual.day_index.iloc[0], relative // 288)
        self.assertEqual(actual.slot_in_day.iloc[0], relative % 288)
        self.assertTrue(np.isfinite(self.predictor.predict(history)))

    def test_missing_model_and_contract_mismatch_fail_without_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Aucun entraînement"):
                XGBoostTrafficPredictor(model_path=Path(folder) / "absent.json")
            metadata = copy.deepcopy(self.metadata)
            metadata["features"][0:2] = reversed(metadata["features"][0:2])
            path = Path(folder) / "metadata.json"
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordre"), \
                    patch.object(XGBRegressor, "fit", side_effect=AssertionError("Entraînement interdit")):
                XGBoostTrafficPredictor(ROOT, metadata_path=path)


class LiveAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = XGBoostTrafficPredictor(ROOT)
        # Un fournisseur synthétique déclaré pour les tests de plomberie multi-agent.
        cls.time = 18013

    @staticmethod
    def history_provider(sensor_id, timestep):
        return pd.DataFrame({"location": [sensor_id] * 13,
                             "timestep": np.arange(timestep - 12, timestep + 1),
                             "flow": np.arange(13, dtype=float) + 100.,
                             "speed": np.linspace(60., 40., 13),
                             "occupy": np.linspace(.02, .08, 13)})

    def make_agent(self, **kwargs):
        return TrafficPredictionControlAgent(ROOT, prediction_mode="live", predictor=self.predictor,
                                             history_provider=kwargs.pop("history_provider", self.history_provider), **kwargs)

    def message(self, sensor=134):
        current = self.history_provider(sensor, self.time).iloc[-1]
        return {"request_id": "live_fixture", "sensor_id": sensor, "timestep": self.time,
                "traffic_data": {"current_flow": float(current.flow), "current_speed": float(current.speed),
                                 "current_occupancy": float(current.occupy)}, "integration_test_fixture": True}

    def test_live_uses_inference_and_preserves_live_spatial_context(self):
        with patch.object(XGBRegressor, "load_model", side_effect=AssertionError("Modèle déjà chargé")), \
                patch("pandas.read_csv", side_effect=AssertionError("CSV historique interdit")):
            agent = self.make_agent()
            self.assertIsNone(agent._init_error)
            self.assertFalse(hasattr(agent.rules, "LOOKUP"))
            self.assertFalse(hasattr(agent.rules, "OBSERVATIONS"))
            self.assertEqual(agent._cached, {})
            state = agent.build_state(self.message())
            expected = self.predictor.predict(self.history_provider(134, self.time))
            self.assertEqual(state["predicted_occupancy_t15"], expected)
            self.assertEqual({n["sensor_id"] for n in state["neighbors"]},
                             set(np.flatnonzero(agent.rules.ADJACENCY[134] > 0)) - {134})
            self.assertTrue(all(n["timestep"] == self.time for n in state["neighbors"]))
            calls = []
            original = self.predictor.predict
            def counted(history, **kwargs):
                calls.append(kwargs["sensor_id"])
                return original(history, **kwargs)
            with patch.object(self.predictor, "predict", side_effect=counted):
                response = agent.handle_message(self.message())
            self.assertEqual(len(calls), len(state["neighbors"]) + 1)
            self.assertEqual(response["forecast"]["predicted_occupancy"], expected)
            self.assertEqual(response["metadata"]["prediction_mode"], "live")
            self.assertEqual(response["decision_source"], "rule_based_fallback")
            self.assertTrue(agent.validate_response(response))

    def test_provider_errors_never_fall_back_to_historical_forecast(self):
        for provider in [lambda s, t: self.history_provider(s, t).iloc[:12],
                         lambda s, t: self.history_provider(s, t + 1)]:
            agent = self.make_agent(history_provider=provider)
            response = agent.handle_message(self.message())
            self.assertEqual(response["status"], "error")
            self.assertIsNone(response["forecast"]["predicted_occupancy"])
        def missing_neighbor(sensor, timestep):
            if sensor != 134:
                raise ValueError("Historique voisin indisponible")
            return self.history_provider(sensor, timestep)
        response = self.make_agent(history_provider=missing_neighbor).handle_message(self.message())
        self.assertEqual(response["status"], "error")
        self.assertIn("voisin", response["reason"])

    def test_live_rejects_current_or_neighbor_measure_inconsistency(self):
        agent = self.make_agent()
        message = self.message()
        message["traffic_data"]["current_occupancy"] += 1
        self.assertEqual(agent.handle_message(message)["status"], "error")
        state = agent.build_state(self.message())
        message = self.message()
        neighbor = copy.deepcopy(state["neighbors"][0])
        neighbor["predicted_occupancy_t15"] += 1
        message["cooperative_context"] = {"neighboring_sensors": [neighbor]}
        self.assertEqual(agent.handle_message(message)["status"], "error")

    def test_explicit_mode_and_live_schema(self):
        with self.assertRaisesRegex(ValueError, "prediction_mode"):
            TrafficPredictionControlAgent(ROOT, prediction_mode="unknown")
        with self.assertRaisesRegex(ValueError, "history_provider"):
            TrafficPredictionControlAgent(ROOT, prediction_mode="live")
        agent = self.make_agent()
        self.assertNotIn("maximum", agent.input_schema["properties"]["timestep"])
        self.assertEqual(agent.input_schema["properties"]["timestep"]["minimum"], 13)
        response = agent.handle_message({**self.message(), "timestep": 12})
        self.assertEqual(response["status"], "error")

    def test_live_thresholds_equal_saved_values_without_readjustment(self):
        rules = create_runtime()
        actual = rules.initialize_live(ROOT)
        record = json.loads((ROOT / "results/decision_agent/decision_agent_methodology.md").read_text(
            encoding="utf-8").split("```json")[1].split("```")[0])
        self.assertEqual(actual, record["thresholds_train_only"])
        actual["speed_low"] = -1
        self.assertEqual(rules.THRESHOLDS, record["thresholds_train_only"])


if __name__ == "__main__":
    unittest.main()

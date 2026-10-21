"""Exporter uniquement les objets XGBoost encore présents dans le noyau du notebook 03.

Depuis ce noyau : %run -i ../scripts/export_existing_xgboost.py
Aucun entraînement et aucune nouvelle inférence ne sont effectués.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Refuser de reconstituer un modèle ou des prédictions lorsque le noyau a été perdu.
required = ["best_model", "test", "y_pred", "feature_columns", "type_capteur"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError("Objets absents du noyau 03 : " + ", ".join(missing)
                       + ". Ne pas relancer l'entraînement pour cet export.")

root = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "notebooks/03_model_benchmark.ipynb").exists()), None)
if root is None:
    raise RuntimeError("Racine GoFlow introuvable.")
folder = root / "results/model_benchmark"
outputs = [folder / name for name in ["xgboost_test_predictions.csv", "xgboost_best.json",
                                     "xgboost_export_metadata.json"]]
if any(p.exists() for p in outputs):
    raise FileExistsError("Export déjà présent : aucun fichier ne sera écrasé.")

# Vérifier la cible, les clés et les scores contre le benchmark sauvegardé, sans le modifier.
prediction = np.asarray(y_pred).reshape(-1)
assert len(test) == len(prediction) == 455090
assert np.isfinite(prediction).all()
keys = test[["location", "timestep"]].reset_index(drop=True)
expected = pd.MultiIndex.from_product([range(170), range(15177, 17854)],
                                     names=["location", "timestep"]).to_frame(index=False)
pd.testing.assert_frame_equal(keys.astype("int64"), expected.astype("int64"))
target = test["target_occupy_t15"].to_numpy()
assert np.isfinite(target).all()
benchmark = pd.read_csv(folder / "benchmark_results.csv")
rows = benchmark.loc[benchmark["model"].eq("XGBoost Regressor")
                     & benchmark["target"].eq("target_occupy_t15")
                     & benchmark["horizon_minutes"].eq(15)]
assert len(rows) == 1
scores = {"MAE": mean_absolute_error(target, prediction),
          "RMSE": float(np.sqrt(mean_squared_error(target, prediction))),
          "R2": r2_score(target, prediction)}
for metric, value in scores.items():
    assert np.isclose(value, float(rows.iloc[0][metric]), rtol=1e-7, atol=1e-10), metric
assert best_model.__class__.__name__ == "XGBRegressor"
assert list(best_model.get_booster().feature_names) == list(feature_columns)

# Conserver les clés explicites ; les cibles servent à l'audit, jamais à la décision.
export = keys.copy()
export["target_occupy_t15"] = target
export["predicted_occupancy_t15"] = prediction
metadata = {"target": "target_occupy_t15", "horizon_minutes": 15,
            "feature_columns": list(feature_columns),
            "sensor_categories": [int(x) for x in type_capteur.categories],
            "test_input_timestep_min": 15177, "test_input_timestep_max": 17853,
            "n_test": len(test), "best_iteration": int(best_model.best_iteration),
            "source": "Objets en mémoire du noyau du notebook 03, sans réentraînement"}
export.to_csv(outputs[0], index=False)
best_model.save_model(outputs[1])
outputs[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("Export vérifié, sans modification des scores :")
for path in outputs:
    print(path)

"""Reconstruire uniquement le XGBoost sélectionné du notebook 03 et exporter après contrôle."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import redirect_stdout
import ast
import gc
import hashlib
import io
import json
import os
import platform
import tempfile

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rebuild():
    root = Path(__file__).resolve().parents[1]
    folder = root / "results/model_benchmark"
    destination = folder / "xgboost"
    if destination.exists():
        raise FileExistsError(f"Export déjà présent : {destination}. Aucun écrasement autorisé.")
    protected = [p for base in [root / "notebooks", root / "data", folder]
                 for p in base.rglob("*") if p.is_file()]
    fingerprints = {p: sha256(p) for p in protected}
    historical = pd.read_csv(folder / "benchmark_results.csv")
    selected = historical.loc[historical.model.eq("XGBoost Regressor")
                              & historical.target.eq("target_occupy_t15")
                              & historical.horizon_minutes.eq(15)]
    if len(selected) != 1:
        raise RuntimeError("Ligne historique XGBoost unique introuvable.")
    reference = selected.iloc[0]
    configuration = json.loads(reference.hyperparameters)
    notebooks = {name: json.loads((root / "notebooks" / name).read_text(encoding="utf-8"))
                 for name in ["02_feature_engineering.ipynb", "03_model_benchmark.ipynb"]}
    def source(name, index):
        return "".join(notebooks[name]["cells"][index]["source"])
    n02, n03 = "02_feature_engineering.ipynb", "03_model_benchmark.ipynb"
    namespace = {"pd": pd, "np": np, "Path": Path}
    raw = pd.read_csv(root / "data/raw/traffic_dataset.csv")
    assert raw.shape == (3035520, 5)
    assert sorted(raw.location.unique().tolist()) == list(range(170))
    assert not raw.duplicated(["location", "timestep"]).any()
    print("Préparation : cellules originales 02, capteur par capteur pour limiter la RAM.", flush=True)
    prepared = []
    for sensor, frame in raw.groupby("location", sort=True):
        namespace["df"] = frame.copy()
        # Exécuter les calculs originaux sans figures ni affichages volumineux.
        with redirect_stdout(io.StringIO()):
            for index in [3, 5, 7, 9, 11, 13]:
                exec(compile(source(n02, index), f"{n02}:cellule_{index}", "exec"), namespace)
        data = namespace["df_features"]
        # Convertir après les calculs float64, comme preparer_features du notebook 03.
        numeric = ["day_index", "slot_in_day", "hour", "minute", "flow", "speed", "occupy"]
        numeric += namespace["colonnes_lags"] + namespace["colonnes_rolling"]
        data[numeric] = data[numeric].astype("float32")
        prepared.append(data)
    del raw
    namespace["df_features"] = pd.concat(prepared, ignore_index=True)
    del prepared
    # La continuité et l'origine communes sont vérifiées ; le split original reste global.
    assert namespace["df_features"].groupby("location").timestep.min().eq(13).all()
    assert namespace["df_features"].groupby("location").timestep.max().eq(17853).all()
    for index in [17, 18]:
        exec(compile(source(n02, index), f"{n02}:cellule_{index}", "exec"), namespace)
    for name, expected in [("train", 2122450), ("validation", 454410), ("test", 455090)]:
        assert len(namespace[name]) == expected
    namespace.pop("df_features")
    namespace.pop("df", None)
    gc.collect()
    namespace.update(RANDOM_STATE=42, N_JOBS=min(4, os.cpu_count() or 1))
    exec(compile(source(n03, 5), f"{n03}:cellule_5", "exec"), namespace)
    namespace["X_test"] = namespace["preparer_features"](namespace["test"])
    namespace["y_test"] = namespace["test"][namespace["target_column"]]
    # Lire uniquement les affectations de configuration : aucun essai du notebook n'est lancé.
    tree = ast.parse(source(n03, 7))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in
                           {"configurations", "parametres_communs"} for t in node.targets)]
    assert len(assignments) == 2
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "configuration_03", "exec"), namespace)
    assert configuration in namespace["configurations"]
    assert namespace["target_column"] == "target_occupy_t15"
    assert len(namespace["feature_columns"]) == int(reference.n_features) == 32
    best_model = XGBRegressor(**namespace["parametres_communs"], **configuration)
    print("Entraînement du seul XGBoost sélectionné :", configuration, flush=True)
    best_model.fit(namespace["X_train"], namespace["y_train"],
                   eval_set=[(namespace["X_validation"], namespace["y_validation"])], verbose=False)
    namespace["best_model"] = best_model
    X_test, y_test, test = (namespace[name] for name in ["X_test", "y_test", "test"])
    y_pred = best_model.predict(X_test)
    namespace["y_pred"] = y_pred
    assert len(X_test) == len(y_test) == len(y_pred) == 455090
    assert X_test.index.equals(y_test.index) and test.index.equals(X_test.index)
    assert np.isfinite(y_pred).all()
    expected_keys = pd.MultiIndex.from_product([range(170), range(15177, 17854)],
                                             names=["location", "timestep"]).to_frame(index=False)
    pd.testing.assert_frame_equal(test[["location", "timestep"]].reset_index(drop=True), expected_keys)
    scores = {"mae": float(mean_absolute_error(y_test, y_pred)),
              "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
              "r2": float(r2_score(y_test, y_pred))}
    # Tolérances fixées avant entraînement : petites différences, pas de nouveau classement.
    tolerances = {"mae": 1e-5, "rmse": 1e-5, "r2": 2e-4}
    for metric, column in [("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2")]:
        if abs(scores[metric] - float(reference[column])) > tolerances[metric]:
            raise RuntimeError(f"Export arrêté : {metric}={scores[metric]:.10f}, historique="
                               f"{reference[column]:.10f}, tolérance={tolerances[metric]}.")
    export = test[["timestep", "location", "flow", "speed", "occupy", "target_occupy_t15",
                   "day_index", "slot_in_day", "hour", "minute"]].copy()
    export = export.rename(columns={"flow": "current_flow", "speed": "current_speed",
                                    "occupy": "current_occupancy"})
    # Exporter les mesures courantes originales float64, sans l'arrondi interne des features.
    current = pd.read_csv(root / "data/raw/traffic_dataset.csv")
    current = current.set_index(["location", "timestep"])
    aligned = current.loc[pd.MultiIndex.from_frame(export[["location", "timestep"]])]
    for old, new in [("flow", "current_flow"), ("speed", "current_speed"), ("occupy", "current_occupancy")]:
        export[new] = aligned[old].to_numpy()
    export["predicted_occupancy_t15"] = y_pred
    metadata = {"model": "XGBoost", "target": namespace["target_column"], "horizon_minutes": 15,
                "n_test": len(y_pred), "features": namespace["feature_columns"], **scores,
                "source_notebook": n03, "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "xgboost_version": xgboost.__version__, "python_version": platform.python_version(),
                "hyperparameters": best_model.get_params(), "best_iteration": best_model.best_iteration,
                "sensor_categories": [int(x) for x in namespace["type_capteur"].categories],
                "metric_absolute_tolerances": tolerances,
                "preparation_cells": [3, 5, 7, 9, 11, 13, 17, 18],
                "source_hashes": {str(p.relative_to(root)): fingerprints[p] for p in
                                  [root / "notebooks" / n02, root / "notebooks" / n03,
                                   root / "data/raw/traffic_dataset.csv"]}}
    assert all(sha256(p) == value for p, value in fingerprints.items()), "Fichier protégé modifié"
    # Écrire dans un dossier temporaire puis publier l'export complet, sans écraser de fichier.
    with tempfile.TemporaryDirectory(prefix=".xgboost_export_", dir=folder) as staging:
        stage = Path(staging) / "xgboost"
        stage.mkdir()
        best_model.save_model(stage / "xgboost_model.json")
        export.to_csv(stage / "xgboost_test_predictions.csv", index=False)
        (stage / "xgboost_export_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(destination)
    print("Shape X_test :", X_test.shape)
    for name, value in scores.items():
        print(f"{name.upper()} : {value:.10f}")
    print("XGBoost export validé.\nModèle :\nresults/model_benchmark/xgboost/xgboost_model.json\n")
    print("Prédictions :\nresults/model_benchmark/xgboost/xgboost_test_predictions.csv\n")
    print("Nombre de prédictions : 455090\nLe notebook 08 peut maintenant être relancé.")
    return namespace


if __name__ == "__main__":
    globals().update(rebuild())

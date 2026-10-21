# Audit de reconstruction XGBoost

Les indices ci-dessous sont les indices des cellules du fichier JSON, à partir de zéro.
Le notebook `03_model_benchmark.ipynb` n'est pas modifié ni exécuté intégralement.

| Élément | Cellule du notebook 03 |
|---|---|
| Imports, graine 42, maximum 4 threads | 1 |
| Chargement via notebook 02, train / validation / test | 3 |
| Features, cible, catégorie location, X_train / X_validation et y_train / y_validation | 5 |
| Configurations, paramètres communs, sélection et variable best_model | 7 |
| X_test, y_test, y_pred et métriques test | 9 |

La préparation réutilise littéralement les cellules de calcul 3, 5, 7, 9, 11 et 13
du notebook 02, indépendamment pour chaque capteur afin de réduire la RAM.
Les 170 séries ont la même origine temporelle. Les features numériques sont
converties en float32 après les calculs originaux en float64, comme dans la
fonction `preparer_features` de 03. Les cibles restent en float64.
Les cellules globales 17 et 18 de 02 réalisent le split original et sa purge.

- Train : timesteps 13–12497, 2 122 450 lignes.
- Validation : 12501–15173, 454 410 lignes.
- Test : 15177–17853, 455 090 lignes.
- Cible : `target_occupy_t15`, shift(-3) par capteur, horizon 15 minutes.
- Ordre des lignes : location puis timestep.

Les 32 features sont, dans cet ordre :

```text
location, day_index, slot_in_day, hour, minute, flow, speed, occupy,
flow_lag_1, flow_lag_2, flow_lag_3, flow_lag_6, flow_lag_12,
speed_lag_1, speed_lag_2, speed_lag_3, speed_lag_6, speed_lag_12,
occupy_lag_1, occupy_lag_2, occupy_lag_3, occupy_lag_6, occupy_lag_12,
flow_rolling_mean_3, flow_rolling_mean_6, flow_rolling_mean_12,
speed_rolling_mean_3, speed_rolling_mean_6, speed_rolling_mean_12,
occupy_rolling_mean_3, occupy_rolling_mean_6, occupy_rolling_mean_12
```

La configuration gagnante est lue dans le benchmark historique et vérifiée parmi
les configurations du notebook : learning_rate=0.1, max_depth=6,
min_child_weight=10, n_estimators=300. Les paramètres communs sont extraits des
affectations originales de la cellule 7 : objective=reg:squarederror,
eval_metric=rmse, tree_method=hist, device=cpu, enable_categorical=True,
max_bin=128, subsample=0.8, colsample_bytree=0.9, reg_lambda=1,
early_stopping_rounds=25, random_state=42, n_jobs=min(4, nombre de CPU).
Un seul modèle est entraîné ; aucune recherche de configuration n'est rejouée.
La validation sélectionne la meilleure itération, jamais le test.

Les tolérances absolues sont fixées avant entraînement : MAE 0.00001,
RMSE 0.00001 et R² 0.0002. Un dépassement arrête l'export. Les scores recalculés
sont conservés dans les métadonnées de reconstruction, sans remplacer les scores
historiques. Les mesures courantes exportées sont relues dans le CSV source en
float64 et réalignées par les clés, pour conserver les valeurs originales.

Les fichiers existants de notebooks/, data/ et results/model_benchmark/ sont
contrôlés par SHA256 avant publication. L'export utilise un dossier temporaire
puis un renommage vers xgboost/ et refuse d'écraser un export déjà présent.

Exécution depuis la racine :

```powershell
python -X utf8 scripts/rebuild_and_export_xgboost.py
```

Depuis notebooks/ :

```python
%run ../scripts/rebuild_and_export_xgboost.py
```

Ce script produit directement les trois fichiers demandés : il n'est pas nécessaire
de relancer l'ancien export, dont les noms de sortie sont différents. Avec `%run`,
les objets reconstruits sont également disponibles dans le noyau appelant.
Le notebook 08 de diagnostic pourra détecter les nouveaux artefacts ; ses étapes
décisionnelles, actuellement arrêtées au prérequis manquant, restent à implémenter.

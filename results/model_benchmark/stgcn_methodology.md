# Benchmark STGCN — méthode et résultats

Sources : https://github.com/Davidham3/ASTGCN-2019-mxnet/tree/d5d40645fa45471f2b473fc49a29ec3f0a993c6d/data/PEMS08

Hypothèse : location=index_NPZ ; identité non prouvée pour les 25 capteurs aux séries identiques

Graphe : noyau gaussien sur distances uniques, symétrisation max(W,W.T), self-loops A+I, normalisation symétrique ; aucune relation basée sur les IDs consécutifs.

Split : notebook 02 réutilisé, mêmes observations test que XGBoost/LSTM/PatchTST, fenêtres de 12 timesteps, horizon +3 (+15 min). Scalers ajustés sur train uniquement.

Comparaison : XGBoost tabulaire, LSTM/PatchTST par capteur, STGCN conjoint/spatial ; périmètres d’inférence différents.

**STGCN : MAE=0.007355, RMSE=0.017009, R²=0.8831**, rang RMSE 4/4 sur 455,090 observations test.

Meilleur selon RMSE : **XGBoost Regressor** (0.016440). Meilleur selon R² : **XGBoost Regressor** (0.8908). Entraînement le plus court : **XGBoost Regressor** (30.326 s).

STGCN a demandé 1425.173 s d’entraînement et 2.672 s d’inférence, avec sélection de l’époque 30 sur validation. Le R² macro est calculé sur 170 capteurs à cible test non constante.

Ces conclusions concernent cette configuration et ce test. Les entrées diffèrent entre architectures ; aucune ablation du graphe n’a été menée pour isoler son effet. Le mapping location=index NPZ reste une hypothèse documentée pour 25 capteurs indiscernables, et le support spatial a été symétrisé pour ce GCN.

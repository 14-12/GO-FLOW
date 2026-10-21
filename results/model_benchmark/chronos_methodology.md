# Chronos Zero-Shot — méthode et résultats

Modèle : amazon/chronos-bolt-tiny, révision a0e552de83495b5c28c14c71c374f3e33280b340, 8652672 paramètres, cpu.

Source : https://huggingface.co/amazon/chronos-bolt-tiny ; package chronos-forecasting 2.3.2 installé uniquement dans .venv-chronos.

Split : cellules temporelles du notebook 02 réutilisées ; clés test identiques aux benchmarks. Aucun entraînement/fine-tuning, aucun fit de scaler global.

Contexte : 48 mesures exclusivement avant t ; 4 pas depuis t−1 pour atteindre t+3. Médiane q=0.5, normalisation intégrée sur historique causal et retour dans l’échelle originale.

**Chronos Zero-Shot : MAE=0.009108, RMSE=0.020526, R²=0.8297**, rang RMSE 5/5 sur 455,090 observations test.

Meilleur selon RMSE : **XGBoost Regressor** (0.016440). Meilleur selon R² : **XGBoost Regressor** (0.8908).

L’inférence Chronos a pris 166.993 s, soit 366.945 µs par ligne en lots. Le temps d’entraînement local est 0 s ; téléchargement, chargement et préentraînement externe sont exclus.

Chronos est univarié, zero-shot local, avec contexte t−48..t−1. Les modèles précédents exploitent d’autres informations, dont la mesure courante t. Aucun contexte ni quantile n’a été choisi sur test ; aucun avantage d’un foundation model n’est présupposé. L’absence de PEMS08 dans son corpus de préentraînement n’est pas certifiée par cet audit.

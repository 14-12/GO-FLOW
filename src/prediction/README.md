# Inférence XGBoost GoFlow

Le prédicteur charge une fois par instance le modèle existant
`results/model_benchmark/xgboost/xgboost_model.json` et vérifie son contrat contre
`xgboost_export_metadata.json`. Il ne contient aucun entraînement, téléchargement,
export, ajustement de seuil ou calcul de métrique benchmark.

```python
from src.prediction import XGBoostTrafficPredictor

predictor = XGBoostTrafficPredictor(project_root=root)
# history : DataFrame chronologique, un seul capteur, de t−12 à t au minimum.
# Colonnes : location, timestep, flow, speed, occupy ; unités source inchangées.
predicted_occupancy_t15 = predictor.predict(history, sensor_id=134, timestep=16252)
```

La sortie est un `float`, sans écrêtage. `build_features(history, ...)` expose la
ligne de features pour l'audit ; `predict` renvoie uniquement la prévision.
La même instance peut être injectée dans plusieurs agents pour partager le modèle
déjà chargé. XGBoost 3.2.0 est la version vérifiée avec l'export existant.

## Audit du contrat existant

Sources lues : notebooks 02 (cellules 5/7/9), 03 (cellule 5 et prédiction après
early stopping), script `rebuild_and_export_xgboost.py`, audit d'export, métadonnées
et noms/types contenus dans le modèle JSON. Aucun de ces fichiers n'est modifié.

Ordre exact des 32 features :

```text
location, day_index, slot_in_day, hour, minute, flow, speed, occupy,
flow_lag_1, flow_lag_2, flow_lag_3, flow_lag_6, flow_lag_12,
speed_lag_1, speed_lag_2, speed_lag_3, speed_lag_6, speed_lag_12,
occupy_lag_1, occupy_lag_2, occupy_lag_3, occupy_lag_6, occupy_lag_12,
flow_rolling_mean_3, flow_rolling_mean_6, flow_rolling_mean_12,
speed_rolling_mean_3, speed_rolling_mean_6, speed_rolling_mean_12,
occupy_rolling_mean_3, occupy_rolling_mean_6, occupy_rolling_mean_12
```

- Historique minimal : **13 observations consécutives**, t−12…t, soit 60 minutes
  entre les extrémités. Aucune observation à t+1…t+3 n'est nécessaire.
- Lags : 1, 2, 3, 6, 12 observations passées du même capteur.
- Moyennes glissantes : `shift(1).rolling(window, min_periods=window).mean()` ;
  fenêtres 3, 6, 12, excluant t. Calculs float64, conversion finale float32.
- Temps : origine **globale 1**, vérifiée dans le CSV source, 288 intervalles de
  cinq minutes par jour. Ne pas recalculer cette origine sur la fenêtre reçue.
  `timestep` n'est pas une feature mais permet de calculer les quatre features
  temporelles exactement comme à l'entraînement.
- `location` : catégorie, catégories ordonnées 0…169 conservées même pour une
  fenêtre d'un seul capteur. Les capteurs inconnus sont refusés.
- Arrêt anticipé : `best_iteration=132`, donc utilisation des 133 premiers arbres.
  Les arbres supplémentaires du modèle sauvegardé ne sont pas utilisés.
- `target_occupy_t15` et toutes les colonnes autres que les cinq entrées source
  sont ignorées. Les features précalculées fournies par un appelant sont ignorées
  et reconstruites à partir des observations.

Les données non finies, timesteps manquants/dupliqués/désordonnés, plusieurs
capteurs, historique insuffisant ou observations après le timestep demandé
déclenchent une erreur explicite. Aucun historique n'est complété artificiellement.
Un timestep postérieur à l'export historique est accepté dans le même repère global.

L'empreinte du notebook 02 actuel diffère de celle inscrite à l'export : SHA256
actuel `fccd8256f05610eb14d42c44f44e98ece804166a87156e02e618ec8791ee6191`,
export `21ed2737fe3bca43c900590a0176a23ed1e077bb0a52e4284af931e1d8d3e2d4`.
Cette divergence préexistante n'est pas attribuée à une cause non vérifiée.
Le notebook 03 et le CSV brut correspondent à leurs empreintes d'export.
Les tests comparent les 32 features aux cellules actuelles, puis les inférences
aux prédictions exportées sur plusieurs capteurs et instants.

## Modes de l'agent

`TrafficPredictionControlAgent(..., prediction_mode="historical_replay")` reste
le défaut de compatibilité : anciennes entrées, réponses, prévisions et tests
inchangés. Ce mode ne charge pas le modèle XGBoost.

```python
from src.agents import TrafficPredictionControlAgent

# Fournisseur applicatif : retourner les observations réelles du capteur de t−12 à t.
# Il doit également servir les voisins directs au même timestep, sans lignes futures.
def history_provider(sensor_id, timestep):
    return observation_store.get_history(sensor_id, timestep - 12, timestep)

agent = TrafficPredictionControlAgent(
    root,
    prediction_mode="live",
    predictor=predictor,
    history_provider=history_provider,
)
response = agent.handle_message(message)
```

`message` garde le contrat usuel : request_id, sensor_id, timestep et traffic_data
(current_flow, current_speed, current_occupancy), avec les contextes optionnels.
Les mesures courantes doivent correspondre à la dernière observation fournie.
`agent.input_schema` décrit le mode actif ; `INPUT_SCHEMA` reste le contrat
historique pour la compatibilité. En live, le timestep est au minimum 13 et n'est
plus limité à la période test 15177–17853.

En live, les seuils train sont relus dans la méthodologie déjà sauvegardée et le
graphe dans `stgcn_adjacency.npy`. Aucun CSV de prédictions ni cible future n'est lu.
L'état courant, ses tendances strictement passées et les prévisions des voisins
sont construits à partir des historiques fournis. Chaque prévision est calculée
une seule fois par message et capteur ; la réponse reprend celle de l'état validé.
Un historique voisin indisponible produit une erreur contrôlée, sans inventer de
voisin ni réutiliser silencieusement sa prévision historique.

La trace indique `metadata.prediction_mode="live"` pour une réponse exploitable.
Les anciennes réponses LLM de 09 ne sont pas rejouées dans ce mode : un backend
LLM explicitement injecté peut être utilisé, sinon les règles de décision restent
le fallback. Un échec d'inférence XGBoost produit `status="error"` et une prévision
null ; il ne déclenche jamais un remplacement silencieux par le CSV historique.

Les valeurs restent dans les unités et le repère PEMS08 existants ; cette couche
ne définit pas de conversion pour un nouveau réseau de capteurs ou une autre horloge.

```powershell
python -B -X utf8 -m unittest tests.test_xgboost_predictor -v
python -B -X utf8 -m unittest discover -s tests -v
```

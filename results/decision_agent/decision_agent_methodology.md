# GoFlow — Agent Traffic Prediction & Adaptive Signal Control

Le forecasting répond : « Que va-t-il se passer dans 15 minutes ? »
L'agent répond : « Que recommander en fonction de cette prévision et du contexte ? »

L'agent réutilise exclusivement le CSV XGBoost sauvegardé : 455 090 observations
test, cible target_occupy_t15, horizon +3 timesteps de 5 minutes. Aucun modèle n'est
réentraîné et aucune nouvelle prévision n'est calculée. Le contrôle initial compare
les clés, les mesures courantes et les cibles au CSV brut. Les cibles futures ne
sont jamais incluses dans les états ni utilisées pour sélectionner les scénarios.

## Temporalité et seuils expérimentaux

Les tendances sont (valeur(t−1) − valeur(t−6))/5, dans les unités source par timestep.
Les six mesures appartiennent au même capteur. Elles excluent t et tout futur.
Les seuils sont calculés seulement sur train, timesteps 13–12497 : occupancy aux
quantiles 50 %, 90 %, 97 %, speed au quantile 25 % et médiane, variation absolue
d'occupancy sur trois pas au quantile 90 %. La cible la plus tardive de ce calcul
est 12500, avant la validation commençant à 12501. Ces seuils sont des choix
opérationnels expérimentaux, pas des classes de congestion scientifiquement validées.
Aucun seuil n'est ajusté sur validation ou test. Le test sert ici à des exemples
décisionnels illustratifs, pas à une estimation de performance de contrôle.

Le niveau utilise le maximum de l'occupation courante et prévue. Une occupation
au quantile 97 % entraîne severe/high ; au quantile 90 % ou avec hausse forte,
congested/high. Au-dessus de la médiane, à faible vitesse, avec un voisin à occupation
élevée ou avec hausse passée
et baisse passée de vitesse conjointes, moderate/medium ; sinon free_flow/low.
Pour high, l'action est coordinate_with_neighbor si au moins un voisin est au
quantile élevé courant, sinon manual_review. Medium entraîne monitor ; low,
keep_current_signal_plan. Une amélioration prévue peut donc conserver un risque
élevé si l'occupation courante est encore élevée. La confidence constante 0.5
est un indicateur neutre non calibré, pas une probabilité statistique de réussite.

## Graphe et mapping

L'adjacence STGCN sauvegardée est une matrice symétrisée 170×170, construite à partir
du fichier de distances ASTGCN PEMS08, avec 274 liens non dirigés. Sa connectivité
est comparée aux arêtes du fichier source. On conserve toutes les arêtes réellement
présentes, même avec un poids gaussien très petit, sans seuil arbitraire. Les voisins
sont directs et leur état ainsi que leur prévision sont lus au même timestep.
L'adjacence porte sur le réseau de capteurs, pas sur des feux de carrefours.

Le mapping location=index NPZ reste une hypothèse pour 25 capteurs aux séries
entièrement identiques : [48,61,80,121,122,123,141,142],
[50,51,74,75,90,91,115,131], [55,70,73,81,83,84,159], [62,109].
Les valeurs seules ne prouvent pas leur identité physique individuelle. Les conseils
de coordination restent donc conditionnels à ce mapping documenté.

## LLM et garde-fous

L'environnement est vérifié avant toute installation : Python, RAM, GPU, packages,
Ollama local, transformers et llama.cpp. Aucun téléchargement ni installation.
Chronos n'est pas un modèle instruct décisionnel. Un petit modèle Ollama déjà local,
identifié comme instruct, peut être utilisé si la mémoire mesurée suffit ; sinon
llm_decision retourne None et le fallback fonctionne. L'interface est remplaçable.
La présence de transformers seule ne prouve pas la présence d'un modèle instruct.

La première version borne volontairement le LLM : les règles imposent état, risque
et confidence ; le LLM peut choisir une action parmi les actions compatibles et une
justification parmi deux formulations exactes dérivées des données. Une justification
libre est refusée, même avec un JSON valide. Ce contrôle permet de vérifier l'ancrage
aux valeurs ; il ne valide pas l'efficacité réelle de la politique de trafic.
Sans information de feux, extend_green_phase et reduce_green_phase sont dans le
schéma général mais interdites par le contrat applicable. Aucune durée de feu
n'est proposée et aucune commande n'est envoyée à un système externe.

incident_detected et emergency_vehicle_present sont null, jamais False inventé.
Chaque justification signale aussi le plan de feux indisponible. À terme, Incident
Detection Agent et Emergency Agent fourniront ces champs ; Cooperative Traffic
Agent fournira le contexte spatial, Traffic Prediction Agent les prévisions, et
l'Orchestrator combinera ces informations. Ce notebook implémente seulement le
rôle Traffic Prediction & Adaptive Signal Control.

## Scénarios et validation

Cinq cas distincts sont sélectionnés par critères déterministes sur le test :
occupation minimale parmi les états fluides, occupation maximale parmi les états
élevés, hausse prévue maximale, baisse prévue maximale et contraste maximal entre
voisins. Il s'agit de cas illustratifs extrêmes, pas d'un échantillon représentatif.
Les états, prévisions et voisins proviennent des données ; aucune valeur n'est
écrite manuellement pour construire un scénario.

La grille vérifie schéma JSON strict, action autorisée, entrée canonique, justification
préautorisée liée aux valeurs, informations manquantes, cohérence risque/action.
Les tests exercent des erreurs de contrat et un faux appel LLM explicitement simulé,
sans les enregistrer comme décisions réelles. Aucun score accuracy LLM ou efficacité
de contrôle n'est revendiqué. Les sources réelles des décisions sont sauvegardées.
Les notebooks 01–07, les données sources et les métriques historiques sont préservés.


## Configuration et exécution réelles

```json
{
  "config": {
    "trend_history": 6,
    "train_first_timestep": 13,
    "train_last_timestep": 12497,
    "occupancy_quantiles": [
      0.5,
      0.9,
      0.97
    ],
    "low_speed_quantile": 0.25,
    "change_quantile": 0.9,
    "llm_timeout_s": 30,
    "llm_model": null,
    "max_local_model_bytes": 2000000000,
    "minimum_available_ram_bytes": 2500000000
  },
  "thresholds_train_only": {
    "occupancy_median": 0.0599,
    "occupancy_high": 0.1071,
    "occupancy_severe": 0.1604,
    "speed_low": 62.6,
    "speed_median": 64.9,
    "strong_change": 0.015000000000000006
  },
  "environment": {
    "python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]",
    "packages": {
      "torch": true,
      "transformers": true,
      "llama_cpp": false,
      "ollama": false,
      "psutil": true
    },
    "ollama_executable": null,
    "gguf_files": [],
    "torch_version": "2.11.0+cpu",
    "cuda_available": false,
    "ram_total_bytes": 16487878656,
    "ram_available_bytes": 3629977600,
    "ollama_models": [],
    "ollama_status": "URLError",
    "selected_llm": null,
    "llm_mode": "placeholder_rule_based"
  },
  "input_hashes": {
    "data\\raw\\traffic_dataset.csv": "7bcc4a2161229a4b5187bd727c1e63f989aa193301d6773137bd4a59e1680293",
    "results\\model_benchmark\\xgboost\\xgboost_test_predictions.csv": "8b55822700ddc6e4d2148d058753acbd984300b179c7e9e1e09d31a365e6cd22",
    "results\\model_benchmark\\stgcn_adjacency.npy": "d17dc8ba48d923013d7d74dc9f11dc8425964ca4e906753a80a552ee85995676"
  },
  "decision_sources": {
    "rule_based": 5
  }
}
```

# Traffic Prediction & Adaptive Signal Control Agent

Point d'entrée : `TrafficPredictionControlAgent.handle_message(message)`.
Les contrats sérialisables `INPUT_SCHEMA` et `OUTPUT_SCHEMA`, ainsi que la politique
expérimentale `PRIORITY_POLICY`, sont exportés par `traffic_prediction_agent.py`.

Les schémas et enums ont leur source unique sous `src/contracts/` ; l'agent
réexporte les mêmes objets pour compatibilité. Les exemples JSON et validateurs
indépendants sont décrits dans le [contrat d'intégration officiel](../../docs/traffic_prediction_agent_contract.md).

La décision est séparée dans `src/decision/` : moteur et fallback dans
`decision_engine.py`, validations et protections critiques dans `guardrails.py`,
seuils et politiques dans `policy.py`. Les points d'entrée historiques restent des
adaptateurs compatibles. Voir [la couche de décision](../decision/README.md).

Le paramètre `prediction_mode` distingue désormais `historical_replay` (défaut
temporaire de compatibilité) et `live` (inférence du XGBoost sauvegardé). Le mode
live reçoit un `history_provider(sensor_id, timestep)` pour le capteur et ses
voisins directs, avec au moins 13 observations causales et contiguës par capteur.
Il peut partager une instance de `XGBoostTrafficPredictor` via `predictor=`.
Le contrat exact des 32 features, l'origine temporelle et des exemples d'appel
figurent dans [la documentation du prédicteur](../prediction/README.md).
Les exemples historiques ci-dessous conservent volontairement le mode par défaut.

Installer le paquet dans l'environnement du noyau Jupyter depuis la racine :

```powershell
python -m pip install --no-index --no-deps --no-build-isolation -e .
```

Cette commande utilise les dépendances déjà présentes et ne télécharge rien.
Les notebooks peuvent ensuite importer `src.agents` depuis leur propre répertoire,
sans modifier `sys.path`. Le noyau doit utiliser cet environnement Python.

Le contexte de données du notebook 08 réside dans `src/agents/decision_agent.py`,
qui délègue les décisions à `src/decision/` :
`build_traffic_state` construit l'état causal et récupère les voisins directs au
même timestep ; `indicators` et `rule_based_decision` appliquent les seuils train
existants ; `validate_decision`, `decision_contract` et `is_grounded_decision`
contrôlent le JSON et l'ancrage aux faits. `final_decision` refuse les états
modifiés et applique le fallback si le LLM est indisponible ou non conforme.
Les garde-fous de priorité multi-agent sont dans `src/decision/guardrails.py`,
appelés par les adaptateurs de `traffic_prediction_agent.py`.

Pour utiliser ces fonctions indépendamment du notebook :

```python
from src.agents.decision_agent import create_runtime

rules = create_runtime()  # Aucun chargement ni appel LLM à la création.
rules.initialize(root)   # Lecture et validation explicites des exports existants.
state = rules.build_traffic_state(sensor_id, timestep)
decision = rules.rule_based_decision(state)
assert rules.validate_decision(decision)
assert rules.is_grounded_decision(decision, state)
```

Chaque runtime possède sa configuration et ses données : il peut être utilisé
par un agent sans modifier celles d'une autre instance. L'import historique
`scripts.decision_agent` reste compatible, notamment pour le notebook 09.
Le notebook 08 conserve ses explications et ses sorties sauvegardées et importe
les fonctions partagées. Les helpers de sélection, d'évaluation et de sauvegarde
sont disponibles dans le même module ; seul un appel explicite à `save_results`
écrit des fichiers.

Tests hors ligne, sans réentraînement ni écriture dans les résultats du dépôt :

```powershell
python -B -X utf8 -m unittest tests.test_decision_agent -v
```

Depuis la racine GoFlow, un échange avec les **mesures réelles déjà sauvegardées** :

```python
import json
from pathlib import Path
from src.agents import TrafficPredictionControlAgent

root = Path.cwd()
state = json.loads(
    (root / "results/decision_agent/agent_inputs.json").read_text(encoding="utf-8")
)[0]["state"]
agent = TrafficPredictionControlAgent(root)
message = {
    "request_id": "goflow-replay-001",
    "timestamp": None,
    "sensor_id": state["sensor_id"],
    "timestep": state["timestep"],
    "traffic_data": {k: state[k] for k in
                     ["current_flow", "current_speed", "current_occupancy"]},
    "incident_context": None,
    "emergency_context": None,
    "cooperative_context": None,
    "public_transport_context": None,
}
response = agent.handle_message(message)
assert agent.validate_response(response)
print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
```

Les mesures reçues doivent correspondre au snapshot historique de la prévision.
`timestamp` décrit l'enveloppe orchestrateur, sans inventer de date PEMS08 ;
`timestep` identifie explicitement l'observation. Un timestep hors de l'export produit
une erreur contrôlée. Les contextes absents restent inconnus. Les signaux vrais
d'incident/urgence/transport doivent venir de leurs fournisseurs, pas de cet agent.
Une urgence ou un incident critique impose `manual_review`, sans laisser le LLM
proposer une action contradictoire. Aucun ordre n'est envoyé à un feu.

Par défaut, les sorties réelles sauvegardées du notebook 09 sont **rejouées**,
pas recalculées ; les réponses non conformes déclenchent le fallback. Le champ
`metadata.llm_mode` rend ce mode explicite. Un backend vivant peut être injecté :

```python
from src.backends.llama_cpp import LlamaCppBackend

with LlamaCppBackend(root) as backend:
    agent = TrafficPredictionControlAgent(root, llm_backend=backend)
    response = agent.handle_message(message)
```

Cette option démarre uniquement le serveur CPU local préparé dans 09. Elle ne change
pas le forecasting et n'effectue aucun entraînement. Le backend doit gérer son
timeout et le contrat enrichi ; le modèle actuel peut toujours être refusé par
les garde-fous de grounding. Le contexte ferme le serveur et supprime uniquement
son journal temporaire, y compris en cas d'erreur. Aucun journal historique n'est
écrasé et aucun téléchargement n'est lancé. `backend.last_call` conserve la sortie
brute, la durée et les erreurs ; `backend.validation_report(agent, state)` expose
les diagnostics partagés sans redéfinir les validateurs dans le notebook.
L'import historique `scripts.local_llm_agent` reste compatible pour le notebook 09.
Les 24 tests de la première partie de 10 utilisent le replay ou des fixtures ;
seule la démonstration finale peut appeler le modèle local.

Les contextes artificiels de test portent `integration_test_fixture=true`, propagé
à la réponse et aux logs. `MockOrchestrator` et les fixtures sont dans
`tests/multi_agent_integration_cases.py`. Un chemin LLM valide simulé y teste le
contrat ; il ne prouve pas une nouvelle décision réelle ni la qualité du modèle.

Vérification des 24 cas existants et de la non-régression de la démo, sans LLM :

```powershell
python -B -X utf8 -m unittest tests.test_llama_cpp_backend -v
```

Régénération explicite du notebook et de ses résultats depuis la racine :

```powershell
python -X utf8 scripts/create_multi_agent_integration_notebook.py
```

Le notebook 10 et ses seuls résultats sont régénérés. Les notebooks 01–09 et tous
les résultats antérieurs sont vérifiés par empreinte SHA256 et restent inchangés.
La couche est prête pour un branchement **de test** avec le contrat historique ;
un fonctionnement temps réel et un contrôle de feux opérationnel exigent une
stratégie de forecasting hors export, un contexte de commande, l'authentification
des fournisseurs et une politique validée par l'équipe.

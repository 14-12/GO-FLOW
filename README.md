# GO-FLOW

## Video incident integration

The integration branch adds an MP4 incident detector and a deterministic
handoff to the existing Traffic Prediction agent. See
[`docs/incident_prediction_integration.md`](docs/incident_prediction_integration.md)
for the provider contract and
[`agents/incident_detection/README.md`](agents/incident_detection/README.md)
for detector setup. Run `python -m scripts.run_incident_pipeline highway.mp4`
to obtain per-window detector results. A genuine +15 minute forecast additionally
requires measured PEMS08 flow, speed, occupancy, and causal history from
callable providers; the MP4 alone cannot supply them.

## Composant GoFlow — Traffic Prediction & Adaptive Signal Control

Composant expérimental de prévision du trafic et d'aide à la décision sur PEMS08,
préparé pour une intégration avec un orchestrateur multi-agent. Il ne constitue
pas un système prêt pour la production.

## Rôle et périmètre

`TrafficPredictionControlAgent` reçoit des observations et des contextes externes,
obtient une prévision d'occupation à **+15 minutes**, interprète l'état du trafic
et renvoie une recommandation structurée avec sa justification et sa source.
Les voisins directs du graphe alimentent le contexte spatial.

L'agent ne détecte lui-même **ni les incidents ni les véhicules d'urgence**.
Il ne fournit pas de plan de feux, n'optimise pas leurs durées et n'envoie aucune
commande à une infrastructure réelle. Les actions retournées sont des
recommandations à destination de l'orchestrateur. Un appel à l'agent ne
réentraîne aucun modèle.

## Architecture courte

```text
Message de l'orchestrateur
  → src/contracts : contrat et validation
  → src/agents : état du capteur, contexte externe et voisins
  → prévision : replay sauvegardé ou src/prediction (XGBoost)
  → src/decision : règles, proposition LLM éventuelle, garde-fous et fallback
  → réponse structurée à l'orchestrateur
```

`src/backends/llama_cpp.py` gère le backend Qwen local et son cycle de vie.
Pour les nouvelles intégrations, la façade
[`src.llm.LocalLLMBackend`](src/llm/README.md) expose `start()`, `generate(state)`
et `stop()` ; elle est directement injectable comme `llm_backend` dans l'agent.
Les notebooks exposent les expériences ; les modules sous `src/` rendent la
logique réutilisable. Voir aussi la [documentation de l'agent](src/agents/README.md)
et celle de la [couche de décision](src/decision/README.md).

## Modèles benchmarkés et modèle retenu

| Modèle | Expérience sauvegardée |
|---|---|
| XGBoost | Benchmark tabulaire, notebook 03 |
| LSTM | Benchmark séquentiel, notebook 03 |
| PatchTST | Benchmark par patches, notebook 03 |
| STGCN | Benchmark spatio-temporel sur graphe, notebook 06 |
| Chronos Bolt Tiny | Benchmark zero-shot, sans fine-tuning PEMS08, notebook 07 |

**XGBoost est le modèle retenu pour la prévision de l'agent.** Son export est
`results/model_benchmark/xgboost/xgboost_model.json`. Le prédicteur réutilisable
conserve les 32 features et leur ordre d'entraînement ; il exige au moins
13 observations contiguës et causales par capteur. Le
[contrat du prédicteur](src/prediction/README.md) décrit ces contraintes.
Les [résultats sauvegardés](results/model_benchmark/benchmark_results.csv)
documentent les comparaisons ; leur présence ne démontre pas une généralisation
à d'autres réseaux ou conditions d'exploitation.

## Qwen, garde-fous et fallback

Qwen2.5-0.5B-Instruct est un **backend de décision facultatif**, distinct du
modèle de prévision. Le parcours local utilise notamment sa version GGUF via
llama.cpp. Le LLM propose une décision à partir de l'état fourni ; il ne calcule
pas la prévision XGBoost et ne prend pas systématiquement la décision finale.

Les contrôles portent sur le JSON, les champs et valeurs autorisés, l'ancrage
aux faits, les incidents ou urgences inventés et la cohérence risque/action.
Les priorités critiques sont protégées avant et après le parcours LLM ; les
contextes d'urgence ou d'incident conduisent à un arbitrage par l'orchestrateur.
Une sortie absente, invalide, non ancrée ou une erreur du backend déclenche le
fallback déterministe. Ces contrôles expérimentaux ne garantissent pas
l'absence de toute hallucination.

En replay historique, sans backend injecté, certaines réponses sauvegardées du
notebook 09 peuvent être rejouées puis validées. Ce parcours est signalé par
`metadata.llm_mode` : il ne s'agit pas d'une nouvelle inférence LLM. En mode
`live`, ces anciennes réponses ne sont pas rejouées implicitement.

## Contrat input/output et exemple minimal

La source unique est `src/contracts/` : schémas input/output, enums, exemples JSON
et validation Python. Le [contrat d'intégration officiel](docs/traffic_prediction_agent_contract.md)
détaille les types, les champs optionnels et les réponses warning/error.

L'entrée exige `request_id`, `sensor_id`, `timestep` et `traffic_data` avec
`current_flow`, `current_speed`, `current_occupancy`. Les contextes optionnels
`incident_context`, `emergency_context`, `public_transport_context` et
`cooperative_context` sont fournis par les agents spécialisés via l'orchestrateur.
Une information absente reste inconnue.

La réponse contient notamment `forecast`, `traffic_state`, `risk_level`,
`recommended_action`, `reason`, `confidence`, `decision_source`, `status` et
`metadata`. La source distingue `llm` de `rule_based_fallback` ; les métadonnées
précisent le replay, les garde-fous et les motifs de fallback. Le score
`confidence` n'est pas une probabilité calibrée. `warning` peut accompagner une
réponse exploitable ; il ne signifie pas nécessairement un échec.

Exemple sur un snapshot historique existant, après installation du socle,
avec les données et exports du dépôt disponibles :

```python
from pathlib import Path
from src.agents import TrafficPredictionControlAgent

# Exécuter depuis la racine du dépôt ; les mesures correspondent au snapshot.
agent = TrafficPredictionControlAgent(
    project_root=Path.cwd(), prediction_mode="historical_replay"
)
message = {
    "request_id": "exemple-trafic",
    "sensor_id": 33,
    "timestep": 15227,
    "traffic_data": {
        "current_flow": 0.0,
        "current_speed": 65.0,
        "current_occupancy": 0.0,
    },
}
response = agent.handle_message(message)
print(response)  # Dans un notebook : display(response).
```

## Limites actuelles

- **Replay historique par défaut** : observations concordantes avec les exports,
  timesteps 15177 à 17853. Le mode `live` existe, mais exige un fournisseur
  d'historique causal pour le capteur et ses voisins ; aucun flux terrain n'est
  automatiquement connecté.
- **Mapping spatial** : le graphe utilise l'hypothèse `location = index NPZ`.
  L'identité reste ambiguë pour 25 capteurs aux séries identiques. Voir
  l'[audit spatial](results/spatial_audit/spatial_audit.md) et la
  [préparation du graphe](results/spatial_audit/graph_preparation.md).
- **Aucun contrôle réel des feux** : ni plan disponible, ni commande exécutée,
  ni validation d'un gain opérationnel sur une infrastructure réelle.
- **LLM facultatif** : disponibilité locale et sorties conformes non garanties ;
  les règles et le fallback restent nécessaires.
- **Périmètre PEMS08** : identifiants, unités et grille temporelle existants.
  Le mode `live` ne définit pas une adaptation automatique à un autre réseau.

## Structure du dépôt

```text
src/
  agents/       API de l'agent et construction de l'état
  prediction/   Inférence XGBoost réutilisable
  decision/     Moteur de décision, garde-fous et politique
  contracts/    Schémas, enums, exemples JSON et validateurs
  backends/     Intégration LLM locale
  llm/          Interface publique du backend local et cycle de vie explicite
  config/       Compatibilité des imports de configuration
notebooks/      Exploration, benchmarks et démonstrations 08–10
scripts/        Préparation, audits et outils expérimentaux
tests/          Tests unitaires, cas d'intégration et fixtures
docs/           Contrat d'intégration et audit avant push
data/           Données source et données de travail
models/         Poids des backends locaux
results/        Exports, métriques et sorties expérimentales sauvegardées
requirements*.txt  Dépendances par mode
pyproject.toml    Métadonnées du paquet et extras
```

## Installation en équipe

Utiliser Python 3.11 (version de référence : 3.11.9), depuis la racine du dépôt.
Créer un **nouvel environnement dédié**, sans modifier les environnements existants :

```powershell
python -m venv .venv-team
.\.venv-team\Scripts\Activate.ps1
```

Choisir ensuite un mode ci-dessous. Les commandes sont des instructions pour
l'équipe : leur préparation n'a installé aucun package. Elles nécessitent un
accès à un index de packages ou à un miroir interne.

### Agent minimal : décisions et replay historique

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Ce socle contient NumPy, pandas, psutil et jsonschema. Le mode par défaut
`prediction_mode="historical_replay"` utilise les prédictions sauvegardées, sans
charger de modèle ML ni de LLM. Le fallback déterministe fonctionne sans Qwen.

Pour l'inférence `prediction_mode="live"`, ajouter l'extra XGBoost existant :

```powershell
python -m pip install -e ".[prediction]"
```

Cet extra impose `xgboost==3.2.0`, version de l'export sauvegardé. Le mode live
nécessite également un `history_provider` causal ; voir le
[contrat des features et de l'historique](src/prediction/README.md).

### Agent avec backend Qwen local

```powershell
python -m pip install -r requirements-llm.txt
python -m pip install -e .
```

Ce fichier inclut le socle et ajoute PyTorch et Transformers, importés par la
préparation et les diagnostics du backend actuel, y compris son parcours GGUF.
Il n'installe pas `llama-cpp-python` : le backend utilise un exécutable llama.cpp
externe. Les poids GGUF, les binaires et l'archive `release.zip` vérifiée par le
backend doivent être fournis séparément aux chemins attendus par
`src/backends/llama_cpp.py`. Leur provenance est enregistrée dans
`models/qwen2.5-0.5b-instruct-gguf/download_provenance.json`.
Les poids Transformers sont également nécessaires pour les diagnostics qui les
chargent explicitement. Aucun téléchargement de modèle n'est lancé par ces
commandes d'installation Python.

### Recherche : benchmarks et notebooks

```powershell
python -m pip install -r requirements-research.txt
python -m pip install -e .
python -m ipykernel install --user --name goflow-team --display-name "GoFlow équipe"
```

Sélectionner ce noyau dans l'éditeur de notebooks. Le fichier recherche inclut
le mode LLM, XGBoost, PyTorch pour LSTM/PatchTST/STGCN, `chronos-forecasting`,
scikit-learn, les figures Matplotlib et les exports Excel openpyxl. Les outils
Jupyter sont utilisés pour afficher, lire et exécuter les notebooks ;
`huggingface-hub` et `packaging` sont importés par les outils Chronos.
L'installation ne lance aucun notebook, entraînement ou benchmark. Les scripts
`prepare_*` peuvent télécharger des poids : les exécuter constitue une opération
distincte, inutile pour simplement lire les résultats sauvegardés.

## Versions et fichiers nécessaires

Les versions directes sont fixées à celles connues localement : XGBoost provient
de `results/model_benchmark/xgboost/xgboost_export_metadata.json` ; Chronos,
PyTorch et Transformers sont cohérents avec
`results/model_benchmark/chronos_environment.json`. Les autres versions ont été
relevées avec les métadonnées des packages installés. L'environnement Chronos
enregistre un build CPU `2.11.0+cpu` ; la contrainte `torch==2.11.0` fixe la version
publique, sans imposer un index ou un build CPU/CUDA particulier.
Ces fichiers listent les dépendances directes et le noyau nécessaire ; ils ne
constituent pas un verrouillage complet des dépendances transitives ni une
garantie de reproduction numérique sur une autre machine.

Le replay dépend notamment de `data/raw/traffic_dataset.csv` et des exports dans
`results/` ; l'inférence live charge
`results/model_benchmark/xgboost/xgboost_model.json` et ses métadonnées. Garder
les fichiers existants : leur récupération n'est pas effectuée par pip. Consulter
l'[audit avant push](docs/pre_push_audit.md) pour les données et poids volumineux.

## Validation finale avant push

Dans l'environnement dédié aux tests, installer le paquet et son extra `test`
(pytest 8.4.2 et XGBoost 3.2.0), puis lancer toute la suite :

```powershell
python -m pip install -e ".[test]"
pytest tests -q
```

Les tests historiques `unittest` restent collectés sans modification. Les tests
`tests/test_pre_push.py` vérifient explicitement les contrats input/output,
le trafic normal, la congestion, le fallback LLM, les priorités incident/urgence,
le transport public, les voisins invalides, les données incohérentes et un
backend indisponible. Les **24 cas d'intégration historiques** sont également
comparés individuellement aux diagnostics sauvegardés.

Ces tests lisent les exports existants, simulent les backends LLM et ne lancent
ni serveur réel ni entraînement XGBoost/LSTM/PatchTST/STGCN/Chronos. La suite
complète inclut l'inférence du XGBoost sauvegardé ; aucun benchmark n'est relancé.
Le cache pytest est désactivé dans `pyproject.toml`.

Validation locale du 21 octobre 2026 : **104 tests réussis en 34,93 secondes**
avec Python 3.11.9, dont les 24 scénarios historiques comparés aux sorties
sauvegardées. Les empreintes SHA-256 des 120 fichiers sous `data/`, `models/`,
`results/` et `notebooks/` sont restées identiques avant et après cette exécution.
Ce résultat décrit cette exécution locale, sans garantir celle d'un autre environnement.

Pour exécuter uniquement la validation ciblée :

```powershell
pytest tests/test_pre_push.py -q
```

Les anciens tests peuvent aussi être lancés avec la commande standard
(`test_pre_push.py` utilise pytest et n'est pas collecté par unittest) :

```powershell
python -B -X utf8 -m unittest discover -s tests -v
```

Cette suite ne réentraîne aucun modèle. Installer le paquet avec `-e .` permet
aux notebooks d'importer `src` sans ajouter de manipulation de `sys.path`.

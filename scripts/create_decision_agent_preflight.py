"""Créer le notebook 08 arrêté au prérequis manquant, sans modifier les notebooks validés."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
protected = [*sorted((root / "notebooks").glob("0[1-7]*.ipynb")),
             *sorted((root / "results/model_benchmark").iterdir()),
             *sorted((root / "data").rglob("*"))]
protected = [p for p in protected if p.is_file()]
def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
before = {p: digest(p) for p in protected}
path = root / "notebooks/08_llm_decision_agent.ipynb"
if path.exists():
    raise FileExistsError(path)
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell
cells = [md("""# GoFlow — Agent décisionnel : contrôle des prérequis

**État : arrêté avant la construction de l'agent.** Les scores et graphiques XGBoost
ne contiennent ni le modèle entraîné ni ses 455 090 prédictions. Aucun scénario,
aucune recommandation et aucune évaluation d'agent ne peuvent être produits à partir
de ces seuls fichiers. Les notebooks 01 à 07, les données et le benchmark restent intacts.

Ce notebook exécute uniquement un audit en lecture seule. Il ne réentraîne aucun
modèle et n'utilise aucune prédiction d'un autre modèle comme substitut XGBoost.
"""), code("""# Identifier le projet sans dépendre du dossier de lancement du noyau.
from pathlib import Path
import sys, json, importlib.util, shutil, urllib.request
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'notebooks/03_model_benchmark.ipynb').exists())
folder = root / 'results/model_benchmark'
print('Projet :', root)
print('Python :', sys.version)
"""), md("""## Environnement local, sans installation

La présence de `transformers` seule ne prouve pas qu'un modèle instruct est disponible.
Chronos est un modèle de prévision temporelle, pas un LLM instruct décisionnel.
La RAM disponible est mesurée au moment de l'exécution. Aucun téléchargement n'est lancé.
"""), code("""# Vérifier les bibliothèques, le GPU, la mémoire et le service Ollama local.
packages = {name: importlib.util.find_spec(name) is not None
            for name in ['torch', 'transformers', 'llama_cpp', 'ollama', 'psutil']}
environment = {'python': sys.version, 'packages': packages,
               'ollama_executable': shutil.which('ollama'),
               'gguf_files': [str(p.relative_to(root)) for p in (root / 'models').rglob('*.gguf')]}
if packages['torch']:
    import torch
    environment.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
if packages['psutil']:
    import psutil
    memory = psutil.virtual_memory()
    environment.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available)
try:
    # Interroger uniquement localhost, sans proxy et sans téléchargement de modèle.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:11434/api/tags', timeout=2) as response:
        environment['ollama_models'] = json.load(response).get('models', [])
except (OSError, ValueError) as error:
    environment['ollama_models'] = []
    environment['ollama_service_status'] = type(error).__name__
print(json.dumps(environment, ensure_ascii=False, indent=2))
"""), md("""## Recherche des artefacts XGBoost

Les graphiques, l'importance des variables et les scores de validation sont exclus
des prévisions réutilisables. Les éventuels fichiers détectés doivent ensuite être
validés : leur présence seule n'autorise pas l'agent.
"""), code("""# Afficher tous les artefacts du benchmark et rechercher les candidats, sans les exécuter.
inventory = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size}
             for p in sorted(folder.rglob('*')) if p.is_file()]
for row in inventory:
    print(row['path'], row['bytes'])
predictions = [p for p in folder.rglob('*') if p.is_file() and p.stat().st_size > 0
               and ('xgboost' in p.name.lower() or 'xgb' in p.name.lower())
               and ('prediction' in p.name.lower() or 'predictions' in p.name.lower())
               and p.suffix.lower() in {'.csv', '.npz', '.npy', '.parquet'}]
models = [p for p in folder.rglob('*') if p.is_file() and p.stat().st_size > 0
          and ('xgboost' in p.name.lower() or 'xgb' in p.name.lower())
          and p.suffix.lower() in {'.json', '.ubj', '.pkl', '.joblib', '.model', '.bin'}
          and 'metadata' not in p.name.lower()]
agent_ready = False
if not predictions and not models:
    status = 'BLOQUÉ : ni prédictions XGBoost ni modèle sauvegardé.'
else:
    status = 'Artefacts détectés : validation et implémentation encore nécessaires.'
print(status)
print('Prédictions candidates :', [str(p) for p in predictions])
print('Modèles candidats :', [str(p) for p in models])
print('Aucune décision générée. Aucun fichier de benchmark modifié.')
"""), md("""## Export requis depuis le noyau encore actif du notebook 03

Dans une **console attachée au noyau qui a déjà exécuté le notebook 03**, lancer le
script fourni avec `%run -i` afin de réutiliser `best_model`, `test`, `y_pred`,
`feature_columns` et `type_capteur`. Ne pas relancer les cellules d'entraînement.
Depuis le dossier `notebooks/` :

```python
%run -i ../scripts/export_existing_xgboost.py
```

Depuis la racine GoFlow :

```python
%run -i scripts/export_existing_xgboost.py
```

Le script refuse les objets manquants et les fichiers déjà présents. Il vérifie
455 090 lignes, l'ordre exact **location puis timestep**, les clés test
`location=0..169`, `timestep=15177..17853`, les valeurs finies et la concordance
MAE/RMSE/R² avec les scores enregistrés. Il exporte, sans entraînement ni inférence :

- `results/model_benchmark/xgboost_test_predictions.csv` : clés, cible de contrôle,
  prévision `predicted_occupancy_t15` ;
- `results/model_benchmark/xgboost_best.json` : modèle existant ;
- `results/model_benchmark/xgboost_export_metadata.json` : features, catégories,
  meilleure itération et horizon réels issus du noyau.

**Si le noyau a été arrêté et qu'aucune autre sauvegarde n'existe, ces objets ne
peuvent pas être récupérés à partir des scores.** Cet export s'arrête alors sans
relancer le forecasting. Aucune cible future ne sera fournie à l'agent.
"""), md("""## Architecture prévue après levée du blocage

```text
PEMS08 → Feature Engineering → XGBoost forecasting → Prediction t+15
  → Traffic state builder
      ├── état courant
      ├── tendances passées uniquement
      └── voisins directs du graphe
  → Decision Agent
      ├── garde-fous déterministes
      └── raisonnement LLM, si disponible et conforme
  → recommandation structurée → Orchestrator / Adaptive Traffic Control
```

La suite reste **à implémenter**, après validation des prévisions :
`rule_based_decision(state)`, `llm_decision(input_state)` et `final_decision(state)`.
Un LLM absent ou invalide devra déclencher le fallback déterministe. La sortie
contiendra exactement `traffic_state`, `risk_level`, `recommended_action`, `reason`,
`confidence`. Cette dernière sera un score opérationnel heuristique documenté,
pas une probabilité de réussite scientifiquement calibrée.

Les seuils seront centralisés et présentés comme expérimentaux, définis sans
ajustement sur test. Les cinq scénarios seront sélectionnés sur les véritables
observations et prévisions : fluide, occupation élevée, hausse prévue, amélioration,
voisinage contrasté. Aucun scénario factice n'est créé pour contourner le blocage.

### Contexte spatial et limites

Le graphe symétrisé enregistré dans `results/model_benchmark/stgcn_adjacency.npy`
provient des distances ASTGCN PEMS08. Il permet de retrouver les voisins directs,
sans inférer une proximité à partir d'identifiants consécutifs.
Le mapping `location = index NPZ` est une hypothèse documentée pour 25 capteurs
aux séries identiques, répartis en quatre groupes. Il ne prouve pas leur identité
physique individuelle. L'état des voisins devra être lu au même timestep courant.

PEMS08 ne fournit pas de plans de feux ni de correspondance capteur/carrefour :
les recommandations ne constituent pas des commandes exécutables de feux.
Les champs `incident_detected` et `emergency_vehicle_present` resteront `null`.
L'Incident Detection Agent et l'Emergency Agent pourront les fournir ultérieurement ;
le Cooperative Traffic Agent fournira du contexte spatial, le Traffic Prediction
Agent les prévisions, et l'Orchestrator combinera ces informations.

### Consigne française prévue pour le LLM

> Tu es l'agent Traffic Prediction & Adaptive Signal Control de GoFlow. Tu reçois
> l'état réel courant, la prévision à +15 minutes, les tendances passées et les
> voisins observés. Utilise uniquement ces informations et signale les informations
> manquantes. Produis uniquement un JSON valide, avec une justification courte liée
> aux valeurs fournies. N'altère pas les prévisions, n'invente ni données, ni voisins,
> ni précision scientifique. Ne prétends pas détecter un accident ou un véhicule
> d'urgence. Choisis seulement keep_current_signal_plan, monitor,
> extend_green_phase, reduce_green_phase, coordinate_with_neighbor ou manual_review.
> Ne donne aucune durée exacte de feu vert. traffic_state appartient à free_flow,
> moderate, congested, severe ; risk_level à low, medium, high ; confidence est
> compris entre 0 et 1.

La validation future contrôlera le schéma JSON, les actions autorisées, les données
manquantes et la cohérence des décisions. La vérification automatique de schéma
ne prouve pas à elle seule l'absence de fabrication dans une justification libre :
une revue de l'ancrage aux valeurs d'entrée restera nécessaire. Aucune accuracy LLM
ne sera revendiquée sans protocole et annotations appropriés.
"""), md("""## Conclusion de cette exécution

**Arrêt propre : les prévisions et le modèle XGBoost ne sont pas sauvegardés.**
L'environnement dispose de PyTorch CPU et de transformers, sans CUDA ; aucun
modèle instruct local utilisable n'a été identifié par les vérifications exécutées.
Aucune installation n'a été effectuée. L'absence de LLM n'empêchera pas le fallback
futur, mais l'absence de prévisions réelles empêche actuellement toute décision.

Les fichiers `selected_scenarios.csv`, `agent_inputs.json`, `agent_outputs.json` et
`validation_results.csv` ne sont volontairement pas créés : ils représenteraient
des résultats qui n'existent pas. Seul le document méthodologique décrit ce blocage.
Les notebooks validés et `benchmark_results.csv` sont préservés.
""")]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbformat.write(notebook, path)
NotebookClient(notebook, timeout=120, kernel_name="python3",
               resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(notebook, path)
output = root / "results/decision_agent"
output.mkdir(parents=True, exist_ok=True)
(output / "decision_agent_methodology.md").write_text(
    "# GoFlow — état de préparation de l'agent décisionnel\n\n"
    "Statut : arrêté, ni modèle ni prédictions XGBoost sauvegardés.\n\n"
    "L'audit du notebook 08 répertorie les artefacts et l'environnement local. "
    "Les fichiers de scores, d'importance et les images ne sont pas des prévisions. "
    "Aucun scénario, résultat d'agent ou métrique d'agent n'a été produit.\n\n"
    "Exporter les objets existants du noyau 03 avec `%run -i "
    "../scripts/export_existing_xgboost.py` depuis notebooks/, sans relancer ses cellules. "
    "Le script vérifie les clés de 455 090 lignes et les scores sauvegardés puis "
    "exporte les prédictions, le modèle et ses métadonnées réelles. "
    "Un noyau perdu ne peut pas être reconstitué à partir des seules métriques.\n\n"
    "Après export : vérifier l'alignement aux données brutes et à la cible t+3 ; "
    "construire les tendances avec le passé uniquement et lire les voisins directs "
    "du graphe STGCN symétrisé au même instant. Le mapping location=index NPZ "
    "reste hypothétique pour 25 capteurs aux séries identiques. "
    "Incidents et véhicules d'urgence restent inconnus (null). "
    "PEMS08 ne fournit pas de contexte de contrôle des feux.\n\n"
    "L'agent hybride, son fallback, ses cinq scénarios réels et ses tests restent "
    "à implémenter après levée de ce prérequis. Aucun téléchargement ni installation. "
    "Les notebooks 01 à 07, les métriques et les données sources sont préservés.\n",
    encoding="utf-8")
assert all(digest(p) == value for p, value in before.items()), "Fichier protégé modifié"
print("Notebook exécuté ; prérequis manquant documenté ; tous les fichiers protégés inchangés.")

if __name__ == "__main__":
    pass

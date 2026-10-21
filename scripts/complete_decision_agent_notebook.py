"""Compléter et exécuter uniquement le notebook 08 avec les exports réels existants."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
protected = [p for base in [root / "data", root / "results/model_benchmark"]
             for p in base.rglob("*") if p.is_file()]
protected += list((root / "notebooks").glob("0[1-7]*.ipynb"))
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
before = {p: digest(p) for p in protected}
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
# Générer uniquement les imports du module partagé, sans recopier ses fonctions.
sections = {
    'Imports et contrat': "# Utiliser la même implémentation que les agents, depuis la racine ou notebooks/.\nfrom pathlib import Path\nimport copy\nimport json\nimport sys\n\nimport numpy as np\nimport pandas as pd\n\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents]\n            if (p / 'src/agents/decision_agent.py').exists())\nif str(root) not in sys.path:\n    sys.path.insert(0, str(root))\nfrom src.agents.decision_agent import create_runtime\n\n# Isoler les données, les seuils et le backend des autres agents du même processus.\n_rules = create_runtime()\nTRAFFIC_STATES = _rules.TRAFFIC_STATES\nRISK_LEVELS = _rules.RISK_LEVELS\nACTIONS = _rules.ACTIONS\nDECISION_KEYS = _rules.DECISION_KEYS\nCONFIG = _rules.CONFIG\nMAPPING_GROUPS = _rules.MAPPING_GROUPS\nSYSTEM_PROMPT = _rules.SYSTEM_PROMPT\nfile_hash = _rules.file_hash\ninspect_environment = _rules.inspect_environment\n",
    'Chargement et validation des données': 'def initialize(root):\n    # Exposer les données pour les explications et tests, sans recopier la logique.\n    result = _rules.initialize(root)\n    for name in ("ROOT", "PREDICTIONS", "OBSERVATIONS", "ADJACENCY", "LOOKUP",\n                 "THRESHOLDS", "ENVIRONMENT"):\n        globals()[name] = getattr(_rules, name)\n    return result\n',
    "Construction causale de l'état": '# Construction causale et récupération des voisins dans le module partagé.\nbuild_traffic_state = _rules.build_traffic_state\nindicators = _rules.indicators\n',
    'Règles et validation stricte': '# Contrat, règles et garde-fous partagés avec le code de production.\nrule_based_decision = _rules.rule_based_decision\nstrict_json_loads = _rules.strict_json_loads\nvalidate_decision = _rules.validate_decision\ndecision_contract = _rules.decision_contract\nis_grounded_decision = _rules.is_grounded_decision\n',
    'Interface LLM et fallback': "# L'interface et le fallback utilisent le même contexte que le state builder.\nllm_decision = _rules.llm_decision\nfinal_decision = _rules.final_decision\n",
    'Sélection des cinq scénarios réels': '# Sélection déterministe conservée, sans réentraîner ni recalculer de prévisions.\nselect_scenarios = _rules.select_scenarios\n',
    'Validation des scénarios et sauvegarde': '# Helpers expérimentaux explicites : aucune sauvegarde au simple import du module.\nevaluate_scenarios = _rules.evaluate_scenarios\nsave_results = _rules.save_results\n',
}
methodology = """# GoFlow — Agent Traffic Prediction & Adaptive Signal Control

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
"""
cells = [md("""# GoFlow — Agent décisionnel à partir des prévisions XGBoost

Le forecasting répond : **« Que va-t-il se passer dans 15 minutes ? »**
L'agent répond : **« Que recommander en fonction de cette prévision et du contexte ? »**

Le prérequis XGBoost est désormais disponible. Ce notebook remplace le diagnostic
bloqué par un pipeline exécutable, sans réentraînement ni modification des benchmarks.
"""), md("""## 1. Contrat et environnement

Vérifier les capacités locales sans installer de dépendances. Si aucun petit LLM
instruct local n'est disponible, utiliser l'agent déterministe et conserver une
interface remplaçable. Les sorties finales sont un JSON strict de cinq champs ;
la source de décision est conservée séparément dans une enveloppe de suivi.
"""), code(sections["Imports et contrat"]),
md("""## 2. Charger et réaligner les 455 090 prévisions

Vérifier toutes les clés location/timestep, les valeurs finies, la provenance du CSV
brut et la concordance des mesures et cibles. L'audit lit les cibles futures pour
contrôler l'export ; le state builder ne les reçoit jamais.

Le graphe 170×170 symétrisé provient des distances réelles ASTGCN PEMS08.
Le mapping reste **une hypothèse pour 25 capteurs aux séries identiques** ;
leur identité physique ne peut pas être prouvée par leurs valeurs seules.
"""), code(sections["Chargement et validation des données"]),
code("""# Résoudre la racine depuis notebooks/ ou depuis le projet.
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'notebooks/03_model_benchmark.ipynb').exists())
display(initialize(root))
print(json.dumps(ENVIRONMENT, ensure_ascii=False, indent=2))
print('Seuils expérimentaux ajustés sur train uniquement :')
print(json.dumps(THRESHOLDS, ensure_ascii=False, indent=2))
"""), md("""## 3. Construire un état causal et son contexte spatial

Les tendances utilisent t−6…t−1 du même capteur : différence entre les extrémités,
divisée par cinq intervalles. Les voisins sont les indices connectés par une arête
réelle, avec observations et prévisions au même t. Aucune relation n'est déduite
de numéros consécutifs. Incidents et urgences restent `null`.
"""), code(sections["Construction causale de l'état"]),
md("""## 4. Règles et validation JSON

Les quantiles train sont des seuils opérationnels **expérimentaux**. La politique
prend en compte l'occupation courante et prévue, la hausse prévue, la vitesse, les
tendances et les voisins. Elle ne connaît aucun plan de feux : aucune extension
ou réduction de phase n'est autorisée dans ce contexte. `confidence=0.5` est
non calibrée et ne représente pas une probabilité scientifique.

La validation refuse aussi les champs supplémentaires, les clés JSON dupliquées,
NaN, Infinity et un booléen utilisé comme confidence. Le contrat de grounding
vérifie ensuite la cohérence et l'ancrage de la justification aux données.
"""), code(sections["Règles et validation stricte"]),
md("""## 5. Interface LLM, consigne française et fallback

Le system prompt est défini dans la première cellule de code. Le LLM reçoit les
valeurs réelles, les indicateurs et les choix autorisés. Cette version lui permet
de choisir une justification parmi deux textes calculés ; elle refuse toute
justification libre susceptible d'introduire des faits non fournis.

`final_decision` vérifie l'entrée contre les sources, tente le LLM, puis utilise
les règles si le LLM est absent, échoue ou produit une sortie invalide/non ancrée.
L'appel reçoit une copie de l'état pour empêcher la modification des entrées.
"""), code(sections["Interface LLM et fallback"]),
md("""## 6. Sélection automatique de cinq vrais cas test

Sélectionner des observations distinctes par critères sur les mesures courantes,
les prévisions et les voisins. Aucun critère ne consulte la cible réellement
observée à t+3. Ce sont des illustrations extrêmes, pas un échantillon représentatif.
"""), code(sections["Sélection des cinq scénarios réels"]),
code("""# Afficher les critères et les valeurs mesurées/prédites réellement sélectionnées.
scenarios = select_scenarios()
display(scenarios[['scenario', 'location', 'timestep', 'current_occupancy',
                   'predicted_occupancy_t15', 'occupancy_change', 'current_speed',
                   'neighbor_occupancy_range', 'selection_criterion']])
assert len(scenarios) == 5
assert not scenarios.duplicated(['location', 'timestep']).any()
"""), md("""## 7. Décisions et grille de validation

Chaque cas affiche l'entrée complète avec ses voisins, la décision stricte, sa
source `llm` ou `rule_based` et la grille. Les résultats décrivent la conformité
au contrat, pas une accuracy LLM ni l'effet sur la congestion réelle.
"""), code(sections["Validation des scénarios et sauvegarde"]),
code("""# Évaluer les cinq états réels avec l'interface actuellement disponible.
inputs, outputs, validations = evaluate_scenarios(scenarios)
for input_record, output_record in zip(inputs, outputs):
    print('\\nSCÉNARIO :', input_record['scenario'])
    print(json.dumps(input_record['state'], ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(output_record, ensure_ascii=False, indent=2, allow_nan=False))
display(validations)
"""), md("""## 8. Tests unitaires de garde-fous

Les entrées de trafic proviennent d'un scénario réel. Les sorties LLM malformées
ou contradictoires ci-dessous sont **des fixtures de test**, pas des observations
ni des décisions de modèle enregistrées. Un appel valide simulé teste le chemin
LLM ; cela ne signifie pas qu'un modèle instruct a été exécuté.
"""), code("""# Vérifier le contrat et les branches de fallback sur un état réel.
state = inputs[0]['state']
baseline = rule_based_decision(state)
assert validate_decision(json.dumps(baseline, allow_nan=False))
assert not validate_decision('{')
assert not validate_decision({**baseline, 'confidence': True})
assert not validate_decision({**baseline, 'confidence': float('nan')})
assert not validate_decision({**baseline, 'confidence': 1.1})
assert not validate_decision({**baseline, 'recommended_action': 'invented_action'})
assert not validate_decision({**baseline, 'reason': ' '})
assert not validate_decision({**baseline, 'extra': 1})
duplicated = json.dumps(baseline)[:-1] + ', "confidence": 0.5}'
assert not validate_decision(duplicated)
assert not is_grounded_decision({**baseline, 'reason': 'Accident détecté.'}, state)
assert not is_grounded_decision({**baseline, 'recommended_action': 'extend_green_phase'}, state)

# Vérifier les tendances exactement sur le passé et le voisinage sur les arêtes réelles.
sensor, timestep = state['sensor_id'], state['timestep']
history = OBSERVATIONS[sensor, timestep - 7:timestep - 1]
assert state['recent_occupancy_trend'] == float((history[-1, 2] - history[0, 2]) / 5)
assert state['recent_speed_trend'] == float((history[-1, 1] - history[0, 1]) / 5)
assert {n['sensor_id'] for n in state['neighbors']} == set(np.flatnonzero(ADJACENCY[sensor] > 0)) - {sensor}
assert all(n['timestep'] == timestep for n in state['neighbors'])
assert state['incident_detected'] is None and state['emergency_vehicle_present'] is None

# Tester l'indisponibilité, l'erreur, le JSON invalide, la fabrication et un retour valide simulé.
original_llm = _rules.llm_decision
try:
    for fixture in [None, '{', {**baseline, 'reason': 'Accident détecté.'}]:
        _rules.llm_decision = lambda input_state, answer=fixture: answer
        result = final_decision(state)
        assert result['source'] == 'rule_based' and result['decision'] == baseline
    def failed_llm(input_state):
        raise OSError('Erreur simulée uniquement pour le test')
    _rules.llm_decision = failed_llm
    assert final_decision(state)['source'] == 'rule_based'
    _rules.llm_decision = lambda input_state: json.dumps(baseline, allow_nan=False)
    assert final_decision(state)['source'] == 'llm'
finally:
    _rules.llm_decision = original_llm
tampered = copy.deepcopy(state)
tampered['predicted_occupancy_t15'] += 1
try:
    final_decision(tampered)
except ValueError:
    pass
else:
    raise AssertionError('Une entrée modifiée doit être refusée')
print('Tests du schéma, du grounding, des tendances, des voisins et du fallback : OK.')
"""), md("""## 9. Sauvegarde des résultats réels

Sauvegarder les cinq scénarios, états, décisions et validations sous
`results/decision_agent/`, ainsi qu'une méthode précisant les seuils, l'environnement
et les empreintes des sources. Le CSV de benchmark n'est jamais écrit.
"""), code("# Documenter les choix et limites avec les valeurs réelles de cette exécution.\n"
            + "methodology = " + repr(methodology) + "\n"
            + "save_results(scenarios, inputs, outputs, validations, methodology)\n"),
md("""## 10. Architecture et interprétation

```text
PEMS08
  ↓
XGBoost (prévisions déjà sauvegardées)
  ↓
Prévision +15 min
  ↓
State Builder
  ├── état courant
  ├── tendances strictement passées
  └── voisins directs
  ↓
Decision Agent
  ├── LLM local si disponible et conforme
  └── fallback rule-based
  ↓
Recommandation structurée
  ↓
Orchestrator futur (aucune commande de feux envoyée)
```

Une hausse prévue motive une vigilance accrue ; une baisse prévue n'annule pas
une occupation courante élevée. Des voisins congestionnés motivent une coordination,
sous l'hypothèse de mapping documentée. Les exemples montrent le fonctionnement
des règles et des garde-fous, sans prouver leur efficacité sur un réseau réel.

Les futurs Incident Detection Agent et Emergency Agent fourniront les informations
actuellement inconnues ; Cooperative Traffic Agent fournira le contexte voisin,
Traffic Prediction Agent les prévisions et Orchestrator combinera les informations.
Le présent notebook implémente uniquement Traffic Prediction & Adaptive Signal Control.
"""), code("""# Résumer seulement ce qui a réellement été exécuté.
print('Sources des cinq décisions :', validations['source'].value_counts().to_dict())
print('LLM sélectionné :', ENVIRONMENT['selected_llm'])
print('Aucun réentraînement et aucune modification des métriques historiques.')
if ENVIRONMENT['selected_llm'] is None:
    print('Agent déterministe opérationnel ; interface LLM disponible pour une intégration ultérieure.')
""")]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
path = root / "notebooks/08_llm_decision_agent.ipynb"
nbformat.write(notebook, path)
client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                        resources={"metadata": {"path": str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
assert all(digest(p) == value for p, value in before.items()), "Fichier protégé modifié"
print("Notebook 08 exécuté ; données, notebooks 01–07 et benchmarks inchangés.")

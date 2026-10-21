"""Créer et exécuter uniquement l'intégration 10, sans toucher aux modèles existants."""
from pathlib import Path
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
protected = [p for base in [root / "data", root / "results/model_benchmark", root / "results/decision_agent", root / "results/llm_agent"]
             for p in base.rglob("*") if p.is_file()]
protected += [p for p in (root / "notebooks").glob("*.ipynb") if p.name[:2] in [f"{i:02d}" for i in range(1, 10)]]
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
before = {p: digest(p) for p in protected}
methodology = """# GoFlow — Intégration multi-agent expérimentale

## Périmètre

TrafficPredictionControlAgent, dans src/agents/traffic_prediction_agent.py, assure
la réutilisation des prévisions XGBoost à +15 minutes, la construction d'états
causaux, l'interprétation du risque et une recommandation structurée. Il ne détecte
ni accident ni véhicule d'urgence, ne calcule pas de routes et ne prédit pas les bus.
Les modèles des autres agents ne sont pas implémentés. MockOrchestrator n'est
qu'un outil de test, pas l'orchestrateur final du groupe.

## Contrats

Le message JSON requiert request_id (texte), sensor_id (0..169), timestep (15177..17853)
et traffic_data (trois mesures finies). timestamp peut être null ou une date ISO 8601
avec fuseau, représentant l'enveloppe du message. PEMS08 ne fournit pas ici d'origine
calendaire vérifiable : le timestep est obligatoire et n'est jamais déduit du timestamp.
Les contextes incident/emergency/cooperative/public_transport peuvent être absents,
null ou des objets partiels. Les valeurs absentes restent null, jamais False inventé.
Les flags doivent être de vrais booléens, pas des chaînes. Une gravité/type sans
incident confirmé, ou une priorité sans urgence confirmée, est rejetée.

Le mode courant est un replay historique : les mesures reçues doivent correspondre
au snapshot associé à la prévision sauvegardée. Un voisin externe doit être une
arête directe réelle du graphe, au même timestep, avec mesures concordantes. Une
liste partielle ne supprime pas les autres voisins du graphe. Les données externes
de trafic incompatibles déclenchent une erreur contrôlée, pas une prévision inventée.

La réponse contient le contrat demandé et un objet metadata : timestep, priorité,
trace LLM/fallback, avertissements et marqueur de fixture. En cas d'erreur, forecast,
traffic_state et risk_level sont null, action=manual_review et confidence=0.0 :
ils signalent l'absence de décision valide, pas des mesures ou scores fabriqués.
Les réponses exploitables portent status=warning à cause des limites connues de
mapping et de contexte de feux. Le statut ok est réservé à une future extension
disposant d'informations suffisantes. Le log structuré limite les données affichées.

Les schémas sérialisables INPUT_SCHEMA et OUTPUT_SCHEMA sont dans le module ; la
validation Python couvre aussi les contraintes temporelles et spatiales que les
schémas JSON seuls ne prouvent pas. handle_message intercepte les erreurs et retourne
un contrat error ; les méthodes internes peuvent lever des erreurs explicites.

## Priorité expérimentale et garde-fous

PRIORITY_POLICY : emergency=100, incident=80, severe_congestion=60,
public_transport=40, normal=20. Cette politique est configurable et expérimentale,
pas une règle définitive du système. Les niveaux emergency et incident doivent
rester strictement supérieurs aux autres niveaux pour préserver les garde-fous.
Une urgence présente, ou un incident high/à gravité inconnue, bloque l'appel LLM
et impose manual_review/high. Un deuxième garde-fou vérifie la même priorité après
la couche LLM. Le traitement reste une demande d'arbitrage à l'orchestrateur :
aucune seconde de feu, extension/réduction de phase ni commande réelle n'est émise.
Les autres incidents confirmés et les demandes de priorité transport public sont
aussi renvoyés à une revue orchestrateur, selon leur priorité et le risque existant.
Un contexte externe reçu est attribué à son fournisseur, jamais présenté comme
une détection propre de cet agent. L'authenticité de ces messages relève du futur
orchestrateur ; les tests de contrat ne prouvent pas la fiabilité des autres agents.

## Réutilisation du LLM et des modèles

Aucun modèle n'est réentraîné. predict consulte les 455 090 prévisions existantes,
sans nouvelle inférence XGBoost. Les fonctions et garde-fous de 08 sont réutilisés.
La première intégration est volontairement stricte : risque, état, action et raison
LLM doivent correspondre au contrat déterministe enrichi ; la raison est un texte
de faits autorisé, pas un raisonnement libre sans contrôle.

Sans callable llm_backend injecté, l'agent relit les sorties réelles sauvegardées de 09
sur les cinq états correspondants. metadata.llm_mode indique explicitement un replay,
sans prétendre une nouvelle génération Qwen. Ces sorties restent refusées par les
garde-fous comme dans 09. Pour un état ou contexte sans réponse applicable, le fallback
est utilisé. Un backend local vivant peut être injecté sans changer le contrat ; il
doit recevoir l'état enrichi, fournir un JSON conforme et gérer son délai d'inférence.
Le notebook ne lance pas de serveur ni de nouveau téléchargement. Un backend simulé
valide teste le chemin source=llm, avec integration_test_fixture=true ; il n'est pas
compté comme une véritable décision Qwen ni une preuve de qualité LLM.

## Données, fixtures et vérification

Les mesures utilisées sont les états réels déjà sélectionnés dans 08. Les contextes
positifs incident/emergency/transport, messages volontairement corrompus et backends
simulés sont marqués integration_test_fixture=true. Ils ne sont jamais présentés
comme incidents PEMS08 réels. Le test de garde-fou après LLM est un test unitaire
complémentaire sans échange orchestrateur ; son compte est séparé des messages.

Les vérifications portent sur contrat JSON, conservation de la prévision, erreurs
contrôlées, priorité critique avant/après LLM, voisins réels, contexte manquant,
timeouts et refus de données incohérentes. Il ne s'agit ni d'une accuracy ni d'une
mesure d'efficacité du contrôle routier. La confidence reste non calibrée.
Le mapping location=index NPZ reste une hypothèse pour 25 capteurs aux séries
identiques documentés dans 05/06/08 ; les conseils spatiaux héritent de cette limite.

## Architecture

```text
Incident Detection Agent ──┐
Emergency Priority Agent ──┤
Public Transport Agent ────┤
Cooperative Traffic Agent ─┤
                          ↓
                     Orchestrator
                          ↓
          Traffic Prediction & Control Agent
                          ↓
              XGBoost forecast +15 (sauvegardé)
                          ↓
            LLM + garde-fous déterministes
                          ↓
               recommandation structurée
                          ↓
                     Orchestrator
```

L'interface est prête pour un branchement de test sur l'orchestrateur réel respectant
ce contrat historique. Elle n'est pas un contrôle de feux en production : un flux
temps réel hors de l'export requiert une autre stratégie de forecasting, une
authentification des fournisseurs, un contexte de feux et une politique validée
par l'équipe. Les notebooks 01–09, benchmarks et résultats antérieurs sont préservés.
"""
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = [md("""# GoFlow — Couche d'intégration multi-agent

Une classe réutilisable reçoit des messages orchestrateur, réutilise les prévisions
XGBoost existantes et retourne une recommandation JSON. Aucun modèle d'incident,
d'urgence, de routes ou de transport public n'est créé. Les priorités sont des
choix **expérimentaux configurables**, pas la politique finale du groupe.
"""), md("## 1. Charger l'agent et les contrats\n\n`timestep` est obligatoire pour réaligner le forecasting historique. `timestamp`\npeut rester null : aucune date de mesure PEMS08 n'est inventée.\nLes contextes manquants restent inconnus. Les schémas et validations sont affichés\npour permettre aux autres membres du groupe d'adapter leurs messages.\n\nPrérequis : installer ce dépôt dans l'environnement du noyau Jupyter, depuis sa\nracine, avec `python -m pip install --no-index --no-deps --no-build-isolation -e .`.\nCette installation locale ne télécharge rien. Les imports fonctionnent ensuite\négalement depuis `notebooks/`, sans modification de `sys.path`.\n"), code("# Importer la classe réutilisable sans exécuter les notebooks précédents.\nfrom pathlib import Path\nimport json, logging\nimport pandas as pd\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents]\n            if (p / 'notebooks/09_llm_integration.ipynb').exists())\nfrom src.agents.traffic_prediction_agent import TrafficPredictionControlAgent, PRIORITY_POLICY, INPUT_SCHEMA, OUTPUT_SCHEMA\nprint('Politique expérimentale :', PRIORITY_POLICY)\nprint(json.dumps(INPUT_SCHEMA, ensure_ascii=False, indent=2))\nprint(json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2))\n"), md("""## 2. Journalisation structurée et état initial

Les logs enregistrent request_id, sensor_id, forecast, source, action, validation,
fallback et erreur, sans imprimer tous les états. Un backend LLM peut être injecté.
Par défaut, les sorties Qwen sauvegardées de 09 sont rejouées quand elles s'appliquent :
ce notebook ne prétend pas effectuer de nouvelles générations.
"""), code("""# Initialiser une fois les accès aux prévisions, séries et voisins réels.
folder = root / 'results/multi_agent_integration'
folder.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('goflow.integration_test')
logger.setLevel(logging.INFO)
logger.propagate = False
for handler in list(logger.handlers):
    logger.removeHandler(handler)
    handler.close()
handler = logging.FileHandler(folder / 'integration_events.jsonl', mode='w', encoding='utf-8')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
agent = TrafficPredictionControlAgent(root, logger=logger)
assert agent._init_error is None, agent._init_error
print('Agent chargé : prévisions existantes, aucun fit et aucune nouvelle inférence.')
"""), md("""## 3. Tests avec MockOrchestrator

Les sept situations demandées sont couvertes, ainsi que conflits de priorité,
données incohérentes, timeout et garde-fous avant/après LLM. Les incidents, urgences,
priorités transport et backends artificiels sont explicitement marqués
`integration_test_fixture=true`. Les mesures de trafic restent issues des états réels.
Le test direct du garde-fou après LLM est distingué des échanges orchestrateur.
"""), code("# Exécuter les tests de contrat ; MockOrchestrator reste dans le module de tests.\nfrom tests.multi_agent_integration_cases import run_integration_cases\norchestrator, test_results = run_integration_cases(agent, root)\nvalidation = pd.DataFrame(test_results)\ndisplay(validation)\nassert validation.passed.all()\nfor message, response in zip(orchestrator.messages, orchestrator.responses):\n    print(message.get('request_id'), '→', response['status'], response['recommended_action'],\n          response['decision_source'])\n"), md("""## 4. Examiner les garde-fous critiques

Une urgence externe ou un incident critique impose une revue manuelle, sans appel
LLM. Le contrôle après LLM réimpose la même décision. Un événement indiqué dans
une fixture reste une simulation d'interface, pas une observation PEMS08.
"""), code("""# Afficher un échange normal et un échange d'urgence simulée.
for request_id in ['trafic_normal', 'urgence_externe']:
    index = next(i for i, m in enumerate(orchestrator.messages) if m.get('request_id') == request_id)
    print(json.dumps({'message': orchestrator.messages[index], 'response': orchestrator.responses[index]},
                     ensure_ascii=False, indent=2, allow_nan=False))
"""), md("""## 5. Sauvegarde et résumé

Sauvegarder uniquement sous `results/multi_agent_integration/`. Une erreur contrôlée
attendue fait réussir un test négatif ; elle ne signifie pas que l'agent a fourni
une prévision pour un message invalide. Le chemin LLM accepté simulé est un test
de contrat, pas une nouvelle décision du modèle Qwen.
"""), code("methodology = " + repr(methodology) + "\n" + """# Écrire les résultats d'intégration sans modifier aucun benchmark existant.
for filename, values in [('integration_test_results.json', test_results),
                         ('orchestrator_messages.json', orchestrator.messages),
                         ('agent_responses.json', orchestrator.responses)]:
    (folder / filename).write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
validation.to_csv(folder / 'integration_validation.csv', index=False)
(folder / 'integration_methodology.md').write_text(methodology, encoding='utf-8')
handler.flush()
logger.removeHandler(handler)
handler.close()
summary = {
    'Agent intégré': 'oui' if validation.passed.all() else 'non',
    'Nombre de tests': len(validation), 'Tests réussis': int(validation.passed.sum()),
    'Tests échoués': int((~validation.passed).sum()),
    'Nombre de messages': len(orchestrator.messages),
    'Messages valides (schéma)': int(validation.input_schema_valid.fillna(False).sum()),
    'Messages exploitables (snapshot cohérent)': sum(r['status'] != 'error' for r in orchestrator.responses),
    'Réponses JSON valides': sum(agent.validate_response(r) for r in orchestrator.responses),
    'Fallbacks (réponses)': sum(r['decision_source'] == 'rule_based_fallback' for r in orchestrator.responses),
    'Erreurs contrôlées': sum(r['status'] == 'error' for r in orchestrator.responses),
    'Acceptations LLM simulées pour le contrat': sum(r['decision_source'] == 'llm' and r['metadata']['integration_test_fixture'] for r in orchestrator.responses)
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
"""), md("""## 6. Architecture et périmètre de branchement

```text
Incident Detection Agent ──┐
Emergency Priority Agent ──┤
Public Transport Agent ────┤
Cooperative Traffic Agent ─┤
                          ↓
                     Orchestrator
                          ↓
          Traffic Prediction & Control Agent
                          ↓
              XGBoost forecast +15 existant
                          ↓
            LLM + garde-fous déterministes
                          ↓
               recommandation structurée
                          ↓
                     Orchestrator
```

L'interface est prête pour un branchement de test avec le contrat décrit. Le mode
actuel couvre les snapshots historiques exportés, pas un forecasting sur un flux
temps réel inédit ni une commande de feux. Le mapping location=index NPZ reste
hypothétique pour 25 capteurs aux séries identiques. La priorité doit être validée
par l'équipe avant un usage opérationnel ; confidence n'est pas calibrée.
""")]
path = root / "notebooks/10_multi_agent_integration.ipynb"
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}})
nbformat.write(notebook, path)
try:
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
finally:
    nbformat.write(notebook, path)
nbformat.validate(notebook)
assert all(digest(p) == value for p, value in before.items()), "Fichier existant modifié"
print("Notebook 10 exécuté ; notebooks 01–09, sources et résultats précédents inchangés.")

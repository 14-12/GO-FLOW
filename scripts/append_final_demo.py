"""Ajouter la démo et exécuter uniquement ses nouvelles cellules, sans réécrire les anciens résultats."""
from pathlib import Path
import copy
import hashlib
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/10_multi_agent_integration.ipynb"
notebook = nbformat.read(path, as_version=4)
if any("# DEMO FINALE — GOFLOW AGENT" in cell.source for cell in notebook.cells):
    raise RuntimeError("La section DEMO FINALE existe déjà ; aucun ajout dupliqué.")
original_cells = copy.deepcopy(notebook.cells)
protected = [p for base in [root / "data", root / "results"] for p in base.rglob("*") if p.is_file()]
protected += [p for p in (root / "notebooks").glob("*.ipynb") if p != path]
def digest(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
before = {p: digest(p) for p in protected}
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
added = [md("""# DEMO FINALE — GOFLOW AGENT

**Un seul cas réel, un seul échange orchestrateur et un seul appel LLM si disponible.**
Les prévisions XGBoost sont déjà sauvegardées. Cette section ne réentraîne rien et
n'écrit aucun résultat. Les 24 tests et leurs sorties précédentes sont conservés.
"""), md("""## 1. Observation réelle

Choisir automatiquement la plus forte hausse d'occupation prévue parmi les lignes
test exportées. Le choix utilise la prévision et l'état courant, jamais la cible
réellement observée à t+3. Aucun incident ni véhicule d'urgence n'est inventé.
"""), code("# Charger l'interface et sélectionner un vrai cas, indépendamment des cellules de tests.\nfrom pathlib import Path\nimport json, copy\nimport pandas as pd\nfrom IPython.display import display, Markdown\ndemo_root = next(p for p in [Path.cwd(), *Path.cwd().parents]\n                 if (p / 'notebooks/09_llm_integration.ipynb').exists())\nfrom src.agents import TrafficPredictionControlAgent\nfrom src.backends.llama_cpp import LlamaCppBackend\ndemo_backend = LlamaCppBackend(demo_root)\ndemo_agent = TrafficPredictionControlAgent(demo_root, llm_backend=demo_backend)\nassert demo_agent._init_error is None, demo_agent._init_error\ndemo_rows = demo_agent.rules.PREDICTIONS\ndemo_changes = demo_rows.predicted_occupancy_t15 - demo_rows.current_occupancy\ndemo_index = demo_changes.idxmax()\nassert demo_changes.loc[demo_index] > 0, 'Aucune hausse réelle disponible.'\ndemo_row = demo_rows.loc[demo_index]\ndemo_sensor, demo_timestep = int(demo_row.location), int(demo_row.timestep)\ndemo_base_state = demo_agent.rules.build_traffic_state(demo_sensor, demo_timestep)\ndisplay(pd.DataFrame([{key: demo_base_state[key] for key in\n    ['sensor_id', 'timestep', 'current_flow', 'current_speed', 'current_occupancy',\n     'predicted_occupancy_t15', 'occupancy_change']}]))\n"), md("""## 2. Prévision +15 min

**Prévision XGBoost à +15 minutes** — lecture d'une prévision existante, sans nouvelle
inférence XGBoost. Les valeurs sont dans l'échelle originale de l'occupation.
"""), code("""# Montrer la variation prévue sans modifier ou écrêter la prévision.
demo_forecast = demo_agent.predict(demo_sensor, demo_timestep)
assert demo_forecast == demo_base_state['predicted_occupancy_t15']
demo_delta = demo_forecast - demo_base_state['current_occupancy']
demo_direction = 'hausse prévue' if demo_delta > 0 else 'baisse prévue' if demo_delta < 0 else 'stable'
display(pd.DataFrame([{'current_occupancy': demo_base_state['current_occupancy'],
                      'predicted_occupancy_t15': demo_forecast, 'différence': demo_delta}]))
print('Interprétation :', demo_direction)
"""), md("""## 3. Contexte temporel et spatial

La tendance est la variation moyenne entre t−6 et t−1, par intervalle de 5 minutes :
positive signifie une hausse passée, négative une baisse passée. Les voisins sont
les connexions directes du graphe, sous l'hypothèse de mapping documentée pour
25 capteurs indiscernables. `timestamp=null` évite d'inventer une date PEMS08 ; le
`timestep` identifie le snapshot historique.
"""), code("# Afficher les tendances causales et tous les voisins directs du cas sélectionné.\ndisplay(pd.DataFrame([{key: demo_base_state[key] for key in\n                       ['recent_occupancy_trend', 'recent_speed_trend']}]))\nfor field, label in [('recent_occupancy_trend', 'occupation'), ('recent_speed_trend', 'vitesse')]:\n    value = demo_base_state[field]\n    direction = 'augmentait' if value > 0 else 'diminuait' if value < 0 else 'restait stable'\n    print(f'Sur les six observations précédentes, {label} {direction} en moyenne.')\ndisplay(pd.DataFrame(demo_base_state['neighbors']))\ndemo_message = {\n    'request_id': 'demo_final_001', 'timestamp': None, 'sensor_id': demo_sensor,\n    'timestep': demo_timestep,\n    'traffic_data': {key: demo_base_state[key] for key in\n                     ['current_flow', 'current_speed', 'current_occupancy']},\n    'incident_context': {'incident_detected': None, 'incident_type': None, 'severity': None},\n    'emergency_context': {'emergency_vehicle_present': None, 'priority_level': None},\n    'cooperative_context': {'neighboring_sensors': copy.deepcopy(demo_base_state['neighbors'])},\n    'public_transport_context': {'priority_requested': None},\n    'integration_test_fixture': False}\ndemo_state = demo_agent.build_state(demo_message)\nprint('Message envoyé à l’agent :')\nprint(json.dumps(demo_message, ensure_ascii=False, indent=2, allow_nan=False))\n"), md("""## 4. Analyse LLM

Appeler le vrai Qwen2.5-0.5B-Instruct GGUF local via llama.cpp, avec le prompt et
le schéma de 09. Un serveur temporaire lié à localhost est arrêté après l'appel.
Son log reste temporaire : aucun fichier de résultats antérieur n'est écrit.
La réponse brute, même mauvaise, est affichée séparément de la décision finale.
"""), code("# Le backend gère le serveur temporaire ; l'agent expose l'API de décision.\nwith demo_backend:\n    demo_response = demo_agent.handle_message(demo_message)\n    demo_available = demo_backend.available\n\ndemo_call = demo_backend.last_call\nprint('Sortie LLM brute :')\nprint(demo_call['raw_text'] if demo_call['raw_text'] is not None else 'Aucune sortie : ' + str(demo_call['error']))\nprint('Temps d’inférence (s) :', demo_call['inference_time_s'])\n\ndemo_output = demo_call['parsed_output']\ndemo_json_valid = demo_agent.validate(demo_output)\nprint('JSON valide :', 'oui' if demo_json_valid else 'non')\nif demo_call['error']:\n    print('Erreur LLM conservée :', demo_call['error'])\n"), md("""## 5. Validation et garde-fous

Un JSON valide ne suffit pas : risque, action et justification doivent aussi
respecter les valeurs et le contrat enrichi. Les contrôles lexicaux d'incident et
d'urgence sont partiels ; une absence de terme suspect ne prouve pas une justesse
sémantique complète. Le grounding de l'agent impose ici sa justification de faits.
"""), code("# Montrer chaque contrôle, sans remplacer la réponse produite par le modèle.\n# Réutiliser les contrôles du backend et de l'agent, sans redéfinir la validation.\ndemo_checks = demo_backend.validation_report(demo_agent, demo_state)\n\ndisplay(pd.DataFrame([{'Garde-fou': label, 'Résultat': 'OK' if passed else 'FAIL'}\n                      for label, passed in demo_checks.items()]))\ndemo_fallback = demo_response['decision_source'] == 'rule_based_fallback'\nif demo_fallback:\n    print('LLM rejeté par les garde-fous' if demo_available else 'LLM indisponible — fallback déterministe')\n    print('Raison :', demo_response['metadata'].get('fallback_reason'))\n    if demo_response['metadata'].get('llm_error'):\n        print('Erreur :', demo_response['metadata']['llm_error'])\n    print('Décision rule_based utilisée :')\n    print(json.dumps(demo_agent._rule_decision(demo_state), ensure_ascii=False, indent=2, allow_nan=False))\nelse:\n    print('LLM accepté : aucun fallback nécessaire.')\n"), md("""## 6. Décision finale

### DÉCISION FINALE

La confidence n'est pas une probabilité calibrée. Cette recommandation reste une
sortie vers l'orchestrateur, sans commande de feux exécutée.
"""), code("""# Afficher seulement la décision finale, distincte de la réponse LLM brute.
demo_final = {key: demo_response[key] for key in
              ['traffic_state', 'risk_level', 'recommended_action', 'reason', 'confidence', 'decision_source']}
print(json.dumps(demo_final, ensure_ascii=False, indent=2, allow_nan=False))
"""), md("""## 7. Réponse à l’orchestrateur

Voici la réponse complète réellement produite par `handle_message`, avec
prévision inchangée, corrélation request_id et trace de décision.
"""), code('# La réponse complète est déjà validée par handle_message.\ndisplay(demo_response)\n'), md("""### Résumé de la démonstration

Le modèle XGBoost prédit l'état futur du trafic. Le LLM propose une interprétation.
Les garde-fous vérifient la cohérence. Si nécessaire, le système utilise un fallback
déterministe. La décision finale est renvoyée à l'orchestrateur sous forme JSON structurée.
"""), code("""# Rapporter les résultats réellement observés de cette démonstration.
print('DEMO RESULT')
print('Sensor :', demo_sensor)
print('Timestep :', demo_timestep)
print('Forecast +15 :', demo_forecast)
print('LLM valid (JSON) :', 'oui' if demo_json_valid else 'non')
print('LLM accepté par les garde-fous :', 'oui' if not demo_fallback else 'non')
print('Fallback used :', 'oui' if demo_fallback else 'non')
print('Final action :', demo_response['recommended_action'])
print('Response status :', demo_response['status'])
""")]
notebook.cells.extend(added)
nbformat.write(notebook, path)
# Exécuter uniquement le nouveau fragment dans un noyau frais ; ne pas relancer les tests ni leurs sauvegardes.
fragment = nbformat.v4.new_notebook(cells=copy.deepcopy(added), metadata=copy.deepcopy(notebook.metadata))
try:
    NotebookClient(fragment, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
finally:
    notebook.cells = original_cells + fragment.cells
    nbformat.write(notebook, path)
nbformat.validate(notebook)
assert notebook.cells[:len(original_cells)] == original_cells, "Cellules précédentes modifiées"
assert all(digest(p) == checksum for p, checksum in before.items()), "Ancien fichier modifié"
print("Démo exécutée ; anciennes cellules et tous les résultats existants inchangés.")

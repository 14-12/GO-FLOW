"""Créer le notebook 09, sans modifier les notebooks 01 à 08."""
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[1]
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
methodology = """# GoFlow — Intégration d'un LLM instruct local

XGBoost fournit les prévisions d'occupation à +15 minutes. Le LLM ne fait pas de
forecasting : il reçoit les cinq états déjà sauvegardés par le notebook 08, avec
mesures, tendances passées et voisins, puis propose une catégorie de risque, une
action et une justification. Les règles constituent le fallback. Aucun scénario
n'est recréé, aucune cible future n'est passée au modèle, aucun modèle trafic
n'est réentraîné. Les fichiers de results/decision_agent/ sont lus, pas modifiés.

## Modèle et environnement

Modèle : Qwen/Qwen2.5-0.5B-Instruct, environ 0,49 milliard de paramètres, issu du
[dépôt du fournisseur](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
La révision téléchargée et les SHA256 de chaque fichier sont enregistrés sous
models/qwen2.5-0.5b-instruct/download_provenance.json. Les informations effectives
de chargement, nombre exact de paramètres, mémoire RSS et versions sont enregistrées
dans llm_model_info.json. Ce modèle est préentraîné/instruct, sans adaptation à PEMS08.

L'environnement .venv-llm utilise --system-site-packages pour lire les dépendances
existantes sans les modifier. Aucune installation de package n'est nécessaire.
Le CPU est utilisé, avec quatre threads ; CUDA n'est pas disponible sur cette
installation PyTorch. Le backend retenu est le binaire CPU Windows officiel
[llama.cpp b11596](https://github.com/ggml-org/llama.cpp/releases/tag/b11596), avec le
[GGUF Qwen officiel Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
Provenance et SHA256 du GGUF et du binaire sont enregistrés sous
models/qwen2.5-0.5b-instruct-gguf/download_provenance.json. Le serveur est lié à
127.0.0.1 sur un port libre et arrêté en fin de notebook. Sa RSS est mesurée
séparément de celle du noyau Python. Le chargement exige 1 Go libre.

Le premier essai Transformers/INT8 a généré cinq réponses, avec zéro JSON valide,
cinq fallbacks et trois délais dépassés. Tous ses résultats, textes, erreurs et
temps sont conservés dans diagnostics/transformers_int8_first_attempt. Le changement
de backend répond à cet échec technique, sans consulter les cibles futures et
sans modifier les scénarios. Aucun échec initial n'est effacé.

Un premier essai GGUF utilisait une enveloppe response_format incomplète : le
serveur appliquait un JSON générique, pas le schéma des cinq champs. Ces résultats
sont conservés dans diagnostics/llama_cpp_schema_first_attempt. L'enveloppe a été
corrigée d'après le code de la version b11596 : response_format.json_schema.schema.
La longueur de reason est bornée à 220 caractères pour conserver une réponse courte.

## Génération et validation

Chat template embarqué dans le GGUF officiel, prompt système français, température
zéro, maximum 300 nouveaux tokens et génération contrainte par schéma JSON dans
llama.cpp. Le schéma impose les champs et enums, pas les décisions ni une raison
préécrite. Le texte utile est récupéré dans message.content.
Le prompt et les paramètres sont fixés avant de parcourir les cinq scénarios ;
aucun réglage par comparaison à la cible future n'est réalisé. Le modèle n'est
pas entraîné pendant cette étape. Un timeout HTTP de 60 secondes et un contrôle
de durée après réponse déclenchent un fallback ; une opération serveur en cours
n'est pas garantie d'être interrompue instantanément. Le log serveur est conservé.
Le texte brut, le JSON extrait, les erreurs, motifs de fallback,
tokens et durées restent disponibles pour chacun des cinq appels réels.

L'extraction accepte du bruit autour d'un unique objet JSON mais ne répare aucune
valeur, clé ou ponctuation et refuse les clés dupliquées, NaN et blocs ambigus.
Le validateur strict du notebook 08 est réutilisé ; confidence doit être un float.
Les catégories de risque et de trafic doivent être cohérentes. Le niveau de risque
des règles est un plancher expérimental, pas une vérité scientifique. Une action
de coordination nécessite au moins un voisin à occupation élevée selon les mêmes
seuils train de 08. Les phases de feux restent interdites faute de contexte de commande.

Le LLM peut produire une justification libre, mais elle doit citer les deux valeurs
d'occupation fournies à six décimales et la phrase : « Incidents inconnus; urgences
inconnues; plan de feux inconnu. » Les autres affirmations incident/urgence/voisin
sont refusées de manière lexicale conservatrice ; les nombres de la justification
doivent correspondre aux occupations fournies ou à l'horizon 15. Ces tests peuvent
rejeter des formulations pourtant légitimes et ne garantissent pas l'absence de
toute hallucination sémantique. Le contrôle d'hallucination porte seulement sur
les raisons JSON analysables ; les réponses non parsées exigent une revue manuelle.
« Reason grounded » signifie conformité à ces
contrôles, pas une annotation humaine démontrant une causalité correcte.

La comparaison rules/LLM est qualitative. L'accord d'action n'est pas une accuracy,
et aucun résultat ne prouve que le LLM est meilleur. Sa confidence est une
auto-évaluation non calibrée, pas une probabilité statistique. Les cinq scénarios
sont les cas illustratifs extrêmes déjà retenus dans 08, pas un échantillon représentatif.
Le mapping location=index NPZ reste hypothétique pour les 25 capteurs aux séries
identiques documentés dans 08 ; les recommandations de voisinage héritent de cette limite.

## Robustesse et architecture

Les tests de JSON invalide, clé manquante, action interdite, confidence >1, faux
incident, modèle indisponible et timeout utilisent des fixtures explicitement
simulées. Ils ne sont pas enregistrés comme sorties du modèle réel.

```text
PEMS08 → Feature Engineering → XGBoost → Prediction +15 min
  → Traffic State Builder (état courant, tendances passées, voisins directs)
  → LLM Decision Agent → JSON et garde-fous → Fallback si nécessaire
  → Recommandation structurée → Orchestrator futur
```

Aucune commande n'est envoyée à des feux ou à un système externe. Les notebooks
01–08, le benchmark et les données sources restent inchangés. L'intégration technique
est réussie si le modèle se charge et génère du texte pour les cinq appels ; le
nombre de décisions acceptées est rapporté séparément, sans masquer les fallbacks.
"""
cells = [md("""# GoFlow — Intégration réelle d'un petit LLM instruct

Réutiliser **exactement les cinq états sauvegardés dans 08**. XGBoost reste le
forecaster ; le LLM interprète les valeurs et propose une recommandation.
Les règles et la validation demeurent les garde-fous. Aucun notebook 01–08 modifié.
"""), md("""## 1. Environnement isolé et réutilisation des résultats

Ce notebook utilise le noyau `.venv-llm`, créé avec les packages système en lecture
seule. Aucune installation dans l'environnement principal. Le module réutilise
les fonctions du notebook 08 via `scripts/decision_agent.py`, les seuils train
déjà enregistrés et les cinq états JSON. Il contrôle aussi les empreintes des sources.
"""), code("""# Charger le module sans réexécuter les notebooks ni refaire du forecasting.
from pathlib import Path
import sys, json, copy
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'notebooks/08_llm_decision_agent.ipynb').exists())
sys.path.insert(0, str(root / 'scripts'))
import local_llm_agent as agent
display(agent.setup_saved_inputs())
display(agent.SCENARIOS[['scenario', 'location', 'timestep']])
"""), md("""## 2. Modèle CPU léger et quantification

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) annonce
environ 0,49 milliard de paramètres. Les poids officiels sont téléchargés avant
l'exécution, avec une révision figée et des empreintes enregistrées. Le notebook
charge exclusivement les fichiers locaux. La quantification GGUF Q4_K_M et le
backend llama.cpp CPU sont utilisés. L'essai initial INT8 est archivé comme diagnostic.
Les versions, le nombre exact de paramètres et la RSS sont affichés ci-dessous.
"""), code("""# Charger le vrai modèle ; conserver une erreur explicite et le fallback si nécessaire.
display(agent.load_model())
print(agent.SYSTEM_PROMPT)
"""), md("""## 3. Appel réel et contrat de sortie

`agent.llm_decision(state)` sérialise l'état, applique le chat template, appelle
le serveur llama.cpp local avec schéma JSON, extrait un seul JSON et
retourne un dict. Le texte brut est conservé ; aucune correction silencieuse.

`agent.final_decision(state)` tente le LLM, valide les cinq clés, catégories,
confidence, justification et cohérence ; sinon elle retourne les règles avec
`source=rule_based_fallback` et un motif explicite. Les méthodes sont visibles
ci-dessous pour rendre la logique reviewable sans la dupliquer.
"""), code("""# Afficher les fonctions effectives utilisées dans cette exécution.
import inspect
for function in [agent.llm_decision, agent.extract_json, agent.validate_decision,
                 agent.qualitative_checks, agent.final_decision]:
    print(inspect.getsource(function))
"""), md("""## 4. Les cinq mêmes scénarios — un appel réel par état

Comparer les sorties brutes du LLM, les règles et les décisions finales. Aucun
nouveau scénario ni cible réelle à t+3 n'est utilisé. Le temps mesuré inclut
tokenisation, génération et parsing, mais pas le chargement du modèle.
"""), code("""# Effectuer exactement cinq appels réels et conserver leurs erreurs éventuelles.
records, comparison, validation, times = agent.run_scenarios()
for record in records:
    print('\\nSCÉNARIO :', record['scenario'])
    print(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
display(comparison)
display(validation)
display(times)
"""), md("""## 5. Tests de robustesse avec fixtures simulées

Les tests ci-dessous utilisent un état réel mais remplacent l'appel LLM par des
sorties erronées ou exceptions contrôlées. Ils ne constituent pas des sorties du
modèle et ne sont pas inclus dans la comparaison des cinq inférences réelles.
"""), code("""# Vérifier le fallback pour chaque défaillance demandée, sans appel supplémentaire au modèle.
state = agent.INPUTS[0]['state']
baseline = agent.rules.rule_based_decision(state)
original = agent.llm_decision
fixtures = [
    ('json_invalide', '{'),
    ('cle_manquante', {k: v for k, v in baseline.items() if k != 'risk_level'}),
    ('action_interdite', {**baseline, 'recommended_action': 'invented_action'}),
    ('type_invalide', {**baseline, 'risk_level': []}),
    ('confidence_hors_limites', {**baseline, 'confidence': 1.5}),
    ('faux_incident', {**baseline, 'reason': 'Accident détecté.'})]
try:
    for name, fixture in fixtures:
        agent.llm_decision = lambda input_state, answer=fixture: answer
        result = agent.final_decision(state)
        assert result['source'] == 'rule_based_fallback', name
        assert result['decision'] == baseline, name
        assert result['fallback_reason'], name
    for error in [RuntimeError('Modèle indisponible simulé'),
                  TimeoutError('Timeout simulé'), OSError('Erreur d’inférence simulée')]:
        def fail(input_state, exception=error):
            raise exception
        agent.llm_decision = fail
        result = agent.final_decision(state)
        assert result['source'] == 'rule_based_fallback'
        assert type(error).__name__ in result['fallback_reason']
finally:
    agent.llm_decision = original
assert not agent.validate_decision({**baseline, 'confidence': True})
assert not agent.validate_decision({**baseline, 'confidence': float('nan')})
assert agent.extract_json('Texte avant ```json ' + json.dumps(baseline) + ' ```') == baseline
try:
    agent.extract_json('{"confidence": 0.5, "confidence": 0.8}')
except ValueError:
    pass
else:
    raise AssertionError('Clés dupliquées non refusées')
print('Tests JSON, contrat, faux incident, indisponibilité et timeout : OK.')
"""), md("""## 6. Sauvegarde et résumé réel

Sauvegarder séparément sous `results/llm_agent/`. Ni les résultats de 08 ni le
benchmark ne sont modifiés. JSON valide, grounding et nombre de fallbacks sont
des contrôles qualitatifs ; ils ne constituent pas une accuracy.
"""), code("methodology = " + repr(methodology)
            + "\nsummary = agent.save_results(records, comparison, validation, times, methodology)\n"),
md("""## 7. Limites et architecture

```text
PEMS08 → Feature Engineering → XGBoost → Prévision +15 min
  → Traffic State Builder
      ├── état courant
      ├── tendances passées
      └── voisins directs
  → LLM Decision Agent → JSON validation → Fallback si nécessaire
  → Recommandation structurée → Orchestrator futur
```

La confidence du LLM est une auto-évaluation **non calibrée**. Les cinq cas ne
permettent pas d'affirmer que le LLM est meilleur. Les contrôles anti-hallucination
sont conservateurs et partiels : ils ne prouvent pas une justesse sémantique totale.
Le mapping reste hypothétique pour les 25 capteurs indiscernables documentés dans 08.
PEMS08 ne fournit pas de plans de feux, d'incidents ni d'urgences : aucune commande
de feux n'est envoyée, aucune durée de phase n'est recommandée.
"""), code("""# Séparer le succès technique de la conformité des décisions.
if summary['Intégration technique']:
    print('LLM techniquement intégré : génération réelle effectuée sur les cinq scénarios.')
else:
    print('Intégration technique incomplète : consulter load_error et les erreurs d’inférence sauvegardées.')
print('Décisions LLM retenues :', int((validation.final_source == 'llm').sum()))
print('Fallbacks documentés :', summary['Fallbacks'])
""")]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "goflow-llm", "display_name": "GoFlow LLM CPU", "language": "python"},
    "language_info": {"name": "python"}})
nbformat.write(notebook, root / "notebooks/09_llm_integration.ipynb")
# Installer uniquement un kernelspec local au projet, sans changer les noyaux principaux.
folder = root / ".venv-llm/share/jupyter/kernels/goflow-llm"
folder.mkdir(parents=True, exist_ok=True)
import json
(folder / "kernel.json").write_text(json.dumps({
    "argv": [str(root / ".venv-llm/Scripts/python.exe"), "-X", "utf8", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "GoFlow LLM CPU", "language": "python",
    "env": {"IPYTHONDIR": str(root / ".venv-llm/.ipython")}}, indent=2), encoding="utf-8")
print("Notebook 09 et noyau local créés.")

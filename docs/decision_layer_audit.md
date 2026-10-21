# Audit et séparation de la décision GoFlow

## Localisation des règles avant et après

| Élément | Avant cette extraction | Définition après extraction |
|---|---|---|
| Indicateurs, `rule_based_decision` | `src/decision/decision_engine.py` | `src/decision/policy.py` ; adaptateurs historiques dans le moteur et `src/agents/decision_agent.py` |
| Priorité, raison externe, décision contextualisée | `decision_engine.py` | `policy.py` ; mêmes callbacks et adaptateurs |
| `validate_decision`, format et valeurs autorisées | Adaptateurs `guardrails.py` vers `src/contracts/validation.py` | Inchangé |
| Cohérence risque/action, plancher de risque | `guardrails.qualitative_checks` avec tables dans `policy.py` | Inchangé |
| Hallucinations incident/urgence et justification | Contrôles lexicaux et numériques dans `guardrails.py`, grounding exact dans `is_grounded_decision` et `validate_contextual_decision` | Inchangé |
| Actions, états, risques autorisés | `src/contracts/enums.py`, réexportés par `policy.py` | Inchangé, aucune seconde définition |
| Rangs, quantiles, seuils nommés | `policy.py` ; valeurs train résolues par le runtime | Inchangé |
| Fallback de 08, 09 et de l'agent | `final_decision`, `final_saved_scenario_decision`, `final_agent_decision` dans le moteur | Inchangé |
| Protection critique avant/après LLM | Orchestration dans le moteur, contrôles dans `guardrails.py` | Inchangé |

L'agent `traffic_prediction_agent.py` déléguait déjà au moteur et aux garde-fous.
Ses appels restent compatibles et atteignent maintenant la politique via les
adaptateurs du moteur. Aucun changement supplémentaire n'est nécessaire dans l'agent.

Le notebook 08 importe `create_runtime` et ne redéfinit pas les règles ; le
notebook 09 utilise l'adaptateur `local_llm_agent`, dont les validations et le
fallback délèguent déjà à ces modules. Le notebook 10 importe l'agent, ses
contrats et le backend. Aucun de ces notebooks n'a besoin d'être modifié.

## Conditions réelles et seuils

Les valeurs ci-dessous proviennent du témoin train existant
`tests/fixtures/traffic_policy_before.json`. Elles restent chargées/calculées par
les chemins existants ; elles ne sont pas figées à nouveau dans le code.
`pic = max(current_occupancy, predicted_occupancy_t15)`.

| Règle | Condition réelle | Seuil réel | Action |
|---|---|---|---|
| Severe, évaluée en premier | pic >= Q97 occupation | 0.1604 | Risque high ; coordination si voisin congestionné, sinon manual_review |
| Congestion | pic >= Q90 occupation OU hausse >= Q90 variation absolue, hors severe | 0.1071 ; 0.015000000000000006 | Même choix d'action high |
| Occupation modérée | pic > médiane, hors règles précédentes | 0.0599 | monitor, risque medium |
| Vitesse faible | vitesse <= Q25, hors règles précédentes | 62.6 | monitor, risque medium |
| Voisin congestionné | occupation voisine >= Q90 ; nombre > 0 | 0.1071 ; 0 | Au moins moderate ; coordination si risque high |
| Tendances défavorables | pente occupation > 0 ET pente vitesse < 0, hors règles précédentes | 0 ; historique t−6 à t−1 | monitor, risque medium |
| Free flow | Aucune condition précédente | Aucun seuil supplémentaire | keep_current_signal_plan, risque low |
| Amélioration, indicateur seulement | variation <= −Q90 variation absolue | −0.015000000000000006 | Aucun changement direct d'action |
| Emergency | emergency_vehicle_present est True | Rang 100 | manual_review, high ; priorité critique |
| Incident | incident_detected est True | Rang 80 | manual_review, high ; critique si gravité high ou inconnue |
| Severe congestion | état déterministe severe | Rang 60 | Action high existante ; pas de verrou critique à elle seule |
| Transport public | priority_requested est True | Rang 40 | manual_review ; risque au moins medium |
| Normal | Priorité de base | Rang 20 | Action déterministe courante |

La priorité active de rang maximal est retenue : emergency > incident > severe
congestion > public transport > normal avec les rangs par défaut. Ainsi une
congestion severe peut précéder une demande transport public. Les urgences
confirmées et incidents critiques contournent le LLM ; les autres incidents et
la priorité transport public retenue le contournent également.

La confiance déterministe reste 0.5, celle d'erreur 0.0. Les justifications gardent
six décimales et leur tolérance numérique existante de 0.000001. Les contrôles
qualitatifs de 09 restent distincts du grounding exact de 08 et du contrat enrichi
de l'agent : aucune validation n'est durcie ou assouplie par ce déplacement.

## Vérification

`tests/test_decision_separation.py` compare 14 décisions et traces capturées avant
le déplacement et 34 cas de frontière issus du témoin historique. Les tests
existants vérifient toujours les contrats, les hallucinations, les injections de
callbacks et les 24 scénarios d'intégration. Aucun entraînement ni nouvelle
inférence Qwen n'est nécessaire.

```powershell
pytest tests -q
```

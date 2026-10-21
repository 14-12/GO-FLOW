# Contrat d'intégration de TrafficPredictionControlAgent

La source unique du contrat est `src/contracts/` : `schemas.py` contient les
schémas input/output, `enums.py` leurs valeurs autorisées et `validation.py` les
validateurs Python. Les anciens imports `INPUT_SCHEMA` et `OUTPUT_SCHEMA` depuis
`src.agents.traffic_prediction_agent` désignent ces mêmes objets.

Le point d'entrée d'intégration recommandé est
`src.contracts.traffic_prediction_contract` : il rassemble ces schémas, les
ensembles de valeurs autorisées, les champs obligatoires/optionnels dérivés des
schémas, les exemples et les validateurs. Il réexporte les définitions existantes
sans créer de second contrat ; les imports depuis `src.contracts` restent valides.

**Cet agent ne détecte pas lui-même les incidents ou les urgences.** Il utilise
les informations transmises par les autres agents via l'orchestrateur, avec les
observations de trafic, une prévision à +15 minutes et ses règles de décision.
Il retourne une recommandation ; il n'envoie aucune commande aux feux.

Il **ne prédit pas les bus** et **ne calcule pas les routes**. Il reçoit les
contextes de trafic spécialisés via l'orchestrateur, sans implémenter les modèles
des autres agents. La demande de priorité transport public est un contexte externe,
pas une prévision de bus. Aucun champ de route ou de prévision de bus n'est ajouté
au contrat actuel. Ce composant reste expérimental.

## Entrée : champs obligatoires

| Champ | Type | Signification |
|---|---|---|
| `request_id` | string non blanche, 1 à 200 caractères | Identifiant de corrélation ; recopié dans la réponse. |
| `sensor_id` | integer de 0 à 169 | Identifiant du capteur PEMS08 dans le mapping existant. Un booléen n'est pas un entier valide. |
| `timestep` | integer | Pas temporel du snapshot, obligatoire même si `timestamp` est fourni. |
| `traffic_data` | object | Exactement les trois mesures courantes ci-dessous. |

| Champ de `traffic_data` | Type | Signification |
|---|---|---|
| `current_flow` | number fini | Débit courant dans les unités source. |
| `current_speed` | number fini | Vitesse courante dans les unités source. |
| `current_occupancy` | number fini | Occupation courante dans l'échelle source ; aucune conversion automatique en pourcentage. |

`NaN`, `Infinity`, booléens ou chaînes à la place des mesures sont refusés.
Aucun champ supplémentaire n'est accepté dans le message, les contextes ou
`traffic_data`, hormis les champs explicitement définis dans le schéma.

## Champs optionnels

| Champ | Type | Signification / valeur absente |
|---|---|---|
| `timestamp` | string ISO 8601 avec fuseau, ou null | Horodatage de l'enveloppe ; ne remplace pas `timestep`. Aucune date PEMS08 n'est inventée. |
| `incident_context` | object partiel ou null | Information d'incident fournie de l'extérieur ; absente = inconnue. |
| `emergency_context` | object partiel ou null | Information de véhicule d'urgence fournie de l'extérieur ; absente = inconnue. |
| `public_transport_context` | object partiel ou null | Demande de priorité transport public ; absente = inconnue. |
| `cooperative_context` | object partiel ou null | Mesures des voisins proposées par le fournisseur de contexte spatial. |
| `integration_test_fixture` | boolean | Marque les contextes/backends simulés de test ; défaut false. |

Un contexte peut être omis, null ou un objet partiel, y compris `{}`. Une valeur
inconnue reste null ; elle n'est jamais transformée automatiquement en false.

### Fournisseurs des contextes

| Contexte | Fournisseur | Champs et sens |
|---|---|---|
| `incident_context` | **Incident Detection Agent**, relayé par l'orchestrateur | `incident_detected`: boolean/null ; `incident_type`: string non blanche/null ; `severity`: `low`, `medium`, `high` ou null. |
| `emergency_context` | **Emergency Priority Agent** (agent d'urgence), relayé par l'orchestrateur | `emergency_vehicle_present`: boolean/null ; `priority_level`: `low`, `medium`, `high` ou null. |
| `public_transport_context` | **Public Transport Agent**, relayé par l'orchestrateur | `priority_requested`: boolean/null, indiquant une demande de priorité. |
| `cooperative_context` | **Cooperative Traffic Agent**, relayé par l'orchestrateur | `neighboring_sensors`: array de voisins ou null. |

L'orchestrateur collecte ces informations et les transmet. L'agent GoFlow les
interprète comme des informations externes, sans les présenter comme ses propres
détections. Si l'incident n'est pas confirmé (`incident_detected` différent de
true), `incident_type` et `severity` doivent rester absents ou null. Si l'urgence
n'est pas confirmée, `priority_level` doit rester absent ou null.

Chaque élément de `neighboring_sensors` contient obligatoirement `sensor_id`
(integer 0–169), `timestep` (integer), `current_speed` et `current_occupancy`
(numbers finis). `predicted_occupancy_t15` (number fini) est optionnel. Aucun autre
champ n'est autorisé dans un voisin. Le contexte est vérifié contre le graphe
réel, au même timestep et avec des mesures concordantes. Les doublons et voisins
inconnus sont rejetés. Une liste partielle ne supprime pas les autres voisins du
graphe et ne remplace pas leurs observations.

## Modes de prévision

`prediction_mode` est un paramètre du constructeur, **pas un champ JSON entrant**.

- `historical_replay`, défaut temporaire : `timestep` entre **15177 et 17853**,
  mesures concordantes avec le snapshot historique et prévision déjà exportée.
- `live` : `timestep` entier **≥ 13**, sans borne supérieure du schéma. Un
  `history_provider` configuré dans l'application fournit au moins 13 observations
  contiguës et causales par capteur, y compris pour les voisins directs. Le modèle
  XGBoost sauvegardé calcule la prévision.

Le repère reste celui de PEMS08 : origine globale timestep=1, un pas de cinq
minutes. Ce contrat ne définit pas de conversion pour un autre réseau ou une
nouvelle horloge. Les historiques et le backend LLM sont injectés dans le
constructeur ; ils ne sont pas ajoutés au message JSON.

## Exemple input

Exemple réel déjà sauvegardé, livré dans `src/contracts/example_json/input.json`.
Il est compatible avec les deux schémas ; en replay, les observations restent
celles du snapshot indiqué.

```json
{
  "request_id": "trafic_normal",
  "timestamp": null,
  "sensor_id": 33,
  "timestep": 15227,
  "traffic_data": {
    "current_flow": 0.0,
    "current_speed": 65.0,
    "current_occupancy": 0.0
  },
  "incident_context": {
    "incident_detected": null,
    "incident_type": null,
    "severity": null
  },
  "emergency_context": {
    "emergency_vehicle_present": null,
    "priority_level": null
  },
  "public_transport_context": {
    "priority_requested": null
  },
  "cooperative_context": {
    "neighboring_sensors": null
  },
  "integration_test_fixture": false
}
```

## Sortie : tous les champs sont obligatoires

| Champ | Type | Signification |
|---|---|---|
| `agent` | string constante `traffic_prediction_control` | Identité de l'agent. |
| `request_id` | string ou null | Corrélation avec la requête ; null si aucun identifiant exploitable n'est fourni. |
| `sensor_id` | integer 0–169 ou null | Capteur de la requête ; null si invalide/inexploitable. |
| `forecast` | object | `horizon_minutes` vaut 15 ; `predicted_occupancy` est un number fini ou null en erreur. |
| `traffic_state` | `free_flow`, `moderate`, `congested`, `severe` ou null | État interprété par les règles ; null en erreur. |
| `risk_level` | `low`, `medium`, `high` ou null | Risque interprété ; null en erreur. |
| `recommended_action` | string de l'enum `Action` | Recommandation structurée. |
| `reason` | string non vide | Justification factuelle, ou explication d'erreur. |
| `confidence` | number fini de 0 à 1 | Indicateur non calibré ; règle déterministe actuelle à 0.5, erreur à 0.0. |
| `decision_source` | `llm` ou `rule_based_fallback` | Source de la décision finale ; la seconde valeur ne décrit pas le mode de forecasting. |
| `status` | `ok`, `warning` ou `error` | Statut de traitement décrit ci-dessous. |
| `metadata` | object | Trace complémentaire ouverte, sans nouveaux champs obligatoires imposés par le schéma. |

`forecast` contient exactement `horizon_minutes` et `predicted_occupancy`.
Aucun autre champ top-level n'est accepté en sortie. Le schéma reste compatible
avec les champs null existants ; les validateurs Python appliquent aussi les
invariants de statut et les bornes de capteur.

Les actions autorisées par le contrat général sont :

| Action | Sens |
|---|---|
| `keep_current_signal_plan` | Conserver le plan courant. |
| `monitor` | Surveiller l'évolution. |
| `extend_green_phase` | Valeur conservée dans le schéma général pour compatibilité. |
| `reduce_green_phase` | Valeur conservée dans le schéma général pour compatibilité. |
| `coordinate_with_neighbor` | Demander une coordination avec le voisinage. |
| `manual_review` | Demander une revue/arbitrage à l'orchestrateur. |

`extend_green_phase` et `reduce_green_phase` sont refusées par les garde-fous
actuels, faute de plan de feux. La conformité au schéma ne suffit donc pas à
prouver l'ancrage d'une décision aux observations ou le respect des priorités.

### Statuts et trace

- `warning` : réponse exploitable, avec les limites documentées de mapping et de
  contexte de feux. **C'est le statut actuellement émis pour toutes les réponses
  exploitables**, y compris quand une décision LLM est acceptée.
- `ok` : valeur autorisée par le schéma, réservée à une extension future. Aucun
  exemple n'est présenté ici comme une véritable sortie `ok` de l'agent actuel.
- `error` : pas de prévision ni de décision exploitable. Prévision, état et risque
  sont null ; action `manual_review`, confiance 0.0 et source `rule_based_fallback`.

Les réponses exploitables contiennent actuellement dans `metadata` : timestep,
politique/rang de priorité, trace LLM/fallback, garde-fous critiques, warnings et
marqueur de fixture. Le mode live ajoute `prediction_mode="live"`. Une erreur
contrôlée contient notamment `error_controlled=true`. La liste des métadonnées
reste extensible. Un replay LLM est identifié explicitement ; il ne représente
pas une nouvelle génération du modèle.

## Exemple output warning

Réponse existante correspondant exactement à l'input ci-dessus, sans nouvelle
inférence. Fichier : `src/contracts/example_json/output_warning.json`.

```json
{
  "agent": "traffic_prediction_control",
  "request_id": "trafic_normal",
  "sensor_id": 33,
  "forecast": {
    "horizon_minutes": 15,
    "predicted_occupancy": 0.0027397831
  },
  "traffic_state": "free_flow",
  "risk_level": "low",
  "recommended_action": "keep_current_signal_plan",
  "reason": "Occupation actuelle=0.000000, prévue à +15 min=0.002740, variation=+0.002740; vitesse=65.000000; voisins à occupation élevée=0/2. Incident inconnu; urgence inconnue; priorité transport public inconnue; plan de feux indisponible.",
  "confidence": 0.5,
  "decision_source": "rule_based_fallback",
  "status": "warning",
  "metadata": {
    "timestep": 15227,
    "priority_policy_experimental": {
      "emergency": 100,
      "incident": 80,
      "severe_congestion": 60,
      "public_transport": 40,
      "normal": 20
    },
    "priority": "normal",
    "priority_rank": 20,
    "llm_mode": "replay_saved_09_no_new_inference",
    "fallback_reason": "llm_output_invalid_or_not_grounded",
    "llm_error": null,
    "critical_guard_before": false,
    "critical_guard_after": false,
    "cached_original_fallback": "Critères refusés : reason_grounded, risk_action_coherent, missing_information_respected",
    "warnings": [
      "Mapping location=index NPZ sous hypothèse pour 25 capteurs indiscernables.",
      "Aucun plan de feux disponible ; recommandation sans commande exécutée.",
      "Décision déterministe : llm_output_invalid_or_not_grounded"
    ],
    "integration_test_fixture": false
  }
}
```

## Exemple output error

Réponse existante à un message incomplet, sans inventer de mesure. Fichier :
`src/contracts/example_json/output_error.json`. Le marqueur de fixture indique
que cet échange provient d'un test d'intégration.

```json
{
  "agent": "traffic_prediction_control",
  "request_id": null,
  "sensor_id": null,
  "forecast": {
    "horizon_minutes": 15,
    "predicted_occupancy": null
  },
  "traffic_state": null,
  "risk_level": null,
  "recommended_action": "manual_review",
  "reason": "ValueError: request_id, sensor_id, timestep et traffic_data sont obligatoires.",
  "confidence": 0.0,
  "decision_source": "rule_based_fallback",
  "status": "error",
  "metadata": {
    "integration_test_fixture": true,
    "error_controlled": true
  }
}
```

## Validation Python réutilisable

```python
from src.contracts import (
    Action, INPUT_SCHEMA, OUTPUT_SCHEMA, get_input_schema,
    load_example, validate_input, validate_output,
)

message = load_example("input")
validate_input(message, prediction_mode="historical_replay")  # True ou ValueError
assert validate_output(load_example("output_warning"))       # bool
assert validate_output(load_example("output_error"))

live_schema = get_input_schema("live")  # Copie indépendante du schéma.
assert Action.MONITOR.value == "monitor"
```

`validate_input` contrôle par défaut le JSON strict, le schéma du mode choisi,
les types, les valeurs finies, le timestamp avec fuseau et la cohérence interne
des contextes. `validate_output` contrôle le schéma et les invariants de statut ;
il renvoie false pour une sortie invalide. Ces fonctions ne chargent aucun modèle,
CSV de trafic ou graphe. `INPUT_SCHEMA` correspond au replay historique ;
`get_input_schema("live")` est l'entrée officielle du mode live. Les dictionnaires
sont sérialisables en JSON et utilisables par les autres agents.

Pour préserver les comportements existants, les adaptateurs internes de l'agent
appellent la validation historique avec `check_schema=False`, puis `build_state`
vérifie le snapshot, les voisins et leurs mesures comme auparavant. Les partenaires
utilisent la validation complète par défaut. Les noms de champs, règles de
priorité, décisions et réponses existantes ne sont pas modifiés par cette extraction.

Une validation indépendante réussie ne certifie pas la correspondance au graphe,
aux données du snapshot ou à l'historique fourni. Ces vérifications nécessitent
l'état construit par l'agent. Les garde-fous décisionnels restent sous
`src/decision/guardrails.py`.

```python
response = agent.handle_message(message)
assert validate_output(response)
```

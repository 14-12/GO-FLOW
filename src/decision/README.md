# Couche de décision GoFlow

Les trois modules contiennent uniquement les règles et contrôles déjà présents.
Le forecasting XGBoost, la construction des états et le cycle de vie llama.cpp
restent dans leurs couches respectives.

Le contrat JSON public, ses enums et ses validateurs généraux ont leur source
unique sous `src/contracts/`. `guardrails.py` leur délègue la conformité des
messages/décisions et conserve les garde-fous métier. Voir le
[contrat d'intégration](../../docs/traffic_prediction_agent_contract.md).

| Module | Responsabilité |
|---|---|
| `decision_engine.py` | Contrat de décision et orchestration LLM/replay/fallback ; adaptateurs compatibles des règles |
| `guardrails.py` | JSON strict, extraction de JSON LLM, valeurs autorisées, ancrage aux faits, diagnostics d'hallucination, cohérence risque/action et protection des priorités critiques |
| `policy.py` | Indicateurs, règles déterministes, contextualisation, sélection de priorité ; seuils train nommés, quantiles, actions, risques, bornes et tolérances existants |

`TrafficPredictionControlAgent` délègue ses décisions et garde-fous aux modules
ci-dessus. Ses méthodes privées historiques restent des adaptateurs pour préserver
les tests, les points d'injection et les traces. `handle_message` conserve son
contrat, en modes `live` comme `historical_replay`.

Les fonctions de base peuvent aussi être utilisées sans construire l'agent :

```python
from functools import partial
from src.decision import decision_engine, guardrails

# state et thresholds sont les données et seuils existants fournis par l'appelant.
decision = decision_engine.rule_based_decision(state, thresholds)
contract = partial(decision_engine.decision_contract, thresholds=thresholds)
assert guardrails.validate_decision(decision)
assert guardrails.is_grounded_decision(
    decision, state, validator=guardrails.validate_decision, contract_builder=contract)
```

Les contrats historiques restent distincts :

- `final_decision` préserve le chemin du notebook 08 : entrée canonique vérifiée,
  grounding strict, enveloppe `source=llm|rule_based`.
- `final_agent_decision` préserve le contrat orchestrateur enrichi : priorités,
  garde-fous avant/après LLM, replay éventuel et `decision_source`.
- `final_saved_scenario_decision` préserve le chemin expérimental de 09 : cinq états
  autorisés, contrôles qualitatifs et enveloppe avec diagnostics.

La validation qualitative du backend exige toujours une confidence de type float,
alors que le contrat JSON général accepte aussi les entiers numériques valides.
L'extraction LLM et le parsing JSON strict conservent leurs comportements propres.
Les contrôles lexicaux ne deviennent pas de nouvelles règles sémantiques ; le
contrat enrichi exige toujours sa justification exacte.

Compatibilité : `src.agents.decision_agent`, les anciens imports de scripts et
`src.config.traffic_policy` restent disponibles. Ce dernier chemin désigne le même
module que `src.decision.policy`, sans dupliquer la configuration. Les ensembles
et seuils propres à chaque runtime restent isolés et modifiables comme auparavant.

```powershell
python -B -X utf8 -m unittest tests.test_decision_layer -v
python -B -X utf8 -m unittest discover -s tests -v
```

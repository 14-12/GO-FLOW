# Backend LLM local GoFlow

`LocalLLMBackend` expose Qwen2.5-0.5B-Instruct GGUF via le transport llama.cpp
existant. La façade ne change ni les prompts, ni les paramètres, ni les délais,
ni les règles métier. Les validations finales restent dans `src/decision/`.

```python
from pathlib import Path
from src.agents import TrafficPredictionControlAgent
from src.llm import LocalLLMBackend

root = Path.cwd()  # Racine du dépôt.
with LocalLLMBackend(root) as backend:
    agent = TrafficPredictionControlAgent(root, llm_backend=backend)
    # message respecte src.contracts.traffic_prediction_contract.
    response = agent.handle_message(message)
```

Le gestionnaire de contexte appelle `start()` puis `stop()`, y compris lors
d'une exception. L'interface explicite équivalente est :

```python
backend = LocalLLMBackend(root)
try:
    backend.start()
    if backend.available:
        proposal = backend.generate(state)
finally:
    backend.stop()
```

`state` est l'état enrichi construit par l'agent, pas le message JSON entrant.
`generate(state)` et `backend(state)` ont le même comportement : ils retournent
la proposition parsée ou lèvent l'erreur du transport. Ils ne démarrent jamais
automatiquement le backend. `stop()` peut être appelé plusieurs fois.
`last_call` expose une copie des diagnostics ; `load_error` décrit l'échec de
démarrage et `available` indique la disponibilité. `validation_report(agent,
state)` délègue au rapport existant lorsqu'un transport a été initialisé.

Comme le backend historique, `start()` conserve une erreur de chargement dans
`load_error` et retourne l'instance indisponible. `generate()` lève alors
`RuntimeError("LLM indisponible : …")`. Injectée dans l'agent, cette erreur suit
le chemin existant `llm_backend_error` puis `rule_based_fallback`. Une sortie
JSON parsée mais non conforme reste rejetée par les garde-fous du moteur.
Le moteur ne connaît aucun détail de llama.cpp et n'a pas besoin d'être modifié.

## Fichiers locaux et compatibilité

Le démarrage vérifie explicitement :

- `models/qwen2.5-0.5b-instruct-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` ;
- `models/qwen2.5-0.5b-instruct-gguf/download_provenance.json` ;
- l'exécutable donné par `llama_server_path` dans cette provenance.

Un fichier absent produit un diagnostic `FileNotFoundError` avec son chemin.
Le transport conserve ensuite ses vérifications SHA-256, notamment celles de
l'archive `release.zip`, ses contrôles mémoire et de santé, ainsi que son arrêt
du processus. Aucun téléchargement n'est effectué. Importer `src.llm` ou
construire la façade ne lance ni serveur, ni subprocess, ni chargement de modèle.

Les imports historiques `src.backends.llama_cpp.LlamaCppBackend` et
`scripts.local_llm_agent` restent disponibles. Le notebook 09 conserve son
parcours expérimental historique ; le notebook 10 utilise déjà le transport
encapsulé. Aucun notebook ni résultat sauvegardé n'est modifié.
Les nouvelles intégrations utilisent `src.llm.LocalLLMBackend`.

Aucune dépendance supplémentaire : `requirements-llm.txt` reste applicable,
avec PyTorch/Transformers pour les diagnostics actuels du transport. Les
tests mockent HTTP et le cycle de vie ; aucun vrai serveur n'est démarré.

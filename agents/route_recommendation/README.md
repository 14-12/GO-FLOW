# Agent 6 : Route Recommendation

Recommande un itinéraire, l'explique en français, et laisse l'humain valider.
Prototype : il n'est pas encore branché sur la ville (Town05) en direct.

## Fonctionnement
- Graphe routier : `sumo_map.py` lit `scenarios/town05/town05.net.xml` (23 nœuds, 78 arêtes).
  `grid_map.py` = grille synthétique 6x6 de test.
- Alertes (mockées pour l'instant) : accident (x5), circulation prévue dans N min (x3), circulation ville (x1,3).
- Calcul : A* sur le graphe avec coûts pénalisés (`step5b_agent.py`).
- LLM : Qwen2.5-7B via Ollama + LangGraph, 3 outils : `get_network_state`, `find_routes`, `compare_routes`.
  Le LLM explique, le code calcule.
- Human-in-the-loop : `step6_hitl.py` (interrupt LangGraph).
- Évaluation : `step8_eval_grid.py` (baseline vs agent, 100 scénarios sur grille).

## Dépendances
Environnement séparé : `pip install -r requirements.txt`.
Ollama doit être installé à part (`ollama pull qwen2.5:7b`).

## État
- Vérifié : chargement de town05.net.xml, A* sur ses edge_id.
- Non vérifié : lecture en direct des incidents et des rapports caméra.
- Limites : pénalités non calibrées, évaluation sur grille synthétique avec les mêmes pénalités que l'agent, LLM non évalué.

## Notes
Les fichiers sont copiés tels quels depuis le dossier de travail : chemins relatifs (`data/...`) à adapter.
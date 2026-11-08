# Agent 6 : Route Recommendation

Recommande 1 à 3 itinéraires distincts sur la carte Town05, les compare,
laisse l'humain choisir, et explique le choix en français (Qwen).

## Fonctionnement
- `sumo_map.py` : lit `scenarios/town05/town05.net.xml` (23 nœuds, 78 arêtes) + index de tous les edge_id.
- `step5b_agent.py` : coûts pénalisés par les alertes, A*, `k_routes` (jusqu'à 3 routes distinctes),
  3 outils LangChain, Qwen2.5-7B via Ollama.
- `adapter_traffic.py` : traduit les messages des autres agents vers le format interne
  - DSO4 `forecast` (fill par approche) -> alerte `congestion` (pénalité 1 + 2 x fill)
  - accident (edge, closed_lanes, total_lanes) -> alerte `accident` (x50 si tout est fermé)
- `step6_hitl.py` : pause humaine (LangGraph interrupt), K options, explication Qwen.
- `app.py` : interface Streamlit (carte Plotly, options, boutons Choisir, explication).

## Lancer
pip install -r requirements.txt  (venv séparé, jamais le .venv du projet)
ollama pull qwen2.5:7b
streamlit run app.py   (depuis ce dossier)

## Vérifié
- Chargement de la carte, A* = Dijkstra (test_astar.py)
- 55/55 approches DSO4 retrouvées dans le graphe (test_nodes_map.py)
- Une alerte (accident ou congestion) change la route (test_accident.py, test_congestion.py)

## Limites
- Alertes simulées par boutons (format DSO4 et accident fixé par nous, pas encore reçu en direct).
- Temps = longueur / vitesse limite, corrigé par des pénalités non calibrées.
- Évaluation sur temps SUMO réels : à faire.
- Pas de recalcul en cours de trajet.
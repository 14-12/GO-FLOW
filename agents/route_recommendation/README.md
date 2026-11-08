# Agent 6 : Route Recommendation

Recommande 1 à 3 itinéraires distincts sur Town05, les compare, laisse l'humain choisir, et explique le choix en
français (Qwen). Dans le viewer, il guide la voiture que l'on suit (touche F) et replanifie quand les alertes changent.

## Fonctionnement
- `router.py` : le réseau SUMO de Town05 route par route (80 routes), avec seulement les virages que chaque carrefour
  permet (connexions SUMO). Recherche : Dijkstra, et jusqu'à 3 itinéraires différents (Yen, moins de 70 % de routes
  en commun). Coût : temps à vide (longueur / vitesse limite) x pénalités des alertes ; ou, si demandé, les temps
  typiques des trajets passés (attente aux feux comprise, `models/route_guidance/movement_times.json`).
  N'utilise que sumolib : le viewer et l'évaluation l'importent sans LangChain.
- `adapter_traffic.py` : messages des autres agents -> alertes
  - DSO4 `forecast` (remplissage par approche à +120 s) -> `congestion` (pénalité 1 + 2 x fill, au-dessus de 0.3)
  - accident (edge, closed_lanes, total_lanes) -> `accident` (x50 si tout est fermé)
- `step5b_agent.py` : 3 outils LangChain sur `router.py`, Qwen2.5-7B via Ollama.
- `step6_hitl.py` : pause humaine (LangGraph interrupt), K options, explication Qwen.
- `app.py` : page Streamlit (carte Plotly des vraies formes de routes, options, boutons Choisir, explication).
- Dans le projet : `cosim/route_guidance.py` (la voiture suivie, et une part des voitures pour l'évaluation),
  `cosim/route_minimap.py` (la mini-carte du viewer), `tools/eval_route_guidance.py` (A/B dans SUMO).

## Lancer
    tools\setup_route_recommendation.bat      (venv séparé, jamais le .venv du projet ; lance les tests)
    ollama pull qwen2.5:7b
    tools\route_recommendation_app.bat        (la page Streamlit)

## Vérifié (8 Nov)
- Toutes les routes proposées sont conduisibles (toutes les paires de routes de Town05 : `tests/test_router.py`).
- Une alerte sur une des deux routes parallèles du périphérique (35, 37) reste sur elle.
- Les 55 approches DSO4 sont des routes du graphe.
- Un accident signalé fait changer l'itinéraire (`tests/test_router.py`, et dans SUMO : `tests/test_route_guidance.py`).

## Limites
- Les pénalités ne sont pas calibrées : elles classent les itinéraires, elles ne prédisent pas un temps.
- Le temps à vide ignore l'attente aux feux (sur Town05 un trajet réel dure environ 6 fois son temps à vide).
- L'effet sur le trafic se mesure avec `tools/eval_route_guidance.py` (graines appariées), voir docs/WORKLOG.md.

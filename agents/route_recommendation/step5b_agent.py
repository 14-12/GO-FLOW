import networkx as nx
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from sumo_map import build_from_sumo, EDGE_INDEX

# ---------- 1. Graphe ----------

# def load_graph():
#     # return build_grid()
#     return build_from_sumo("data/town05.net.xml")

from pathlib import Path


NET = Path(__file__).resolve().parents[2] / "scenarios" / "town05" / "town05.net.xml"

def load_graph():
    return build_from_sumo(str(NET))



G = load_graph()


ALERTS = []   # état courant : rempli par les adaptateurs (accident, DSO4)

def set_alerts(new_alerts):
    ALERTS.clear()
    ALERTS.extend(new_alerts)

def add_alert(alert):
    ALERTS.append(alert)
    
P_ACC = {"haute": 5.0, "moyenne": 2.0, "faible": 1.5}
P_TRAF = {"forte": 3.0, "moyenne": 1.5, "faible": 1.2}
P_CITY = {"forte": 1.5, "moyenne": 1.3, "faible": 1.1}

VMAX = max(d["speed"] for _, _, d in G.edges(data=True)) / 3.6   # m/s

def heuristic(u, v):
    dx = G.nodes[u]["x"] - G.nodes[v]["x"]
    dy = G.nodes[u]["y"] - G.nodes[v]["y"]
    return ((dx**2 + dy**2) ** 0.5) / VMAX

# def edge_by_id(eid):
#     for u, v, d in G.edges(data=True):
#         if d["edge_id"] == eid:
#             return u, v

def edge_by_id(eid):
    return EDGE_INDEX.get(eid)

# def set_costs(alerts):
#     for u, v, d in G.edges(data=True):
#         d["cost"] = d["travel_time"]
#     for a in alerts:
#         if a.get("type") == "congestion":
#             e = edge_by_id(a["route"])
#             if e and G.has_edge(*e):
#                 G[e[0]][e[1]]["cost"] *= a["penalty"]
#         elif a["agent"] == 1:
#             e = edge_by_id(a["route"])
#             if e:
#                 G[e[0]][e[1]]["cost"] *= P_ACC[a["severite"]]
#         elif a["agent"] == 4:
#             for u, v, d in G.edges(data=True):
#                 d["cost"] *= P_CITY[a["niveau"]]

def set_costs(alerts):
    for u, v, d in G.edges(data=True):
        d["cost"] = d["travel_time"]
    for a in alerts:
        if a.get("type") in ("congestion", "accident") and "penalty" in a:
            e = edge_by_id(a["route"])
            if e and G.has_edge(*e):
                G[e[0]][e[1]]["cost"] *= a["penalty"]
        elif a.get("agent") == 1:                      # ancien format mock
            e = edge_by_id(a["route"])
            if e and G.has_edge(*e):
                G[e[0]][e[1]]["cost"] *= P_ACC[a["severite"]]
        elif a.get("agent") == 4:
            for u, v, d in G.edges(data=True):
                d["cost"] *= P_CITY[a["niveau"]]
        

def apply_timed(alerts, path):
    t = 0.0
    for u, v in zip(path, path[1:]):
        d = G[u][v]
        for a in alerts:
            if a["agent"] == 2 and "dans_minutes" in a and a["route"] == d["edge_id"]:
                if abs(t / 60 - a["dans_minutes"]) <= 5:
                    d["cost"] *= P_TRAF[a["niveau"]]
        t += d["cost"]

def compute(alerts, o, d):
    set_costs(alerts)
    p1 = nx.astar_path(G, o, d, heuristic=heuristic, weight="cost")
    apply_timed(alerts, p1)
    return nx.astar_path(G, o, d, heuristic=heuristic, weight="cost")

from itertools import islice

K_MAX = 3          # maximum d'options affichées
MAX_OVERLAP = 0.7  # une alternative doit différer d'au moins 30 % des routes

def _edge_ids(path):
    return [G[u][v]["edge_id"] for u, v in zip(path, path[1:])]

def k_routes(alerts, o, d, k_max=K_MAX):
    """Jusqu'à k_max itinéraires réellement différents (1 à k_max)."""
    set_costs(alerts)
    kept = []
    for path in islice(nx.shortest_simple_paths(G, o, d, weight="cost"), 50):
        ids = set(_edge_ids(path))
        if all(len(ids & set(_edge_ids(p))) / len(ids) < MAX_OVERLAP for p in kept):
            kept.append(path)
        if len(kept) == k_max:
            break
    return kept



def summarize(path, alerts):
    ids = [G[u][v]["edge_id"] for u, v in zip(path, path[1:])]
    real_time = sum(G[u][v]["travel_time"] for u, v in zip(path, path[1:]))
    dist = sum(G[u][v]["length"] for u, v in zip(path, path[1:]))
    hit = [a for a in alerts if a.get("route") in ids]
    city = [a for a in alerts if a["agent"] == 4]
    m, s = divmod(int(round(real_time)), 60)
    return {
        "chemin": " -> ".join(path),
        "routes": ids,
        "temps_reel_secondes": round(real_time, 1),
        "temps_formate": f"{m} min {s} s",
        "distance_metres": dist,
        "alertes_rencontrees": hit + city,
    }

# ---------- 3. Outils ----------
@tool
def get_network_state() -> list:
    """Retourne les alertes actuelles du reseau (accidents, circulation prevue,
    circulation en ville) envoyees par les agents 1, 2 et 4."""
    return ALERTS

@tool
def find_routes(origin: str, destination: str) -> dict:
    """Calcule la meilleure route entre deux noeuds (ex: 'A' et 'E') en tenant
    compte des alertes actuelles. Retourne chemin, routes, temps, distance."""
    if origin not in G or destination not in G:
        return {"erreur": f"Noeud inconnu. Noeuds valides : {list(G.nodes)}"}
    path = compute(ALERTS, origin, destination)
    return summarize(path, ALERTS)

@tool
def compare_routes(origin: str, destination: str) -> dict:
    """Compare la route SANS alerte (reference) a la route AVEC alertes.
    Utile pour justifier le choix et montrer le detour."""
    if origin not in G or destination not in G:
        return {"erreur": f"Noeud inconnu. Noeuds valides : {list(G.nodes)}"}
    ref = compute([], origin, destination)
    ref_s = summarize(ref, [])
    new = compute(ALERTS, origin, destination)
    new_s = summarize(new, ALERTS)

    ref_ids = set(ref_s["routes"])
    new_ids = set(new_s["routes"])
    evitees = [a for a in ALERTS if a.get("route") in ref_ids - new_ids]

    diff = round(new_s["temps_reel_secondes"] - ref_s["temps_reel_secondes"], 1)
    if diff > 0:
        verdict = f"La route recommandee est PLUS LONGUE de {int(diff)} s que la route de reference."
    elif diff < 0:
        verdict = f"La route recommandee est PLUS COURTE de {int(-diff)} s que la route de reference."
    else:
        verdict = "Meme duree que la route de reference."

    return {
        "route_reference_sans_alerte": ref_s,
        "route_recommandee_avec_alertes": new_s,
        "difference_temps_secondes": diff,
        "verdict_temps": verdict,
        "alertes_evitees": evitees,
    }

#- Les noeuds valides sont A, B, C, D, E. Extrais origine et destination de la demande.
# ---------- 4. Agent ----------
SYSTEM_PROMPT = """Tu es l'agent de recommandation d'itineraire.
Regles strictes :
- Utilise TOUJOURS les outils pour obtenir les chiffres. N'invente jamais un temps, une distance ou une route.
- Les noeuds sont des identifiants SUMO (ex: 1126, 979)
- Appelle d'abord get_network_state, puis find_routes, puis compare_routes.
- Cite uniquement les chiffres renvoyes par les outils. Si une info manque, dis-le.
- Reponds en francais, de facon courte : route recommandee, temps, alertes evitees ou rencontrees, et pourquoi.
- Respecte les unites : les champs *_secondes sont en secondes, *_minutes en minutes, *_metres en metres. Ne convertis jamais toi-meme.
- Pour les alertes evitees, utilise uniquement le champ "alertes_evitees" de compare_routes.
- Pour le temps, recopie exactement le champ "temps_formate". Ne fais aucune conversion ni arrondi.
- Pour comparer les durees, recopie exactement le champ "verdict_temps". N'ajoute aucune affirmation de ton cru (comme "plus rapide" ou "plus sur").
"""

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
agent = create_agent(llm, [get_network_state, find_routes, compare_routes],
                     system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    question = "Je veux aller de A a E le plus vite possible, il y a des problemes ?"
    result = agent.invoke({"messages": [("user", question)]})
    for m in result["messages"]:
        kind = m.type
        if kind == "ai" and m.tool_calls:
            print(f"[Qwen appelle] {[(c['name'], c['args']) for c in m.tool_calls]}")
        elif kind == "tool":
            print(f"[Outil {m.name} renvoie] {str(m.content)[:300]}")
        elif kind == "ai":
            print(f"\n[Reponse finale]\n{m.content}")
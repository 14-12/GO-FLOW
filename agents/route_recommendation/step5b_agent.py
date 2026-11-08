"""Route recommendation agent: the route search (router.py, road by road), the alerts in force, and three tools for
an LLM (Qwen2.5-7B through Ollama) that explains the result. Routes go from one Town05 junction to another."""
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from router import Router

R = Router()

ALERTS = []   # alerts in force: filled by the adapters (accident, DSO4)


def set_alerts(new_alerts):
    ALERTS.clear()
    ALERTS.extend(new_alerts)


def add_alert(alert):
    ALERTS.append(alert)


def compute(alerts, o, d):
    """Cheapest route (list of SUMO edge ids) from junction o to junction d with these alerts."""
    return R.best_between(o, d, alerts)


def k_routes(alerts, o, d, k_max=3):
    """Up to k_max different routes from junction o to junction d."""
    return R.k_between(o, d, alerts, k_max=k_max)


def summarize(route, alerts):
    s = R.summary(route, alerts, skip_first=False)
    s["depart"], s["arrivee"] = R.roads[route[0]].frm, R.roads[route[-1]].to
    return s


def nodes():
    return sorted(R.junctions)


# ---------- tools ----------
@tool
def get_network_state() -> list:
    """Retourne les alertes actuelles du reseau (accidents, circulation prevue) envoyees par les autres agents."""
    return ALERTS


@tool
def find_routes(origin: str, destination: str) -> dict:
    """Calcule la meilleure route entre deux carrefours (identifiants SUMO, ex: '24.11' et '359') en tenant
    compte des alertes actuelles. Retourne routes, temps sans attente aux feux, distance."""
    if origin not in R.junctions or destination not in R.junctions:
        return {"erreur": f"Carrefour inconnu. Carrefours valides : {nodes()}"}
    path = compute(ALERTS, origin, destination)
    if path is None:
        return {"erreur": "Aucune route possible entre ces carrefours."}
    return summarize(path, ALERTS)


@tool
def compare_routes(origin: str, destination: str) -> dict:
    """Compare la route SANS alerte (reference) a la route AVEC alertes. Utile pour justifier le choix et montrer
    le detour."""
    if origin not in R.junctions or destination not in R.junctions:
        return {"erreur": f"Carrefour inconnu. Carrefours valides : {nodes()}"}
    ref, new = compute([], origin, destination), compute(ALERTS, origin, destination)
    if ref is None or new is None:
        return {"erreur": "Aucune route possible entre ces carrefours."}
    ref_s, new_s = summarize(ref, []), summarize(new, ALERTS)
    diff = round(new_s["temps_libre_secondes"] - ref_s["temps_libre_secondes"], 1)
    if diff > 0:
        verdict = f"La route recommandee est PLUS LONGUE de {int(diff)} s (sans attente aux feux) que la reference."
    elif diff < 0:
        verdict = f"La route recommandee est PLUS COURTE de {int(-diff)} s (sans attente aux feux) que la reference."
    else:
        verdict = "Meme duree que la route de reference."
    return {"route_reference_sans_alerte": ref_s, "route_recommandee_avec_alertes": new_s,
            "difference_temps_secondes": diff, "verdict_temps": verdict,
            "alertes_evitees": R.avoided(ref, new, ALERTS)}


# ---------- agent ----------
SYSTEM_PROMPT = """Tu es l'agent de recommandation d'itineraire.
Regles strictes :
- Utilise TOUJOURS les outils pour obtenir les chiffres. N'invente jamais un temps, une distance ou une route.
- Les carrefours sont des identifiants SUMO (ex: 24.11, 359).
- Appelle d'abord get_network_state, puis find_routes, puis compare_routes.
- Cite uniquement les chiffres renvoyes par les outils. Si une info manque, dis-le.
- Reponds en francais, de facon courte : route recommandee, temps, alertes evitees ou rencontrees, et pourquoi.
- Les temps sont SANS attente aux feux : dis-le quand tu donnes un temps.
- Respecte les unites : les champs *_secondes sont en secondes, *_metres en metres. Ne convertis jamais toi-meme.
- Pour les alertes evitees, utilise uniquement le champ "alertes_evitees" de compare_routes.
- Pour le temps, recopie exactement le champ "temps_formate". Ne fais aucune conversion ni arrondi.
- Pour comparer les durees, recopie exactement le champ "verdict_temps". N'ajoute aucune affirmation de ton cru.
"""

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
agent = create_agent(llm, [get_network_state, find_routes, compare_routes], system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    question = "Je veux aller de 24.11 a 359 le plus vite possible, il y a des problemes ?"
    result = agent.invoke({"messages": [("user", question)]})
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            print(f"[Qwen appelle] {[(c['name'], c['args']) for c in m.tool_calls]}")
        elif m.type == "tool":
            print(f"[Outil {m.name} renvoie] {str(m.content)[:300]}")
        elif m.type == "ai":
            print(f"\n[Reponse finale]\n{m.content}")

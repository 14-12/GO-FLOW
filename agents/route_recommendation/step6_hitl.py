"""Human in the loop (LangGraph interrupt): the code computes 1 to 3 different routes, a person picks one, Qwen
explains it from the numbers only."""
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from step5b_agent import ALERTS, SYSTEM_PROMPT, R, compute, k_routes, summarize

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


class State(TypedDict, total=False):
    origin: str
    destination: str
    options: dict      # {"1": {...}, "2": {...}}
    choix: str
    explication: str


def compute_options(state: State):
    o, d = state["origin"], state["destination"]
    ref = compute([], o, d) or []                     # the route with no alert: what each option avoids
    options = {}
    for i, p in enumerate(k_routes(ALERTS, o, d), start=1):
        s = summarize(p, ALERTS)
        s["nom"] = "Route recommandee" if i == 1 else f"Alternative {i - 1}"
        s["alertes_evitees"] = [f"{a['type']} sur {a['route']}" for a in R.avoided(ref, p, ALERTS)]
        options[str(i)] = s
    return {"options": options}


def human_choice(state: State):
    choix = interrupt({
        "question": "Quelle route choisissez-vous ?",
        "options": {k: f"{v['nom']} : {' -> '.join(v['routes'])} ({v['temps_formate']} sans attente aux feux, "
                       f"{v['cout_avec_alertes_secondes']} s avec alertes), "
                       f"{len(v['alertes_rencontrees'])} alerte(s) sur le trajet"
                    for k, v in state["options"].items()},
    })
    return {"choix": choix}


def explain(state: State):
    route = state["options"][state["choix"]]
    prompt = (SYSTEM_PROMPT
              + "\nL'utilisateur a choisi cette route. Explique-la en 3 phrases maximum, en recopiant exactement les "
                "valeurs ci-dessous, sans rien inventer. Mentionne aussi les alertes_evitees : c'est la raison de ce "
                "choix. Mentionne explicitement les alertes rencontrees si la liste n'est pas vide.\n"
              + str(route))
    return {"explication": llm.invoke(prompt).content}


builder = StateGraph(State)
builder.add_node("compute_options", compute_options)
builder.add_node("human_choice", human_choice)
builder.add_node("explain", explain)
builder.add_edge(START, "compute_options")
builder.add_edge("compute_options", "human_choice")
builder.add_edge("human_choice", "explain")
builder.add_edge("explain", END)
graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}
    result = graph.invoke({"origin": "24.11", "destination": "359"}, config)   # stops at interrupt()
    info = result["__interrupt__"][0].value
    print(info["question"])
    for k, v in info["options"].items():
        print(f"  [{k}] {v}")
    choix = input(f"Votre choix (1 a {len(info['options'])}) : ").strip()
    final = graph.invoke(Command(resume=choix), config)                     # resumes after the pause
    print("\n[Explication]\n" + final["explication"])

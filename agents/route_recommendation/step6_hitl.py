from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_ollama import ChatOllama

from step5b_agent import compute, summarize, ALERTS, G, SYSTEM_PROMPT

llm = ChatOllama(model="qwen2.5:7b", temperature=0)


class State(TypedDict, total=False):
    origin: str
    destination: str
    options: dict      # {"A": {...}, "B": {...}}
    choix: str
    explication: str


# ---- Noeud 1 : le code calcule les 2 candidates ----
def compute_options(state: State):
    o, d = state["origin"], state["destination"]
    ref = summarize(compute([], o, d), ALERTS)   # <- ALERTS au lieu de []
    rec = summarize(compute(ALERTS, o, d), ALERTS)
    return {"options": {
        "1": {"nom": "Route la plus courte (ignore les alertes)", **ref},
        "2": {"nom": "Route recommandee (evite les alertes)", **rec},
    }}

# ---- Noeud 2 : PAUSE, l'humain choisit ----
def human_choice(state: State):
    choix = interrupt({
        "question": "Quelle route choisissez-vous ?",
        "options": {k: f"{v['nom']} : {v['chemin']} ({v['temps_formate']}), "
                        f"{len(v['alertes_rencontrees'])} alerte(s) sur le trajet"
                    for k, v in state["options"].items()},
    })
    return {"choix": choix}


# ---- Noeud 3 : Qwen explique la route choisie ----
def explain(state: State):
    route = state["options"][state["choix"]]
    prompt = (
        SYSTEM_PROMPT
        + "\nL'utilisateur a choisi cette route. Explique-la en 3 phrases maximum, "
            "en recopiant exactement les valeurs ci-dessous, sans rien inventer. "
            "Mentionne explicitement les alertes rencontrees si la liste n'est pas vide.\n"
        + str(route)
    )
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

    # 1re execution : s'arrete a interrupt()
    result = graph.invoke({"origin": "A", "destination": "E"}, config)
    info = result["__interrupt__"][0].value
    print(info["question"])
    for k, v in info["options"].items():
        print(f"  [{k}] {v}")

    # Reponse humaine
    choix = input("Votre choix (1 ou 2) : ").strip()

    # 2e execution : reprend apres la pause
    final = graph.invoke(Command(resume=choix), config)
    print("\n[Explication]\n" + final["explication"])
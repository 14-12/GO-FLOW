import plotly.graph_objects as go
import streamlit as st

import step5b_agent as a

st.set_page_config(page_title="Agent 6 : Route Recommendation", layout="wide")
st.title("Town05 : recommandation d'itinéraire")


def base_map():
    fig = go.Figure()
    xs, ys = [], []
    for u, v in a.G.edges():
        xs += [a.G.nodes[u]["x"], a.G.nodes[v]["x"], None]
        ys += [a.G.nodes[u]["y"], a.G.nodes[v]["y"], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="lightgray", width=2),
                             hoverinfo="skip", name="Routes"))
    fig.add_trace(go.Scatter(
        x=[d["x"] for _, d in a.G.nodes(data=True)],
        y=[d["y"] for _, d in a.G.nodes(data=True)],
        mode="markers+text", text=list(a.G.nodes),
        textposition="top center", textfont=dict(size=9),
        marker=dict(size=7, color="#9aa5b1"), name="Carrefours"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
    return fig

def est(p):
    """Temps estimé avec alertes (somme des coûts pénalisés)."""
    return sum(a.G[u][v]["cost"] for u, v in zip(p, p[1:]))

def fmt(sec):
    sec = int(round(sec))
    return f"{sec} s" if sec < 60 else f"{sec // 60} min {sec % 60} s"


st.caption(f"{len(a.G.nodes)} carrefours, {len(a.G.edges)} routes")

from adapter_traffic import accident_to_alert

# --- Panneau gauche : trajet ---
nodes = sorted(a.G.nodes)
c1, c2 = st.sidebar.columns(2)
origin = c1.selectbox("Départ", nodes, index=nodes.index("24.11"))
dest = c2.selectbox("Arrivée", nodes, index=nodes.index("359"))

# --- Accident de test (temporaire, remplacé plus tard par les agents) ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []
base_ids = a._edge_ids(a.k_routes([], origin, dest)[0]) if origin != dest else []
if st.sidebar.button("Simuler un accident sur le trajet") and len(base_ids) > 2:
    st.session_state.alerts = [accident_to_alert({
        "edge": base_ids[len(base_ids) // 2], "severity": "high",
        "closed_lanes": 2, "total_lanes": 2})]
if st.sidebar.button("Effacer les alertes"):
    st.session_state.alerts = []

from adapter_traffic import forecast_to_alerts, load_approach_map

if st.sidebar.button("Simuler une congestion (DSO4)") and len(base_ids) > 1:
    amap = load_approach_map()                     # approche -> edge_id
    edge2approach = {e: n for n, e in amap.items()}
    # on cherche une route du trajet qui est une approche DSO4
    cand = [e for e in base_ids if e in edge2approach]
    if cand:
        msg = {"nodes": {edge2approach[cand[0]]: {"fill": [0.7, 0.9, 1.0]}}}
        st.session_state.alerts += forecast_to_alerts(msg, amap)
    else:
        st.sidebar.warning("Aucune approche DSO4 sur ce trajet : change de trajet.")
a.set_alerts(st.session_state.alerts)

# --- Routes colorées ---
fig = base_map()
colors = ["#22c55e", "#3b82f6", "#f59e0b"]
if origin != dest:
    paths = a.k_routes(a.ALERTS, origin, dest)
    costs = [est(p) for p in paths]          # juste après k_routes (coûts pénalisés)
    for i, p in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=[a.G.nodes[n]["x"] for n in p], y=[a.G.nodes[n]["y"] for n in p],
            mode="lines", line=dict(color=colors[i], width=6 - i),
            name=f"Option {i + 1} ({fmt(costs[i])} estimé)"))
for al in a.ALERTS:
    e = a.edge_by_id(al["route"])
    color = "red" if al["type"] == "accident" else "orange"
    label = "Accident" if al["type"] == "accident" else "Congestion"
    if e:
        (x1, y1), (x2, y2) = [(a.G.nodes[n]["x"], a.G.nodes[n]["y"]) for n in e]
        fig.add_trace(go.Scatter(x=[(x1 + x2) / 2], y=[(y1 + y2) / 2], mode="markers",
                                 marker=dict(symbol="x", size=18, color=color),
                                 name=f"{label} sur {al['route']}"))
st.plotly_chart(fig, width="stretch", key="carte_routes")

if origin != dest:
    alert_edges = {al["route"]: al for al in a.ALERTS}
    rows = []
    for i, p in enumerate(paths):
        s = a.summarize(p, a.ALERTS)
        hit = [f"{alert_edges[r]['type']} sur {r}" for r in s["routes"] if r in alert_edges]
        rows.append({"Option": i + 1,
                     "Temps estimé": fmt(costs[i]),
                     "Temps libre (sans alerte)": fmt(s["temps_reel_secondes"]),
                     "Distance (m)": round(s["distance_metres"]),
                     "Alertes traversées": ", ".join(hit) if hit else "aucune"})
    st.dataframe(rows, hide_index=True, width="stretch")

    from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

if origin != dest and paths:
    st.subheader("Votre choix")
    ref_ids = set(a._edge_ids(a.k_routes([], origin, dest)[0]))
    a.k_routes(a.ALERTS, origin, dest)        # remet les coûts pénalisés
    cols = st.columns(len(paths))
    for i, p in enumerate(paths):
        if cols[i].button(f"Choisir l'option {i + 1}", key=f"choix_{i}"):
            s = a.summarize(p, a.ALERTS)
            s["cout_estime"] = fmt(costs[i])
            s["alertes_evitees"] = [
                f"{al['type']} sur {al['route']}" for al in a.ALERTS
                if al.get("route") in ref_ids - set(s["routes"])]
            s["alertes_traversees"] = [
                f"{al['type']} sur {al['route']}" for al in a.ALERTS
                if al.get("route") in set(s["routes"])]
            prompt = (
                "Tu es l'agent de recommandation d'itineraire. Explique en 3 phrases "
                "maximum, en francais, la route choisie par l'utilisateur. Recopie "
                "exactement les valeurs ci-dessous, n'invente rien. Mentionne les "
                "alertes evitees et traversees.\n"
                f"Option {i + 1} : routes {s['routes']}, temps estime {s['cout_estime']}, "
                f"distance {round(s['distance_metres'])} m, "
                f"alertes evitees {s['alertes_evitees']}, "
                f"alertes traversees {s['alertes_traversees']}")
            with st.spinner("Qwen réfléchit..."):
                st.session_state.explication = llm.invoke(prompt).content
if "explication" in st.session_state:
    st.info(st.session_state.explication)
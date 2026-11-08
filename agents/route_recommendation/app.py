"""Streamlit page: Town05's roads, up to 3 routes between two junctions, test alerts, the choice and Qwen's
explanation. Times are free-flow times (no waiting at signals)."""
import plotly.graph_objects as go
import step5b_agent as a
import streamlit as st
from adapter_traffic import accident_to_alert, forecast_to_alerts, load_approach_map

R = a.R
st.set_page_config(page_title="Agent 6 : Route Recommendation", layout="wide")
st.title("Town05 : recommandation d'itinéraire")


def line(route):
    xs, ys = [], []
    for r in route:
        for x, y in R.roads[r].shape:
            xs.append(x)
            ys.append(y)
    return xs, ys


def base_map():
    fig = go.Figure()
    xs, ys = [], []
    for road in R.roads.values():
        for x, y in road.shape:
            xs.append(x)
            ys.append(y)
        xs.append(None)
        ys.append(None)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "lightgray", "width": 2}, hoverinfo="skip",
                             name="Routes"))
    js = sorted(R.junctions)
    fig.add_trace(go.Scatter(x=[R.junctions[j][0] for j in js], y=[R.junctions[j][1] for j in js],
                             mode="markers+text", text=js, textposition="top center", textfont={"size": 9},
                             marker={"size": 7, "color": "#9aa5b1"}, name="Carrefours"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=650, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    return fig


def fmt(sec):
    sec = round(sec)
    return f"{sec} s" if sec < 60 else f"{sec // 60} min {sec % 60} s"


st.caption(f"{len(R.junctions)} carrefours, {len(R.roads)} routes. Temps sans attente aux feux.")

nodes = a.nodes()
c1, c2 = st.sidebar.columns(2)
origin = c1.selectbox("Départ", nodes, index=nodes.index("24.11"))
dest = c2.selectbox("Arrivée", nodes, index=nodes.index("359"))

# --- test alerts (until the agents send them live) ---
if "alerts" not in st.session_state:
    st.session_state.alerts = []
ref = a.compute([], origin, dest) if origin != dest else None
base_ids = ref or []
if st.sidebar.button("Simuler un accident sur le trajet") and len(base_ids) > 2:
    road = base_ids[len(base_ids) // 2]
    st.session_state.alerts = [accident_to_alert({"edge": road, "severity": "high", "closed_lanes": R.roads[road].lanes,
                                                  "total_lanes": R.roads[road].lanes})]
if st.sidebar.button("Effacer les alertes"):
    st.session_state.alerts = []
if st.sidebar.button("Simuler une congestion (DSO4)") and len(base_ids) > 1:
    amap = load_approach_map()
    edge2approach = {e: n for n, e in amap.items()}
    cand = [e for e in base_ids if e in edge2approach]
    if cand:
        msg = {"nodes": {edge2approach[cand[0]]: {"fill": [0.7, 0.9, 1.0]}}}
        st.session_state.alerts += forecast_to_alerts(msg, amap)
    else:
        st.sidebar.warning("Aucune approche DSO4 sur ce trajet : change de trajet.")
a.set_alerts(st.session_state.alerts)
pen = R.penalties(a.ALERTS)

fig = base_map()
colors = ["#22c55e", "#3b82f6", "#f59e0b"]
paths = a.k_routes(a.ALERTS, origin, dest) if origin != dest else []
costs = [R.route_cost(p, pen, skip_first=False) for p in paths]
for i, p in enumerate(paths):
    xs, ys = line(p)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": colors[i], "width": 6 - i},
                             name=f"Option {i + 1} ({fmt(costs[i])} avec alertes)"))
for al in a.ALERTS:
    road = R.roads.get(al["route"])
    if road is None:
        continue
    x, y = road.shape[len(road.shape) // 2]
    color = "red" if al["type"] == "accident" else "orange"
    label = "Accident" if al["type"] == "accident" else "Congestion"
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker={"symbol": "x", "size": 18, "color": color},
                             name=f"{label} sur {al['route']}"))
st.plotly_chart(fig, width="stretch", key="carte_routes")

if paths:
    rows = []
    for i, p in enumerate(paths):
        s = a.summarize(p, a.ALERTS)
        hit = [f"{al['type']} sur {al['route']}" for al in s["alertes_rencontrees"]]
        rows.append({"Option": i + 1, "Avec alertes": fmt(costs[i]),
                     "Sans attente aux feux": fmt(s["temps_libre_secondes"]),
                     "Distance (m)": round(s["distance_metres"]), "Alertes traversées": ", ".join(hit) or "aucune"})
    st.dataframe(rows, hide_index=True, width="stretch")

    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="qwen2.5:7b", temperature=0)
    st.subheader("Votre choix")
    cols = st.columns(len(paths))
    for i, p in enumerate(paths):
        if cols[i].button(f"Choisir l'option {i + 1}", key=f"choix_{i}"):
            s = a.summarize(p, a.ALERTS)
            evitees = [f"{al['type']} sur {al['route']}" for al in R.avoided(base_ids, p, a.ALERTS)]
            traversees = [f"{al['type']} sur {al['route']}" for al in s["alertes_rencontrees"]]
            prompt = ("Tu es l'agent de recommandation d'itineraire. Explique en 3 phrases maximum, en francais, la "
                      "route choisie par l'utilisateur. Recopie exactement les valeurs ci-dessous, n'invente rien. "
                      "Les temps sont sans attente aux feux. Mentionne les alertes evitees et traversees.\n"
                      f"Option {i + 1} : routes {s['routes']}, temps avec alertes {fmt(costs[i])}, "
                      f"distance {round(s['distance_metres'])} m, alertes evitees {evitees}, "
                      f"alertes traversees {traversees}")
            with st.spinner("Qwen réfléchit..."):
                st.session_state.explication = llm.invoke(prompt).content
if "explication" in st.session_state:
    st.info(st.session_state.explication)

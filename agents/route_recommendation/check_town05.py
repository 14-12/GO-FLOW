import networkx as nx
from sumo_map import build_from_sumo

G = build_from_sumo()
vmax = max(d["speed"] for _, _, d in G.edges(data=True)) / 3.6

def h(u, v):
    dx = G.nodes[u]["x"] - G.nodes[v]["x"]
    dy = G.nodes[u]["y"] - G.nodes[v]["y"]
    return ((dx**2 + dy**2) ** 0.5) / vmax

nodes = sorted(G.nodes)
print("Noeuds :", nodes)
o, d = nodes[0], nodes[-1]
p = nx.astar_path(G, o, d, heuristic=h, weight="travel_time")
ids = [G[u][v]["edge_id"] for u, v in zip(p, p[1:])]
print(f"\nA* {o} -> {d} : {len(ids)} routes")
print("edge_id :", ids)
print(f"Temps : {nx.path_weight(G, p, weight='travel_time'):.1f} s")

# test de conversion voie -> arete (format SUMO "<edge>_<index>")
lane = ids[0] + "_0"
print("\nVoie", lane, "-> arete", lane.rsplit("_", 1)[0],
      "| existe dans le graphe :", any(dd["edge_id"] == lane.rsplit("_", 1)[0] for _, _, dd in G.edges(data=True)))
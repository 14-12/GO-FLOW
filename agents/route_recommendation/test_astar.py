import networkx as nx
import step5b_agent as a

nodes = list(a.G.nodes)
far = max(((u, v) for u in nodes for v in nodes if u != v),
          key=lambda p: (a.G.nodes[p[0]]["x"] - a.G.nodes[p[1]]["x"]) ** 2
                      + (a.G.nodes[p[0]]["y"] - a.G.nodes[p[1]]["y"]) ** 2)
o, d = far
p = a.compute([], o, d)
ids = [a.G[u][v]["edge_id"] for u, v in zip(p, p[1:])]
print("Départ/arrivée les plus éloignés :", o, "->", d)
print("Nombre de routes :", len(ids))
print("edge_id :", ids)
print("Temps :", a.summarize(p, [])["temps_formate"])
print("Même chemin que Dijkstra :", p == nx.dijkstra_path(a.G, o, d, weight="travel_time"))
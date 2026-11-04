import networkx as nx
import sumolib


def build_from_sumo(net_path="data/town05.net.xml"):
    net = sumolib.net.readNet(net_path)
    G = nx.DiGraph()
    for n in net.getNodes():
        x, y = n.getCoord()
        G.add_node(n.getID(), x=x, y=y)
    for e in net.getEdges():
        if e.getFunction() != "":        # ignore les edges internes
            continue
        if not e.allows("passenger"):    # ignore les routes interdites aux voitures
            continue
        u, v = e.getFromNode().getID(), e.getToNode().getID()
        length = e.getLength()
        speed_ms = e.getSpeed()
        tt = length / speed_ms
        # DiGraph : si 2 edges relient les memes noeuds, on garde le plus rapide
        if G.has_edge(u, v) and G[u][v]["travel_time"] <= tt:
            continue
        G.add_edge(u, v, edge_id=e.getID(), length=length,
                   speed=speed_ms * 3.6, travel_time=tt, cost=tt)
    return G


if __name__ == "__main__":
    G = build_from_sumo()
    print("Noeuds :", len(G.nodes), "| Routes :", len(G.edges))
    print("Exemple :", list(G.edges(data=True))[:2])
    print("Fortement connexe :", nx.is_strongly_connected(G))
    print("Plus grande composante :",
          len(max(nx.strongly_connected_components(G), key=len)))
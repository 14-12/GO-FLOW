import networkx as nx
import sumolib

EDGE_INDEX = {}   # NOUVEAU : edge_id SUMO -> (u, v), pour TOUS les edges


def build_from_sumo(net_path="data/town05.net.xml"):
    net = sumolib.net.readNet(net_path)
    G = nx.DiGraph()
    EDGE_INDEX.clear()                               # NOUVEAU
    for n in net.getNodes():
        x, y = n.getCoord()
        G.add_node(n.getID(), x=x, y=y)
    for e in net.getEdges():
        if e.getFunction() != "":
            continue
        if not e.allows("passenger"):
            continue
        u, v = e.getFromNode().getID(), e.getToNode().getID()
        EDGE_INDEX[e.getID()] = (u, v)               # NOUVEAU (avant le continue)
        length = e.getLength()
        speed_ms = e.getSpeed()
        tt = length / speed_ms
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
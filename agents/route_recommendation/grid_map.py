import random
import networkx as nx

def build_grid(n=6, block=200, seed=42):
    """Grille n x n, routes a double sens, vitesses variees. Noeuds: N_ligne_colonne."""
    rnd = random.Random(seed)
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            G.add_node(f"N_{i}_{j}", x=j * block, y=i * block)

    k = 0
    def add_road(u, v):
        nonlocal k
        speed = rnd.choice([30, 50, 70])          # km/h
        length = block
        tt = length / (speed / 3.6)
        for a, b in ((u, v), (v, u)):              # double sens
            k += 1
            G.add_edge(a, b, edge_id=f"R{k}", length=length, speed=speed,
                       travel_time=tt, cost=tt)

    for i in range(n):
        for j in range(n):
            if j + 1 < n:
                add_road(f"N_{i}_{j}", f"N_{i}_{j+1}")
            if i + 1 < n:
                add_road(f"N_{i}_{j}", f"N_{i+1}_{j}")
    return G

if __name__ == "__main__":
    G = build_grid()
    print("Noeuds :", len(G.nodes), "| Routes :", len(G.edges))
    print("Exemple :", list(G.edges(data=True))[:2])
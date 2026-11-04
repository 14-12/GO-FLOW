import random, statistics
from step5b_agent import G, compute, P_ACC, P_TRAF, P_CITY


def real_time(path, alerts):
    """Temps reellement subi sur ce chemin, alertes comprises."""
    t, graves = 0.0, 0
    for u, v in zip(path, path[1:]):
        d = G[u][v]
        dur = d["travel_time"]
        for a in alerts:
            if a["agent"] == 1 and a["route"] == d["edge_id"]:
                dur *= P_ACC[a["severite"]]
                graves += 1
            elif a["agent"] == 2 and a["route"] == d["edge_id"]:
                if abs(t / 60 - a["dans_minutes"]) <= 5:
                    dur *= P_TRAF[a["niveau"]]
                    graves += 1
            elif a["agent"] == 4:
                dur *= P_CITY[a["niveau"]]
        t += dur
    return t, graves


rnd = random.Random(0)
nodes = list(G.nodes)

gains, graves_base, graves_agent = [], 0, 0
for _ in range(100):
    o, d = rnd.sample(nodes, 2)
    base = compute([], o, d)
    ids = [G[u][v]["edge_id"] for u, v in zip(base, base[1:])]
    if len(ids) < 2:
        continue
    alerts = [
        {"agent": 1, "route": rnd.choice(ids), "type": "accident", "severite": "haute"},
        {"agent": 2, "route": rnd.choice(ids), "dans_minutes": 1, "niveau": "forte"},
    ]
    agent = compute(alerts, o, d)
    tb, gb = real_time(base, alerts)
    ta, ga = real_time(agent, alerts)
    gains.append((1 - ta / tb) * 100)
    graves_base += gb > 0
    graves_agent += ga > 0

print(f"Scenarios evalues : {len(gains)}")
print(f"Gain moyen du temps : {statistics.mean(gains):.1f} % (ecart-type {statistics.stdev(gains):.1f})")
print(f"Cas ou l'agent est meilleur ou egal : {sum(g >= 0 for g in gains)}/{len(gains)}")
print(f"Scenarios avec alerte grave subie : baseline {graves_base}, agent {graves_agent}")
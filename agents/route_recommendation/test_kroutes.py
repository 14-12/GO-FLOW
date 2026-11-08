import step5b_agent as a

o, d = "24.11", "359"
routes = a.k_routes([], o, d)
print(len(routes), "itinéraire(s) distinct(s)")
for p in routes:
    s = a.summarize(p, [])
    print(s["routes"], s["temps_formate"], s["distance_metres"], "m")
import step5b_agent as a

o, d = "24.11", "359"
base = a.k_routes([], o, d)[0]
ids = a._edge_ids(base)
alerts = [{"agent": 2, "route": ids[2], "type": "congestion", "penalty": 5.0, "fill": 1.0}]
new = a.k_routes(alerts, o, d)[0]
print("sans alerte :", ids)
print("avec congestion sur", ids[2], ":", a._edge_ids(new))
print("route changée :", a._edge_ids(new) != ids)
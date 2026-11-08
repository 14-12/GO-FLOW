import step5b_agent as a
from adapter_traffic import accident_to_alert

o, d = "24.11", "359"
ids = a._edge_ids(a.k_routes([], o, d)[0])
print("alertes au départ :", a.ALERTS)

a.add_alert(accident_to_alert({"edge": ids[2], "severity": "high",
                               "closed_lanes": 2, "total_lanes": 2}))
print("après accident :", len(a.ALERTS), "alerte(s)")
print("find_routes évite l'edge :",
      ids[2] not in a.find_routes.invoke({"origin": o, "destination": d})["routes"])

a.set_alerts([])
print("après effacement :", a.ALERTS)
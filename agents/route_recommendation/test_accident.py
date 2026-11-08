import step5b_agent as a
from adapter_traffic import accident_to_alert

o, d = "24.11", "359"
ids = a._edge_ids(a.k_routes([], o, d)[0])
msg = {"type": "accident", "t": 150.0, "edge": ids[2],
       "severity": "high", "closed_lanes": 2, "total_lanes": 2}
alert = accident_to_alert(msg)
new = a._edge_ids(a.k_routes([alert], o, d)[0])
print("sans alerte :", ids)
print("accident sur", ids[2], ":", new)
print("évité :", ids[2] not in new, "| pénalité :", alert["penalty"])
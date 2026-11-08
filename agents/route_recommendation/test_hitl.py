import step5b_agent as a
from adapter_traffic import accident_to_alert
import step6_hitl as h
from langgraph.types import Command

o, d = "24.11", "359"
ids = a._edge_ids(a.k_routes([], o, d)[0])
a.add_alert(accident_to_alert({"edge": ids[2], "severity": "high",
                               "closed_lanes": 2, "total_lanes": 2}))
cfg = {"configurable": {"thread_id": "t1"}}
res = h.graph.invoke({"origin": o, "destination": d}, cfg)
for k, v in res["__interrupt__"][0].value["options"].items():
    print(k, v)
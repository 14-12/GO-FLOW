import json, subprocess
import step5b_agent as a
from sumo_map import EDGE_INDEX

raw = subprocess.run(
    ["git", "show", "origin/main:models/goflow/graph/nodes.json"],
    capture_output=True, text=True, encoding="utf-8", cwd=a.NET.parents[2]).stdout
nodes = json.loads(raw)
# mine = {d["edge_id"] for _, _, d in a.G.edges(data=True)}
mine = set(EDGE_INDEX)
ok = [n for n in nodes if n["edge"] in mine]
print("approches :", len(nodes))
print("edge présent dans mon graphe :", len(ok))
print("absents :", [(n["node"], n["edge"]) for n in nodes if n["edge"] not in mine][:10])
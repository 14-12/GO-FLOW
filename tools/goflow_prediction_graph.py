"""The 55 approaches of Town05 and the graph between them, for agents/goflow_prediction and agents/goflow_traffic.

    .venv\\Scripts\\python tools\\goflow_prediction_graph.py [--out out\\goflow_prediction\\graph]

* An approach = one traffic camera (out/town05_cameras/layout.json): the SUMO edge that camera watches coming into
  its junction. Found from the lanes the camera owns with the role "in" for its junction (camera_reports.Layout),
  keeping the edges that end at that junction.
* storage = sum over the edge's lanes of length / 7.5 m (cosim/agents_actuated.VEH_M).
* Graph: approach i -> approach j when traffic that has crossed i's junction can reach j's edge, following the
  network's connections without crossing another signalised junction (priority junctions are passed through);
  cost = road metres from i's junction to the end of j's edge. Written as distance.csv (from,to,cost), the format
  goflow_prediction.data.load_graph reads.

Writes nodes.json ([{node, tls, edge, lanes, storage}]) and distance.csv, into a NEW folder.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VEH_M = 7.5
MAX_M = 1500.0


def approaches(net, layout) -> list[dict]:
    tls_nodes = {t.getID() for t in net.getTrafficLights()}
    out = []
    for c, cam in enumerate(layout.cams):
        name = layout.names[c].split(" ")[0]
        tls = cam.spec.tls
        edges = set()
        for ln in layout.owned(c):
            if ln.startswith(":"):
                continue
            if any(t == tls and role == "in" for t, role in layout.roles.get(ln, [])):
                e = net.getLane(ln).getEdge()
                if e.getToNode().getID() == tls:
                    edges.add(e.getID())
        if len(edges) != 1:
            raise ValueError(f"camera {name}: {len(edges)} approach edges into {tls} ({sorted(edges)})")
        e = net.getEdge(edges.pop())
        lanes = [ln.getID() for ln in e.getLanes() if ln.allows("passenger") or ln.allows("bus") or ln.allows("truck")]
        storage = sum(net.getLane(ln).getLength() for ln in lanes) / VEH_M
        out.append({"node": name, "tls": tls, "edge": e.getID(), "lanes": lanes, "storage": round(storage, 2)})
    assert tls_nodes >= {a["tls"] for a in out}
    edges = [a["edge"] for a in out]
    if len(set(edges)) != len(edges):
        raise ValueError("two cameras on one approach edge")
    return out


def graph(net, appr: list[dict]) -> list[dict]:
    """Shortest road distance from each approach's junction to every approach edge reachable without crossing
    another signalised junction."""
    tls_nodes = {t.getID() for t in net.getTrafficLights()}
    by_edge = {a["edge"]: a["node"] for a in appr}
    rows = []
    for a in appr:
        start = net.getEdge(a["edge"])
        best: dict[str, float] = {}
        heap = [(e.getLength(), e.getID()) for e in start.getOutgoing()]
        heapq.heapify(heap)
        while heap:
            d, eid = heapq.heappop(heap)
            if eid in best or d > MAX_M:
                continue
            best[eid] = d
            if eid in by_edge and by_edge[eid] != a["node"]:
                rows.append({"from": a["node"], "to": by_edge[eid], "cost": round(d, 1)})
                continue          # stop at the next approach: beyond it lies another signalised junction
            e = net.getEdge(eid)
            if e.getToNode().getID() in tls_nodes and eid not in by_edge:
                continue          # into a signal by an edge no camera watches: not followed further
            for nxt in e.getOutgoing():
                if nxt.getID() not in best:
                    heapq.heappush(heap, (d + nxt.getLength(), nxt.getID()))
    return rows


def main(argv=None) -> int:
    import sumolib

    from cosim.camera_reports import Layout

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "goflow_prediction" / "graph"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists: results are never overwritten")
        return 1
    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    layout = Layout.load(ROOT / "out" / "town05_cameras")
    appr = approaches(net, layout)
    rows = graph(net, appr)
    out.mkdir(parents=True)
    (out / "nodes.json").write_text(json.dumps(appr, indent=1), encoding="utf-8")
    with open(out / "distance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["from", "to", "cost"])
        w.writeheader()
        w.writerows(rows)
    lonely = sorted({x["node"] for x in appr} - {r["to"] for r in rows} - {r["from"] for r in rows})
    print(f"{len(appr)} approaches, {len(rows)} edges; storage {min(x['storage'] for x in appr):.1f}-"
          f"{max(x['storage'] for x in appr):.1f} vehicles; with no neighbour: {lonely or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

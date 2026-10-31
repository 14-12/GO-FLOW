"""Bus lines for the Town05 free-fly (public transport): stops for SUMO and the lines' loops.

    python tools/build_town05_pt.py            # writes scenarios/town05/town05.pt.add.xml and town05.pt.json

Three lines across the town, each a closed loop (there and back) on town streets, never on the raised ring
road (the 4 edges over 500 m) or the two 5 km/h edges netconvert made of 791 m and 590 m roads:
  L1 west - east, L2 north - south, L3 south-west - north-east (the extreme edges of the street grid).
A stop every ~150 m of loop on an edge of at least 45 m, 20 m long, half-way along the edge, on the
rightmost lane buses may use. The stops are SUMO busStops (they must exist when SUMO starts); the buses
themselves are added and kept running by cosim/freefly.py (BusLines).
"""

from __future__ import annotations

import heapq
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sumolib  # noqa: E402

SCEN = ROOT / "scenarios" / "town05"
STOP_EVERY_M = 150.0
MIN_EDGE_M = 45.0


def usable(e) -> bool:
    return (not e.getFunction() and e.allows("bus") and e.getLength() <= 500.0 and e.getSpeed() > 5.0)


def path(net, a, b) -> list[str]:
    """Shortest path by length from edge a to edge b over usable edges (Dijkstra)."""
    dist, prev = {a.getID(): 0.0}, {}
    heap = [(0.0, a.getID())]
    while heap:
        d, eid = heapq.heappop(heap)
        if eid == b.getID():
            break
        if d > dist.get(eid, 1e18):
            continue
        e = net.getEdge(eid)
        for nxt, conns in e.getOutgoing().items():
            if not usable(nxt) or not any(c.getFromLane().allows("bus") and c.getToLane().allows("bus") for c in conns):
                continue
            nd = d + nxt.getLength()
            if nd < dist.get(nxt.getID(), 1e18):
                dist[nxt.getID()], prev[nxt.getID()] = nd, eid
                heapq.heappush(heap, (nd, nxt.getID()))
    if b.getID() not in dist:
        return []
    out, cur = [b.getID()], b.getID()
    while cur != a.getID():
        cur = prev[cur]
        out.append(cur)
    return out[::-1]


def centre(e):
    pts = e.getShape()
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def bus_lane(e):
    for lane in e.getLanes():  # index 0 is the rightmost
        if lane.allows("bus") and lane.allows("passenger"):
            return lane
    return None


def main() -> int:
    net = sumolib.net.readNet(str(SCEN / "town05.net.xml"))
    edges = [e for e in net.getEdges() if usable(e) and bus_lane(e) is not None and e.getLength() >= MIN_EDGE_M]
    keys = {"L1": lambda e: centre(e)[0], "L2": lambda e: centre(e)[1],
            "L3": lambda e: centre(e)[0] + centre(e)[1]}
    root = ET.Element("additional")
    lines, used_stop_edges = {}, set()
    for name, key in keys.items():
        ranked = sorted(edges, key=key)
        loop = []
        for a in ranked[:6]:  # the extreme edges; the first pair that gives a closed loop
            for b in ranked[::-1][:6]:
                there, back = path(net, a, b), path(net, b, a)
                if there and back:
                    loop = there + back[1:]
                    break
            if loop:
                break
        if not loop:
            print(f"{name}: no loop found")
            continue
        stops, run, since = [], 0.0, STOP_EVERY_M / 2
        for i, eid in enumerate(loop):
            e = net.getEdge(eid)
            L = e.getLength()
            if 0 < i < len(loop) - 1 and since >= STOP_EVERY_M and L >= MIN_EDGE_M and eid not in used_stop_edges:
                lane = bus_lane(e)
                start = max(10.0, L * 0.5 - 10.0)
                end = min(L - 10.0, start + 20.0)
                sid = f"{name}_S{len(stops) + 1}"
                ET.SubElement(root, "busStop", id=sid, lane=lane.getID(), startPos=f"{start:.1f}", endPos=f"{end:.1f}",
                              name=f"{name} stop {len(stops) + 1}")
                stops.append({"id": sid, "edge": eid, "lane": lane.getID(), "pos": round((start + end) / 2, 1)})
                used_stop_edges.add(eid)
                since = 0.0
            since += L
            run += L
        lines[name] = {"edges": loop, "length_m": round(run), "stops": stops}
        print(f"{name}: {len(loop)} edges, {run:.0f} m, {len(stops)} stops")
    ET.indent(root)
    ET.ElementTree(root).write(SCEN / "town05.pt.add.xml", encoding="UTF-8", xml_declaration=True)
    (SCEN / "town05.pt.json").write_text(json.dumps(lines, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

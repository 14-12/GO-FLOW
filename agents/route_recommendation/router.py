"""Routes on the Town05 SUMO network, road by road, with only the turns the network allows.

The first version (sumo_map.py) made a graph of junctions: one link per pair of junctions, so of two parallel roads
only the faster was kept (ring road 35 and 37, both DSO4 approaches: an alert on them landed on the road next to them),
and a route could need a turn the junction does not have (9 of 1518 suggested routes on Town05, 8 Nov). Here the
graph's points are the roads (SUMO edges) and its links are the network's own connections between their lanes, for
the vehicle class asked (passenger). A route is a list of SUMO edge ids that a car can be given as it is
(traci.vehicle.setRoute).

Costs are free-flow times (length / speed limit) multiplied by the alerts' penalties (adapter_traffic.py). They leave
out the waiting at signals (on Town05 a car's real trip took about 6 times its free-flow time, 8 Nov): a route's time
here is an estimate "without waiting", to be shown as such.

Only sumolib and the standard library: the viewer's own environment can import it (no networkx, no LLM).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NET = REPO / "scenarios" / "town05" / "town05.net.xml"

K_MAX = 3            # options offered at most
MAX_OVERLAP = 0.7    # an alternative shares less than this share of its roads with every option kept
CANDIDATES = 50      # loop-free routes looked at, cheapest first, to find K_MAX different ones


@dataclass
class Road:
    id: str
    frm: str                  # junction it starts at
    to: str                   # junction it ends at
    length: float             # m
    speed: float              # m/s, the limit
    lanes: int                # lanes the vehicle class may use
    shape: list = field(default_factory=list)   # [(x, y)] SUMO coordinates

    @property
    def free_s(self) -> float:
        return self.length / max(self.speed, 0.1)


class Router:
    def __init__(self, net_file: str | Path = NET, vclass: str = "passenger"):
        import sumolib

        net = sumolib.net.readNet(str(net_file))
        self.vclass = vclass
        self.roads: dict[str, Road] = {}
        for e in net.getEdges():
            if e.getFunction() != "" or not e.allows(vclass):
                continue
            lanes = [ln for ln in e.getLanes() if ln.allows(vclass)]
            self.roads[e.getID()] = Road(e.getID(), e.getFromNode().getID(), e.getToNode().getID(), e.getLength(),
                                         e.getSpeed(), len(lanes), [tuple(p) for p in e.getShape()])
        self.next: dict[str, list[str]] = {r: [] for r in self.roads}
        for rid in self.roads:
            e = net.getEdge(rid)
            for to_e, conns in e.getOutgoing().items():
                t = to_e.getID()
                if t in self.roads and any(c.getFromLane().allows(vclass) and c.getToLane().allows(vclass)
                                           for c in conns):
                    self.next[rid].append(t)
        self.junctions = {n.getID(): tuple(n.getCoord()) for n in net.getNodes()}
        self.tls_ids = {tl.getID() for tl in net.getTrafficLights()}
        self.out_of = {j: [] for j in self.junctions}
        self.into = {j: [] for j in self.junctions}
        for r in self.roads.values():
            self.out_of[r.frm].append(r.id)
            self.into[r.to].append(r.id)
        self.typical: dict[tuple[str, str], float] = {}
        self.typical_ratio = 1.0
        self.use_typical = False

    # ------------------------------------------------------------------ typical times (tools/route_delay_table.py)
    def set_typical(self, card: dict, routing: bool = False, min_n: int = 3) -> None:
        """Typical time on each road by the road taken next, the wait at its signal included, from past trips on the
        train seeds (models/route_guidance/movement_times.json). routing: the route search uses them as costs
        (the alerts' penalties multiply them) instead of the free-flow times."""
        import statistics

        self.typical, ratios = {}, []
        for k, v in card["movements"].items():
            a, _, b = k.partition(">")
            if v["n"] >= min_n and a in self.roads:
                self.typical[(a, b)] = float(v["mean_s"])
                ratios.append(v["mean_s"] / max(self.roads[a].free_s, 0.1))
        self.typical_ratio = statistics.median(ratios) if ratios else 1.0
        self.use_typical = routing and bool(self.typical)

    def move_s(self, a: str, b: str) -> float:
        """Typical time on road a when leaving it for road b ("" = the trip ends on it)."""
        t = self.typical.get((a, b))
        if t is None:
            t = self.roads[a].free_s * self.typical_ratio
        return t

    def typical_s(self, route: list[str], skip_first: bool = True) -> float:
        """The route's typical time, signals included. The first road counts when skip_first is False (a trip from
        its start), not when the car is on it already."""
        if not self.typical:
            return self.free_s(route, skip_first)
        nxt = list(route[1:]) + [""]
        return sum(self.move_s(a, b) for i, (a, b) in enumerate(zip(route, nxt)) if i > 0 or not skip_first)

    # ------------------------------------------------------------------ costs
    @staticmethod
    def penalties(alerts) -> dict[str, float]:
        """road -> cost multiplier: the product of the penalties of the alerts on it (adapter_traffic.py)."""
        pen: dict[str, float] = {}
        for a in alerts or ():
            r, p = a.get("route"), a.get("penalty")
            if r is not None and p is not None:
                pen[r] = pen.get(r, 1.0) * float(p)
        return pen

    def cost(self, rid: str, pen: dict) -> float:
        return self.roads[rid].free_s * pen.get(rid, 1.0)

    def route_cost(self, route: list[str], pen: dict | None = None, skip_first: bool = True) -> float:
        """Sum of the roads' costs. skip_first: the road the car is on is the same for every option (not counted).
        With typical times (use_typical): each road's time by the road taken next, the first road's included (where
        it turns decides which queue it waits in), times the road's penalty."""
        pen = pen or {}
        if self.use_typical:
            nxt = list(route[1:]) + [""]
            return sum(self.move_s(a, b) * pen.get(a, 1.0) for a, b in zip(route, nxt))
        return sum(self.cost(r, pen) for r in route[(1 if skip_first else 0):])

    def link_cost(self, a: str, b: str, pen: dict) -> float:
        """The search's cost of going on from road a into road b."""
        if self.use_typical:
            return self.move_s(a, b) * pen.get(a, 1.0)
        return self.cost(b, pen)

    def free_s(self, route: list[str], skip_first: bool = True) -> float:
        """Free-flow time (length / limit), whatever the costs are."""
        return sum(self.roads[r].free_s for r in route[(1 if skip_first else 0):])

    def length_m(self, route: list[str], skip_first: bool = True) -> float:
        return sum(self.roads[r].length for r in route[(1 if skip_first else 0):])

    def drivable(self, route: list[str]) -> bool:
        """Every road known and each next road reachable through a connection."""
        return all(r in self.roads for r in route) and all(b in self.next[a] for a, b in itertools.pairwise(route))

    # ------------------------------------------------------------------ search
    def _dijkstra(self, starts: list[str], goals: set[str], pen: dict, banned_roads=frozenset(),
                  banned_links=frozenset()) -> list[str] | None:
        """Cheapest route from one of `starts` (its cost not counted) to one of `goals` (counted)."""
        best: dict[str, float] = {}
        prev: dict[str, str | None] = {}
        heap = []
        tie = itertools.count()
        for s in starts:
            if s in banned_roads or s not in self.roads:
                continue
            best[s], prev[s] = 0.0, None
            heap.append((0.0, next(tie), s))
        heapq.heapify(heap)
        while heap:
            c, _, r = heapq.heappop(heap)
            if c > best.get(r, float("inf")):
                continue
            if r in goals:
                out = [r]
                while prev[out[-1]] is not None:
                    out.append(prev[out[-1]])
                return out[::-1]
            for n in self.next[r]:
                if n in banned_roads or (r, n) in banned_links:
                    continue
                nc = c + self.link_cost(r, n, pen)
                if nc < best.get(n, float("inf")) - 1e-9:
                    best[n], prev[n] = nc, r
                    heapq.heappush(heap, (nc, next(tie), n))
        return None

    def best(self, frm: str, to: str, alerts=None) -> list[str] | None:
        """Cheapest route from road `frm` (where the car is) to road `to`, both included."""
        if frm == to:
            return [frm]
        return self._dijkstra([frm], {to}, self.penalties(alerts))

    def simple_routes(self, frm: str, to: str, pen: dict, limit: int = CANDIDATES):
        """Loop-free routes from road frm to road to, cheapest first (Yen's algorithm)."""
        first = self._dijkstra([frm], {to}, pen)
        if first is None:
            return
        found = [first]
        yield first
        cands: list = []
        seen = {tuple(first)}
        tie = itertools.count()
        while len(found) < limit:
            last = found[-1]
            for i in range(len(last) - 1):
                root = last[:i + 1]
                links = {(p[i], p[i + 1]) for p in found if len(p) > i + 1 and p[:i + 1] == root}
                spur = self._dijkstra([root[-1]], {to}, pen, banned_roads=frozenset(root[:-1]),
                                      banned_links=frozenset(links))
                if spur is None:
                    continue
                path = root[:-1] + spur
                key = tuple(path)
                if key in seen:
                    continue
                seen.add(key)
                heapq.heappush(cands, (self.route_cost(path, pen), next(tie), path))
            if not cands:
                return
            _, _, path = heapq.heappop(cands)
            found.append(path)
            yield path

    @staticmethod
    def _inner(route: list[str]) -> set[str]:
        """The roads that can differ between two routes with the same start and end."""
        return set(route[1:-1]) if len(route) > 2 else set(route)

    def k_routes(self, frm: str, to: str, alerts=None, k_max: int = K_MAX, max_overlap: float = MAX_OVERLAP):
        """Up to k_max routes from road frm to road to that differ: each shares under max_overlap of its roads
        (between the start and the end) with every one kept before it. Cheapest (with the alerts' penalties) first."""
        pen = self.penalties(alerts)
        kept: list[list[str]] = []
        for path in self.simple_routes(frm, to, pen):
            ids = self._inner(path)
            if all(len(ids & self._inner(p)) / max(len(ids), 1) < max_overlap for p in kept):
                kept.append(path)
            if len(kept) == k_max:
                break
        return kept

    # ------------------------------------------------------------------ junction to junction (the Streamlit page)
    def best_between(self, j_from: str, j_to: str, alerts=None) -> list[str] | None:
        """Cheapest route leaving junction j_from and arriving at junction j_to (every road counted)."""
        pen = self.penalties(alerts)
        best, best_c = None, float("inf")
        for s in self.out_of.get(j_from, ()):
            p = self._dijkstra([s], set(self.into.get(j_to, ())), pen)
            if p is not None:
                c = self.route_cost(p, pen, skip_first=False)
                if c < best_c:
                    best, best_c = p, c
        return best

    def k_between(self, j_from: str, j_to: str, alerts=None, k_max: int = K_MAX, max_overlap: float = MAX_OVERLAP):
        """Up to k_max different routes from junction j_from to junction j_to (all roads counted)."""
        pen = self.penalties(alerts)
        cands = []
        for s in self.out_of.get(j_from, ()):
            for e in self.into.get(j_to, ()):
                for p in itertools.islice(self.simple_routes(s, e, pen), 10):
                    cands.append((self.route_cost(p, pen, skip_first=False), p))
        cands.sort(key=lambda x: x[0])
        kept: list[list[str]] = []
        seen = set()
        for _, p in cands:
            if tuple(p) in seen:
                continue
            seen.add(tuple(p))
            ids = set(p)
            if all(len(ids & set(q)) / max(len(ids), 1) < max_overlap for q in kept):
                kept.append(p)
            if len(kept) == k_max:
                break
        return kept

    # ------------------------------------------------------------------ what a route meets
    @staticmethod
    def on_route(route: list[str], alerts) -> list[dict]:
        ids = set(route)
        return [a for a in alerts or () if a.get("route") in ids]

    @staticmethod
    def avoided(reference: list[str], route: list[str], alerts) -> list[dict]:
        """The alerts on the reference route that this route does not go through."""
        gone = set(reference) - set(route)
        return [a for a in alerts or () if a.get("route") in gone]

    def summary(self, route: list[str], alerts=None, skip_first: bool = True) -> dict:
        pen = self.penalties(alerts)
        free = self.free_s(route, skip_first)
        m, s = divmod(round(free), 60)
        return {"routes": list(route), "temps_libre_secondes": round(free, 1), "temps_formate": f"{m} min {s} s",
                "cout_avec_alertes_secondes": round(self.route_cost(route, pen, skip_first), 1),
                "distance_metres": round(self.length_m(route, skip_first), 1),
                "alertes_rencontrees": self.on_route(route, alerts)}

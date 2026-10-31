"""Signal plans in which every green is a free "go": nothing crossing the path of a green lane is green with it, cars
or walkers (27 Oct).

The 25 Oct programs (netconvert's "opposites") let left and right turns go on the same green as the traffic or the
people crossing their path, giving way ("g", permissive). Cars entered the junction on green and stopped inside it
for oncoming cars or for walkers. Here:

* the unit is the lane: a lane is green only when nothing in conflict with any of its directions is green, by SUMO's
  own junction geometry (`foes`), so a car that has its green never stops inside the junction. A lane's directions
  go together: a direction of its own (the straight-on arrow green while the left arrow of the same lane stays red)
  leaves the first car of the lane, turning left, standing at the line in front of the cars behind it (tried first,
  27 Oct morning: at x0.8 the town locked);
* the greens are chosen from the sets of lanes that may go together (`choose_greens`): as few as cover every lane
  (each green costs a yellow and an all red), each lane in as many as possible. On Town05's two-lane approaches
  (left lane straight on + left, right lane straight on + right) that is each approach on its own; with SIGNAL_GREENS
  "all", lead-lag (N alone, the two right lanes N+S together with the left lanes red, S alone); with left-turn lanes
  (tools/build_town05_signals.py LEFT_LANES "dedicated"), N+S straight on and right, then N+S left. Measured with the
  junction agents (tools/check_junction_stops.py, README "Signals where no green gives way"): each approach on its
  own keeps the town moving furthest; lead-lag costs a transition more per road; a left-turn lane leaves straight on
  one lane and the town locks at x0.8;
* pedestrians cross on a walk of their own, every car red, all crossings at once, since on Town05 every lane carries
  a direction that crosses a crosswalk (the one at its own stop line, or the one it turns over). A crossing that no
  lane of a green crosses gets its walk with that green. The walk is skipped when nobody waits
  (sumo_side.WalkSkip, the agents' skip), its clearance ends once the crossings are empty, and nobody steps on at
  DON'T WALK (sumo_side.PedRedHold).

After the walk, its flashing DON'T WALK ("u" on the crossings: red to SUMO's walkers, flashing on the lamps), and 2 s
of steady DON'T WALK before the cars. A lane green in two following phases stays green between them; a lane that
ends shows yellow (ITE, from its speed), then all red (from the junction's width) when what starts next conflicts
with it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

BUFFER_S = 2.0
GREENS = os.environ.get("SIGNAL_GREENS", "min")  # "min": as few greens as cover every lane; "all": lead-lag


@dataclass
class Phase:
    greens: frozenset[int]            # link indices green
    name: str                         # "green" | "walk" | "walk clearance" | "walk buffer"
    duration: float


# "approach" (29 Oct, the default): all the lanes of an approach go together, so every head of its CARLA pole shows
# the same colour and CARLA's own lamps can show it (a CARLA traffic light has one state for its four heads); "lane":
# a lane's directions go together; "direction": each its own
UNIT = os.environ.get("SIGNAL_UNIT", "approach")


def lanes_of(links: list) -> dict[str, list[int]]:
    """The units that get a signal: lanes (all their directions together), or with SIGNAL_UNIT "direction" each
    direction of each lane (the left arrow of a straight-and-left lane red while its straight on is green)."""
    out: dict[str, list[int]] = {}
    for lk in links:
        if lk.kind == "vehicle":
            key = (lk.from_lane if UNIT == "lane" else lk.approach if UNIT == "approach"
                   else f"{lk.from_lane}|{lk.dir}")
            out.setdefault(key, []).append(lk.index)
    return out


def _maximal_sets(names: list[str], compatible) -> list[frozenset[str]]:
    """Every maximal set of lanes that may be green together (Bron-Kerbosch; Town05 has at most 10 lanes)."""
    out: list[frozenset[str]] = []

    def bk(r: set, p: set, x: set) -> None:
        if not p and not x:
            out.append(frozenset(r))
            return
        for v in list(p):
            nb = {u for u in names if u != v and compatible(u, v)}
            bk(r | {v}, p & nb, x & nb)
            p = p - {v}
            x = x | {v}

    bk(set(), set(names), set())
    return out


def choose_greens(lanes: dict, app: dict, ok, lane_order, parking) -> list[frozenset[int]]:
    """The greens of the cycle: as few as cover every lane (each green is a transition's lost time: yellow and
    all red), among those the ones that give the lanes the most green (a lane in two greens has more), in the order
    that keeps the most lanes green from one to the next (a lane green in both stays green between them). Each
    green: a maximal set of lanes that may go together."""
    import itertools

    names = sorted(lanes)

    def compatible(a: str, b: str) -> bool:
        return ok(lanes[a], set(lanes[b]))

    sets = _maximal_sets(names, compatible)
    best = None
    if GREENS == "all" and len(sets) <= 8:
        # lead-lag (SIGNAL_GREENS=all): every maximal set its green, e.g. on shared lanes N alone, then the two right
        # lanes N+S together (the left lanes red: the oncoming traffic), then S alone. More lane green, more yellow
        # and all red (one more transition per road)
        best = ((0, 0, 0), tuple(sets))
    for k in range(1, len(sets) + 1):
        if best is not None:
            break
        for combo in itertools.combinations(sets, k):
            if set().union(*combo) != set(names):
                continue
            # the lanes' green: how many greens each (non-parking) lane is in
            score = (sum(1 for g in combo for ln in g if not parking(ln)),
                     # then greens of one road (N-S or E-W): what drivers expect, and the walks of the other road
                     sum(1 for g in combo if len({"NS" if app[ln] in "NS" else "EW" for ln in g}) == 1),
                     # then opposite approaches together (N+S straight on, then N+S left: each head of a pole its
                     # own green) rather than one approach at a time, which gives the lanes as much green
                     sum(len({app[ln] for ln in g}) for g in combo))
            if best is None or score > best[0]:
                best = (score, combo)
        if best is not None:
            break
    combo = list(best[1])

    def axis_first(g: frozenset[str]) -> tuple:
        sides = {app[ln] for ln in g}
        return (0 if sides & set("NS") else 1, "".join(sorted(sides)))

    first = min(combo, key=axis_first)
    rest = [g for g in combo if g is not first]
    best_order, best_keep = None, -1
    for perm in itertools.permutations(rest):
        order = [first, *perm]
        keep = sum(len(order[i] & order[(i + 1) % len(order)]) for i in range(len(order)))
        axis_runs = sum(axis_first(order[i])[0] != axis_first(order[(i + 1) % len(order)])[0] for i in range(len(order)))
        key = (keep, -axis_runs)
        if best_order is None or key > best_keep:
            best_order, best_keep = order, key
    return [frozenset(i for ln in g for i in lanes[ln]) for g in best_order]


def plan(links: list, conflict, timing: dict) -> list[tuple[str, float, float | None, float | None, str]]:
    """links: cosim.signal_audit.Link list; conflict(i, j): SUMO foes; timing: {"yellow": f(link)->s,
    "allred": f(link)->s, "walk_speed": m/s, "clear_speed": m/s}. Returns [(state, duration, minDur, maxDur, name)]
    (minDur/maxDur None: tools/build_town05_signals.actuated sets them)."""
    n = len(links)
    lanes = lanes_of(links)
    app = {ln: links[idx[0]].approach for ln, idx in lanes.items()}
    cross = [lk.index for lk in links if lk.kind == "crossing"]

    def ok(group: list[int], members: set[int]) -> bool:
        return not any(conflict(i, j) for i in group for j in members if i != j)

    def parking(ln: str) -> bool:
        lk = links[lanes[ln][0]]
        return lk.lane_type == "parking" or lk.speed < 5.0

    def lane_order(ln: str) -> tuple:
        idx = lanes[ln]
        return (sum(1 for i in idx for j in range(n) if conflict(i, j)), ln)

    phases = [Phase(g, "green", 15.0) for g in choose_greens(lanes, app, ok, lane_order, parking)]
    # crossings: with a green no lane of which crosses them; else on the walk of their own
    walked: set[int] = set()
    out_phases: list[Phase] = []
    for ph in phases:
        walks = [c for c in cross if ok([c], set(ph.greens))]
        if walks:
            ln = max(links[c].length for c in walks)
            fdw = max(5.0, math.ceil(max(ln / timing["walk_speed"], ln / timing["clear_speed"]) * 2) / 2)
            out_phases.append(Phase(ph.greens | frozenset(walks), "green", max(ph.duration, 10.0)))
            out_phases.append(Phase(ph.greens, "walk clearance", fdw))
            walked |= set(walks)
        else:
            out_phases.append(ph)
    rest = [c for c in cross if c not in walked]
    if rest:
        ln = max(links[c].length for c in rest)
        fdw = max(5.0, math.ceil(max(ln / timing["walk_speed"], ln / timing["clear_speed"] - BUFFER_S) * 2) / 2)
        out_phases.append(Phase(frozenset(rest), "walk", 7.0))
        out_phases.append(Phase(frozenset(), "walk clearance", fdw))
        out_phases.append(Phase(frozenset(), "walk buffer", BUFFER_S))
    # states with the transitions: yellow for the lanes that end, then all red if what starts conflicts with them
    out: list[tuple[str, float, float | None, float | None, str]] = []

    def state(greens, flashing=()) -> str:
        return "".join("G" if i in greens else ("u" if i in flashing else "r") for i in range(n))

    seq = out_phases
    for k, ph in enumerate(seq):
        # the crossings whose walk just ended show "u" in their clearance: red to SUMO's walkers (nobody starts),
        # flashing DON'T WALK on the lamps (cosim/ped_lights.ped_aspect), told apart from an all red that way
        fl = [c for c in cross if c in seq[k - 1].greens and c not in ph.greens] if ph.name == "walk clearance" else []
        out.append((state(ph.greens, fl), ph.duration, None, None, ph.name))
        nxt = seq[(k + 1) % len(seq)]
        ending_veh = [i for i in ph.greens if i not in nxt.greens and links[i].kind == "vehicle"]
        if not ending_veh:
            if ph.name == "walk clearance" and ph.greens:  # cars go on: before what crosses the walk starts, the
                walked_now = [c for c in cross if c in seq[k - 1].greens]  # steady don't walk (MUTCD buffer)
                starting = [i for i in nxt.greens if i not in ph.greens]
                if any(conflict(c, j) for c in walked_now for j in starting):
                    out.append((state(ph.greens), BUFFER_S, None, None, "walk buffer"))
            continue
        keep = ph.greens & nxt.greens
        y = max(timing["yellow"](i) for i in ending_veh)
        out.append(("".join("G" if i in keep else ("y" if i in ending_veh else "r") for i in range(n)), y, None, None,
                    "yellow"))
        starting = [i for i in nxt.greens if i not in ph.greens]
        if any(conflict(i, j) for i in ending_veh for j in starting):
            ar = max(timing["allred"](i) for i in ending_veh)
            out.append(("".join("G" if i in keep else "r" for i in range(n)), ar, None, None, "all red"))
    return out

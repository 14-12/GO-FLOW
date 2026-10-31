"""Put every movement of a signalised junction under its signal.

CARLA's netconvert_carla.py controls only the lanes an OpenDRIVE signal references. On CARLA's
own towns that leaves movements at signalised junctions with no signal at all: SUMO gives them
permanent right of way (state "M"), so a stream crosses the junction through every red phase of
the others. Measured (SUMO 1.27.1, CARLA 0.9.16 nets): Town12 2140 of 5094 movements at its 320
signalised junctions, Town10HD 24 of 80; the Ingolstadt and Cologne nets, made by netconvert from
SUMO networks, have none. On Town12 the queues that grow behind these junctions locked the city
centre within 20 minutes (tools/diag_town12.py).

Each uncontrolled movement is added to its junction's signal (a new link index) and, in every
phase, takes the signal of a "donor" movement:

1. a controlled movement from the same lane-group (same incoming edge, same direction) - the
   lane beside it on the same arrow: its state is copied as it is;
2. otherwise a controlled movement from the same incoming edge;
3. otherwise one from the approach most nearly opposite (opposite approaches share their green
   in the programs netconvert_carla writes).

From 2 and 3 a green is given as "g" (go, but yield to conflicting traffic as the junction's
right-of-way rules say), never "G", so two greens can never claim the same space. The network's
geometry, junction logic and the linkSignalID parameters (tlmap.py) are left as they are; the
CARLA light poles keep showing the links they showed, and a new link moves with the lane next
to it, so what CARLA shows still matches.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_CONN = re.compile(r'<connection (?P<attrs>[^>]*?)(?P<end>/?>)')
_ATTR = re.compile(r'(\w+)="([^"]*)"')
DIR_ORDER = {"s": 0, "r": 1, "l": 2, "t": 3, "R": 4, "L": 5}


def _heading(shape: str) -> float:
    pts = [tuple(map(float, p.split(",")[:2])) for p in shape.split()]
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return math.atan2(y1 - y0, x1 - x0)


def _angle(a: float, b: float) -> float:
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def complete(net_in: Path, net_out: Path, details: list | None = None) -> dict:
    """Rewrite net_in to net_out with every movement at a signalised junction controlled.
    Returns counts (per donor rule) for the build report. If `details` is a list, one
    (internal lane, tls id, donor link index, rule) per added link is appended to it."""
    text = net_in.read_text(encoding="utf-8")
    tls_ids = set(re.findall(r'<tlLogic id="([^"]*)"', text))
    lane_shape = dict(re.findall(r'<lane id="([^"]*)"[^>]*?shape="([^"]*)"', text))

    conns = []  # (match span, attrs dict)
    for m in _CONN.finditer(text):
        a = dict(_ATTR.findall(m.group("attrs")))
        via = a.get("via", "")
        if not a.get("from", "").startswith(":") and via.startswith(":"):
            j = via[1:].rsplit("_", 2)[0]
            if j in tls_ids:
                conns.append((m, a, j))

    by_tls: dict[str, list] = {}
    for m, a, j in conns:
        by_tls.setdefault(j, []).append((m, a))

    edits = []  # (start, end, replacement)
    added: dict[str, list[tuple[int, str, bool]]] = {}  # tls -> [(donor index, rule, exact)]
    stats = Counter()
    for j, items in by_tls.items():
        ctl = [(m, a) for m, a in items if "tl" in a]
        if not ctl:
            stats["junctions_without_any_signal"] += 1
            continue
        heading = {}
        for _, a in items:
            lane = f'{a["from"]}_{a["fromLane"]}'
            if lane in lane_shape:
                heading.setdefault(a["from"], _heading(lane_shape[lane]))
        n_links = 1 + max(int(a["linkIndex"]) for _, a in ctl)
        for m, a in items:
            if "tl" in a:
                continue
            same_edge = [c for c in ctl if c[1]["from"] == a["from"]]
            same_dir = [c for c in same_edge if c[1].get("dir") == a.get("dir")]
            if same_dir:
                donor, rule = min(same_dir, key=lambda c: abs(int(c[1]["fromLane"]) - int(a["fromLane"]))), "same_lane_group"
            elif same_edge:
                donor, rule = min(same_edge, key=lambda c: DIR_ORDER.get(c[1].get("dir"), 9)), "same_approach"
            else:
                h = heading.get(a["from"])
                cand = [c for c in ctl if c[1]["from"] in heading]
                if h is None or not cand:
                    stats["left_uncontrolled"] += 1
                    continue
                donor = min(cand, key=lambda c: (abs(_angle(heading[c[1]["from"]], h) - math.pi),
                                                 c[1].get("dir") != a.get("dir")))
                rule = "opposite_approach"
            idx = n_links + len(added.get(j, []))
            added.setdefault(j, []).append((int(donor[1]["linkIndex"]), rule, rule == "same_lane_group"))
            stats[rule] += 1
            if details is not None:
                details.append((a["via"], j, int(donor[1]["linkIndex"]), rule))
            attrs = m.group("attrs") + f' tl="{j}" linkIndex="{idx}"'
            edits.append((m.start(), m.end(), f'<connection {attrs}{m.group("end")}'))

    # phase states: one more character per added link, from its donor
    def extend(state: str, extra: list[tuple[int, str, bool]]) -> str:
        out = []
        for donor, _, exact in extra:
            ch = state[donor]
            out.append(ch if exact or ch != "G" else "g")
        return state + "".join(out)

    for tl in re.finditer(r'<tlLogic id="([^"]*)".*?</tlLogic>', text, re.S):
        extra = added.get(tl.group(1))
        if not extra:
            continue
        body = re.sub(r'(<phase [^>]*?state=")([^"]*)(")',
                      lambda p: p.group(1) + extend(p.group(2), extra) + p.group(3), tl.group(0))
        edits.append((tl.start(), tl.end(), body))

    edits.sort()
    parts, pos = [], 0
    for s, e, rep in edits:
        assert s >= pos, "overlapping edits"
        parts.append(text[pos:s])
        parts.append(rep)
        pos = e
    parts.append(text[pos:])
    net_out.write_text("".join(parts), encoding="utf-8", newline="\n")
    stats["signalised_junctions"] = len(by_tls)
    stats["movements"] = sum(len(v) for v in by_tls.values())
    stats["were_controlled"] = sum(1 for v in by_tls.values() for _, a in v if "tl" in a)
    return dict(stats)

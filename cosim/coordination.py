"""Coordinated signal timing for Town05: one cycle for every signal, offsets that make green waves (27 Oct).

Town05's 15 signals stand 23-65 m apart (2-5 s at 50 km/h). Run on their own (actuated, the junction agents) each
kept its own short cycle: greens of 4-6 s, a car leaving one junction on green met red at the next. Here every
signal runs the same fixed cycle (`plan`):

* the cycle: Webster's (1.5 L + 5) / (1 - Y) for the busiest junction (L: its lost time, yellows, all reds and the
  walk; Y: the sum over its greens of the critical lane flow / 1800 veh/h, from the routes at the design demand),
  rounded up to 5 s, at least MIN_CYCLE_S;
* each junction's greens share what the cycle leaves after its yellows, all reds and walk, in proportion to their
  critical lane flows, each at least MIN_GREEN_S (a green with straight on) or MIN_TURN_GREEN_S (turns only): no more
  greens of 4-5 s;
* the offsets (where each signal's cycle starts) put the greens one after the other along the way the traffic goes
  (`offsets`): for every pair of signals one after the other on the vehicles' routes, the share of the cars leaving
  the first on green that reach the second on green, weighted by how many make that trip, is made as large as it
  goes (coordinate search over the offsets from several starts, 1 s steps): a green wave on the busy ways, both
  directions where the short distances allow.

In the simulation (`sumo_side.CycleKeeper`, controller "coordinated") the programs run as coordinated-actuated
controllers do: a green other than the first ends early once it has had its minimum and nobody is coming; a walk
nobody waits for is skipped (`sumo_side.WalkSkip`); what they leave over goes to the first green (the busiest, the
one the wave is built on), which always ends at its planned second of the cycle, so the signals stay in step.
"""

from __future__ import annotations

import itertools
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SAT = 1800.0            # veh/h per lane of green
MIN_CYCLE_S = float(os.environ.get("COORD_MIN_CYCLE", "90"))
MAX_CYCLE_S = 120.0
FORCE_CYCLE_S = float(os.environ.get("COORD_CYCLE", "0"))   # a cycle given (experiments); 0: Webster's
MIN_GREEN_S = float(os.environ.get("COORD_MIN_GREEN", "12"))      # a green with straight-on movements
MIN_TURN_GREEN_S = float(os.environ.get("COORD_MIN_TURN", "7"))   # a green of turns only (a protected left)
SPEED_FACTOR = 0.85     # platoons travel at this share of the limit


@dataclass
class Signal:
    tls: str
    phases: list[list]                    # [state, duration, name]
    greens: list[int] = field(default_factory=list)   # indices of car greens
    offset: float = 0.0


def car_green(jn, state: str, name: str = "green") -> bool:
    """A green of the program: a phase named "green" with cars green (the yellows and all reds after a green keep
    the lanes that go on green, but they are transitions, not greens to stretch)."""
    return name == "green" and any(state[lk.index] in "Gg" for lk in jn.links if lk.kind == "vehicle")


def lane_flows(jn, flows: dict[int, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lk in jn.links:
        if lk.kind == "vehicle":
            out[lk.from_lane] = out.get(lk.from_lane, 0.0) + flows.get(lk.index, 0.0)
    return out


def green_ratios(jn, phases, flows: dict[int, float]) -> dict[int, float]:
    """Each car green's critical flow ratio: the busiest lane it serves (a lane served by two greens counts half in
    each)."""
    lf = lane_flows(jn, flows)
    greens = [k for k, (s, _d, n) in enumerate(phases) if car_green(jn, s, n)]
    idx = {ln: [lk.index for lk in jn.links if lk.kind == "vehicle" and lk.from_lane == ln] for ln in lf}
    out = {}
    for k in greens:
        s = phases[k][0]
        ys = []
        for ln, f in lf.items():
            if any(s[i] in "Gg" for i in idx[ln]):
                n = sum(1 for g in greens if any(phases[g][0][i] in "Gg" for i in idx[ln]))
                ys.append(f / SAT / n)
        out[k] = max(ys, default=0.0)
    return out


def min_green(jn, state: str) -> float:
    straight = any(state[lk.index] in "Gg" and lk.dir == "s" for lk in jn.links if lk.kind == "vehicle")
    return MIN_GREEN_S if straight else MIN_TURN_GREEN_S


def cycle_length(js: dict, programs: dict[str, list[list]], flows: dict, scale: float = 1.0) -> float:
    need = MIN_CYCLE_S
    for tid, phases in programs.items():
        jn = js[tid]
        ratios = green_ratios(jn, phases, flows.get(tid, {}))
        fixed = sum(d for k, (s, d, _n) in enumerate(phases) if k not in ratios)
        y = sum(ratios.values()) * scale
        webster = (1.5 * fixed + 5.0) / max(0.05, 1.0 - y)
        floor = fixed + sum(min_green(jn, phases[k][0]) for k in ratios)
        need = max(need, webster, floor)
    if FORCE_CYCLE_S > 0:
        return FORCE_CYCLE_S
    return min(max(MIN_CYCLE_S, math.ceil(need / 5.0) * 5.0), max(MAX_CYCLE_S, math.ceil(need / 5.0) * 5.0))


def splits(jn, phases: list[list], flows: dict[int, float], cycle: float) -> list[list]:
    """The phases with their greens stretched to fill the cycle: each at least its minimum, the rest by flow."""
    ratios = green_ratios(jn, phases, flows)
    fixed = sum(d for k, (s, d, _n) in enumerate(phases) if k not in ratios)
    mins = {k: min_green(jn, phases[k][0]) for k in ratios}
    spare = cycle - fixed - sum(mins.values())
    tot = sum(ratios.values())
    out = [list(p) for p in phases]
    for k in ratios:
        share = (ratios[k] / tot) if tot > 0 else 1.0 / len(ratios)
        out[k][1] = mins[k] + spare * share
    # whole seconds, the rounding left to the first green
    for k in ratios:
        out[k][1] = float(math.floor(out[k][1]))
    gap = cycle - sum(p[1] for p in out)
    if ratios:
        out[min(ratios, key=lambda k: -ratios[k])][1] += gap
    return out


# ---------------------------------------------------------------------- trips between signals
def signal_pairs(net_file: Path, route_files: list[Path]) -> dict[tuple, float]:
    """(tls a, link at a, tls b, link at b) -> (vehicles/h making that step, travel s from a's line to b's line)."""
    import sumolib

    net = sumolib.net.readNet(str(net_file), withInternal=False)
    conns: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "connection" and el.get("tl"):
            conns.setdefault((el.get("from"), el.get("to")), []).append((el.get("tl"), int(el.get("linkIndex"))))
    length, speed = {}, {}
    for e in net.getEdges():
        length[e.getID()] = e.getLength()
        speed[e.getID()] = e.getSpeed()
    out: dict[tuple, list[float]] = {}
    for rf in route_files:
        for _, el in ET.iterparse(str(rf), events=("end",)):
            if el.tag != "vehicle":
                continue
            route = el.find("route")
            edges = route.get("edges", "").split() if route is not None else []
            seq = []  # (tls, links, the edge index after the signal)
            for k, (a, b) in enumerate(itertools.pairwise(edges)):
                links = conns.get((a, b))
                if links:
                    seq.append((links[0][0], tuple(sorted(i for _, i in links)), k + 1))
            for (ta, la, ka), (tb, lb, kb) in itertools.pairwise(seq):
                t = sum(length.get(e, 0.0) / max(1.0, speed.get(e, 13.89) * SPEED_FACTOR) for e in edges[ka:kb])
                if t > 30.0:
                    continue  # too far for a platoon to hold together (the ring road between 2014 and 2328)
                rec = out.setdefault((ta, la, tb, lb), [0.0, t])
                rec[0] += 1.0
            el.clear()
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---------------------------------------------------------------------- offsets
def _green_mask(phases: list[list], links: tuple[int, ...], cycle: int) -> list[bool]:
    mask = [False] * cycle
    t = 0.0
    for s, d, _n in phases:
        on = any(s[i] in "Gg" for i in links if i < len(s))
        for u in range(int(round(t)), int(round(t + d))):
            mask[u % cycle] = on
        t += d
    return mask


def offsets(signals: dict[str, Signal], pairs: dict[tuple, tuple[float, float]], cycle: float, seed: int = 1,
            starts: int = 12, step: int = 1) -> tuple[dict[str, float], float, float]:
    """Offsets maximising the platoons that arrive on green (see the module's docstring). Returns (offsets, score
    with them, score with every offset 0), scores as the share of the vehicles making a signal-to-signal step on a
    green that arrive on green."""
    import numpy as np

    C = int(round(cycle))
    tids = sorted(signals)
    masks: dict[tuple[str, tuple], np.ndarray] = {}

    def mask(tid, links):
        key = (tid, links)
        if key not in masks:
            masks[key] = np.array(_green_mask(signals[tid].phases, links, C), dtype=float)
        return masks[key]

    terms = []  # (a, b, weight, depart profile at a (rolled by the travel time), green at b)
    total = 0.0
    for (ta, la, tb, lb), (n, t) in pairs.items():
        if ta not in signals or tb not in signals or ta == tb:
            continue
        dep = mask(ta, la)
        if dep.sum() == 0:
            continue
        arr = np.roll(dep, int(round(t))) / dep.sum()
        terms.append((ta, tb, n, arr, mask(tb, lb)))
        total += n
    by_sig: dict[str, list[int]] = {t: [] for t in tids}
    for k, (a, b, *_r) in enumerate(terms):
        by_sig[a].append(k)
        by_sig[b].append(k)

    def term_score(k, off):
        a, b, n, arr, grn = terms[k]
        shift = int(round(off[b] - off[a])) % C  # b's cycle starts later by this: its green mask moves right
        return n * float(np.dot(arr, np.roll(grn, shift)))

    def score(off):
        return sum(term_score(k, off) for k in range(len(terms)))

    rng = random.Random(seed)
    base = {t: 0.0 for t in tids}
    best, best_s = dict(base), score(base)
    s0 = best_s
    for r in range(starts):
        off = dict(base) if r == 0 else {t: float(rng.randrange(0, C, step)) for t in tids}
        cur = score(off)
        for _ in range(8):
            improved = False
            for t in tids:
                ks = by_sig[t]
                if not ks:
                    continue
                here = sum(term_score(k, off) for k in ks)
                keep = off[t]
                best_o, best_v = keep, here
                for o in range(0, C, step):
                    off[t] = float(o)
                    v = sum(term_score(k, off) for k in ks)
                    if v > best_v + 1e-9:
                        best_o, best_v = float(o), v
                off[t] = best_o
                if best_o != keep:
                    cur += best_v - here
                    improved = True
            if not improved:
                break
        if cur > best_s:
            best, best_s = dict(off), cur
    return best, best_s / max(total, 1e-9), s0 / max(total, 1e-9)


COORD_TYPE = os.environ.get("COORD_TYPE", "static")  # "actuated": coordinated-actuated; "static": fixed time


def write(signals: dict[str, Signal], cycle: float, out: Path, program_id: str = "coordinated",
          note: str = "", js: dict | None = None) -> None:
    """The programs. COORD_TYPE "actuated" (SUMO's coordinated actuated mode, param coordinated=true): the cycle and
    the offsets are kept by SUMO itself; each green may end early on its loops once it has had its minimum (no one
    coming), but never after its planned end (latestEnd), and the first green (phase 0, the one the green wave of
    that signal is built on) never before its planned end (earliestEnd): what the other greens and a skipped walk
    leave over goes to it. "static": fixed times (sumo_side.CycleKeeper then keeps the cycles in step)."""
    root = ET.Element("additional")
    if note:
        root.append(ET.Comment(note))
    for tid in sorted(signals, key=lambda t: int(t) if t.isdigit() else t):
        sg = signals[tid]
        tl = ET.SubElement(root, "tlLogic", id=tid, type="static" if COORD_TYPE == "static" else "actuated",
                           programID=program_id, offset=f"{sg.offset:g}")
        t_end = 0.0
        first = sg.greens[0] if sg.greens else None
        for k, (s, d, nm) in enumerate(sg.phases):
            t_end += d
            attrs = {"duration": f"{d:g}", "state": s}
            if COORD_TYPE != "static" and k in sg.greens:
                mn = min_green(js[tid], s) if js is not None else MIN_TURN_GREEN_S
                mn = min(mn, d)
                attrs["minDur"] = f"{mn:g}"
                attrs["maxDur"] = f"{d + (cycle / 2 if k == first else 0.0):g}"
                end = t_end % cycle
                attrs["latestEnd"] = f"{end:g}"
                if k == first:
                    attrs["earliestEnd"] = f"{end:g}"
            if nm:
                attrs["name"] = nm
            ET.SubElement(tl, "phase", attrib=attrs)
        ET.SubElement(tl, "param", key="cycle", value=f"{cycle:g}")
        if COORD_TYPE != "static":
            ET.SubElement(tl, "param", key="coordinated", value="true")
            ET.SubElement(tl, "param", key="cycleTime", value=f"{cycle:g}")
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)


def build(net_file: Path, route_files: list[Path], out: Path, scale: float = 1.0) -> dict:
    """The coordinated programs of every signal in net_file (from its program "0"), written to out. Returns a
    summary: cycle, the offsets, arrivals on green with and without them."""
    from cosim.signal_audit import load
    from cosim.town import link_flows

    js = load(net_file, [])
    flows = link_flows(net_file, route_files)
    programs = {tid: [[s, d, nm] for s, d, _mn, nm in jn.programs["0"]] for tid, jn in js.items()}
    cycle = cycle_length(js, programs, flows, scale)
    signals = {}
    for tid, phases in programs.items():
        ph = splits(js[tid], phases, flows.get(tid, {}), cycle)
        # the busiest green first: phase 0 is the one the cycle is kept on (sumo_side.CycleKeeper) and that gets
        # what the others leave over
        ratios = green_ratios(js[tid], ph, flows.get(tid, {}))
        if ratios:
            k0 = max(ratios, key=lambda k: (ratios[k], -k))
            ph = ph[k0:] + ph[:k0]
        sg = Signal(tid, ph, [k for k, p in enumerate(ph) if car_green(js[tid], p[0], p[2])])
        signals[tid] = sg
    pairs = signal_pairs(net_file, route_files)
    offs, s_best, s_zero = offsets(signals, pairs, cycle)
    for tid, o in offs.items():
        signals[tid].offset = o
    write(signals, cycle, out, js=js, note=f" cosim/coordination.py: cycle {cycle:g} s; arrivals on green "
                                    f"{100 * s_best:.0f} % (all offsets 0: {100 * s_zero:.0f} %) ")
    return {"cycle": cycle, "offsets": offs, "on_green": round(s_best, 3), "on_green_offsets_0": round(s_zero, 3),
            "greens": {tid: [round(sg.phases[k][1]) for k in sg.greens] for tid, sg in signals.items()},
            "pairs": len(pairs)}

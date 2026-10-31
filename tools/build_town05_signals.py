"""Town05's signal programs as real ones (25 Oct): opposite approaches go together.

CARLA's netconvert_carla.py gave every signalised junction CARLA's own cycle: one approach (one pole) green at a
time, all its movements together, the others red; then a 10 s walk on every crossing at once with every car red
(tools/build_town05_peds.py). That is "split phasing", which signal manuals keep for odd junctions (FHWA Traffic
Signal Timing Manual, ch. 4: when lefts cannot be separated, or one approach is heavy and the other light): it
serves one pole in four, so a car waits through three greens.

Here, as at most real junctions (NEMA ring-and-barrier: phase 2 with 6, 4 with 8):
* opposite approaches have their green together: straight on and right; the left turn too, giving way to
  the oncoming traffic ("g", a permissive left: Town05's left turns share their lane with the straight-on cars,
  so there is no separate left-arrow phase, which needs a lane of its own); turning cars give way to people on
  the crossing they turn over;
* pedestrians walk with the traffic parallel to them (the crossings over the side roads walk while the main
  road is green), then a clearance (flashing don't-walk: no one starts, the cars keep their green) of
  crossing length / 1.2 m/s, 5-15 s;
* yellow from the approach speed (ITE kinematic: 1 s reaction + v / (2 x 3.05 m/s2), rounded up to 0.5 s,
  3-6 s): 3.5 s at 50 km/h, 5 s on the ring road (80 km/h);
* all red from the junction's width (ITE: (W + 6 m) / v, rounded up to 0.5 s, 1-3 s).

SUMO's netconvert builds this ("--tls.rebuild --tls.layout opposites", see sumo.dlr.de Simulation/Traffic_Lights)
and, which matters, recomputes the junctions' right of way for it (who gives way to whom inside the junction
depends on what is green together; a program set over TraCI alone would leave that as CARLA's cycle had it).
Lanes, shapes and connections stay as they were (checked: 1027 lanes, no shape moved); the link numbers change,
so the CARLA poles' links (linkSignalID, cosim/tlmap.py) are carried over by (from lane, to lane).

The originals are kept as town05.carla_cycle.* (net, actuated, webster). Then:
    python tools/build_town05_signals.py            # rewrites town05.net.xml, .actuated.add.xml, .webster.add.xml
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCEN = Path(os.environ.get("TOWN05_DIR", ROOT / "scenarios" / "town05"))
KEEP = "carla_cycle"          # the originals' suffix
DECEL = 3.05                  # m/s2 (ITE: 10 ft/s2)
REACT = 1.0                   # s
WALK_SPEED = 1.2              # m/s, for the clearance after the walk (MUTCD 3.5 ft/s = 1.07; SUMO's walkers ~1.2)
PED_CLEAR_SPEED = 1.07        # m/s = 3.5 ft/s: MUTCD (11th ed., 4I.06) pedestrian clearance time = crossing / 3.5 ft/s,
#                               met by the flashing don't walk plus the buffer (steady don't walk: the yellow and all red
#                               after it) before conflicting traffic is released (cosim/signal_audit.py checks it)
RING_SPEED = 22.22            # m/s: the ring road's limit in the viewer (cosim/highway.py)
RING_LANES = 3                # driving lanes one way: the ring road (no crosswalk over it since 27 Oct)
EXTRA = os.environ.get("SIGNALS_EXTRA", "").split()  # more netconvert options (experiments)
MIN_GREEN, MAX_GREEN = 7.0, 45.0  # actuated: 7 s at least (a walk), 45 s at most (urban 20-45 s)
WALK_MIN_S = 7.0             # MUTCD 4I.06: the walk interval at least 7 s
# protected car greens (actuated), min-max s (FHWA STM tables 5-3 / 5-5: minimum 2-15 s, maximum 20-60 s by road);
# env PROTECTED_GREEN="5,45" for another
PROTECTED_GREEN = tuple(float(x) for x in os.environ.get("PROTECTED_GREEN", "5,30").split(","))
# "protected" (27 Oct, cosim/signal_plan.py): every direction has its own signal and is green only with nothing
# crossing its path, cars or walkers; "opposites": the 25 Oct programs (permissive left and right turns)
LAYOUT = os.environ.get("SIGNAL_LAYOUT", "protected")


def up(x: float, step: float = 0.5) -> float:
    return math.ceil(x / step - 1e-9) * step


def yellow_s(v: float) -> float:
    return min(max(up(REACT + v / (2 * DECEL)), 3.0), 6.0)


ALLRED_MIN = float(os.environ.get("ALLRED_MIN", "1.0"))


def allred_s(width: float, v: float) -> float:
    return min(max(up((width + 6.0) / max(v, 5.0)), ALLRED_MIN), max(3.0, ALLRED_MIN))


def conn_key(c: ET.Element) -> tuple:
    return c.get("from"), c.get("to"), c.get("fromLane"), c.get("toLane")


# "dedicated" (27 Oct): at a signalised junction, an approach's leftmost driving lane that goes straight on and left
# while another lane also goes straight on becomes a left-turn lane (its straight-on connection removed): a protected
# left arrow needs a lane of its own (FHWA Signal Timing Manual, ch. 4: from a shared lane, only split phasing),
# and with it both directions of a road can turn left together and go straight on together. "shared": as CARLA has it
# (measured 27 Oct, tools/check_junction_stops.py: "dedicated" leaves straight on one lane and the town locks at
# x0.8; the default is "shared", each approach on its own green: cosim/signal_plan.py)
LEFT_LANES = os.environ.get("LEFT_LANES", "shared")


def left_lane_deletions(src: Path, out: Path) -> int:
    """A netconvert connection file deleting, at the signalised junctions, the parking lanes' connections and (LEFT_LANES
    "dedicated") the straight-on connections of the left lanes made left-turn lanes. Returns how many."""
    root = ET.parse(src).getroot()
    tls_nodes = {j.get("id") for j in root.iter("junction") if j.get("type", "").startswith("traffic_light")}
    edge_to = {e.get("id"): e.get("to") for e in root.iter("edge") if not e.get("function")}
    lane_ok, parking = {}, set()
    for e in root.iter("edge"):
        if e.get("function"):
            continue
        for ln in e.findall("lane"):
            dis = ln.get("disallow") or ""
            lane_ok[ln.get("id")] = (ln.get("type") in (None, "driving") and "passenger" not in dis
                                     and (ln.get("allow") is None or "passenger" in ln.get("allow")))
            if ln.get("type") == "parking":
                parking.add(ln.get("id"))
    by_edge: dict[str, dict[int, list]] = {}
    park_conns = []
    for c in root.iter("connection"):
        frm = c.get("from", "")
        if frm.startswith(":") or edge_to.get(frm) not in tls_nodes or not c.get("tl"):
            continue
        if f"{frm}_{c.get('fromLane')}" in parking:
            park_conns.append(c)
        if not lane_ok.get(f"{frm}_{c.get('fromLane')}", False):
            continue
        by_edge.setdefault(frm, {}).setdefault(int(c.get("fromLane")), []).append(c)
    dels = ET.Element("connections")
    n = 0
    # a parking lane (a car leaving its spot, 5 km/h) does not run on across a signalised junction beside the
    # driving lanes: its car moves over to the right lane first. Beside it, the right lane's right turn would cross
    # its straight on (SUMO's foes), and it would need a green of its own
    for c in park_conns:
        if c.get("from") in by_edge:
            ET.SubElement(dels, "delete", attrib={"from": c.get("from"), "to": c.get("to"),
                                                  "fromLane": c.get("fromLane"), "toLane": c.get("toLane")})
            n += 1
    for frm, lanes in by_edge.items():
        if len(lanes) < 2 or LEFT_LANES != "dedicated":
            continue
        left = max(lanes)
        dirs = {c.get("dir") for c in lanes[left]}
        others_s = any(c.get("dir") == "s" for k, cs in lanes.items() if k != left for c in cs)
        if "s" in dirs and ({"l", "L", "t"} & dirs) and others_s:
            for c in lanes[left]:
                if c.get("dir") == "s":
                    ET.SubElement(dels, "delete", attrib={"from": frm, "to": c.get("to"), "fromLane": c.get("fromLane"),
                                                          "toLane": c.get("toLane")})
                    n += 1
    # no crosswalk over the ring road (27 Oct): three lanes each way at 80 km/h, people do not walk across it; at 2014
    # and 2328 its 24.5 m crossings are removed (the side road's crossing stays)
    lanes_of_edge = {e.get("id"): sum(1 for ln in e.findall("lane") if lane_ok.get(ln.get("id"), False))
                     for e in root.iter("edge") if not e.get("function")}
    for e in root.iter("edge"):
        if e.get("function") != "crossing":
            continue
        node = e.get("id")[1:].rsplit("_c", 1)[0]
        crossed = (e.get("crossingEdges") or "").split()
        if node in tls_nodes and any(lanes_of_edge.get(x, 0) >= RING_LANES for x in crossed):
            ET.SubElement(dels, "crossing", attrib={"node": node, "edges": " ".join(crossed), "discard": "true"})
            n += 1
    ET.ElementTree(dels).write(out, encoding="UTF-8", xml_declaration=True)
    return n


def netconvert(src: Path, out: Path) -> None:
    cmd = ["netconvert", "-s", str(src), "--tls.rebuild", "--tls.layout", "opposites",
           "--tls.default-type", "static", "--tls.allred.time", "2", "--tls.left-green.time", "6",
           "--tls.crossing-min.time", "7", "--tls.crossing-clearance.time", "5"] + EXTRA + ["-o", str(out)]
    dels = None
    if LAYOUT == "protected":  # parking lanes' connections; with "dedicated", the left lanes' straight on
        dels = out.with_suffix(".del.xml")
        left_lane_deletions(src, dels)
        cmd[3:3] = ["--connection-files", str(dels)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if dels is not None:
        dels.unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)


def build(src: Path, out: Path) -> dict:
    import sumolib

    from cosim.highway import find_loops

    tmp = out.with_suffix(".tmp.xml")
    netconvert(src, tmp)
    old = ET.parse(src).getroot()
    new_tree = ET.parse(tmp)
    new = new_tree.getroot()
    net = sumolib.net.readNet(str(tmp), withInternal=True, withPedestrianConnections=True)
    ring = {e for lp in find_loops(net) for e in lp}
    from cosim import signal_plan
    from cosim.signal_audit import load as audit_load

    audit_js = audit_load(tmp)  # links, approaches and SUMO's foes (geometry) of each signal

    # the CARLA poles' links: old (tl, index) -> landmark, by (from, to, fromLane, toLane)
    old_params = {}
    for tl in old.iter("tlLogic"):
        for p in tl.iter("param"):
            if p.get("key", "").startswith("linkSignalID:"):
                old_params[(tl.get("id"), int(p.get("key").split(":")[1]))] = p.get("value")
    old_idx = {conn_key(c): (c.get("tl"), int(c.get("linkIndex"))) for c in old.iter("connection") if c.get("tl")}
    new_idx = {conn_key(c): (c.get("tl"), int(c.get("linkIndex"))) for c in new.iter("connection") if c.get("tl")}
    link_lane, link_via = {}, {}
    for c in new.iter("connection"):
        if c.get("tl"):
            k = (c.get("tl"), int(c.get("linkIndex")))
            link_lane.setdefault(k, f"{c.get('from')}_{c.get('fromLane')}")
            if c.get("via"):
                link_via.setdefault(k, c.get("via"))
    carried = {}
    for key, (tl, i) in new_idx.items():
        o = old_idx.get(key)
        if o is not None and o in old_params:
            carried[(tl, i)] = old_params[o]

    stats = {"tls": 0, "phases": {}, "carried_links": len(carried), "lost_links": 0}
    lost = {o for o in old_params if o not in {old_idx.get(k) for k in new_idx}}
    stats["lost_links"] = len(lost)
    for tl in new.iter("tlLogic"):
        tid = tl.get("id")
        stats["tls"] += 1
        phases = tl.findall("phase")
        n_links = len(phases[0].get("state"))

        def speed(i: int) -> float:
            lane = link_lane.get((tid, i))
            if lane is None:
                return 13.89
            e = lane.rsplit("_", 1)[0]
            return RING_SPEED if e in ring else net.getLane(lane).getSpeed()

        def width(i: int) -> float:
            via = link_via.get((tid, i))
            try:
                return net.getLane(via).getLength() if via else 15.0
            except Exception:  # noqa: BLE001
                return 15.0

        def cross_len(i: int) -> float:
            for c in new.iter("connection"):
                if c.get("tl") == tid and int(c.get("linkIndex")) == i:
                    try:
                        return net.getEdge(c.get("to")).getLength()
                    except Exception:  # noqa: BLE001
                        return 10.0
            return 10.0

        crossing = set()
        for c in new.iter("connection"):
            if c.get("tl") == tid and c.get("to", "").startswith(":") and "_c" in c.get("to"):
                crossing.add(int(c.get("linkIndex")))
        if LAYOUT == "protected":
            jn = audit_js[tid]
            prog = signal_plan.plan(jn.links, jn.conflict, {
                "yellow": lambda i: yellow_s(speed(i)), "allred": lambda i: allred_s(width(i), speed(i)),
                "walk_speed": WALK_SPEED, "clear_speed": PED_CLEAR_SPEED})
            for ph in phases:
                tl.remove(ph)
            at = list(tl).index(tl.find("param")) if tl.find("param") is not None else len(tl)
            phases = []
            for st, d, mn, mx, nm in prog:
                ph = ET.Element("phase", duration=f"{d:g}", state=st)
                if nm:
                    ph.set("name", nm)
                if mn is not None:  # for the actuated program (actuated(): kept as they are)
                    ph.set("minDur", f"{mn:g}")
                    ph.set("maxDur", f"{mx:g}")
                tl.insert(at + len(phases), ph)
                phases.append(ph)
            tl.set("type", "static")
            phases_done = True
        else:
            phases_done = False
        prev = phases[-1].get("state")
        for k, ph in enumerate(phases if not phases_done else []):
            s = ph.get("state")
            greens = {i for i, ch in enumerate(s) if ch in "Gg"}
            if "y" in s:
                ys = [i for i, ch in enumerate(s) if ch == "y"]
                ph.set("duration", f"{max(yellow_s(speed(i)) for i in ys):g}")
            elif not greens:
                ended = [i for i, ch in enumerate(prev) if ch in "y" and i not in crossing]
                if ended:
                    ph.set("duration", f"{max(allred_s(width(i), speed(i)) for i in ended):g}")
            elif greens <= crossing:
                pass  # (none in this layout)
            else:
                was = phases[k - 1].get("state") if k > 0 else prev
                walk_ended = [i for i in crossing if was[i] in "Gg" and s[i] not in "Gg"]
                if walk_ended and any(was[i] in "Gg" for i in greens - crossing):
                    # the clearance after the walk: no one starts, those on the crossing get across
                    ph.set("duration", f"{min(max(up(max(cross_len(i) for i in walk_ended) / WALK_SPEED), 5.0), 15.0):g}")
                    ph.set("name", "walk clearance")
                else:
                    ph.set("name", "green")
            prev = s
        # the walk clearance is at least the pedestrian clearance time less the buffer that follows it (26 Oct: the
        # 15 s cap left the 24.5 m crossings over the ring road at 2014 and 2328 with 21.5 s from the end of the walk
        # to the conflicting green, the MUTCD's 3.5 ft/s needs 22.9 s)
        for k, ph in enumerate(phases if not phases_done else []):
            if ph.get("name") != "walk clearance":
                continue
            was = phases[k - 1].get("state")
            ended = [i for i in crossing if was[i] in "Gg" and ph.get("state")[i] not in "Gg"]
            buffer = 0.0
            for q in range(k + 1, k + len(phases)):
                nxt = phases[q % len(phases)]
                if any(ch in "Gg" for ch in nxt.get("state")):
                    break
                buffer += float(nxt.get("duration"))
            need = up(max(cross_len(i) for i in ended) / PED_CLEAR_SPEED - buffer)
            if need > float(ph.get("duration")):
                ph.set("duration", f"{need:g}")
        # the CARLA poles' links
        for p in list(tl.findall("param")):
            if p.get("key", "").startswith("linkSignalID:"):
                tl.remove(p)
        for i in range(n_links):
            if (tid, i) in carried:
                ET.SubElement(tl, "param", key=f"linkSignalID:{i}", value=carried[(tid, i)])
        stats["phases"][tid] = [(p.get("state"), p.get("duration")) for p in phases]
    new_tree.write(out, encoding="UTF-8", xml_declaration=True)
    tmp.unlink()
    if LAYOUT == "protected":
        rightofway(out)
    return stats


def rightofway(net_file: Path) -> None:
    """netconvert again with the net's own programs loaded (--tllogic-files): who gives way to whom inside each
    junction is worked out for these programs (no green has to give way: none shares the junction with a conflict).
    Lanes, connections, link numbers and the programs (with their linkSignalID) stay as they are."""
    root = ET.parse(net_file).getroot()
    add = ET.Element("additional")
    for tl in root.iter("tlLogic"):
        add.append(tl)
    tls = net_file.with_suffix(".tls.tmp.xml")
    ET.ElementTree(add).write(tls, encoding="UTF-8", xml_declaration=True)
    tmp = net_file.with_suffix(".row.tmp.xml")
    r = subprocess.run(["netconvert", "-s", str(net_file), "--tllogic-files", str(tls), "-o", str(tmp)],
                       capture_output=True, text=True)
    tls.unlink()
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)
    tmp.replace(net_file)


_CROSSING_LINKS: dict[str, set[int]] = {}


def crossing_links(tid: str) -> set[int]:
    return _CROSSING_LINKS.get(tid, set())


def actuated(net_file: Path, out: Path) -> int:
    """programID "actuated": the greens that may run longer or shorter on SUMO's loops (MIN_GREEN-MAX_GREEN); the
    walk clearances, yellows and all reds as they are."""
    import copy

    root = ET.Element("additional")
    n = 0
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "connection" and el.get("tl") and el.get("to", "").startswith(":") and "_c" in el.get("to"):
            _CROSSING_LINKS.setdefault(el.get("tl"), set()).add(int(el.get("linkIndex")))
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag != "tlLogic":
            continue
        tl = copy.deepcopy(el)
        tl.set("type", "actuated")
        tl.set("programID", "actuated")
        phs = tl.findall("phase")
        for k, ph in enumerate(phs):
            if ph.get("name") != "green" or ph.get("minDur") is not None:
                continue
            if LAYOUT != "protected":
                ph.set("minDur", f"{MIN_GREEN:g}")
                ph.set("maxDur", f"{MAX_GREEN:g}")
                continue
            # protected (cosim/signal_plan.py): a green with a walk 10-40 s (the MUTCD's 7 s walk and more);
            # the others PROTECTED_GREEN
            walk = any(c == "G" for c in (ph.get("state")[i] for i in crossing_links(tl.get("id"))))
            mn, mx = (10.0, 40.0) if walk else PROTECTED_GREEN
            ph.set("minDur", f"{mn:g}")
            ph.set("maxDur", f"{mx:g}")
        root.append(tl)
        n += 1
        el.clear()
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    return n


def webster_walks(net_file: Path, web_file: Path) -> int:
    """Webster's split (cosim/town.webster) takes every phase with a green as a green to share out, the walk
    clearances too: it gave them 5-8 s (26 Oct, cosim/signal_audit.py: 11-16.5 s from the end of a walk to the
    conflicting green where 13-23 s are needed) and the walks themselves 5-6 s (MUTCD: 7 s). Here each walk
    clearance gets back its duration from the network's program, the green before it keeps the rest of the time
    Webster gave the two (at least MIN_GREEN), and the phases get the network's names. Returns phases changed."""
    net_tl = {}
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "tlLogic":
            net_tl[el.get("id")] = [(p.get("state"), float(p.get("duration")), p.get("name", "")) for p in el.findall("phase")]
            el.clear()
    tree = ET.parse(web_file)
    changed = 0
    for tl in tree.getroot().iter("tlLogic"):
        ref = net_tl.get(tl.get("id"))
        ph = tl.findall("phase")
        if ref is None or len(ref) != len(ph) or any(p.get("state") != r[0] for p, r in zip(ph, ref, strict=True)):
            continue
        for k, (p, (_, dur, name)) in enumerate(zip(ph, ref, strict=True)):
            if name:
                p.set("name", name)
            if name == "walk" and float(p.get("duration")) < WALK_MIN_S:  # the exclusive walk (27 Oct): MUTCD 7 s
                p.set("duration", f"{WALK_MIN_S:g}")
                changed += 1
            if name != "walk clearance" or k == 0:
                continue
            g = ph[k - 1]
            total = float(g.get("duration")) + float(p.get("duration"))
            if float(p.get("duration")) != dur:
                p.set("duration", f"{dur:g}")
                g.set("duration", f"{max(MIN_GREEN, total - dur):g}")
                changed += 1
    ET.indent(tree.getroot())
    tree.write(web_file, encoding="UTF-8", xml_declaration=True)
    return changed


def main() -> int:
    from cosim.town import webster_programs

    net = SCEN / "town05.net.xml"
    kept = SCEN / f"town05.{KEEP}.net.xml"
    if not kept.is_file():
        for suffix in ("net.xml", "actuated.add.xml", "webster.add.xml"):
            shutil.copy2(SCEN / f"town05.{suffix}", SCEN / f"town05.{KEEP}.{suffix}")
    stats = build(kept, net)
    n_act = actuated(net, SCEN / "town05.actuated.add.xml")
    routes = [SCEN / f"town05.{k}.rou.xml" for k in ("passenger", "delivery", "truck", "bus", "motorcycle")]
    n_web = webster_programs(net, [r for r in routes if r.is_file()], SCEN / "town05.webster.add.xml")
    webster_walks(net, SCEN / "town05.webster.add.xml")
    # the coordinated programs (27 Oct): one cycle for every signal, green waves (cosim/coordination.py)
    from cosim.coordination import build as coordinate

    coord = coordinate(net, [r for r in routes if r.is_file()], SCEN / "town05.coordinated.add.xml")
    import json

    from cosim.town import content_sha256

    sj = SCEN / "scenario.json"
    meta = json.loads(sj.read_text(encoding="utf-8"))
    meta["net_content_sha256"] = content_sha256(net)
    meta["signal_layout"] = {"layout": "opposites (opposite approaches together, permissive lefts, walk with the "
                                       "parallel traffic)", "tool": "tools/build_town05_signals.py",
                             "original_net": f"town05.{KEEP}.net.xml", "carried_links": stats["carried_links"],
                             "lost_links": stats["lost_links"]}
    meta["actuated_programs"], meta["webster_programs"] = n_act, n_web
    meta["coordinated"] = {"cycle_s": coord["cycle"], "arrivals_on_green": coord["on_green"],
                           "arrivals_on_green_offsets_0": coord["on_green_offsets_0"]}
    sj.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for tid, ph in stats["phases"].items():
        print(tid, ph)
    print({k: v for k, v in stats.items() if k != "phases"}, "actuated", n_act, "webster", n_web, "coordinated",
          {k: coord[k] for k in ("cycle", "on_green", "on_green_offsets_0")})
    return 0


if __name__ == "__main__":
    sys.exit(main())

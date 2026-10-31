"""Signalised crosswalks for Town05's SUMO network (scenarios/town05).

    python tools/build_town05_peds.py            # rewrites town05.net.xml, .actuated.add.xml, .webster.add.xml
                                                 # (the first run keeps the originals as *.noped.*)

The network was built for vehicles (cosim/town.py): sidewalks, but no walking areas and no crossings,
so a pedestrian could not cross a road (cosim/extras.py kept each on one sidewalk). This adds:

* walking areas at every junction (pedestrians go round corners from one sidewalk to the next);
* a crossing over each road at the 15 signalised junctions, where netconvert's own guess puts one
  (52 crossings: one per road that has sidewalks). None at the unsignalised junctions: there a
  crossing would be a zebra that every car must stop for;
* a pedestrian signal for each crossing, inside the junction's existing programs (the network's own
  program "0" and the actuated and Webster ones):
  - a pedestrian phase at the end of each cycle: every car red, every crossing walk for 10 s, then 5 s
    all red while the last ones finish crossing. cosim/sumo_side.WalkSkip skips it when nobody is
    waiting at that junction, so the cars lose nothing when no one wants to cross;
  - walk also during a car phase on a crossing no green car movement touches (a quiet side road).
  Crossings that ran alongside the cars instead (turning cars waiting for walkers) locked the town:
  measured with SUMO alone, cars stopped inside junctions behind walkers until queues spilled back.

Vehicle edges, lanes, link indices and the CARLA light mapping (linkSignalID params) stay as they were:
the crossings' links are numbered after the vehicle links, so routes and the CARLA lights still match.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim.sumo_tools import binary  # noqa: E402

SCEN = ROOT / "scenarios" / "town05"


def run(cmd: list[str], cwd: Path) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stdout}\n{r.stderr}")


WALK_S, CLEAR_S = 10, 5  # the pedestrian phase: walk, then all red while the last ones finish crossing


def walk_states(states: list[str], links: dict[int, tuple[str, str]], crossings: list[set[str]]) -> list[str]:
    """Phase states with one pedestrian signal per crossing appended (rules in the module doc)."""
    out = []
    for state in states:
        green = any(c in "Gg" for c in state)
        walk = ["G" if green and all(state[i] == "r" for i, (f, t) in links.items() if f in edges or t in edges)
                else "r" for edges in crossings]
        out.append(state + "".join(walk))
    return out


def add_walk_phase(tl: ET.Element, n_links: int, n_cross: int) -> None:
    """The pedestrian phase at the end of the cycle: every car red, every crossing walk; then all red."""
    phases = tl.findall("phase")
    at = list(tl).index(phases[-1]) + 1
    for k, (dur, walk) in enumerate(((WALK_S, "G"), (CLEAR_S, "r"))):
        ph = ET.Element("phase", duration=str(dur), state="r" * n_links + walk * n_cross)
        ph.tail = phases[-1].tail
        tl.insert(at + k, ph)


def extend_programs(root: ET.Element, links: dict[str, dict[int, tuple[str, str]]],
                    crossings: dict[str, list[set[str]]]) -> int:
    n = 0
    for tl in root.iter("tlLogic"):
        tid = tl.get("id")
        if not crossings.get(tid):
            continue
        phases = tl.findall("phase")
        width = len(phases[0].get("state"))
        if width != len(links[tid]):
            raise ValueError(f"{tid}: program has {width} links, the network {len(links[tid])}")
        for ph, st in zip(phases, walk_states([p.get("state") for p in phases], links[tid], crossings[tid]),
                          strict=True):
            ph.set("state", st)
        add_walk_phase(tl, width, len(crossings[tid]))
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=str(SCEN))
    a = ap.parse_args(argv)
    scen = Path(a.scenario)
    name = scen.name
    net = scen / f"{name}.net.xml"
    orig = scen / f"{name}.noped.net.xml"
    if not orig.exists():
        shutil.copy2(net, orig)
        for prog in ("actuated", "webster"):
            shutil.copy2(scen / f"{name}.{prog}.add.xml", scen / f"{name}.{prog}.noped.add.xml")
    nc = binary("netconvert")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run([nc, "-s", str(orig), "--plain-output-prefix", "p0"], tmp)
        run([nc, "-s", str(orig), "--crossings.guess", "true", "--plain-output-prefix", "p1"], tmp)
        nodes = ET.parse(tmp / "p0.nod.xml").getroot()
        tl_of = {n.get("id"): n.get("tl") for n in nodes.iter("node") if n.get("tl")}
        tll = ET.parse(tmp / "p0.tll.xml")
        links: dict[str, dict[int, tuple[str, str]]] = {}
        for c in tll.getroot().iter("connection"):
            links.setdefault(c.get("tl"), {})[int(c.get("linkIndex"))] = (c.get("from"), c.get("to"))
        crossings: dict[str, list[set[str]]] = {}
        lines = []
        for c in ET.parse(tmp / "p1.con.xml").getroot().iter("crossing"):
            tid = tl_of.get(c.get("node"))
            if tid is None:
                continue  # unsignalised junction: no crossing
            k = len(links[tid]) + len(crossings.get(tid, []))
            crossings.setdefault(tid, []).append(set(c.get("edges").split()))
            lines.append(f'    <crossing node="{c.get("node")}" edges="{c.get("edges")}" priority="1" '
                         f'linkIndex="{k}"/>')
        con = (tmp / "p0.con.xml").read_text(encoding="utf-8")
        con = con.replace("</connections>", "\n".join(lines) + "\n</connections>")
        (tmp / "con.xml").write_text(con, encoding="utf-8")
        extend_programs(tll.getroot(), links, crossings)
        tll.write(tmp / "tll.xml", encoding="UTF-8", xml_declaration=True)
        cfg = (tmp / "p0.netccfg").read_text(encoding="utf-8")
        opts = re.findall(r"<([\w.-]+) value=\"([^\"]*)\"/>", cfg.split("<processing>")[1]) if "<processing>" in cfg else []
        cmd = [nc, "--node-files", "p0.nod.xml", "--edge-files", "p0.edg.xml", "--connection-files", "con.xml",
               "--tllogic-files", "tll.xml", "--walkingareas", "true", "-o", "out.net.xml",
               "--xml-validation", "never"]  # its own plain output fails its schema (params in tll connections)
        for k, v in opts:
            if k != "connection-files":
                cmd += [f"--{k}", v]
        run(cmd, tmp)
        shutil.copy2(tmp / "out.net.xml", net)
    for prog in ("actuated", "webster"):
        src = scen / f"{name}.{prog}.noped.add.xml"
        tree = ET.parse(src)
        n = extend_programs(tree.getroot(), links, crossings)
        ET.indent(tree)
        tree.write(scen / f"{name}.{prog}.add.xml", encoding="UTF-8", xml_declaration=True)
        print(f"{prog}: {n} programs with pedestrian signals")
    print(f"{sum(len(v) for v in crossings.values())} crossings at {len(crossings)} signalised junctions; "
          f"{net.name} rewritten (original: {orig.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build a SUMO scenario for one of CARLA's own towns (Town10HD first).

    python -m cosim.town --xodr "<CARLA>/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD_Opt.xodr"
    python -m cosim.town --from-server --carla-map Town10HD_Opt        # asks a running CARLA

Here the map comes first and SUMO is derived from it, the reverse of the corridor maps:

1. **Network**: CARLA's own netconvert_carla.py (vendored, MIT, third_party/carla_sumo). It
   runs SUMO's netconvert on the OpenDRIVE file and adds the town's traffic lights, recording
   for every signal link the CARLA landmark that shows it (``linkSignalID`` params, tlmap.py).
   Its signal program ("0") gives each approach 42 s green, 3 s yellow, 3 s all-red in turn.
   It leaves some movements of signalised junctions without a signal (lanes no OpenDRIVE
   signal references); tlcomplete.py puts them under the signal of the lane beside them.
2. **Demand**: SUMO's randomTrips.py, one run per vehicle class (cars, vans, trucks, buses,
   motorbikes), routes validated by duarouter. Seeded, so the same command gives the same
   file.
3. **Signal programs** the corridor scenarios also have:
   * ``webster``: Webster's cycle and green split from the demand's flows (webster_programs).
     SUMO's own tlsCycleAdaptation.py cannot be used here: it finds signals through the
     junction type, and netconvert_carla adds them to plain junctions (measured: it reports
     "no flows" for every one of them);
   * ``actuated``: the default phases as gap-actuated control (5 s min, 60 s max green).
4. ``scenario.json`` records the CARLA map name, the step length, the hashes of the inputs
   and the tool versions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from cosim import tlcomplete
from cosim.opendrive import sha256
from cosim.sumo_tools import binary, sumo_home

ROOT = Path(__file__).resolve().parents[1]
NETCONVERT_CARLA = ROOT / "third_party" / "carla_sumo" / "netconvert_carla.py"
log = logging.getLogger("cosim.town")

# vClass -> mean seconds between departures at scale 1 (scale divides them). Roughly an urban
# mix: ~84 % cars, the rest vans, trucks, buses and motorbikes.
#
# Town10 is small (five signalised junctions, one driving lane per direction on most streets,
# blocks of 50-100 m), and SUMO runs without teleporting, so queues that spill round a block
# lock it. Measured with seed 7 (SUMO 1.27.1, Webster programs, 1800-3600 s):
#   scale 0.4: no gridlock under Webster or actuated over a full hour (1070 / 1080 of 1102
#              trips completed, mean trip 81 s / 76 s); CARLA's default program (42 s green
#              per approach) gridlocks (250 completed);
#   scale 0.5: actuated still flows, Webster gridlocks; scale 0.6: both gridlock.
# So the default is 0.4.
DEFAULT_SCALE = 0.4
DEMAND = {"passenger": 1.6, "delivery": 16.0, "truck": 40.0, "bus": 90.0, "motorcycle": 24.0}


def _env() -> dict[str, str]:
    home = str(sumo_home())
    # PYTHONHASHSEED: netconvert_carla iterates over Python sets when it numbers signal links
    # and orders phases; without a fixed seed the same map gives a different network each run.
    return dict(os.environ, SUMO_HOME=home, PYTHONHASHSEED="0",
                PATH=str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", ""))


def _run(cmd: list[str], cwd: Path) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=_env(), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[1] if len(cmd) > 1 else cmd[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def content_sha256(path: Path) -> str:
    """sha256 without XML comments (SUMO tools stamp the generation time into a comment)."""
    text = re.sub(r"<!--.*?-->", "", Path(path).read_text(encoding="utf-8"), flags=re.S)
    return hashlib.sha256(text.encode()).hexdigest()


def sumo_version() -> str:
    out = subprocess.run([binary("sumo"), "--version"], capture_output=True, text=True).stdout
    return out.splitlines()[0].strip() if out else "?"


def link_flows(net_file: Path, route_files: list[Path], begin: float = 0.0, hours: float = 1.0
               ) -> dict[str, dict[int, float]]:
    """Vehicles per hour on each signal link (tls id -> link index -> veh/h), from the routes.
    A vehicle going from edge a to edge b is shared evenly over the controlled lane-to-lane
    connections a -> b (it uses one of them; which one is SUMO's lane choice)."""
    conns: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "connection" and el.get("tl"):
            conns.setdefault((el.get("from"), el.get("to")), []).append((el.get("tl"), int(el.get("linkIndex"))))
    flows: dict[str, dict[int, float]] = {}
    end = begin + 3600 * hours
    for rf in route_files:
        for _, el in ET.iterparse(str(rf), events=("end",)):
            if el.tag != "vehicle":
                continue
            if begin <= float(el.get("depart", "0")) < end:
                route = el.find("route")
                edges = route.get("edges", "").split() if route is not None else []
                pce = {"bus": 2.0, "truck": 2.0, "motorcycle": 0.5}.get(_vclass_of(el), 1.0)
                for a, b in itertools.pairwise(edges):
                    links = conns.get((a, b), [])
                    for tl, li in links:
                        f = flows.setdefault(tl, {})
                        f[li] = f.get(li, 0.0) + pce / len(links) / hours
            el.clear()
    return flows


def _vclass_of(vehicle: ET.Element) -> str:
    # randomTrips names types "<prefix>_<vclass>"
    return vehicle.get("type", "").rpartition("_")[2]


def link_lanes(net_file: Path) -> dict[str, dict[int, str]]:
    """tls id -> link index -> the incoming lane it leaves from ("<edge>_<lane>")."""
    out: dict[str, dict[int, str]] = {}
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag == "connection" and el.get("tl"):
            out.setdefault(el.get("tl"), {})[int(el.get("linkIndex"))] = f"{el.get('from')}_{el.get('fromLane')}"
    return out


def webster(phases: list[tuple[str, float]], flows: dict[int, float], sat: float = 1800.0,
            min_green: float = 5.0, cycle_range: tuple[float, float] = (30.0, 120.0),
            lanes: dict[int, str] | None = None) -> list[tuple[str, float]]:
    """Webster's optimal cycle C = (1.5 L + 5) / (1 - Y), greens split by flow ratio.

    phases: (state, duration) of a fixed-time program. Green phases are those with a g/G;
    the others (yellow, all-red) are kept as they are and make up the lost time L. A green
    phase's flow ratio y is its critical lane's flow over the saturation flow (per lane):
    the flows of the links green in the phase are added up per incoming lane (`lanes`, link
    -> lane; without it every link counts as its own lane), and the largest sum is taken.
    A lane feeding a straight and a turning link carries both, so taking single links
    would understate y (measured on Town10: cycles stuck at the 30 s minimum, 5-8 s greens,
    and gridlock at 0.5x demand).
    """
    greens = [i for i, (s, _) in enumerate(phases) if any(c in "Gg" for c in s)]
    lost = sum(d for i, (_, d) in enumerate(phases) if i not in greens)
    lanes = lanes or {}

    def critical(state: str) -> float:
        per_lane: dict[str, float] = {}
        for k, c in enumerate(state):
            if c in "Gg":
                lane = lanes.get(k, f"link{k}")
                per_lane[lane] = per_lane.get(lane, 0.0) + flows.get(k, 0.0)
        return max(per_lane.values(), default=0.0)

    y = {i: critical(phases[i][0]) / sat for i in greens}
    total = min(sum(y.values()), 0.9)  # beyond 0.9 the formula explodes: saturated anyway
    cycle = (1.5 * lost + 5.0) / (1.0 - total)
    cycle = min(max(cycle, cycle_range[0]), cycle_range[1])
    effective = cycle - lost
    n = len(greens)
    out = []
    for i, (s, d) in enumerate(phases):
        if i in greens:
            share = y[i] / sum(y.values()) if sum(y.values()) > 0 else 1.0 / n
            d = max(min_green, round(effective * share))
        out.append((s, float(d)))
    return out


def webster_programs(net_file: Path, route_files: list[Path], out: Path) -> int:
    flows = link_flows(net_file, route_files)
    lanes = link_lanes(net_file)
    root = ET.Element("additional")
    n = 0
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag != "tlLogic":
            continue
        phases = [(p.get("state"), float(p.get("duration"))) for p in el.findall("phase")]
        tl = ET.SubElement(root, "tlLogic", id=el.get("id"), type="static", programID="webster", offset="0")
        for s, d in webster(phases, flows.get(el.get("id"), {}), lanes=lanes.get(el.get("id"))):
            ET.SubElement(tl, "phase", duration=f"{d:g}", state=s)
        n += 1
        el.clear()
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    return n


def xodr_from_server(carla_map: str, host: str, port: int) -> str:
    import carla

    client = carla.Client(host, port)
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != carla_map:
        world = client.load_world(carla_map)
    return world.get_map().to_opendrive()


def actuated_programs(net_file: Path, out: Path, min_dur: int = 5, max_dur: int = 60) -> int:
    """The network's static programs as actuated ones with programID "actuated"."""
    root = ET.Element("additional")
    n = 0
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag != "tlLogic":
            continue
        tl = copy.deepcopy(el)
        tl.set("type", "actuated")
        tl.set("programID", "actuated")
        for ph in tl.findall("phase"):
            if any(c in "Gg" for c in ph.get("state", "")):
                ph.set("minDur", str(min_dur))
                ph.set("maxDur", str(max_dur))
        root.append(tl)
        n += 1
        el.clear()
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding="UTF-8", xml_declaration=True)
    return n


def build(xodr_text: str, out_dir: Path, name: str, carla_map: str, end: float = 3600.0,
          scale: float = 1.0, seed: int = 7, step_length: float = 0.5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    xodr = out_dir / f"{name}.xodr"
    xodr.write_text(xodr_text, encoding="utf-8", newline="\n")  # same bytes on Windows
    net = out_dir / f"{name}.net.xml"
    py = sys.executable
    tools = sumo_home() / "tools"

    _run([py, NETCONVERT_CARLA, xodr.name, "--output", net.name], out_dir)
    signals = tlcomplete.complete(net, net)  # every movement at a signalised junction under its signal
    log.info("signal control completed: %s", signals)

    routes = []
    for i, (vclass, period) in enumerate(DEMAND.items()):
        rou = f"{name}.{vclass}.rou.xml"
        _run([py, tools / "randomTrips.py", "-n", net.name, "-r", rou, "-o", f"{name}.{vclass}.trips.xml",
              "--vehicle-class", vclass, "--prefix", vclass[:3], "-b", "0", "-e", str(end),
              "-p", str(period / scale), "--seed", str(seed + i), "--min-distance", "120",
              "--fringe-factor", "10", "--validate",
              "--trip-attributes", 'departLane="best" departSpeed="max"'], out_dir)
        (out_dir / f"{name}.{vclass}.trips.xml").unlink(missing_ok=True)
        routes.append(rou)

    n_web = webster_programs(net, [out_dir / r for r in routes], out_dir / f"{name}.webster.add.xml")
    n_act = actuated_programs(net, out_dir / f"{name}.actuated.add.xml")

    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    ET.SubElement(inp, "net-file", value=net.name)
    ET.SubElement(inp, "route-files", value=",".join(routes))
    t = ET.SubElement(cfg, "time")
    ET.SubElement(t, "begin", value="0")
    ET.SubElement(t, "end", value=str(int(end)))
    ET.SubElement(t, "step-length", value=str(step_length))
    ET.indent(cfg)
    ET.ElementTree(cfg).write(out_dir / f"{name}.sumocfg", encoding="UTF-8", xml_declaration=True)

    vehicles = {}
    for vclass in DEMAND:
        vehicles[vclass] = sum(1 for _, el in ET.iterparse(str(out_dir / f"{name}.{vclass}.rou.xml"))
                               if el.tag == "vehicle")
    tls = sum(1 for _, el in ET.iterparse(str(net)) if el.tag == "tlLogic")
    info = {
        "kind": "carla-town",
        "carla_map": carla_map,
        "xodr_sha256": sha256(xodr),
        "net_content_sha256": content_sha256(net),
        "tls": tls,
        "signal_control_completed": signals,
        "actuated_programs": n_act,
        "webster_programs": n_web,
        "vehicles": vehicles,
        "end": end,
        "step_length": step_length,
        "seed": seed,
        "clock_start_s": 16 * 3600,  # SUMO t = 0 is shown as 16:00, like Ingolstadt7's 57600 s
        "scale": scale,
        "tools": {"sumo": sumo_version(), "netconvert_carla": "CARLA 0.9.16"},
    }
    (out_dir / "scenario.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xodr", type=Path, help="OpenDRIVE file of the town (from the CARLA package)")
    src.add_argument("--from-server", action="store_true", help="ask a running CARLA server for it")
    ap.add_argument("--carla-map", default="Town10HD_Opt")
    ap.add_argument("--name", default="town10")
    ap.add_argument("--end", type=float, default=3600.0)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="demand multiplier (see DEMAND)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step-length", type=float, default=0.5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    text = a.xodr.read_text(encoding="utf-8") if a.xodr else xodr_from_server(a.carla_map, a.host, a.port)
    info = build(text, ROOT / "scenarios" / a.name, a.name, a.carla_map, a.end, a.scale, a.seed, a.step_length)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

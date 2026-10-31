"""What a scenario folder holds, for both kinds of map.

* ``sumo-corridor`` (Ingolstadt7, Cologne3): the SUMO network is the source; the CARLA map is
  derived from it (opendrive.py) and loaded in OpenDRIVE standalone mode. Signal links come
  from the landmark ids. The sun follows the scenario's own clock and place.
* ``carla-town`` (Town10HD): the CARLA town is the source; the SUMO network was derived from
  it (town.py). Signal links come from the network's linkSignalID params. The town has no
  real place, so the sun is computed for Ingolstadt; its clock starts at ``clock_start_s``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import sumolib

from cosim.mapping import Frame
from cosim.opendrive import MapInfo, build
from cosim.tlmap import Links, links_from_landmark_ids, links_from_net

INGOLSTADT = (48.7685, 11.4163)  # lat, lon of the Ingolstadt7 corridor's centre (its net, convBoundary middle)


@dataclass
class Scenario:
    name: str
    kind: str
    folder: Path
    net_file: Path
    frame: Frame
    lat: float
    lon: float
    clock_start_s: float  # added to SUMO time for the sun (0 when SUMO time is the clock)
    carla_map: str | None = None  # carla-town: the town to load
    xodr: Path | None = None  # sumo-corridor: the derived map
    map_info: MapInfo | None = None
    meta: dict = field(default_factory=dict)

    def links(self, landmark_ids: list[str]) -> Links:
        if self.kind == "carla-town":
            return links_from_net(self.net_file)
        net = sumolib.net.readNet(str(self.net_file))
        tls_ids = [t.getID() for t in net.getTrafficLights()]
        merged = self.map_info.signal_links if self.map_info else {}
        out: Links = {}
        for lid in landmark_ids:
            # a kept head stands for every head of its lane (opendrive.py, point 4)
            for alias in merged.get(lid, [lid]):
                out.setdefault(lid, []).extend(links_from_landmark_ids([alias], tls_ids).get(alias, []))
        return {k: v for k, v in out.items() if v}


def load(name: str, root: Path) -> Scenario:
    folder = root / "scenarios" / name
    net_file = folder / f"{name}.net.xml"
    meta_file = folder / "scenario.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    if meta.get("kind") == "carla-town":
        net = sumolib.net.readNet(str(net_file))
        ox, oy = net.getLocationOffset()
        lat, lon = INGOLSTADT
        return Scenario(name, "carla-town", folder, net_file, Frame(ox, oy), lat, lon,
                        float(meta.get("clock_start_s", 16 * 3600)), carla_map=meta["carla_map"], meta=meta)
    info = build(net_file, root / "maps" / name)
    net = sumolib.net.readNet(str(net_file))
    lon, lat = net.convertXY2LonLat(info.cx, info.cy)
    return Scenario(name, "sumo-corridor", folder, net_file, Frame(info.cx, info.cy), lat, lon, 0.0,
                    xodr=root / "maps" / name / info.xodr, map_info=info, meta=meta)


def tls_centres(net_file: Path, frame: Frame) -> dict[str, tuple[float, float, int]]:
    """CARLA (x, y) of each signal's junction and its number of links, busiest first."""
    net = sumolib.net.readNet(str(net_file))
    out = {}
    for tls in net.getTrafficLights():
        conns = tls.getConnections()
        nodes = {c[0].getEdge().getToNode() for c in conns}
        xs = [n.getCoord()[0] for n in nodes]
        ys = [n.getCoord()[1] for n in nodes]
        x, y = frame.to_carla_xy(sum(xs) / len(xs), sum(ys) / len(ys))
        out[tls.getID()] = (x, y, len(conns))
    return dict(sorted(out.items(), key=lambda kv: -kv[1][2]))

"""Turn a SUMO network into an OpenDRIVE map CARLA can load (OpenDRIVE standalone mode).

The SUMO network stays the source of truth: the simulation, its demand and its signal
programs are untouched. This module only derives a *drawing* of the same roads for CARLA.

What has to be right, each checked in tests/test_opendrive.py:

1. **Re-centring.** SUMO keeps the RESCO networks in UTM metres (Ingolstadt7 sits about
   213 km east of the origin). Unreal Engine 4 cannot place actors that far out (its world
   is about +-10 km), so the map is written shifted by the centre of the network's bounding
   box. The shift (cx, cy) is stored next to the map and the bridge applies it to every
   position: carla = (x - cx, -(y - cy)). The minus on y is the change from SUMO's
   right-handed axes to Unreal's left-handed ones.
2. **Traffic lights.** netconvert writes signal heads with arrows under the German StVO code
   1000011, plain ones under 1000001; CARLA treats only 1000001 as a traffic light (measured:
   0 of Ingolstadt7's 72 heads recognised before the rewrite, 72 after). The type is rewritten. Each
   head keeps netconvert's id "<tlsID>_<linkIndex>", which is exactly the SUMO link whose
   state it shows, so the bridge needs no lookup table.
3. **Traffic-light actors.** CARLA spawns a traffic-light actor for each signal of a
   `<controller>`, but links it to its signal only if the controller is referenced by a
   `<junction>` (Unreal side, TrafficLightManager.cpp: "No junctions in controller", then
   it gives up). netconvert writes the controllers but not those references; measured on
   the server: 0 of Ingolstadt7's 72 landmarks had a traffic-light actor. `_link_controllers`
   adds `<controller id=.. type="0"/>` to the junction of every node the signal controls.
4. **Signal heads at the kerb.** netconvert puts one head per controlled link on the lane
   centre line, and CARLA draws each as a full pole with an arm: poles stood in the middle of
   the lanes, and heads of the same lane (straight and right turn) on the same spot, of
   which CARLA linked only one (measured: 72 heads, 59 distinct positions, 60 actors linked).
   `_kerb_signals` keeps one head per lane, moves it 0.5 m beyond the outer edge of the
   carriageway, 2 m apart along the road for neighbouring lanes, and records which link ids
   each kept head stands for (`MapInfo.signal_links`); the runner shows the most
   restrictive of their states.
5. **Short ids.** Measured on the server: the 12 heads of Ingolstadt7's signal
   `cluster_306484187_cluster_1200363791_...` (a 196-character SUMO id) got traffic-light
   actors that CARLA never linked to them, while all others were linked. Signal ids longer
   than `MAX_ID` are therefore renamed to a short stable alias (`MapInfo.tls_alias`).
6. **Lanes line up.** Sampled SUMO lane midpoints, mapped to CARLA coordinates, must fall
   on a CARLA driving lane, close to its centre.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cosim.sumo_tools import binary

CARLA_TRAFFIC_LIGHT = "1000001"
_SIGNAL_TYPE = re.compile(r'type="1000011" subtype="[-0-9]*"')


@dataclass
class MapInfo:
    """What was written, so the bridge and the report use the same numbers."""

    net: str
    net_sha256: str
    xodr: str
    cx: float
    cy: float
    roads: int
    junctions: int
    controllers: int
    signal_heads: int
    signal_links: dict[str, list[str]] = field(default_factory=dict)  # kept head -> the heads it stands for
    tls_alias: dict[str, str] = field(default_factory=dict)  # short id used in the map -> SUMO tls id


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def net_centre(net_file: Path) -> tuple[float, float]:
    """Centre of the network's convBoundary (SUMO coordinates), rounded to the metre."""
    for _, el in ET.iterparse(net_file, events=("start",)):
        if el.tag == "location":
            x0, y0, x1, y1 = (float(v) for v in el.get("convBoundary").split(","))
            return round((x0 + x1) / 2), round((y0 + y1) / 2)
    raise ValueError(f"{net_file}: no <location> element")


_JUNCTION = re.compile(r'(<junction\b[^>]*\bname="([^"]*)"[^>]*>)(.*?)(\n([ \t]*)</junction>)', re.S)


def tls_nodes(net_file: Path) -> dict[str, str]:
    """SUMO node id -> id of the traffic light controlling it (from the connections)."""
    out: dict[str, str] = {}
    edges_to: dict[str, str] = {}
    for _, el in ET.iterparse(net_file, events=("end",)):
        if el.tag == "edge" and el.get("function") != "internal":
            edges_to[el.get("id")] = el.get("to")
        elif el.tag == "connection" and el.get("tl") and el.get("from") in edges_to:
            out[edges_to[el.get("from")]] = el.get("tl")
    return out


def _link_controllers(text: str, net_file: Path) -> str:
    """Reference each signal controller from the junction(s) it controls (see point 3)."""
    node_tls = tls_nodes(net_file)
    controllers = set(re.findall(r'<controller id="([^"]+)"', text))

    def add(m: re.Match) -> str:
        tls = node_tls.get(m.group(2))
        if tls is None or tls not in controllers:
            return m.group(0)
        # After the connections, as the OpenDRIVE 1.4 junction element orders its children
        # (CARLA's JunctionParser.cpp reads junction/controller wherever it is).
        ref = f'\n{m.group(5)}    <controller id="{tls}" type="0" sequence="0"/>'
        return m.group(1) + m.group(3) + ref + m.group(4)

    return _JUNCTION.sub(add, text)


_ROAD = re.compile(r"<road\b.*?</road>", re.S)
_SIGNAL = re.compile(r'([ \t]*)<signal\b([^>]*)>(.*?)</signal>\n?', re.S)
_CARRIAGEWAY = {"driving", "biking", "parking", "shoulder", "restricted", "bidirectional"}
KERB_GAP_M = 0.5
STAGGER_M = 2.0


def _attr(attrs: str, name: str) -> str | None:
    m = re.search(rf'\b{name}="([^"]*)"', attrs)
    return m.group(1) if m else None


def _set_attr(attrs: str, name: str, value: str) -> str:
    return re.sub(rf'\b{name}="[^"]*"', f'{name}="{value}"', attrs, count=1)


def _right_edge(road: ET.Element, s: float) -> float | None:
    """Lateral offset t (negative: right of the reference line) of the outer edge of the
    carriageway at s: the right lanes from -1 outwards while they carry vehicles."""
    sections = sorted(road.iter("laneSection"), key=lambda e: float(e.get("s")))
    sec = None
    for e in sections:
        if float(e.get("s")) <= s + 1e-6:
            sec = e
    right = sec.find("right") if sec is not None else None
    if right is None:
        return None
    ds = s - float(sec.get("s"))
    total = 0.0
    for lane in sorted(right.findall("lane"), key=lambda e: -int(e.get("id"))):
        if lane.get("type") not in _CARRIAGEWAY:
            break
        widths = [w for w in lane.findall("width") if float(w.get("sOffset")) <= ds + 1e-6] or lane.findall("width")
        w = widths[-1]
        x = ds - float(w.get("sOffset"))
        total += sum(float(w.get(k)) * x**i for i, k in enumerate("abcd"))
    offset = road.find("lanes/laneOffset")
    base = float(offset.get("a")) if offset is not None else 0.0
    return base - total


def _kerb_signals(text: str) -> tuple[str, dict[str, list[str]]]:
    """One traffic-light head per lane, at the kerb (see point 4). Returns the new text and
    {kept signal id: [ids it stands for, itself first]}."""
    merged: dict[str, list[str]] = {}
    removed: set[str] = set()

    def per_road(m: re.Match) -> str:
        block = m.group(0)
        heads = [h for h in _SIGNAL.finditer(block) if _attr(h.group(2), "type") == CARLA_TRAFFIC_LIGHT]
        if not heads:
            return block
        road = ET.fromstring(block)
        groups: dict[str, list[re.Match]] = {}
        for h in heads:
            v = re.search(r'fromLane="(-?\d+)"', h.group(3))
            groups.setdefault(v.group(1) if v else _attr(h.group(2), "t"), []).append(h)
        # inner lane first (-1, -2, ...): the head of the innermost lane is the one at the stop line
        order = sorted(groups, key=lambda k: -int(k) if k.lstrip("-").isdigit() else 0)
        replace: dict[int, str] = {}
        for k, lane in enumerate(order):
            hs = sorted(groups[lane], key=lambda h: _attr(h.group(2), "id"))
            keep = hs[0]
            ids = [_attr(h.group(2), "id") for h in hs]
            merged[ids[0]] = ids
            removed.update(ids[1:])
            s = float(_attr(keep.group(2), "s"))
            attrs = keep.group(2)
            edge = _right_edge(road, s) if _attr(attrs, "orientation") == "+" else None
            if edge is not None:
                # Back along the kerb; on a road too short for that, further out instead.
                back = s - STAGGER_M * k
                out = 0.0 if back >= 0.5 else STAGGER_M * k
                attrs = _set_attr(attrs, "t", f"{edge - KERB_GAP_M - out:.8f}")
                attrs = _set_attr(attrs, "s", f"{back if back >= 0.5 else s:.8f}")
            replace[keep.start()] = f"{keep.group(1)}<signal{attrs}>{keep.group(3)}</signal>\n"
            for h in hs[1:]:
                replace[h.start()] = ""
        out, pos = [], 0
        for h in _SIGNAL.finditer(block):
            out.append(block[pos:h.start()])
            out.append(replace.get(h.start(), h.group(0)))
            pos = h.end()
        out.append(block[pos:])
        return "".join(out)

    text = _ROAD.sub(per_road, text)
    if removed:
        text = re.sub(r'[ \t]*<control signalId="([^"]+)"[^>]*/>\n?',
                      lambda c: "" if c.group(1) in removed else c.group(0), text)
    return text, merged


MAX_ID = 40


def _short_ids(text: str, merged: dict[str, list[str]]) -> tuple[str, dict[str, list[str]], dict[str, str]]:
    """Rename controllers (= SUMO tls ids) longer than MAX_ID, and their signals (see point 5).
    Returns the text, `merged` with its keys renamed (values keep the SUMO names), and
    {alias: tls id}."""
    alias: dict[str, str] = {}
    for tls in sorted(set(re.findall(r'<controller id="([^"]+)">', text))):
        if len(tls) > MAX_ID:
            alias[tls] = "tls_" + hashlib.blake2b(tls.encode(), digest_size=4).hexdigest()
    for tls, short in alias.items():
        q = re.escape(tls)
        text = re.sub(rf'(<signal id="|signalId="){q}_(\d+)"', rf'\g<1>{short}_\2"', text)
        text = re.sub(rf'<controller id="{q}"', f'<controller id="{short}"', text)
    renamed = {}
    for key, ids in merged.items():
        tls, _, link = key.rpartition("_")
        renamed[f"{alias[tls]}_{link}" if tls in alias else key] = ids
    return text, renamed, {v: k for k, v in alias.items()}


def build(net_file: Path, out_dir: Path) -> MapInfo:
    """Write <name>.xodr (CARLA-ready) and <name>.map.json into out_dir."""
    net_file = Path(net_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = net_file.name.replace(".net.xml", "")
    raw = out_dir / f"{name}.raw.xodr"
    xodr = out_dir / f"{name}.xodr"
    cx, cy = net_centre(net_file)
    # netconvert adds --offset.* to every coordinate: subtract the centre.
    cmd = [
        binary("netconvert"), "-s", str(net_file),
        "--offset.x", str(-cx), "--offset.y", str(-cy),
        "--opendrive-output", str(raw),
        "--no-warnings", "true",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = raw.read_text(encoding="utf-8")
    # Plain lights may already carry 1000001; lights with arrows (1000011) are rewritten.
    text = _SIGNAL_TYPE.sub(f'type="{CARLA_TRAFFIC_LIGHT}" subtype="-1"', text)
    n_heads = text.count(f'type="{CARLA_TRAFFIC_LIGHT}"')
    text, signal_links = _kerb_signals(text)
    text = _link_controllers(text, net_file)
    text, signal_links, tls_alias = _short_ids(text, signal_links)
    xodr.write_text(text, encoding="utf-8", newline="\n")
    raw.unlink()
    info = MapInfo(
        net=net_file.name, net_sha256=sha256(net_file), xodr=xodr.name, cx=cx, cy=cy,
        roads=text.count("<road "), junctions=text.count("<junction "),
        controllers=len(re.findall(r'<controller id="[^"]+">', text)),  # not the junction references
        signal_heads=n_heads, signal_links=signal_links, tls_alias=tls_alias,
    )
    (out_dir / f"{name}.map.json").write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")
    return info


def load_info(map_json: Path) -> MapInfo:
    return MapInfo(**json.loads(Path(map_json).read_text(encoding="utf-8")))

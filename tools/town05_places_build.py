"""The places of Town05 (step 1 of giving the town's trips real destinations, 8 Nov): homes, work, shops, car parks,
the park, the hospital... each with its outline, where it is entered from the road, and where its cars park.

From what CARLA's map really holds (tools/town05_places_probe.py: out/town05_places/env_objects.json, the buildings'
meshes and boxes; sat_rgb.png, the town from above): the places are drawn around the buildings CARLA has, and the car
parks' spaces are put on the bays painted on the ground, the bays where CARLA keeps a car parked for good left out.

Writes (both in git, read at run time):
* scenarios/town05/places.json: every place (id, name, kind, outline and label point in SUMO coordinates, entrances,
  car park, capacity, what trips it makes and draws);
* scenarios/town05/town05.places.add.xml: SUMO parking areas, one per car park / kerb, with its spaces.

    python tools/town05_places_build.py            # build, check, write
    python tools/town05_places_build.py --check    # only the checks on the files written
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "scenarios" / "town05"
PLACES = SCEN / "places.json"
ADD = SCEN / "town05.places.add.xml"
PROBE = ROOT / "out" / "town05_places"

# the top view (tools/town05_places_probe.py, sat.json): full-size pixel -> SUMO
SAT_CENTRE, SAT_MPP, SAT_SIZE = (209.41, 199.645), 0.263046875, 2048

SPACE_GAP = 2.4        # two spaces' middles at least this far apart (m)
STATIC_CAR_M = 2.2     # a space this close to a car CARLA keeps parked for good is left out
KERB_SPACE_M = 6.5     # a kerb space's length
HEIGHT = {"parking": 8.0, "park": 6.0, "kerb": 3.0}   # a place with no building of its own: its label this high


def xy(u: float, v: float) -> tuple[float, float]:
    return (round(SAT_CENTRE[0] + (u - SAT_SIZE / 2) * SAT_MPP, 2), round(SAT_CENTRE[1] - (v - SAT_SIZE / 2) * SAT_MPP, 2))


def rect(u0, v0, u1, v1) -> list:
    return [xy(u0, v0), xy(u1, v0), xy(u1, v1), xy(u0, v1)]


# --------------------------------------------------------------------------- the places (from the top view)
# box: where its buildings are (full-size pixels of sat_rgb.png); meshes: which of CARLA's meshes make it up (its
# outline is drawn around them); door: where it is entered (a point by the road it is entered from)
PLACES_DEF = [
    # --- downtown
    {"id": "marche", "name": "Marché Central", "kind": "market", "sub": "Fresh market", "box": (1105, 730, 1340, 822),
         "meshes": ("SM_Shop", "Awning", "StreetCounter"), "park": "lot_marche", "hours": (7, 14), "jobs": 25, "visits": 260},
    {"id": "lot_marche", "name": "Parking du Marché", "kind": "parking", "sub": "Car park", "box": (1105, 826, 1340, 986),
         "door": (1222, 990), "lot": {"columns": (1156.7, 1218.3, 1281.7), "v0": 848.3, "v1": 956.7, "pitch": 10.84, "off": 9.9}},
    {"id": "lot_square", "name": "Parking du Square", "kind": "parking", "sub": "Car park", "box": (832, 730, 1025, 822),
         "door": (826, 792), "lot": {"rows": (776.0, 812.0), "u0": 868.0, "u1": 1004.0, "pitch": 10.6}},
    {"id": "silo", "name": "Parking Central", "kind": "parking", "sub": "Multi-storey car park", "box": (835, 826, 1025, 982),
         "door": (930, 988), "hidden": {"n": 60}},
    {"id": "atlas", "name": "Galerie Atlas", "kind": "mall", "sub": "Shopping centre and offices", "box": (1415, 740, 1605, 1290),
         "meshes": ("Office", "SM_Mall", "MetallicDoor"), "door": (1392, 880), "hidden": {"n": 70}, "hours": (9, 21), "jobs": 160,
         "visits": 420},
    {"id": "horizon", "name": "Tour Horizon", "kind": "office", "sub": "Offices", "box": (560, 700, 760, 990),
         "meshes": ("Skycraper", "MultipleFloorBuilding"), "door": (770, 840), "hidden": {"n": 40}, "hours": (8, 18), "jobs": 220},
    {"id": "jumelles", "name": "Tours Jumelles", "kind": "office", "sub": "Offices", "box": (560, 1070, 760, 1310),
         "meshes": ("Skycraper",), "door": (770, 1190), "hidden": {"n": 40}, "hours": (8, 18), "jobs": 240},
    {"id": "cafe_coin", "name": "Café du Coin", "kind": "cafe", "sub": "Café, terrace", "box": (560, 500, 720, 640),
         "meshes": ("Awning", "Block", "PlasticChair", "PlasticTable"), "park": "kerb_4", "hours": (7, 23), "jobs": 8, "visits": 90},
    {"id": "brasserie", "name": "Brasserie du Nord", "kind": "restaurant", "sub": "Brasserie", "box": (830, 500, 1040, 640),
         "meshes": ("Awning", "Block", "PlasticChair", "PlasticTable", "Kiosk"), "park": "lot_square", "hours": (11, 23), "jobs": 12,
         "visits": 120},          # Town05 has no lot of its own for a restaurant: the Square's, across the street
    {"id": "resto_parc", "name": "Restaurant du Parc", "kind": "restaurant", "sub": "Restaurant", "box": (840, 1380, 1050, 1520),
         "meshes": ("Awning", "Block", "PlasticChair", "PlasticTable"), "park": "kerb_51", "hours": (11, 23), "jobs": 10, "visits": 100},
    {"id": "jardin", "name": "Jardin des Enfants", "kind": "park", "sub": "Park and playground", "box": (1180, 1110, 1330, 1260),
         "park": "lot_marche", "visits": 80},
    # --- homes
    {"id": "oliviers", "name": "Résidence Les Oliviers", "kind": "apartments", "sub": "Apartments", "box": (830, 1080, 1040, 1300),
         "meshes": ("MultipleFloorBuilding",), "door": (1060, 1196), "hidden": {"n": 60}, "homes": 140},
    {"id": "lac", "name": "Résidence du Lac", "kind": "apartments", "sub": "Apartments", "box": (550, 1380, 740, 1530),
         "meshes": ("MultipleFloorBuilding",), "door": (760, 1450), "hidden": {"n": 36}, "homes": 70},
    {"id": "pins_res", "name": "Résidence Les Pins", "kind": "apartments", "sub": "Apartments", "box": (1640, 540, 1800, 720),
         "meshes": ("MultipleFloorBuilding",), "park": "kerb_19", "homes": 60},
    {"id": "bellevue", "name": "Résidence Bellevue", "kind": "apartments", "sub": "Apartments", "box": (1690, 780, 1800, 970),
         "meshes": ("MultipleFloorBuilding",), "park": "kerb_19", "homes": 60},
    {"id": "q_pins", "name": "Quartier des Pins", "kind": "houses", "sub": "Houses", "box": (1440, 340, 1920, 1010),
         "meshes": ("TerracedHouse", "Block", "SM_Garage"), "park": "kerb_19", "homes": 70},
    {"id": "q_cypres", "name": "Quartier des Cyprès", "kind": "houses", "sub": "Houses", "box": (1420, 1150, 1920, 1660),
         "meshes": ("TerracedHouse", "Block"), "park": "kerb_20", "homes": 90},
    # --- the kerbs where people park on the street (SUMO's parking lanes, besides the depots' slots and bus stops)
    {"id": "kerb_4", "name": "Rue du Nord", "kind": "kerb", "sub": "Street parking", "edges": ("4", "-4")},
    {"id": "kerb_51", "name": "Rue du Sud", "kind": "kerb", "sub": "Street parking", "edges": ("51", "-51")},
    {"id": "kerb_19", "name": "Allée des Pins", "kind": "kerb", "sub": "Street parking", "edges": ("19", "-19")},
    {"id": "kerb_20", "name": "Allée des Cyprès", "kind": "kerb", "sub": "Street parking", "edges": ("20", "-20")},
]

# the depots (cosim/depots.py) are places too: their own vehicles, plus people who go there
DEPOT_PLACES = [
    {"id": "hopital", "name": "Hôpital", "kind": "hospital", "sub": "Emergency 24/7", "depot": "hospital", "park": "lot_square",
         "jobs": 120, "visits": 150, "hours": (0, 24)},
    {"id": "police", "name": "Commissariat", "kind": "police", "sub": "Police station", "depot": "police", "park": "kerb_51", "jobs": 40,
         "visits": 30, "hours": (0, 24)},
    {"id": "depannage", "name": "Dépannage", "kind": "garage", "sub": "Recovery, 24/7", "depot": "tow", "park": "kerb_51", "jobs": 10,
         "visits": 10, "hours": (0, 24)},
]


def hull(points: list) -> list:
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def inside(poly: list, p) -> bool:
    x, y = p
    n, c = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1:
            c = not c
    return c


def corners(o) -> list:
    x, y = o["sumo"]
    ex, ey = o["ext"][0], o["ext"][1]
    a = math.radians(-o["yaw"])                      # CARLA yaw is clockwise seen in SUMO's frame (y flipped)
    ca, sa = math.cos(a), math.sin(a)
    return [(x + dx * ca - dy * sa, y + dx * sa + dy * ca) for dx, dy in ((ex, ey), (ex, -ey), (-ex, -ey), (-ex, ey))]


def shrink(poly, m=0.0):
    return poly


def build(net, objs) -> tuple[list, list]:
    from sumolib.geomhelper import distancePointToPolygon

    static_cars = [tuple(o["sumo"]) for o in objs if o["label"] in ("Car", "Truck")]
    lanes = [ln for e in net.getEdges() if not e.getFunction() for ln in e.getLanes() if ln.allows("passenger")]
    driving = [ln for ln in lanes if ln.getSpeed() > 3.0]

    def lane_dist(p, which=driving):
        return min(distancePointToPolygon(p, ln.getShape()) for ln in which)

    def access(door, centre, want_edges=None):
        """The lane a place is entered from: the driving lane (the rightmost of its edge) nearest the door that has the
        place on its right, and the position along it."""
        best = None
        for e in net.getEdges():
            if e.getFunction() or (want_edges and e.getID() not in want_edges):
                continue
            drv = [ln for ln in e.getLanes() if ln.allows("passenger") and ln.getSpeed() > 3.0]
            if not drv:
                continue
            ln = drv[0]                                   # the rightmost driving lane
            sh = ln.getShape()
            from sumolib.geomhelper import (
                polygonOffsetWithMinimumDistanceToPoint,
                positionAtShapeOffset,
            )
            pos = polygonOffsetWithMinimumDistanceToPoint(door, sh)
            px, py = positionAtShapeOffset(sh, pos)
            d = math.hypot(px - door[0], py - door[1])
            x2, y2 = positionAtShapeOffset(sh, min(pos + 1.0, ln.getLength()))
            dx, dy = x2 - px, y2 - py
            right = (dx * (centre[1] - py) - dy * (centre[0] - px)) < 0
            if right and d < 30.0 and (best is None or d < best[0]):
                best = (d, ln, pos)
        if best is None:
            raise RuntimeError(f"no lane by the door {door}")
        _, ln, pos = best
        L = ln.getLength()
        a, b = max(1.0, pos - 6.0), min(L - 1.0, pos + 6.0)
        if b - a < 8.0:
            a, b = max(1.0, L / 2 - 5.0), min(L - 1.0, L / 2 + 5.0)
        return {"lane": ln.getID(), "edge": ln.getEdge().getID(), "start": round(a, 1), "end": round(b, 1)}

    places, areas = [], []
    by_id = {p["id"]: p for p in PLACES_DEF}
    for d in PLACES_DEF + DEPOT_PLACES:
        p = {k: d[k] for k in ("id", "name", "kind", "sub") if k in d}
        for k in ("hours", "jobs", "visits", "homes", "depot"):
            if k in d:
                p[k] = d[k]
        if "box" in d:
            box = rect(*d["box"])
            pts = []
            if d.get("meshes"):
                for o in objs:
                    part = o["label"] == "Buildings" or any(m in o["name"] for m in ("StreetCounter", "PlasticChair",
                                                                                    "PlasticTable", "Kiosk"))
                    if part and any(m in o["name"] for m in d["meshes"]) and inside(box, o["sumo"]):
                        pts += corners(o)
            p["outline"] = [[round(x, 1), round(y, 1)] for x, y in hull(pts)] if len(pts) >= 3 else \
                [[x, y] for x, y in box]
            if d.get("meshes"):                           # each of its buildings' own footprint (CARLA's boxes)
                blocks = []
                for o in objs:
                    if (o["label"] == "Buildings" and any(m in o["name"] for m in d["meshes"]) and inside(box, o["sumo"])
                            and o["ext"][0] * o["ext"][1] * 4 >= 6.0):
                        blocks.append([[round(x, 1), round(y, 1)] for x, y in corners(o)])
                p["blocks"] = blocks
            ox = [q[0] for q in p["outline"]]
            oy = [q[1] for q in p["outline"]]
            p["label"] = [round((min(ox) + max(ox)) / 2, 1), round((min(oy) + max(oy)) / 2, 1)]
            tops = [o["carla"][2] + o["ext"][2] for o in objs if o["label"] == "Buildings" and d.get("meshes")
                    and any(m in o["name"] for m in d["meshes"]) and inside(box, o["sumo"])]
            p["height"] = round(max(tops), 1) if tops else HEIGHT.get(d["kind"], 6.0)   # its roof (m), for its label
            p["n_meshes"] = len(pts) // 4
        area = None
        if "lot" in d:                                    # painted bays: spaces on them, the bays CARLA fills left out
            lot, spaces = d["lot"], []
            if "columns" in lot:
                n = round((lot["v1"] - lot["v0"]) / lot["pitch"])
                for cu in lot["columns"]:
                    for side, ang in ((-1, 90.0), (1, 270.0)):
                        for i in range(n):
                            v = lot["v0"] + (i + 0.5) * lot["pitch"]
                            spaces.append((*xy(cu + side * lot["off"], v), ang))
            else:
                n = round((lot["u1"] - lot["u0"]) / lot["pitch"])
                for rv, ang in zip(lot["rows"], (180.0, 0.0), strict=True):
                    for i in range(n):
                        spaces.append((*xy(lot["u0"] + (i + 0.5) * lot["pitch"], rv), ang))
            free = [s for s in spaces if all(math.hypot(s[0] - c[0], s[1] - c[1]) > STATIC_CAR_M for c in static_cars)]
            p["static_taken"] = len(spaces) - len(free)
            area = {"id": f"pa_{d['id']}", "spaces": [[round(a, 2), round(b, 2), ang] for a, b, ang in free],
                    **access(xy(*d["door"]), p["label"])}
        elif "hidden" in d:                               # inside the building: the car drives in and is out of sight
            ox = [q[0] for q in p["outline"]]
            oy = [q[1] for q in p["outline"]]
            cx0, cy0 = (min(ox) + max(ox)) / 2, (min(oy) + max(oy)) / 2
            n = d["hidden"]["n"]
            k = math.ceil(math.sqrt(n))
            spaces = [(cx0 + (i % k - k / 2) * 2.5, cy0 + (i // k - k / 2) * 2.5, 0.0) for i in range(n)]
            area = {"id": f"pa_{d['id']}", "hidden": True, "spaces": [[round(a, 2), round(b, 2), ang] for a, b, ang in
                                                                       spaces], **access(xy(*d["door"]), (cx0, cy0))}
        elif "edges" in d:                                # the street's parking lane, its whole length
            kerbs = []
            for eid in d["edges"]:
                e = net.getEdge(eid)
                pl = [ln for ln in e.getLanes() if ln.allows("passenger") and ln.getSpeed() <= 3.0]
                if not pl:
                    continue
                ln = pl[0]
                L = ln.getLength()
                kerbs.append({"id": f"pa_{d['id']}_{eid}", "lane": ln.getID(), "edge": eid, "start": 3.0,
                              "end": round(L - 3.0, 1), "roadside": int((L - 6.0) // KERB_SPACE_M)})
            p["kerbs"] = kerbs
            sh = [q for k_ in kerbs for q in net.getLane(k_["lane"]).getShape()]
            p["outline"] = [[round(x, 1), round(y, 1)] for x, y in sh[::max(1, len(sh) // 12)]]
            p["label"] = [round(sum(q[0] for q in sh) / len(sh), 1), round(sum(q[1] for q in sh) / len(sh), 1)]
            areas += kerbs
            p["capacity"] = sum(k_["roadside"] for k_ in kerbs)
            p["door"] = {"edge": kerbs[0]["edge"], "lane": kerbs[0]["lane"], "pos": round(kerbs[0]["end"] / 2, 1)}
        if area is not None:
            p["parking"] = area["id"]
            p["capacity"] = len(area["spaces"])
            p["door"] = {"lane": area["lane"], "edge": area["edge"], "pos": round((area["start"] + area["end"]) / 2, 1)}
            areas.append(area)
        if "park" in d:
            p["parks_at"] = d["park"]
        places.append(p)
    # places that park elsewhere: their door is that car park's
    pid = {p["id"]: p for p in places}
    for p in places:
        if "parks_at" in p and "door" not in p:
            q = pid[p["parks_at"]]
            p["door"] = q.get("door") or {"edge": q["kerbs"][0]["edge"], "lane": q["kerbs"][0]["lane"],
                                          "pos": q["kerbs"][0]["end"] / 2}
        if p.get("depot") and "label" not in p:
            from cosim.depots import TOWN05

            plan = next(t for t in TOWN05 if t.name == p["depot"])
            e = net.getEdge(plan.edge)
            sh = e.getLanes()[plan.lane].getShape()
            p["label"] = [round(sum(q[0] for q in sh) / len(sh), 1), round(sum(q[1] for q in sh) / len(sh), 1)]
            p["outline"] = [[round(x, 1), round(y, 1)] for x, y in sh]
            p["depot_edge"] = plan.edge
    _ = by_id, lane_dist
    return places, areas


def write_add(areas: list, path: Path) -> None:
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           "<!-- The places' car parks and kerbs (tools/town05_places_build.py, scenarios/town05/places.json).",
           "     Lot spaces sit on the bays CARLA paints; hidden ones inside the building (the car drives in). -->",
           "<additional>"]
    for a in areas:
        if "spaces" in a:
            out.append(f'    <parkingArea id="{a["id"]}" lane="{a["lane"]}" startPos="{a["start"]}" endPos="{a["end"]}" '
                       f'roadsideCapacity="0" onRoad="false" friendlyPos="true">')
            for x, y, ang in a["spaces"]:
                out.append(f'        <space x="{x}" y="{y}" angle="{ang}"/>')
            out.append("    </parkingArea>")
        else:
            out.append(f'    <parkingArea id="{a["id"]}" lane="{a["lane"]}" startPos="{a["start"]}" endPos="{a["end"]}" '
                       f'roadsideCapacity="{a["roadside"]}" friendlyPos="true"/>')
    out.append("</additional>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def check(net, places: list, areas: list) -> list[str]:
    """Problems found (empty: all good)."""
    from sumolib.geomhelper import distancePointToPolygon

    sys.path.insert(0, str(ROOT / "agents" / "route_recommendation"))
    from router import Router

    r = Router()
    bad = []
    driving = [ln for e in net.getEdges() if not e.getFunction() for ln in e.getLanes()
               if ln.allows("passenger") and ln.getSpeed() > 3.0]
    ids = [p["id"] for p in places]
    if len(set(ids)) != len(ids):
        bad.append("duplicate place ids")
    for a in areas:
        ln = net.getLane(a["lane"])
        if not ln.allows("passenger"):
            bad.append(f"{a['id']}: lane {a['lane']} closed to cars")
        if not (0 <= a["start"] < a["end"] <= ln.getLength()):
            bad.append(f"{a['id']}: {a['start']}-{a['end']} not on {a['lane']} ({ln.getLength():.1f} m)")
        sp = a.get("spaces") or []
        for i, (x, y, _) in enumerate(sp):
            if not a.get("hidden") and min(distancePointToPolygon((x, y), d.getShape()) for d in driving) < 3.0:
                bad.append(f"{a['id']}: space {i} on a driving lane")
            if any(math.hypot(x - u, y - v) < SPACE_GAP for u, v, _ in sp[:i]):
                bad.append(f"{a['id']}: space {i} on another")
    import xml.etree.ElementTree as ET

    stops = [(b.get("lane"), float(b.get("startPos")), float(b.get("endPos")))
             for b in ET.parse(SCEN / "town05.pt.add.xml").getroot().iter("busStop")]
    for a in areas:                                   # never on a bus stop
        for ln, s0, s1 in stops:
            if a["lane"] == ln and a["start"] < s1 and a["end"] > s0:
                bad.append(f"{a['id']}: on the bus stop of {ln} ({s0}-{s1})")
    pid = {p["id"]: p for p in places}
    for p in places:
        if "parks_at" in p and p["parks_at"] not in pid:
            bad.append(f"{p['id']}: parks at unknown {p['parks_at']}")
        if "parking" in p:
            a = next(a for a in areas if a["id"] == p["parking"])
            if not a.get("hidden") and "outline" in p:
                out = [s for s in a["spaces"] if not inside(p["outline"], s[:2])]
                if out:
                    bad.append(f"{p['id']}: {len(out)} spaces outside its outline")
    doors = {p["id"]: p["door"]["edge"] for p in places if "door" in p}
    for a_id, a in doors.items():
        for b_id, b in doors.items():
            if a != b and r.best(a, b) is None:
                bad.append(f"no route {a_id} -> {b_id}")
    return bad


def main(argv=None) -> int:
    import sumolib

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    net = sumolib.net.readNet(str(SCEN / "town05.net.xml"))
    if a.check:
        card = json.loads(PLACES.read_text(encoding="utf-8"))
        places, areas = card["places"], card["parking_areas"]
    else:
        objs = json.loads((PROBE / "env_objects.json").read_text(encoding="utf-8"))
        places, areas = build(net, objs)
    bad = check(net, places, areas)
    for b in bad:
        print("PROBLEM", b)
    if a.check:
        return 1 if bad else 0
    card = {"what": "Town05's places (tools/town05_places_build.py)", "sat": {"centre": SAT_CENTRE, "mpp": SAT_MPP,
                                                                              "size": SAT_SIZE},
            "places": places, "parking_areas": areas}
    PLACES.write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    write_add(areas, ADD)
    kinds: dict = {}
    for p in places:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"{len(places)} places {kinds}, {len(areas)} parking areas, "
          f"{sum(len(x.get('spaces', ())) for x in areas)} spaces + {sum(x.get('roadside', 0) for x in areas)} at kerbs; "
          f"{len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

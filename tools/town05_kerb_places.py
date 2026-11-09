"""(9 Nov) Where a car can stand at each kerb of Town05 as the viewer shows it (cosim/parking_motion.py): CARLA's
avenues have a pine or a lamp post on the pavement edge every 6 m or so, and a car shown parked on one (or driving
into its place past one) went through it. Along each kerb, every 0.25 m, a car's body (5.0 m, 0.1 m kept; and a 5.5 m one's, 0.15 m) beside its
lane as SUMO puts it (2.6 m off its line; 3.1 m on the narrow streets 4 and 51) is tested against
scenarios/town05/parking_obstacles.json (cosim/obstacles.py); the places are packed from the lane's start, KERB_ROOM_M
apart at least, KERB_OUT_M kept free before the lane's end (the drive out ends there), each on a clear stretch.

    python tools/town05_kerb_places.py            -> prints them
    python tools/town05_kerb_places.py --write    -> places.json "shown_fronts" per kerb, and its capacity (SUMO's
                                                     roadsideCapacity, in town05.places.add.xml too) made the number
                                                     of places found when that is fewer
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LAT = {"4": -3.1, "-4": -3.1, "51": -3.1, "-51": -3.1}     # SUMO's side offset of a parked car (measured), else
LAT_DEFAULT = -2.6
CAR_L = 5.0
CAR_L_LONG = 5.5    # (9 Nov, 6th) a longer car's rear too (its front where a 5 m car's is: a 5.4 m car's rear
                    # touched a lamp post, Rue 20)
ROOM = 6.0          # front to front: 1.0 m or more between two cars
STEP = 0.25


def places_of(aid, a, net, grid):
    import sumolib

    from cosim import obstacles as po
    from cosim import parking_motion as pm

    lane = net.getLane(a["lane"])
    sh, L = lane.getShape(), lane.getLength()
    lat = LAT.get(a["edge"], LAT_DEFAULT)

    def clear(front):
        mid = front - CAR_L / 2
        x0, y0 = sumolib.geomhelper.positionAtShapeOffset(sh, max(mid, 0.0))
        x1, y1 = sumolib.geomhelper.positionAtShapeOffset(sh, min(mid + 1.0, L))
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        u = ((x1 - x0) / n, (y1 - y0) / n)
        mx, my = x0 - u[1] * lat, y0 + u[0] * lat
        face = pm._angle(*u)
        # (the car's body, and the metre before and after it it swings through pulling in and out)
        mx2, my2 = mx - u[0] * (CAR_L_LONG - CAR_L) / 2, my - u[1] * (CAR_L_LONG - CAR_L) / 2
        return not grid.touching(po.car_box(mx, my, face, CAR_L, margin=-0.1)) and \
            not grid.touching(po.car_box(mx2, my2, face, CAR_L_LONG, margin=-0.15)) and \
            not grid.touching(po.car_box(mx, my, face, CAR_L + 2.0, width=1.2))

    lo, hi = max(a["start"], 0.5) + CAR_L, L - pm.KERB_OUT_M
    out, f = [], lo
    while f <= hi:
        if clear(f):
            out.append(round(f, 2))
            f += ROOM if L > 60.0 else ROOM + 1.0      # (the short streets: 1 m more, the cars pull out in less room)
        else:
            f += STEP
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true")
    a_ = ap.parse_args(argv)
    import sumolib

    from cosim import obstacles as po

    pj = ROOT / "scenarios" / "town05" / "places.json"
    add = ROOT / "scenarios" / "town05" / "town05.places.add.xml"
    card = json.loads(pj.read_text(encoding="utf-8"))
    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    ent = json.loads((pj.with_name("entrances.json")).read_text(encoding="utf-8"))
    skip = set(ent.get("removed", ())) | set(ent.get("gates", {}))
    grid = po.Grid([o for o in po.load_file() if o[0] not in skip])
    xml = add.read_text(encoding="utf-8")
    for a in card["parking_areas"]:
        if not a.get("roadside") or "spaces" in a or a.get("hidden"):
            continue
        fronts = places_of(a["id"], a, net, grid)
        n0 = int(a["roadside"])
        print(f"{a['id']}: {len(fronts)} places clear (SUMO {n0}): {fronts}")
        if a_.write:
            a.pop("_lat", None)
            n = min(n0, len(fronts))
            a["shown_fronts"] = fronts
            a["shown_why"] = ("9 Nov: the places as the viewer shows them, clear of the pines and lamp posts on the "
                              "pavement edge (tools/town05_kerb_places.py)")
            if n < n0:
                a["roadside_was"] = n0
                a["roadside"] = n
                xml, k = re.subn(rf'(<parkingArea id="{re.escape(a["id"])}"[^>]*roadsideCapacity=")\d+(")',
                                 rf"\g<1>{n}\g<2>", xml)
                assert k == 1, a["id"]
            for p in card["places"]:
                if p.get("parking") == a["id"] and "capacity" in p:
                    p["capacity"] = a["roadside"]
    if a_.write:
        pj.write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
        add.write_text(xml, encoding="utf-8")
        print("->", pj, add)
    return 0


if __name__ == "__main__":
    sys.exit(main())

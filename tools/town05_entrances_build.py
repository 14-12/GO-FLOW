"""The ways into the car parks inside buildings (9 Nov), from what CARLA has there (tools/town05_entrance_probe.py ->
out/town05_places/entrances_probe.json): scenarios/town05/entrances.json, read by the viewer (cosim/parking_motion.py),
and a picture of each way from above (docs/img/town05_entrances.png).

    python tools/town05_entrances_build.py [--probe out/town05_places/entrances_probe.json] [--write]

For each building, from CARLA's vertical rays every 0.6 m: where the ground is, what stands between 0.3 m and 2.3 m
over it (a wall, a fence, a post, a parked car: no way through), and what covers it (a floor or a roof, at least
2.3 m up). A car's way (1.1 m kept from anything standing) from the road beside the car park to a covered place at
least DEEP_M from any open sky: out of sight from above and from the street. A building's door (the Galerie Atlas's
metal shutters) counts as open when the probe saw room behind it; the viewer opens it as a car comes.

--write also moves each such car park's stop on its lane (SUMO, scenarios/town05/town05.places.add.xml and
places.json) to where its way leaves the road, ROAD_ON_M before it, as the viewer's drives expect (a car stops, drives
on along the road, turns in).
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "out" / "town05_places" / "entrances_probe.json"
OUT = ROOT / "scenarios" / "town05" / "entrances.json"
FIG = ROOT / "docs" / "img" / "town05_entrances.png"

LOW_M, HIGH_M = 0.3, 2.3     # standing between these over the ground: in a car's way; covering above HIGH_M
CLEAR_M = 1.1                # a car's half width and a little: kept from anything standing
ROAD_ON_M = 9.0              # as cosim/parking_motion.ROAD_ON_M
ROAD_FREE_M = 10.0           # a way in is on the road only for this long (leaving its lane)
OTHER_LANE_M = 15.0          # a way from another road than the car park's own: as if this much longer
DEEP_M = 7.0                 # where a car stops inside: this far from open sky (out of sight from above and the street)
GROUND_LABELS = {1, 2, 10, 24, 25, 3, 22, 20}   # roads, sidewalks, terrain, road lines, ground, building floors...
ROAD_LABELS = {1, 24}


def cell_kinds(area):
    """Per cell: (ground z or None, blocked, covered, on a road)."""
    out = []
    side = area.get("side")
    for j, row in enumerate(area["cols"]):
        r = []
        for i, hits in enumerate(row):
            # the ground: the highest surface within 1 m of the street (a road at 0.15 m, the car park's floor at
            # 0.6 m; under it CARLA has a plane at -0.3 m, label 0: taken for the ground, it made everything a step)
            g = None
            for z, lab in hits:
                if -1.0 <= z <= 1.0 and (g is None or z > g[0]):
                    g = (z, lab)
            if g is None:
                r.append((None, True, False, False))
                continue
            gz = g[0]
            blocked = any(gz + LOW_M < z < gz + HIGH_M for z, _l in hits) or (side is not None and side[j][i] == "1")
            covered = any(gz + HIGH_M <= z <= gz + 60.0 for z, _l in hits)
            r.append((gz, blocked, covered, g[1] in ROAD_LABELS))
        out.append(r)
    return out


def dilate(blocked, nx, ny, r_cells):
    out = [[False] * nx for _ in range(ny)]
    offs = [(di, dj) for di in range(-r_cells, r_cells + 1) for dj in range(-r_cells, r_cells + 1)
            if di * di + dj * dj <= r_cells * r_cells]
    for j in range(ny):
        for i in range(nx):
            if blocked[j][i]:
                for di, dj in offs:
                    a, b = i + di, j + dj
                    if 0 <= a < nx and 0 <= b < ny:
                        out[b][a] = True
    return out


def depth_under_cover(kinds, nx, ny, step):
    """For each covered cell: how far (m) to the nearest cell open to the sky (by 8-neighbour steps)."""
    INF = float("inf")
    d = [[INF] * nx for _ in range(ny)]
    q = []
    for j in range(ny):
        for i in range(nx):
            if not kinds[j][i][2]:
                d[j][i] = 0.0
                q.append((0.0, i, j))
    heapq.heapify(q)
    while q:
        dd, i, j = heapq.heappop(q)
        if dd > d[j][i]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny:
                nd = dd + step * math.hypot(di, dj)
                if nd < d[b][a]:
                    d[b][a] = nd
                    heapq.heappush(q, (nd, a, b))
    return d


def lane_cells(shape, lane_from, lane_to):
    """Points under a lane (its shape) between two positions along it (m), every 0.5 m: (pos, x, y)."""
    out, acc = [], 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(shape):
        seg = math.dist((x0, y0), (x1, y1))
        k = 0.0
        while k <= seg:
            pos = acc + k
            if lane_from <= pos <= lane_to:
                x, y = x0 + (x1 - x0) * k / seg, y0 + (y1 - y0) * k / seg
                out.append((pos, x, y))
            k += 0.5
        acc += seg
    return out


def find_way(area, free, kinds, depth, lanes, deep_m=DEEP_M):
    """The shortest way (cells) from one of the lanes (id, edge, shape, own?) to a covered cell DEEP_M under cover,
    over free cells, on the road only for its first ROAD_FREE_M (a car leaves its lane for the building on its
    right; it does not cross the road or drive along others). (cells, lane id, edge, position on it) or None."""
    nx, ny, step = area["nx"], area["ny"], area["step"]
    x0, y0 = area["x0"], area["y0"]
    INF = float("inf")
    dist = [[INF] * nx for _ in range(ny)]
    base = [[0.0] * nx for _ in range(ny)]       # the cost the way had where it started
    prev = {}
    q = []
    for lane_id, edge, shape, own in lanes:
        L = _length(shape)
        # (SUMO stops the car ROAD_ON_M before it and puts it back as far past it: room on the lane for both)
        for pos, x, y in lane_cells(shape, ROAD_ON_M + 2.7, L - ROAD_ON_M - 2.8):
            i, j = round((x - x0) / step), round((y - y0) / step)
            c0 = 0.0 if own else OTHER_LANE_M
            if 0 <= i < nx and 0 <= j < ny and free[j][i] and dist[j][i] > c0:
                dist[j][i] = c0
                base[j][i] = c0
                prev[(i, j)] = ("lane", lane_id, edge, pos, c0)
                q.append((c0, i, j))
    heapq.heapify(q)
    goal = None
    while q:
        dd, i, j = heapq.heappop(q)
        if dd > dist[j][i]:
            continue
        if depth[j][i] >= deep_m:
            goal = (i, j)
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[b][a]:
                nd = dd + step * math.hypot(di, dj)
                if kinds[b][a][3] and nd - base[j][i] > ROAD_FREE_M:
                    continue
                if nd < dist[b][a]:
                    dist[b][a] = nd
                    base[b][a] = base[j][i]
                    prev[(a, b)] = (i, j)
                    heapq.heappush(q, (nd, a, b))
    if goal is None:
        return None
    cells = [goal]
    while not isinstance(prev[cells[-1]][0], str):
        cells.append(prev[cells[-1]])
    _tag, lane_id, edge, lane_pos, _c = prev[cells[-1]]
    cells.reverse()
    return cells, lane_id, edge, lane_pos


def line_free(free, a, b):
    (i0, j0), (i1, j1) = a, b
    n = max(abs(i1 - i0), abs(j1 - j0)) * 2 + 1
    for k in range(n + 1):
        i = round(i0 + (i1 - i0) * k / n)
        j = round(j0 + (j1 - j0) * k / n)
        if not free[j][i]:
            return False
    return True


def smooth(cells, free):
    """String pulling: the fewest corners with a free straight line between them."""
    out = [cells[0]]
    k = 0
    while k < len(cells) - 1:
        m = len(cells) - 1
        while m > k + 1 and not line_free(free, cells[k], cells[m]):
            m -= 1
        out.append(cells[m])
        k = m
    return out


# (9 Nov) two jeeps CARLA keeps parked for good in the Parking du Square's south row, side by side in front of
# Parking Central's open ground floor: the only way into it is between them and the guard hut. The viewer takes them
# away (world.enable_environment_objects) and the ground there is a way.
REMOVED = ("Vh_Car_JeepWranglerRubicon_NOrig2_SM_0", "Vh_Car_JeepWranglerRubicon_NOrig_560_SM_0")


def removed_boxes(env_objects, names=REMOVED):
    """The CARLA objects the viewer takes away: name -> [x, y, half x, half y] (SUMO)."""
    return {n: b for n, b in gate_boxes(env_objects, names).items() if n in names}


def gate_boxes(env_objects, names=("ParkingBarrier",)):
    """The barrier arms (and other gates) a car passes once opened: name -> [x, y, half x, half y] (SUMO)."""
    out = {}
    for o in env_objects:
        if not any(n in o["name"] for n in names):
            continue
        ex, ey = o["ext"][0], o["ext"][1]
        across_x = abs(math.cos(math.radians(o["yaw"]))) > 0.7
        hx, hy = (ex, ey) if across_x else (ey, ex)
        out[o["name"]] = [round(o["sumo"][0], 2), round(o["sumo"][1], 2), round(hx, 2), round(hy, 2)]
    return out


def standing_points(pillars):
    """The points where tools/town05_pillars_probe.py saw something standing (SUMO x, y)."""
    if not pillars:
        return []
    c, x0, y0 = pillars["cell"], pillars["x0"], pillars["y0"]
    fx0, fx1, fy0, fy1 = pillars.get("floor", (158.6, 209.2, 211.8, 250.6))
    out = []
    for j, (row, seen) in enumerate(zip(pillars["stand"], pillars["seen"], strict=True)):
        for i, (ch, sn) in enumerate(zip(row, seen, strict=True)):
            x, y = x0 + i * c, y0 + j * c
            # something standing; or, on the floor, not seen at all: inside a pillar (the camera sees its sides)
            if ch == "1" or (sn == "0" and fx0 <= x <= fx1 and fy0 <= y <= fy1):
                out.append((x, y))
    return out


def build(probe, deep_m=DEEP_M, lanes_of=None, gates=None, standing=(), removed=None):
    """{area: way in} from the probe. lanes_of(area) -> [(lane id, edge, shape, own?)] the lanes a way may start
    from (default: the car park's own lane); gates: name -> box, passable (opened by the viewer)."""
    res = {}
    gates = dict(gates or {})
    for area in probe["areas"].values():         # (the probe gives the grid's step once, for all)
        area.setdefault("step", probe.get("step", 0.6))
    for aid, area in probe["areas"].items():
        nx, ny, step = area["nx"], area["ny"], area["step"]
        x0, y0 = area["x0"], area["y0"]
        kinds = cell_kinds(area)
        passable = dict(gates)
        for name, dd in (probe.get("doors") or {}).items():    # a door with room behind it: a way through it
            rays = (probe.get("door_rays") or {})
            # each side: the nearest thing behind the door over the three heights (no hit: nothing in 45 m)
            room = [min((hs[0][0] if hs else 45.0) for hs in rays.get(f"{name}_{s}", [[]] * 3)) for s in "ab"]
            if max(room) >= 9.0 + 8.0:            # (rays from 9 m out: 8 m behind the door at least)
                ex, ey = dd["ext"][0], dd["ext"][1]
                across_x = abs(math.cos(math.radians(dd["yaw"]))) > 0.7
                hx, hy = (ex, ey) if across_x else (ey, ex)
                passable[name] = [dd["sumo"][0], dd["sumo"][1], hx, hy]
        for _name, (sx, sy, hx, hy) in passable.items():
            if not (x0 - hx <= sx <= x0 + nx * step + hx and y0 - hy <= sy <= y0 + ny * step + hy):
                continue
            for j in range(ny):
                for i in range(nx):
                    x, y = x0 + i * step, y0 + j * step
                    if abs(x - sx) <= hx + 0.5 and abs(y - sy) <= hy + 0.5:
                        g = kinds[j][i]
                        kinds[j][i] = (g[0] if g[0] is not None else 0.0, False, g[2], False)
        blocked = [[k[1] or k[0] is None for k in row] for row in kinds]
        for _n, (sx, sy, hx, hy) in (removed or {}).items():      # (taken away by the viewer)
            for j in range(ny):
                for i in range(nx):
                    x, y = x0 + i * step, y0 + j * step
                    if abs(x - sx) <= hx + 0.4 and abs(y - sy) <= hy + 0.4:
                        blocked[j][i] = False
        standing = [(sx, sy) for sx, sy in standing if not any(
            abs(sx - bx) <= bhx + 0.4 and abs(sy - by) <= bhy + 0.4 for bx, by, bhx, bhy in (removed or {}).values())]
        for sx, sy in standing:             # (what the depth camera saw standing under a floor: pillars)
            for i in (math.floor((sx - x0) / step), math.ceil((sx - x0) / step)):
                for j in (math.floor((sy - y0) / step), math.ceil((sy - y0) / step)):
                    if 0 <= i < nx and 0 <= j < ny:
                        blocked[j][i] = True
        nb = dilate(blocked, nx, ny, max(1, round(CLEAR_M / step)))
        free = [[not nb[j][i] for i in range(nx)] for j in range(ny)]
        depth = depth_under_cover(kinds, nx, ny, step)
        lanes = lanes_of(area) if lanes_of else [(area["lane"], area["lane"].rsplit("_", 1)[0], area["lane_shape"],
                                                  True)]
        found = find_way(area, free, kinds, depth, lanes, deep_m)
        entry = {"place": area["place"], "found": found is not None}
        if found is not None:
            cells, lane_id, edge, lane_pos = found
            L = _length(next(sh for lid, _e, sh, _o in lanes if lid == lane_id))
            pts = [(x0 + i * step, y0 + j * step) for i, j in smooth(cells, free)]
            raw = [(x0 + i * step, y0 + j * step) for i, j in cells]
            # the opening: the first covered cell of the way; front: 5 m before it (outside)
            k_in = next((n for n, (i, j) in enumerate(cells) if kinds[j][i][2]), len(cells) - 1)
            thr = raw[k_in]
            k_front = max(0, k_in - round(5.0 / step))
            front = raw[k_front]
            way = [thr] + [p for p in pts if _along(raw, p) > _along(raw, thr) + 0.5]
            # from the road to the front: the way's corners before it (the road's point itself is the viewer's)
            approach = [p for p in pts[1:] if 0.5 < _along(raw, p) < _along(raw, front) - 0.5]
            crossed = sorted(n for n, (sx, sy, hx, hy) in passable.items()
                             if any(abs(p[0] - sx) <= hx + 1.0 and abs(p[1] - sy) <= hy + 1.0 for p in raw))
            # SUMO: a car stops ROAD_ON_M (its middle) before where its way leaves the road, and is put back as far
            # past it, as the viewer's drives start and end (cosim/parking_motion.py)
            half = 2.3
            stop_end = lane_pos - ROAD_ON_M + half
            depart = min(lane_pos + ROAD_ON_M + half, L - 0.5)
            entry.update(lane=lane_id, edge=edge, front=[round(v, 2) for v in front],
                         way=[[round(v, 2) for v in p] for p in way],
                         approach=[[round(v, 2) for v in p] for p in approach],
                         door=next((n for n in crossed if "Door" in n), None), gates=crossed,
                         lane_pos=round(lane_pos, 1), stop_end=round(stop_end, 1),
                         depart_pos=round(depart, 1), length_m=round(_length(raw), 1),
                         cover_depth_m=round(depth[cells[-1][1]][cells[-1][0]], 1),
                         raw=[[round(v, 2) for v in p] for p in raw])
        res[aid] = entry
    return res, passable_all(probe, gates)


def passable_all(probe, gates):
    """Every gate the viewer may open: the barrier arms, and the doors with room behind them."""
    out = dict(gates or {})
    for name, dd in (probe.get("doors") or {}).items():
        rays = (probe.get("door_rays") or {})
        room = [min((hs[0][0] if hs else 45.0) for hs in rays.get(f"{name}_{s}", [[]] * 3)) for s in "ab"]
        if max(room) >= 17.0:
            ex, ey = dd["ext"][0], dd["ext"][1]
            across_x = abs(math.cos(math.radians(dd["yaw"]))) > 0.7
            hx, hy = (ex, ey) if across_x else (ey, ex)
            out[name] = [round(dd["sumo"][0], 2), round(dd["sumo"][1], 2), round(hx, 2), round(hy, 2)]
    return out


def net_lanes(net_file):
    """lanes_of for build(): every edge's rightmost driving lane in the area's grid (not the ring road's fast
    lanes), the car park's own first."""
    import sumolib

    net = sumolib.net.readNet(str(net_file))
    lanes = []
    for e in net.getEdges():
        if e.getID().startswith(":"):
            continue
        cand = [ln for ln in e.getLanes() if ln.allows("passenger") and 5.0 < ln.getSpeed() <= 14.5]
        if cand:
            ln = min(cand, key=lambda ln: ln.getIndex())
            lanes.append((ln.getID(), e.getID(), [tuple(p) for p in ln.getShape()]))

    def lanes_of(area):
        x0, y0 = area["x0"], area["y0"]
        x1, y1 = x0 + area["nx"] * area["step"], y0 + area["ny"] * area["step"]
        out = []
        for lid, eid, sh in lanes:
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in sh):
                out.append((lid, eid, sh, lid == area["lane"]))
        return sorted(out, key=lambda r: not r[3])
    return lanes_of


def _length(pts):
    return sum(math.dist(a, b) for a, b in itertools.pairwise(pts))


def _along(raw, p):
    k = min(range(len(raw)), key=lambda n: math.dist(raw[n], p))
    return _length(raw[:k + 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", default=str(PROBE))
    ap.add_argument("--deep", type=float, default=DEEP_M)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--figure", action="store_true", help="out/town05_places/entrances.png")
    a = ap.parse_args(argv)
    probe = json.loads(Path(a.probe).read_text(encoding="utf-8"))
    env = json.loads((ROOT / "out" / "town05_places" / "env_objects.json").read_text(encoding="utf-8"))
    pil = ROOT / "out" / "town05_places" / "silo_pillars.json"
    standing = standing_points(json.loads(pil.read_text(encoding="utf-8"))) if pil.is_file() else []
    print(f"{len(standing)} points with something standing under a floor ({pil.name})")
    res, gates = build(probe, a.deep, lanes_of=net_lanes(ROOT / "scenarios" / "town05" / "town05.net.xml"),
                       gates=gate_boxes(env), standing=standing, removed=removed_boxes(env))
    for aid, e in res.items():
        print(aid, {k: v for k, v in e.items() if k not in ("raw", "way", "approach")})
    if a.write:
        OUT.write_text(json.dumps({"what": __doc__.split("\n\n")[0], "entrances": {
            aid: {k: v for k, v in e.items() if k != "raw"} for aid, e in res.items() if e["found"]},
            "gates": {n: {"box": b} for n, b in sorted(gates.items())},
            "removed": sorted(removed_boxes(env))}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print("->", OUT)
        move_stops({aid: e for aid, e in res.items() if e["found"]})
    if a.figure:
        figure(probe, res, gates)
    return 0


def move_stops(found):
    """SUMO: each building's car park stops its cars before its way in and puts them back past it (startPos,
    endPos, departPos), in places.json and town05.places.add.xml."""
    import re

    pj = ROOT / "scenarios" / "town05" / "places.json"
    add = ROOT / "scenarios" / "town05" / "town05.places.add.xml"
    card = json.loads(pj.read_text(encoding="utf-8"))
    xml = add.read_text(encoding="utf-8")
    for a in card["parking_areas"]:
        e = found.get(a["id"])
        if e is None:
            continue
        end = e["stop_end"]
        a["start"], a["end"], a["depart"] = round(max(end - 12.0, 0.0), 1), round(end, 1), e["depart_pos"]
        a["lane"], a["edge"] = e["lane"], e["edge"]
        pat = re.compile(r'(<parkingArea id="' + re.escape(a["id"]) + r'"[^>]*?)\s*/?>', re.DOTALL)
        m = pat.search(xml)
        if m is None:
            continue
        tag = m.group(1)
        tag = re.sub(r'lane="[^"]*"', f'lane="{a["lane"]}"', tag)
        tag = re.sub(r'startPos="[^"]*"', f'startPos="{a["start"]}"', tag)
        tag = re.sub(r'endPos="[^"]*"', f'endPos="{a["end"]}"', tag)
        tag = re.sub(r'\s*departPos="[^"]*"', "", tag) + f' departPos="{a["depart"]}"'
        xml = xml[:m.start(1)] + tag + xml[m.end(1):]
        print(f"{a['id']}: on {a['lane']}, stops at {a['start']}-{a['end']}, back on the road at {a['depart']}")
    pj.write_text(json.dumps(card, indent=1, ensure_ascii=False), encoding="utf-8")
    add.write_text(xml, encoding="utf-8")


def figure(probe, res, gates):
    """Each building's way from above on the town's picture (out/town05_places/sat_rgb.png): the cells a car
    cannot be in (red), the covered ones (blue), the way (green), where it stops (white)."""
    from PIL import Image, ImageDraw

    sat = json.loads((ROOT / "out" / "town05_places" / "sat.json").read_text(encoding="utf-8"))
    im = Image.open(ROOT / "out" / "town05_places" / "sat_rgb.png").convert("RGB")
    mpp, (cx, cy), size = sat["metres_per_pixel_at_ground"], sat["sumo_centre"], sat["size"]

    def px(x, y):
        return (size / 2 + (x - cx) / mpp, size / 2 - (y - cy) / mpp)

    tiles = []
    for aid, area in probe["areas"].items():
        nx, ny, step, x0, y0 = area["nx"], area["ny"], area["step"], area["x0"], area["y0"]
        a0, a1 = px(x0, y0 + ny * step), px(x0 + nx * step, y0)
        crop = im.crop((int(a0[0]), int(a0[1]), int(a1[0]), int(a1[1])))
        k = 2.0
        crop = crop.resize((int(crop.width * k), int(crop.height * k)))
        ov = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        def P(x, y, a0=a0, k=k):
            u, v = px(x, y)
            return ((u - int(a0[0])) * k, (v - int(a0[1])) * k)
        kinds = cell_kinds(area)
        r = step / mpp * k / 2
        for j in range(ny):
            for i in range(nx):
                g, bl, cov, _road = kinds[j][i]
                u, v = P(x0 + i * step, y0 + j * step)
                if bl or g is None:
                    d.rectangle((u - r, v - r, u + r, v + r), fill=(255, 40, 40, 90))
                elif cov:
                    d.rectangle((u - r, v - r, u + r, v + r), fill=(60, 120, 255, 70))
        e = res.get(aid, {})
        for _n, (gx, gy, ghx, ghy) in gates.items():
            d.rectangle((*P(gx - ghx, gy + ghy), *P(gx + ghx, gy - ghy)), outline=(255, 200, 0, 255), width=2)
        if e.get("found"):
            d.line([P(*p) for p in e["raw"]], fill=(40, 255, 120, 255), width=4)
            for p in [e["front"], *e["way"]]:
                u, v = P(*p)
                d.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 255, 255, 255), width=2)
        crop = Image.alpha_composite(crop.convert("RGBA"), ov).convert("RGB")
        ImageDraw.Draw(crop).text((10, 10), f"{area['place']}: {'way in ' + str(e.get('length_m')) + ' m' if e.get('found') else 'no way in'}"
                                  f"{' door ' + e['door'] if e.get('door') else ''}", fill=(255, 255, 255))
        tiles.append(crop)
    if not tiles:
        return
    w = max(t.width for t in tiles)
    h = sum(t.height for t in tiles)
    sheet = Image.new("RGB", (w, h), (12, 14, 20))
    yy = 0
    for t in tiles:
        sheet.paste(t, (0, yy))
        yy += t.height
    out = ROOT / "out" / "town05_places" / "entrances.png"
    sheet.save(out)
    print("->", out)


if __name__ == "__main__":
    sys.exit(main())

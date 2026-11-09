"""(9 Nov) What a car on a parking drive could run into that CARLA has and SUMO does not: the map's parked cars and
trucks, posts, poles, signs, fences, walls, low hedges, tree trunks, street furniture (out/town05_places/env_objects.json,
from tools/town05_places_probe.py), as boxes over the ground at a car's height; and a test of a car's body (an oriented
box) against them. Used by tools/check_parking_motion.py and tools/check_parking_crowd.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "out" / "town05_places" / "env_objects.json"
SOLID = {"Car", "Truck", "Static", "Dynamic", "Poles", "TrafficLight", "TrafficSigns", "Fences", "Walls", "GuardRail",
         "Buildings", "Vegetation", "Other"}
NOT = ("Road_", "Sidewalk", "Curb", "Grass", "Terrain", "concreteFloor", "buildingTown05_456", "InstancedFoliage")
CAR_W = 1.9
LOW, HIGH = 0.4, 1.5           # a car's body over the ground (m): under LOW a car rolls over it (debris, kerbs)


def load(removed=(), gates=()):
    """[(name, x, y, half along, half across, heading rad (SUMO, x east y north))] at a car's height."""
    if not ENV.is_file():
        return []
    out = []
    objs = json.loads(ENV.read_text(encoding="utf-8"))
    heads = {}                       # a light's head (high up, same name but its last part): where its arm points
    for o in objs:
        if o["label"] == "Poles" and o["carla"][2] > 6.0 and max(o["ext"][:2]) < 1.0:
            heads.setdefault(o["name"].rsplit("_SM_", 1)[0], o["sumo"])
    for o in objs:
        if o["label"] not in SOLID or any(n in o["name"] for n in NOT) or o["name"] in removed or o["name"] in gates:
            continue
        z, ez = o["carla"][2], o["ext"][2]
        ex, ey = o["ext"][0], o["ext"][1]
        if o["label"] == "Poles" and max(ex, ey) > 0.8:
            # a post with an arm: its box is mostly the arm, over the road. The post: the box's end away from the
            # lamp head (a highway light: 7.9 m box, the post in the barrier line); without a head, its middle
            a = -math.radians(o["yaw"])
            u = (math.cos(a), math.sin(a)) if ex >= ey else (-math.sin(a), math.cos(a))
            half = max(ex, ey)
            ends = [(o["sumo"][0] + u[0] * k * half, o["sumo"][1] + u[1] * k * half) for k in (-1, 1)]
            head = heads.get(o["name"].rsplit("_SM_", 1)[0])
            px, py = max(ends, key=lambda e: math.dist(e, head)) if head else o["sumo"]
            out.append((o["name"], px, py, 0.25, 0.25, 0.0))
            continue
        if o["label"] == "Vegetation":
            if ez > 1.5:                               # a tree: its trunk, not its crown
                ex = ey = 0.3
                z, ez = 1.0, 1.0
            elif ez < 0.3:
                continue                               # (grass)
        if z - ez > HIGH or z + ez < LOW:
            continue                                   # over a car (a roof, a crown) or flat on the ground
        if max(ex, ey) > 30.0:
            continue                                   # (a whole block's box: not a solid)
        a = -math.radians(o["yaw"])                    # CARLA yaw (y south) -> SUMO (y north): the box's x axis
        out.append((o["name"], o["sumo"][0], o["sumo"][1], ex, ey, a))
    return out


def _axes(a):
    return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))


def hits(x, y, face_deg, length, obstacles, margin=0.0, near=8.0):
    """The obstacles a car's body (middle x, y, SUMO heading face_deg) overlaps, by more than `margin` (SAT)."""
    h = math.radians(face_deg)
    fx, fy = math.sin(h), math.cos(h)                 # SUMO heading: 0 north, clockwise
    car = ((x, y), (fx, fy), (fy, -fx), length / 2 - margin, CAR_W / 2 - margin)
    out = []
    for name, ox, oy, ex, ey, a in obstacles:
        if abs(ox - x) > near + ex + ey or abs(oy - y) > near + ex + ey:
            continue
        u, v = _axes(a)
        box = ((ox, oy), u, v, ex, ey)
        if _overlap(car, box):
            out.append(name)
    return out


def _overlap(b1, b2):
    (c1, u1, v1, a1, b1_), (c2, u2, v2, a2, b2_) = b1, b2
    d = (c2[0] - c1[0], c2[1] - c1[1])
    for ax in (u1, v1, u2, v2):
        r1 = a1 * abs(u1[0] * ax[0] + u1[1] * ax[1]) + b1_ * abs(v1[0] * ax[0] + v1[1] * ax[1])
        r2 = a2 * abs(u2[0] * ax[0] + u2[1] * ax[1]) + b2_ * abs(v2[0] * ax[0] + v2[1] * ax[1])
        if abs(d[0] * ax[0] + d[1] * ax[1]) > r1 + r2:
            return False
    return True

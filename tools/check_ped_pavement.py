"""Where do the free-fly viewer's pedestrians stand? SUMO alone, CARLA's Town05 map built offline (no server).

    python tools/check_ped_pavement.py [--seconds 900]

Runs Town05 with the viewer's pedestrians (routed trips over the signalised crosswalks, actuated signals) and
places every person every step the way the viewer does (cosim.carla_side.CarlaSide.walker_pose with its
glide, height left out), and looks every second. For each placed point it asks CARLA's map what is under it. A person who is not on a crossing
should be on a sidewalk; "in the road" is a driving lane, shoulder, parking or bike lane, or no lane at all
inside a junction. Prints the counts for people standing still (waiting) and for everyone off the crossings.
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import carla  # noqa: E402

from cosim import scenario as scenario_mod  # noqa: E402
from cosim.carla_side import CarlaSide  # noqa: E402
from cosim.sumo_side import SumoSide  # noqa: E402


class _Placer(CarlaSide):
    """walker_pose without a server: the map only, flat ground."""

    def __init__(self, cmap, frame):  # noqa: D107 - no CarlaSide.__init__ (it needs a world)
        self.map, self.frame = cmap, frame

    def _surface_z(self, cx, cy, z_lane, guess=0.0):
        return z_lane


def _inside(cmap, loc, lane_type) -> bool:
    """Within the lateral extent of the nearest lane of that type (get_waypoint without projection
    sometimes answers None for points well inside a sidewalk)."""
    wp = cmap.get_waypoint(loc, project_to_road=True, lane_type=lane_type)
    if wp is None:
        return False
    c = wp.transform.location
    r, f = wp.transform.get_right_vector(), wp.transform.get_forward_vector()
    dx, dy = loc.x - c.x, loc.y - c.y
    side = abs(dx * r.x + dy * r.y) / (math.hypot(r.x, r.y) or 1.0)
    along = abs(dx * f.x + dy * f.y) / (math.hypot(f.x, f.y) or 1.0)
    return side <= wp.lane_width / 2.0 + 1e-3 and along <= 0.5


def under(cmap, x: float, y: float) -> str:
    loc = carla.Location(x, y, 0.0)
    if _inside(cmap, loc, carla.LaneType.Sidewalk):
        return "Sidewalk"
    for t in (carla.LaneType.Driving, carla.LaneType.Shoulder, carla.LaneType.Parking, carla.LaneType.Biking):
        if _inside(cmap, loc, t):
            return str(t)
    return "no lane (kerb corner or off the map's lanes)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--port", type=int, default=9790)
    a = ap.parse_args(argv)
    scen = scenario_mod.load("town05", ROOT)
    cmap = carla.Map("Town05", (scen.folder / "town05.xodr").read_text(encoding="utf-8"))
    placer = _Placer(cmap, scen.frame)
    s = SumoSide(scen.folder, "actuated", seed=7, port=a.port, label="pave", extras=True, emergency_every=1e9,
                 extra_args=["--scale", "0.5", "--end", f"{a.seconds + 10:g}"])
    s.extras.config.pedestrian_trips = True
    waiting, walking = collections.Counter(), collections.Counter()
    try:
        while s.time < a.seconds:
            r = s.step()
            placed = {p.id: placer.walker_pose(p.x, p.y, p.angle, p.road, pid=p.id) for p in r.persons.values()}
            placer.forget_walkers(placed)
            if int(s.time * 2) % 2:
                continue
            for p in r.persons.values():
                if p.road.startswith(":") and "_c" in p.road:
                    continue  # on a crossing: on the road by design
                tf = placed[p.id]
                kind = under(cmap, tf.location.x, tf.location.y)
                (waiting if p.speed < 0.1 else walking)[kind] += 1
    finally:
        s.close()
    for name, c in (("standing still", waiting), ("walking, off the crossings", walking)):
        n = sum(c.values())
        road = n - c.get("Sidewalk", 0)
        print(f"{name}: {n} samples, {road} not on a sidewalk ({100 * road / max(n, 1):.1f} %): {dict(c.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

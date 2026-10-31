"""The direction of a SUMO lane at a position along it (SUMO's angle convention: degrees, 0 = north, clockwise).

The free-fly viewer turns each car's body along the path its front has taken (cosim/interp.Trails). SUMO's
sublane model moves cars sideways at low speed (a short edge after a junction, a car that must reach another
lane), and a body following that sideways path stood up to 49 deg across the lane, poking into the next one
while SUMO, which keeps every body parallel to its lane, let a neighbour pass at 3 m (tools/check_overlaps.py).
The body is therefore kept within Trails.MAX_SLANT_DEG of its lane's own direction (12 deg when slow)."""

from __future__ import annotations

import bisect
import math


class LaneGeometry:
    def __init__(self, net_file):
        import sumolib

        net = sumolib.net.readNet(str(net_file), withInternal=True)
        self.lanes: dict[str, tuple[list[float], list[float]]] = {}
        for e in net.getEdges(withInternal=True):
            for ln in e.getLanes():
                shape = ln.getShape()
                cum, ang = [0.0], []
                for (ax, ay), (bx, by) in zip(shape, shape[1:], strict=False):
                    cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
                    ang.append(math.degrees(math.atan2(bx - ax, by - ay)) % 360.0)
                if ang:
                    # SUMO scales a lane's shape to its length: positions are scaled the same way
                    length = ln.getLength()
                    k = length / cum[-1] if cum[-1] > 0 else 1.0
                    self.lanes[ln.getID()] = ([c * k for c in cum], ang)

    def angle(self, lane: str, pos: float) -> float | None:
        g = self.lanes.get(lane)
        if g is None:
            return None
        cum, ang = g
        i = min(max(bisect.bisect_right(cum, pos) - 1, 0), len(ang) - 1)
        return ang[i]

"""Coordinates between SUMO and CARLA for a map written by cosim.opendrive.

SUMO: x east, y north, right-handed, a vehicle's position is the middle of its FRONT bumper,
heading in degrees clockwise from north. CARLA (Unreal): x east, y SOUTH (left-handed),
a vehicle's location is the centre of its bounding box, yaw in degrees clockwise from +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Frame:
    """The map's shift, from cosim.opendrive.MapInfo."""

    cx: float
    cy: float

    def to_carla_xy(self, x: float, y: float) -> tuple[float, float]:
        return x - self.cx, -(y - self.cy)

    def to_sumo_xy(self, x: float, y: float) -> tuple[float, float]:
        return x + self.cx, -y + self.cy


def sumo_angle_to_carla_yaw(angle_deg: float) -> float:
    """SUMO heading (0 = north, clockwise) -> CARLA yaw (0 = +x east, clockwise seen from
    above in Unreal's left-handed frame, i.e. 90 = south)."""
    return (angle_deg - 90.0 + 180.0) % 360.0 - 180.0


def front_to_centre(x: float, y: float, angle_deg: float, half_length: float) -> tuple[float, float]:
    """SUMO reports the front bumper; CARLA places the box centre half a length behind it."""
    a = math.radians(angle_deg)
    return x - math.sin(a) * half_length, y - math.cos(a) * half_length

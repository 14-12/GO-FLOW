"""Sun position for the scenario clock, so CARLA's light matches the time SUMO simulates.

Same low-precision almanac formulae as the main project's twin (environment.ts), which were
checked there against the `astral` library to within 0.01 deg.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def solar_position(utc: datetime, lat: float, lon: float) -> tuple[float, float]:
    """(azimuth deg clockwise from north, elevation deg) of the sun."""
    rad = math.pi / 180
    d = utc.timestamp() / 86400.0 + 2440587.5 - 2451545.0
    g = (357.529 + 0.98560028 * d) * rad
    q = 280.459 + 0.98564736 * d
    L = (q + 1.915 * math.sin(g) + 0.02 * math.sin(2 * g)) * rad
    e = (23.439 - 0.00000036 * d) * rad
    ra = math.atan2(math.cos(e) * math.sin(L), math.cos(L))
    dec = math.asin(math.sin(e) * math.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = (gmst * 15 + lon) * rad - ra
    phi = lat * rad
    el = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha))
    az = math.atan2(-math.sin(ha), math.tan(dec) * math.cos(phi) - math.sin(phi) * math.cos(ha))
    return (math.degrees(az) + 360) % 360, math.degrees(el)


def scenario_sun(seconds_of_day: float, lat: float, lon: float, month: int = 6, day: int = 21,
                 utc_offset_h: float = 2.0, year: int = 2026) -> tuple[float, float]:
    """Sun for a scenario clock. The RESCO scenarios carry no date: 21 June (CEST) is
    assumed, as in the main project, and reported as an assumption."""
    base = datetime(year, month, day, tzinfo=timezone.utc).timestamp()
    t = datetime.fromtimestamp(base + seconds_of_day - utc_offset_h * 3600, tz=timezone.utc)
    return solar_position(t, lat, lon)


def carla_sun_azimuth(bearing_deg: float) -> float:
    """CARLA's `sun_azimuth_angle` for a sun at this compass bearing (clockwise from north).

    Measured on CARLA 0.9.16 (tools/check_sun.py, out/sun_check.json): with the sun at 55 deg
    altitude in Town10HD_Opt, a truck's shadow gives sun bearing = azimuth + 268.3 deg (fit over
    7 settings, max residual 16.9 deg, the truck's box shape biasing the oblique ones); the two
    axis-aligned settings give 269.0 (azimuth 0) and 90.8 (azimuth 180). So bearing =
    azimuth - 90, i.e. azimuth = bearing + 90. Image orientation was checked in the same run
    with a car placed 9 m along +x (seen at 8.74, 0.16 m).
    """
    return (bearing_deg + 90.0) % 360.0

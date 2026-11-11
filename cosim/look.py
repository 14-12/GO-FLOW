"""The town's look: CARLA's weather for the time of day (the sky and its clouds).

The viewer used one weather all day (run.WEATHER["clear"]) and only moved the sun. `weather_fields` gives the
fields of carla.WeatherParameters for a sun elevation; the sun's own angles are set by the caller.
"""

from __future__ import annotations

OLD = dict(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wetness=0.0, fog_density=0.0)


def _mix(a: float, b: float, f: float) -> float:
    f = max(0.0, min(1.0, f))
    return a + (b - a) * f


def weather_fields(sun_elevation: float, hour: float = 12.0, look: str = "town") -> dict:
    """The fields of carla.WeatherParameters for this sun (degrees above the horizon). Chosen on the look lab's
    sheets (tools/look_lab.py --sweep, 11 Nov): by day more clouds and a deeper blue (less Rayleigh scattering);
    towards sunrise and sunset more Rayleigh scattering (an orange sky) and fewer clouds (many of them darken the
    low sun's sky to grey); at night few clouds. Fog, wetness and the map's 82 "building" lights changed nothing
    worth having on the sheets and are left alone.
    The other fields are CARLA's own defaults (the old weather left them at 0)."""
    if look == "old":
        return dict(OLD)
    el = sun_elevation
    low = 1.0 - max(0.0, min(1.0, (el - 3.0) / 19.0))  # 1 at the horizon, 0 from 22 deg up
    night = max(0.0, min(1.0, (-el - 2.0) / 6.0))      # 0 until the sun is 2 deg down, 1 from 8 deg down
    return dict(cloudiness=_mix(_mix(28.0, 16.0, low), 12.0, night), precipitation=0.0, precipitation_deposits=0.0,
                wetness=0.0, wind_intensity=5.0, fog_density=3.0, fog_distance=0.75, fog_falloff=0.9,
                scattering_intensity=1.0, mie_scattering_scale=0.03,
                rayleigh_scattering_scale=_mix(_mix(0.026, 0.05, low), 0.0331, night), dust_storm=0.0)


def light_report(world) -> dict:
    """How many lights the map has in each of CARLA's groups, how many are on, their mean intensity."""
    import carla

    lm = world.get_lightmanager()
    out = {}
    for name in ("Street", "Building", "Other", "Vehicle", "NONE"):
        g = getattr(carla.LightGroup, name, None)
        if g is None:
            continue
        ls = lm.get_all_lights(g)
        on = [x for x in ls if x.is_on]
        out[name] = {"lights": len(ls), "on": len(on),
                     "intensity": round(sum(x.intensity for x in ls) / len(ls), 1) if ls else None,
                     "colour": (lambda c: (c.r, c.g, c.b))(ls[0].color) if ls else None}
    return out

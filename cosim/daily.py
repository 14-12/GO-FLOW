"""A day in the city: how much traffic each hour brings, and the game clock that runs it.

* PROFILE: the demand of each hour of a weekday as a share of the evening peak hour (17:00-18:00):
  a quiet night, a morning build-up (07:00-09:00), a midday plateau, and the evening rush. The
  shape is assumed, not measured in Ingolstadt (no Ingolstadt hourly counts are in this project):
  a commuter weekday as traffic counts typically show it. Its peak hour carries 9.2 % of the
  day's traffic, inside the 8-12 % that design-hour factors usually take.
* The peak hour is the Town12 benchmark hour itself (scenario.json "vehicles", 4408 veh/h), so
  at 17:00 the city carries exactly the demand every Town12 controller was compared on.
* day_routes(): a 24-hour route file, each hour's vehicles drawn from the benchmark's own routes
  (all valid on the network; duarouter made them), in the benchmark's vehicle mix.
* GameClock: 2 real minutes = 1 game hour (30x; 60x, a GTA-like minute an hour, until 7 Nov: the evening rush
  went by in 2.5 minutes, the forecasters' 6-minute warm-up took 6 town hours). It is saved to a small JSON file while
  it runs and when the viewer closes, and starts from there the next time.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Hour h covers [h:00, h+1:00). Share of the 17:00-18:00 peak.
PROFILE = (0.12, 0.07, 0.05, 0.05, 0.07, 0.15, 0.32, 0.55, 0.62, 0.52, 0.50, 0.54,
           0.58, 0.58, 0.64, 0.76, 0.90, 1.00, 0.88, 0.66, 0.48, 0.38, 0.28, 0.18)
PEAK_HOUR = 17

# Names shown on the clock, by hour. (start hour, label)
PHASES = ((0, "QUIET NIGHT"), (5, "EARLY MORNING"), (7, "MORNING RUSH"), (9, "DAYTIME"),
          (15, "EVENING BUILD-UP"), (16, "EVENING RUSH"), (19, "EVENING"), (22, "NIGHT"))

WEEKDAYS = ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY")
DAY_S = 86400.0
GAME_SPEED = 30.0  # game seconds per real second: 2 real minutes = 1 game hour (a day in 48 min)


def peak_share() -> float:
    """Share of the whole day's traffic carried by the peak hour."""
    return max(PROFILE) / sum(PROFILE)


def demand_factor(hour: float) -> float:
    """Demand at a time of day (hours, any float), smooth between hours: each hour's value
    sits at its middle (h + 0.5) and the curve is cosine-blended between neighbours."""
    x = (hour - 0.5) % 24.0
    i = int(x)
    f = x - i
    w = (1 - math.cos(math.pi * f)) / 2
    return PROFILE[i] * (1 - w) + PROFILE[(i + 1) % 24] * w


def phase(hour: float) -> str:
    h = hour % 24.0
    name = PHASES[0][1]
    for start, label in PHASES:
        if h >= start:
            name = label
    return name


# ---------------------------------------------------------------- routes

_VTYPE = re.compile(r"<vType\b[^>]*/>")
_VEH = re.compile(r'<vehicle\b([^>]*)>\s*<route edges="([^"]+)"\s*/>\s*</vehicle>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class Pool:
    """The benchmark's routes by vehicle type: what a day's vehicles are drawn from."""
    vtypes: dict[str, str] = field(default_factory=dict)      # type id -> its <vType .../> line
    routes: dict[str, list[str]] = field(default_factory=dict)  # type id -> edge lists
    per_hour: dict[str, int] = field(default_factory=dict)      # type id -> vehicles in the benchmark hour
    depart_attrs: dict[str, str] = field(default_factory=dict)  # type id -> departLane/departSpeed attrs

    @property
    def total_per_hour(self) -> int:
        return sum(self.per_hour.values())


def load_pool(folder: Path, name: str | None = None) -> Pool:
    """Read scenarios/<name>/<name>.<class>.rou.xml (the route files the .sumocfg lists)."""
    folder = Path(folder)
    name = name or folder.name
    cfg = (folder / f"{name}.sumocfg").read_text(encoding="utf-8")
    files = re.search(r'<route-files value="([^"]+)"', cfg).group(1).split(",")
    pool = Pool()
    for f in files:
        text = (folder / f.strip()).read_text(encoding="utf-8")
        for vt in _VTYPE.findall(text):
            pool.vtypes[dict(_ATTR.findall(vt))["id"]] = vt.strip()
        for attrs, edges in _VEH.findall(text):
            a = dict(_ATTR.findall(attrs))
            t = a["type"]
            pool.routes.setdefault(t, []).append(edges)
            pool.per_hour[t] = pool.per_hour.get(t, 0) + 1
            pool.depart_attrs.setdefault(t, " ".join(f'{k}="{v}"' for k, v in a.items()
                                                     if k in ("departLane", "departSpeed", "departPos")))
    if not pool.routes:
        raise ValueError(f"no vehicles with routes in {files}")
    return pool


def hourly_counts(pool: Pool, scale: float = 1.0) -> list[dict[str, int]]:
    """Vehicles of each type to insert in each hour 0..23 (the peak hour = the benchmark hour)."""
    return [{t: round(n * PROFILE[h] * scale) for t, n in pool.per_hour.items()} for h in range(24)]


def day_routes(pool: Pool, out: Path, seed: int = 7, scale: float = 1.0, hours: int = 24,
               start_hour: int = 0) -> dict:
    """Write a route file with `hours` hours of traffic, SUMO time 0 being `start_hour`:00.
    In each hour the vehicles of a type are spread at random over the hour; each takes a route
    drawn at random from the benchmark's routes of its type. Returns a summary."""
    rng = random.Random(seed)
    counts = hourly_counts(pool, scale)
    vehicles = []
    for h in range(hours):
        for t, n in counts[(start_hour + h) % 24].items():
            for _ in range(n):
                vehicles.append((h * 3600.0 + rng.uniform(0, 3600.0), t, rng.choice(pool.routes[t])))
    vehicles.sort(key=lambda v: v[0])
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f"<!-- {hours} h of traffic from {start_hour:02d}:00, cosim/daily.py PROFILE x the benchmark hour, seed {seed} -->",
             "<routes>"]
    lines += [f"    {vt}" for vt in pool.vtypes.values()]
    for i, (dep, t, edges) in enumerate(vehicles):
        lines.append(f'    <vehicle id="d{i}" type="{t}" depart="{dep:.2f}" {pool.depart_attrs.get(t, "")}>'
                     f'<route edges="{edges}"/></vehicle>')
    lines.append("</routes>")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"vehicles": len(vehicles), "per_clock_hour": [sum(c.values()) for c in counts], "seed": seed,
            "scale": scale, "start_hour": start_hour, "hours": hours}


# ---------------------------------------------------------------- the clock

class GameClock:
    """Game time in seconds since day 1, 00:00. Runs GAME_SPEED times faster than real time
    while not paused. `save()` writes it atomically; `load()` restores it (or the default)."""

    def __init__(self, game_s: float = 7 * 3600.0, speed: float = GAME_SPEED, path: Path | None = None,
                 now=time.monotonic):
        self.speed = speed
        self.path = Path(path) if path else None
        self._now = now
        self._base = float(game_s)
        self._t0 = now()
        self.paused = False

    @classmethod
    def load(cls, path: Path, default_s: float = 7 * 3600.0, **kw) -> "GameClock":
        """The clock as saved at `path`; a missing or unreadable file gives `default_s`."""
        game_s = default_s
        try:
            game_s = float(json.loads(Path(path).read_text(encoding="utf-8"))["game_seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return cls(game_s, path=path, **kw)

    @property
    def game_s(self) -> float:
        if self.paused:
            return self._base
        return self._base + (self._now() - self._t0) * self.speed

    def set(self, game_s: float) -> None:
        self._base = max(0.0, float(game_s))
        self._t0 = self._now()

    def shift(self, game_seconds: float) -> None:
        self.set(self.game_s + game_seconds)

    def pause(self, on: bool | None = None) -> None:
        on = (not self.paused) if on is None else on
        if on and not self.paused:
            self._base, self.paused = self.game_s, True
        elif not on and self.paused:
            self.paused = False
            self._t0 = self._now()

    @property
    def day(self) -> int:
        """Day number, 1-based."""
        return int(self.game_s // DAY_S) + 1

    @property
    def seconds_of_day(self) -> float:
        return self.game_s % DAY_S

    @property
    def hour(self) -> float:
        return self.seconds_of_day / 3600.0

    @property
    def weekday(self) -> str:
        return WEEKDAYS[(self.day - 1) % 7]

    def hhmm(self) -> str:
        m = int(self.seconds_of_day // 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"game_seconds": round(self.game_s, 1), "day": self.day, "time": self.hhmm(),
                                   "weekday": self.weekday, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                                  indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash mid-write never leaves half a file


# ---------------------------------------------------------------- vehicles in the city

def population_curve(study: dict | None, pool: Pool | None = None, mean_trip_min: float = 10.0) -> list[float]:
    """Vehicles on the road at each minute of the day (1440 values): measured by the 24 h study
    (tools/day_study_town12.py, its "running_per_minute"), or, without one, estimated as
    insertions per minute x the mean trip (Little's law; 10 min on Town12, ~2.3 min on Town05)."""
    if study and len(study.get("running_per_minute", [])) >= 1440:
        return [float(v) for v in study["running_per_minute"][:1440]]
    per_hour = pool.total_per_hour if pool else 4408
    return [per_hour * demand_factor(m / 60.0) / 60.0 * mean_trip_min for m in range(1440)]


def target_at(curve: list[float], seconds_of_day: float) -> float:
    """The curve at a time of day, linear between minutes."""
    m = (seconds_of_day / 60.0) % 1440.0
    i = int(m)
    f = m - i
    return curve[i] * (1 - f) + curve[(i + 1) % 1440] * f


def write_pool_routes(pool: Pool, out: Path) -> dict[str, list[str]]:
    """A route file with the pool's vehicle types and every route as <route id=...>, no vehicles:
    what the live viewer starts SUMO with, adding vehicles itself. Returns type -> route ids."""
    ids: dict[str, list[str]] = {}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    lines += [f"    {vt}" for vt in pool.vtypes.values()]
    for t, routes in pool.routes.items():
        for i, edges in enumerate(routes):
            rid = f"pool_{t}_{i}"
            ids.setdefault(t, []).append(rid)
            lines.append(f'    <route id="{rid}" edges="{edges}"/>')
    lines.append("</routes>")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids

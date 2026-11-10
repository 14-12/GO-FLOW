"""The day's demand profile, the game clock and its saving, the day's route file, and the
viewer's population control (cosim/daily.py, cosim/live.Population, cosim/hud.py)."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from cosim import daily
from cosim.live import Population


def test_profile_shape():
    p = daily.PROFILE
    assert len(p) == 24 and max(p) == 1.0 and p.index(1.0) == daily.PEAK_HOUR
    assert 0.08 <= daily.peak_share() <= 0.12  # a usual design-hour share
    assert max(p[0:5]) < 0.15                  # quiet night
    assert p[8] > p[5] and p[17] > p[8]        # morning build-up, then an evening rush above it


def test_demand_factor_is_smooth_and_hits_the_hour_values():
    for h in range(24):
        assert daily.demand_factor(h + 0.5) == pytest.approx(daily.PROFILE[h])
    for h in range(25):  # no jump at the hour boundaries
        assert abs(daily.demand_factor(h - 1e-6) - daily.demand_factor(h + 1e-6)) < 1e-4
    assert daily.demand_factor(24.5) == pytest.approx(daily.demand_factor(0.5))


def test_phase_names():
    assert daily.phase(3) == "QUIET NIGHT"
    assert daily.phase(7.5) == "MORNING RUSH"
    assert daily.phase(17.2) == "EVENING RUSH"
    assert daily.phase(23.9) == "NIGHT"


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_clock_runs_thirty_times_faster_and_pauses():
    """2 real minutes a town hour (daily.GAME_SPEED 30 since 7 Nov; 60 before)."""
    now = FakeTime()
    c = daily.GameClock(7 * 3600, now=now)
    assert daily.GAME_SPEED == 30.0
    assert c.hhmm() == "07:00" and c.day == 1 and c.weekday == "MONDAY"
    now.t += 120  # two real minutes
    assert c.hhmm() == "08:00"
    c.pause(True)
    now.t += 600
    assert c.hhmm() == "08:00"
    c.pause(False)
    now.t += 34 * 60  # 34 real minutes, 17 town hours: past midnight
    assert c.hhmm() == "01:00" and c.day == 2 and c.weekday == "TUESDAY"
    c.shift(-3600)
    assert c.hhmm() == "00:00" and c.day == 2


def test_clock_saved_and_restored(tmp_path: Path):
    now = FakeTime()
    f = tmp_path / "clock.json"
    c = daily.GameClock.load(f, default_s=7 * 3600, now=now)  # no file yet: the default
    assert c.hhmm() == "07:00"
    now.t += 21 * 60  # 21 real minutes, 10.5 town hours: 17:30
    c.save()
    saved = json.loads(f.read_text())
    assert saved["time"] == "17:30" and saved["day"] == 1
    again = daily.GameClock.load(f, now=FakeTime())
    assert again.hhmm() == "17:30" and again.game_s == pytest.approx(c.game_s, abs=0.1)
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_clock_survives_a_damaged_file(tmp_path: Path):
    f = tmp_path / "clock.json"
    f.write_text("{not json")
    assert daily.GameClock.load(f, default_s=3600).hhmm() == "01:00"


def _tiny_scenario(folder: Path) -> Path:
    folder.mkdir()
    (folder / "tiny.sumocfg").write_text('<configuration><input><route-files value="tiny.a.rou.xml,tiny.b.rou.xml"/>'
                                         "</input></configuration>")
    veh = '<vehicle id="{i}" type="{t}" depart="{d}" departLane="best" departSpeed="max">\n  <route edges="{e}"/>\n</vehicle>'
    a = "\n".join(veh.format(i=f"a{i}", t="car", d=i * 36.0, e=f"e{i} f{i}") for i in range(100))
    b = "\n".join(veh.format(i=f"b{i}", t="bus", d=i * 360.0, e="x y z") for i in range(10))
    (folder / "tiny.a.rou.xml").write_text(f'<routes><vType id="car" vClass="passenger"/>\n{a}\n</routes>')
    (folder / "tiny.b.rou.xml").write_text(f'<routes><vType id="bus" vClass="bus"/>\n{b}\n</routes>')
    return folder


def test_pool_and_day_routes(tmp_path: Path):
    pool = daily.load_pool(_tiny_scenario(tmp_path / "tiny"))
    assert pool.per_hour == {"car": 100, "bus": 10} and len(pool.routes["car"]) == 100
    out = tmp_path / "day.rou.xml"
    made = daily.day_routes(pool, out, seed=3, hours=25, start_hour=23)
    root = ET.parse(out).getroot()
    vs = root.findall("vehicle")
    assert len(vs) == made["vehicles"]
    deps = [float(v.get("depart")) for v in vs]
    assert deps == sorted(deps)
    # hour k of the file is clock hour (23 + k) % 24
    for k in (0, 1, 18):
        n = sum(1 for d in deps if k * 3600 <= d < (k + 1) * 3600)
        h = (23 + k) % 24
        assert n == round(100 * daily.PROFILE[h]) + round(10 * daily.PROFILE[h])
    assert {v.get("type") for v in vs} == {"car", "bus"}
    assert all(v.find("route").get("edges") for v in vs)
    ids = daily.write_pool_routes(pool, tmp_path / "pool.rou.xml")
    pr = ET.parse(tmp_path / "pool.rou.xml").getroot()
    assert len(pr.findall("route")) == 110 and not pr.findall("vehicle") and len(ids["bus"]) == 10


def test_population_curve_and_target():
    curve = daily.population_curve({"running_per_minute": list(range(1440))})
    assert daily.target_at(curve, 60.0 * 100) == 100
    assert daily.target_at(curve, 60.0 * 100 + 30) == pytest.approx(100.5)
    est = daily.population_curve(None)
    assert len(est) == 1440 and max(est) > min(est) * 5
    full = daily.population_curve(None, base=1.0)
    assert est[600] == pytest.approx(full[600] * daily.DEMAND_BASE)  # (9 Nov) the town's own traffic: 0.45 of the routes'


class FakeConn:
    def __init__(self):
        self.added, self.routes, self.removed, self.pending = [], {}, [], []
        self.vehicle = self
        self.route = self
        self.simulation = self

    def add(self, vid, rid, typeID=None, **kw):  # noqa: N803 - TraCI's names
        if isinstance(rid, list):
            self.routes[vid] = rid  # route.add(id, edges)
        else:
            self.added.append((vid, rid, typeID, kw))

    def remove(self, vid):
        self.removed.append(vid)

    def getPendingVehicles(self):  # noqa: N802
        return self.pending


def test_population_fills_up_and_thins_out_of_sight(tmp_path: Path):
    pool = daily.load_pool(_tiny_scenario(tmp_path / "tiny"))
    ids = {t: [f"pool_{t}_{i}" for i in range(len(r))] for t, r in pool.routes.items()}
    conn = FakeConn()
    pop = Population(conn, pool, ids, [200.0] * 1440, seed=1, flow=False)   # the count mode (before 9 Nov)
    pop.step({}, 3600.0, lambda vid, v: True)  # far below: filled along the routes, 100 at once
    assert len(conn.added) == 100 and len(conn.routes) == 100
    assert all(kw.get("departPos") == "random_free" for *_, kw in conn.added)
    conn.added.clear()
    pop.pending = 0
    pop._k = 1  # no pending query this step
    running = {f"v{i}": None for i in range(190)}
    pop.step(running, 3600.0, lambda vid, v: True)  # a little below: new trips from route starts
    assert len(conn.added) == 10 and all(rid.startswith("pool_") for _, rid, *_ in conn.added)
    running = {f"v{i}": None for i in range(300)}
    pop.step(running, 3600.0, lambda vid, v: vid.endswith("7"))  # above: only out-of-sight ones go
    assert conn.removed and all(v.endswith("7") for v in conn.removed) and len(conn.removed) <= 40


def test_population_flow_starts_trips_at_the_curves_rate_and_takes_nobody_off(tmp_path: Path):
    """(9 Nov) The default: new trips at curve / mean trip a second; above the curve nobody is taken off."""
    pool = daily.load_pool(_tiny_scenario(tmp_path / "tiny"))
    ids = {t: [f"pool_{t}_{i}" for i in range(len(r))] for t, r in pool.routes.items()}
    conn = FakeConn()
    pop = Population(conn, pool, ids, [140.0] * 1440, seed=1, mean_trip_s=140.0)
    pop.step_s = 0.5
    pop.step({}, 3600.0, lambda vid, v: True)   # the start: filled along the routes up to the curve
    assert len(conn.added) == 100 and all(kw.get("departPos") == "random_free" for *_, kw in conn.added)
    conn.added.clear()
    pop.pending, pop._k = 0, 1
    full = {f"v{i}": None for i in range(300)}  # far above the curve: still no one taken off
    sod = 3600.0
    for _ in range(120):                        # 60 s of simulation, 15 town seconds a step at clock 30
        sod += 15.0
        pop.pending, pop._k = 0, 1
        pop.step(full, sod, lambda vid, v: True)
    assert not conn.removed
    assert len(conn.added) == 60               # 140 cars / 140 s = 1 trip a second
    conn.added.clear()
    pop.pending, pop._k = 0, 1
    pop.step(full, sod + 3600.0, lambda vid, v: True)  # a jump of the clock (F7) with the town full: nothing added
    assert not conn.added and not conn.removed


def test_hud_draws_at_every_hour():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from cosim.hud import Hud, HudState, dial_angle

    pygame.init()
    screen = pygame.Surface((1280, 720))
    hud = Hud()
    for h in range(0, 24, 3):
        hud.draw(screen, HudState(game_s=h * 3600 + 86400, vehicles=500, sun_elevation=10 if 6 <= h <= 20 else -10))
    hud.loading(screen, HudState(game_s=0, paused=True), "Loading", 0.5, 1.0)
    # the dial: midnight at the bottom, noon at the top (screen y grows downwards)
    import math
    assert math.sin(dial_angle(0)) == pytest.approx(1.0)
    assert math.sin(dial_angle(12)) == pytest.approx(-1.0)
    assert math.cos(dial_angle(6)) == pytest.approx(-1.0)  # morning on the left


def test_population_flow_counts_simulation_time_with_the_hour_held(tmp_path: Path):
    """(10 Nov) A tool that holds the hour still (tools/check_overlaps.py) still gets new trips."""
    pool = daily.load_pool(_tiny_scenario(tmp_path / "tiny"))
    ids = {t: [f"pool_{t}_{i}" for i in range(len(r))] for t, r in pool.routes.items()}
    conn = FakeConn()
    pop = Population(conn, pool, ids, [140.0] * 1440, seed=1, mean_trip_s=140.0)
    pop.step_s = 0.5
    full = {f"v{i}": None for i in range(140)}
    pop.step(full, 3600.0, lambda vid, v: True)   # the start (the town already full)
    for _ in range(120):
        pop.pending, pop._k = 0, 1
        pop.step(full, 3600.0, lambda vid, v: True)
    assert len(conn.added) == 60

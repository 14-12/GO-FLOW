"""Trips with a purpose (cosim/purposes.py) in SUMO on Town05: a car leaves its home, parks at its destination, comes
back, parks at home and is the one taken out for the next trip from there; the route agent follows such a car to the
car park it is going to."""
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import purposes as P


@dataclass
class V:
    x: float
    y: float
    angle: float
    speed: float
    vclass: str = "passenger"


@dataclass
class Cur:
    time: float
    vehicles: dict
    arrived: list = field(default_factory=list)


def test_open_hours_and_choices():
    places = P.Places()
    assert places.is_open("marche", 9.0) and not places.is_open("marche", 16.0)
    rng = random.Random(1)
    seen = {places.choose(rng, "shops", 16.0) for _ in range(50)}
    assert "marche" not in seen and "atlas" in seen            # the market has closed at 14:00
    jobs = [places.choose(rng, "jobs", 8.0) for _ in range(400)]
    assert jobs.count("jumelles") > jobs.count("cafe_coin")      # by their jobs
    for pid in places.places:                                    # every place parks somewhere
        assert places.park_of[pid], pid


def test_the_hours_weigh_the_purposes():
    w = {n: f[0] for n, f in P.PURPOSES.items()}
    assert w["work"](8.0) > 5 * w["work"](20.0)
    assert w["eating"](12.6) > 2 * w["eating"](10.0)


def _sumo(label):
    import sumolib
    import traci
    import traci.constants as tc

    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a", str(P.ADD), "--step-length",
                 "0.5", "--no-step-log", "--no-warnings", "--time-to-teleport", "-1"], label=label)
    c = traci.getConnection(label)
    c.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_IDS])
    return c


def _cur(c):
    import traci.constants as tc

    vs = {v: V(*c.vehicle.getPosition(v), c.vehicle.getAngle(v), c.vehicle.getSpeed(v)) for v in c.vehicle.getIDList()}
    return Cur(c.simulation.getTime(), vs, list(c.simulation.getSubscriptionResults()[tc.VAR_ARRIVED_VEHICLES_IDS]))


@pytest.mark.slow
def test_a_car_goes_parks_comes_home_and_is_taken_out_again():
    c = _sumo("purposes_test")
    try:
        d = P.PurposeDemand(c, clock_speed=1800.0, seed=3)        # a fast clock: an hour's dwell is 2 s
        d.rng = random.Random(5)
        d.purpose_at = lambda hour, vclass: "shopping"
        assert d.launch("car", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ())
        plan = d.plans["car"]
        home_area, dest_area = d.places.park_of[plan.home][0], plan.areas[0]
        seen = []
        for _ in range(3000):
            c.simulationStep()
            d.step(_cur(c).arrived)
            where = d.parked.get("car")
            if where and (not seen or seen[-1] != where):
                seen.append(where)
            if "car" in d.idle.get(home_area, []):
                break
        assert seen[:1] == [dest_area] and seen[-1] in d.places.park_of[plan.home]
        assert "car" not in d.plans and d.stats["arrived_parked"] == 2
        # the next trip from that home takes this car out again
        d.places.home = lambda rng: plan.home
        assert d.launch("other", "DEFAULT_VEHTYPE", "passenger", 10.0, (), ())
        assert d.stats["reused"] == 1 and "car" in d.plans and "other" not in d.plans
        for _ in range(20):
            c.simulationStep()
            d.step(_cur(c).arrived)
        assert "car" not in d.parked                             # it pulled out
    finally:
        c.close()


@pytest.mark.slow
def test_the_route_agent_follows_a_purpose_car_to_its_car_park(tmp_path):
    from cosim import route_guidance as rg

    c = _sumo("purposes_guide")
    try:
        d = P.PurposeDemand(c, clock_speed=30.0, seed=4)
        d.rng = random.Random(9)
        d.purpose_at = lambda hour, vclass: "work"
        assert d.launch("car", "DEFAULT_VEHTYPE", "passenger", 8.0, (), ())
        plan = d.plans["car"]
        guide = rg.RouteGuide(router=d.places.router, adapter=rg.load_router()[1], log_dir=tmp_path)
        guide.demand = d
        for _ in range(4):
            c.simulationStep()
            d.step(_cur(c).arrived)
        guide.follow("car")
        snap = None
        for _ in range(2000):
            cur = _cur(c)
            guide.step(cur, c)
            snap = guide.snapshot()
            if snap and snap["status"] != "following":
                break
            c.simulationStep()
            d.step(_cur(c).arrived)
        assert snap["status"] == "arrived"
        assert snap["to_place"]["id"] == plan.dest and snap["purpose"] == "work"
        assert snap["from_place"]["id"] == plan.home
        assert d.parked.get("car") == d.places.park_of[plan.dest][0] or d.parked.get("car") in d.places.areas
    finally:
        c.close()


@pytest.mark.slow
def test_prefill_puts_cars_in_their_spaces(tmp_path):
    """The town at noon: cars put straight into the spaces the table gives (cosim/purposes.prefill_one); those at
    home stay there until a trip takes them out, those at work go home after what is left of their day."""
    c = _sumo("purposes_prefill")
    try:
        d = P.PurposeDemand(c, clock_speed=60.0, seed=5)          # a workday: 8 min
        d.rng = random.Random(2)
        home_area = d.places.park_of["oliviers"][0]
        work_area = d.places.park_of["atlas"][0]
        made = [v for v in ("h0", "h1", "w0", "w1")
                if d.prefill_one(v, "DEFAULT_VEHTYPE", home_area if v[0] == "h" else work_area, 3.0 if v[0] == "h"
                                 else 12.0)]
        assert made == ["h0", "h1", "w0", "w1"]
        for _ in range(4):
            c.simulationStep()
            d.step(_cur(c).arrived)
        assert d.parked == {"h0": home_area, "h1": home_area, "w0": work_area, "w1": work_area}
        assert sorted(d.idle[home_area]) == ["h0", "h1"] and d.here()["oliviers"] == 2
        for _ in range(3000):                    # the workers' day ends: they drive home and park there
            c.simulationStep()
            d.step(_cur(c).arrived)
            if all(v not in d.plans for v in ("w0", "w1")):
                break
        assert all(v not in d.plans for v in ("w0", "w1"))
        assert all(d.parked.get(v) in d.places.areas for v in ("w0", "w1"))
        assert d.stats["arrived_parked"] == 2      # (the noon parking was not an arrival)
    finally:
        c.close()


def test_the_places_are_drawn_on_the_map_and_over_the_town():
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import json
    import math

    import pygame

    pygame.init()
    from cosim.mapping import Frame
    from cosim.place_hud import PlaceHud, info_line
    from cosim.route_minimap import MiniMap

    cards = json.loads(P.PLACES.read_text(encoding="utf-8"))["places"]
    places = P.Places()
    occ = {a: n // 2 for a, n in places.cap.items()}
    live = {"occ": occ, "cap": places.cap, "park_of": places.park_of, "here": {"atlas": 7}, "hour": 12.5,
            "parked": frozenset()}
    by_id = {p["id"]: p for p in cards}
    assert info_line(by_id["atlas"], 12.5, live)[0].endswith("7 here")
    assert info_line(by_id["marche"], 16.0, live)[0].startswith("CLOSED")
    assert info_line(by_id["lot_marche"], 12.5, live)[0] == f"{43 - 21} free of 43"
    m = MiniMap(places.router, places=cards)
    free = m.render(None, 1.0, free={"cars": [], "places": live})
    snap = {"vid": "live1", "status": "following", "guided": True, "xy": by_id["lac"]["label"], "angle": 0.0,
            "trail": [], "signals": [], "alerts": [], "left": [], "own_left": [], "start": None, "dest": None,
            "purpose": "work", "from_place": {"id": "lac", "name": "Résidence du Lac", "kind": "apartments"},
            "to_place": {"id": "atlas", "name": by_id["atlas"]["name"], "kind": "mall"}, "stop_area": "pa_atlas",
            "elapsed_s": 3.0, "events": [], "options": [], "inputs": {}}
    follow = m.render(snap, 2.0, free={"cars": [], "places": live})
    assert free.get_width() == follow.get_width() and follow.get_height() > free.get_height()

    class Vec:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Tf:          # a CARLA view 40 m up over the market, looking west and down
        location = Vec(*Frame(209.41, 199.645).to_carla_xy(263.0, 180.0), 40.0)

        @staticmethod
        def get_forward_vector():
            return Vec(-math.cos(0.3), 0.0, -math.sin(0.3))

        @staticmethod
        def get_right_vector():
            return Vec(0.0, -1.0, 0.0)

        @staticmethod
        def get_up_vector():
            return Vec(-math.sin(0.3), 0.0, math.cos(0.3))
    hud = PlaceHud(cards, Frame(209.41, 199.645))
    screen = pygame.Surface((1600, 900))
    assert hud.draw(screen, (1600, 900), Tf(), 12.5, live)       # some labels are in view

"""Town05's places (scenarios/town05/places.json, tools/town05_places_build.py): every check passes, and SUMO parks a
car on the space it was given."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "agents" / "route_recommendation"))
CARD = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))


def test_the_places_pass_every_check():
    import sumolib
    import town05_places_build as b

    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    assert b.check(net, CARD["places"], CARD["parking_areas"]) == []


def test_homes_jobs_and_spaces_are_there():
    places = CARD["places"]
    kinds = {p["kind"] for p in places}
    assert {"houses", "apartments", "office", "market", "mall", "restaurant", "parking", "hospital"} <= kinds
    assert sum(p.get("homes", 0) for p in places) >= 300
    assert sum(p.get("jobs", 0) for p in places) >= 300
    for p in places:            # every place can be reached: its own door, or the car park it uses
        assert "door" in p, p["id"]


@pytest.mark.slow
def test_a_car_parks_on_its_space_and_leaves_from_it():
    import sumolib
    import traci
    from router import Router

    areas = {a["id"]: a for a in CARD["parking_areas"]}
    scen = ROOT / "scenarios" / "town05"
    traci.start([sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-a",
                 str(scen / "town05.places.add.xml"), "--step-length", "0.5", "--no-step-log", "--no-warnings"],
                label="places_test")
    c = traci.getConnection("places_test")
    try:
        r = Router()
        a = areas["pa_lot_marche"]
        c.route.add("in", r.best("46", a["edge"]))
        c.vehicle.add("car", "in")
        c.vehicle.setParkingAreaStop("car", a["id"], duration=30)
        at = None
        for _ in range(800):
            c.simulationStep()
            if "car" in c.vehicle.getIDList() and c.vehicle.isStoppedParking("car"):
                at = c.vehicle.getPosition("car")
                break
        assert at is not None
        assert min(math.hypot(at[0] - s[0], at[1] - s[1]) for s in a["spaces"]) < 0.1
        c.route.add("out", r.best(a["edge"], "45"))
        c.vehicle.add("leaver", "out", depart="now", departPos="stop")
        c.vehicle.setParkingAreaStop("leaver", a["id"], duration=2)
        parked = moved = False
        for _ in range(200):
            c.simulationStep()
            if "leaver" in c.vehicle.getIDList():
                if c.vehicle.isStoppedParking("leaver"):
                    parked = True
                elif parked:
                    moved = True
                    break
        assert parked and moved
    finally:
        c.close()

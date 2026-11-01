"""The kinds of incident added on 31 Oct (cosim/incidents.py): speeding drivers, breakdowns, pile-ups on the ring road,
look-alikes; and how the overlay names them (cosim/incident_hud.describe). The whole run, SUMO alone, is
tools/check_accidents.py --ring --buses."""

from __future__ import annotations

import random
from types import SimpleNamespace as NS

from cosim import incidents as inc_mod
from cosim.incident_hud import describe
from cosim.incidents import Incident, IncidentManager, Lookalike, Wreck


def _im(seed: int = 1, speeder_share: float = 0.08, share: float = 0.10) -> IncidentManager:
    """An IncidentManager without SUMO: only what type_for, snapshot and the ring test read."""
    im = IncidentManager.__new__(IncidentManager)
    im.typer = NS(type_for=lambda vid, base: base, base={"pas": "passenger", "bus": "bus", "del": "delivery"})
    im.rng = random.Random(seed)
    im.share, im.speeder_share = share, speeder_share
    im.max_active, im.cooldown_s = 2, 60.0
    im.incidents, im.done = [], []
    im.now, im.last_t = 1000.0, -1e9
    im.careless, im.speeders, im._speed_pending = {}, {}, {}
    im.stats = {"careless_made": 0, "speeders_made": 0}
    im._ring_prefix, im.pileups, im.ring_brake_mean_s = "hw", True, 400.0
    im.ring = {"35", "37"}
    im._via = {":2014_3_0": ("35_2", "12_1"), ":900_1_0": ("4_1", "5_1")}
    im.lookalikes, im.lookalikes_done = [], []
    return im


def test_a_share_of_the_cars_speed_and_they_are_more_often_careless():
    im = _im(seed=3)
    for k in range(20000):
        im.type_for(f"live{k}", "pas")
    for k in range(500):
        im.type_for(f"bus{k}", "bus")
    share = im.stats["speeders_made"] / 20000
    assert 0.07 < share < 0.09
    assert not any(v.startswith("bus") for v in im.speeders)  # buses keep to the limit
    assert all(inc_mod.SPEEDER_FACTOR[0] <= f <= inc_mod.SPEEDER_FACTOR[1] for f in im.speeders.values())
    careless_fast = sum(1 for v in im.speeders if v in im.careless) / len(im.speeders)
    careless_rest = sum(1 for v in im.careless if v not in im.speeders) / (20000 - len(im.speeders))
    assert careless_fast > 2.0 * careless_rest


def test_no_speeders_when_turned_off():
    im = _im(speeder_share=0.0)
    for k in range(2000):
        im.type_for(f"live{k}", "pas")
    assert not im.speeders


def test_the_ring_roads_own_cars_are_more_often_careless():
    im = _im(seed=5, speeder_share=0.0)
    for k in range(20000):
        im.type_for(f"live{k}", "pas")
        im.type_for(f"hw{k}", "pas")
    town = sum(1 for v in im.careless if v.startswith("live")) / 20000
    ring = sum(1 for v in im.careless if v.startswith("hw")) / 20000
    assert ring > 2.0 * town


def test_on_the_ring_road_and_in_the_junctions_it_leads_into():
    im = _im()
    assert im._on_ring("35_2") and im._on_ring("37_0")
    assert im._on_ring(":2014_3_0")      # from 35 into its junction
    assert not im._on_ring(":900_1_0")   # a town junction
    assert not im._on_ring("4_1") and not im._on_ring("")


def test_snapshot_names_the_cause_and_the_speeders():
    im = _im()
    w1 = Wreck("a", "vehicle.audi.tt", "collider", "35_2", 50.0, 0, 0, 0, 4.5, 30.0, speeding=True)
    w2 = Wreck("b", "vehicle.audi.tt", "victim", "35_2", 56.0, 0, 6, 0, 4.5, 10.0)
    i = Incident(1, 900.0, "rear-end", 20.0, "severe", "35", "35_2", 0.0, 3.0, wrecks=[w1, w2])
    i.cause = "phone"
    im.incidents = [i]
    snap = im.snapshot()[0]
    assert snap["cause"] == "phone" and snap["speeding"] == ["a"]


def test_lookalikes_are_reported_once_they_stand():
    im = _im()
    im.lookalikes = [Lookalike(1, "double-parked", "v1", "3_3", 10.0, 90.0),
                     Lookalike(2, "bus-long-stop", "pt_L1_0", "3_3", 20.0, 120.0, t0=20.0, stop="L1_S1")]
    im.lookalikes_done = [Lookalike(3, "double-parked", "v2", "4_3", 1.0, 60.0, t0=5.0, t1=66.0)]
    kinds = {lk["id"]: lk["kind"] for lk in im.lookalike_snapshot()}
    assert kinds == {3: "double-parked", 2: "bus-long-stop"}  # 1 is still on its way to its stop


def test_the_overlay_names_each_kind():
    base = {"label": "severe", "kmh": 60, "cars": 2}
    assert describe(dict(base, kind="rear-end")) == "severe rear-end crash, 2 cars, 60 km/h"
    assert describe(dict(base, kind="pile-up", cars=4)) == "severe pile-up, 4 cars, 60 km/h"
    assert describe(dict(base, kind="breakdown", cars=1, kmh=0)) == "breakdown, 1 car"


def test_rates_are_sensible():
    # a breakdown every 40 min of simulated time at most, hard braking on the ring every few minutes, a long bus stop
    # now and then: look-alikes are for the cameras, not the main event
    assert inc_mod.BREAKDOWN_MEAN_S >= 1200.0
    assert 0.0 < inc_mod.BUS_LONG_P <= 0.1
    assert inc_mod.PILEUP_MAX_WRECKS > 3
    assert all(0.0 < p < 1.0 for p in inc_mod.PILEUP_LATE_P)

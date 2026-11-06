"""Option A of the GO-FLOW crash delivery (6 Nov): crashes made likelier where a recorded camera sees clearly
(cosim/clip_recorder.Focus, cosim/incidents.IncidentManager._focus_step). Measured with tools/check_accidents.py
--focus-cams ... --crash-focus P."""

from __future__ import annotations

import random
from types import SimpleNamespace as NS

from cosim.clip_recorder import Focus
from cosim.incidents import IncidentManager
from cosim.perception import CamSpec

CAM = CamSpec(tls="1", node="1", road_to="2", x=0.0, y=0.0, z=9.0, yaw=0.0, pitch=-22.0)


def _focus():
    # SUMO and CARLA the same here but for y (CARLA y points south): SUMO (x, y) -> CARLA (x, -y)
    return Focus([("cam", CAM)], 960, 540, 75.0, lambda x, y: (x, -y))


def test_a_car_close_ahead_is_in_clear_view():
    f = _focus()
    assert f.seen_by(18.0, 0.0, 90.0) == "cam"       # 18 m ahead, along the view (SUMO 90 deg: east)
    assert f(18.0, 0.0, 0.0)                         # across it, still whole in the picture


def test_too_far_too_close_or_aside_is_not():
    f = _focus()
    assert not f(45.0, 0.0, 90.0)                    # beyond 30 m
    assert not f(3.0, 0.0, 90.0)                     # under the camera: off the bottom of the picture
    assert not f(15.0, 20.0, 90.0)                   # off to the side
    assert not f(-15.0, 0.0, 90.0)                   # behind it


def _im(p: float):
    im = IncidentManager.__new__(IncidentManager)
    im.focus = lambda x, y, a=None: x > 100.0
    im.conn = None  # (no car ahead: where it will be in 1.5 and 3 s)
    im.focus_p = p
    im.focus_rng = random.Random(1)
    im.rng = random.Random(5)
    im.focus_seen, im.focus_made, im.careless, im.role = set(), set(), {}, {}
    im.stats = {}
    im.incidents, im.max_active, im.cooldown_s, im.now, im.last_t = [], 3, 60.0, 1000.0, -1e9
    return im


def _car(x, speed=12.0, vclass="passenger"):
    return NS(x=x, y=0.0, angle=90.0, speed=speed, length=4.5, vclass=vclass, type_id="t")


def test_cars_heading_into_view_are_made_careless_once():
    im = _im(1.0)
    cars = {"a": _car(70.0), "b": _car(0.0), "c": _car(90.0, vclass="bus"), "d": _car(95.0, speed=1.0)}
    near = im._focus_step(cars)
    assert near == {"a"}                     # 70 m + 12 m/s x 3 s ahead reaches x > 100; b does not; c a bus; d slow
    assert set(im.careless) == {"a"} and im.focus_made == {"a"}
    im._focus_step(cars)                     # decided once
    assert im.stats["focus_careless"] == 1


def test_without_the_chance_nobody_is_made_careless_and_the_town_draws_stay():
    im = _im(0.0)
    state = im.rng.getstate()
    assert im._focus_step({"a": _car(70.0)}) == {"a"}
    assert im.careless == {} and im.rng.getstate() == state


def test_nothing_is_decided_while_no_new_incident_may_start():
    im = _im(1.0)
    im.last_t = im.now - 10.0                # in the cooldown
    im._focus_step({"a": _car(70.0)})
    assert im.careless == {} and "a" not in im.focus_seen

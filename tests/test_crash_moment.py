"""The moment of a crash as drawn, and the police dispatch delay (3 Nov, cosim/incidents.py, cosim/interp.py,
cosim/pose_proc.py, cosim/incident_scene.py). Measured end to end, SUMO alone, with tools/measure_impact.py."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from cosim import incidents as inc_mod
from cosim.incident_scene import IncidentScene
from cosim.incidents import (Incident, IncidentManager, Wreck, body_box, boxes_touch, contact_fraction,
                             police_delay, police_delay_args)
from cosim.interp import Timed
from cosim.pose_proc import damage_share, hit_fraction

CAR = (4.5, 1.85)


# ------------------------------------------------------------------ where and when the cars touched
def test_boxes_touch():
    a = body_box(0.0, 4.5, 0.0, *CAR)              # front at y 4.5, heading north: body from y 0 to 4.5
    assert boxes_touch(a, body_box(0.0, 8.0, 0.0, *CAR))       # body 3.5-8.0: overlaps
    assert not boxes_touch(a, body_box(0.0, 9.5, 0.0, *CAR))   # body 5.0-9.5: a gap
    assert not boxes_touch(a, body_box(2.5, 4.5, 0.0, *CAR))   # side by side, 2.5 m apart
    assert boxes_touch(a, body_box(0.5, 3.0, 90.0, *CAR))      # nose into its side (heading east)
    assert not boxes_touch(a, body_box(-1.0, 3.0, 90.0, *CAR))  # stopped just short of it


def test_a_rear_end_touches_where_the_follower_reaches_the_leaders_back():
    lead = (0.0, 20.0, 0.0)                         # standing, its back at y 15.5
    f = contact_fraction((0.0, 5.0, 0.0), (0.0, 17.0, 0.0), lead, lead, CAR, CAR)  # 12 m in the step, 1.5 m inside
    assert f == pytest.approx((15.5 - 5.0) / 12.0, abs=0.025)


def test_a_side_crash_touches_when_the_bodies_first_meet():
    # A heads east through y 0, B north through x 0; at the end of the step they are well inside each other
    a0, a1 = (-12.0, 0.0, 90.0), (2.0, 0.0, 90.0)
    b0, b1 = (0.0, -9.0, 0.0), (0.0, 0.5, 0.0)
    f = contact_fraction(a0, a1, b0, b1, CAR, CAR)
    assert 0.3 < f < 0.95
    at = lambda p, q, k: (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, p[2])  # noqa: E731
    assert boxes_touch(body_box(*at(a0, a1, f), *CAR), body_box(*at(b0, b1, f), *CAR))
    g = f - 0.04
    assert not boxes_touch(body_box(*at(a0, a1, g), *CAR), body_box(*at(b0, b1, g), *CAR))


def test_no_overlap_on_the_way_leaves_the_step_whole():
    lead = (0.0, 40.0, 0.0)
    assert contact_fraction((0.0, 5.0, 0.0), (0.0, 17.0, 0.0), lead, lead, CAR, CAR) == 1.0


# ------------------------------------------------------------------ drawn: on at its speed, then a dead stop
def test_a_hit_step_runs_at_an_even_speed_and_stops_dead():
    tm = Timed((-10.0, 0.0), (0.0, 0.0), (6.0, 0.0), (12.0, 0.0), stop_at=0.6)
    xs = [tm.at(i / 60.0)[0] for i in range(61)]
    steps = [b - a for a, b in zip(xs, xs[1:])]
    moving = [d for d in steps[:35]]
    assert max(moving) - min(moving) < 0.02           # even: no easing into the stop
    assert xs[36] == pytest.approx(6.0, abs=1e-6)       # there at 0.6 of the step...
    assert all(x == pytest.approx(6.0) for x in xs[37:])  # ... and stands


def test_a_normal_step_still_eases():
    tm = Timed((-10.0, 0.0), (0.0, 0.0), (6.0, 0.0), (6.0, 0.0))
    xs = [tm.at(i / 60.0)[0] for i in range(61)]
    assert xs[-1] == pytest.approx(6.0) and xs[30] < 6.0


def test_hit_fraction_only_in_the_hit_step():
    dmg = {"t_hit": 100.3}
    assert hit_fraction(dmg, 100.5, 0.5) == pytest.approx(0.6)
    assert hit_fraction(dmg, 101.0, 0.5) is None       # the step after: standing
    assert hit_fraction(dmg, 100.0, 0.5) is None
    assert hit_fraction({"ramp_s": 0.4}, 100.5, 0.5) is None  # (a damage from before 3 Nov)
    assert hit_fraction(None, 100.5, 0.5) is None


def test_the_push_and_turn_start_at_the_hit_not_before():
    dmg = {"ramp_s": 0.5, "t_hit": 100.3, "t0": 50.0}
    assert damage_share(dmg, 51.0, 100.2) == 0.0       # the cars not touching yet
    assert damage_share(dmg, 51.0, 100.3) == 0.0
    assert 0.0 < damage_share(dmg, 51.0, 100.55) < 1.0
    assert damage_share(dmg, 51.0, 100.8) == 1.0
    # a damage without t_hit: wall clock from when the scene saw it, as before
    assert damage_share({"ramp_s": 0.5, "t0": 50.0}, 50.6, None) == 1.0
    assert damage_share({"ramp_s": 0.5, "t0": 50.0}, 50.0, None) == 0.0


def test_the_debris_waits_for_the_hit():
    inc = {"now": 100.5}
    # the step 100.0-100.5 is shown from SHOW_LEAD_S after this update; the hit 0.3 s into it
    d = IncidentScene.hit_delay(inc, [{"t_hit": 100.3}, {"t_hit": 100.3}])
    assert d == pytest.approx(0.2 + 0.3)
    assert IncidentScene.hit_delay(inc, [{"t_hit": None}]) == 0.0


def _im() -> IncidentManager:
    im = IncidentManager.__new__(IncidentManager)
    im.incidents, im.done, im.role = [], [], {}
    im.breaking, im.lookalikes = {}, []
    im.now = 100.5
    return im


def test_hit_poses_name_the_cars_that_crashed_in_this_step():
    im = _im()
    w1 = Wreck("a", "vehicle.audi.tt", "collider", "35_2", 50.0, 1.0, 2.0, 90.0, 4.5, 14.0, t_hit=100.3)
    w2 = Wreck("b", "vehicle.audi.tt", "victim", "35_2", 56.0, 1.0, 8.0, 90.0, 4.5, 0.0, t_hit=100.3)
    w3 = Wreck("c", "vehicle.audi.tt", "collider", "35_2", 40.0, 1.0, 0.0, 90.0, 4.5, 9.0, t_hit=95.0)
    im.incidents = [Incident(1, 100.5, "rear-end", 14.0, "severe", "35", "35_2", 0.0, 3.0, wrecks=[w1, w2, w3])]
    assert im.hit_poses(100.5) == {"a": (1.0, 2.0, 90.0), "b": (1.0, 8.0, 90.0)}
    assert im.hit_poses(101.0) == {}


def test_hazards_come_on_after_the_hit():
    im = _im()
    w = Wreck("a", "vehicle.audi.tt", "collider", "35_2", 50.0, 1.0, 2.0, 90.0, 4.5, 14.0, t_hit=100.3)
    im.incidents = [Incident(1, 100.5, "rear-end", 14.0, "severe", "35", "35_2", 0.0, 3.0, wrecks=[w])]
    im.role = {"a": (1, "wreck")}
    im.now = 100.5
    assert im.light_bits("a") == 0
    im.now = 100.3 + inc_mod.HAZARDS_AFTER_S
    assert im.light_bits("a") == 0x10 | 0x20


# ------------------------------------------------------------------ the police: minutes, not seconds
def test_police_delay_without_spread_is_the_median():
    assert police_delay(random.Random(1), 6.0, 0.0) == 6.0


def test_police_delay_is_minutes_around_its_median_and_within_bounds():
    r = random.Random(3)
    xs = [police_delay(r, 240.0, 0.5) for _ in range(4000)]
    assert statistics.median(xs) == pytest.approx(240.0, rel=0.08)
    lo, hi = inc_mod.POLICE_BOUNDS
    assert min(xs) >= lo * 240.0 and max(xs) <= hi * 240.0
    q = statistics.quantiles(xs, n=20)
    assert 80.0 < q[0] < 130.0 and 450.0 < q[-1] < 650.0   # 5-95 %: about 105-545 s
    assert inc_mod.POLICE_MEDIAN_S >= 120.0                  # the default is minutes (it was 6 s)


def test_police_delay_argument():
    assert police_delay_args(None) == {}
    assert police_delay_args("6:0") == {"police_delay_s": 6.0, "police_spread": 0.0}
    assert police_delay_args("300") == {"police_delay_s": 300.0}
    with pytest.raises(ValueError):
        police_delay_args("-5")


def test_the_snapshot_carries_the_contact_and_the_police_record():
    im = _im()
    w = Wreck("a", "vehicle.audi.tt", "collider", "35_2", 50.0, 1.0, 2.0, 90.0, 4.5, 14.0, t_hit=100.3)
    i = Incident(1, 100.5, "rear-end", 14.0, "severe", "35", "35_2", 0.0, 3.0, wrecks=[w])
    i.t_contact = 100.3
    i.police_record = {"due": 330.5, "sent": None, "on_scene": None, "how": None, "cars": 0}
    im.incidents = [i]
    snap = im.snapshot()[0]
    assert snap["t_contact"] == 100.3 and snap["police_record"]["due"] == 330.5
    assert snap["wrecks"][0]["t_hit"] == 100.3 and snap["wrecks"][0]["speed"] == 14.0
    assert not math.isnan(snap["severity"])

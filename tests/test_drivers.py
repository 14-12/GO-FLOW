"""Drivers at the wheel (cosim/drivers.py): their places in the models."""

from __future__ import annotations

import json
from pathlib import Path

from cosim.drivers import POSE, seat_of, seat_rel

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / "config" / "vehicle_models.json").read_text(encoding="utf-8"))


def test_every_four_wheeler_has_a_seat_inside_its_box_on_the_left():
    n = 0
    for m, info in MODELS.items():
        rel = seat_rel(m, info)
        if info["base_type"] in ("bicycle", "motorcycle") or any(w in m for w in ("crossbike", "century", "omafiets")):
            assert rel is None, m
            continue
        assert rel is not None, m
        x, y, z = rel[:3]
        ex, ey, ez = info["extent"]
        cx, _, cz = info["centre"]
        assert cx - ex < x < cx + ex and -ey < y < 0.0, m  # inside the box, the driver's (left) side
        assert 0.3 < z < cz + ez - 0.8, m                   # above the floor, the head under the roof
        n += 1
    assert n >= 25


def test_low_cars_keep_the_head_under_the_roof():
    x, y, z = seat_of("vehicle.ford.mustang", (2.36, 0.95, 0.65), (0.03, 0.65))
    assert z + 0.04 + 0.85 <= 1.3  # roof at 1.30 m
    assert seat_of("vehicle.dodge.charger_2020", (2.5, 0.94, 0.77), (0.0, 0.77))[2] > 0.5


def test_the_pose_sits():
    assert POSE["crl_thigh__L"][2] < -80 and POSE["crl_thigh__R"][2] < -80  # thighs forward (roll -: forward)
    assert POSE["crl_leg__L"][2] > 30 and POSE["crl_leg__R"][2] > 30        # knees bent

"""The Tunisian number plates (cosim/plates.py) and where each model carries them (config/vehicle_models.json)."""

from __future__ import annotations

import random

import pytest

from cosim import plates as pl
from cosim import vmodels


def test_plates_are_never_repeated_and_look_plausible():
    rng, used = random.Random(1), set()
    got = [pl.random_plate(rng, used) for _ in range(3000)]
    assert len({(p.series, p.number) for p in got}) == 3000
    assert all(1 <= p.series <= pl.NEWEST_SERIES and 1 <= p.number <= 9999 for p in got)
    series = sorted(p.series for p in got)
    assert series[len(series) // 2] > pl.NEWEST_SERIES - 60   # most cars on the road are recent
    assert all(p.kind == "private" for p in got)  # no blue rental plates on the road (24 Oct)


def test_plate_text_latin_and_arabic():
    p = pl.Plate(247, 3816)
    assert p.text == "247 TU 3816"
    assert p.arabic == "247 تونس 3816"


def test_rendered_plate_is_the_texture_size_white_on_black_or_blue():
    pytest.importorskip("PIL")
    img = pl.render(pl.Plate(198, 7528))
    assert img.size == (pl.TEX_W, pl.TEX_H)
    assert abs(pl.TEX_W / pl.TEX_H - 520 / 110) < 0.05      # the rear plate's proportions
    px = img.convert("RGB").getpixel((pl.TEX_W // 2, 4))       # the black ground inside the border, top centre
    assert max(px) < 70
    ink = [img.getpixel((x, y)) for x in range(pl.TEX_W // 8) for y in range(pl.TEX_H)]
    assert max(min(c) for c in ink) > 180                      # the white series digits on the left
    blue = pl.render(pl.Plate(198, 7528, "rental")).getpixel((pl.TEX_W // 2, 4))
    assert blue[2] > blue[0] + 40                               # a rental's plate is blue


def test_every_car_model_has_plates_where_a_plate_belongs():
    models = vmodels.load()
    four_wheels = [m for m, i in models.items() if i["base_type"] in ("car", "van", "truck", "bus")
                   and "firetruck" not in m]
    for m in four_wheels:
        ends = set((models[m].get("plates") or {}))
        assert ends == {"rear", "front"}, (m, ends)
    for m, i in models.items():
        if i["base_type"] == "motorcycle":
            assert set(i.get("plates") or {}) == {"rear"}, m


def test_plates_sit_on_the_car_body_and_face_out():
    models = vmodels.load()
    for m, i in models.items():
        ex, ey, ez = i["extent"]
        cz = i["centre"][2]
        for end, (x, y, z, pitch, yaw, scale) in (i.get("plates") or {}).items():
            # at the body's end (a plate is on the bumper or the boot lid, never inside the car), centred, on the body
            assert abs(abs(x) - ex) < 0.45 * ex, (m, end, x, ex)
            assert (x < 0) == (end == "rear"), (m, end)
            assert abs(y) + 0.26 * scale < ey, (m, end, y)   # within the body's width (a Wrangler's is off centre)
            assert 0.15 < z < cz + ez, (m, end, z)
            assert yaw == (0.0 if end == "rear" else 180.0), (m, end, yaw)
            assert abs(pitch) < 45.0, (m, end, pitch)
            # SM_TnPlate is 52 cm wide at scale 1 (tools/plates/make_plate_mesh.py): rear 40-52 cm, front 28-45 cm
            width = 0.52 * scale
            lo, hi = (0.28, 0.521) if end == "rear" and i["base_type"] == "motorcycle" else \
                ((0.399, 0.521) if end == "rear" else (0.279, 0.451))
            assert lo <= width <= hi, (m, end, width)


def test_police_cars_and_ambulances_carry_the_state_plate():
    rng, used = random.Random(4), set()
    for model in ("vehicle.dodge.charger_police_2020", "vehicle.ford.ambulance"):
        p = pl.plate_for(model, rng, used)
        assert p.kind == "state" and p.series == pl.STATE_CODES[model]
        assert p.text == f"{p.series:02d} - {p.number:05d}"
    assert pl.plate_for("vehicle.audi.tt", rng, used).kind in ("private", "rental")
    got = [pl.plate_for("vehicle.ford.ambulance", rng, used).number for _ in range(500)]
    assert len(set(got)) == 500


def test_state_plate_is_red_on_white():
    pytest.importorskip("PIL")
    img = pl.render(pl.state_plate(random.Random(2), 12)).convert("RGB")
    assert img.size == (pl.TEX_W, pl.TEX_H)
    assert min(img.getpixel((pl.TEX_W // 2, 5))) > 190            # white ground
    reds = [img.getpixel((x, pl.TEX_H // 2)) for x in range(pl.TEX_W)]
    assert any(r > 150 and g < 80 and b < 80 for r, g, b in reds)  # red characters

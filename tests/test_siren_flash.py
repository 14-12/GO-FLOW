"""(10 Nov) The siren-light model (cosim/siren_flash.py, models/siren_flash.json from tools/ev_flash_model.py)."""

import numpy as np

from cosim.siren_flash import SirenFlash, light_numbers, window_features


def _bar(red, blue, w=60, h=40):
    """A vehicle's box: grey body, a bar on top lit red and/or blue."""
    img = np.full((h, w, 3), 120, np.uint8)
    if red:
        img[2:6, 5:28] = (250, 40, 40)
    if blue:
        img[2:6, 32:55] = (40, 60, 250)
    return light_numbers(img, (0, 0, w - 1, h - 1))


def test_flashing_bar_is_on_steady_body_is_off():
    m = SirenFlash()
    flashing = [_bar(i % 2 == 0, i % 2 == 1) for i in range(m.pictures)]
    steady = [_bar(False, False) for _ in range(m.pictures)]
    assert m.prob(flashing) > 0.9
    assert m.prob(steady) < 0.5
    assert m.prob(flashing[:-1]) is None            # too few pictures yet


def test_window_needs_the_vehicle_in_half_the_pictures():
    s = np.full((8, 4), np.nan)
    s[:3] = 0.1
    assert window_features(s) is None
    s[:4] = 0.1
    assert window_features(s) is not None

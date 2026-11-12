"""(12 Nov) The free-flight radar (cosim/radar.py) and the sky clock (cosim/hud.py, ui.ALPHA)."""

import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from cosim import ui


@pytest.fixture(autouse=True)
def _alpha(monkeypatch):
    pygame.init()
    pygame.font.init()
    monkeypatch.setattr(ui, "ALPHA", True)


def _land(px) -> bool:
    """The map's empty ground (the scaled layer may be a level or two darker)."""
    from cosim import radar

    return all(abs(a - b) <= 4 for a, b in zip(px[:3], radar.LAND))


def _router():
    roads = {"ns": SimpleNamespace(id="ns", shape=[(130.0, 0.0), (130.0, 400.0)], lanes=2, speed=13.9),
             "ew": SimpleNamespace(id="ew", shape=[(0.0, 50.0), (400.0, 50.0)], lanes=1, speed=13.9)}
    return SimpleNamespace(roads=roads)


def test_heading_up_turns_the_map_and_the_cars_alike():
    from cosim import radar

    r = radar.Radar(_router(), [])
    # looking east (heading 90) from (100, 200): the road 30 m east runs across the map above the centre
    free = {"cam": (100.0, 200.0, 90.0), "z": 20.0, "cars": [("a", 130.0, 200.0, 10.0)], "alerts": []}
    for i in range(20):   # the zoom eases in (by time)
        out = r.render(free, i * 0.2)
    c = radar.D / 2
    s = c / r.R
    road_y = int(c - 30 * s)
    assert not _land(r.disc.get_at((int(c - 40), road_y)))     # the road, left of the cone
    assert _land(r.disc.get_at((int(c - 40), int(c + 30 * s))))   # nothing behind the camera
    assert r.road_of(130.0, 200.0) == "ns"
    assert out.get_width() >= radar.D + 2 * radar.BEZEL


def test_far_alert_is_a_blip_with_its_distance():
    from cosim import radar

    r = radar.Radar(_router(), [])
    free = {"cam": (130.0, 390.0, 0.0), "z": 10.0, "cars": [], "alerts": [{"route": "ew", "type": "accident"}]}
    for i in range(10):
        r.render(free, i * 0.2)
    assert r.R < 200     # low: zoomed in
    free["z"] = 400.0
    for i in range(10, 40):
        r.render(free, i * 0.2)
    assert r.R > 300     # high: zoomed out


def test_moving_between_traffic_layers_matches_a_fresh_one():
    """The traffic layer is redrawn 5 times a second; in between each picture turns and shifts it. After the camera
    moved 6 m and turned, the shifted layer puts the roads where a freshly drawn one does (within a pixel)."""
    from cosim import radar

    def edges(disc):
        c = int(radar.D / 2)
        col = [sum(disc.get_at((c - 40, y))[:3]) > 180 for y in range(20, radar.D - 20)]   # (a road's fill)
        row = [sum(disc.get_at((x, c + 30))[:3]) > 180 for x in range(20, radar.D - 20)]
        return [i for i in range(1, len(col)) if col[i] != col[i - 1]], [i for i in range(1, len(row)) if row[i] != row[i - 1]]

    r = radar.Radar(_router(), [])
    free = {"cam": (110.0, 70.0, 80.0), "z": 20.0, "cars": [], "alerts": []}
    for i in range(30):
        r.render(free, i * 0.01)
    t_layer = r._w["t"]
    free["cam"] = (104.3, 66.1, 63.0)
    r.render(free, t_layer + 0.05)
    assert r._w["t"] == t_layer          # (shifted, not redrawn)
    shifted = edges(r.disc)
    r._w = None
    r.render(free, t_layer + 0.05)
    fresh = edges(r.disc)
    for a, b in zip(shifted, fresh):
        assert len(a) == len(b) and len(a) > 0
        assert all(abs(x - y) <= 1 for x, y in zip(a, b))


def test_bearing_names():
    from cosim.radar import _bearing_name

    assert [_bearing_name(h) for h in (0, 44, 46, 90, 181, 270, 359)] == ["N", "NE", "NE", "E", "S", "W", "N"]


@pytest.mark.parametrize("hour,el", [(3.0, -35.0), (7.25, 8.0), (12.5, 55.0), (19.75, -2.5), (22.0, -20.0)])
def test_sky_clock_draws_every_hour(hour, el):
    from cosim.hud import Hud, HudState

    h = Hud()
    st = HudState(game_s=hour * 3600, sun_elevation=el)
    card = h.card(st)
    assert card.get_size() == (h.W, h.H)
    assert h.sky_col_x + 60 < h.W


def test_sky_clock_windows_light_up_at_night():
    from cosim.hud import Hud, HudState

    h = Hud()

    def warm(card):
        n = 0
        for x in range(12, h.SKY_W + 8, 2):
            for y in range(90, h.H - 12, 2):
                r, g, b, _ = card.get_at((x, y))
                n += r > 200 and g > 150 and b < 150
        return n

    night = warm(h.card(HudState(game_s=21.5 * 3600, sun_elevation=-15.0)).copy())
    day = warm(h.card(HudState(game_s=12.0 * 3600, sun_elevation=55.0)).copy())
    assert night > 20 and day == 0

"""cosim/hud.py: the clock card says how fast the day goes (7 Nov: it said "1 MIN = 1 HOUR" at any --clock-speed)."""
import pytest

pytest.importorskip("pygame")

from cosim.hud import HudState, speed_label  # noqa: E402


def test_speed_label():
    assert speed_label(60) == "1 MIN = 1 HOUR"
    assert speed_label(30) == "2 MIN = 1 HOUR"
    assert speed_label(20) == "3 MIN = 1 HOUR"
    assert speed_label(24) == "2.5 MIN = 1 HOUR"
    assert speed_label(1) == "REAL TIME"
    assert speed_label(45) == "45 X REAL TIME"


def test_the_state_carries_the_speed():
    from cosim.daily import GAME_SPEED

    assert HudState(game_s=0.0).speed == GAME_SPEED == 30.0
    assert HudState(game_s=0.0, speed=30.0).speed == 30.0

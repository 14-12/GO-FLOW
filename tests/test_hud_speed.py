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


def test_the_strip_says_who_runs_the_signals_and_the_traffic_level():
    """(9 Nov) A picture says what it was taken under: the screenshots of 9 Nov were taken under the fixed-cycle
    signals and, from the forecast launcher, twice the traffic, and nothing on screen said so."""
    from cosim.hud import setup_text

    assert setup_text("coordinated") == "fixed-cycle signals"
    assert setup_text("agents_observed", 2.0) == "junction agents   ·   traffic x2"
    assert setup_text("agents_observed_goflow+forecast") == "junction agents + GO-FLOW priority + forecasts"
    assert setup_text("max_pressure") == "max_pressure"
    assert setup_text("agents_observed+bus@bus_cut=1") == "junction agents + bus priority"

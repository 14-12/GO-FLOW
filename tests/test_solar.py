"""Sun position: values the main project checked against `astral` (environment.ts)."""

import pytest

from cosim.solar import scenario_sun


def test_ingolstadt_16h_midsummer():
    az, el = scenario_sun(16 * 3600, 48.766, 11.425)
    # Afternoon sun in the west-south-west, high in the sky.
    assert 240 < az < 255 and 45 < el < 52


def test_night_is_below_horizon():
    _, el = scenario_sun(23.5 * 3600, 48.766, 11.425)
    assert el < -5


def test_cologne_07h_morning_east():
    az, el = scenario_sun(7 * 3600, 50.94, 6.96)
    assert 60 < az < 90 and 10 < el < 25


def test_carla_azimuth_matches_the_shadow_measurement():
    from cosim.solar import carla_sun_azimuth

    # out/sun_check.json: azimuth 0 -> sun at bearing 269.0, azimuth 180 -> 90.8.
    assert carla_sun_azimuth(270.0) == 0.0
    assert carla_sun_azimuth(90.0) == 180.0
    assert carla_sun_azimuth(301.06) == pytest.approx(31.06)  # Ingolstadt, 20.7 h

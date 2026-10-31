import carla

from cosim.convert import BLINKER_LEFT, BRAKELIGHT, light_state, tl_state, vehicle_transform
from cosim.mapping import Frame

S = carla.TrafficLightState


def test_tl_agreeing_links():
    assert tl_state(["G", "g"]) == (S.Green, False)
    assert tl_state(["y"]) == (S.Yellow, False)
    assert tl_state(["r", "u"]) == (S.Red, False)


def test_tl_disagreeing_links_show_most_restrictive():
    assert tl_state(["G", "r"]) == (S.Red, True)
    assert tl_state(["G", "y"]) == (S.Yellow, True)


def test_lights():
    v = carla.VehicleLightState
    s = light_state(BRAKELIGHT | BLINKER_LEFT, night=False)
    assert int(s) & int(v.Brake) and int(s) & int(v.LeftBlinker)
    assert not int(s) & int(v.LowBeam)
    assert int(light_state(0, night=True)) & int(v.LowBeam)


def test_transform_front_bumper_to_centre_and_axes():
    f = Frame(100.0, 200.0)
    # Heading east (SUMO 90 deg), front bumper at (110, 200), 4 m long: centre at (108, 200).
    t = vehicle_transform(f, 110.0, 200.0, 90.0, 4.0)
    assert abs(t.location.x - 8.0) < 1e-6 and abs(t.location.y) < 1e-6
    assert abs(t.rotation.yaw - 0.0) < 1e-6
    # Heading north (SUMO 0 deg) at y = 210: CARLA y is flipped, yaw -90.
    t = vehicle_transform(f, 100.0, 210.0, 0.0, 4.0)
    assert abs(t.location.y - (-8.0)) < 1e-6 and abs(t.rotation.yaw + 90.0) < 1e-6

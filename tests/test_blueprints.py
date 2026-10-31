from cosim.blueprints import Candidate, choose

LIB = [
    Candidate("vehicle.a.old", "car", "", 1, ("1,1,1",), has_lights=False),
    Candidate("vehicle.b.new", "car", "", 2, ("255,0,0", "0,0,255")),
    Candidate("vehicle.c.new", "car", "electric", 2, ()),
    Candidate("vehicle.taxi", "car", "taxi", 2, ()),
    Candidate("vehicle.police", "car", "emergency", 2, ()),
    Candidate("vehicle.ambulance", "van", "emergency", 2, ()),
    Candidate("vehicle.sprinter", "van", "", 2, ()),
    Candidate("vehicle.bus", "bus", "", 2, ()),
    Candidate("vehicle.truck", "truck", "", 1, ()),
    Candidate("vehicle.cybertruck", "truck", "electric", 2, ()),
    Candidate("vehicle.d.lit_old", "car", "", 1, ()),
]


def test_classes_get_matching_models():
    assert choose(LIB, "bus", "x")[0] == "vehicle.bus"
    assert choose(LIB, "emergency", "x")[0] == "vehicle.ambulance"
    assert choose(LIB, "authority", "x")[0] == "vehicle.police"
    assert choose(LIB, "truck", "x")[0] == "vehicle.truck"


def test_passenger_cars_have_lights_and_are_never_emergency():
    ids = {choose(LIB, "passenger", f"veh{i}")[0] for i in range(200)}
    assert ids == {"vehicle.b.new", "vehicle.c.new", "vehicle.d.lit_old"}


def test_electric_models_only_for_cars():
    assert {choose(LIB, "truck", f"t{i}")[0] for i in range(50)} == {"vehicle.truck"}


def test_choice_is_stable_per_vehicle():
    assert choose(LIB, "passenger", "carIn126006:1") == choose(LIB, "passenger", "carIn126006:1")


def test_unknown_class_falls_back_to_a_car():
    assert choose(LIB, "rail", "x")[0] in {"vehicle.b.new", "vehicle.c.new", "vehicle.d.lit_old"}

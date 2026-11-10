"""(10 Nov) The H window labels each vehicle by its track's vote, not by one picture (cosim/inspector.TrackVote)."""

from cosim.inspector import TrackVote, count


def _box(i, x0=100.0):
    x = x0 + 3.0 * i
    return (x, 200.0, x + 60.0, 240.0)


def test_siren_on_confirmed_then_kept():
    v = TrackVote()
    seq = ["ambulance", "ambulance_lights", "ambulance", "ambulance_lights", "car", "ambulance_lights",
           "ambulance", "ambulance", "ambulance", "car"]
    labels = [v.relabel(0.2 * i, [(*_box(i), n, 0.8)])[0][4] for i, n in enumerate(seq)]
    assert labels[:5] == ["ambulance"] * 5           # 1-2 votes of 3: not yet
    assert labels[5:] == ["ambulance_lights"] * 5    # the third: on, and kept
    assert count([(*_box(9), labels[-1], 0.8)])["emergency, SIREN ON"] == 1


def test_one_picture_does_not_make_a_car_an_emergency_vehicle():
    v = TrackVote()
    seq = ["car", "car", "police_lights", "car", "car"]
    labels = [v.relabel(0.2 * i, [(*_box(i), n, 0.7)])[0][4] for i, n in enumerate(seq)]
    assert labels == ["car"] * 5


def test_siren_off_police_stays_off_and_two_vehicles_apart():
    v = TrackVote()
    for i in range(8):
        n = "police_lights" if i in (2, 6) else ("police" if i % 3 else "car")
        out = v.relabel(0.2 * i, [(*_box(i), n, 0.7), (*_box(i, 500.0), "car", 0.9)])
    assert out[0][4] == "police" and out[1][4] == "car"

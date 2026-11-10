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


class _Flash:
    """A stand-in for the siren-light model: lights on when the red number swings over 0.1."""
    pictures, threshold = 4, 0.5

    def prob(self, series):
        if len(series) < self.pictures:
            return None
        red = [s[0] for s in series[-self.pictures:]]
        return 0.9 if max(red) - min(red) > 0.1 else 0.1


def test_siren_light_model_decides_once_it_has_its_pictures():
    v = TrackVote(flash=_Flash())
    on, off = [0.05, 0.05, 0.05, 0.05], [0.3, 0.05, 0.05, 0.05]
    labels = []
    for i in range(8):
        # the detector says siren on in 3 of the first 4 pictures, but the bar does not flash
        n = "police_lights" if i in (0, 1, 2) else "police"
        labels.append(v.relabel(0.2 * i, [(*_box(i), n, 0.7, on)])[0][4])
    assert labels[2] == "police_lights"            # the detector's vote, before the model has 4 pictures
    assert labels[3:] == ["police"] * 5             # then the model: no flashing, siren off
    v = TrackVote(flash=_Flash())
    labels = [v.relabel(0.2 * i, [(*_box(i), "ambulance", 0.7, off if i % 2 else on)])[0][4] for i in range(6)]
    assert labels[3:] == ["ambulance_lights"] * 3   # never called siren on by the detector, but its bar flashes
    assert all(len(b) == 6 for b in v.relabel(2.0, [(*_box(10), "car", 0.9, on)]))

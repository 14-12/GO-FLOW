import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents/incident_detection/evaluation"))
from crash_features import window_crash_features  # noqa: E402

FPS = 5


def track(track_id, positions, start=0.0, h=40.0, w=40.0):
    """positions: list of (x, y), one per frame at 5 fps."""
    return [{"source_time_sec": round(start + i / FPS, 3), "track_id": track_id, "class_id": 2, "category": "car",
             "confidence": 0.9, "center_x": x, "center_y": y, "bbox_height": h, "bbox_area": h * w}
            for i, (x, y) in enumerate(positions)]


def drive(x0, y, speed_px, frames):
    return [(x0 + speed_px * i, y) for i in range(frames)]


def features(rows, duration=30.0):
    return window_crash_features(pd.DataFrame(rows), duration).set_index("window_start_sec")


def test_sudden_stop_touching_another_car_is_a_crash_moment():
    # Car 1 drives at 2 bbox heights/s, stops dead at 3 s next to car 2, which stands there.
    moving = drive(0, 100, 16, 15) + [(224, 100)] * 60
    rows = track(1, moving) + track(2, [(250, 100)] * 75)
    f = features(rows)
    assert f.loc[0.0:5.0, "contact_hard_stops"].sum() >= 1
    assert f.crash_born_spot_age_s.max() > 5


def test_gradual_stop_in_a_queue_is_not():
    # Same distance but slowing down over five seconds, as in a jam.
    x, speed, positions = 0.0, 16.0, []
    for i in range(75):
        positions.append((x, 100))
        x += speed
        speed = max(0.0, speed - 0.7)
    rows = track(1, positions) + track(2, [(x + 26, 100)] * 75)
    f = features(rows)
    assert f.contact_hard_stops.sum() == 0 and f.crash_born_spot_age_s.max() == 0


def test_cars_passing_a_stopped_car_the_same_way_count_as_avoidance():
    # Car 1 drives east and stops at x=200; from 25 s on, cars 10+ drive east past it one lane over.
    stopped = drive(0, 100, 16, 13) + [(192, 100)] * 200
    rows = track(1, stopped)
    for n in range(5):
        rows += track(10 + n, drive(100, 140, 16, 15), start=25 + 2 * n)
    rows += track(20, drive(300, 140, -16, 15), start=26)   # one car going the other way
    f = features(rows, duration=45.0)
    assert f.loc[25.0:40.0, "passing_same_direction"].sum() >= 4
    assert f.passing_any_direction.sum() > f.passing_same_direction.sum()


def test_a_jam_where_nobody_moves_has_no_avoidance():
    rows = []
    for n in range(6):
        rows += track(n, drive(0, 100 + 40 * n, 16, 10) + [(144, 100 + 40 * n)] * 200)
    f = features(rows, duration=40.0)
    assert f.passing_same_direction.sum() == 0 and f.passing_any_direction.sum() == 0

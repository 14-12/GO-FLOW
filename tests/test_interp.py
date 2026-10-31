"""Hermite motion between SUMO steps (cosim/interp.py)."""


from cosim.interp import State, hermite


def ang_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_ends_are_the_sumo_states():
    a, b = State(0, 0, 0, 10), State(0, 10, 0, 10)
    assert hermite(a, b, 0.0, 1.0) == (0, 0, 0)
    assert hermite(a, b, 1.0, 1.0) == (0, 10, 0)


def test_straight_constant_speed_is_uniform():
    a, b = State(0, 0, 90, 12), State(12, 0, 90, 12)  # east at 12 m/s
    xs = [hermite(a, b, k / 20, 1.0)[0] for k in range(21)]
    steps = [xs[i + 1] - xs[i] for i in range(20)]
    assert max(steps) - min(steps) < 1e-9
    assert all(ang_diff(hermite(a, b, k / 20, 1.0)[2], 90) < 1e-6 for k in range(1, 20))


def test_quarter_turn_heading_changes_monotonically():
    # North at the start, east at the end: a quarter circle of radius 10 m (15.7 m of arc)
    # driven in one 1-s step, so 15.7 m/s at both ends.
    a, b = State(0, 0, 0, 15.7), State(10, 10, 90, 15.7)
    hs = [hermite(a, b, k / 40, 1.0)[2] for k in range(41)]
    assert all(hs[i + 1] >= hs[i] - 1e-9 for i in range(40))
    # A true circle turns 90/40 = 2.25 deg per sample; allow 2x that, no jumps.
    assert max(hs[i + 1] - hs[i] for i in range(40)) < 4.5


def test_stopping_car_does_not_overshoot():
    a, b = State(0, 0, 0, 3), State(0, 1.0, 0, 0)  # brakes to a stop 1 m ahead
    ys = [hermite(a, b, k / 20, 1.0)[1] for k in range(21)]
    assert max(ys) <= 1.0 + 1e-9
    assert all(ys[i + 1] >= ys[i] - 1e-9 for i in range(20))


def test_standing_still_keeps_heading():
    a = b = State(5, 5, 123, 0)
    assert hermite(a, b, 0.5, 1.0) == (5, 5, 123)

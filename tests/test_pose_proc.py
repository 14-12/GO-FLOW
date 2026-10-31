"""The per-picture pose process (cosim/pose_proc.py): its poses, and its judder measurement."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

carla = pytest.importorskip("carla")

from cosim.interp import Timed  # noqa: E402
from cosim.pose_proc import FAR_M, NEAR_M, Driver, FramePoser, Judder  # noqa: E402


def _seg(start=0.0, dt=0.5):
    pts = [(0.0, 0.0), (5.0, 0.0), (10.5, 0.0), (16.5, 0.0)]  # speeding up along +x (SUMO coordinates)
    return {"start": start, "dt": dt,
            "veh": [(7, "car", *pts, 4.5, 90.0, 0.0, 1.0, 0.0, 2.0),
                    (8, "new", None, None, (50.0, 50.0), None, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0)],
            "walk": [(9, (0.0, 0.0, 0.0, 170.0), (1.0, 0.0, 0.0, -170.0), 1.2)]}


def test_frame_poser_follows_the_viewer_curve_and_holds_at_the_end():
    fp = FramePoser(0.0, 0.0)
    fp.set_segment(_seg())
    tm = Timed((0.0, 0.0), (5.0, 0.0), (10.5, 0.0), (16.5, 0.0))
    for t in (0.0, 0.1, 0.25, 0.4):
        s = fp.s_at(t)
        (aid, tf), (aid2, tf2) = fp.vehicles(s)
        x, _ = tm.at(t / 0.5)
        # the body's centre is half a length behind the front, along +x; CARLA y is SUMO -y
        assert aid == 7 and tf.location.x == pytest.approx(x - 2.25, abs=1e-6)
        assert tf.location.z == pytest.approx(s)  # height blended across the step
        assert aid2 == 8 and tf2.location.y == pytest.approx(-50.0 + 2.25, abs=1e-6)  # no step before: in place
    assert fp.s_at(0.9) == 1.0 and fp.held == 1  # the next step is late: hold its end
    fp.frozen_s = 0.3
    assert fp.s_at(5.0) == 0.3  # paused


def test_walkers_turn_the_short_way():
    fp = FramePoser(0.0, 0.0)
    fp.set_segment(_seg())
    (_, tf, speed), = fp.walkers(0.5)
    assert tf.location.x == pytest.approx(0.5) and speed == 1.2
    assert abs((tf.rotation.yaw + 180.0) % 360.0 - 180.0) == pytest.approx(180.0, abs=1e-6)  # through 180, not 0


class _Snap:
    def __init__(self, xs):
        self.xs = xs

    def find(self, aid):
        x = self.xs.get(aid)
        if x is None:
            return None
        return NS(get_transform=lambda: NS(location=NS(x=x, y=0.0)))


def _judder(pose_rate=None, fps=90.0, seconds=3.0, speed=10.0):
    """A car at a steady speed; CARLA draws `fps` pictures a second. pose_rate: poses sent at this rate on
    their own clock (the old way: each picture shows the last one sent); None: one per picture."""
    j = Judder()
    dt = 1.0 / fps
    for k in range(int(seconds * fps)):
        t = k * dt
        shown_t = t if pose_rate is None else int(t * pose_rate) / pose_rate
        j.picture(_Snap({1: speed * shown_t}), [1], dt)
    counted, judders = j.take()
    return judders / counted if counted else 0.0


def test_judder_measures_poses_out_of_step_with_the_pictures():
    assert _judder(pose_rate=None) == 0.0          # one pose per picture: even
    assert _judder(pose_rate=60.0) > 0.3           # 60 a second against 90 pictures: repeats and doubles
    assert _judder(pose_rate=44.0) > 0.3           # what the log showed
    assert _judder(pose_rate=None, speed=1.0) == 0.0  # creeping cars are not counted


def test_feeder_packs_a_step_into_a_segment():
    import queue

    from cosim.freefly import FrameFeeder

    car = lambda x: NS(x=x, y=0.0, length=4.5, angle=90.0)  # noqa: E731
    poser = NS(prev=NS(vehicles={"a": car(0.0)}), cur=NS(vehicles={"a": car(5.0), "b": car(30.0)}),
               pre={"a": car(-5.0)}, nxt=None, _from={"a": (0.0, 0.0), "b": (1.0, 0.0)},
               _g={"a": (0.5, 1.0), "b": (1.0, 0.0)})
    tf = lambda x: NS(location=NS(x=x, y=1.0, z=0.2), rotation=NS(yaw=10.0))  # noqa: E731
    lights = []
    cs = NS(actors={"a": 11, "b": 12}, walkers={"p": 21}, parked_v=[13], parked_w=[22],
            light_changes=lambda ls: (lights.append(ls), [(11, 3)])[1])
    ff = NS(cs=cs, sumo=NS(step_length=0.5))
    pk = NS(poser=poser, walk={"p": (tf(0.0), tf(0.6), 1.3), "gone": (tf(0.0), tf(0.0), 0.0)}, lights={"a": 1})
    q = queue.Queue()
    feeder = FrameFeeder(ff, NS(q=queue.Queue()), NS(paused=False), pk, q)
    feeder.send(pk, 100.0)
    kind, seg, at_once = q.get_nowait()
    assert not at_once
    assert kind == "seg" and seg["start"] == 100.0 and seg["dt"] == 0.5
    a, b = seg["veh"]
    assert a[:6] == (11, "a", (-5.0, 0.0), (0.0, 0.0), (5.0, 0.0), (10.0, 0.0))  # p3 extrapolated: no step after
    assert a[8:12] == (0.0, 0.5, 0.0, 1.0)
    assert a[12:] == (None, None, None, None, None)  # no lane angle, speed or damage known in this fake
    assert b[:6] == (12, "b", None, None, (30.0, 0.0), None)  # new this step
    assert seg["walk"] == [(21, (0.0, 1.0, 0.2, 10.0), (0.6, 1.0, 0.2, 10.0), 1.3)]
    assert seg["park_v"] == [13] and seg["park_w"] == [22]
    assert cs.parked_v == [] and cs.parked_w == []  # handed over once
    assert lights == [{"a": 1}]
    assert seg["lights"] == [(11, 3)]  # the lights that changed go with the poses


def test_driver_takes_over_a_segment_sent_ahead_at_its_start_with_its_parks():
    d = Driver(0.0, 0.0)
    first = dict(_seg(start=0.0), park_v=[], park_w=[])
    d.receive(("seg", first, True))
    nxt = dict(_seg(start=0.5), park_v=[50, 7], park_w=[60])  # 7 is in use again: not parked
    d.receive(("seg", nxt, False))
    vs, ws, pv, pw, posed, _ = d.picture(0.45)
    assert d.poser.seg is first and not pv and d.poser.held == 0  # still the first step
    vs, ws, pv, pw, posed, _ = d.picture(0.55)
    assert d.poser.seg is nxt and pv == [50] and pw == [60]
    assert d.poser.s_at(0.55) == pytest.approx(0.1)  # no held picture across the step boundary


def test_driver_skips_actors_that_have_not_moved():
    d = Driver(0.0, 0.0)
    d.receive(("seg", _seg(start=0.0), True))
    vs, ws, *_ , skipped = d.picture(0.1)
    assert {a for a, _ in vs} == {7, 8} and skipped == 0
    vs, ws, *_ , skipped = d.picture(0.2)
    assert {a for a, _ in vs} == {7} and skipped == 1  # 8 stands where it was


def test_driver_keeps_the_parks_of_a_segment_cancelled_by_a_pause():
    d = Driver(0.0, 0.0)
    d.receive(("seg", _seg(start=0.0), True))
    d.receive(("seg", dict(_seg(start=0.5), park_v=[50], park_w=[]), False))
    d.receive(("pause", 0.9))
    assert not d.pending and d.poser.s_at(3.0) == 0.9
    d.receive(("seg", dict(_seg(start=10.0), park_v=[], park_w=[]), True))  # resent on resume
    *_, pv, pw, posed, skipped = d.picture(10.0)
    assert pv == [50]


def test_feeder_sends_each_step_ahead_and_evenly():
    import queue
    import time as _t

    from cosim.freefly import FrameFeeder

    class Feeder(FrameFeeder):
        def segment(self, pk, start):
            return {"start": start, "veh": [], "walk": [], "pk": pk, "sent_at": _t.time()}

    worker_q = queue.Queue()
    for k in range(1, 6):
        worker_q.put(NS(lights={}, k=k))
    applied = []
    ff = NS(sumo=NS(step_length=0.5), cs=NS(light_changes=lambda lights: []), apply=applied.append)
    q = queue.Queue()
    f = Feeder(ff, NS(q=worker_q), NS(paused=False), NS(lights={}, k=0), q)
    f.start()
    _t.sleep(1.6)
    f.stop = True
    f.join(timeout=1.0)
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    assert msgs[0][2] is True and all(m[2] is False for m in msgs[1:])  # only the first takes over at once
    starts = [m[1]["start"] for m in msgs]
    assert len(msgs) >= 4
    for a, b in zip(starts, starts[1:], strict=False):
        assert b - a == pytest.approx(0.5, abs=1e-9)  # evenly, step after step
    for m in msgs[1:]:
        lead = m[1]["start"] - m[1]["sent_at"]
        assert 0.15 <= lead <= 0.21  # sent ~0.2 s ahead of its start
    assert [p.k for p in applied] == [m[1]["pk"].k for m in msgs[1:]]


def test_distant_vehicles_take_turns():
    d = Driver(0.0, 0.0)
    seg = _seg(start=0.0)
    far = NEAR_M + 10.0  # CARLA x of a vehicle: SUMO x - 2.25 (the body's centre behind the front)
    veh = [(10 + i, f"v{i}", (x - 1, 0.0), (x, 0.0), (x + 1, 0.0), (x + 2, 0.0), 4.5, 90.0, 0, 0, 0, 0)
           for i, x in enumerate((10.0, far, far, FAR_M + 50.0, FAR_M + 50.0, FAR_M + 50.0, FAR_M + 50.0))]
    d.receive(("seg", dict(seg, veh=veh, walk=[]), True))
    sent = {a: 0 for a, *_ in veh}
    for k in range(8):
        vs, *_ = d.picture(0.05 * k, eye=(0.0, 0.0), k=k)
        for a, _ in vs:
            sent[a] += 1
    assert sent[10] == 8                       # near: every picture
    assert sent[11] == sent[12] == 4           # middle: every 2nd
    assert all(sent[a] == 2 for a in (13, 14, 15, 16))  # far: every 4th, each on its own pictures


def test_judder_counts_only_vehicles_near_the_camera():
    j = Judder()
    for k in range(30):
        t = k / 90.0
        shown = int(t * 30) / 30  # far away, posed every 3rd picture: looks like judder, but is not counted
        j.picture(_Snap({1: 1000.0 + 10.0 * shown, 2: 10.0 * t}), [1, 2], 1 / 90.0, eye=(0.0, 0.0))
    counted, judders = j.take()
    assert counted > 0 and judders == 0


def test_lag_finds_which_sent_pose_each_picture_shows():
    from cosim.pose_proc import Lag

    tf = lambda x: NS(location=NS(x=x, y=0.0))  # noqa: E731
    lag = Lag()
    shown = []
    for k in range(10):
        # the server shows the pose sent at the picture before, except every 3rd picture: the one before that
        back = 2 if k % 3 == 0 else 1
        if len(lag.sent) >= back:
            x = lag.sent[back - 1][1][0]
            lag.picture(_Snap({1: x}))
            shown.append(back)
        lag.sent_now([(1, tf(10.0 * k))])
    h = lag.take()
    assert h["1"] > 0.5 and h["2"] > 0.2 and "other" not in h


def test_lag_tells_how_many_batches_the_server_has_not_shown_yet():
    from cosim.pose_proc import Lag

    tf = lambda x: NS(location=NS(x=x, y=0.0))  # noqa: E731
    lag = Lag()
    for k in range(5):
        lag.sent_now([(1, tf(10.0 * k))])
    assert lag.picture(_Snap({1: 40.0})) == 0     # the last one sent is shown
    assert lag.picture(_Snap({1: 20.0})) == 2     # two sent since are waiting
    assert lag.picture(_Snap({1: -50.0})) == 5    # none of them: further behind than kept
    lag.sent_now([(1, tf(40.0))])                 # standing still: nothing to tell by
    assert lag.picture(_Snap({1: 40.0})) is None


def test_frame_clock_follows_the_server_and_ignores_late_callbacks():
    from cosim.pose_proc import FrameClock, Pacing

    fc = FrameClock(window_s=3.0)
    out = []
    for k in range(200):
        game = k / 40.0
        late = (k * 7919 % 13) / 1000.0  # 0-12 ms late, unevenly
        out.append(fc.wall(game, 1000.0 + game + 0.004 + late))
    steps = [b - a for a, b in zip(out[20:], out[21:], strict=False)]
    assert max(abs(s - 1 / 40.0) for s in steps) < 1e-3  # even, whatever the lateness
    p = Pacing()
    for dt in (0.02, 0.02, 0.04, 0.02, 0.021):
        p.picture(dt)
    assert p.take() == 0.5  # 2 of 4 changes over 1.5x


def test_driver_hands_the_lights_of_a_segment_over_once_it_starts():
    d = Driver(0.0, 0.0)
    d.receive(("seg", dict(_seg(start=0.0), lights=[(7, 3)]), True))
    d.receive(("seg", dict(_seg(start=0.5), lights=[(8, 1)]), False))
    assert d.lights == [(7, 3)]
    d.lights = []
    d.picture(0.55)
    assert d.lights == [(8, 1)]


def test_pose_reports_merge_weighted():
    from cosim.freefly import merge_pose_figures

    m = merge_pose_figures([
        {"span_s": 10.0, "pictures": 400, "moving_vehicle_pictures": 1000, "judder_share": 0.1, "uneven_pictures": 0.0,
         "pictures_waited_for_server": 2, "held": 0, "shown_lag": {"2": 1.0}},
        {"span_s": 5.0, "pictures": 100, "moving_vehicle_pictures": 3000, "judder_share": 0.0, "uneven_pictures": 0.5,
         "pictures_waited_for_server": 1, "held": 3, "shown_lag": {"2": 0.5, "3": 0.5}}])
    assert m["span_s"] == 15.0 and m["pictures_per_s"] == pytest.approx(33.3, abs=0.05)
    assert m["judder_share"] == 0.025 and m["uneven_pictures"] == 0.1
    assert m["pictures_waited_for_server"] == 3 and m["held"] == 3
    assert m["shown_lag"] == {"2": 0.9, "3": 0.1}


def test_drivers_sit_in_the_cars_nearest_the_camera():
    """cosim/drivers.py: the pool's walkers go with the nearest cars, not into wrecks or parked cars, and let go."""
    from cosim.pose_proc import DRIVER_KEEP_M

    def car(aid, x):
        return (aid, f"v{aid}", None, None, (x, 0.0), None, 4.5, 90.0, 0.0, 0.0, 0.0, 0.0)

    seat = (0.0, -0.4, 0.5, 0.0, 0.0, 0.0)
    drv = Driver(0.0, 0.0)
    drv.receive(("seg", {"start": 0.0, "dt": 0.5, "veh": [car(7, 10.0), car(8, 100.0), car(9, 20.0), car(10, 30.0)],
                         "walk": [], "empty": [9]}, True))
    drv.receive(("seats", {7: seat, 8: seat, 9: seat, 10: seat}))
    drv.receive(("drivers", [101]))
    vs, _, pv, _, posed, _ = drv.picture(0.0, eye=(0.0, 0.0))
    sent = dict(vs)
    assert drv.driver_in == {7: 101}  # the nearest; 8 too far, 9 parked or wrecked ("empty"), no walker left for 10
    car_tf = dict(posed)[7]
    want = car_tf.transform(carla.Location(0.0, -0.4, 0.5))
    assert sent[101].location.distance(want) < 1e-3 and sent[101].rotation.yaw == pytest.approx(car_tf.rotation.yaw)
    drv.receive(("door", 7, True))  # the driver got out (an accident scene): the walker goes to the next car
    vs, _, pv, _, _, _ = drv.picture(0.0, eye=(0.0, 0.0))
    assert drv.driver_in == {10: 101} and 101 not in pv
    vs, _, pv, _, _, _ = drv.picture(0.0, eye=(30.0 + DRIVER_KEEP_M + 10.0, 300.0))  # the camera flew away
    assert drv.driver_in == {} and 101 in pv and drv.drivers_free == [101]

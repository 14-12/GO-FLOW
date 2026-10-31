"""Camera placement (cosim/camera.py): the clear arc around a junction."""

import math

from cosim.camera import arc_pose, clear_arc


def blocked_between(lo, hi, label="Buildings"):
    """A fake cast_ray: positions whose angle (deg) around the origin lies in [lo, hi) see a wall."""
    def cast(a, b):
        ang = round(math.degrees(math.atan2(a[1], a[0])) % 360, 6)
        return [label, "Roads"] if lo <= ang < hi else ["Roads"]
    return cast


def test_full_circle_when_nothing_blocks():
    assert clear_arc(lambda a, b: ["Roads", "Poles"], 0, 0, None) == (0.0, 360.0)


def test_longest_clear_run_is_chosen():
    start, span = clear_arc(blocked_between(90, 200), 0, 0, None)
    assert (start, span) == (200.0, 250.0)  # 200..440 (= 80) deg, wrapping through 0


def test_trees_block_but_lamp_posts_do_not():
    assert clear_arc(blocked_between(0, 360, "Vegetation"), 0, 0, None) is None
    assert clear_arc(blocked_between(0, 360, "Poles"), 0, 0, None) == (0.0, 360.0)


def test_obstacle_beside_the_camera_rejects_the_position():
    """A sign next to the camera, missed by the ray to the target, still blocks the view: the
    short rays towards the picture's edges catch it."""
    def cast(a, b):
        ang = math.degrees(math.atan2(a[1], a[0])) % 360
        to_centre = abs(b[0]) < 1e-9 and abs(b[1]) < 1e-9 and b[2] == 1.5
        return ["Roads"] if to_centre or not (100 <= ang < 130) else ["Static", "Roads"]
    start, span = clear_arc(cast, 0, 0, None)
    assert (start, span) == (130.0, 330.0)  # 130..460 (= 100) deg: 100..125 rejected


def test_camera_looks_towards_the_map_centre():
    # Map centre far along +x: the camera must stand on the -x side, looking +x.
    start, span = clear_arc(lambda a, b: [], 0, 0, (1000.0, 0.0))
    mid = math.radians(start + span / 2)
    assert math.cos(mid) < -0.9


def test_arc_pose_stays_inside_the_arc():
    arc = (200.0, 120.0)
    for t in range(0, 200, 7):
        p = arc_pose(0, 0, t, arc)
        ang = math.degrees(math.atan2(p.y, p.x)) % 360
        assert 200 <= ang <= 320, (t, ang)


def test_tour_stops_by_count_or_by_name():
    from cosim.run import tour_stops

    centres = {"a": (0, 0, 4), "b": (1, 1, 3), "c": (2, 2, 2)}
    assert tour_stops(centres, "") == ["a", "b", "c"]
    assert tour_stops(centres, "2") == ["a", "b"]
    assert tour_stops(centres, "c,a") == ["c", "a"]


def test_draw_radius_keeps_the_district_with_a_margin():
    from cosim.mapping import Frame
    from cosim.run import within
    from cosim.sumo_side import Person, StepResult, Vehicle

    def veh(i, x):
        return Vehicle(i, "passenger", "t", 4.5, x, 0.0, 90.0, 10.0, 0)

    step = StepResult(0.0, {"near": veh("near", 50.0), "edge": veh("edge", 130.0), "far": veh("far", 500.0)}, [],
                      persons={"p": Person("p", 20.0, 0.0, 0.0, 1.0)})
    frame = Frame(0.0, 0.0)
    out = within(step, frame, 0.0, 0.0, 100.0, keep=set())
    assert set(out.vehicles) == {"near"} and set(out.persons) == {"p"}
    out = within(step, frame, 0.0, 0.0, 100.0, keep={"edge"})  # drawn already: kept to 160 m
    assert set(out.vehicles) == {"near", "edge"}


def test_poses_are_lifted_to_the_road_height():
    from cosim.camera import Pose
    from cosim.run import lift
    p = Pose(1.0, 2.0, 22.0, 90.0, -30.0)
    assert lift(p, 0.0) is p
    q = lift(p, 361.5)  # Town12's centre: the road ~360 m up
    assert (q.x, q.y, q.z, q.yaw, q.pitch) == (1.0, 2.0, 383.5, 90.0, -30.0)


def test_settle_waits_for_fast_ticks(monkeypatch):
    import cosim.run as run
    durations = iter([5.0, 4.0, 0.1, 0.1, 3.0] + [0.1] * 20)  # slow, then a relapse, then steady
    clock = {"t": 0.0}
    monkeypatch.setattr(run.time, "time", lambda: clock["t"])

    def tick():
        clock["t"] += next(durations)
    assert run.settle(tick, fast_s=1.0, need=10) == 15

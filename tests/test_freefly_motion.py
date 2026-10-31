"""Free-fly viewer motion: the pose thread (cosim.freefly.Mover), the speed-true timing of each step's curve
(cosim.interp.Timed) and the pedestrians' pavement glide (cosim.carla_side.CarlaSide.walker_pose)."""

from __future__ import annotations

import math
import queue
import time
from types import SimpleNamespace

import pytest

from cosim.interp import Timed, catmull_rom, ease_profile


def test_ease_profile_runs_from_0_to_1_forwards():
    for m0, m1 in ((1, 1), (0, 2), (2, 0), (5, 5), (0, 0)):
        xs = [ease_profile(m0, m1, i / 50) for i in range(51)]
        assert xs[0] == 0 and abs(xs[-1] - 1) < 1e-9
        assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:], strict=False)), (m0, m1)


def test_timed_keeps_speed_continuous_across_steps():
    """A car accelerating along a line, one SUMO position per step: the old curve's speed jumped at every
    step; the timed one changes smoothly and still passes through every SUMO position."""
    xs = [0.5 * 2.0 * (k * 0.5) ** 2 for k in range(12)]  # 2 m/s2 from standing, positions every 0.5 s
    pts = [(x, 0.0) for x in xs]
    sub = 30

    def speeds(fn):
        out = []
        for k in range(1, len(pts) - 2):
            for i in range(sub):
                out.append(fn(k, i / sub))
        v = [(b - a) * sub / 0.5 for a, b in zip(out, out[1:], strict=False)]
        return max(abs(b - a) for a, b in zip(v, v[1:], strict=False))

    old = speeds(lambda k, s: catmull_rom(pts[k - 1], pts[k], pts[k + 1], pts[k + 2], s)[0])
    new = speeds(lambda k, s: Timed(pts[k - 1], pts[k], pts[k + 1], pts[k + 2]).at(s)[0])
    assert new < 0.3 * old, (old, new)
    seg = Timed(pts[2], pts[3], pts[4], pts[5])
    assert seg.at(0.0) == pytest.approx(pts[3]) and seg.at(1.0) == pytest.approx(pts[4])


class _FakeFF:
    def __init__(self):
        self.sumo = SimpleNamespace(step_length=0.5)
        self.applied: list[int] = []
        self.moves: list[tuple[int, float]] = []

    def apply(self, pk):
        self.applied.append(pk)

    def move(self, pk, s):
        self.moves.append((pk, s))
        return {"car": SimpleNamespace(rotation=SimpleNamespace(pitch=0.0, yaw=10.0 * s))}


def test_mover_sends_poses_at_an_even_rate_and_follows_the_steps():
    from cosim.freefly import Mover

    ff = _FakeFF()
    worker = SimpleNamespace(q=queue.Queue())
    for k in range(1, 4):
        worker.q.put(k)
    clock = SimpleNamespace(paused=False)
    m = Mover(ff, worker, clock, rate=60.0, pk=0, demo=True)
    m.start()
    time.sleep(1.2)  # 1.2 s: steps 1 and 2 become due; 3 is there too
    m.stop = True
    m.join(timeout=1.0)
    n = len(ff.moves)
    assert 55 <= n <= 80, n  # ~60 a second
    assert ff.applied[:2] == [1, 2]
    assert [pk for pk, _ in ff.moves] == sorted(pk for pk, _ in ff.moves)  # steps in order
    assert all(0.0 <= s <= 1.0 for _, s in ff.moves)
    assert m.error is None
    gaps = sorted(m.gaps[5:])
    assert gaps[len(gaps) // 2] == pytest.approx(1 / 60, abs=0.006)


def test_mover_holds_the_last_pose_when_sumo_is_late():
    from cosim.freefly import Mover

    ff = _FakeFF()
    worker = SimpleNamespace(q=queue.Queue())  # SUMO never delivers
    m = Mover(ff, worker, SimpleNamespace(paused=False), rate=60.0, pk=0, demo=False)
    m.start()
    time.sleep(0.8)
    m.stop = True
    m.join(timeout=1.0)
    assert m.held > 0 and not ff.applied
    assert max(s for _, s in ff.moves) == 1.0  # held at the end of the step, never beyond


carla = pytest.importorskip("carla")


@pytest.fixture(scope="module")
def town05_side():
    from pathlib import Path

    from cosim import scenario as scenario_mod
    from cosim.carla_side import CarlaSide

    root = Path(__file__).resolve().parents[1]
    scen = scenario_mod.load("town05", root)
    side = CarlaSide.__new__(CarlaSide)  # the map only: no server
    side.map = carla.Map("Town05", (scen.folder / "town05.xodr").read_text(encoding="utf-8"))
    side.frame = scen.frame
    side._surface_z = lambda cx, cy, z, guess=0.0: z
    return side


def test_a_point_at_a_junction_kerb_goes_onto_the_pavement(town05_side):
    side = town05_side
    cmap = side.map
    sw = cmap.get_waypoint(carla.Location(0.0, 0.0, 0.0), project_to_road=True, lane_type=carla.LaneType.Sidewalk)
    r = sw.transform.get_right_vector()
    out = sw.lane_width / 2 + 1.0  # 1 m beyond the sidewalk's edge, on the road side or the other
    for sign in (1.0, -1.0):
        x, y = sw.transform.location.x + sign * out * r.x, sw.transform.location.y + sign * out * r.y
        nx, ny = side._onto_pavement(x, y)
        c = sw.transform.location
        lateral = abs((nx - c.x) * r.x + (ny - c.y) * r.y)
        assert lateral <= sw.lane_width / 2 - side.PAVEMENT_MARGIN_M + 1e-6


def test_the_move_onto_the_pavement_glides(town05_side):
    side = town05_side
    side._pave_off = {}
    sx, sy = side.frame.to_sumo_xy(0.0, 0.0)
    a = side.walker_pose(sx, sy, 0.0, ":X_c0", pid="p")       # on a crossing: not moved
    b = side.walker_pose(sx, sy, 0.0, ":X_w0", pid="p")       # off it: moved, but at most the glide
    assert math.hypot(b.location.x - a.location.x, b.location.y - a.location.y) <= side.PAVEMENT_GLIDE_M + 1e-6
    side.forget_walkers(set())
    assert not side._pave_off


def test_a_walkers_ground_height_never_waits_for_the_server():
    """CarlaSide._surface_z answers at once (the lane plus the kerb guess) and asks the server from a thread
    of its own; the answer is used once it has come."""
    import threading
    import time as _t

    from cosim.carla_side import CarlaSide

    release = threading.Event()

    class _World:
        def ground_projection(self, loc, dist):
            release.wait(5)  # a slow server
            return SimpleNamespace(label=SimpleNamespace(name="Sidewalks"), location=SimpleNamespace(z=0.17))

    side = CarlaSide.__new__(CarlaSide)
    side.world = _World()
    t0 = _t.perf_counter()
    assert side._surface_z(1.0, 2.0, 0.0, guess=0.15) == pytest.approx(0.15)
    assert _t.perf_counter() - t0 < 0.05  # did not wait
    release.set()
    for _ in range(100):
        if side._surface_z(1.0, 2.0, 0.0, guess=0.15) == pytest.approx(0.17):
            break
        _t.sleep(0.01)
    assert side._surface_z(1.0, 2.0, 0.0, guess=0.15) == pytest.approx(0.17)


def test_junction_views_look_down_an_approach_at_the_junction():
    """FreeFly.approach_views: above an approach road, ~70 m back and 28 m up, looking at the junction."""
    from pathlib import Path

    import sumolib

    from cosim import scenario as scenario_mod
    from cosim.freefly import FreeFly

    root = Path(__file__).resolve().parents[1]
    scen = scenario_mod.load("town05", root)
    ff = FreeFly.__new__(FreeFly)
    ff.scen = scen
    ff.centres = scenario_mod.tls_centres(scen.net_file, scen.frame)
    ff.zs = {}
    net = sumolib.net.readNet(str(scen.net_file))
    tid = next(iter(ff.centres))
    views = ff.approach_views(net, tid)
    assert len(views) >= 3
    cx, cy, _ = ff.centres[tid]
    for tf, target in views:
        loc = tf.location
        assert loc.z == pytest.approx(ff.VIEW_UP_M)
        assert 55.0 <= math.hypot(loc.x - cx, loc.y - cy) <= 95.0
        yaw_to = math.degrees(math.atan2(cy - loc.y, cx - loc.x))
        assert abs((tf.rotation.yaw - yaw_to + 180) % 360 - 180) < 1.0 and -30 < tf.rotation.pitch < -12
    ff.views = {tid: views[0][0]}
    assert ff.junction_pose(tid) is views[0][0]

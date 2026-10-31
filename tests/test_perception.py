"""Counting from cameras (cosim/perception.py) and max-pressure on given counts."""

from pathlib import Path

import numpy as np

from cosim.perception import CamSpec, LaneCamera, build_masks, count_lanes, ground_point_lane, plan_cameras, project

ROOT = Path(__file__).resolve().parents[1]


def test_projection_pinhole():
    cam = CamSpec("t", "n", "o", 0.0, 0.0, 10.0, yaw=0.0, pitch=0.0)
    px, depth = project(np.array([[20.0, 0.0, 10.0], [20.0, 5.0, 10.0], [20.0, 0.0, 15.0]]), cam, 800, 400, 90.0)
    assert np.allclose(depth, 20.0)
    assert np.allclose(px[0], [400, 200])  # straight ahead: image centre
    assert np.allclose(px[1], [400 + 400 * 5 / 20, 200])  # to the right: +u (focal 400 at fov 90)
    assert np.allclose(px[2], [400, 200 - 400 * 5 / 20])  # above: -v
    # yaw 90 faces +y (Unreal), pitch -90 looks straight down
    down = CamSpec("t", "n", "o", 0.0, 0.0, 10.0, yaw=90.0, pitch=-90.0)
    px, depth = project(np.array([[0.0, 0.0, 0.0]]), down, 800, 400, 90.0)
    assert np.allclose(depth, 10.0) and np.allclose(px[0], [400, 200])


def _quad(i, x0, x1, y0, y1):
    return i, np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])


def test_masks_give_each_lane_to_one_camera():
    # two cameras 9 m up at x = 0 and x = 100, looking at each other along x; two lanes along x
    a = LaneCamera(CamSpec("t", "a", "b", 0.0, 0.0, 9.0, yaw=0.0, pitch=-14.0), 320, 180, 75.0)
    b = LaneCamera(CamSpec("t", "b", "a", 100.0, 0.0, 9.0, yaw=180.0, pitch=-14.0), 320, 180, 75.0)
    lanes = ["near_a", "near_b"]
    quads = [_quad(0, x, x + 2, -1.6, 1.6) for x in range(5, 40, 2)] + \
            [_quad(1, x, x + 2, -1.6, 1.6) for x in range(60, 95, 2)]
    owner = build_masks([a, b], lanes, quads)
    assert owner == {"near_a": 0, "near_b": 1}
    assert set(np.unique(a.mask)) == {0, 1} and set(np.unique(b.mask)) == {0, 2}
    # a car on near_a, 20 m ahead of camera a: its box bottom lies on the lane
    (u, v), = project(np.array([[20.0, 0.0, 0.0]]), a.spec, 320, 180, 75.0)[0]
    box = [u - 15, v - 20, u + 15, v]
    assert ground_point_lane(a.mask, box) == 1
    assert count_lanes(a, [box, box], ["car", "pedestrian"]) == {"near_a": 1}


def test_town05_ring_road_has_a_camera_each_way():
    """Both carriageways of Town05's ring road lead from 2014 to 2328 (it is a loop): one camera each way, three at
    each of those junctions, 55 in all (27 Oct; before, one of them had none)."""
    import collections

    import sumolib

    from cosim.mapping import Frame

    net = sumolib.net.readNet(str(ROOT / "scenarios" / "town05" / "town05.net.xml"))
    cams = plan_cameras(net, Frame(0.0, 0.0), lambda x, y: 0.0)
    per = collections.Counter(c.tls for c in cams)
    assert len(cams) == 55 and per["2014"] == 3 and per["2328"] == 3

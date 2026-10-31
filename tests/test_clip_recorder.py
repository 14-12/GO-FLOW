"""cosim/clip_recorder.py (30 Oct): which camera sees a point, the labels' clock, the clip cut."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cosim.clip_recorder import near_view, sees, slug, video_s_at
from cosim.perception import CamSpec


def _cam(yaw=0.0):
    # 9 m up, pitch -22, looking along +x (yaw 0)
    return CamSpec(tls="J", node="J", road_to="K", x=0.0, y=0.0, z=9.0, yaw=yaw, pitch=-22.0)


def test_a_point_ahead_on_the_road_is_seen_one_behind_or_far_is_not():
    cam = _cam()
    ok, d = sees(cam, 960, 540, 75.0, 30.0, 0.0)
    assert ok and 25.0 < d < 35.0
    assert not sees(cam, 960, 540, 75.0, -20.0, 0.0)[0]       # behind
    assert not sees(cam, 960, 540, 75.0, 150.0, 0.0)[0]       # beyond 90 m
    assert not sees(cam, 960, 540, 75.0, 20.0, 40.0)[0]       # far to the side, out of the picture
    # the same point for a camera turned the other way
    assert sees(_cam(180.0), 960, 540, 75.0, -30.0, 0.0)[0]


def test_near_view_is_wider_than_the_picture():
    cam = _cam()
    assert near_view(cam, 75.0, 50.0, 0.0)
    ang = math.radians(75.0 / 2 + 10.0)  # just outside the picture, inside the margin
    assert near_view(cam, 75.0, 60 * math.cos(ang), 60 * math.sin(ang))
    assert not near_view(cam, 75.0, -60.0, 0.0)
    assert not near_view(cam, 75.0, 300.0, 0.0)


def test_labels_follow_the_simulation_clock_not_the_frame_count():
    # a frame dropped between video 0.2 s and 0.3 s: sim 0.2 -> 0.4
    rows = [(0, 0.0, 100.0), (1, 0.1, 100.1), (2, 0.2, 100.2), (3, 0.3, 100.4), (4, 0.4, 100.5)]
    assert video_s_at(rows, 100.3) == 0.3
    assert video_s_at(rows, 99.0) == 0.0
    assert video_s_at(rows, 200.0) == 0.4


def test_slug():
    assert slug("1427-1 looking N") == "1427-1_looking_N"


def test_cut_makes_a_clip_of_the_asked_length(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from cosim.clip_recorder import cut

    src = tmp_path / "rec.mp4"
    w = imageio_ffmpeg.write_frames(str(src), (64, 48), fps=10, codec="libx264", pix_fmt_in="rgb24",
                                    macro_block_size=8)
    w.send(None)
    for k in range(60):  # 6 s
        w.send(np.full((48, 64, 3), k * 4, dtype=np.uint8).tobytes())
    w.close()
    dst = tmp_path / "clip.mp4"
    cut(src, dst, 2.0, 3.0)
    n, _secs = imageio_ffmpeg.count_frames_and_secs(str(dst))
    assert 28 <= n <= 32

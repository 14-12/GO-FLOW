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


class _World:
    def __init__(self):
        self.cb, self.removed = None, False

    def on_tick(self, cb):
        self.cb = cb
        return 7

    def remove_on_tick(self, i):
        self.removed = i == 7


class _Lay:
    def __init__(self):
        cam = type("C", (), {"width": 960, "height": 540, "fov": 75.0})()
        self.cams, self.names = [cam], ["A"]


def test_a_picture_is_labelled_with_the_sumo_time_it_shows_not_sumos_latest_step(tmp_path, monkeypatch):
    """6 Nov: frames.csv sim_s was SUMO's own time when the picture arrived (sim_now), about 2 s ahead of what the
    picture showed. Now it is depicted_at(the picture's time on the wall clock); sim_now stays as sumo_s."""
    import cosim.clip_recorder as cr

    w = _World()
    rec = cr.ClipRecorder(w, _Lay(), tmp_path, lambda: 102.0, depicted=lambda wall: 100.0 + (wall - 1000.0))
    # CARLA's clock 50.0 s showed up at wall 1000.3, 50.1 at 1000.35: the lowest gap is the offset (950.25)
    snap = type("S", (), {})()
    for wall, el in ((1000.3, 50.0), (1000.35, 50.1)):
        monkeypatch.setattr(cr.time, "time", lambda wall=wall: wall)
        snap.timestamp = type("T", (), {"elapsed_seconds": el})()
        w.cb(snap)
    assert rec.offset == pytest.approx(950.25)
    # a picture of CARLA time 50.5, arriving late at wall 1001.9: it was drawn at wall 1000.75, which shows 100.75
    assert rec.picture_sim(50.5, 1001.9, 102.0) == pytest.approx(100.75)
    # depicted_at knows nothing yet: SUMO's own time
    rec.depicted = lambda wall: None
    assert rec.picture_sim(50.5, 1001.9, 102.0) == 102.0
    rec.close()
    assert w.removed


def test_without_depicted_the_label_is_sumos_time(tmp_path):
    import cosim.clip_recorder as cr

    w = _World()
    rec = cr.ClipRecorder(w, _Lay(), tmp_path, lambda: 5.0)
    assert w.cb is None and rec.picture_sim(1.0, 2.0, 5.0) == 5.0

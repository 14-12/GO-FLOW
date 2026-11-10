"""The full-screen traffic camera feed (cosim/cctv.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from cosim.cctv import CCTV, KEY_SAFE, split_name  # noqa: E402


def test_split_name():
    assert split_name("1126-2 looking north") == ("1126-2", "1126", "north")


@pytest.mark.parametrize("with_picture", [True, False])
def test_frame_fills_the_screen_and_never_shows_the_see_through_colour(with_picture):
    pygame.init()
    rgb = np.zeros((54, 96, 3), np.uint8)
    rgb[:, :] = (1, 0, 1)  # the overlay's colour key everywhere: would be a hole in the feed
    view = (rgb, {"vehicles": 3, "pedestrians": 1}, 10.0, "53-1 looking east") if with_picture else None
    surf = CCTV().frame((640, 360), view, "53-1 looking east", "detector", "MON  ·  DAY 1  ·  08:00", now=0.1)
    assert surf.get_size() == (640, 360)
    px = pygame.surfarray.array3d(surf)
    assert not ((px[:, :, 0] == 1) & (px[:, :, 1] == 0) & (px[:, :, 2] == 1)).any()
    assert tuple(CCTV.grade(rgb)[0, 0]) != (1, 0, 1)
    assert KEY_SAFE != (1, 0, 1)


def test_rec_dot_blinks():
    pygame.init()
    c = CCTV()
    on = pygame.surfarray.array3d(c.frame((640, 360), None, "53-1 looking east", "", "", now=0.1))
    off = pygame.surfarray.array3d(c.frame((640, 360), None, "53-1 looking east", "", "", now=0.9))
    red_on = ((on[:, :, 0] > 200) & (on[:, :, 1] < 80)).sum()
    red_off = ((off[:, :, 0] > 200) & (off[:, :, 1] < 80)).sum()
    assert red_on > red_off + 50


def test_native_overlay_is_see_through_except_its_labels_and_boxes():
    pygame.init()
    key = (1, 0, 1)
    screen = pygame.Surface((960, 540))
    screen.fill(key)
    boxes = [(100.0, 200.0, 180.0, 260.0, "car", 0.9), (400.0, 220.0, 520.0, 300.0, "ambulance_lights", 0.7)]
    view = (None, {"vehicles": 1, "emergency, SIREN ON": 1}, 9.0, "53-1 looking east")
    rects = CCTV().draw_native(screen, (960, 540), view, "53-1 looking east", "detector", "MON", boxes, now=0.1)
    px = pygame.surfarray.array3d(screen)
    see_through = (px[:, :, 0] == 1) & (px[:, :, 1] == 0) & (px[:, :, 2] == 1)
    assert see_through.mean() > 0.6  # CARLA's own picture shows through most of the window
    assert not see_through[450, 221] and see_through[450, 260]  # on the box outline / inside the box
    drawn = ~see_through
    covered = np.zeros_like(drawn)
    for r in rects:  # every pixel drawn lies in a rectangle sent to the window
        r = r.clip(screen.get_rect())
        covered[r.left:r.right, r.top:r.bottom] = True
    assert not (drawn & ~covered).any()


def test_boxes_are_scaled_to_the_window_and_carried_forward():
    from cosim.inspector import CARRY_MAX_S, TrafficCameras

    cams = TrafficCameras.__new__(TrafficCameras)
    cams.det = [(10.0, [(100.0, 100.0, 140.0, 120.0, "car", 0.9)], (960, 540)),
                (10.1, [(110.0, 100.0, 150.0, 120.0, "car", 0.9)], (960, 540))]  # 100 px/s to the right
    (x0, y0, x1, y1, cls, _), = cams.boxes_now(10.2, (1920, 1080))
    assert cls == "car"
    assert x0 == pytest.approx((110 + 100 * 0.1) * 2) and y0 == pytest.approx(200)
    (x0, *_), = cams.boxes_now(99.0, (1920, 1080))  # long after: carried at most CARRY_MAX_S
    assert x0 == pytest.approx((110 + 100 * CARRY_MAX_S) * 2)


@pytest.mark.parametrize("mode", ["plain", "panel", "sensor", "native"])
def test_compose_overlay_in_every_mode(mode):
    """The overlay as the viewer composes it (cosim.freefly.compose_overlay): the clock, the key help, a
    prompt, the sensor feed's picture or the native feed's labels. The first native version crashed here."""
    from cosim.freefly import compose_overlay

    pygame.init()
    key = (1, 0, 1)
    size = (640, 360)
    screen = pygame.Surface(size)
    screen.fill(key)
    c = CCTV()
    view = (None, {"vehicles": 2}, 8.0, "53-1 looking east")
    small = pygame.Surface((200, 100))
    kw = {"plain": {}, "panel": {"panel": small},
          "sensor": {"panel": c.frame(size, None, "53-1 looking east", "detector", "MON")},
          "native": {"native": lambda scr: c.draw_native(scr, size, view, "53-1 looking east", "d", "MON", [])}}[mode]
    hud = (lambda scr: pygame.draw.rect(scr, (20, 20, 30), (0, 0, 300, 120)), 300, 120)
    rects = compose_overlay(screen, size, [pygame.Rect(0, 0, 10, 10)], False, key, hud=hud,
                            keys_pill=pygame.Surface((100, 20)), prompt=pygame.Surface((120, 30)), **kw)
    assert rects
    px = pygame.surfarray.array3d(screen)
    see = (px[:, :, 0] == 1) & (px[:, :, 1] == 0) & (px[:, :, 2] == 1)
    if mode == "sensor":
        assert not see.any()  # its picture covers the window
    elif mode == "native":
        assert see.mean() > 0.5 and see[5, 5]  # CARLA shows through; the clock card is not drawn
    else:
        assert not see[5, 5]  # the clock card


def test_native_camera_waits_for_the_view_to_arrive():
    import time as _t
    from types import SimpleNamespace as NS

    from cosim.inspector import ARRIVE_S, TrafficCameras

    cams = TrafficCameras.__new__(TrafficCameras)
    cams.native, cams.active = True, 0
    cams.cams = [NS(x=10.0, y=20.0, z=9.0, yaw=90.0, pitch=-22.0)]
    cams.pose = (10.0, 20.0, 9.0)
    cams._at_pose, cams._opened_t = False, _t.time()

    def tf(x, yaw=90.0):
        return NS(location=NS(x=x, y=20.0, z=9.0), rotation=NS(yaw=yaw, pitch=-22.0))

    assert not cams.flown_away(tf(500.0))  # still the old place, read before CARLA moved the view
    assert not cams.flown_away(tf(10.0))   # arrived
    assert cams.flown_away(tf(13.0))       # then flown away
    assert cams.flown_away(tf(10.0, yaw=100.0))  # or turned
    cams._at_pose, cams._opened_t = False, _t.time() - ARRIVE_S - 1
    assert cams.flown_away(tf(500.0))      # never arrived: something else moved it


def test_detector_process_runs_its_own_sensor_for_the_native_feed(monkeypatch):
    """cosim.inspector.detector_process with stand-ins for YOLO and CARLA: ("open", ...) puts its own sensor at
    the camera, each picture it gets is detected there, and only the boxes come back (with the picture's time)."""
    import queue
    import sys
    import threading
    import time as _t
    import types

    from cosim import inspector

    class _Res:
        names = {0: "car"}
        boxes = types.SimpleNamespace(xyxy=_Arr([[1.0, 2.0, 3.0, 4.0]]), cls=_Arr([0]), conf=_Arr([0.9]))

    class _YOLO:
        def __init__(self, path):
            pass

        def predict(self, img, **kw):
            assert img.flags["C_CONTIGUOUS"]
            return [_Res()]

    spawned = {}

    class _Sensor:
        def listen(self, cb):
            spawned["cb"] = cb

        def stop(self):
            spawned["stopped"] = True

        def destroy(self):
            pass

    class _BP:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

    bp = _BP()
    world = types.SimpleNamespace(get_blueprint_library=lambda: types.SimpleNamespace(find=lambda name: bp),
                                  spawn_actor=lambda b, tf: (spawned.update(tf=tf), _Sensor())[1])
    fake_carla = types.SimpleNamespace(Client=lambda h, p: types.SimpleNamespace(set_timeout=lambda s: None,
                                                                                    get_world=lambda: world),
                                       Transform=lambda loc, rot: (loc, rot), Location=lambda *a: a,
                                       Rotation=lambda *a: a)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_YOLO))
    monkeypatch.setitem(sys.modules, "carla", fake_carla)
    q_in, q_out = queue.Queue(), queue.Queue()
    th = threading.Thread(target=inspector.detector_process, args=(q_in, q_out, "m.pt", "h", 1), daemon=True)
    th.start()
    assert q_out.get(timeout=5)[0] == "ready"
    q_in.put(("open", (1.0, 2.0, 9.0, -22.0, 90.0), (96, 54), 90.0, 0.1))
    for _ in range(100):
        if "cb" in spawned:
            break
        _t.sleep(0.01)
    assert bp.attrs == {"image_size_x": "96", "image_size_y": "54", "fov": "90.0", "sensor_tick": "0.1"}
    img = types.SimpleNamespace(raw_data=bytes(96 * 54 * 4), width=96, height=54)
    t0 = _t.time()
    spawned["cb"](img)
    kind, boxes, dt, t_pic, size = q_out.get(timeout=5)
    # (10 Nov) each vehicle box also carries its light numbers, for the siren-light model (cosim/siren_flash.py)
    assert kind == "ndet" and [b[:6] for b in boxes] == [(1.0, 2.0, 3.0, 4.0, "car", pytest.approx(0.9))]
    assert len(boxes[0][6]) == 4 and size == (96, 54)
    assert t0 - 0.01 <= t_pic <= _t.time()
    q_in.put(("close",))
    q_in.put(None)
    th.join(timeout=5)
    assert spawned.get("stopped") and not th.is_alive()


class _Arr(list):
    def cpu(self):
        return self

    def numpy(self):
        return self

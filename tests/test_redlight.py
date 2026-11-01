"""The red-light camera (31 Oct): the controller's log, the camera's geometry, the agent's rule on made-up tracks, the
plate reader on the plate design itself, the register, the police office. No CARLA, no detector model: the whole
chain on recorded video is graded by tools/eval_redlight.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from cosim import enforcement as enf
from cosim import red_runners
from cosim import redlight
from cosim import register as reg_mod

ROOT = Path(__file__).resolve().parents[1]


def _site() -> enf.Site:
    # a two-lane road going east (CARLA +x) into a junction; stop line at x = 0, lanes 3.5 m wide, y from 0 (right)
    # to -7 (left: CARLA's y is south, so left of an eastbound road is negative y)
    s = enf.Site(name="J-r", tls="J", edge="r", lanes=["r_0", "r_1"], links={"r_0": [0, 1], "r_1": [2, 3]},
                 line=[[0.0, 0.0], [0.0, -7.0]], lane_lines={}, heading=0.0, lane_width=3.5)
    s.cam = {"x": -22.0, "y": -3.5, "z": 6.5, "yaw": 0.0, "pitch": -14.0, "fov": 30.0, "width": 1920, "height": 1080,
             "ground_z": 0.0}
    return s


def test_the_signal_log_holds_a_steps_state_from_the_step_before():
    log = enf.SignalLog.from_changes([(30.5, "yyyy"), (32.0, "rrrr"), (109.5, "GGGG")], until=112)
    assert log.state_at(31.4) == "yyyy" and log.state_at(31.6) == "rrrr"   # red from 31.5
    assert log.red_since(40.0, [0, 1]) == pytest.approx(8.5)
    assert log.red_since(31.8, [0]) == pytest.approx(0.3)
    assert log.red_since(110.0, [0]) is None                              # green again
    assert log.changes([0, 2]) == [(30.5, "yy"), (32.0, "rr"), (109.5, "GG")]


def test_a_pixel_goes_back_to_the_road_where_it_came_from():
    cam = _site().cam
    pts = np.array([[-2.0, -1.0, 0.0], [5.0, -5.5, 0.0], [20.0, -3.0, 0.0]])
    px, depth = enf.project(pts, cam)
    assert (depth > 0).all() and (px[:, 0] > 0).all() and (px[:, 0] < 1920).all()
    back = enf.to_ground(px, cam, 0.0)
    assert np.allclose(back, pts, atol=1e-6)


def test_the_road_measures_along_and_across_from_the_stop_line():
    road = redlight.Road(_site())
    s, d = road.sd(-4.0, -5.0)
    assert s == pytest.approx(-4.0) and d == pytest.approx(5.0)
    assert road.lane(1.0) == "r_0" and road.lane(5.0) == "r_1" and road.lane(8.0) is None


class _Boxes:
    """A made-up detector: one car driving east along lane r_0 at `v` m/s, its back at s(t) = s0 + v t; its box is
    where the camera would see it (the back's bottom edge at the right pixel)."""

    def __init__(self, cam, s0: float, v: float, cls: int = 0, stop_at: float | None = None):
        self.cam, self.s0, self.v, self.cls, self.stop_at = cam, s0, v, cls, stop_at

    def at(self, t: float):
        s = self.s0 + self.v * t
        if self.stop_at is not None:
            s = min(s, self.stop_at)
        (px,), _ = enf.project([[s, -1.75, 0.0]], self.cam)
        x, y = px
        h = 4000.0 / max(8.0, s + 22.0)   # roughly as big as a car looks from the pole 22 m before the line
        return [redlight.Det((x - 0.9 * h, y - h, x + 0.9 * h, y), self.cls, 0.9)]


def _drive(site, log, boxes, t0=0.0, t1=12.0, fps=10.0):
    agent = redlight.RedLightAgent(site, log, detector=None)
    cases = []
    k = 0
    t = t0
    while t <= t1 + 1e-9:
        cases += agent.frame(k, t, None, dets=boxes.at(t))
        k += 1
        t += 1.0 / fps
    for tr in agent.tracker.tracks:
        tr.done = True
    cases += agent.frame(k, t + 5.0, None, dets=[])
    return agent, cases


def test_a_car_entering_two_seconds_into_red_is_a_case():
    site = _site()
    log = enf.SignalLog.from_changes([(0.0, "GGGG"), (2.0, "yyyy"), (5.5, "rrrr")], until=30)  # red from 5.0
    # the back at -16 m at t=0, 3 m/s: the front (4.5 m on) crosses the line at t = 11.5/3 = 3.83 s -> amber: no case
    agent, cases = _drive(site, log, _Boxes(site.cam, -16.0, 3.0))
    assert not cases and agent.crossings and agent.crossings[0]["red_s"] is None
    # 2 m/s: crosses at 5.75 s, 0.75 s into red: a case, with its photos before and after
    agent, cases = _drive(site, log, _Boxes(site.cam, -16.0, 2.0), t1=14.0)
    assert len(cases) == 1
    c = cases[0]
    assert c.t_cross == pytest.approx(5.75, abs=0.15) and c.red_s == pytest.approx(0.75, abs=0.15)
    assert c.lane == "r_0" and c.frame_a < c.frame_b and c.speed_kmh == pytest.approx(7.2, abs=1.0)


def test_within_the_grace_time_or_stopping_over_the_line_is_no_case():
    site = _site()
    log = enf.SignalLog.from_changes([(0.0, "GGGG"), (2.0, "yyyy"), (5.5, "rrrr")], until=30)
    # crosses 0.25 s into red: within the tolerance
    _, cases = _drive(site, log, _Boxes(site.cam, -16.0 + 1.0, 2.0))   # 1 m on at 2 m/s: 0.5 s earlier
    assert not cases
    # crosses on red but stops with its back 1 m past the line: not confirmed
    agent, cases = _drive(site, log, _Boxes(site.cam, -16.0, 2.0, stop_at=-3.0), t1=16.0)
    assert not cases and agent.crossings[0]["outcome"].startswith("stopped")


def test_an_emergency_vehicle_with_its_lights_on_is_exempt():
    site = _site()
    log = enf.SignalLog.from_changes([(0.0, "GGGG"), (2.0, "yyyy"), (5.5, "rrrr")], until=30)
    # an ambulance is 5.8 m long: its back at -17.3 m puts its front over the line at 5.75 s, 0.75 s into red
    agent, cases = _drive(site, log, _Boxes(site.cam, -17.3, 2.0, cls=5), t1=14.0)   # ambulance_lights
    assert not cases and agent.crossings[0]["outcome"].startswith("exempt")


def test_the_plate_reader_reads_the_plate_design_at_camera_sizes():
    import cv2

    from cosim import anpr, plates

    rng = random.Random(4)
    right = wrong_sure = 0
    n = 60
    for _ in range(n):
        p = plates.random_plate(rng)
        img = np.asarray(plates.render(p).convert("RGB"))
        small = cv2.resize(img, (115, 24), interpolation=cv2.INTER_AREA)
        r = anpr.read_plate(cv2.resize(small, (anpr.PLATE_W, anpr.PLATE_H), interpolation=cv2.INTER_CUBIC))
        right += r.text == p.text
        wrong_sure += r.text is not None and r.text != p.text and r.conf >= 0.3
    assert right >= 0.85 * n
    assert wrong_sure <= 1   # a wrong reading is almost never a sure one


def test_the_register_is_fictional_and_stable():
    e = reg_mod.record_for("214 TU 751", "vehicle.audi.tt", "255,255,255")
    assert e == reg_mod.record_for("214 TU 751", "vehicle.audi.tt", "255,255,255")
    assert e["fictional"] and e["owner"]["id"].startswith("SIM-")
    assert e["vehicle"]["make"] == "Audi" and e["vehicle"]["model"] == "TT" and e["vehicle"]["colour"] == "white"
    assert reg_mod.record_for("07 - 49858", "vehicle.dodge.charger_police_2020", "0,0,0")["owner"]["kind"] == "state"


def test_gamblers_are_a_fixed_share_of_cars():
    n = sum(red_runners.is_gambler(f"live{k}", 0.1) for k in range(20000))
    assert 1800 < n < 2200
    assert red_runners.is_gambler("live7", 0.1) == red_runners.is_gambler("live7", 0.1)


def test_the_office_issues_a_notice_with_the_registered_keeper(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("police_office", ROOT / "tools" / "police_office.py")
    po = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(po)
    reg = reg_mod.Register(tmp_path / "register.json")
    reg.add("214 TU 751", "vehicle.audi.tt", "255,255,255")
    reg.save()
    d = tmp_path / "cases" / "J-r-0000057"
    d.mkdir(parents=True)
    (d / "photo_a.jpg").write_bytes(b"a")
    case = {"id": "J-r-0000057", "site": "J-r", "lane": "r_0", "red_s": 0.75, "speed_kmh": 7.2, "plate": "214 TU 751",
            "plate_conf": 0.6, "kind": "car", "plate_reads": []}
    (d / "case.json").write_text(json.dumps({"case": case, "when": "WED 17:31:05", "files": {
        "photo_a.jpg": po.sha256(d / "photo_a.jpg")}, "rules": redlight.RULES, "grace_s": 0.5}), encoding="utf-8")
    office = po.Office(tmp_path / "cases", tmp_path / "register.json")
    rec = office.all()[0]
    assert office.integrity(rec) == {"photo_a.jpg": True}
    dec = office.decide(rec, True, "214 TU 751", None, "test officer")
    assert dec["notice"].startswith("SIM-") and dec["register"]["vehicle"]["make"] == "Audi"
    text = (d / "notice.html").read_text(encoding="utf-8")
    assert "fictional" in text.lower() and "214 TU 751" in text
    (d / "photo_a.jpg").write_bytes(b"changed")
    assert office.integrity(office.all()[0]) == {"photo_a.jpg": False}


def test_the_plate_camera_finds_a_tracked_vehicles_plate_from_where_it_is_on_the_road():
    import math

    import cv2

    from cosim import plates

    site = _site()
    site.plate_cam = dict(site.cam, fov=enf.PLATE_FOV_DEG,
                          pitch=-math.degrees(math.atan2(6.5 - enf.PLATE_Z_M, 22.0 + enf.PLATE_AIM_S)))
    road = redlight.Road(site)
    x, y = road.xy(-5.0, 1.75)
    assert road.sd(x, y) == pytest.approx((-5.0, 1.75))
    # a car along lane r_0, its back from -8 m to -2 m over 2 s; the plate camera's picture at t = 1 s (back at -5 m)
    tr = redlight.Track(1, [redlight.Obs(k, k * 0.5, (0, 0, 1, 1), 0, 0.9, -8.0 + 1.5 * k, 1.75) for k in range(5)])
    p = plates.random_plate(random.Random(11))
    plate = np.asarray(plates.render(p).convert("RGB"))
    (c, l_, r_), _ = enf.project([[*road.xy(-5.0, 1.75), 0.5], [*road.xy(-5.0, 1.49), 0.5],
                                  [*road.xy(-5.0, 2.01), 0.5]], site.plate_cam)
    w = int(abs(r_[0] - l_[0]))
    small = cv2.resize(plate, (w, int(w * plate.shape[0] / plate.shape[1])), interpolation=cv2.INTER_AREA)
    img = np.full((1080, 1920, 3), (120, 30, 30), np.uint8)     # (a dark red car's back all round it)
    y0, x0 = int(c[1] - small.shape[0] / 2), int(c[0] - w / 2)
    img[y0:y0 + small.shape[0], x0:x0 + w] = small
    _, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    store = redlight.PlateStore()
    store.add(10, 1.0, buf.tobytes())
    reader = redlight.PlateReader(site, store)
    views = reader.plate_cam_views(tr)
    assert [k for k, *_ in views] == [10]
    text, _, _, order = reader.read(tr)
    assert text == p.text and order[0] == "p10" and reader.crop(tr, "p10") is not None

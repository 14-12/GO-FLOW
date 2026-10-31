"""The free-fly viewer's loading screen (cosim/splash.py): the viewer's status, the stages, the progress."""

from __future__ import annotations

import time

from cosim import splash


def test_status_written_by_the_viewer_is_read_only_if_newer(tmp_path):
    p = tmp_path / "loading.json"
    splash.write_status("Starting SUMO", 0.5, {"vehicles": 3}, "Tuesday 08:00", path=p)
    st = splash.read_status(time.time() - 5, path=p)
    assert st["stage"] == "Starting SUMO" and st["counts"] == {"vehicles": 3}
    assert splash.read_status(time.time() + 5, path=p) is None  # from an earlier run
    assert splash.read_status(0, path=tmp_path / "missing.json") is None


def test_stage_texts_map_to_the_longest_matching_stage():
    names = [n for _, n, *_ in splash.STAGES]
    assert names[splash.stage_index("Starting")] == "Opening the city"
    assert names[splash.stage_index("Starting SUMO")] == "Waking the traffic"
    assert names[splash.stage_index("Loading Town05")] == "Raising the skyline"
    assert names[splash.stage_index("Filling the town for 08:00: 69 vehicles")] == "Filling the streets"


def test_progress_only_grows_and_reaches_one_when_ready():
    t = 1000.0
    pr = splash.Progress(t)
    shown = []
    for k, stage in enumerate(["Starting", "Connecting to CARLA", "Loading Town05", "Starting SUMO",
                               "Filling the town for 08:00: 69 vehicles"]):
        pr.update({"stage": stage, "progress": -1.0}, t + 60 + k * 5)
        for i in range(30):
            shown.append(pr.step(t + 60 + k * 5 + i * 0.1, 0.1))
    assert all(b >= a for a, b in zip(shown, shown[1:], strict=False))
    assert 0.3 < shown[-1] < 0.9
    pr.update({"stage": "Ready", "ready": True}, t + 200)
    for i in range(100):
        v = pr.step(t + 200 + i * 0.1, 0.1)
    assert v > 0.99 and pr.cur == len(splash.STAGES) - 1


def test_the_screen_draws(tmp_path, monkeypatch):
    import os

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    s = splash.Splash(screen, "town05", 0.0)
    s.status = {"stage": "Filling the town", "progress": 0.4, "counts": {"vehicles": 12}, "clock": "Tuesday 08:00"}
    s.prog.update(s.status, 70.0)
    for k in range(40):
        s.draw(70.0 + k / 30, 1 / 30)
    assert s.revealed > 0 and s.cars
    lit = sum(1 for x in range(0, 540, 6) for y in range(0, 540, 6) if sum(screen.get_at((x, y))[:3]) > 120)
    assert lit > 30  # roads and cars are drawn on the map side
    pygame.quit()

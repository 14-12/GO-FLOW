"""The arrow keys' turning, in a process of its own (`--arrows proc`, the default).

Done in CARLA's tick callback with the place of the picture just made (cosim/flycam.ArrowTurn), the turn
does not pull the camera back while CARLA moves it for Z Q S D, provided the callback runs at once. In the
viewer's own process it waits for Python's lock behind the SUMO thread and the main loop (Python work
for every vehicle, 60 times a second); a late callback sends a place that is already old, and the camera
shook again (tools/probe_viewer_arrows.bat measures it in the running viewer). Here nothing else runs.
"""

from __future__ import annotations

import time


def run(host: str, port: int) -> None:
    import multiprocessing as mp

    import carla

    from cosim.flycam import ArrowTurn
    from cosim.overlay import Overlay

    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    turn = ArrowTurn(world.get_spectator())
    ov = Overlay(0)
    ov.carla = ov._find_carla()

    def tick(snap):
        try:
            turn.tick(snap, ov.held())
        except Exception:  # noqa: BLE001 - never stop CARLA's callback thread
            pass

    cb = world.on_tick(tick)
    parent = mp.parent_process()
    try:
        while parent is None or parent.is_alive():
            if not ov.carla or not ov.u.IsWindow(ov.carla):
                ov.carla = ov._find_carla()
            time.sleep(0.5)
    finally:
        try:
            world.remove_on_tick(cb)
        except Exception:  # noqa: BLE001
            pass

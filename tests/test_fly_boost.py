"""cosim/fly_proc.Fly: Shift held with Z Q S D flies the camera BOOST times faster (29 Oct)."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest


def _fly(shift: bool):
    pytest.importorskip("carla")
    from cosim import fly_proc

    keys = {fly_proc.MOVE_KEYS["Z"]: True, fly_proc.VK_SHIFT: shift}
    ov = NS(carla_in_front=lambda: True, held=lambda: set(),
            u=NS(GetAsyncKeyState=lambda vk: 0x8000 if keys.get(vk) else 0))
    tf = NS(location=NS(x=0.0, y=0.0, z=10.0), rotation=NS(yaw=0.0, pitch=0.0))
    sent = []
    sp = NS(id=1, set_transform=sent.append, set_location=sent.append)
    snap = NS(find=lambda i: NS(get_transform=lambda: tf), timestamp=NS(delta_seconds=0.05))
    fly = fly_proc.Fly(sp, ov)
    for _ in range(40):  # 2 s: well past the easing
        fly.tick(snap)
    return fly.v[0]


def test_shift_flies_faster():
    from cosim import fly_proc

    slow, fast = _fly(False), _fly(True)
    assert slow == pytest.approx(fly_proc.SPEED_MS, rel=0.01)
    assert fast == pytest.approx(fly_proc.SPEED_MS * fly_proc.BOOST, rel=0.01)

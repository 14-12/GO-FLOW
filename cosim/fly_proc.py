"""The free-fly camera flown by the viewer alone, in a process of its own (`--arrows fly`).

Z Q S D (forward where it looks, left, back, right), A / E (down, up), Shift held with them (BOOST times
faster) and the arrows (turn, look up / down) all go through here, and CARLA's own key movement is off
(CarlaUE4/Config/DefaultInput.ini), so one place
moves and turns the camera and nothing can pull it back. Measured in the running viewer
(tools/probe_viewer_arrows.bat): CARLA moving it and Python turning it shook (the place sent with a turn
is late); CARLA's own mouse look driven by fake mouse moves turned 23 deg of 225 asked, in jerks.

Once per CARLA picture (tick callback), at high priority, nothing else in this process: the pose for the
*next* picture is sent (the server applies a pose a picture after the one it was worked out from), from
this process's own place and turn, never read back. The right mouse button still turns the view with
CARLA's mouse look: then only the place is sent (set_location), and the turn followed.
"""

from __future__ import annotations

import math
import time

SPEED_MS = 20.0        # as CARLA's own spectator (measured)
BOOST = 2.0            # Shift held: this many times faster (40 m/s), eased in and out like the rest (29 Oct)
VK_SHIFT = 0x10        # either Shift key
EASE_S = 0.12          # speed eases in and out over this
TURN_DEG_S, PITCH_DEG_S = 90.0, 60.0
MOVE_KEYS = {"Z": 0x5A, "Q": 0x51, "S": 0x53, "D": 0x44, "A": 0x41, "E": 0x45}


class Fly:
    def __init__(self, sp, ov):
        self.sp, self.ov = sp, ov
        self.active = False
        self.pos = [0.0, 0.0, 0.0]
        self.yaw = self.pitch = 0.0
        self.v = [0.0, 0.0, 0.0]

    def keys(self) -> tuple[set, set, bool]:
        ov = self.ov
        if not ov.carla_in_front():
            return set(), set(), False
        move = {k for k, vk in MOVE_KEYS.items() if ov.u.GetAsyncKeyState(vk) & 0x8000}
        return move, ov.held(), bool(ov.u.GetAsyncKeyState(0x02) & 0x8000)

    def tick(self, snap) -> None:
        import carla

        move, arrows, mouse = self.keys()
        fast = bool(move) and self.ov.carla_in_front() and bool(self.ov.u.GetAsyncKeyState(VK_SHIFT) & 0x8000)
        a = snap.find(self.sp.id)
        if a is None:
            return
        tf = a.get_transform()
        if not self.active:
            if not move and not arrows:
                return
            self.pos = [tf.location.x, tf.location.y, tf.location.z]
            self.yaw, self.pitch = tf.rotation.yaw, tf.rotation.pitch
            self.v = [0.0, 0.0, 0.0]
            self.active = True
        dt = min(max(snap.timestamp.delta_seconds, 0.001), 0.1)
        if arrows:
            self.yaw += (("RIGHT" in arrows) - ("LEFT" in arrows)) * TURN_DEG_S * dt
            self.pitch = max(-89.0, min(89.0, self.pitch + (("UP" in arrows) - ("DOWN" in arrows)) * PITCH_DEG_S * dt))
        elif mouse:
            self.yaw, self.pitch = tf.rotation.yaw, tf.rotation.pitch
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp_ = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        fwd, right = (cy * cp, sy * cp, sp_), (-sy, cy, 0.0)
        f = ("Z" in move) - ("S" in move)
        r = ("D" in move) - ("Q" in move)
        u = ("E" in move) - ("A" in move)
        want = [f * fwd[i] + r * right[i] + (u if i == 2 else 0.0) for i in range(3)]
        n = math.sqrt(sum(w * w for w in want))
        speed = SPEED_MS * (BOOST if fast else 1.0)
        want = [speed * w / n for w in want] if n > 1e-6 else [0.0, 0.0, 0.0]
        k = min(1.0, dt / EASE_S)
        self.v = [v + (w - v) * k for v, w in zip(self.v, want, strict=True)]
        if not move and not arrows and all(abs(c) < 0.05 for c in self.v):
            self.active = False  # let go: whatever else moves the camera (F8, the mouse) is picked up next time
            return
        self.pos = [p + v * dt for p, v in zip(self.pos, self.v, strict=True)]
        nxt = carla.Location(*(p + v * dt for p, v in zip(self.pos, self.v, strict=True)))  # the next picture's
        if arrows or not mouse:
            self.sp.set_transform(carla.Transform(nxt, carla.Rotation(self.pitch, self.yaw, 0.0)))
        else:
            self.sp.set_location(nxt)


def run(host: str, port: int) -> None:
    import ctypes
    import multiprocessing as mp

    import carla

    from cosim.overlay import Overlay

    try:  # above the viewer and CARLA: the pose must reach the server before its next picture
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:  # noqa: BLE001
        pass
    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    ov = Overlay(0)
    ov.carla = ov._find_carla()
    fly = Fly(world.get_spectator(), ov)

    def tick(snap):
        try:
            fly.tick(snap)
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

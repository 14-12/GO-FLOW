"""The arrow keys turn the free-fly camera through CARLA's own mouse look (the default, `--arrows mouse`).

Why: CARLA moves its spectator itself for Z Q S D and turns it only with the mouse (right button held).
Every way of turning it from Python sends a whole transform, place included, and that place is always a
little old by the time CARLA applies it; while Z Q S D moves the camera, each turn pulled it back, and the
camera shook. That was so from the main loop, from CARLA's tick callback and from a process of its own,
because the server is busy with the traffic. Through the mouse, CARLA is the only one that moves and turns
the camera, so nothing can pull it back.

How: while an arrow is held (CARLA in front, right button not held by you), the right button is pressed for
CARLA and the turn is sent as relative mouse moves, 120 times a second from a thread of its own, in steps
of at most 2 px (bigger steps were sped up by Windows' pointer acceleration: 26.5 deg asked, up to 38 got).
Two fixes for presses that did nothing:
* the cursor is moved by a few pixels before each press, so two presses close together are never a
  double-click (Unreal does not start its mouse look on a double-click);
* moves wait CAPTURE_S after the press: Unreal takes the mouse a few frames after it and drops moves
  sent before; a tap shorter than that is sent once it has.
The button is let go when the arrows are, and when the viewer quits.
"""

from __future__ import annotations

import threading
import time

DEG_PER_PX = 26.5 / 360.0  # measured: 360 px of mouse move -> 26.5 deg of yaw, 26.0 of pitch
TURN_DEG_S, PITCH_DEG_S = 90.0, 60.0
CAPTURE_S = 0.08
STEP_PX = 2
_MOVE, _RDOWN, _RUP, _VK_RBUTTON = 0x0001, 0x0008, 0x0010, 0x02


class ArrowMouse(threading.Thread):
    def __init__(self, ov, rate_hz: float = 120.0):
        super().__init__(daemon=True)
        self.ov = ov
        self.dt = 1.0 / rate_hz
        self.stop = False
        self.down = False      # the right button is down (pressed by us or by you)
        self.injected = False  # ... and it was us who pressed it
        self.t_down = 0.0
        self.acc = [0.0, 0.0]
        self.nudge = 1
        self.presses = 0
        self.releases = 0
        self.max_gap_s = 0.0  # longest wait between two runs of the thread

    # -- the thread ---------------------------------------------------------------------------------
    def run(self) -> None:
        last = time.perf_counter()
        while not self.stop:
            now = time.perf_counter()
            self.max_gap_s = max(self.max_gap_s, now - last)
            dt = min(now - last, 0.05)
            last = now
            try:
                arrows = self.ov.held()
                if arrows:
                    self._turn((("RIGHT" in arrows) - ("LEFT" in arrows)) * TURN_DEG_S * dt,
                               (("UP" in arrows) - ("DOWN" in arrows)) * PITCH_DEG_S * dt)
                elif self.down:
                    self.release()
            except Exception:  # noqa: BLE001 - never leave the button held because of an error
                self.release()
            time.sleep(self.dt)
        self.release()

    # -- turning --------------------------------------------------------------------------------------
    def _turn(self, yaw_deg: float, pitch_deg: float) -> None:
        u = self.ov.u
        if not self.down:
            self.injected = not (u.GetAsyncKeyState(_VK_RBUTTON) & 0x8000)
            if self.injected:
                self._place_cursor()
                u.mouse_event(_RDOWN, 0, 0, 0, 0)
                self.presses += 1
            self.down = True
            self.t_down = time.perf_counter()
            self.acc = [0.0, 0.0]
        self.acc[0] += yaw_deg / DEG_PER_PX
        self.acc[1] -= pitch_deg / DEG_PER_PX  # mouse down = look down
        if self.injected and time.perf_counter() - self.t_down < CAPTURE_S:
            return  # Unreal has not taken the mouse yet: keep the turn
        self._send()

    def _send(self) -> None:
        u = self.ov.u
        ix, iy = int(self.acc[0]), int(self.acc[1])
        self.acc[0] -= ix
        self.acc[1] -= iy
        while ix or iy:
            sx = max(-STEP_PX, min(STEP_PX, ix))
            sy = max(-STEP_PX, min(STEP_PX, iy))
            u.mouse_event(_MOVE, sx, sy, 0, 0)
            ix -= sx
            iy -= sy

    def release(self) -> None:
        if not self.down:
            return
        u = self.ov.u
        if self.injected:
            wait = CAPTURE_S - (time.perf_counter() - self.t_down)
            if wait > 0 and (abs(self.acc[0]) >= 1 or abs(self.acc[1]) >= 1):
                time.sleep(wait)  # a tap: give Unreal the mouse first, then the turn
            self._send()
            u.mouse_event(_RUP, 0, 0, 0, 0)
        self.releases += 1
        self.down = False
        self.injected = False
        self.acc = [0.0, 0.0]

    def _place_cursor(self) -> None:
        """On CARLA's picture (the press goes to the window under the cursor), and a few pixels from where
        the last press was, so that two presses are never one double-click."""
        import ctypes
        from ctypes import wintypes

        u = self.ov.u
        rect = wintypes.RECT()
        u.GetWindowRect(wintypes.HWND(self.ov.carla), ctypes.byref(rect))
        pt = wintypes.POINT()
        u.GetCursorPos(ctypes.byref(pt))
        cx, cy = pt.x, pt.y
        if not (rect.left + 20 < cx < rect.right - 20 and rect.top + 20 < cy < rect.bottom - 20):
            cx, cy = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
        self.nudge = -self.nudge
        u.SetCursorPos(cx + 8 * self.nudge, cy)

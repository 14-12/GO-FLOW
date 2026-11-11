"""The free-fly clock drawn over the CARLA window, like a game's HUD (Windows).

CARLA's packaged server cannot be given a widget of our own: a HUD inside Unreal needs the editor
and a rebuild. So the clock is a borderless window laid exactly over the top-left corner of the
CARLA picture: everything but the card is see-through (since 11 Nov every pixel has its own transparency,
AlphaPresenter below; before, and as the fallback, one see-through colour key), mouse clicks go through it
to CARLA, it never takes the keyboard focus, it follows the CARLA window when that moves, and it
hides itself when another application is in front.

The window covers the whole CARLA picture (everything but what is drawn is see-through): the clock in
the top-left corner, the "H inspect" prompt and the camera inspector (cosim/inspector.py).

Its keys are read while the CARLA window has the focus (GetAsyncKeyState), so you keep flying in
CARLA: F5 pause, F6 / F7 one hour back / forward, F8 next junction, F9 hide or show the clock,
F10 the forecast's road most at risk (--goflow-prediction), H inspect the traffic camera you are near, Esc quit. Function keys and H, because CARLA's
spectator uses the other letters, the arrows, Space and C.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

KEY_COLOUR = (1, 0, 1)  # drawn nowhere in the clock: these pixels are see-through

VK = {"F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "ESC": 0x1B, "H": 0x48,
      "F": 0x46, "V": 0x56, "M": 0x4D,  # F follow a car, V its camera, M the mini-map (8 Nov, route_guidance.py)
      "L": 0x4C}  # L the places' labels (8 Nov, cosim/place_hud.py)

_GWL_EXSTYLE = -20
_WS_EX = 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000 | 0x00000008  # LAYERED TRANSPARENT TOOLWINDOW NOACTIVATE TOPMOST
_LWA_COLORKEY = 0x1
_HWND_TOPMOST = -1
_SWP = 0x0001 | 0x0010 | 0x0040  # NOSIZE NOACTIVATE SHOWWINDOW
_SW_HIDE = 0


def dpi_aware() -> None:
    """Per-monitor DPI awareness, so window positions are in the same pixels as CARLA's."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001 - older Windows
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass


class Overlay:
    def __init__(self, hwnd: int, margin: int = 0):
        self.ok = sys.platform == "win32"
        self.margin = margin
        self.hwnd = hwnd
        self.carla: int | None = None
        self.shown = None
        self._down: set[str] = set()
        if not self.ok:
            return
        u = ctypes.windll.user32
        self.u = u
        LONG_PTR = ctypes.c_ssize_t
        u.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetWindowLongPtrW.restype = LONG_PTR
        u.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
        u.SetWindowLongPtrW.restype = LONG_PTR
        u.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD]
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint]
        u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.IsIconic.argtypes = [wintypes.HWND]
        u.IsWindow.argtypes = [wintypes.HWND]
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = ctypes.c_short
        # The quick questions asked every loop (keys, the window in front, where it is), without letting go of
        # Python's lock: a ctypes call through windll lets go and then has to win the lock back from the
        # viewer's other threads, which with SUMO, the poses and the camera busy took ~1-5 ms a call (the
        # profile showed 15-20 % of the main loop in pressed(), 7 key checks).
        q = ctypes.PyDLL("user32")
        q.GetAsyncKeyState.argtypes = [ctypes.c_int]
        q.GetAsyncKeyState.restype = ctypes.c_short
        q.GetForegroundWindow.restype = wintypes.HWND
        q.IsIconic.argtypes = [wintypes.HWND]
        q.IsWindow.argtypes = [wintypes.HWND]
        q.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        q.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.q = q
        if hwnd:  # 0: keys only (cosim/arrow_proc.py has no window of its own)
            self.restyle(hwnd)

    def restyle(self, hwnd: int, alpha: bool = False) -> None:
        """See-through, click-through, never focused, on top (again after the window was resized).
        alpha: the picture comes from AlphaPresenter (every pixel its own transparency), not the colour key."""
        if not self.ok:
            return
        self.hwnd = hwnd
        u = self.u
        style = u.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE)
        if alpha:  # a window given a colour key cannot take UpdateLayeredWindow until the layered bit is set anew
            u.SetWindowLongPtrW(hwnd, _GWL_EXSTYLE, (style | _WS_EX) & ~0x00080000)
            u.SetWindowLongPtrW(hwnd, _GWL_EXSTYLE, style | _WS_EX)
        else:
            u.SetWindowLongPtrW(hwnd, _GWL_EXSTYLE, style | _WS_EX)
            r, g, b = KEY_COLOUR
            u.SetLayeredWindowAttributes(hwnd, r | (g << 8) | (b << 16), 0, _LWA_COLORKEY)
        self.shown = None

    def fullscreen_carla(self) -> bool:
        """Makes the CARLA window borderless and as large as its monitor (borderless full screen: the
        overlay still shows over it, which exclusive full screen would not allow). Unreal follows the
        window size with its picture. True once done."""
        if not self.ok:
            return False
        if not self.carla or not self.q.IsWindow(self.carla):
            self.carla = self._find_carla()
        if not self.carla:
            return False

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                        ("dwFlags", wintypes.DWORD)]

        u = self.u
        u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.MonitorFromWindow.restype = wintypes.HMONITOR
        u.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not u.GetMonitorInfoW(u.MonitorFromWindow(self.carla, 2), ctypes.byref(mi)):  # 2: nearest monitor
            return False
        style = u.GetWindowLongPtrW(self.carla, -16)  # GWL_STYLE
        frame = 0x00C00000 | 0x00040000 | 0x00020000 | 0x00010000 | 0x00080000  # caption, sizing frame, buttons, menu
        u.SetWindowLongPtrW(self.carla, -16, style & ~frame)
        r = mi.rcMonitor
        u.SetWindowPos(self.carla, wintypes.HWND(0), r.left, r.top, r.right - r.left, r.bottom - r.top,
                       0x0020 | 0x0040)  # FRAMECHANGED SHOWWINDOW
        return True

    def client_size(self) -> tuple[int, int] | None:
        """Size of the CARLA picture in pixels, once the CARLA window is found."""
        if not self.ok:
            return None
        if not self.carla or not self.q.IsWindow(self.carla):
            self.carla = self._find_carla()
        if not self.carla:
            return None
        rect = wintypes.RECT()
        self.q.GetClientRect(self.carla, ctypes.byref(rect))
        return (rect.right, rect.bottom) if rect.right > 100 and rect.bottom > 100 else None

    def _find_carla(self) -> int | None:
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(h, _):
            cls = ctypes.create_unicode_buffer(64)
            self.u.GetClassNameW(h, cls, 64)
            if cls.value == "UnrealWindow" and self.u.IsWindowVisible(h):
                found.append(h)
                return False
            return True

        self.u.EnumWindows(cb, 0)
        return found[0] if found else None

    def carla_in_front(self) -> bool:
        if not self.ok or not self.carla:
            return False
        fg = self.q.GetForegroundWindow()
        return fg in (self.carla, self.hwnd)

    def follow(self, visible: bool = True, need_front: bool = True) -> None:
        """Lays the clock over the CARLA picture's top-left corner, or hides it (CARLA not found,
        minimised or, with need_front, not in front; or the clock switched off)."""
        if not self.ok:
            return
        if not self.carla or not self.q.IsWindow(self.carla):
            self.carla = self._find_carla()
        show = bool(visible and self.carla and not self.q.IsIconic(self.carla)
                    and (self.carla_in_front() or not need_front))
        if show:
            pt = wintypes.POINT(0, 0)
            self.q.ClientToScreen(self.carla, ctypes.byref(pt))
            self.u.SetWindowPos(self.hwnd, wintypes.HWND(_HWND_TOPMOST), pt.x + self.margin, pt.y + self.margin,
                                0, 0, _SWP)
        elif self.shown is not False:
            self.u.ShowWindow(self.hwnd, _SW_HIDE)
        self.shown = show

    ARROWS = {"LEFT": 0x25, "UP": 0x26, "RIGHT": 0x27, "DOWN": 0x28}
    # CARLA's spectator turns only with the mouse (right button held): no key or input axis turns it (measured,
    # tools/probe_keys.py). So the arrows turn it the same way: right button down, relative mouse moves,
    # right button up, given to CARLA as input. Measured: 360 px -> 26.5 deg yaw, 26.0 deg pitch, and Z
    # still moves the full 20.4 m while it turns. (The other way, cosim/flycam.ArrowTurn, is
    # `--arrows tick`.)
    DEG_PER_PX = 26.5 / 360.0
    _MOVE, _RDOWN, _RUP, _VK_RBUTTON = 0x0001, 0x0008, 0x0010, 0x02
    _rmb = False
    _acc = (0.0, 0.0)

    def look(self, yaw_deg: float, pitch_deg: float) -> None:
        """Turn CARLA's view by these degrees (pitch up positive), as the mouse would."""
        if not self.ok:
            return
        if not self._rmb:
            if self.q.GetAsyncKeyState(self._VK_RBUTTON) & 0x8000:
                pass  # you are holding the right button yourself: just move
            else:
                self._cursor_on_carla()
                self.u.mouse_event(self._RDOWN, 0, 0, 0, 0)
                self._rmb = True
                self._acc = (0.0, 0.0)
                return  # let CARLA take the mouse first
        ax = self._acc[0] + yaw_deg / self.DEG_PER_PX
        ay = self._acc[1] - pitch_deg / self.DEG_PER_PX
        ix, iy = int(ax), int(ay)
        self._acc = (ax - ix, ay - iy)
        if ix or iy:
            self.u.mouse_event(self._MOVE, ix, iy, 0, 0)

    def _cursor_on_carla(self) -> None:
        """The button goes to the window under the cursor: put the cursor in the middle of CARLA's picture
        first if it is elsewhere (a click on the taskbar or another window must not happen)."""
        pt = wintypes.POINT()
        self.u.GetCursorPos(ctypes.byref(pt))
        rect = wintypes.RECT()
        self.u.GetWindowRect(wintypes.HWND(self.carla), ctypes.byref(rect))
        if not (rect.left < pt.x < rect.right and rect.top < pt.y < rect.bottom):
            self.u.SetCursorPos((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)

    def look_end(self) -> None:
        if self.ok and self._rmb:
            self.u.mouse_event(self._RUP, 0, 0, 0, 0)
        self._rmb = False

    def held(self) -> set[str]:
        """Arrow keys held down now, only while CARLA (or the overlay) is in front."""
        if not self.ok or not self.carla_in_front():
            return set()
        return {name for name, vk in self.ARROWS.items() if self.q.GetAsyncKeyState(vk) & 0x8000}

    def moving(self) -> bool:
        """Z Q S D A E or an arrow held now, while CARLA is in front: the free-fly camera is being flown."""
        if not self.ok or not self.carla_in_front():
            return False
        from cosim.fly_proc import MOVE_KEYS

        return any(self.q.GetAsyncKeyState(vk) & 0x8000 for vk in list(MOVE_KEYS.values()) + list(self.ARROWS.values()))

    def pressed(self) -> set[str]:
        """Keys pressed since the last call, only while CARLA (or the clock) is in front."""
        if not self.ok:
            return set()
        if not self.carla_in_front():
            for vk in VK.values():  # clears bit 0, so Esc pressed in another window does not quit later
                self.q.GetAsyncKeyState(vk)
            self._down.clear()
            return set()
        # held now, or pressed and let go since the last look (bit 0): a quick tap between two loops of the
        # viewer was missed and the key had to be pressed again
        now, tapped = set(), set()
        for name, vk in VK.items():
            st = self.q.GetAsyncKeyState(vk)
            if st & 0x8000:
                now.add(name)
            elif st & 0x0001:
                tapped.add(name)
        new = (now - self._down) | (tapped - self._down)
        self._down = now
        return new


def premultiplied(surface, rect):
    """The pixels of `rect` (x, y, w, h) of a pygame surface with per-pixel alpha, as Windows wants them for a
    layered window: an array (h, w, 4) of B, G, R, A with the colours multiplied by their alpha."""
    import numpy as np

    x, y, w, h = rect
    if hasattr(surface, "premul_alpha"):
        # pygame's own multiply (C): several times faster than numpy's on the clock card and the mini-map. Of a
        # copy: premul_alpha on a subsurface itself reads the wrong rows (pygame 2.6.1; seen as stripes over the
        # whole picture, 11 Nov; tests/test_overlay_alpha.py)
        sub = surface.subsurface((x, y, w, h)).copy().premul_alpha()
        return np.frombuffer(sub.get_view("0"), dtype=np.uint8).reshape(h, sub.get_pitch() // 4, 4)[:, :w]
    src = np.frombuffer(surface.get_view("0"), dtype=np.uint8).reshape(surface.get_height(), surface.get_pitch() // 4, 4)
    s = src[y:y + h, x:x + w]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = ((s[:, :, :3].astype(np.uint16) * s[:, :, 3:4].astype(np.uint16) + 127) // 255).astype(np.uint8)
    out[:, :, 3] = s[:, :, 3]
    return out


class AlphaPresenter:
    """The overlay's picture with a transparency of its own in every pixel (11 Nov): smooth edges, glass, soft
    shadows over CARLA's picture, where the colour key gave see-through or solid and nothing between.

    The viewer draws on a pygame surface with per-pixel alpha (straight alpha, as pygame blends); `present`
    copies the changed rectangles into a 32-bit bitmap, colours multiplied by their alpha as Windows wants them,
    and gives the window that bitmap (UpdateLayeredWindowIndirect, only the changed part)."""

    def __init__(self, hwnd: int, size: tuple[int, int]):
        import numpy as np

        self.np = np
        self.hwnd, self.size = hwnd, (int(size[0]), int(size[1]))
        self.ok = False
        self.error = ""
        g, u = ctypes.windll.gdi32, ctypes.windll.user32
        self.g, self.u = g, u
        w, h = self.size

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class BLENDFUNCTION(ctypes.Structure):
            _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                        ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_byte)]

        class ULWINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("hdcDst", wintypes.HDC), ("pptDst", ctypes.POINTER(wintypes.POINT)),
                        ("psize", ctypes.POINTER(wintypes.SIZE)), ("hdcSrc", wintypes.HDC),
                        ("pptSrc", ctypes.POINTER(wintypes.POINT)), ("crKey", wintypes.COLORREF),
                        ("pblend", ctypes.POINTER(BLENDFUNCTION)), ("dwFlags", wintypes.DWORD),
                        ("prcDirty", ctypes.POINTER(wintypes.RECT))]

        self._ULWINFO = ULWINFO
        g.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g.CreateCompatibleDC.restype = wintypes.HDC
        g.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
        g.CreateDIBSection.restype = wintypes.HBITMAP
        g.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        g.SelectObject.restype = wintypes.HGDIOBJ
        g.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        g.DeleteDC.argtypes = [wintypes.HDC]
        u.UpdateLayeredWindowIndirect.argtypes = [wintypes.HWND, ctypes.POINTER(ULWINFO)]
        u.UpdateLayeredWindowIndirect.restype = wintypes.BOOL
        bi = BITMAPINFOHEADER()
        bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.biWidth, bi.biHeight, bi.biPlanes, bi.biBitCount = w, -h, 1, 32  # top-down, BGRA
        self.dc = g.CreateCompatibleDC(None)
        bits = ctypes.c_void_p()
        self.bmp = g.CreateDIBSection(self.dc, ctypes.byref(bi), 0, ctypes.byref(bits), None, 0)
        if not self.dc or not self.bmp or not bits.value:
            self.error = "no bitmap for the overlay"
            return
        self.old = g.SelectObject(self.dc, self.bmp)
        self.px = np.ctypeslib.as_array((ctypes.c_ubyte * (w * h * 4)).from_address(bits.value)).reshape(h, w, 4)
        self.px[:] = 0
        self.blend = BLENDFUNCTION(0, 0, 255, 1)  # AC_SRC_OVER, the bitmap's own alpha (AC_SRC_ALPHA)
        self.sz = wintypes.SIZE(w, h)
        self.src = wintypes.POINT(0, 0)
        self.ok = True
        self.failed = 0

    def close(self) -> None:
        if getattr(self, "bmp", None):
            self.g.SelectObject(self.dc, self.old)
            self.g.DeleteObject(self.bmp)
            self.g.DeleteDC(self.dc)
            self.bmp = None
        self.ok = False

    def present(self, surface, rects, full: bool = False) -> None:
        """Send `rects` of the per-pixel-alpha `surface` (everything when `full`) to the window."""
        if not self.ok:
            return
        w, h = self.size
        if full:
            rects = [(0, 0, w, h)]
        rects = {(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rects}  # (the clock's is in both lists)
        x0 = y0 = 1 << 30
        x1 = y1 = -1
        sent, area = [], 0
        for r in rects:
            rx, ry, rw, rh = r
            ax, ay, bx, by = max(0, rx), max(0, ry), min(w, rx + rw), min(h, ry + rh)
            if bx <= ax or by <= ay:
                continue
            self.px[ay:by, ax:bx] = premultiplied(surface, (ax, ay, bx - ax, by - ay))
            x0, y0, x1, y1 = min(x0, ax), min(y0, ay), max(x1, bx), max(y1, by)
            sent.append((ax, ay, bx, by))
            area += (bx - ax) * (by - ay)
        if x1 < 0:
            return
        # far-apart parts (the clock top left, the mini-map bottom right): each on its own, not the screen between
        if len(sent) <= 8 and (x1 - x0) * (y1 - y0) > 2 * area:
            for q in sent:
                self._send(*q)
        else:
            self._send(x0, y0, x1, y1)

    def _send(self, x0: int, y0: int, x1: int, y1: int) -> None:
        info = self._ULWINFO()
        info.cbSize = ctypes.sizeof(self._ULWINFO)
        info.psize = ctypes.pointer(self.sz)
        info.hdcSrc = self.dc
        info.pptSrc = ctypes.pointer(self.src)
        info.pblend = ctypes.pointer(self.blend)
        info.dwFlags = 0x2  # ULW_ALPHA
        dirty = wintypes.RECT(x0, y0, x1, y1)
        info.prcDirty = ctypes.pointer(dirty)
        if not self.u.UpdateLayeredWindowIndirect(self.hwnd, ctypes.byref(info)):
            self.failed += 1
            self.error = f"UpdateLayeredWindowIndirect failed ({ctypes.GetLastError()})"

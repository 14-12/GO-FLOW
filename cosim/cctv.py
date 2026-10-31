"""The traffic camera you open with H, full screen, as a live street camera (CCTV) feed.

The camera's picture (with the detector's boxes, cosim/inspector.py) fills the CARLA window, cropped to
fill it, slightly cooler and less saturated, with scanlines, a vignette and a little grain. Over it:

* top left: the camera's name, its junction and direction, the town, the date and time of the game clock;
* top right: a red dot and REC (the dot blinks once a second), and LIVE with the picture's rate;
* bottom left: what the detector sees (vehicles, buses, pedestrians, emergency vehicles with the siren on
  and off) and which detector runs;
* bottom right: how to close it (H);
* viewfinder corners.

Two ways (cosim/inspector.py): native (default) draws only these and the detector's boxes over CARLA's own
window, which then is the picture, at CARLA's frame rate (draw_native); sensor draws the camera sensor's
picture too, graded, with scanlines, vignette and grain (frame). Drawn by the clock overlay
(cosim/freefly.py) while a camera is open.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pygame

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
REC_RED = (235, 40, 40)
WHITE = (240, 244, 250)
DIM = (178, 188, 204)
PANEL = (8, 10, 16, 165)
PANEL_SOLID = (12, 14, 20, 255)
BOX_RGB = {"car": (80, 200, 80), "van": (80, 200, 80), "truck": (80, 200, 80), "bus": (255, 190, 0),
           "pedestrian": (60, 220, 255), "ambulance": (60, 140, 255), "police": (60, 140, 255),
           "ambulance_lights": (255, 60, 60), "police_lights": (255, 60, 60)}  # cosim/inspector.GROUPS, as RGB
BOX_LABEL = {"ambulance_lights": "ambulance SIREN ON", "police_lights": "police SIREN ON"}
KEY_SAFE = (2, 0, 2)  # the overlay's see-through colour is (1, 0, 1): never draw it, not even in the picture


def _font(name: str, size: int) -> pygame.font.Font:
    p = FONTS / name
    return pygame.font.Font(str(p) if p.is_file() else None, size)


def split_name(name: str) -> tuple[str, str, str]:
    """'1126-2 looking north' -> ('1126-2', '1126', 'north')."""
    cam, _, rest = name.partition(" ")
    direction = rest.replace("looking", "").strip()
    return cam, cam.split("-")[0], direction


class CCTV:
    def __init__(self, town: str = "Town05"):
        if not pygame.font.get_init():
            pygame.font.init()
        self.town = town
        self.f_cam = _font("BarlowCondensed-SemiBold.ttf", 44)
        self.f_sub = _font("Barlow-SemiBold.ttf", 17)
        self.f_time = _font("BarlowCondensed-Medium.ttf", 30)
        self.f_rec = _font("BarlowCondensed-SemiBold.ttf", 40)
        self.f_small = _font("Barlow-Medium.ttf", 15)
        self.f_count_n = _font("BarlowCondensed-SemiBold.ttf", 34)
        self.f_count_l = _font("Barlow-Medium.ttf", 13)
        self.f_box = _font("Barlow-SemiBold.ttf", 14)
        self._tcache: dict = {}
        self._static = None
        self._fx: dict[tuple[int, int], pygame.Surface] = {}
        self._grain: pygame.Surface | None = None
        self._rng = random.Random(5)

    # ------------------------------------------------------------ effects (made once per size)
    def _effects(self, size: tuple[int, int]) -> pygame.Surface:
        fx = self._fx.get(size)
        if fx is not None:
            return fx
        w, h = size
        a = np.zeros((h, w), np.float32)
        a[::3, :] = 26.0  # scanlines
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        a = np.maximum(a, np.clip((r - 0.55) / 0.85, 0.0, 1.0) ** 1.6 * 175.0)  # vignette
        fx = pygame.Surface(size, pygame.SRCALPHA)
        fx.fill((0, 0, 0, 0))
        alpha = pygame.surfarray.pixels_alpha(fx)
        alpha[:, :] = a.T.astype(np.uint8)
        del alpha
        self._fx[size] = fx
        return fx

    def _grain_tile(self) -> pygame.Surface:
        if self._grain is None:
            n = 256
            rng = np.random.default_rng(3)
            v = rng.integers(0, 255, (n, n), dtype=np.uint8)
            g = pygame.Surface((n, n), pygame.SRCALPHA)
            rgb = pygame.surfarray.pixels3d(g)
            rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = v
            del rgb
            al = pygame.surfarray.pixels_alpha(g)
            al[:, :] = 16
            del al
            self._grain = g
        return self._grain

    # ------------------------------------------------------------ the picture
    _LUT = None

    @classmethod
    def grade(cls, rgb: np.ndarray) -> np.ndarray:
        """A street camera's colours: a little less saturated (78 %), a touch more contrast, a little cooler.
        Integer arithmetic and one lookup table per channel: ~4 ms for 960 x 540 (float numpy took 19)."""
        if cls._LUT is None:
            x = np.arange(256, dtype=np.float32)
            base = (x - 128.0) * 1.08 + 124.0
            cls._LUT = [np.clip(base + d, 0, 255).astype(np.uint8) for d in (-3.0, 0.0, 6.0)]
        c = rgb.astype(np.uint16)
        grey = (c[:, :, 0] + c[:, :, 1] + c[:, :, 2]) // 3
        mixed = (c * 200 + grey[:, :, None] * 56) >> 8
        out = np.empty(rgb.shape, np.uint8)
        for k in range(3):
            out[:, :, k] = cls._LUT[k][mixed[:, :, k]]
        key = (out[:, :, 0] == 1) & (out[:, :, 1] == 0) & (out[:, :, 2] == 1)
        out[key] = KEY_SAFE
        return out

    @staticmethod
    def cover(img: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
        """Scaled to fill `size`, the overflow cropped evenly."""
        w, h = img.get_size()
        k = max(size[0] / w, size[1] / h)
        sw, sh = max(1, round(w * k)), max(1, round(h * k))
        big = pygame.transform.smoothscale(img, (sw, sh))
        return big.subsurface(((sw - size[0]) // 2, (sh - size[1]) // 2, size[0], size[1]))

    # ------------------------------------------------------------ text helpers
    @staticmethod
    def _text(surf, font, text, colour, pos, anchor="topleft", shadow=True, pill=False):
        """Draws text; returns everything it drew (the shadow too, so it is erased with the text). pill: on a
        small dark plate (the native feed: small text over the picture needs one to read cleanly)."""
        img = font.render(text, True, colour)
        r = img.get_rect(**{anchor: pos})
        drawn = r
        if pill:
            plate = r.inflate(14, 6)
            pygame.draw.rect(surf, PANEL_SOLID[:3], plate, border_radius=5)
            drawn = plate
        elif shadow:
            sh = font.render(text, True, (0, 0, 0))
            sh.set_alpha(150)
            surf.blit(sh, r.move(2, 2))
            drawn = r.union(r.move(2, 2))
        surf.blit(img, r)
        return drawn

    @staticmethod
    def _corners(surf, size, inset=26, arm=46, width=3, colour=(235, 240, 248)) -> list:
        w, h = size
        rects = []
        for cx, cy, dx, dy in ((inset, inset, 1, 1), (w - inset, inset, -1, 1),
                               (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1)):
            pygame.draw.line(surf, colour, (cx, cy), (cx + dx * arm, cy), width)
            pygame.draw.line(surf, colour, (cx, cy), (cx, cy + dy * arm), width)
            rects.append(pygame.Rect(min(cx, cx + dx * arm) - 3, min(cy, cy + dy * arm) - 3, arm + 7, arm + 7))
        return rects

    def _panel(self, size, alpha_rgba=PANEL, radius=10) -> pygame.Surface:
        p = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(p, alpha_rgba, p.get_rect(), border_radius=radius)
        return p

    def _hud(self, out, size, counts, fps, name, model_state, stamp, now, solid: bool = False) -> list:
        """The feed's labels, drawn on `out`; the rectangles drawn. solid: over CARLA's own window through
        the see-through overlay, which knows no half-transparency: opaque panels, no glow."""
        w, h = size
        rects = self._corners(out, size)

        # top left: which camera, where, when
        cam, tls, direction = split_name(name)
        m = 48
        r = self._text(out, self.f_cam, f"CAM {cam}", WHITE, (m, m - 6))
        rects.append(r)
        sub = f"JUNCTION {tls}  ·  LOOKING {direction.upper()}  ·  {self.town.upper()}"
        r = self._text(out, self.f_sub, sub, DIM, (m + 2, r.bottom + 2))
        rects.append(r)
        rects.append(self._text(out, self.f_time, stamp, WHITE, (m + 1, r.bottom + 6)))

        # top right: recording
        on = (now % 1.0) < 0.62
        rec = self.f_rec.render("REC", True, WHITE)
        rr = rec.get_rect(topright=(w - m, m - 8))
        dot_c = (rr.left - 30, rr.centery + 1)
        rects.append(pygame.Rect(dot_c[0] - 32, dot_c[1] - 32, 64, 64))
        if on and not solid:
            glow = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*REC_RED, 70), (30, 30), 22)
            out.blit(glow, (dot_c[0] - 30, dot_c[1] - 30))
        if on:
            pygame.draw.circle(out, REC_RED, dot_c, 12)
        else:
            pygame.draw.circle(out, (90, 30, 30), dot_c, 12, 2)
        sh = self.f_rec.render("REC", True, (0, 0, 0))
        sh.set_alpha(150)
        out.blit(sh, rr.move(2, 2))
        out.blit(rec, rr)
        rects.append(rr.union(rr.move(2, 2)).inflate(4, 4))
        live = "LIVE" + (f"  ·  {fps:.0f} detections/s" if fps else "")
        rects.append(self._text(out, self.f_small, live, DIM, (w - m, rr.bottom + 6), "topright", pill=solid))

        # bottom left: what the detector sees
        items = [("VEHICLES", counts.get("vehicles", 0), WHITE),
                 ("BUSES", counts.get("public transport", 0), (120, 205, 255)),
                 ("PEDESTRIANS", counts.get("pedestrians", 0), (255, 222, 110)),
                 ("SIREN ON", counts.get("emergency, SIREN ON", 0), (255, 110, 110)),
                 ("SIREN OFF", counts.get("emergency, siren off", 0), (255, 170, 120))]
        col_w, pad = 118, 16
        pw, ph = pad * 2 + col_w * len(items), 104
        px, py = m - 4, h - m - ph - 4
        out.blit(self._panel((pw, ph), PANEL_SOLID if solid else PANEL), (px, py))
        rects.append(pygame.Rect(px, py, pw, ph))
        self._text(out, self.f_count_l, "DETECTED IN VIEW", DIM, (px + pad, py + 10), shadow=False)
        for k, (label, n, colour) in enumerate(items):
            x = px + pad + k * col_w
            self._text(out, self.f_count_n, str(n), colour, (x, py + 28), shadow=False)
            self._text(out, self.f_count_l, label, DIM, (x + 1, py + 70), shadow=False)
        rects.append(self._text(out, self.f_small, model_state, DIM, (px + 8, py - 10), "bottomleft", pill=solid))

        # bottom right: how to close
        hint = self._panel((176, 40), PANEL_SOLID if solid else PANEL, radius=8)
        hx, hy = w - m - 176 + 4, h - m - 40 - 4
        out.blit(hint, (hx, hy))
        rects.append(pygame.Rect(hx, hy, 176, 40))
        key = pygame.Rect(hx + 12, hy + 8, 26, 24)
        pygame.draw.rect(out, (235, 240, 248), key, border_radius=5)
        self._text(out, self.f_small, "H", (14, 16, 22), key.center, "center", shadow=False)
        self._text(out, self.f_small, "close camera", WHITE, (key.right + 10, key.centery), "midleft", shadow=False)
        return rects

    def _cached(self, font, text: str, colour, shadow: bool = False, pill: bool = False) -> pygame.Surface:
        """A text drawn once: with its drop shadow, or on its dark plate (pill). Rendering each text again
        every redraw made the native feed's ~60 pygame calls, each giving up and waiting for Python's lock
        behind the viewer's SUMO thread: 5 redraws a second instead of 30 (measured on the laptop; the
        same drawing alone does 470 a second there, tools/bench_overlay.py)."""
        key = (id(font), text, colour, shadow, pill)
        img = self._tcache.get(key)
        if img is None:
            t = font.render(text, True, colour)
            if pill:
                img = pygame.Surface((t.get_width() + 14, t.get_height() + 6), pygame.SRCALPHA)
                pygame.draw.rect(img, PANEL_SOLID, img.get_rect(), border_radius=5)
                img.blit(t, (7, 3))
            elif shadow:
                img = pygame.Surface((t.get_width() + 2, t.get_height() + 2), pygame.SRCALPHA)
                sh = font.render(text, True, (0, 0, 0))
                sh.set_alpha(150)
                img.blit(sh, (2, 2))
                img.blit(t, (0, 0))
            else:
                img = t
            if len(self._tcache) > 400:
                self._tcache.clear()
            self._tcache[key] = img
        return img

    def _put(self, surf, img, **anchor) -> pygame.Rect:
        r = img.get_rect(**anchor)
        surf.blit(img, r)
        return r

    def _static_layer(self, size, name: str):
        """Everything of the native feed that does not change while a camera is open, drawn once: the
        corners, the camera's name and place, the panels and their labels. Returns (layer, rectangles,
        where the dynamic parts go)."""
        key = (size, name)
        if self._static is not None and self._static[0] == key:
            return self._static[1:]
        w, h = size
        layer = pygame.Surface(size, pygame.SRCALPHA)
        rects = self._corners(layer, size)
        cam, tls, direction = split_name(name)
        m = 48
        r = self._put(layer, self._cached(self.f_cam, f"CAM {cam}", WHITE, shadow=True), topleft=(m, m - 6))
        rects.append(r)
        sub = f"JUNCTION {tls}  ·  LOOKING {direction.upper()}  ·  {self.town.upper()}"
        r2 = self._put(layer, self._cached(self.f_sub, sub, DIM, shadow=True), topleft=(m + 2, r.bottom + 2))
        rects.append(r2)
        col_w, pad = 118, 16
        pw, ph = pad * 2 + col_w * 5, 104
        px, py = m - 4, h - m - ph - 4
        layer.blit(self._panel((pw, ph), PANEL_SOLID), (px, py))
        rects.append(pygame.Rect(px, py, pw, ph))
        self._put(layer, self._cached(self.f_count_l, "DETECTED IN VIEW", DIM), topleft=(px + pad, py + 10))
        for k, label in enumerate(("VEHICLES", "BUSES", "PEDESTRIANS", "SIREN ON", "SIREN OFF")):
            self._put(layer, self._cached(self.f_count_l, label, DIM), topleft=(px + pad + k * col_w + 1, py + 70))
        hx, hy = w - m - 176 + 4, h - m - 40 - 4
        layer.blit(self._panel((176, 40), PANEL_SOLID, radius=8), (hx, hy))
        rects.append(pygame.Rect(hx, hy, 176, 40))
        keyr = pygame.Rect(hx + 12, hy + 8, 26, 24)
        pygame.draw.rect(layer, (235, 240, 248), keyr, border_radius=5)
        self._put(layer, self._cached(self.f_small, "H", (14, 16, 22)), center=keyr.center)
        self._put(layer, self._cached(self.f_small, "close camera", WHITE), midleft=(keyr.right + 10, keyr.centery))
        places = {"stamp": (m + 1, r2.bottom + 6), "rec_right": (w - m, m - 8), "panel": (px, py, pad, col_w)}
        # only what is drawn (the layer is see-through elsewhere): its bounding rectangles
        self._static = (key, layer, rects, places)
        return layer, rects, places

    def draw_native(self, screen, size, view, name: str, model_state: str, stamp: str, boxes,
                    now: float | None = None) -> list:
        """Native feed: CARLA's window is the picture; draws the detector's boxes (window pixels) and the
        labels on the see-through overlay `screen`. Returns the rectangles drawn (the overlay sends only
        those to the window). The parts that never change come from one layer drawn once
        (_static_layer); texts are rendered once each (_cached)."""
        now = time.time() if now is None else now
        counts, fps = ({}, 0.0) if view is None else (view[1], view[2])
        if view is not None:
            name = view[3]
        rects = []
        for x0, y0, x1, y1, cls, p in boxes:
            colour = BOX_RGB.get(cls, (200, 200, 200))
            r = pygame.Rect(round(x0), round(y0), max(2, round(x1 - x0)), max(2, round(y1 - y0)))
            pygame.draw.rect(screen, colour, r, 2)
            label = self._cached(self.f_box, f"{BOX_LABEL.get(cls, cls)} {p:.2f}", (16, 18, 24))
            bg = label.get_rect(bottomleft=(r.left, r.top)).inflate(8, 4)
            pygame.draw.rect(screen, colour, bg)
            screen.blit(label, label.get_rect(center=bg.center))
            rects += [r.inflate(4, 4), bg]
        layer, static_rects, places = self._static_layer(size, name)
        screen.blit(layer, (0, 0))
        rects += static_rects
        rects.append(self._put(screen, self._cached(self.f_time, stamp, WHITE, shadow=True), topleft=places["stamp"]))
        rec = self._cached(self.f_rec, "REC", WHITE, shadow=True)
        rr = self._put(screen, rec, topright=places["rec_right"])
        rects.append(rr)
        dot = (rr.left - 30, rr.centery)
        if (now % 1.0) < 0.62:
            pygame.draw.circle(screen, REC_RED, dot, 12)
        else:
            pygame.draw.circle(screen, (90, 30, 30), dot, 12, 2)
        rects.append(pygame.Rect(dot[0] - 14, dot[1] - 14, 28, 28))
        live = "LIVE" + (f"  ·  {fps:.0f} detections/s" if fps else "")
        rects.append(self._put(screen, self._cached(self.f_small, live, DIM, pill=True), topright=(rr.right, rr.bottom + 6)))
        px, py, pad, col_w = places["panel"]
        for k, (key, colour) in enumerate((("vehicles", WHITE), ("public transport", (120, 205, 255)),
                                           ("pedestrians", (255, 222, 110)), ("emergency, SIREN ON", (255, 110, 110)),
                                           ("emergency, siren off", (255, 170, 120)))):
            rects.append(self._put(screen, self._cached(self.f_count_n, str(counts.get(key, 0)), colour),
                                   topleft=(px + pad + k * col_w, py + 28)))
        rects.append(self._put(screen, self._cached(self.f_small, model_state, DIM, pill=True),
                               bottomleft=(px + 2, py - 4)))
        return rects

    # ------------------------------------------------------------ one frame
    def frame(self, size: tuple[int, int], view, name: str, model_state: str, stamp: str,
              now: float | None = None) -> pygame.Surface:
        """The whole screen. view: (RGB picture, counts, detections a second, camera name) or None."""
        now = time.time() if now is None else now
        w, h = size
        out = pygame.Surface(size)
        out.fill((6, 7, 10))
        if view is not None:
            rgb, counts, fps, name = view
            cached = getattr(self, "_pic", None)
            if cached is None or cached[0] is not rgb or cached[1] != size:  # a new picture from the detector
                img = pygame.image.frombuffer(self.grade(rgb).tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
                self._pic = cached = (rgb, size, self.cover(img, size))
            out.blit(cached[2], (0, 0))
        else:
            counts, fps = {}, 0.0
            g = self._grain_tile()
            for y in range(0, h, g.get_height()):
                for x in range(0, w, g.get_width()):
                    out.blit(g, (x, y))
            self._text(out, self.f_cam, "CONNECTING ...", WHITE, (w // 2, h // 2 - 20), "center")
            self._text(out, self.f_sub, "waiting for the camera's first picture", DIM, (w // 2, h // 2 + 22), "center")
        out.blit(self._effects(size), (0, 0))
        g = self._grain_tile()
        ox, oy = self._rng.randrange(g.get_width()), self._rng.randrange(g.get_height())
        for y in range(-oy, h, g.get_height()):
            for x in range(-ox, w, g.get_width()):
                out.blit(g, (x, y))
        self._hud(out, size, counts, fps, name, model_state, stamp, now)
        return out


def stamp_of(st) -> str:
    """The game clock for the feed: 'MON  DAY 3  18:39'."""
    return f"{st.weekday.upper()}  ·  DAY {st.day}  ·  {st.hhmm}"


if __name__ == "__main__":  # a preview: python -m cosim.cctv picture.jpg out.png [W H]
    import sys

    from PIL import Image

    pygame.init()
    src, dst = sys.argv[1], sys.argv[2]
    W, H = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1920, 1080)
    rgb = np.asarray(Image.open(src).convert("RGB"))
    counts = {"vehicles": 7, "public transport": 1, "pedestrians": 3, "emergency, SIREN ON": 1,
              "emergency, siren off": 0}
    t0 = time.perf_counter()
    cctv = CCTV()
    surf = cctv.frame((W, H), (rgb, counts, 12.0, "1882-3 looking north"), "",
                      "YOLO26n (CARLA-trained) on GPU", "TUE  ·  DAY 2  ·  18:39", now=0.2)
    t1 = time.perf_counter()
    for _ in range(10):
        cctv.frame((W, H), (rgb, counts, 12.0, "1882-3 looking north"), "", "YOLO26n (CARLA-trained) on GPU",
                   "TUE  ·  DAY 2  ·  18:39")
    print(f"first frame {1000 * (t1 - t0):.0f} ms, then {100 * (time.perf_counter() - t1):.0f} ms a frame")
    pygame.image.save(surf, dst)

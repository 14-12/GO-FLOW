"""The accident markers on the free-fly overlay (23 Oct).

On screen, a marker stands over the accident: a warning sign (a triangle with "!"), coloured by what is happening
(red: just happened, blue: police on the way or there, amber: the recovery trucks at work, green: clearing up),
with a label: what (ACCIDENT, POLICE ON SCENE, RECOVERY, ...) and how far. Off screen, or behind you, the marker
sits at the edge of the screen in the accident's direction, with an arrow pointing there and the distance: fly
that way and it slides onto the scene. Close by (under 120 m) the label also names the cars' plates.

Further than EDGE_MAX_M off screen (or SHOWN_MAX_M on screen) there is no marker: the accident is not yours to see.
Its notice is made once per text (the distance in steps of 10 m) and moves only when it has moved 3 px or more,
so the window is not redrawn under it on every picture.

Drawn into the overlay window (cosim/overlay.py), whose see-through colour is a colour key: no shape is drawn with
blended edges against it (they would fringe); text is drawn on its own solid pill.
"""

from __future__ import annotations

import math

import pygame

from cosim import ui
from cosim.hud import FONTS

COLOURS = {"crash": (232, 64, 56), "police en route": (64, 132, 255), "secured": (64, 132, 255),
           "recovery": (255, 176, 46), "clearing": (72, 196, 120), "closed": (72, 196, 120)}
EDGE = 64          # px kept free at the screen's edge for an off-screen marker
NEAR_PLATES_M = 120.0
EDGE_MAX_M = 250.0  # off screen: the edge marker only this close (25 Oct: flown to another part of the town, the
                    # marker went on sitting at the screen's edge, its notice with it: "it follows me")
SHOWN_MAX_M = 900.0  # on screen: only this close


def describe(m: dict) -> str:
    """The marker's second line: what happened (31 Oct: breakdowns and pile-ups too)."""
    if m["kind"] == "breakdown":
        return "breakdown, 1 car" if m["cars"] == 1 else f"breakdown, {m['cars']} cars"
    if m["kind"] == "pile-up":
        return f"{m['label']} pile-up, {m['cars']} cars, {m['kmh']} km/h"
    return f"{m['label']} {m['kind']} crash, {m['cars']} cars, {m['kmh']} km/h"


class MarkerHud:
    def __init__(self, fov_deg: float = 90.0):
        self.fov = fov_deg
        self._pills: dict = {}
        self._at: dict = {}
        self.f_title = pygame.font.Font(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 17)
        self.f_small = pygame.font.Font(str(FONTS / "Barlow-Medium.ttf"), 13)

    def project(self, tf, size, x: float, y: float, z: float):
        """Screen (sx, sy, in_front) of CARLA point (x, y, z) seen from transform `tf` (CARLA's view, fov_deg across)."""
        w, h = size
        f = tf.get_forward_vector()
        r = tf.get_right_vector()
        u = tf.get_up_vector()
        dx, dy, dz = x - tf.location.x, y - tf.location.y, z - tf.location.z
        cx = dx * f.x + dy * f.y + dz * f.z
        cy = dx * r.x + dy * r.y + dz * r.z
        cz = dx * u.x + dy * u.y + dz * u.z
        fpx = (w / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        if cx > 0.5:
            return w / 2.0 + cy / cx * fpx, h / 2.0 - cz / cx * fpx, True
        return w / 2.0 + cy * 1e3, h / 2.0 - cz * 1e3, False  # behind: only the direction counts

    _signs: dict = {}

    @classmethod
    def _triangle(cls, surf, cx, cy, s, colour):
        if ui.ALPHA:  # smooth edges: drawn 4x once per size and colour, scaled down
            key = (round(s), colour)
            sign = cls._signs.get(key)
            if sign is None:
                n = 2 * (round(s) + 8)
                big = pygame.Surface((n * 4, n * 4), pygame.SRCALPHA)
                cls._triangle_shape(big, n * 2, n * 2, round(s) * 4, colour)
                sign = cls._signs[key] = pygame.transform.smoothscale(big, (n, n))
            surf.blit(sign, (int(cx) - sign.get_width() // 2, int(cy) - sign.get_height() // 2))
            return
        cls._triangle_shape(surf, cx, cy, s, colour)

    @staticmethod
    def _triangle_shape(surf, cx, cy, s, colour):
        pts = [(cx, cy - s), (cx + s * 0.95, cy + s * 0.68), (cx - s * 0.95, cy + s * 0.68)]
        pygame.draw.polygon(surf, (16, 17, 20), [(px + (px - cx) * 0.22, py + (py - cy) * 0.22) for px, py in pts])
        pygame.draw.polygon(surf, colour, pts)
        pygame.draw.rect(surf, (16, 17, 20), (cx - s * 0.1, cy - s * 0.45, s * 0.2, s * 0.62))
        pygame.draw.rect(surf, (16, 17, 20), (cx - s * 0.1, cy + s * 0.27, s * 0.2, s * 0.2))

    def _pill(self, title: str, sub: str, colour):
        if ui.ALPHA:   # (12 Nov) the overlay's design system (cosim/hudkit.py)
            from cosim import hudkit

            kind = {v: k for k, v in COLOURS.items()}.get(colour, "crash")
            return hudkit.tag(hudkit.INCIDENT_KIND.get(kind, "accident"), title, sub, accent=colour)
        t = self.f_title.render(title, True, (245, 246, 248), (22, 24, 30))
        b = self.f_small.render(sub, True, (200, 206, 216), (22, 24, 30)) if sub else None
        w = max(t.get_width(), b.get_width() if b else 0) + 22
        h = t.get_height() + (b.get_height() if b else 0) + 10
        surf = pygame.Surface((w, h))
        surf.fill((22, 24, 30))
        pygame.draw.rect(surf, colour, (0, 0, 5, h))
        surf.blit(t, (13, 4))
        if b:
            surf.blit(b, (13, 4 + t.get_height()))
        return ui.glassify(surf)

    def draw(self, screen, size, view_tf, markers: list[dict], now: float, plates: dict | None = None) -> list:
        """Draws the markers; returns the rectangles drawn (the overlay sends only those)."""
        rects = []
        w, h = size
        loc = view_tf.location
        for m in markers:
            colour = COLOURS.get(m["phase"], COLOURS["crash"])
            d = math.dist((loc.x, loc.y, loc.z), (m["x"], m["y"], m["z"]))
            sx, sy, front = self.project(view_tf, size, m["x"], m["y"], m["z"])
            on = front and EDGE <= sx <= w - EDGE and EDGE <= sy <= h - EDGE
            if d > (SHOWN_MAX_M if on else EDGE_MAX_M):
                self._at.pop(m.get("id"), None)
                continue
            last = self._at.get(m.get("id"))
            if last is not None and last[2] == on and math.hypot(sx - last[0], sy - last[1]) < 3.0:
                sx, sy = last[0], last[1]  # (still: not redrawn elsewhere for a pixel or two)
            else:
                self._at[m.get("id")] = (sx, sy, on)
            dist = f"{d:.0f} m" if d < 100 else (f"{round(d, -1):.0f} m" if d < 1000 else f"{d / 1000:.1f} km")
            title = f"{m['status']}  ·  {dist}"
            sub = describe(m)
            if plates and d < NEAR_PLATES_M:
                names = [plates.get(v) for v in m.get("plate_ids", []) if plates.get(v)]
                if names:
                    sub += "  ·  " + "  ".join(names)
            pulse = 0.5 + 0.5 * math.sin(now * 5.0) if m["phase"] == "crash" else 1.0
            size_px = 15 + 5 * pulse
            if on:
                ax, ay = sx, sy
            else:  # at the edge, towards it
                vx, vy = sx - w / 2.0, sy - h / 2.0
                if not front and abs(vx) < 1e-3 and abs(vy) < 1e-3:
                    vx = 1.0
                k = min((w / 2.0 - EDGE) / max(abs(vx), 1e-6), (h / 2.0 - EDGE) / max(abs(vy), 1e-6))
                ax, ay = w / 2.0 + vx * k, h / 2.0 + vy * k
                ang = math.atan2(vy, vx)
                tip = (ax + math.cos(ang) * 30, ay + math.sin(ang) * 30)
                left = (ax + math.cos(ang + 2.5) * 14, ay + math.sin(ang + 2.5) * 14)
                right = (ax + math.cos(ang - 2.5) * 14, ay + math.sin(ang - 2.5) * 14)
                rects.append(pygame.draw.polygon(screen, colour, [tip, left, right]))
            self._triangle(screen, ax, ay, size_px, colour)
            rects.append(pygame.Rect(ax - 26, ay - 26, 52, 52))
            key = (title, sub if on else "", colour)
            pill = self._pills.get(key)
            if pill is None:
                if len(self._pills) > 64:
                    self._pills.clear()
                pill = self._pills[key] = self._pill(title, key[1], colour)  # (off screen: the title alone)
            px = min(max(ax - pill.get_width() / 2, 8), w - pill.get_width() - 8)
            py = ay + 26 if ay < h / 2 or on else ay - 26 - pill.get_height()
            py = min(max(py, 8), h - pill.get_height() - 8)
            rects.append(screen.blit(pill, (px, py)))
        return rects

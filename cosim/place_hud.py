"""(8 Nov) The town's places named over the CARLA picture (scenarios/town05/places.json, cosim/purposes.py): above each
place near you, a label on a stem down to its roof: its icon, its name, and what is happening there now: a car park's
free spaces (green, amber, red), a shop's or a restaurant's hours (open until..., closed), how many are at work in an
office, how many cars are home. L shows or hides them.

Drawn into the overlay window (cosim/overlay.py) like the junction agents' pills (cosim/agent_hud.py, whose projection
it reuses): solid colours only, the see-through colour is a colour key. Each label is rendered once per text and kept.
"""
from __future__ import annotations

import math

import pygame

from cosim import ui
from cosim.agent_hud import EDGE, AgentHud
from cosim.hud import FONTS
from cosim.place_icons import Painter, colour, occupancy_colour

NEAR_M = 420.0          # labels for places this close
FULL_M = 160.0          # closer than this: the full label (name and what is happening); further, the name only
MAX_LABELS = 9
BG = (12, 16, 26)
TEXT = (238, 242, 250)
MUTED = (160, 172, 196)
OPEN = (52, 211, 153)
STEM_UP_M = 10.0        # the label this far above the roof


def info_line(p: dict, hour: float, live: dict) -> tuple[str, tuple]:
    """(the line under the name, its colour) for place p at town hour `hour`. live: {"occ": area -> cars parked,
    "cap": area -> spaces, "park_of": place -> its areas, "here": place -> its cars parked now} (cosim/purposes.py)."""
    kind = p["kind"]
    here = live.get("here", {}).get(p["id"], 0)
    if kind in ("parking", "kerb"):
        mine = live.get("park_of", {}).get(p["id"], [])
        room = sum(live.get("cap", {}).get(a, 0) for a in mine)
        free = max(0, room - sum(live.get("occ", {}).get(a, 0) for a in mine))
        return f"{free} free of {room}", occupancy_colour(free / room if room else 0.0)
    if kind in ("apartments", "houses"):
        return f"{here} cars home", MUTED
    hrs = p.get("hours")
    who = "at work" if kind == "office" else "here"
    if hrs and list(hrs) != [0, 24]:
        a, b = hrs
        if a <= hour % 24 < b:
            return f"OPEN until {b}:00  ·  {here} {who}", OPEN
        return f"CLOSED  ·  opens {a}:00", MUTED
    if kind == "office":
        return f"{here} at work now", MUTED
    if hrs:
        return f"24/7  ·  {here} {who}", OPEN
    return p.get("sub", ""), MUTED


class PlaceHud(AgentHud):
    def __init__(self, places: list, frame, fov_deg: float = 90.0):
        super().__init__(fov_deg=fov_deg, mode="places")
        self.frame = frame
        self.places = [p for p in places if "label" in p]
        self.f_name = pygame.font.Font(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 19)
        self.f_info = pygame.font.Font(str(FONTS / "Barlow-Medium.ttf"), 13)
        self._cache: dict = {}
        self.carla_xy = {p["id"]: frame.to_carla_xy(*p["label"]) for p in self.places}

    def _label(self, p: dict, line: str, col, full: bool) -> pygame.Surface:
        key = (p["id"], line if full else None)
        surf = self._cache.get(key)
        if surf is not None:
            return surf
        name = self.f_name.render(p["name"], True, TEXT, BG)
        info = self.f_info.render(line, True, col, BG) if full else None
        icon = 30
        h = 44 if full else 34
        w = icon + 22 + max(name.get_width(), info.get_width() if info else 0) + 22
        surf = pygame.Surface((w, h))
        surf.fill(BG if ui.ALPHA else (1, 0, 1))               # (the overlay's colour key: see-through corners)
        kc = colour(p["kind"])
        pygame.draw.rect(surf, kc, surf.get_rect(), border_radius=h // 2)
        pygame.draw.rect(surf, BG, surf.get_rect().inflate(-4, -4), border_radius=h // 2 - 2)
        Painter(surf, "pygame").icon(p["kind"], h // 2 + 1, h // 2, icon)
        x = icon + 16
        if full:
            surf.blit(name, (x, 3))
            pygame.draw.circle(surf, col, (x + 4, 33), 3)
            surf.blit(info, (x + 11, 25))
        else:
            surf.blit(name, (x, (h - name.get_height()) // 2))
        if ui.ALPHA:  # smooth round ends and a soft shadow; r.midbottom in draw() allows for the margin
            surf = ui.glassify(surf, radius=h // 2, body_alpha=232, border=0, shadow=110, bg=BG)
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[key] = surf
        return surf

    def draw(self, screen, size, view_tf, hour: float, live: dict, avoid=()) -> list:
        """The labels; none over the rects in `avoid` (the clock, the mini-map)."""
        w, h = size
        loc = view_tf.location
        near = []
        for p in self.places:
            x, y = self.carla_xy[p["id"]]
            d = math.dist((loc.x, loc.y), (x, y))
            if d > NEAR_M or (p["kind"] == "kerb" and d > 90.0):
                continue
            near.append((d, p, x, y))
        near.sort(key=lambda t: t[0])
        rects, taken, shown = [], [], []
        for d, p, x, y in near[:MAX_LABELS * 2]:      # the nearest first: they get their place on screen
            if len(shown) >= MAX_LABELS:
                break
            top = float(p.get("height") or 6.0)
            base = self.project(view_tf, size, x, y, top)
            tip = self.project(view_tf, size, x, y, top + STEM_UP_M + min(d, 300.0) * 0.04)
            if base is None or tip is None:
                continue
            if not (EDGE <= tip[0] <= w - EDGE and EDGE <= tip[1] <= h - EDGE):
                continue
            full = d <= FULL_M
            line, col = info_line(p, hour, live) if full else ("", None)
            lab = self._label(p, line, col, full)
            r = lab.get_rect(midbottom=(int(tip[0]), int(tip[1]) + (ui.MARGIN if ui.ALPHA else 0)))
            r.clamp_ip(pygame.Rect(8, 8, w - 16, h - 16))          # (whole on screen; its stem still to its place)
            if any(r.colliderect(t) for t in taken) or any(r.colliderect(a) for a in avoid):
                continue
            taken.append(r.inflate(8, 6))
            shown.append((p, lab, r, (r.centerx, r.bottom - (ui.MARGIN if ui.ALPHA else 0)), (int(base[0]), int(base[1]))))
        for p, _lab, _r, tip, (bx, by) in shown:        # the stems under every label
            if 0 <= bx < w and 0 <= by < h:
                kc = colour(p["kind"])
                rects.append(pygame.draw.line(screen, kc, tip, (bx, by), 2))
                rects.append(pygame.draw.circle(screen, kc, (bx, by), 4))
        for _p, lab, r, _tip, _base in reversed(shown):   # the nearest on top
            rects.append(screen.blit(lab, r))
        return rects

"""The free-flight mini-map as a radar (12 Nov): round, heading-up, zooming with the camera's height.

What it takes from the open-world games' maps (docs/WORKLOG.md, 12 Nov): GTA V's round radar turning with you, a
compass bezel whose N moves round the edge (The Witcher 3, Cyberpunk 2077), the zoom following speed (here, height:
the critique of Cyberpunk's driving map), things off the map clamped to the edge as blips with their distance (GTA's
blips), live traffic on the roads in green, amber and red (navigation apps), a view cone for where the camera looks.

The town's map (roads, the places' buildings) is drawn once at two scales; each picture crops it round the camera and
turns and scales it in one step (pygame.transform.rotozoom), then draws what moves on top in screen space. Rendered
5 times a second while flying (cosim/freefly.py holds it between).
"""

from __future__ import annotations

import math

import pygame

from cosim import hudkit, ui
from cosim.place_icons import Painter
from cosim.place_icons import colour as kind_colour

D = 330            # the map's diameter (px)
BEZEL = 16         # the compass ring
BG = (12, 16, 25)
LAND = (17, 22, 33)
ROAD_CASE = (31, 39, 56)
ROAD = (63, 74, 98)
ROAD_LINE = (88, 101, 130)
TEXT = (238, 242, 250)
DIM = (150, 160, 182)
FLOW = {0: (52, 211, 153), 1: (251, 191, 36), 2: (248, 82, 82)}   # moving, slow, standing
LEVELS = (2.0, 1.0, 0.75)    # the base map's scales (px per metre)
CHIPS_W = 500           # the chips' room above the radar: the picture keeps one size whatever their numbers
CELL_M = 20.0


def _bearing_name(h: float) -> str:
    return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")[int(((h % 360) + 22.5) // 45) % 8]


class Radar:
    def __init__(self, router, places: list | None = None):
        self.r = router
        self.places = [p for p in places or () if "label" in p]
        xs = [x for rd in router.roads.values() for x, _ in rd.shape]
        ys = [y for rd in router.roads.values() for _, y in rd.shape]
        m = 60.0
        self.x0, self.x1, self.y0, self.y1 = min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m
        self.bases = {k: self._base(k) for k in LEVELS}
        self._grid()
        self.f_card = hudkit.font("BarlowCondensed-Bold.ttf", 15)
        self.f_small = hudkit.font("Barlow-SemiBold.ttf", 11)
        self.f_head = hudkit.font("BarlowCondensed-Bold.ttf", 14)
        self._mask = self._circle_mask(D)
        self._cone = self._view_cone(D)
        self._icons: dict = {}
        self.R = 220.0      # the radius shown (m), eased to the camera's height

    # ---------------------------------------------------------------- the town, drawn once
    def _px(self, k: float, x: float, y: float) -> tuple[float, float]:
        return ((x - self.x0) * k, (self.y1 - y) * k)

    def _base(self, k: float) -> pygame.Surface:
        w, h = int((self.x1 - self.x0) * k), int((self.y1 - self.y0) * k)
        s = pygame.Surface((w, h))
        s.fill(LAND)
        step = int(50 * k)       # a faint 50 m grid
        for x in range(0, w, step):
            pygame.draw.line(s, (20, 26, 39), (x, 0), (x, h))
        for y in range(0, h, step):
            pygame.draw.line(s, (20, 26, 39), (0, y), (w, y))
        for p in self.places:    # the places' buildings in their kind's colour
            col = kind_colour(p["kind"])
            polys = p.get("blocks") or ([p["outline"]] if len(p.get("outline") or ()) > 2 else [])
            for poly in polys:
                pts = [self._px(k, *q) for q in poly]
                if len(pts) > 2:
                    pygame.draw.polygon(s, tuple(int(a + (b - a) * 0.22) for a, b in zip(LAND, col)), pts)
                    pygame.draw.polygon(s, tuple(int(a + (b - a) * 0.5) for a, b in zip(LAND, col)), pts,
                                        max(1, int(k)))
        for widen, col in ((2.5, ROAD_CASE), (0.0, ROAD)):
            for rd in self.r.roads.values():
                pts = [self._px(k, *p) for p in rd.shape]
                if len(pts) > 1:
                    wm = max(int(getattr(rd, "lanes", 1) or 1), 1) * 3.3 + widen
                    wpx = max(2, int(wm * k))
                    pygame.draw.lines(s, col, False, pts, wpx)
                    for q in (pts[0], pts[-1]):
                        pygame.draw.circle(s, col, q, wpx // 2)
        for rd in self.r.roads.values():   # a thin centre line
            pts = [self._px(k, *p) for p in rd.shape]
            if len(pts) > 1 and k >= 1.5:
                pygame.draw.lines(s, ROAD_LINE, False, pts, 1)
        return s

    def _grid(self) -> None:
        """Road sample points in cells, to put each car on its road (the live traffic)."""
        self.cells: dict = {}
        self._near: dict = {}
        for rid, rd in self.r.roads.items():
            for (ax, ay), (bx, by) in zip(rd.shape, rd.shape[1:]):
                n = max(1, int(math.dist((ax, ay), (bx, by)) / 5.0))
                for i in range(n + 1):
                    x, y = ax + (bx - ax) * i / n, ay + (by - ay) * i / n
                    self.cells.setdefault((int(x // CELL_M), int(y // CELL_M)), []).append((x, y, rid))

    def road_of(self, x: float, y: float) -> str | None:
        """The road nearest a point (within 8 m), remembered per 3 m square: the search runs once a square."""
        key = (int(x // 3.0), int(y // 3.0))
        try:
            return self._near[key]
        except KeyError:
            rid = self._near[key] = self._search(key[0] * 3.0 + 1.5, key[1] * 3.0 + 1.5)
            return rid

    def _search(self, x: float, y: float) -> str | None:
        cx, cy = int(x // CELL_M), int(y // CELL_M)
        best, rid = 64.0, None
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for px, py, r in self.cells.get((cx + i, cy + j), ()):
                    d = (px - x) ** 2 + (py - y) ** 2
                    if d < best:
                        best, rid = d, r
        return rid

    @staticmethod
    def _circle_mask(d: int) -> pygame.Surface:
        big = pygame.Surface((d * 4, d * 4), pygame.SRCALPHA)
        pygame.draw.circle(big, (255, 255, 255, 255), (d * 2, d * 2), d * 2)
        return pygame.transform.smoothscale(big, (d, d))

    @staticmethod
    def _view_cone(d: int) -> pygame.Surface:
        """Where the camera looks: a soft wedge up from the centre (its 90 degree view)."""
        s = pygame.Surface((d, d), pygame.SRCALPHA)
        c = d / 2
        for i in range(24, 0, -1):
            r = c * 0.92 * i / 24
            a = int(46 * (1 - i / 24) + 6)
            pts = [(c, c)] + [(c + r * math.sin(math.radians(t)), c - r * math.cos(math.radians(t)))
                              for t in range(-45, 46, 5)]
            pygame.draw.polygon(s, (190, 225, 255, a), pts)
        return s

    def _icon(self, kind: str, size: int) -> pygame.Surface:
        key = (kind, size)
        s = self._icons.get(key)
        if s is None:
            n = size * 4
            big = pygame.Surface((n + 8, n + 8), pygame.SRCALPHA)
            pygame.draw.circle(big, (0, 0, 0, 120), (n / 2 + 4, n / 2 + 6), n / 2 + 2)
            Painter(big, "pygame").icon(kind, n / 2 + 4, n / 2 + 4, n, ring=(255, 255, 255))
            s = self._icons[key] = pygame.transform.smoothscale(big, (size + 2, size + 2))
        return s

    # ---------------------------------------------------------------- each picture
    # ---------------------------------------------------------------- the moving part, 5 times a second
    WORLD_S = 0.2    # the traffic layer's age at most (s)

    def _world(self, cx: float, cy: float, R: float, cars, t: float) -> None:
        """The town round the camera, north up, with the traffic on it (the roads coloured by their cars' speed):
        drawn 5 times a second, a margin wide enough for the camera to move and the zoom to change before the
        next. Each picture only turns, scales and shifts it (render)."""
        s = (D / 2) / R
        k = min((lv for lv in LEVELS if lv >= s), default=LEVELS[0])   # the base map just sharper than the screen
        half = 1.6 * R                      # metres each side of the camera
        Lk = int(2 * half * k)
        wx0, wy1 = cx - half, cy + half
        bx, by = self._px(k, wx0, wy1)
        src = pygame.Surface((Lk, Lk))
        src.fill(LAND)
        src.blit(self.bases[k], (0, 0), pygame.Rect(int(round(bx)), int(round(by)), Lk, Lk))
        wx0 = self.x0 + int(round(bx)) / k   # (the pixel grid's own corner)
        wy1 = self.y1 - int(round(by)) / k
        # the layer at the screen's scale now: each picture then only turns it (and scales it by the zoom eased since)
        L = int(Lk * s / k)
        w = pygame.transform.smoothscale(src, (L, L)) if L != Lk else src
        k = L / Lk * k
        g = 1.0                              # layer pixels per screen pixel

        def W(x, y):
            return (x - wx0) * k, (wy1 - y) * k

        per: dict = {}
        for _vid, x, y, v in cars:
            if abs(x - cx) > half or abs(y - cy) > half:
                continue
            rid = self.road_of(x, y)
            if rid is not None:
                per.setdefault(rid, []).append(v)
        flow = pygame.Surface((L, L), pygame.SRCALPHA)
        lw = max(2, int(max(3, int(4.5 * s)) * g))
        for rid, vs in per.items():
            rd = self.r.roads.get(rid)
            if rd is None:
                continue
            ref = max(getattr(rd, "speed", 13.9) or 13.9, 5.0)
            m = sum(vs) / len(vs)
            lev = 2 if m < 1.0 else 1 if m < 0.45 * ref else 0
            pygame.draw.lines(flow, (*FLOW[lev], (95, 165, 185)[lev]), False, [W(*p) for p in rd.shape],
                              lw + (int(g) if lev > 0 else 0))
        w.blit(flow, (0, 0))      # (the cars themselves: every picture, render)
        self._w = {"surf": w, "k": k, "R": R, "x0": wx0, "y1": wy1, "cx": cx, "cy": cy, "t": t}

    def render(self, free: dict, t: float) -> pygame.Surface:
        """One picture: the traffic layer (redrawn 5 times a second) turned heading-up, scaled and shifted to where
        the camera is now, then what must stay upright or move every picture (icons, alerts, the camera's marker)."""
        self._count(t)
        cx, cy, heading = free.get("cam") or ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2, 0.0)
        z = free.get("z", 60.0)
        want = min(max(110.0 + 1.5 * max(z, 0.0), 120.0), 330.0)
        dt = min(max(t - getattr(self, "_t", t - 0.2), 0.0), 0.5)
        self._t = t
        self.R += (want - self.R) * (1.0 - math.exp(-dt * 5.0))   # eased by time, not by pictures
        R = self.R
        s = (D / 2) / R
        h = heading % 360.0
        wl = getattr(self, "_w", None)
        if (wl is None or t - wl["t"] > self.WORLD_S or t < wl["t"] or abs(R / wl["R"] - 1.0) > 0.12
                or math.hypot(cx - wl["cx"], cy - wl["cy"]) > 0.3 * wl["R"]):
            self._world(cx, cy, R, free.get("cars") or (), t)
            wl = self._w
        k = wl["k"]
        bx, by = (cx - wl["x0"]) * k, (wl["y1"] - cy) * k
        L = int(2 * R * 1.04 * k) + 4   # the disc only: whatever the turn, the rest is masked off
        ox, oy = int(bx - L / 2), int(by - L / 2)
        crop = wl["surf"].subsurface(pygame.Rect(ox, oy, L, L).clip(wl["surf"].get_rect()))
        if crop.get_size() != (L, L):   # (at the layer's edge: never with its margin, but safe)
            full = pygame.Surface((L, L))
            full.fill(LAND)
            full.blit(wl["surf"], (0, 0), pygame.Rect(ox, oy, L, L))
            crop = full
        g = s / k
        rot = pygame.transform.rotozoom(crop, h, g)
        # the camera is (fx, fy) layer pixels off the crop's centre: that offset, turned and scaled, is taken off
        fx, fy = bx - (ox + L / 2), by - (oy + L / 2)
        sh, ch = math.sin(math.radians(h)), math.cos(math.radians(h))
        ux, uy = (fx * ch + fy * sh) * g, (-fx * sh + fy * ch) * g
        disc = pygame.Surface((D, D), pygame.SRCALPHA)
        disc.blit(rot, (D / 2 - rot.get_width() / 2 - ux, D / 2 - rot.get_height() / 2 - uy))
        c = D / 2

        def P(x, y):
            dx, dy = x - cx, y - cy
            return c + s * (dx * ch - dy * sh), c - s * (dx * sh + dy * ch)

        # the cars, where the town's picture shows them now (every picture: in the 5-a-second layer they jumped)
        lim = (c - 3) ** 2
        for _vid, x, y, v in free.get("cars") or ():
            dx, dy = x - cx, y - cy
            qx, qy = s * (dx * ch - dy * sh), -s * (dx * sh + dy * ch)
            if qx * qx + qy * qy > lim:
                continue
            col = FLOW[2] if v < 1.0 else FLOW[1] if v < 5.0 else (214, 224, 240)
            pygame.draw.circle(disc, (0, 0, 0, 110), (c + qx + 0.6, c + qy + 1.0), 2.8)
            pygame.draw.circle(disc, col, (c + qx, c + qy), 2.3)
        # the places: their icons, upright
        isz = 18 if R < 200 else 14
        for p in self.places:
            q = P(*p["label"])
            if math.dist(q, (c, c)) < c - 14:
                ic = self._icon(p["kind"], isz)
                disc.blit(ic, (q[0] - ic.get_width() / 2, q[1] - ic.get_height() / 2))
        pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 1.2)
        blips = []
        for a in free.get("alerts") or ():
            rd = self.r.roads.get(a.get("route"))
            if rd is None:
                continue
            mx, my = rd.shape[len(rd.shape) // 2]
            kind = "accident" if a.get("type") == "accident" else "forecast"
            q = P(mx, my)
            if math.dist(q, (c, c)) < c - 18:
                col = hudkit.ACCENT[kind]
                pygame.draw.circle(disc, (*col, int(90 * (1 - pulse))), q, 10 + 10 * pulse, 2)
                disc.blit(hudkit.icon(kind, 20), (q[0] - 10, q[1] - 10))
            else:
                blips.append((kind, q, math.dist((mx, my), (cx, cy))))
        cand = free.get("cand")
        if cand is not None:
            q = P(cand[1], cand[2])
            if math.dist(q, (c, c)) < c - 10:
                gr = (52, 211, 153)
                pygame.draw.circle(disc, (*gr, int(120 + 100 * pulse)), q, 8 + 3 * pulse, 2)
                pygame.draw.circle(disc, gr, q, 3.5)
        # where the camera looks, and the camera
        disc.blit(self._lens(), (0, 0))   # where the camera looks, the camera, the shading towards the edge
        disc.blit(self._mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.disc = disc   # (for the tests)
        return self._frame(disc, h, R, s, blips, free, t)

    def _count(self, t: float) -> None:
        """How many pictures a second (the viewer's log, every 10 s): how smooth it turns."""
        n, t0 = getattr(self, "_n", (0, t))
        if t - t0 >= 10.0:
            import logging

            logging.getLogger("freefly").info("radar: %.1f pictures a second", n / (t - t0))
            n, t0 = 0, t
        self._n = (n + 1, t0)

    def _lens(self) -> pygame.Surface:
        ln = getattr(self, "_ln", None)
        if ln is None:
            ln = self._ln = self._cone.copy()
            c = D / 2
            ln.blit(self._marker(), (c - 12, c - 14))
            ln.blit(self._vignette(), (0, 0))
        return ln

    def _marker(self) -> pygame.Surface:
        m = getattr(self, "_mk", None)
        if m is None:
            m = self._mk = pygame.Surface((24, 26), pygame.SRCALPHA)
            c, cy = 12, 14
            arrow = [(c, cy - 11), (c + 8, cy + 8), (c, cy + 3), (c - 8, cy + 8)]
            pygame.draw.polygon(m, (0, 0, 0, 120), [(x + 1, y + 2) for x, y in arrow])
            pygame.draw.polygon(m, (255, 255, 255), arrow)
            pygame.draw.polygon(m, (96, 165, 250), arrow, 2)
        return m

    def _vignette(self) -> pygame.Surface:
        v = getattr(self, "_vig", None)
        if v is None:
            v = pygame.Surface((D, D), pygame.SRCALPHA)
            for i in range(30):
                r = D / 2 - i
                pygame.draw.circle(v, (6, 8, 14, int(5 * (30 - i) * 0.9)), (D / 2, D / 2), r, 1)
            self._vig = v
        return v

    def _frame(self, disc, h: float, R: float, s: float, blips, free: dict, t: float) -> pygame.Surface:
        """The bezel with the compass turning, the heading, the scale, the blips, the chips above."""
        m = ui.MARGIN if ui.ALPHA else 0
        full = D + 2 * BEZEL
        head = hudkit.chips(self._stats(free))
        hint = None   # (the F prompt is the viewer's own, bottom middle)
        W = max(full, head.get_width(), CHIPS_W) + 2 * m + 8   # (one size: the viewer redraws the ring alone)
        H = head.get_height() + 34 + full + (hint.get_height() - 2 * m + 6 if hint else 0) + 2 * m + 6
        out = pygame.Surface((W, H), pygame.SRCALPHA)
        ox = W - m - full - 4
        oy = m + head.get_height() + 34
        C = (ox + full / 2, oy + full / 2)
        # the part that changes with every picture (the ring, its heading and scale), for the viewer to send alone
        self.ring_box = pygame.Rect(ox - 22, oy - 30, full + 44, full + 52).clip(out.get_rect())
        rg = self._ring(full, h)   # (with its soft round shadow)
        out.blit(rg, (ox - (rg.get_width() - full) // 2, oy - (rg.get_height() - full) // 2))
        out.blit(disc, (ox + BEZEL, oy + BEZEL))
        return self._frame_rest(out, C, full, oy, h, s, blips, head, hint, W, m)

    def _shadow(self, full: int) -> pygame.Surface:
        sh = getattr(self, "_sh", None)
        if sh is None:
            sh = pygame.Surface((full // 4 + 8, full // 4 + 8), pygame.SRCALPHA)
            pygame.draw.circle(sh, (0, 0, 0, 150), (sh.get_width() / 2, sh.get_height() / 2 + 1), full / 8)
            sh = self._sh = pygame.transform.smoothscale(sh, (full + 32, full + 32))
        return sh

    def _ring(self, full: int, h: float) -> pygame.Surface:
        """The bezel: dark glass ring, ticks every 15 degrees, all one weight (N, E, S and W have their letters), so the
        ring repeats every 15 degrees: 15 turns a degree apart are drawn once and kept."""
        rings = self.__dict__.setdefault("_rings", {})
        key = (int(round(h)) % 15, ui.ALPHA)
        if key in rings:
            return rings[key]
        h = float(key[0])
        ring = pygame.Surface((full * 2, full * 2), pygame.SRCALPHA)
        rc = full
        pygame.draw.circle(ring, (*hudkit.INK, 236), (rc, rc), rc)
        pygame.draw.circle(ring, (255, 255, 255, 40), (rc, rc), rc, 2)
        pygame.draw.circle(ring, (255, 255, 255, 26), (rc, rc), rc - BEZEL * 2 + 1, 2)
        for b in range(0, 360, 15):
            a = math.radians(b - h)
            major = False   # one weight: the turns repeat every 15 degrees (N E S W have their letters)
            r0 = rc - (12 if major else 7)
            r1 = rc - 3
            col = (255, 255, 255, 170 if major else 70)
            pygame.draw.line(ring, col, (rc + r0 * math.sin(a), rc - r0 * math.cos(a)),
                             (rc + r1 * math.sin(a), rc - r1 * math.cos(a)), 3 if major else 2)
        ring = pygame.transform.smoothscale(ring, (full, full))
        if ui.ALPHA:   # the soft round shadow under it
            sh = self._shadow(full)
            both = pygame.Surface(sh.get_size(), pygame.SRCALPHA)
            both.blit(sh, (0, 4))
            both.blit(ring, ((sh.get_width() - full) // 2, (sh.get_height() - full) // 2))
            ring = both
        rings[key] = ring
        return ring

    def _frame_rest(self, out, C, full: int, oy: float, h: float, s: float, blips, head, hint, W: int, m: int):
        for b, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(b - h)
            r = full / 2 - BEZEL / 2
            col = (255, 90, 90) if name == "N" else TEXT
            g = hudkit.text(self.f_card, name, col)
            p = (C[0] + r * math.sin(a), C[1] - r * math.cos(a))
            pygame.draw.circle(out, (*hudkit.INK, 255), p, 9)
            out.blit(g, (p[0] - g.get_width() / 2, p[1] - g.get_height() / 2))
        # blips: what is off the map, on its edge, with its distance (the nearest of those in one direction)
        kept = []
        for b in sorted(blips, key=lambda b: (b[0] != "accident", b[2])):
            if all(abs(math.atan2(b[1][1] - D / 2, b[1][0] - D / 2) - math.atan2(k[1][1] - D / 2, k[1][0] - D / 2))
                   > 0.25 for k in kept):
                kept.append(b)
        for kind, q, dist in kept:
            dx, dy = q[0] - D / 2, q[1] - D / 2
            a = math.atan2(dy, dx)
            r = D / 2 - 4
            p = (C[0] + r * math.cos(a), C[1] + r * math.sin(a))
            out.blit(hudkit.icon(kind, 22), (p[0] - 11, p[1] - 11))
            tip = (C[0] + (r + 14) * math.cos(a), C[1] + (r + 14) * math.sin(a))
            col = hudkit.ACCENT[kind]
            pygame.draw.polygon(out, col, [tip, (p[0] + 9 * math.cos(a + 1.9), p[1] + 9 * math.sin(a + 1.9)),
                                           (p[0] + 9 * math.cos(a - 1.9), p[1] + 9 * math.sin(a - 1.9))])
            d = hudkit.text(self.f_small, f"{dist:.0f} m" if dist < 1000 else f"{dist / 1000:.1f} km", TEXT)
            lab = (p[0] - 22 * math.cos(a) - d.get_width() / 2, p[1] - 22 * math.sin(a) - d.get_height() / 2)
            out.blit(d, lab)
        # the heading, top of the ring
        hd = hudkit.tracked(self.f_head, f"{_bearing_name(h)}  {int(round(h)) % 360:03d}°", TEXT, 1.0)
        pill = hudkit.glass(hd.get_width() + 22, 22, radius=11, alpha=236).copy()
        pill.blit(hd, (11, (22 - hd.get_height()) // 2))
        out.blit(pill, (C[0] - pill.get_width() / 2, oy - 28))
        pygame.draw.polygon(out, TEXT, [(C[0] - 6, oy - 5), (C[0] + 6, oy - 5), (C[0], oy + 3)])
        # the scale, bottom of the ring
        nice = next(v for v in (25, 50, 100, 200, 250, 500) if v * s >= 50)
        bar = int(nice * s)
        sc = hudkit.text(self.f_small, f"{nice} m", TEXT)
        sw = bar + sc.get_width() + 26
        spill = hudkit.glass(sw, 20, radius=10, alpha=236).copy()
        pygame.draw.line(spill, TEXT, (10, 13), (10 + bar, 13), 2)
        pygame.draw.line(spill, TEXT, (10, 9), (10, 13), 2)
        pygame.draw.line(spill, TEXT, (10 + bar, 9), (10 + bar, 13), 2)
        spill.blit(sc, (16 + bar, (20 - sc.get_height()) // 2))
        out.blit(spill, (C[0] - sw / 2, oy + full - 12))
        # above: what the town is doing; below: the car F would follow
        out.blit(head, (W - m - 4 - head.get_width(), m))
        if hint is not None:
            out.blit(hint, (W - hint.get_width() + m - 4 - (full - hint.get_width() + 2 * m) / 2, oy + full + 6 - m))
        return out

    @staticmethod
    def _stats(free: dict) -> list:
        cars = free.get("cars") or ()
        live = free.get("places") or {}
        why = live.get("why") or {}
        items = [("Town05 live", (125, 211, 252))]
        if why:
            red = why.get("red_signal", 0)
            queued = why.get("total", 0) - red - why.get("depot", 0)
            parked = sum((live.get("occ") or {}).values()) + why.get("depot", 0)
            items += [(f"{len(cars) - why.get('total', 0)} moving", None), (f"{red} at red", None),
                      (f"{queued} queued", None), (f"{parked} parked", None)]
        else:
            items += [(f"{len(cars)} cars", None), (f"{sum(1 for x in cars if x[3] < 1.0)} standing", None)]
        return items

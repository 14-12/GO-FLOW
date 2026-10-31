"""ARTERIA, the free-fly viewer's loading screen: a window of its own, over CARLA, from before CARLA starts until the town
is ready.

    python -m cosim.splash [--scenario town05]          (tools/town05_freefly.bat starts it with pythonw)

Town05's real road network (SUMO's) draws itself from the centre out as the load goes on, with traffic running
along it as points of light that follow the roads' real connections, and the signals lit at the junctions. Beside
it: the stages of the load, ticked off as they finish (the one under way with a spinner and its own progress), the
overall progress, the town's counts as they become known, and the keys.

It reads what the viewer is doing from out/freefly/loading.json (`write_status`, a few times a second): a process
of its own, because the viewer's Python is held for seconds at a time while CARLA loads the map (CARLA's calls
keep Python's lock while they wait), and a loading screen drawn there stood still exactly then. When the viewer
says it is ready, it fades out and puts CARLA's window in front; on an error it says so and closes; without news
from the viewer for STALE_S it closes by itself. Esc closes it (the loading goes on).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "out" / "freefly" / "loading.json"
FONTS = ROOT / "assets" / "fonts"
STALE_S = 240.0        # no news from the viewer for this long (after it started): close
STALE_LAST_S = 12.0    # the same once the viewer is at its last stage (it writes several times a second until ready)
NO_VIEWER_S = 600.0    # the viewer never started (CARLA did not come up): close
FADE_IN_S, FADE_OUT_S = 0.5, 0.8

# the stages, in order: (start of the viewer's stage text, label, expected seconds, weight in the progress)
STAGES = [
    ("", "Building the world", 55.0, 26.0),
    ("Starting", "Opening the city", 3.0, 1.0),
    ("Reading the road network", "Drawing the streets", 3.0, 2.0),
    ("Connecting to CARLA", "Switching on the lights", 3.0, 2.0),
    ("Loading ", "Raising the skyline", 20.0, 14.0),
    ("Starting SUMO", "Waking the traffic", 5.0, 4.0),
    ("Filling the town", "Filling the streets", 20.0, 18.0),
    ("Handing the traffic to CARLA", "Setting the cars in motion", 4.0, 4.0),
    ("Parking the vehicles", "Parking the cars of the day", 8.0, 7.0),
    ("Putting the pedestrians", "Bringing people outside", 4.0, 4.0),
    ("Putting up the pedestrian lights", "Lighting the crosswalks", 3.0, 3.0),
    ("Placing the traffic cameras", "Mounting the street cameras", 4.0, 4.0),
    ("Choosing the camera views", "Finding the best view", 5.0, 4.0),
    ("Ready", "Welcome to the city", 0.0, 0.0),
]

NAME = "ARTERIA"
TAGLINE = "The living city, in perfect flow."
EYEBROW = "A CITY THAT THINKS FOR ITSELF"

TIPS = [
    ("F8", "Jump to the next intersection"),
    ("H", "Watch a street camera, live"),
    ("F5", "Pause time      F6 / F7  travel an hour back or ahead"),
    ("Z Q S D", "Fly around (Shift: faster)      E / A  rise and descend      right mouse  look around"),
    ("F9", "Hide the clock      Esc  leave (the city remembers the time)"),
]

# colours
BG_TOP, BG_BOTTOM = (10, 14, 26), (3, 4, 9)
ROAD_DIM, ROAD = (14, 40, 78), (70, 170, 255)
AMBER, WHITE, MUTED, FAINT = (255, 196, 84), (240, 244, 252), (150, 162, 186), (70, 80, 104)
GREEN, RED = (90, 230, 150), (255, 92, 84)


# ------------------------------------------------------------------------------------------- the viewer's side
def write_status(stage: str, progress: float = -1.0, counts: dict | None = None, clock: str = "",
                 ready: bool = False, error: str | None = None, path: Path = STATUS) -> None:
    """For the viewer (cosim/freefly.py): what it is doing now. Written whole and renamed into place, so the
    splash never reads half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"t": time.time(), "stage": stage, "progress": progress, "counts": counts or {},
                               "clock": clock, "ready": ready, "error": error}), encoding="utf-8")
    # Windows refuses to replace a file another process has open: the splash reading it at that instant. The write
    # saying "ready" was lost that way once (27 Oct) and the loading screen stayed up over the running town for
    # STALE_S: try again for up to half a second
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == 19:
                raise
            time.sleep(0.025)


def read_status(since: float, path: Path = STATUS) -> dict | None:
    """The viewer's last status, if written after `since` (an older file is from an earlier run)."""
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - not there yet, or being replaced
        return None
    return st if st.get("t", 0.0) >= since else None


def stage_index(text: str) -> int:
    """Which of STAGES the viewer's stage text is (the first, CARLA, has no text of its own)."""
    best, best_len = 1, -1
    for i, (prefix, *_rest) in enumerate(STAGES[1:], start=1):
        if text.startswith(prefix) and len(prefix) > best_len:  # the longest: "Starting SUMO" is not "Starting"
            best, best_len = i, len(prefix)
    return best


class Progress:
    """Where the load is: the stage under way, when each began, and the overall share done (by the stages'
    weights; within a stage, its own progress when the viewer gives one, else its time against the time it
    usually takes, never more than 90 % of it). The share shown only ever grows, and eases towards its target."""

    def __init__(self, now: float):
        self.cur = 0
        self.began = {0: now}
        self.done_at: dict[int, float] = {}
        self.sub = -1.0
        self.shown = 0.0
        self.total = sum(w for *_x, w in STAGES)

    def update(self, st: dict | None, now: float) -> None:
        if st is None:
            return
        i = stage_index(st.get("stage", ""))
        if st.get("ready"):
            i = len(STAGES) - 1
        if i > self.cur:
            for k in range(self.cur, i):
                self.done_at.setdefault(k, now)
                self.began.setdefault(k + 1, now)
            self.cur = i
        self.sub = float(st.get("progress", -1.0))

    def target(self, now: float) -> float:
        if self.cur >= len(STAGES) - 1:
            return 1.0
        done = sum(STAGES[k][3] for k in range(self.cur))
        _, _, expect, w = STAGES[self.cur]
        if self.sub >= 0.0:
            part = min(self.sub, 1.0)
        else:
            el = now - self.began.get(self.cur, now)
            part = 0.9 * (1.0 - math.exp(-el / max(expect, 0.5)))
        return (done + w * part) / self.total

    def step(self, now: float, dt: float) -> float:
        t = self.target(now)
        self.shown = max(self.shown, self.shown + (t - self.shown) * min(1.0, dt * 3.0))
        return self.shown


# ------------------------------------------------------------------------------------------------ the map
class Network:
    """The SUMO network as polylines in screen space: passenger roads (with their lane count and their
    successors), the signals' junctions."""

    def __init__(self, net_file: Path, rect: tuple[int, int, int, int]):
        import sumolib

        net = sumolib.net.readNet(str(net_file), withInternal=True)
        x0, y0, x1, y1 = net.getBoundary()
        rx, ry, rw, rh = rect
        k = min(rw / (x1 - x0), rh / (y1 - y0))
        ox, oy = rx + (rw - (x1 - x0) * k) / 2, ry + (rh - (y1 - y0) * k) / 2
        self.scale = k

        def to_screen(x, y):
            return ox + (x - x0) * k, oy + (y1 - y) * k  # SUMO's y points up, the screen's down

        cx, cy = to_screen((x0 + x1) / 2, (y0 + y1) / 2)
        self.centre = (cx, cy)
        self.edges = []  # (points, lanes, distance of its middle from the centre, id)
        index = {}
        self.inner = []  # the junctions' own lanes (the turns), drawn but not driven
        for e in net.getEdges():
            if e.getFunction() == "internal":
                if e.allows("passenger"):
                    for ln in e.getLanes():
                        self.inner.append([to_screen(x, y) for x, y in ln.getShape()])
                continue
            if not e.allows("passenger"):
                continue
            pts = [to_screen(x, y) for x, y in e.getShape()]
            mx, my = pts[len(pts) // 2]
            index[e.getID()] = len(self.edges)
            self.edges.append((pts, e.getLaneNumber(), math.hypot(mx - cx, my - cy), e))
        self.radius = max((d for _, _, d, _ in self.edges), default=1.0) + 20.0
        self.next = []
        for pts, _lanes, _d, e in self.edges:
            outs = [index[o.getID()] for o in e.getOutgoing() if o.getID() in index]
            self.next.append(outs)
        self.lengths = [sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(p, p[1:], strict=False))
                        for p, *_r in self.edges]
        self.signals = [to_screen(*n.getCoord()) for n in net.getNodes() if n.getType() == "traffic_light"]
        self.edges_sorted = sorted(range(len(self.edges)), key=lambda i: self.edges[i][2])

    def point(self, i: int, s: float) -> tuple[float, float]:
        """The point `s` pixels along edge i."""
        pts = self.edges[i][0]
        for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
            seg = math.hypot(bx - ax, by - ay)
            if s <= seg:
                k = s / max(seg, 1e-6)
                return ax + (bx - ax) * k, ay + (by - ay) * k
            s -= seg
        return pts[-1]


class Car:
    __slots__ = ("edge", "s", "v", "hue")

    def __init__(self, edge: int, s: float, v: float, hue: int):
        self.edge, self.s, self.v, self.hue = edge, s, v, hue


# ------------------------------------------------------------------------------------------------ drawing
def _font(name: str, size: int):
    import pygame

    p = FONTS / name
    return pygame.font.Font(str(p) if p.is_file() else None, size)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _spaced(font, text: str, colour, spacing: float):
    import pygame

    glyphs = [font.render(ch, True, colour) for ch in text]
    w = sum(g.get_width() for g in glyphs) + int(spacing * max(len(glyphs) - 1, 0))
    s = pygame.Surface((max(w, 1), font.get_height()), pygame.SRCALPHA)
    x = 0
    for g in glyphs:
        s.blit(g, (x, 0))
        x += g.get_width() + spacing
    return s


def _glow_dot(radius: int, colour) -> "object":
    """A soft round light, for additive drawing."""
    import pygame

    size = radius * 2 + 1
    s = pygame.Surface((size, size))
    for r in range(radius, 0, -1):
        k = (1.0 - r / (radius + 1)) ** 2.2
        pygame.draw.circle(s, tuple(int(c * k) for c in colour), (radius, radius), r)
    return s


class Splash:
    def __init__(self, screen, scenario: str, t0: float):
        import pygame

        self.pg = pygame
        self.screen = screen
        self.W, self.H = W, H = screen.get_size()
        self.t0 = t0
        self.scenario = scenario
        u = H / 1080.0  # sizes for a 1080-line screen, scaled
        self.u = u
        self.f_eyebrow = _font("Barlow-SemiBold.ttf", max(11, int(15 * u)))
        self.f_title = _font("BarlowCondensed-SemiBold.ttf", int(132 * u))
        self.f_title2 = _font("BarlowCondensed-Medium.ttf", int(132 * u))
        self.f_sub = _font("Barlow-Medium.ttf", max(12, int(19 * u)))
        self.f_pct = _font("BarlowCondensed-SemiBold.ttf", int(104 * u))
        self.f_stage = _font("Barlow-Medium.ttf", max(12, int(19 * u)))
        self.f_small = _font("Barlow-SemiBold.ttf", max(10, int(14 * u)))
        self.f_count = _font("BarlowCondensed-SemiBold.ttf", int(46 * u))
        self.f_key = _font("BarlowCondensed-SemiBold.ttf", max(12, int(20 * u)))
        self.f_tip = _font("Barlow-Medium.ttf", max(11, int(17 * u)))
        # the layout: the map on the left, the panel on the right
        self.map_rect = (int(W * 0.045), int(H * 0.09), int(W * 0.51), int(H * 0.80))
        self.px = int(W * 0.625)
        self.pw = int(W * 0.335)
        self.bg = self._background()
        self.net = Network(ROOT / "scenarios" / scenario / f"{scenario}.net.xml", self.map_rect)
        self.roads = pygame.Surface((W, H))
        self.roads.fill((0, 0, 0))
        self.glow_layer = pygame.Surface((W, H))
        self.glow_layer.fill((0, 0, 0))
        self.inner_done: set[int] = set()
        self.ring = pygame.Surface((int(64 * u), int(64 * u)))
        self.revealed = 0
        self.rng = random.Random(5)
        self.cars: list[Car] = []
        def comet(colour):  # a head and a fading tail
            return [_glow_dot(max(2, int(r * u)), tuple(int(c * k) for c in colour))
                    for r, k in ((7, 1.0), (5, 0.55), (4, 0.32), (3, 0.18))]

        self.dot_w = comet((255, 240, 210))   # headlights
        self.dot_r = comet((255, 60, 50))     # tail lights
        self.dot_a = comet((255, 176, 60))    # buses, deliveries
        self.sig_g = _glow_dot(max(6, int(16 * u)), (50, 255, 140))
        self.sig_r = _glow_dot(max(6, int(16 * u)), (255, 70, 60))
        self.cache: dict = {}
        self.prog = Progress(t0)
        self.status: dict | None = None
        self.counts_shown: dict[str, float] = {}

    # -- static layers
    def _background(self):
        pg = self.pg
        W, H = self.W, self.H
        small = pg.Surface((64, 36))
        for y in range(36):
            for x in range(64):
                g = _lerp(BG_TOP, BG_BOTTOM, y / 35)
                # a vignette, and a faint cool light behind the map
                dx, dy = (x - 20) / 32, (y - 18) / 18
                light = max(0.0, 1.0 - math.hypot(dx, dy)) ** 2 * 0.55
                vx, vy = (x - 32) / 32, (y - 18) / 18
                vig = min(1.0, (vx * vx + vy * vy) ** 1.2 * 0.6)
                c = tuple(min(255, int(g[i] * (1 - vig) + (14, 30, 60)[i] * light)) for i in range(3))
                small.set_at((x, y), c)
        bg = pg.transform.smoothscale(small, (W, H))
        # a fine dot grid
        step = max(18, int(26 * self.H / 1080))
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                bg.set_at((x, y), (26, 32, 48))
        # the panel's separator
        pg.draw.line(bg, (30, 38, 58), (self.px - int(W * 0.022), int(H * 0.08)), (self.px - int(W * 0.022),
                                                                                   int(H * 0.92)), 1)
        return bg

    def _text(self, key, make):
        s = self.cache.get(key)
        if s is None:
            s = self.cache[key] = make()
        return s

    # -- the map
    def reveal(self, share: float) -> None:
        """Draw the roads up to `share` of the network (from the centre out) onto the road layer."""
        pg = self.pg
        want = int(len(self.net.edges) * min(1.0, share))
        while self.revealed < want:
            i = self.net.edges_sorted[self.revealed]
            pts, lanes, *_r = self.net.edges[i]
            if len(pts) >= 2:
                w = max(2, int(lanes * 1.1 * self.u + 1))
                for extra, col in ((14, (5, 13, 28)), (9, (7, 20, 42)), (5, (10, 30, 62))):
                    ww = w + int(extra * self.u)
                    self._thick(self.glow_layer, pts, ww, col)
                self._thick(self.roads, pts, w + 2, ROAD_DIM)
                pg.draw.lines(self.roads, (24, 74, 140), False, pts, max(1, w - 1))
                pg.draw.aalines(self.roads, ROAD, False, pts)
            self.revealed += 1
            if len(self.cars) < 420 and self.rng.random() < 0.75:
                r = self.rng.random()
                self.cars.append(Car(i, self.rng.uniform(0, self.net.lengths[i]), self.rng.uniform(30, 70) * self.u,
                                     0 if r < 0.62 else (1 if r < 0.92 else 2)))

    def _thick(self, surf, pts, w: int, col) -> None:
        """A wide polyline with round joins (pygame's wide lines leave notches at the bends)."""
        self.pg.draw.lines(surf, col, False, pts, w)
        r = max(1, w // 2)
        for x, y in pts:
            self.pg.draw.circle(surf, col, (int(x), int(y)), r)

    def _draw_inner(self) -> None:
        """The junctions' turning lanes, once the roads around them are drawn."""
        cx, cy = self.net.centre
        rr = self.reveal_radius()
        for k, pts in enumerate(self.net.inner):
            if k in self.inner_done or len(pts) < 2:
                continue
            mx, my = pts[len(pts) // 2]
            if math.hypot(mx - cx, my - cy) <= rr:
                self.inner_done.add(k)
                self.pg.draw.lines(self.roads, (16, 48, 92), False, pts, max(2, int(2 * self.u)))
                self.pg.draw.aalines(self.roads, (40, 110, 190), False, pts)

    def reveal_radius(self) -> float:
        if self.revealed <= 0:
            return 0.0
        i = self.net.edges_sorted[min(self.revealed, len(self.net.edges)) - 1]
        return self.net.edges[i][2]

    def move_cars(self, dt: float) -> None:
        live = set(self.net.edges_sorted[:self.revealed])
        for c in self.cars:
            c.s += c.v * dt
            while c.s > self.net.lengths[c.edge]:
                c.s -= self.net.lengths[c.edge]
                nxt = [n for n in self.net.next[c.edge] if n in live]
                if nxt:
                    c.edge = self.rng.choice(nxt)
                else:
                    c.edge = self.rng.choice(tuple(live))
                    c.s = 0.0

    # -- one picture
    def draw(self, now: float, dt: float) -> None:
        pg, sc, W, H, u = self.pg, self.screen, self.W, self.H, self.u
        share = self.prog.step(now, dt)
        self.reveal(0.08 + 0.92 * share)
        self.move_cars(dt)
        self._draw_inner()
        sc.blit(self.bg, (0, 0))
        sc.blit(self.glow_layer, (0, 0), special_flags=pg.BLEND_ADD)
        sc.blit(self.roads, (0, 0), special_flags=pg.BLEND_ADD)
        # the reveal front: a soft ring going out from the centre
        if share < 0.995:
            r = int(self.reveal_radius())
            if r > 4:
                ring = pg.Surface((2 * r + 8, 2 * r + 8))
                ring.fill((0, 0, 0))
                pulse = 0.6 + 0.4 * math.sin(now * 3.0)
                pg.draw.circle(ring, (int(20 * pulse), int(60 * pulse), int(110 * pulse)), (r + 4, r + 4), r, 2)
                cx, cy = self.net.centre
                sc.blit(ring, (int(cx - r - 4), int(cy - r - 4)), special_flags=pg.BLEND_ADD)
        # traffic: headlights white, tail lights red, a few amber (buses, deliveries)
        trail = max(3.0, 7.0 * u)
        for c in self.cars:
            dots = (self.dot_w, self.dot_r, self.dot_a)[c.hue]
            for j, sprite in enumerate(((dots[0]), dots[1], dots[2], dots[3])):
                if c.s - j * trail < 0:
                    break
                x, y = self.net.point(c.edge, c.s - j * trail)
                sc.blit(sprite, (int(x) - sprite.get_width() // 2, int(y) - sprite.get_height() // 2),
                        special_flags=pg.BLEND_ADD)
        # the signals: green and red in turn
        rr = self.reveal_radius()
        cx, cy = self.net.centre
        for k, (x, y) in enumerate(self.net.signals):
            if math.hypot(x - cx, y - cy) > rr + 10:
                continue
            green = (int(now / 4.0 + k * 0.37)) % 2 == 0
            dot = self.sig_g if green else self.sig_r
            sc.blit(dot, (int(x) - dot.get_width() // 2, int(y) - dot.get_height() // 2), special_flags=pg.BLEND_ADD)
            ph = (now * 0.8 + k * 0.23) % 1.0  # a ring going out
            rad = int((6 + 22 * ph) * u)
            col = (50, 255, 140) if green else (255, 70, 60)
            fade = (1.0 - ph) ** 2 * 0.6
            ring = self.ring
            ring.fill((0, 0, 0))
            half = ring.get_width() // 2
            pg.draw.circle(ring, tuple(int(c * fade) for c in col), (half, half), min(rad, half - 1),
                           max(1, int(1.5 * u)))
            sc.blit(ring, (int(x) - half, int(y) - half), special_flags=pg.BLEND_ADD)
        self._legend(now)
        self._panel(now, share)

    def _legend(self, now: float) -> None:
        pg, sc, u = self.pg, self.screen, self.u
        rx, ry, rw, rh = self.map_rect
        y = ry + rh + int(22 * u)
        x = rx
        items = [((255, 240, 210), "cars"), ((255, 60, 50), "brake lights"), ((255, 176, 60), "buses and deliveries"),
                 ((50, 255, 140), "traffic lights")]
        for col, name in items:
            pg.draw.circle(sc, col, (x + int(5 * u), y + int(8 * u)), max(2, int(4 * u)))
            t = self._text(("lg", name), lambda name=name: self.f_small.render(name, True, MUTED))
            sc.blit(t, (x + int(16 * u), y))
            x += t.get_width() + int(40 * u)
        net = self._text("lgnet", lambda: self.f_small.render("A live map of the city", True, FAINT))
        sc.blit(net, (rx + rw - net.get_width(), y))

    def _panel(self, now: float, share: float) -> None:
        pg, sc, W, H, u = self.pg, self.screen, self.W, self.H, self.u
        x = self.px
        y = int(H * 0.085)
        eb = self._text("eyebrow", lambda: _spaced(self.f_eyebrow, EYEBROW, AMBER, 2.6 * u))
        sc.blit(eb, (x, y))
        y += eb.get_height() + int(6 * u)
        t1 = self._text("t1", lambda: self._title())
        sc.blit(t1, (x - int(4 * u), y))
        y += t1.get_height() - int(10 * u)
        sub = self._text("sub", lambda: self.f_sub.render(TAGLINE, True, MUTED))
        sc.blit(sub, (x, y))
        y += sub.get_height() + int(38 * u)

        # the percentage and the bar
        pct = int(round(share * 100))
        ps = self._text(("pct", pct), lambda: self.f_pct.render(f"{pct}", True, WHITE))
        sc.blit(ps, (x - int(3 * u), y))
        sign = self._text("pctsign", lambda: self.f_count.render("%", True, MUTED))
        sc.blit(sign, (x + ps.get_width() + int(6 * u), y + ps.get_height() - sign.get_height() - int(14 * u)))
        cur = self.prog.cur
        err = self.status.get("error") if self.status else None
        label = "SOMETHING WENT WRONG" if err else ("READY" if cur >= len(STAGES) - 1 else "WAKING THE CITY")
        lb = self._text(("lbl", label), lambda: _spaced(self.f_small, label, RED if err else AMBER, 2.2 * u))
        sc.blit(lb, (x + int(190 * u), y + int(28 * u)))
        clock = (self.status or {}).get("clock", "")
        if clock:
            cl = self._text(("clock", clock), lambda: self.f_stage.render(clock, True, MUTED))
            sc.blit(cl, (x + int(190 * u), y + int(28 * u) + lb.get_height() + int(8 * u)))
        y += ps.get_height() + int(4 * u)
        bw, bh = self.pw, max(4, int(6 * u))
        pg.draw.rect(sc, (30, 36, 54), (x, y, bw, bh), border_radius=bh // 2)
        fw = max(bh, int(bw * share))
        bar = self._text(("bar", fw), lambda: self._bar(fw, bh))
        sc.blit(bar, (x, y))
        head = self.dot_a[0]
        sc.blit(head, (x + fw - head.get_width() // 2, y + bh // 2 - head.get_height() // 2), special_flags=pg.BLEND_ADD)
        y += bh + int(30 * u)

        # the stages: done, under way, to come (a window of them around the current one)
        row = int(31 * u)
        first = max(0, min(cur - 3, len(STAGES) - 1 - 8))
        for k in range(first, min(first + 9, len(STAGES))):
            _, name, _, _ = STAGES[k]
            cy = y + row // 2
            ready = k == len(STAGES) - 1 and k == cur and bool(self.status and self.status.get("ready"))
            if k < cur or ready:
                pg.draw.circle(sc, (50, 190, 120), (x + int(9 * u), cy), int(8 * u))
                pts = [(x + int(5 * u), cy), (x + int(8 * u), cy + int(3 * u)), (x + int(13 * u), cy - int(3 * u))]
                pg.draw.lines(sc, (8, 20, 14), False, pts, max(2, int(2 * u)))
                col = (190, 200, 220)
                el = self.prog.done_at.get(k, now) - self.prog.began.get(k, now)
                detail = f"{el:.0f} s" if el >= 1 else ""
            elif k == cur and not err:
                a0 = now * 5.0
                rect = (x + int(1 * u), cy - int(8 * u), int(16 * u), int(16 * u))
                pg.draw.circle(sc, (40, 46, 66), (x + int(9 * u), cy), int(8 * u), max(2, int(2 * u)))
                pg.draw.arc(sc, AMBER, rect, a0, a0 + 1.9, max(2, int(2.5 * u)))
                col = WHITE
                sub = self.prog.sub
                detail = f"{sub * 100:.0f} %" if sub >= 0 else "…"
                if k == 0:
                    detail = f"{now - self.t0:.0f} s"
            else:
                pg.draw.circle(sc, (44, 50, 70), (x + int(9 * u), cy), int(8 * u), max(1, int(2 * u)))
                col = FAINT
                detail = ""
            text = name
            ts = self._text(("st", k, text, col), lambda: self.f_stage.render(text, True, col))
            sc.blit(ts, (x + int(30 * u), cy - ts.get_height() // 2))
            if detail:
                ds = self._text(("dt", detail, col), lambda: self.f_small.render(detail, True, MUTED))
                sc.blit(ds, (x + bw - ds.get_width(), cy - ds.get_height() // 2))
            y += row
        y += int(22 * u)

        # the counts
        counts = (self.status or {}).get("counts", {})
        boxes = [("CARS", counts.get("vehicles")), ("TRAFFIC LIGHTS", counts.get("signals")),
                 ("CAMERAS", counts.get("cameras")), ("PEOPLE", counts.get("pedestrians"))]
        gap = int(12 * u)
        cw = (bw - 3 * gap) // 4
        ch = int(86 * u)
        for k, (name, val) in enumerate(boxes):
            bx = x + k * (cw + gap)
            box = self._text(("box", cw, ch), lambda: self._box(cw, ch))
            sc.blit(box, (bx, y))
            if val is None:
                vs = "–"
            else:
                shown = self.counts_shown.get(name, 0.0)
                shown += (val - shown) * 0.12
                if abs(val - shown) < 0.5:
                    shown = float(val)
                self.counts_shown[name] = shown
                vs = f"{shown:.0f}"
            v = self._text(("cv", vs), lambda: self.f_count.render(vs, True, WHITE if val is not None else FAINT))
            sc.blit(v, (bx + int(14 * u), y + int(8 * u)))
            nm = self._text(("cn", name), lambda: _spaced(self.f_small, name, MUTED, 1.6 * u))
            sc.blit(nm, (bx + int(14 * u), y + ch - nm.get_height() - int(12 * u)))
        y += ch + int(30 * u)

        # an error, or the keys, one at a time
        if err:
            msg = "Something got in the way while the city was opening."
            es = self._text(("err", msg), lambda: self.f_tip.render(msg, True, RED))
            sc.blit(es, (x, y))
            hint = self._text("errhint", lambda: self.f_small.render("The city could not open this time. Please try again.",
                                                                     True, MUTED))
            sc.blit(hint, (x, y + es.get_height() + int(6 * u)))
        else:
            k = int((now - self.t0) / 4.5) % len(TIPS)
            key, tip = TIPS[k]
            phase = ((now - self.t0) / 4.5) % 1.0
            alpha = int(255 * min(1.0, phase * 6, (1.0 - phase) * 6))
            ks = self._text(("key", key), lambda: self._keycap(key))
            tp = self._text(("tip", tip), lambda: self.f_tip.render(tip, True, (200, 208, 226)))
            ks.set_alpha(alpha)
            tp.set_alpha(alpha)
            sc.blit(ks, (x, y))
            sc.blit(tp, (x + ks.get_width() + int(14 * u), y + (ks.get_height() - tp.get_height()) // 2))
        foot = self._text("foot", lambda: self.f_small.render(
            "Press Esc to hide this screen. The city keeps getting ready behind it.", True, FAINT))
        sc.blit(foot, (x, int(H * 0.92) - foot.get_height()))

    def _title(self):
        """The name, wide-set, its first letter in the accent blue."""
        pg, u = self.pg, self.u
        head = _spaced(self.f_title, NAME[0], (90, 170, 255), 0)
        rest = _spaced(self.f_title, NAME[1:], WHITE, 10 * u)
        s = pg.Surface((head.get_width() + int(10 * u) + rest.get_width(), head.get_height()), pg.SRCALPHA)
        s.blit(head, (0, 0))
        s.blit(rest, (head.get_width() + int(10 * u), 0))
        return s

    def _bar(self, w: int, h: int):
        pg = self.pg
        s = pg.Surface((w, h), pg.SRCALPHA)
        for i in range(w):
            c = _lerp((90, 160, 255), AMBER, i / max(w - 1, 1))
            pg.draw.line(s, c, (i, 0), (i, h))
        mask = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=h // 2)
        s.blit(mask, (0, 0), special_flags=pg.BLEND_RGBA_MIN)
        return s

    def _box(self, w: int, h: int):
        pg = self.pg
        s = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.rect(s, (18, 24, 40, 235), (0, 0, w, h), border_radius=int(10 * self.u))
        pg.draw.rect(s, (40, 50, 76, 255), (0, 0, w, h), 1, border_radius=int(10 * self.u))
        return s

    def _keycap(self, key: str):
        pg, u = self.pg, self.u
        t = self.f_key.render(key, True, (16, 18, 26))
        w, h = t.get_width() + int(22 * u), t.get_height() + int(8 * u)
        s = pg.Surface((w, h), pg.SRCALPHA)
        pg.draw.rect(s, AMBER, (0, 0, w, h), border_radius=int(6 * u))
        s.blit(t, ((w - t.get_width()) // 2, (h - t.get_height()) // 2))
        return s


# ------------------------------------------------------------------------------------------- the window (Windows)
class Window:
    """The splash's window: on top of everything, without a taskbar button, faded in and out with the window's
    own transparency (Windows); elsewhere a plain window."""

    def __init__(self, hwnd: int):
        self.ok = sys.platform == "win32" and bool(hwnd)
        self.hwnd = hwnd
        if not self.ok:
            return
        import ctypes
        from ctypes import wintypes

        u = self.u = ctypes.windll.user32
        u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        u.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        u.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD]
        u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint]
        ex = u.GetWindowLongPtrW(hwnd, -20)
        u.SetWindowLongPtrW(hwnd, -20, ex | 0x00080000 | 0x00000080 | 0x00000008)  # LAYERED TOOLWINDOW TOPMOST
        self.alpha(0)
        self.on_top()

    def alpha(self, a: float) -> None:
        if self.ok:
            self.u.SetLayeredWindowAttributes(self.hwnd, 0, max(0, min(255, int(a * 255))), 0x2)  # LWA_ALPHA

    def on_top(self) -> None:
        if self.ok:
            self.u.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010 | 0x0040)  # TOPMOST, no move

    def carla_to_front(self) -> None:
        """CARLA's window in front, so its keys work at once (this window may hand the focus over)."""
        if not self.ok:
            return
        try:
            from cosim.overlay import Overlay

            ov = Overlay(0)
            carla = ov._find_carla()
            if carla:
                self.u.SetForegroundWindow(carla)
        except Exception:  # noqa: BLE001 - not essential
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the free-fly viewer's loading screen")
    ap.add_argument("--scenario", default="town05")
    ap.add_argument("--size", default=None, help="WxH: a window of this size instead of the whole screen")
    ap.add_argument("--frames", default=None, help="test: write these seconds' pictures (comma-separated) to "
                                                   "out/freefly/splash_<s>.png from a scripted load, and quit")
    a = ap.parse_args(argv)
    if a.frames:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    else:
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        try:
            from cosim.overlay import dpi_aware

            dpi_aware()
        except Exception:  # noqa: BLE001
            pass
    import pygame

    pygame.init()
    if a.size:
        w, h = (int(v) for v in a.size.lower().split("x"))
    else:
        info = pygame.display.Info()
        w, h = info.current_w, info.current_h
    screen = pygame.display.set_mode((w, h), pygame.NOFRAME)
    pygame.display.set_caption(NAME)
    t0 = time.time()
    splash = Splash(screen, a.scenario, t0)
    if a.frames:
        return _test_frames(splash, [float(v) for v in a.frames.split(",")])
    win = Window(pygame.display.get_wm_info().get("window", 0))
    # SPLASH_SHOTS=1: save a few pictures of the live screen (out/freefly/splash_live_<s>.png), to check it
    shots = [20.0, 70.0, 100.0, 9999.0] if os.environ.get("SPLASH_SHOTS") == "1" else []
    clock = pygame.time.Clock()
    seen_viewer = None
    last_news = t0
    end_at = None
    last = time.time()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return 0
        now = time.time()
        dt, last = min(now - last, 0.1), now
        st = read_status(t0)
        if st is not None:
            if st != splash.status:
                last_news = now
            seen_viewer = seen_viewer or now
            splash.status = st
            splash.prog.update(st, now)
            if end_at is None and (st.get("ready") or st.get("error")):
                end_at = now + (0.6 if st.get("ready") else 8.0)
        last_stage = bool(splash.status) and stage_index(splash.status.get("stage", "")) >= len(STAGES) - 2
        stale = STALE_LAST_S if last_stage else STALE_S
        if end_at is None and ((seen_viewer and now - last_news > stale) or (not seen_viewer and now - t0 > NO_VIEWER_S)):
            end_at = now
            if last_stage and splash.status is not None:  # it got there and went quiet: it is running, show it
                splash.status = dict(splash.status, ready=True)
        splash.draw(now, dt)
        pygame.display.flip()
        if shots and (now - t0 >= shots[0] or (end_at is not None and now >= end_at)):
            pygame.image.save(screen, str(ROOT / "out" / "freefly" / f"splash_live_{shots.pop(0):g}.png"))
        # in and out
        if end_at is not None and now >= end_at:
            k = (now - end_at) / FADE_OUT_S
            win.alpha(1.0 - k)
            if k >= 1.0:
                if splash.status and splash.status.get("ready"):
                    win.carla_to_front()
                pygame.quit()
                return 0
        else:
            win.alpha(min(1.0, (now - t0) / FADE_IN_S))
        if int(now * 2) != int((now - dt) * 2):
            win.on_top()
        clock.tick(60)


def _test_frames(splash: Splash, at: list[float]) -> int:
    """Pictures of a scripted load (no viewer): the viewer's stages at their usual times."""
    import pygame

    script, t = [], 55.0
    for prefix, name, expect, _w in STAGES[1:-1]:
        script.append((t, prefix if prefix != "Loading " else "Loading Town05"))
        t += max(expect, 1.0)
    script.append((t, "Ready"))
    counts_at = {"Filling the town": {"vehicles": 41}, "Handing": {"vehicles": 69}, "Placing": {"signals": 15},
                 "Choosing": {"cameras": 53, "pedestrians": 141}}
    t0 = splash.t0
    counts: dict = {}
    sim = 0.0
    dt = 1 / 30
    out = ROOT / "out" / "freefly"
    out.mkdir(parents=True, exist_ok=True)
    targets = sorted(at)
    while targets:
        sim += dt
        now = t0 + sim
        stage = None
        for ts, name in script:
            if sim >= ts:
                stage = name
        if stage is not None:
            for k, v in counts_at.items():
                if stage.startswith(k):
                    counts.update(v)
            if stage.startswith("Filling"):
                counts.setdefault("vehicles", 41)
            prog = -1.0
            if stage.startswith("Filling"):
                start = next(ts for ts, n in script if n.startswith("Filling"))
                prog = min(1.0, (sim - start) / 20.0)
            splash.status = {"stage": stage, "progress": prog, "counts": dict(counts), "clock": "TUESDAY  08:00",
                             "ready": stage == "Ready", "error": None}
            splash.prog.update(splash.status, now)
        splash.draw(now, dt)
        if sim >= targets[0]:
            p = out / f"splash_{targets.pop(0):g}.png"
            pygame.image.save(splash.screen, str(p))
            print(p)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""(8 Nov) The route agent's mini-map, bottom right of the viewer while you follow a car (F): what the agent knows and
what it decided, live (cosim/route_guidance.py snapshot).

* flying (free-fly camera): the whole town, every car as a dot coloured by its speed (live traffic), the camera and
  what it looks at, the alerts, and the car you may follow (F) ringed;
* following: the map flies in to the trip (its view eases from the town to the trip's box as the camera flies to the
  car), the camera's marker slides onto the car and the routes draw themselves in; then
* the town's roads from SUMO's own shapes, the signalised junctions, the car (an arrow, its trail behind it);
* A where the trip started, B where it goes;
* the car's own route, dashed grey, where it is not the agent's; the agent's route in green, with dots running
  along it towards B;
* the alerts the agent sees: queues DSO4 forecasts (orange, how full in 2 min) and reported crashes (red), pulsing;
* the next signals on its way with their colour for this car, on the map and in a row below it;
* the time since it was picked, the agent's estimate (free flow, no waiting at signals, and said so), the metres
  left; on arrival the real time beside the estimate;
* the last decision and why (the alerts avoided), the options with their costs, and the LLM's sentence when there
  is one.

Drawn at twice the size and scaled down (smooth edges), on an opaque panel: the overlay's see-through colour is a
colour key (cosim/overlay.py), so nothing here may blend into it. The base map is drawn once; the text block only
when what it says changes.
"""
from __future__ import annotations

import itertools
import math

import pygame

from cosim.hud import FONTS

W, MAP_H, MAP_H_FREE, PAD = 430, 300, 250, 14
SS = 2   # supersampling
BOX_EASE_S = 0.45      # the map's view eases to its target (the whole town flying, the trip following) at this pace
INTRO_S = 1.8          # following starts: the camera flies in (cosim/pose_proc.ChaseCam.FLY_IN_S), the map with it
MARGIN_M, MIN_SPAN_M = 70.0, 230.0

BG = (13, 17, 26)
PANEL = (18, 23, 34)
EDGE = (52, 64, 88)
ROAD_CASE = (28, 35, 50)
ROAD = (66, 77, 100)
TEXT = (230, 236, 248)
MUTED = (139, 151, 175)
GREEN = (34, 214, 110)
GREEN_DIM = (22, 94, 58)
BLUE = (96, 165, 250)
GREY = (150, 162, 184)
ORANGE = (249, 140, 34)
RED = (244, 63, 70)
AMBER = (250, 204, 21)
SIGNAL = {"G": GREEN, "g": GREEN, "y": AMBER, "Y": AMBER, "r": RED, "R": RED, "u": AMBER, "s": RED}


def mix(a, b, k: float):
    return tuple(round(x + (y - x) * k) for x, y in zip(a, b))


def mmss(s: float | None) -> str:
    if s is None:
        return "--"
    s = max(0, round(s))
    return f"{s // 60}:{s % 60:02d}"


def wrap(font, text: str, width: int, max_lines: int = 3) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        t = f"{line} {w}".strip()
        if font.size(t)[0] <= width:
            line = t
        else:
            if line:
                lines.append(line)
            line = w
        if len(lines) == max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        lines[-1] = lines[-1].rstrip(".,;") + "..."
    return lines


class MiniMap:
    def __init__(self, router):
        self.r = router
        xs = [x for rd in router.roads.values() for x, _ in rd.shape]
        ys = [y for rd in router.roads.values() for _, y in rd.shape]
        self.town = self.fit((min(xs) - 15, min(ys) - 15, max(xs) + 15, max(ys) + 15), MAP_H_FREE)
        self.box = self.town
        self.box_t: float | None = None
        self.map_h = float(MAP_H_FREE)
        self.tls = {j: xy for j, xy in router.junctions.items() if j in getattr(router, "tls_ids", set())}
        f = FONTS
        self.f_head = pygame.font.Font(str(f / "BarlowCondensed-SemiBold.ttf"), 19)
        self.f_big = pygame.font.Font(str(f / "BarlowCondensed-SemiBold.ttf"), 30)
        self.f_lab = pygame.font.Font(str(f / "BarlowCondensed-Medium.ttf"), 13)
        self.f_txt = pygame.font.Font(str(f / "Barlow-Medium.ttf"), 14)
        self.f_bold = pygame.font.Font(str(f / "Barlow-SemiBold.ttf"), 14)
        self.f_map = pygame.font.Font(str(f / "Barlow-SemiBold.ttf"), 13 * SS)
        self._base_key, self._base = None, None
        self._info_key, self._info = None, None

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def fit(box, map_h: float) -> tuple:
        """box (x0, y0, x1, y1) grown to the map's shape (W x map_h), centred."""
        x0, y0, x1, y1 = box
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        want = W / map_h
        if w / h < want:
            d = (h * want - w) / 2
            x0, x1 = x0 - d, x1 + d
        else:
            d = (w / want - h) / 2
            y0, y1 = y0 - d, y1 + d
        return (x0, y0, x1, y1)

    def trip_box(self, snap: dict, map_h: float) -> tuple:
        """The view while following: the car and what is left of the agent's route (its own old route may run off
        the map, dashed), never wider than the town."""
        pts = [p for rid in (snap.get("left") or []) if rid in self.r.roads for p in self.r.roads[rid].shape]
        if snap.get("xy"):
            pts.append(snap["xy"])
        if not pts:
            return self.fit(self.town, map_h)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        hw = max((max(xs) - min(xs)) / 2 + MARGIN_M, MIN_SPAN_M / 2)
        hh = max((max(ys) - min(ys)) / 2 + MARGIN_M, MIN_SPAN_M / 2 * map_h / W)
        box = self.fit((cx - hw, cy - hh, cx + hw, cy + hh), map_h)
        town = self.fit(self.town, map_h)
        if box[2] - box[0] >= town[2] - town[0]:
            return town
        return box

    def ease_to(self, target: tuple, map_h: float, t: float) -> None:
        dt = 0.0 if self.box_t is None else min(max(t - self.box_t, 0.0), 0.25)
        self.box_t = t
        k = 1.0 - math.exp(-dt / BOX_EASE_S) if dt > 0 else 0.0
        self.box = tuple(a + (b - a) * k for a, b in zip(self.box, target))
        self.map_h += (map_h - self.map_h) * k
        if all(abs(a - b) < 0.3 for a, b in zip(self.box, target)):
            self.box = target
        if abs(self.map_h - map_h) < 0.5:
            self.map_h = float(map_h)

    def px(self, x: float, y: float) -> tuple[float, float]:
        """SUMO (x, y) -> pixel of the supersampled map, in the map's current view."""
        x0, _y0, x1, y1 = self.box
        k = W / (x1 - x0)
        return (SS * (x - x0) * k, SS * (y1 - y) * k)

    def poly(self, route: list[str]) -> list[tuple[float, float]]:
        pts: list = []
        for rid in route:
            rd = self.r.roads.get(rid)
            if rd is None:
                continue
            for p in rd.shape:
                q = self.px(*p)
                if not pts or math.dist(pts[-1], q) > 0.5:
                    pts.append(q)
        return pts

    def _base_map(self, h: int) -> pygame.Surface:
        key = (tuple(round(v, 1) for v in self.box), h)
        if key == self._base_key:
            return self._base.copy()
        s = pygame.Surface((W * SS, h * SS))
        s.fill(BG)
        zoom = (self.town[2] - self.town[0]) / (self.box[2] - self.box[0])
        for step in range(0, W * SS, 40 * SS):          # a faint grid, like a navigation screen
            pygame.draw.line(s, (17, 22, 33), (step, 0), (step, h * SS), SS)
        for step in range(0, h * SS, 40 * SS):
            pygame.draw.line(s, (17, 22, 33), (0, step), (W * SS, step), SS)
        wide, thin = int(9 * SS * max(1.0, zoom ** 0.5)), int(2 * SS * max(1.0, zoom ** 0.5))
        for width, col in ((wide, ROAD_CASE), (thin, ROAD)):
            for rd in self.r.roads.values():
                pts = [self.px(*p) for p in rd.shape]
                if len(pts) > 1:
                    pygame.draw.lines(s, col, False, pts, width)
        for xy in self.tls.values():
            pygame.draw.circle(s, (92, 104, 130), self.px(*xy), 4 * SS, SS)
        self._base_key, self._base = key, s
        return s.copy()

    # ------------------------------------------------------------------ drawing on the map
    @staticmethod
    def _dashed(s, colour, pts, width, dash=10 * SS, gap=7 * SS) -> None:
        on, left = True, dash
        for a, b in itertools.pairwise(pts):
            seg = math.dist(a, b)
            pos = 0.0
            while pos < seg:
                step = min(left, seg - pos)
                if on:
                    p0 = (a[0] + (b[0] - a[0]) * pos / seg, a[1] + (b[1] - a[1]) * pos / seg)
                    p1 = (a[0] + (b[0] - a[0]) * (pos + step) / seg, a[1] + (b[1] - a[1]) * (pos + step) / seg)
                    pygame.draw.line(s, colour, p0, p1, width)
                pos += step
                left -= step
                if left <= 1e-6:
                    on = not on
                    left = dash if on else gap

    @staticmethod
    def _cut(pts, share: float):
        """The first `share` of a polyline (the routes draw themselves in)."""
        if share >= 1.0 or len(pts) < 2:
            return pts
        total = sum(math.dist(a, b) for a, b in itertools.pairwise(pts))
        want, out = total * max(share, 0.0), [pts[0]]
        for a, b in itertools.pairwise(pts):
            seg = math.dist(a, b)
            if want <= seg:
                k = want / seg if seg else 0.0
                out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
                return out
            out.append(b)
            want -= seg
        return out

    @classmethod
    def _along(cls, pts, d: float):
        for a, b in itertools.pairwise(pts):
            seg = math.dist(a, b)
            if d <= seg:
                k = d / seg if seg else 0.0
                return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
            d -= seg
        return None

    def _flow(self, s, pts, t: float) -> None:
        """Dots running along the agent's route towards B."""
        total = sum(math.dist(a, b) for a, b in itertools.pairwise(pts))
        spacing = 26 * SS
        d = (t * 40 * SS) % spacing
        while d < total:
            p = self._along(pts, d)
            if p is not None:
                pygame.draw.circle(s, (200, 255, 222), p, 2 * SS)
            d += spacing

    def _pin(self, s, xy, colour, label: str) -> None:
        x, y = xy
        pygame.draw.circle(s, colour, (x, y - 11 * SS), 8 * SS)
        pygame.draw.polygon(s, colour, [(x - 6 * SS, y - 7 * SS), (x + 6 * SS, y - 7 * SS), (x, y)])
        pygame.draw.circle(s, BG, (x, y - 11 * SS), 5 * SS)
        tt = self.f_map.render(label, True, colour)
        s.blit(tt, (x - tt.get_width() / 2, y - 11 * SS - tt.get_height() / 2))

    @staticmethod
    def _dir(angle: float):
        a = math.radians(angle)              # SUMO: degrees clockwise from north; the map has north up
        return (math.sin(a), -math.cos(a))

    def _car(self, s, p, angle: float, size: float = 1.0) -> None:
        x, y = p
        f = self._dir(angle)
        r = (-f[1], f[0])
        u = SS * size
        tip = (x + f[0] * 11 * u, y + f[1] * 11 * u)
        bl = (x - f[0] * 7 * u + r[0] * 7 * u, y - f[1] * 7 * u + r[1] * 7 * u)
        br = (x - f[0] * 7 * u - r[0] * 7 * u, y - f[1] * 7 * u - r[1] * 7 * u)
        back = (x - f[0] * 3 * u, y - f[1] * 3 * u)
        pygame.draw.circle(s, mix(BG, BLUE, 0.25), (x, y), 15 * u)
        pygame.draw.polygon(s, (255, 255, 255), [tip, bl, back, br])
        pygame.draw.polygon(s, BLUE, [tip, bl, back, br], 2 * SS)

    def _camera(self, s, p, angle: float, fade: float = 1.0) -> None:
        """The free-fly camera: a dot and its view cone (90 deg), faded towards the background by `fade`."""
        x, y = p
        col = mix(BG, (125, 211, 252), 0.85 * fade)
        cone = mix(BG, (56, 120, 170), 0.55 * fade)
        L = 34 * SS
        pts = [(x, y)]
        for k in range(-45, 46, 15):
            f = self._dir(angle + k)
            pts.append((x + f[0] * L, y + f[1] * L))
        pygame.draw.polygon(s, cone, pts)
        pygame.draw.circle(s, col, (x, y), 6 * SS)
        pygame.draw.circle(s, BG, (x, y), 3 * SS)

    def _traffic(self, s, cars, skip: str | None = None) -> None:
        """Every car as a dot: red standing, amber slow, grey moving (live traffic, from the town's own state)."""
        x0, y0, x1, y1 = self.box
        for vid, x, y, v in cars:
            if vid == skip or not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            col = (232, 72, 85) if v < 1.0 else (234, 179, 8) if v < 5.0 else (110, 128, 160)
            pygame.draw.circle(s, col, self.px(x, y), int(2.2 * SS))

    def _alerts(self, s, alerts, pulse: float) -> None:
        for a in alerts or ():
            rd = self.r.roads.get(a["route"])
            if rd is None:
                continue
            col = RED if a["type"] == "accident" else ORANGE
            pts = [self.px(*p) for p in rd.shape]
            if len(pts) > 1:
                pygame.draw.lines(s, mix(BG, col, 0.55), False, pts, 7 * SS)
            mx, my = pts[len(pts) // 2]
            pygame.draw.circle(s, mix(BG, col, 0.35 + 0.4 * pulse), (mx, my), int((9 + 6 * pulse) * SS), 2 * SS)
            pygame.draw.circle(s, col, (mx, my), 5 * SS)
            lab = "X" if a["type"] == "accident" else f"{a.get('fill', 0) * 100:.0f}%"
            tt = self.f_map.render(lab, True, col)
            s.blit(tt, (mx + 9 * SS, my - tt.get_height() - 2 * SS))

    def _map(self, snap: dict | None, free: dict, t: float, intro: float) -> pygame.Surface:
        h = round(self.map_h)
        s = self._base_map(h)
        pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 1.3)
        self._traffic(s, free.get("cars") or (), skip=snap["vid"] if snap else None)
        self._alerts(s, (snap or {}).get("alerts") or free.get("alerts"), pulse)
        if snap is None:            # flying: the camera, the car F would follow
            c = free.get("cand")
            if c is not None:
                p = self.px(c[1], c[2])
                pygame.draw.circle(s, mix(BG, GREEN, 0.3 + 0.5 * pulse), p, int((10 + 5 * pulse) * SS), 2 * SS)
                pygame.draw.circle(s, GREEN, p, 4 * SS)
                tt = self.f_map.render(f"F  {c[0]}", True, GREEN)
                s.blit(tt, (p[0] + 12 * SS, p[1] - tt.get_height() / 2))
            if free.get("cam"):
                cx, cy, ca = free["cam"]
                self._camera(s, self.px(cx, cy), ca)
            return pygame.transform.smoothscale(s, (W, h))
        reveal = ChaseEase.ease(max(0.0, (intro - 0.25) / 0.75))
        trail = [self.px(*p) for p in snap.get("trail", ())]
        if len(trail) > 1:
            n = len(trail)
            for i, (a, b) in enumerate(itertools.pairwise(trail)):
                pygame.draw.line(s, mix(BG, BLUE, 0.25 + 0.6 * i / n), a, b, 3 * SS)
        if snap.get("guided"):
            left, own_left = snap.get("left") or [], snap.get("own_left") or []
            if own_left and own_left != left:
                self._dashed(s, GREY, self._cut(self.poly(own_left), reveal), 2 * SS)
            pts = self._cut(self.poly(left), reveal)
            if len(pts) > 1:
                pygame.draw.lines(s, GREEN_DIM, False, pts, 11 * SS)
                pygame.draw.lines(s, mix(GREEN_DIM, GREEN, 0.5), False, pts, 7 * SS)
                pygame.draw.lines(s, GREEN, False, pts, 3 * SS)
                if snap.get("status") == "following" and reveal >= 1.0:
                    self._flow(s, pts, t)
            if reveal > 0.0:
                st_rd, de_rd = self.r.roads.get(snap["start"]), self.r.roads.get(snap["dest"])
                if st_rd is not None:
                    self._pin(s, self.px(*st_rd.shape[0]), BLUE, "A")
                if de_rd is not None and reveal >= 1.0:
                    self._pin(s, self.px(*de_rd.shape[-1]), GREEN, "B")
        for tl, _d, state in snap.get("signals", ())[:3]:   # the next signals, in this car's colour
            xy = self.r.junctions.get(tl)
            if xy is not None:
                col = SIGNAL.get(state, GREY)
                p = self.px(*xy)
                pygame.draw.circle(s, mix(BG, col, 0.3), p, 9 * SS)
                pygame.draw.circle(s, col, p, 5 * SS)
        if snap.get("xy"):
            car = self.px(*snap["xy"])
            k = ChaseEase.ease(intro)
            cam0 = free.get("cam0")
            if cam0 is not None and k < 1.0:                # the camera's marker slides onto the car
                c0 = self.px(cam0[0], cam0[1])
                p = (c0[0] + (car[0] - c0[0]) * k, c0[1] + (car[1] - c0[1]) * k)
                self._camera(s, p, cam0[2] + ((snap.get("angle", 0.0) - cam0[2] + 180) % 360 - 180) * k, 1.0 - k)
                if k > 0.5:
                    self._car(s, car, snap.get("angle", 0.0), size=(k - 0.5) * 2.0)
            else:
                self._car(s, car, snap.get("angle", 0.0))
        return pygame.transform.smoothscale(s, (W, h))

    # ------------------------------------------------------------------ text block
    def _tile(self, label: str, value: str, colour, width: int) -> pygame.Surface:
        s = pygame.Surface((width, 54))
        s.fill(PANEL)
        pygame.draw.rect(s, EDGE, s.get_rect(), 1, border_radius=6)
        s.blit(self.f_lab.render(label, True, MUTED), (10, 5))
        s.blit(self.f_big.render(value, True, colour), (10, 17))
        return s

    def _info_block(self, snap: dict, view: str) -> pygame.Surface:
        rows: list[tuple[pygame.Surface, int]] = []
        iw = W - 2 * PAD
        arrived = snap["status"] == "arrived"
        if snap.get("guided"):
            tw = (iw - 16) // 3
            row = pygame.Surface((iw, 54))
            row.fill(BG)
            if arrived:
                row.blit(self._tile("ARRIVED IN", mmss(snap.get("real_s")), GREEN, tw), (0, 0))
            else:
                row.blit(self._tile("ON THE WAY", mmss(snap.get("elapsed_s")), TEXT, tw), (0, 0))
            typ = snap.get("est_typical_s")
            est = typ if typ is not None else snap.get("est_free_s")
            row.blit(self._tile("AGENT'S ESTIMATE*", mmss(est), BLUE, tw), (tw + 8, 0))
            if arrived and snap.get("real_s") is not None and est:
                off = snap["real_s"] - est
                col = GREEN if abs(off) <= 0.2 * max(snap["real_s"], 1.0) else AMBER
                row.blit(self._tile("ESTIMATE WAS OFF BY", ("+" if off >= 0 else "-") + mmss(abs(off)), col, tw),
                         (2 * tw + 16, 0))
            else:
                left = snap.get("left_m")
                row.blit(self._tile("LEFT", "--" if left is None else f"{left:.0f} m", TEXT, tw), (2 * tw + 16, 0))
            rows.append((row, 6))
            if typ is not None:
                foot = (f"* from where it was picked, with the typical waits at signals (past trips); free flow "
                        f"{mmss(snap.get('est_free_s'))}")
            else:
                foot = "* free-flow time from where it was picked: no waiting at signals in it"
            rows.append((self.f_lab.render(foot, True, MUTED), 8))
            ev = (snap.get("events") or [])[-1:] or []
            if ev:
                e = ev[0]
                col = {"reroute": ORANGE, "plan": GREEN, "keep": BLUE}.get(e["kind"], TEXT)
                head = {"reroute": "REROUTED", "plan": "AGENT'S ROUTE", "keep": "KEEPS ITS ROUTE"}.get(e["kind"], "")
                at = mmss(e["t"] - (snap["now"] - snap["elapsed_s"])) if e["kind"] == "reroute" else ""
                text = e["text"].split(": ", 1)[-1]
                rows.append((self.f_bold.render(f"{head}  {at}".strip(), True, col), 2))
                for ln in wrap(self.f_txt, text, iw, 2):
                    rows.append((self.f_txt.render(ln, True, TEXT), 1))
                rows[-1] = (rows[-1][0], 8)
            opts = snap.get("options") or []
            if opts:
                row = pygame.Surface((iw, 20))
                row.fill(BG)
                row.blit(self.f_lab.render("OPTIONS (s, alerts in)", True, MUTED), (0, 3))
                hi = max(o["cost"] for o in opts) or 1.0
                x = 118
                bw = (iw - x) // len(opts) - 6
                for i, o in enumerate(opts):
                    col = GREEN if i == 0 else GREY
                    fill = max(4, int((bw - 34) * o["cost"] / hi))
                    pygame.draw.rect(row, mix(BG, col, 0.75), (x, 6, fill, 9), border_radius=4)
                    row.blit(self.f_lab.render(f"{o['cost']:.0f}", True, col), (x + fill + 4, 2))
                    x += bw + 6
                rows.append((row, 6))
            sig = snap.get("signals") or []
            if sig and not arrived:
                row = pygame.Surface((iw, 20))
                row.fill(BG)
                row.blit(self.f_lab.render("SIGNALS AHEAD", True, MUTED), (0, 3))
                x = 118
                for _tl, d, state in sig[:4]:
                    col = SIGNAL.get(state, GREY)
                    pygame.draw.circle(row, col, (x + 6, 10), 6)
                    t = self.f_txt.render(f"{d:.0f} m", True, TEXT)
                    row.blit(t, (x + 16, 1))
                    x += 30 + t.get_width()
                rows.append((row, 6))
            if snap.get("explain"):
                lines = wrap(self.f_txt, "“" + snap["explain"] + "”", iw - 50, 3)
                rows.append((self.f_lab.render("LOCAL LLM (QWEN 2.5)", True, MUTED), 1))
                for ln in lines:
                    rows.append((self.f_txt.render(ln, True, (196, 208, 232)), 1))
            inp = snap.get("inputs") or {}
            row = pygame.Surface((iw, 18))
            row.fill(BG)
            row.blit(self.f_lab.render("IT SEES", True, MUTED), (0, 2))
            x = 118
            for name, on in (("map + its trip", True), ("DSO4 forecasts", inp.get("forecast")),
                             ("reported crashes", inp.get("incidents"))):
                pygame.draw.circle(row, GREEN if on else (70, 80, 100), (x + 4, 9), 4)
                t = self.f_lab.render(name, True, TEXT if on else MUTED)
                row.blit(t, (x + 12, 2))
                x += 22 + t.get_width()
            rows.insert(0, (row, 6))
        elif snap.get("pending"):
            rows.append((self.f_bold.render("PLANNING", True, BLUE), 2))
            rows.append((self.f_txt.render("The agent is reading the car's trip and the alerts...", True, MUTED), 1))
        else:
            rows.append((self.f_bold.render("FOLLOWING ONLY", True, AMBER), 2))
            for ln in wrap(self.f_txt, f"Not routed by the agent: {snap.get('why_not') or 'no destination'}.", iw, 2):
                rows.append((self.f_txt.render(ln, True, TEXT), 1))
        h = sum(r.get_height() + g for r, g in rows)
        s = pygame.Surface((iw, h))
        s.fill(BG)
        y = 0
        for r, g in rows:
            s.blit(r, (0, y))
            y += r.get_height() + g
        return s

    def _header(self, snap: dict, view: str) -> pygame.Surface:
        s = pygame.Surface((W, 36))
        s.fill(BG)
        badge = self.f_head.render("ROUTE AGENT", True, BG)
        pygame.draw.rect(s, GREEN, (PAD, 7, badge.get_width() + 16, 23), border_radius=11)
        s.blit(badge, (PAD + 8, 8))
        t = self.f_head.render(f"following {snap['vid']}", True, TEXT)
        s.blit(t, (PAD + badge.get_width() + 26, 8))
        chip = self.f_lab.render(f"{view.upper()} VIEW  ·  V", True, MUTED)
        s.blit(chip, (W - PAD - chip.get_width(), 12))
        return s

    def _free_header(self, free: dict) -> pygame.Surface:
        s = pygame.Surface((W, 32))
        s.fill(BG)
        badge = self.f_head.render("TOWN05  LIVE", True, BG)
        pygame.draw.rect(s, (125, 211, 252), (PAD, 5, badge.get_width() + 16, 23), border_radius=11)
        s.blit(badge, (PAD + 8, 6))
        n = len(free.get("cars") or ())
        jam = sum(1 for c in free.get("cars") or () if c[3] < 1.0)
        tt = self.f_lab.render(f"{n} CARS   \u00b7   {jam} STANDING   \u00b7   M HIDE", True, MUTED)
        s.blit(tt, (W - PAD - tt.get_width(), 10))
        return s

    def _free_footer(self, free: dict) -> pygame.Surface:
        s = pygame.Surface((W, 28))
        s.fill(BG)
        c = free.get("cand")
        if c is not None:
            tt = self.f_bold.render(f"F   follow car {c[0]}  \u00b7  the route agent drives it", True, GREEN)
        else:
            tt = self.f_txt.render("Fly close to a car to follow it with the route agent", True, MUTED)
        s.blit(tt, (PAD, 5))
        return s

    def render(self, snap: dict | None, t: float, view: str = "chase", free: dict | None = None,
               intro: float = 1.0) -> pygame.Surface:
        """snap: the followed trip (cosim/route_guidance snapshot), or None flying. free: {"cam": (x, y, angle)
        SUMO, "cam0": the camera when following started, "cand": (vid, x, y), "cars": [(vid, x, y, speed)],
        "alerts"}. intro: 0..1 since following started (INTRO_S)."""
        free = free or {}
        if snap is None and free.get("pending"):
            # following has started but the agent's first plan is not here yet (the SUMO thread runs a step or two
            # ahead): the same fly-in, on the car alone, with "planning" under the map
            vid, x, y, ang = free["pending"]
            snap = {"vid": vid, "status": "following", "guided": False, "pending": True, "xy": (x, y), "angle": ang,
                    "trail": [], "signals": [], "alerts": free.get("alerts") or [], "left": [], "own_left": [],
                    "start": None, "dest": None}
        if snap is None:
            self.ease_to(self.town, MAP_H_FREE, t)
            head, foot = self._free_header(free), self._free_footer(free)
            m = self._map(None, free, t, 1.0)
            h = head.get_height() + m.get_height() + foot.get_height()
            out = pygame.Surface((W + 2, h + 2))
            out.fill(EDGE)
            inner = pygame.Surface((W, h))
            inner.fill(BG)
            inner.blit(head, (0, 0))
            inner.blit(m, (0, head.get_height()))
            inner.blit(foot, (0, head.get_height() + m.get_height()))
            out.blit(inner, (1, 1))
            self._info_key = None
            return out
        self.ease_to(self.trip_box(snap, MAP_H), MAP_H, t)
        key = (snap["vid"], snap.get("pending"), snap["status"], int(snap.get("elapsed_s") or 0), snap.get("est_free_s"),
               snap.get("left_m"), tuple((e["t"], e["text"]) for e in snap.get("events") or ()),
               tuple((o["cost"],) for o in snap.get("options") or ()), tuple(snap.get("signals") or ()),
               snap.get("explain"), snap.get("real_s"), snap.get("est_typical_s"),
               tuple(sorted((snap.get("inputs") or {}).items())), view)
        if key != self._info_key:
            self._info_key = key
            self._info = (self._header(snap, view), self._info_block(snap, view))
        head, info = self._info
        m = self._map(snap, free, t, intro)
        shown = round((info.get_height() + PAD) * ChaseEase.ease(intro))   # the text unfolds under the map
        h = head.get_height() + m.get_height() + shown
        s = pygame.Surface((W + 2, h + 2))
        s.fill(EDGE)
        inner = pygame.Surface((W, h))
        inner.fill(BG)
        inner.blit(head, (0, 0))
        inner.blit(m, (0, head.get_height()))
        if shown > 0:
            inner.blit(info, (PAD, head.get_height() + m.get_height()), pygame.Rect(0, 0, W - 2 * PAD, shown))
        s.blit(inner, (1, 1))
        return s


class ChaseEase:
    @staticmethod
    def ease(k: float) -> float:
        k = min(max(k, 0.0), 1.0)
        return k * k * k * (k * (6.0 * k - 15.0) + 10.0)

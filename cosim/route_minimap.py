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
from cosim.place_icons import Painter, occupancy_colour
from cosim.place_icons import colour as kind_colour

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
DARK_RED = (150, 22, 36)
TRAFFIC = {1: ORANGE, 2: RED, 3: DARK_RED}      # (10 Nov) a road ahead by its live time against its usual one
DELAY = {0: GREEN, 1: ORANGE, 2: RED}           # the time left by its delay against usual traffic
PURPLE = (192, 132, 252)
# (8 Nov) a trip with a purpose (cosim/purposes.py): what it is called on the header, its colour
PURPOSE = {"work": ("TO WORK", BLUE), "shopping": ("SHOPPING", PURPLE), "eating": ("EATING OUT", (248, 113, 113)),
           "park": ("TO THE PARK", (74, 222, 128)), "care": ("HOSPITAL VISIT", (239, 68, 68)),
           "errand": ("ERRAND", AMBER), "visitor": ("VISITOR", (148, 163, 184)), "deliver": ("DELIVERY", ORANGE),
           "home": ("GOING HOME", (255, 183, 77))}
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
    def __init__(self, router, places: list | None = None):
        self.r = router
        # (8 Nov) the town's places (scenarios/town05/places.json): their buildings on the map, an icon each, a car
        # park's ring coloured by its free spaces, names when the map is close; the trip's A and B are places
        self.places = [p for p in places or () if "label" in p]
        self.place = {p["id"]: p for p in self.places}
        self.area_name = {}
        for p in self.places:
            for a in [p.get("parking")] + [k["id"] for k in p.get("kerbs") or ()]:
                if a:
                    self.area_name[a] = p["name"]
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
        for p in self.places:                          # the places' buildings, faint in their kind's colour
            col = kind_colour(p["kind"])
            polys = p.get("blocks") or ([p["outline"]] if len(p.get("outline") or ()) > 2 else [])
            for poly in polys:
                pts = [self.px(*q) for q in poly]
                if len(pts) > 2:
                    pygame.draw.polygon(s, mix(BG, col, 0.17), pts)
                    pygame.draw.polygon(s, mix(BG, col, 0.42), pts, SS)
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

    def _places(self, s, live: dict | None, snap: dict | None) -> None:
        """An icon per place (a car park's ring: green, amber, red as it fills); their names when the map is close.
        The trip's own places are drawn as its A and B (_place_pin)."""
        if not self.places:
            return
        x0, y0, x1, y1 = self.box
        zoom = (self.town[2] - self.town[0]) / (x1 - x0)
        size = (13 + 3 * min(zoom - 1.0, 1.0)) * SS
        paint = Painter(s, "pygame")
        live = live or {}
        occ, cap, park_of = live.get("occ", {}), live.get("cap", {}), live.get("park_of", {})
        mine = {(snap.get("from_place") or {}).get("id"), (snap.get("to_place") or {}).get("id")} if snap else set()
        taken: list = []
        named: list = []
        for p in sorted(self.places, key=lambda q: q["kind"] == "kerb"):
            if p["id"] in mine:
                continue
            lx, ly = p["label"]
            if not (x0 - 20 <= lx <= x1 + 20 and y0 - 20 <= ly <= y1 + 20):
                continue
            if p["kind"] == "kerb" and zoom < 1.6:
                continue
            cx, cy = self.px(lx, ly)
            ring = None
            if p["kind"] in ("parking", "kerb") and park_of.get(p["id"]):
                room = sum(cap.get(a, 0) for a in park_of[p["id"]])
                used = sum(occ.get(a, 0) for a in park_of[p["id"]])
                ring = occupancy_colour((room - used) / room if room else 0.0)
            paint.icon(p["kind"], cx, cy, size * (0.8 if p["kind"] == "kerb" else 1.0), ring=ring)
            icon = pygame.Rect(0, 0, size, size)
            icon.center = (int(cx), int(cy))
            taken.append(icon)
            if zoom >= 1.5 and p["kind"] != "kerb":
                named.append((p, cx, cy, icon))
        for p, cx, cy, icon in named:          # names after every icon: none written over another's icon
            tt = self.f_map.render(p["name"], True, mix(BG, TEXT, 0.8))
            r = tt.get_rect(midleft=(cx + size * 0.7, cy))
            if not any(r.colliderect(q) for q in taken if q is not icon):
                taken.append(r.inflate(6 * SS, 2 * SS))
                s.blit(tt, r)

    def _place_pin(self, s, place: dict, letter: str, colour, live: dict | None) -> None:
        """The trip's A or B at its place: a pin in the place's colour with its icon, and a tag with its name."""
        p = self.place.get(place.get("id"))
        if p is None:
            return
        x, y = self.px(*p["label"])
        col = kind_colour(p["kind"])
        top = y - 17 * SS
        pygame.draw.polygon(s, col, [(x - 7 * SS, top + 7 * SS), (x + 7 * SS, top + 7 * SS), (x, y)])
        Painter(s, "pygame").icon(p["kind"], x, top, 22 * SS, ring=(255, 255, 255))
        tag = self.f_map.render(f"{letter}  {p['name']}", True, BG)
        r = tag.get_rect(midleft=(x + 15 * SS, top))
        if r.right + 6 * SS > s.get_width():       # near the map's right edge: the tag on the pin's left
            r = tag.get_rect(midright=(x - 15 * SS, top))
        pygame.draw.rect(s, colour, r.inflate(10 * SS, 5 * SS), border_radius=8 * SS)
        s.blit(tag, r)

    def _map(self, snap: dict | None, free: dict, t: float, intro: float) -> pygame.Surface:
        h = round(self.map_h)
        s = self._base_map(h)
        pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 1.3)
        self._places(s, free.get("places"), snap)
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
                live = (snap.get("trip") or {}).get("live") or {}
                for road, lev in live.get("roads") or ():      # (10 Nov) live traffic on the roads ahead
                    if lev and road in left:
                        rp = self._cut(self.poly([road]), reveal)
                        if len(rp) > 1:
                            pygame.draw.lines(s, TRAFFIC[min(lev, 3)], False, rp, 5 * SS)
                if snap.get("status") == "following" and reveal >= 1.0:
                    self._flow(s, pts, t)
            if reveal > 0.0:
                st_rd, de_rd = self.r.roads.get(snap["start"]), self.r.roads.get(snap["dest"])
                fp, tp = snap.get("from_place") or {}, snap.get("to_place") or {}
                if fp.get("id") in self.place:
                    self._place_pin(s, fp, "A", BLUE, free.get("places"))
                elif st_rd is not None:
                    self._pin(s, self.px(*st_rd.shape[0]), BLUE, "A")
                if reveal >= 1.0:
                    if tp.get("id") in self.place:
                        self._place_pin(s, tp, "B", GREEN, free.get("places"))
                    elif de_rd is not None:
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

    def _park_row(self, snap: dict, park: tuple, iw: int) -> pygame.Surface:
        """(8 Nov) where it will park (or has): the car park's name and its free spaces now."""
        name, free, room = park
        row = pygame.Surface((iw, 22))
        row.fill(BG)
        row.blit(self.f_lab.render("PARKED AT" if snap["status"] == "arrived" else "PARKS AT", True, MUTED), (0, 4))
        Painter(row, "pygame").icon("parking", 127, 11, 18,
                                    ring=occupancy_colour(free / room if room else 0.0))
        t = self.f_bold.render(name, True, TEXT)
        row.blit(t, (142, 2))
        f = self.f_txt.render(f"{free} free of {room}", True, occupancy_colour(free / room if room else 0.0))
        row.blit(f, (152 + t.get_width(), 2))
        return row

    def _info_block(self, snap: dict, view: str, park: tuple | None = None) -> pygame.Surface:
        rows: list[tuple[pygame.Surface, int]] = []
        iw = W - 2 * PAD
        arrived = snap["status"] == "arrived"
        if park is not None:
            rows.append((self._park_row(snap, park, iw), 6))
        trip = snap.get("trip")
        if trip is not None:
            rows.extend(self._trip_rows(trip, iw, None if arrived else snap.get("left_m")))
        agent_at = len(rows)   # the agent's rows start here ("IT SEES" heads them)
        if snap.get("guided") and trip is None:   # (no trip estimate: the agent's own times, from where you pressed F)
            tw = (iw - 16) // 3
            row = pygame.Surface((iw, 54))
            row.fill(BG)
            if arrived:
                row.blit(self._tile("ARRIVED IN", mmss(snap.get("real_s")), GREEN, tw), (0, 0))
            else:
                row.blit(self._tile("SINCE YOU FOLLOWED", mmss(snap.get("elapsed_s")), TEXT, tw), (0, 0))
            typ = snap.get("est_typical_s")
            est = typ if typ is not None else snap.get("est_free_s")
            row.blit(self._tile("AGENT FROM HERE*", mmss(est), BLUE, tw), (tw + 8, 0))
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
                foot = (f"* the agent's: from where you pressed F, typical waits at signals in; free flow "
                        f"{mmss(snap.get('est_free_s'))}")
            else:
                foot = "* the agent's: free flow from where you pressed F, no waiting at signals in it"
            rows.append((self.f_lab.render(foot, True, MUTED), 8))
        if snap.get("guided"):
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
            rows.insert(agent_at, (row, 6))
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

    @staticmethod
    def _trip_key(trip: dict | None):
        if trip is None:
            return None
        lv = trip.get("live") or {}
        return (int(trip.get("elapsed_s") or 0), trip.get("est_s"), trip.get("ended"), len(trip.get("changes") or ()),
                trip.get("start"), trip.get("dest"), int(lv.get("left_s") or 0), lv.get("delay_level"),
                int(lv.get("usual_s") or 0), int((lv.get("left_m") or 0) // 10))

    def _trip_rows(self, trip: dict, iw: int, left_m: float | None = None) -> list:
        """(10 Nov) The car's own trip, from where it set off: driving for, the estimate made then (cosim/
        trip_estimates.py, never made again), and how it stands against it; where from and to; when the town gave it
        another way."""
        rows = []
        tw = (iw - 16) // 3
        row = pygame.Surface((iw, 54))
        row.fill(BG)
        ended = trip.get("ended")
        done = ended in ("parked", "arrived", "stopped")
        el, est = trip.get("elapsed_s"), trip.get("est_s")
        live = None if done else trip.get("live")
        if live is not None:   # (10 Nov) as a navigation app: the time left now, live traffic in, its distance
            lvl = live.get("delay_level") or 0
            row.blit(self._tile("TIME LEFT", mmss(live["left_s"]), DELAY.get(lvl, GREEN), tw), (0, 0))
            m = live.get("left_m") or 0
            row.blit(self._tile("DISTANCE LEFT", f"{m / 1000:.1f} km" if m >= 1000 else f"{m:.0f} m", TEXT, tw),
                     (tw + 8, 0))
            row.blit(self._tile("ESTIMATED AT START", mmss(est), BLUE, tw), (2 * tw + 16, 0))
            rows.append((row, 4))
            d = live["left_s"] - live["usual_s"]
            usual = (f"+{mmss(d)} vs usual traffic" if d >= 5 else
                     f"-{mmss(-d)} vs usual traffic" if d <= -5 else "usual traffic")
            known = round(100 * (live.get("known") or 0.0))
            rows.append((self.f_lab.render(f"driving {mmss(el)}  ·  {usual}  ·  live traffic in {known} % of it",
                                           True, DELAY.get(lvl, TEXT) if lvl else MUTED), 2))
            row = None
        if row is not None:
            row.blit(self._tile("ITS TRIP: DONE IN" if done else "ITS TRIP: DRIVING", mmss(el),
                                GREEN if done else TEXT, tw), (0, 0))
            row.blit(self._tile("ESTIMATED AT START", mmss(est), BLUE, tw), (tw + 8, 0))
        if row is not None and el is not None and est:
            off = el - est
            if done:
                col = GREEN if abs(off) <= 0.2 * max(el, 1.0) else AMBER
                row.blit(self._tile("OFF BY", ("+" if off >= 0 else "-") + mmss(abs(off)), col, tw), (2 * tw + 16, 0))
            elif off > 0:
                row.blit(self._tile("OVER ESTIMATE", "+" + mmss(off), AMBER, tw), (2 * tw + 16, 0))
            else:
                row.blit(self._tile("EST. TIME LEFT", mmss(-off), GREEN, tw), (2 * tw + 16, 0))
        if row is not None:
            rows.append((row, 4))
        frm = trip.get("start") or "the road it set off on"
        to = trip.get("dest") or "the end of its route"
        left = f"  ·  {left_m:.0f} m left" if left_m is not None and not done and live is None else ""
        rows.append((self.f_lab.render(f"from {frm} to {to}{left}", True, TEXT), 2))
        share = trip.get("table_share") or 0.0
        how = ("typical times of past trips, waits at signals in" if share >= 0.5 else
               "mostly free-flow times (few past trips on its roads)")
        rows.append((self.f_lab.render(f"estimated when it set off: {how}", True, MUTED), 2))
        ch = [c for c in trip.get("changes") or [] if c[1] != "new destination given on the way"]
        if any(c[1] == "new destination given on the way" for c in trip.get("changes") or []):
            rows.append((self.f_lab.render("its first destination was changed on the way: this is its new trip",
                                           True, AMBER), 2))
        if ch:
            rows.append((self.f_lab.render("given another way at " + ", ".join(mmss(t) for t, _ in ch[-3:]), True,
                                           AMBER), 2))
        rows[-1] = (rows[-1][0], 10)
        return rows

    def _header(self, snap: dict, view: str) -> pygame.Surface:
        purpose = PURPOSE.get(snap.get("purpose") or "")
        s = pygame.Surface((W, 36 + (30 if purpose else 0)))
        s.fill(BG)
        badge = self.f_head.render("ROUTE AGENT", True, BG)
        pygame.draw.rect(s, GREEN, (PAD, 7, badge.get_width() + 16, 23), border_radius=11)
        s.blit(badge, (PAD + 8, 8))
        t = self.f_head.render(f"following {snap['vid']}", True, TEXT)
        s.blit(t, (PAD + badge.get_width() + 26, 8))
        chip = self.f_lab.render(f"{view.upper()} VIEW  ·  V", True, MUTED)
        s.blit(chip, (W - PAD - chip.get_width(), 12))
        if purpose:     # (8 Nov) why it drives: its purpose, from which place to which
            name, col = purpose
            tag = self.f_lab.render(name, True, BG)
            pygame.draw.rect(s, col, (PAD, 38, tag.get_width() + 14, 20), border_radius=10)
            s.blit(tag, (PAD + 7, 40))
            x = PAD + tag.get_width() + 24
            paint = Painter(s, "pygame")
            for i, pl in enumerate((snap.get("from_place") or {}, snap.get("to_place") or {})):
                if i:           # an arrow, drawn (the font has none)
                    pygame.draw.line(s, MUTED, (x, 48), (x + 12, 48), 2)
                    pygame.draw.polygon(s, MUTED, [(x + 16, 48), (x + 10, 43), (x + 10, 53)])
                    x += 24
                if pl.get("kind") in ("out", None):
                    nm = self.f_txt.render(pl.get("name") or "?", True, MUTED)
                else:
                    paint.icon(pl["kind"], x + 9, 48, 18)
                    x += 23
                    nm = self.f_bold.render(pl["name"], True, TEXT)
                s.blit(nm, (x, 39))
                x += nm.get_width() + 10
        return s

    def _free_header(self, free: dict) -> pygame.Surface:
        s = pygame.Surface((W, 32))
        s.fill(BG)
        badge = self.f_head.render("TOWN05  LIVE", True, BG)
        pygame.draw.rect(s, (125, 211, 252), (PAD, 5, badge.get_width() + 16, 23), border_radius=11)
        s.blit(badge, (PAD + 8, 6))
        n = len(free.get("cars") or ())
        jam = sum(1 for c in free.get("cars") or () if c[3] < 1.0)
        live = free.get("places") or {}
        if live:     # (the cars list leaves the parked ones out: cosim/freefly.py free_view)
            hr = live.get("hour", 0.0) % 24
            why = live.get("why") or {}
            parked = sum(live.get("occ", {}).values()) + why.get("depot", 0)   # (the depots' cars wait at home)
            if why:  # standing: before a red signal, or queued (behind a standing car, the ring road, a crash...)
                red = why.get("red_signal", 0)
                queued = why.get("total", 0) - red - why.get("depot", 0)
                moving = n - why.get("total", 0)
                txt = f"{moving} MOVING  \u00b7  {red} AT RED  \u00b7  {queued} QUEUED  \u00b7  {parked} PARKED"
            else:
                txt = (f"{n} DRIVING   \u00b7   {jam} STANDING   \u00b7   {parked} PARKED   \u00b7   "
                       f"{int(hr):02d}:{int(hr % 1 * 60):02d}")
            tt = self.f_lab.render(txt, True, MUTED)
        else:
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
        park = None
        live = free.get("places") or {}
        area = snap.get("stop_area")
        if area and live and area in live.get("cap", {}):
            room = live["cap"][area]
            park = (self.area_name.get(area, area), max(0, room - live.get("occ", {}).get(area, 0)), room)
        key = (park, snap.get("purpose"), snap["vid"], snap.get("pending"), snap["status"], int(snap.get("elapsed_s") or 0), snap.get("est_free_s"),
               snap.get("left_m"), tuple((e["t"], e["text"]) for e in snap.get("events") or ()),
               tuple((o["cost"],) for o in snap.get("options") or ()), tuple(snap.get("signals") or ()),
               snap.get("explain"), snap.get("real_s"), snap.get("est_typical_s"),
               tuple(sorted((snap.get("inputs") or {}).items())), view, snap.get("guided"),
               self._trip_key(snap.get("trip")))
        if key != self._info_key:
            self._info_key = key
            self._info = (self._header(snap, view), self._info_block(snap, view, park))
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

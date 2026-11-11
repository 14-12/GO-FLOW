"""The junction agents' decisions on the free-fly overlay (24 Oct).

With --controller agents_observed (or agents), each signalised junction has an agent on top of the actuated program
(cosim/agents_actuated.py). What it does shows where it happens:

* over the junction, for 10 s after a decision, a pill with what it did ("SKIPPED AN EMPTY GREEN",
  "GREEN HELD FOR AN AMBULANCE", ...) and the junction's id, coloured by the kind of decision;
* at the stop lines of the green it was about, a chevron in the direction of travel, in the same colour
  (an empty green skipped, a green ended, a green held), under 250 m away;
* top right, the decisions near you of the last minute, newest first, with how far and how long ago, and a count
  of the minute's decisions (all of them, and those for emergency vehicles).

Drawn into the overlay window (cosim/overlay.py), whose see-through colour is a colour key: shapes have no blended
edges and text sits on its own solid pill, as the accident markers (cosim/incident_hud.py).
"""

from __future__ import annotations

import math

import pygame

from cosim import ui
from cosim.hud import FONTS

COLOURS = {"skip": (96, 165, 250), "walk": (148, 163, 184), "spill": (239, 68, 68), "noflow": (249, 115, 22),
           "trickle": (234, 179, 8), "balance": (192, 132, 252), "ambulance": (52, 211, 153),
           "corridor": (45, 212, 191), "extend": (148, 163, 184)}
SHOW_S = 10.0        # a decision stays over its junction this long (simulated seconds)
NEAR_M = 400.0       # junction pills within this distance
CHEVRON_M = 250.0    # stop-line chevrons within this distance
FEED_M = 700.0       # the list top right: decisions this close (else the newest anywhere)
EDGE = 48
BG = (22, 24, 30)


class AgentHud:
    def __init__(self, fov_deg: float = 90.0, mode: str = "cameras"):
        self.fov = fov_deg
        self.mode = mode
        self.f_title = pygame.font.Font(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 16)
        self.f_small = pygame.font.Font(str(FONTS / "Barlow-Medium.ttf"), 12)
        self.f_head = pygame.font.Font(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 15)

    def project(self, tf, size, x: float, y: float, z: float):
        """Screen (sx, sy) of CARLA point (x, y, z) seen from `tf`, or None behind the view."""
        w, h = size
        f, r, u = tf.get_forward_vector(), tf.get_right_vector(), tf.get_up_vector()
        dx, dy, dz = x - tf.location.x, y - tf.location.y, z - tf.location.z
        cx = dx * f.x + dy * f.y + dz * f.z
        if cx <= 0.5:
            return None
        cy = dx * r.x + dy * r.y + dz * r.z
        cz = dx * u.x + dy * u.y + dz * u.z
        fpx = (w / 2.0) / math.tan(math.radians(self.fov) / 2.0)
        return w / 2.0 + cy / cx * fpx, h / 2.0 - cz / cx * fpx

    def _pill(self, title: str, sub: str, colour) -> pygame.Surface:
        t = self.f_title.render(title, True, (245, 246, 248), BG)
        b = self.f_small.render(sub, True, (196, 202, 214), BG)
        w = max(t.get_width(), b.get_width()) + 22
        h = t.get_height() + b.get_height() + 10
        surf = pygame.Surface((w, h))
        surf.fill(BG)
        pygame.draw.rect(surf, colour, (0, 0, 5, h))
        surf.blit(t, (13, 4))
        surf.blit(b, (13, 4 + t.get_height()))
        return ui.glassify(surf)

    @staticmethod
    def _chevron(screen, tf, project, size, x, y, z, yaw_deg, colour) -> pygame.Rect | None:
        """A chevron on the road at (x, y, z) pointing along yaw (CARLA degrees): drawn from three projected points."""
        a = math.radians(yaw_deg)
        fx, fy = math.cos(a), math.sin(a)
        rx, ry = -fy, fx
        pts3 = [(x + fx * 1.0, y + fy * 1.0), (x - fx * 2.4 + rx * 1.5, y - fy * 2.4 + ry * 1.5),
                (x - fx * 1.2, y - fy * 1.2), (x - fx * 2.4 - rx * 1.5, y - fy * 2.4 - ry * 1.5)]
        pts = [project(tf, size, px, py, z + 0.4) for px, py in pts3]
        if any(p is None for p in pts):
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) - min(xs) > 400 or max(ys) - min(ys) > 400:
            return None  # right under the view: not a useful sign
        pygame.draw.polygon(screen, (16, 17, 20), pts, 0)
        inner = [((p[0] - sum(xs) / 4) * 0.72 + sum(xs) / 4, (p[1] - sum(ys) / 4) * 0.72 + sum(ys) / 4) for p in pts]
        return pygame.draw.polygon(screen, colour, inner, 0).union(pygame.Rect(min(xs), min(ys), max(xs) - min(xs) + 1,
                                                                            max(ys) - min(ys) + 1))

    def draw(self, screen, size, view_tf, centres: dict, stop_lines: dict, events: list, now: float) -> list:
        """centres: tls -> CARLA (x, y, z) of the junction; stop_lines: tls -> phase -> [(x, y, z, yaw)];
        events: the agents' decisions (cosim/agents_actuated.py ActuatedAgents.events) of the last minute;
        now: the simulation's time. Returns the rectangles drawn."""
        rects: list = []
        w, h = size
        loc = view_tf.location
        latest: dict[str, dict] = {}
        marks: dict[tuple[str, int], str] = {}  # (junction, green) -> the kind of its latest decision
        for e in events:
            if now - e["t"] <= SHOW_S and e["kind"] != "walk":
                latest[e["tls"]] = e
                for k in e["phases"]:
                    marks[(e["tls"], k)] = e["kind"]
        for (tls, k), kind in marks.items():
            c = centres.get(tls)
            if c is None or math.dist((loc.x, loc.y, loc.z), c) > CHEVRON_M:
                continue
            for sx, sy, sz, yaw in stop_lines.get(tls, {}).get(k, []):
                r = self._chevron(screen, view_tf, self.project, size, sx, sy, sz, yaw, COLOURS.get(kind, COLOURS["skip"]))
                if r is not None:
                    rects.append(r)
        for tls, e in latest.items():
            if tls not in centres:
                continue
            x, y, z = centres[tls]
            d = math.dist((loc.x, loc.y, loc.z), (x, y, z))
            if d > NEAR_M:
                continue
            colour = COLOURS.get(e["kind"], COLOURS["skip"])
            p = self.project(view_tf, size, x, y, z + 9.0)
            if p is None or not (EDGE <= p[0] <= w - EDGE and EDGE <= p[1] <= h - EDGE):
                continue
            ago = max(0, int(now - e["t"]))
            pill = self._pill(e["text"].upper(), f"junction agent {tls}  ·  {self.mode}  ·  {ago} s ago", colour)
            px = min(max(p[0] - pill.get_width() / 2, 8), w - pill.get_width() - 8)
            py = min(max(p[1] - pill.get_height(), 8), h - pill.get_height() - 8)
            rects.append(screen.blit(pill, (px, py)))
        rects += self._feed(screen, size, loc, centres, events, now)
        return rects

    def _feed(self, screen, size, loc, centres: dict, events: list, now: float) -> list:
        w, _ = size
        if not events:
            return []

        def dist(e):
            c = centres.get(e["tls"])
            return math.dist((loc.x, loc.y), (c[0], c[1])) if c else 1e9

        near = [e for e in events if dist(e) <= FEED_M]
        shown = (near or events)[-5:][::-1]
        evs = sum(1 for e in events if e["kind"] in ("ambulance", "corridor"))
        head = self.f_head.render(f"JUNCTION AGENTS  ·  {self.mode.upper()}", True, (245, 246, 248), BG)
        sub = self.f_small.render(f"last minute: {len(events)} decisions, {evs} for emergency vehicles"
                                  + ("" if near else "  ·  none near you"), True, (170, 178, 192), BG)
        lines = []
        for e in shown:
            d = dist(e)
            far = f"{d:.0f} m" if d < 1000 else f"{d / 1000:.1f} km"
            lines.append((COLOURS.get(e["kind"], COLOURS["skip"]),
                          self.f_small.render(f"{e['tls']}  ·  {far}  ·  {e['text']}  ·  {max(0, int(now - e['t']))} s",
                                              True, (226, 230, 238), BG)))
        pw = max([head.get_width(), sub.get_width()] + [t.get_width() + 18 for _, t in lines]) + 24
        ph = 10 + head.get_height() + sub.get_height() + 6 + sum(t.get_height() + 3 for _, t in lines) + 8
        surf = pygame.Surface((pw, ph))
        surf.fill(BG)
        yy = 6
        surf.blit(head, (12, yy))
        yy += head.get_height()
        surf.blit(sub, (12, yy))
        yy += sub.get_height() + 6
        for colour, t in lines:
            pygame.draw.rect(surf, colour, (12, yy + t.get_height() // 2 - 4, 8, 8))
            surf.blit(t, (28, yy))
            yy += t.get_height() + 3
        surf = ui.glassify(surf, radius=10)
        return [screen.blit(surf, (w - surf.get_width() - 16 + (ui.MARGIN if ui.ALPHA else 0), 16))]

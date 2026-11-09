"""The places' look (scenarios/town05/places.json): a colour and a small vector icon per kind, drawn the same way on the
atlas (PIL, tools/town05_places_atlas.py), the mini-map and the labels over the town (pygame, cosim/route_minimap.py,
cosim/place_hud.py). An icon is a list of shapes in a unit square (-1..1, y down), drawn by a painter that knows how to
fill a polygon, a circle and a line on its own surface."""
from __future__ import annotations

STYLE = {   # kind: (colour, icon, what it is called on the legend)
    "houses": ((255, 183, 77), "house", "Homes"),
    "apartments": ((255, 145, 77), "flats", "Apartments"),
    "office": ((96, 165, 250), "tower", "Offices"),
    "mall": ((192, 132, 252), "bag", "Shopping"),
    "market": ((52, 211, 153), "awning", "Market"),
    "cafe": ((251, 191, 36), "cup", "Café"),
    "restaurant": ((248, 113, 113), "cutlery", "Restaurants"),
    "park": ((74, 222, 128), "tree", "Park"),
    "parking": ((56, 189, 248), "P", "Car parks"),
    "kerb": ((125, 211, 252), "p", "Street parking"),
    "hospital": ((239, 68, 68), "cross", "Hospital"),
    "police": ((59, 130, 246), "shield", "Police"),
    "garage": ((250, 204, 21), "wrench", "Recovery"),
}
WHITE = (255, 255, 255)


def colour(kind: str):
    return STYLE.get(kind, ((200, 200, 200), "dot", kind))[0]


def shapes(icon: str) -> list:
    """[(kind, data)]: ("poly", [(x, y)...]), ("circle", (x, y, r)), ("line", ((x0, y0), (x1, y1), width)),
    ("ring", (x, y, r, width)). Unit square, y down, drawn in white on the kind's colour."""
    if icon == "house":
        return [("poly", [(-0.62, -0.05), (0.0, -0.62), (0.62, -0.05), (0.45, -0.05), (0.45, 0.6), (-0.45, 0.6),
                          (-0.45, -0.05)])]
    if icon == "flats":
        out = [("poly", [(-0.5, -0.65), (0.5, -0.65), (0.5, 0.65), (-0.5, 0.65)])]
        return out
    if icon == "tower":
        return [("poly", [(-0.3, -0.75), (0.3, -0.75), (0.3, 0.7), (-0.3, 0.7)]),
                ("poly", [(-0.62, -0.1), (-0.3, -0.1), (-0.3, 0.7), (-0.62, 0.7)]),
                ("poly", [(0.3, 0.1), (0.62, 0.1), (0.62, 0.7), (0.3, 0.7)])]
    if icon == "bag":
        return [("poly", [(-0.55, -0.25), (0.55, -0.25), (0.45, 0.65), (-0.45, 0.65)]),
                ("ring", (0.0, -0.3, 0.3, 0.12))]
    if icon == "awning":
        out = []
        for i in range(4):
            x0 = -0.66 + i * 0.33
            out.append(("poly", [(x0, -0.5), (x0 + 0.33, -0.5), (x0 + 0.33, -0.1), (x0, -0.1)] if i % 2 == 0 else
                        [(x0, -0.5), (x0 + 0.33, -0.5), (x0 + 0.165, -0.1)]))
        out.append(("poly", [(-0.55, 0.0), (0.55, 0.0), (0.55, 0.6), (-0.55, 0.6)]))
        return out
    if icon == "cup":
        return [("poly", [(-0.5, -0.3), (0.35, -0.3), (0.25, 0.5), (-0.4, 0.5)]), ("ring", (0.42, 0.0, 0.2, 0.1)),
                ("line", ((-0.25, -0.75), (-0.2, -0.45), 0.08)), ("line", ((0.05, -0.75), (0.1, -0.45), 0.08))]
    if icon == "cutlery":
        return [("line", ((-0.3, -0.7), (-0.3, 0.7), 0.16)), ("line", ((-0.5, -0.7), (-0.5, -0.2), 0.08)),
                ("line", ((-0.1, -0.7), (-0.1, -0.2), 0.08)), ("line", ((-0.5, -0.2), (-0.1, -0.2), 0.1)),
                ("poly", [(0.25, -0.7), (0.5, -0.5), (0.5, 0.0), (0.38, 0.0), (0.38, 0.7), (0.25, 0.7)])]
    if icon == "tree":
        return [("circle", (0.0, -0.2, 0.48)), ("poly", [(-0.1, 0.1), (0.1, 0.1), (0.1, 0.72), (-0.1, 0.72)])]
    if icon in ("P", "p"):
        r = 0.6 if icon == "P" else 0.5
        return [("poly", [(-0.38 * r / 0.6, -0.7 * r / 0.6), (-0.15 * r / 0.6, -0.7 * r / 0.6),
                          (-0.15 * r / 0.6, 0.7 * r / 0.6), (-0.38 * r / 0.6, 0.7 * r / 0.6)]),
                ("ring", (0.05 * r / 0.6, -0.27 * r / 0.6, 0.3 * r / 0.6, 0.2 * r / 0.6))]
    if icon == "cross":
        return [("poly", [(-0.2, -0.62), (0.2, -0.62), (0.2, 0.62), (-0.2, 0.62)]),
                ("poly", [(-0.62, -0.2), (0.62, -0.2), (0.62, 0.2), (-0.62, 0.2)])]
    if icon == "shield":
        return [("poly", [(0.0, -0.7), (0.58, -0.45), (0.5, 0.15), (0.0, 0.7), (-0.5, 0.15), (-0.58, -0.45)])]
    if icon == "wrench":
        return [("line", ((-0.45, 0.45), (0.25, -0.25), 0.22)), ("ring", (0.35, -0.35, 0.28, 0.14))]
    return [("circle", (0.0, 0.0, 0.35))]


def detail(icon: str) -> list:
    """Shapes drawn in the kind's colour over the white ones (windows, a door)."""
    if icon == "house":
        return [("poly", [(-0.12, 0.2), (0.12, 0.2), (0.12, 0.6), (-0.12, 0.6)])]
    if icon == "flats":
        out = []
        for r in range(3):
            for c in range(2):
                x, y = -0.32 + c * 0.4, -0.45 + r * 0.35
                out.append(("poly", [(x, y), (x + 0.24, y), (x + 0.24, y + 0.2), (x, y + 0.2)]))
        return out
    if icon == "tower":
        return [("line", ((0.0, -0.6), (0.0, 0.6), 0.1))]
    if icon == "shield":
        return [("circle", (0.0, -0.05, 0.2))]
    return []


class Painter:
    """Draws an icon, centred at (cx, cy), `size` across, on a PIL ImageDraw or a pygame Surface."""

    def __init__(self, target, lib: str):
        self.t, self.lib = target, lib

    def _poly(self, pts, col):
        if self.lib == "pil":
            self.t.polygon(pts, fill=col)
        else:
            import pygame
            pygame.draw.polygon(self.t, col, pts)

    def _circle(self, c, r, col, width=0):
        if self.lib == "pil":
            box = [c[0] - r, c[1] - r, c[0] + r, c[1] + r]
            self.t.ellipse(box, fill=None if width else col, outline=col if width else None, width=int(width) or 1)
        else:
            import pygame
            pygame.draw.circle(self.t, col, (int(c[0]), int(c[1])), max(1, int(r)), int(width))

    def _line(self, a, b, w, col):
        if self.lib == "pil":
            self.t.line([a, b], fill=col, width=max(1, int(w)))
        else:
            import pygame
            pygame.draw.line(self.t, col, a, b, max(1, int(w)))

    def icon(self, kind: str, cx: float, cy: float, size: float, badge: bool = True, ring=None) -> None:
        col = colour(kind)
        icon = STYLE.get(kind, (None, "dot"))[1]
        h = size / 2
        if badge:
            if ring is not None:
                self._circle((cx, cy), h + max(2, size * 0.09), ring)
            self._circle((cx, cy), h, col)
        s = h * 0.95

        def P(p):
            return (cx + p[0] * s, cy + p[1] * s)
        for layer, c in ((shapes(icon), WHITE), (detail(icon), col)):
            for kind_, d in layer:
                if kind_ == "poly":
                    self._poly([P(p) for p in d], c)
                elif kind_ == "circle":
                    self._circle(P(d[:2]), d[2] * s, c)
                elif kind_ == "ring":
                    self._circle(P(d[:2]), d[2] * s, c, max(1, d[3] * s))
                elif kind_ == "line":
                    self._line(P(d[0]), P(d[1]), max(1.0, d[2] * s), c)


def occupancy_colour(free_share: float):
    """Green when there is room, amber when it fills, red when full."""
    if free_share > 0.3:
        return (52, 211, 153)
    if free_share > 0.08:
        return (250, 204, 21)
    return (244, 63, 94)


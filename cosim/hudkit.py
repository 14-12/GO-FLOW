"""The overlay's design system (12 Nov): one look for every panel, notice and label drawn over the town.

Drawn from how the best open-world games build their HUDs (docs/WORKLOG.md, 12 Nov), combined:
* little on screen, and only while it matters (Red Dead Redemption 2, The Last of Us Part II, Cyberpunk 2077: parts
  appear with what they are about and leave after): events come as notices that slide in, stay a few seconds and go;
* one notice stack in one corner, newest on top, each with its icon, a title in condensed capitals, one line, its age
  and a thin bar running out (Forza Horizon's and GTA V's feeds);
* a large title in the middle of the top for the few events worth stopping for (GTA V's mission banners);
* labels in the town that fade with distance and stand on a thin leader line to what they name (Watch Dogs 2,
  Cyberpunk 2077's scanner);
* one colour per kind of thing, used for that only (Half-Life 2's single HUD colour, Cyberpunk's colour critique):
  red emergencies, blue police, yellow buses, orange forecasts, cyan the junction agents, sky blue parking;
* two type families: Barlow Condensed in capitals for titles and numbers, Barlow for the lines under them; dark glass
  behind every text (legibility over any street).

Everything is drawn on the overlay's per-pixel-alpha canvas (cosim/overlay.AlphaPresenter, cosim/ui.ALPHA); with the
colour key (`--overlay key`) the same parts are drawn solid. Rendered surfaces are cached by content; only positions
and transparency change from one picture to the next.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import pygame

from cosim import ui
from cosim.hud import FONTS

INK = (10, 13, 21)
TEXT = (244, 246, 250)
DIM = (168, 176, 192)
FAINT = (112, 120, 138)
ACCENT = {"recovery": (255, 176, 46), "ambulance": (255, 84, 84), "accident": (255, 84, 84), "police": (84, 146, 255), "bus": (250, 204, 21),
          "forecast": (251, 146, 60), "agent": (45, 212, 191), "parking": (56, 189, 248), "good": (74, 222, 128),
          "info": (226, 232, 240)}
SS = 4  # supersampling of shapes

_fonts: dict = {}
_cache: dict = {}


def font(name: str, size: int) -> pygame.font.Font:
    key = (name, size)
    f = _fonts.get(key)
    if f is None:
        if not pygame.font.get_init():
            pygame.font.init()
        path = FONTS / name
        if not path.is_file():  # (the bold cuts came on 12 Nov; older checkouts fall back)
            path = FONTS / name.replace("ExtraBold", "SemiBold").replace("Bold", "SemiBold").replace("Regular", "Medium")
        f = _fonts[key] = pygame.font.Font(str(path), size)
    return f


def title_font(size: int) -> pygame.font.Font:
    return font("BarlowCondensed-Bold.ttf", size)


def body_font(size: int) -> pygame.font.Font:
    return font("Barlow-Medium.ttf", size)


def _cached(key, make):
    s = _cache.get(key)
    if s is None:
        if len(_cache) > 600:
            _cache.clear()
        s = _cache[key] = make()
    return s


def tracked(f: pygame.font.Font, text: str, colour, spacing: float = 1.2) -> pygame.Surface:
    """Text with letter spacing (pygame has none): titles in capitals."""
    def make():
        glyphs = [f.render(ch, True, colour) for ch in text]
        w = int(sum(g.get_width() for g in glyphs) + spacing * max(0, len(glyphs) - 1))
        s = pygame.Surface((max(w, 1), f.get_height()), pygame.SRCALPHA)
        x = 0.0
        for g in glyphs:
            s.blit(g, (int(round(x)), 0))
            x += g.get_width() + spacing
        return s
    return _cached(("tracked", id(f), text, colour, spacing), make)


def text(f: pygame.font.Font, s: str, colour) -> pygame.Surface:
    return _cached(("text", id(f), s, colour), lambda: f.render(s, True, colour))


def fit(s: pygame.Surface, width: int) -> pygame.Surface:
    """A text surface cut to `width` with a fade at its end (instead of running off the card)."""
    if s.get_width() <= width:
        return s
    out = pygame.Surface((width, s.get_height()), pygame.SRCALPHA)
    out.blit(s, (0, 0))
    if ui.ALPHA:
        fade = 28
        pa = pygame.surfarray.pixels_alpha(out)
        for i in range(fade):
            pa[width - fade + i, :] = (pa[width - fade + i, :] * (1.0 - (i + 1) / fade)).astype(pa.dtype)
        del pa
    return out


def ease_out(k: float) -> float:
    k = min(max(k, 0.0), 1.0)
    return 1.0 - (1.0 - k) ** 3


# ---------------------------------------------------------------- shapes
def glass(w: int, h: int, radius: int = 12, accent=None, bar: int = 3, alpha: int = 214) -> pygame.Surface:
    """A dark glass card: a slight vertical gradient, a hairline border, a light edge on top, an accent bar on the left
    (or none). Drawn at SS x, scaled down: smooth corners."""
    def make():
        W, H, R = w * SS, h * SS, radius * SS
        body = pygame.Surface((W, H), pygame.SRCALPHA)
        for y in range(0, H, SS):
            f = y / H
            c = (int(INK[0] + 10 * (1 - f)), int(INK[1] + 12 * (1 - f)), int(INK[2] + 16 * (1 - f)))
            pygame.draw.rect(body, (*c, int(alpha + 18 * f) if ui.ALPHA else 255), (0, y, W, SS))
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, W, H), border_radius=R)
        body.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        edge = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(edge, (255, 255, 255, 30), (0, 0, W, H), width=SS, border_radius=R)
        pygame.draw.line(edge, (255, 255, 255, 34), (R, SS), (W - R, SS), SS)
        if accent is not None and bar:
            pygame.draw.rect(edge, (*accent, 255), (0, 0, bar * SS + R, H), border_top_left_radius=R,
                             border_bottom_left_radius=R)
            pygame.draw.rect(edge, (0, 0, 0, 0), (bar * SS, 0, R, H))
        body.blit(edge, (0, 0))
        return pygame.transform.smoothscale(body, (w, h))
    return _cached(("glass", w, h, radius, accent, bar, alpha, ui.ALPHA), make)


def shadowed(card: pygame.Surface, radius: int = 12, strength: int = 120) -> pygame.Surface:
    """The card on a surface ui.MARGIN larger with its soft shadow (nothing added without per-pixel alpha)."""
    if not ui.ALPHA:
        return card
    w, h = card.get_size()
    out = ui._shadow(w, h, radius, strength).copy()
    out.blit(card, (ui.MARGIN, ui.MARGIN))
    return out


# ---------------------------------------------------------------- icons
def icon(kind: str, size: int, accent=None) -> pygame.Surface:
    """A round badge with a white glyph: the kind of event at a glance."""
    accent = accent or ACCENT.get(kind, ACCENT["info"])

    def make():
        n = size * SS
        s = pygame.Surface((n, n), pygame.SRCALPHA)
        c = n / 2
        pygame.draw.circle(s, (*accent, 255), (c, c), c)
        pygame.draw.circle(s, (255, 255, 255, 60), (c, c), c, SS)
        u = n / 2 * 0.55
        w = (255, 255, 255)
        P = lambda x, y: (c + x * u, c + y * u)  # noqa: E731
        if kind in ("ambulance", "accident_old"):
            pygame.draw.rect(s, w, (*P(-0.28, -0.9), 0.56 * u, 1.8 * u))
            pygame.draw.rect(s, w, (*P(-0.9, -0.28), 1.8 * u, 0.56 * u))
        elif kind == "accident":
            pygame.draw.polygon(s, w, [P(0, -1.0), P(0.95, 0.75), P(-0.95, 0.75)])
            pygame.draw.rect(s, accent, (*P(-0.1, -0.45), 0.2 * u, 0.7 * u))
            pygame.draw.rect(s, accent, (*P(-0.1, 0.38), 0.2 * u, 0.2 * u))
        elif kind == "police":
            pygame.draw.polygon(s, w, [P(0, -1.0), P(0.85, -0.6), P(0.7, 0.4), P(0, 1.0), P(-0.7, 0.4), P(-0.85, -0.6)])
            pygame.draw.polygon(s, accent, [P(0, -0.55), P(0.42, -0.32), P(0.34, 0.25), P(0, 0.58), P(-0.34, 0.25),
                                            P(-0.42, -0.32)])
        elif kind == "bus":
            pygame.draw.rect(s, w, (*P(-0.75, -0.95), 1.5 * u, 1.65 * u), border_radius=int(0.25 * u))
            pygame.draw.rect(s, accent, (*P(-0.55, -0.7), 1.1 * u, 0.6 * u), border_radius=int(0.08 * u))
            pygame.draw.circle(s, accent, P(-0.4, 0.35), 0.14 * u)
            pygame.draw.circle(s, accent, P(0.4, 0.35), 0.14 * u)
            pygame.draw.rect(s, w, (*P(-0.6, 0.65), 0.3 * u, 0.35 * u))
            pygame.draw.rect(s, w, (*P(0.3, 0.65), 0.3 * u, 0.35 * u))
        elif kind == "forecast":
            for i, hgt in enumerate((0.6, 1.0, 1.6)):
                x = -0.8 + i * 0.6
                pygame.draw.rect(s, w, (*P(x, 0.8 - hgt), 0.42 * u, hgt * u))
        elif kind == "agent":   # a signal head
            pygame.draw.rect(s, w, (*P(-0.45, -1.0), 0.9 * u, 2.0 * u), border_radius=int(0.3 * u))
            for i, y in enumerate((-0.6, 0.0, 0.6)):
                pygame.draw.circle(s, accent if i != 2 else (40, 40, 40), P(0, y), 0.22 * u)
        elif kind == "parking":
            pygame.draw.rect(s, w, (*P(-0.55, -0.9), 0.32 * u, 1.8 * u))
            pygame.draw.circle(s, w, P(0.05, -0.38), 0.55 * u)
            pygame.draw.circle(s, accent, P(0.05, -0.38), 0.27 * u)
            pygame.draw.rect(s, accent, (*P(-0.23, -0.65), 0.28 * u, 0.54 * u))
        elif kind == "recovery":   # a wrench
            pygame.draw.line(s, w, P(-0.6, 0.6), P(0.25, -0.25), int(0.36 * u))
            pygame.draw.circle(s, w, P(0.42, -0.42), 0.5 * u)
            pygame.draw.circle(s, accent, P(0.6, -0.6), 0.26 * u)
        elif kind == "good":
            pygame.draw.lines(s, w, False, [P(-0.65, 0.0), P(-0.15, 0.5), P(0.7, -0.5)], int(0.32 * u))
        else:
            pygame.draw.circle(s, w, (c, c), 0.35 * u)
        return pygame.transform.smoothscale(s, (size, size))
    return _cached(("icon", kind, size, accent), make)


# ---------------------------------------------------------------- notices (the feed)
@dataclass
class Notice:
    kind: str
    title: str
    line: str = ""
    t: float = field(default_factory=time.time)
    life: float = 6.0
    key: str | None = None     # a notice with the key of one on screen replaces it (no repeats)
    y: float = -1.0            # where it is drawn (eased towards its place in the stack)


class Feed:
    """Notices in one corner, newest on top: each slides in, stays `life` seconds with a bar running out, and fades.
    At most MAX on screen; a repeat (same key) refreshes the one on screen instead of adding a line."""
    W = 380
    MAX_W = 520
    MAX = 4
    IN_S, OUT_S = 0.22, 0.35

    def __init__(self):
        self.items: deque[Notice] = deque()
        self.counts: dict[str, int] = {}     # notices shown, by kind (the demo's report)
        self.f_title = title_font(19)
        self.f_line = body_font(13)
        self.f_age = font("Barlow-SemiBold.ttf", 11)

    def push(self, kind: str, title: str, line: str = "", life: float = 6.0, key: str | None = None,
             now: float | None = None) -> None:
        now = time.time() if now is None else now
        if key is not None:
            for n in list(self.items):
                if n.key == key and now - n.t < n.life:
                    if (n.kind, n.title, n.line) == (kind, title, line):
                        n.t, n.life = now - self.IN_S, life       # the same again: kept on screen longer
                        return
                    self.items.remove(n)                          # news about it: the new notice, on top
                    break
        self.items.appendleft(Notice(kind, title, line, now, life, key))
        self.counts[kind] = self.counts.get(kind, 0) + 1
        while len(self.items) > 8:
            self.items.pop()

    def busy(self, now: float) -> bool:
        """Something moves (sliding in, fading, its bar running): the overlay redraws often while so."""
        return any(now - n.t < n.life + self.OUT_S for n in self.items)

    def _card(self, n: Notice) -> pygame.Surface:
        def make():
            acc = ACCENT.get(n.kind, ACCENT["info"])
            h = 58 if n.line else 44
            tt = fit(tracked(self.f_title, n.title.upper(), TEXT, 0.8), self.MAX_W - 72)
            ln = fit(text(self.f_line, n.line, DIM), self.MAX_W - 72) if n.line else None
            w = min(self.MAX_W, max(self.W, 72 + max(tt.get_width(), ln.get_width() if ln else 0)))
            card = glass(w, h, radius=10, accent=acc).copy()
            card.blit(icon(n.kind, 30, acc), (14, (h - 30) // 2))
            card.blit(tt, (56, 7 if n.line else (h - tt.get_height()) // 2))
            if ln is not None:
                card.blit(ln, (57, 7 + tt.get_height() - 1))
            return shadowed(card, 10)
        return _cached(("notice", n.kind, n.title, n.line, ui.ALPHA, self.W), make)

    def draw(self, screen: pygame.Surface, x_right: int, y_top: int, now: float | None = None) -> list:
        now = time.time() if now is None else now
        rects = []
        live = [n for n in self.items if now - n.t < n.life + self.OUT_S][:self.MAX]
        y = float(y_top)
        m = ui.MARGIN if ui.ALPHA else 0
        for n in live:
            card = self._card(n)
            target = y
            n.y = target if n.y < 0 else n.y + (target - n.y) * 0.35   # others make room smoothly
            age = now - n.t
            k_in = ease_out(age / self.IN_S)
            k_out = 1.0 - min(1.0, max(0.0, (age - n.life) / self.OUT_S))
            a = k_in * k_out
            if a <= 0.01:
                continue
            x = x_right - card.get_width() + m + int((1.0 - k_in) * 46)
            if a < 0.99:
                card = card.copy()
                card.set_alpha(int(255 * a))
            r = screen.blit(card, (x, int(n.y) - m))
            rects.append(r)
            # the bar running out, under the text
            if ui.ALPHA and age < n.life:
                acc = ACCENT.get(n.kind, ACCENT["info"])
                bw = card.get_width() - 2 * m - 28
                left = int(bw * (1.0 - age / n.life))
                bar = pygame.Rect(x + m + 14, int(n.y) + card.get_height() - 2 * m - 5, bw, 2)
                pygame.draw.rect(screen, (*acc, int(70 * a)), bar)
                pygame.draw.rect(screen, (*acc, int(230 * a)), (bar.x, bar.y, left, 2))
            y += card.get_height() - 2 * m + 8
        return rects


# ---------------------------------------------------------------- the banner (the few events worth stopping for)
class Banner:
    """A large title in the middle of the top: a small tag line in the kind's colour, the title in heavy condensed
    capitals, a rule that opens from the middle. One at a time, at most one every MIN_GAP_S."""
    LIFE = 3.4
    MIN_GAP_S = 20.0

    def __init__(self):
        self.cur = None
        self.last = -1e9
        self.f_tag = font("Barlow-SemiBold.ttf", 14)
        self.f_title = font("BarlowCondensed-ExtraBold.ttf", 54)
        self.f_sub = body_font(16)

    def show(self, kind: str, tag: str, title: str, sub: str = "", now: float | None = None, force: bool = False) -> bool:
        now = time.time() if now is None else now
        if not force and now - self.last < self.MIN_GAP_S:
            return False
        self.cur, self.last = (kind, tag, title, sub, now), now
        self.shown = getattr(self, "shown", 0) + 1
        return True

    def busy(self, now: float) -> bool:
        return self.cur is not None and now - self.cur[4] < self.LIFE

    def draw(self, screen: pygame.Surface, size, now: float | None = None, y: int = 96) -> list:
        now = time.time() if now is None else now
        if self.cur is None:
            return []
        kind, tag, title, sub, t0 = self.cur
        age = now - t0
        if age >= self.LIFE:
            self.cur = None
            return []
        acc = ACCENT.get(kind, ACCENT["info"])
        a = min(ease_out(age / 0.25), 1.0 - max(0.0, (age - self.LIFE + 0.5) / 0.5))

        def make():
            tg = tracked(self.f_tag, tag.upper(), acc, 3.0)
            tt = tracked(self.f_title, title.upper(), TEXT, 1.5)
            sb = text(self.f_sub, sub, (214, 220, 232)) if sub else None
            w = max(tg.get_width(), tt.get_width(), sb.get_width() if sb else 0) + 120
            h = tg.get_height() + tt.get_height() + (sb.get_height() + 6 if sb else 0) + 26
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            if ui.ALPHA:   # a dark band behind the text, fading out to the sides (no hard card)
                band = pygame.Surface((w // 8, 1), pygame.SRCALPHA)
                for i in range(w // 8):
                    f = 1.0 - abs(i - w / 16) / (w / 16)
                    band.set_at((i, 0), (*INK, int(205 * min(1.0, f * 2.2))))
                s.blit(pygame.transform.smoothscale(band, (w, h)), (0, 0))
            yy = 6
            s.blit(tg, ((w - tg.get_width()) // 2, yy))
            yy += tg.get_height() - 2
            sh = tracked(self.f_title, title.upper(), (0, 0, 0), 1.5).copy()
            sh.set_alpha(120)
            s.blit(sh, ((w - tt.get_width()) // 2 + 2, yy + 2))
            s.blit(tt, ((w - tt.get_width()) // 2, yy))
            yy += tt.get_height() + 2
            if sb:
                s.blit(sb, ((w - sb.get_width()) // 2, yy + 6))
            return s
        surf = _cached(("banner", kind, tag, title, sub, ui.ALPHA), make)
        w, h = surf.get_size()
        x = (size[0] - w) // 2
        out = surf
        if a < 0.99:
            out = surf.copy()
            out.set_alpha(int(255 * max(a, 0.0)))
        rects = [screen.blit(out, (x, y))]
        # the rule opening from the middle
        k = ease_out(age / 0.5)
        half = int((w - 160) / 2 * k)
        ry = y + h - 4
        pygame.draw.rect(screen, (*acc, int(220 * max(a, 0.0))) if ui.ALPHA else acc,
                         (size[0] // 2 - half, ry, 2 * half, 2))
        rects.append(pygame.Rect(x, ry, w, 3))
        return rects


# ---------------------------------------------------------------- labels in the town
def tag(kind: str, title: str, sub: str = "", accent=None, glyph=None, sub_colour=None) -> pygame.Surface:
    """A label for a thing in the town: an icon, a title in capitals, a line under it, on glass with the kind's bar.
    glyph: draw(surface, cx, cy, size) for an icon of its own (the places': cosim/place_icons.py)."""
    acc = accent or ACCENT.get(kind, ACCENT["info"])

    def make():
        tf, bf = title_font(17), body_font(12)
        tt = tracked(tf, title.upper(), TEXT, 0.7)
        sb = text(bf, sub, sub_colour or DIM) if sub else None
        h = 44 if sub else 32
        isz = 24 if sub else 20
        w = 12 + isz + 10 + max(tt.get_width(), sb.get_width() if sb else 0) + 14
        card = glass(w, h, radius=8, accent=acc, bar=3).copy()
        if glyph is not None:
            glyph(card, 12 + isz / 2, h / 2, isz)
        else:
            card.blit(icon(kind, isz, acc), (12, (h - isz) // 2))
        x = 12 + isz + 10
        card.blit(tt, (x, 4 if sub else (h - tt.get_height()) // 2))
        if sb:
            card.blit(sb, (x + 1, 4 + tt.get_height() - 2))
        return shadowed(card, 8, 110)
    return _cached(("tag", kind, title, sub, acc, glyph is not None, sub_colour, ui.ALPHA), make)


def place_tag(screen, surf: pygame.Surface, anchor, tip, accent, alpha: float = 1.0) -> list:
    """`surf` standing on a leader line from `tip` (its foot) down to `anchor` (the thing), a dot at the anchor;
    `alpha` fades it with distance."""
    m = ui.MARGIN if ui.ALPHA else 0
    if alpha < 0.99:
        surf = surf.copy()
        surf.set_alpha(int(255 * alpha))
    r = surf.get_rect(midbottom=(int(tip[0]), int(tip[1]) + m))
    out = []
    la = int(200 * alpha)
    if ui.ALPHA:
        pygame.draw.aaline(screen, (*accent, la), (tip[0], tip[1]), anchor)
        pygame.draw.aaline(screen, (*accent, la), (tip[0] + 1, tip[1]), (anchor[0] + 1, anchor[1]))
    else:
        pygame.draw.line(screen, accent, tip, anchor, 2)
    out.append(pygame.Rect(min(tip[0], anchor[0]) - 2, min(tip[1], anchor[1]) - 2, abs(tip[0] - anchor[0]) + 5,
                           abs(tip[1] - anchor[1]) + 5))
    pygame.draw.circle(screen, (*accent, int(255 * alpha)) if ui.ALPHA else accent, anchor, 4)
    pygame.draw.circle(screen, (255, 255, 255, int(200 * alpha)) if ui.ALPHA else (255, 255, 255), anchor, 4, 1)
    out.append(pygame.Rect(anchor[0] - 5, anchor[1] - 5, 11, 11))
    out.append(screen.blit(surf, r))
    return out


def distance_alpha(d: float, near: float, far: float) -> float:
    """1 up to `near` metres, 0 from `far`, smooth between."""
    if d <= near:
        return 1.0
    if d >= far:
        return 0.0
    k = (d - near) / (far - near)
    return 1.0 - k * k * (3 - 2 * k)


# ---------------------------------------------------------------- the status chips (under the clock)
def chips(items: list[tuple[str, tuple | None]], h: int = 26) -> pygame.Surface:
    """A row of small glass chips: (text, dot colour or None)."""
    def make():
        f = font("Barlow-SemiBold.ttf", 12)
        parts = []
        for s, dot in items:
            t = tracked(f, s.upper(), TEXT if dot else DIM, 0.8)
            w = t.get_width() + (30 if dot else 20)
            c = glass(w, h, radius=h // 2, accent=None, alpha=196).copy()
            if dot:
                pygame.draw.circle(c, dot, (14, h // 2), 4)
            c.blit(t, (24 if dot else 10, (h - t.get_height()) // 2))
            parts.append(c)
        w = sum(p.get_width() for p in parts) + 6 * (len(parts) - 1)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        x = 0
        for p in parts:
            s.blit(p, (x, 0))
            x += p.get_width() + 6
        return s
    return _cached(("chips", tuple(items), h, ui.ALPHA), make)


# ---------------------------------------------------------------- the key prompt (bottom middle)
def prompt(spec: str) -> pygame.Surface:
    """The viewer's prompt ("H   Inspect traffic camera ...      F   Follow car ...") as key caps and their action,
    on one glass bar."""
    def make():
        segs = [seg.strip() for seg in spec.split("      ") if seg.strip()]
        kf, tf = title_font(17), font("Barlow-SemiBold.ttf", 15)
        parts = []
        for seg in segs:
            key, _, act = seg.partition(" ")
            k = text(kf, key, INK)
            cap = pygame.Surface((max(26, k.get_width() + 14) * SS, 26 * SS), pygame.SRCALPHA)
            pygame.draw.rect(cap, (246, 247, 250, 255), cap.get_rect(), border_radius=6 * SS)
            pygame.draw.rect(cap, (190, 196, 210, 255), (0, cap.get_height() - 3 * SS, cap.get_width(), 3 * SS),
                             border_bottom_left_radius=6 * SS, border_bottom_right_radius=6 * SS)
            cap = pygame.transform.smoothscale(cap, (cap.get_width() // SS, 26))
            cap.blit(k, ((cap.get_width() - k.get_width()) // 2, (23 - k.get_height()) // 2 + 1))
            a = text(tf, act.strip(), TEXT)
            parts.append((cap, a))
        w = sum(c.get_width() + 10 + a.get_width() for c, a in parts) + 26 * (len(parts) - 1) + 28
        h = 40
        bar = glass(w, h, radius=h // 2, accent=None, alpha=206).copy()
        x = 14
        for c, a in parts:
            bar.blit(c, (x, (h - 26) // 2))
            x += c.get_width() + 10
            bar.blit(a, (x, (h - a.get_height()) // 2))
            x += a.get_width() + 26
        return shadowed(bar, h // 2, 110)
    return _cached(("prompt", spec, ui.ALPHA), make)


def keys(spec: str) -> pygame.Surface:
    """The key help ("F5 pause    F6 / F7  -1 h / +1 h    ...") as small key caps and their action on one glass bar."""
    def make():
        kf, tf = title_font(14), font("Barlow-Medium.ttf", 13)
        items = []
        for seg in [x.strip() for x in spec.split("    ") if x.strip()]:
            toks = seg.split(" ")
            ks = []
            while toks and (toks[0].isupper() or toks[0] in ("/", "Esc") or toks[0][:1] == "F" and toks[0][1:].isdigit()):
                ks.append(toks.pop(0))
            items.append(([k for k in ks if k != "/"], " ".join(toks).strip()))
        parts = []
        for ks, act in items:
            caps = []
            for k in ks:
                t = text(kf, k, INK)
                c = pygame.Surface((max(22, t.get_width() + 10) * SS, 20 * SS), pygame.SRCALPHA)
                pygame.draw.rect(c, (236, 239, 245, 255), c.get_rect(), border_radius=4 * SS)
                c = pygame.transform.smoothscale(c, (c.get_width() // SS, 20))
                c.blit(t, ((c.get_width() - t.get_width()) // 2, (20 - t.get_height()) // 2))
                caps.append(c)
            parts.append((caps, text(tf, act, DIM)))
        w = sum(sum(c.get_width() + 3 for c in cs) + 5 + a.get_width() for cs, a in parts) + 16 * (len(parts) - 1) + 24
        h = 32
        bar = glass(w, h, radius=h // 2, accent=None, alpha=200).copy()
        x = 12
        for cs, a in parts:
            for c in cs:
                bar.blit(c, (x, (h - 20) // 2))
                x += c.get_width() + 3
            x += 5
            bar.blit(a, (x, (h - a.get_height()) // 2))
            x += a.get_width() + 16
        return shadowed(bar, h // 2, 100)
    return _cached(("keys", spec, ui.ALPHA), make)


AGENT_KIND = {"bus": "bus", "ambulance": "ambulance", "corridor": "ambulance", "forecast": "forecast",
              "gate": "forecast"}


INCIDENT_KIND = {"crash": "accident", "police en route": "police", "secured": "police", "recovery": "recovery",
                 "clearing": "good", "closed": "good"}


class Events:
    """Turns what the town does into notices and banners (the viewer calls `update` before drawing): accidents as they
    happen and as they are cleared, the junction agents' decisions for buses and ambulances near you, the forecasters'
    alerts. Each source is read for what is new since the last call."""
    AGENT_KINDS = {"bus": "bus", "ambulance": "ambulance"}
    AGENT_M = 700.0       # the agents' decisions this close (as their list did)
    BANNER_M = 450.0      # an accident this close gets the banner; further, a notice

    def __init__(self, feed: Feed, banner: Banner):
        self.feed, self.banner = feed, banner
        self.inc: dict = {}
        self.ev_t = None
        self.alert_t = None

    @staticmethod
    def _far(d: float) -> str:
        return f"{d:.0f} m" if d < 1000 else f"{d / 1000:.1f} km"

    def update(self, loc, markers: list, events: list, centres: dict, alerts: list, describe, title_of=None) -> None:
        """loc: the view (CARLA x, y); markers: the accident markers (cosim/incident_scene); events: the agents'
        decisions; centres: junction -> CARLA (x, y, ...); alerts: the forecasters' (cosim/goflow_prediction)."""
        now = time.time()
        live = set()
        for m in markers:
            live.add(m["id"])
            prev = self.inc.get(m["id"])
            if prev == m["status"]:
                continue
            self.inc[m["id"]] = m["status"]
            d = math.dist((loc.x, loc.y), (m["x"], m["y"]))
            what = describe(m)
            kind = INCIDENT_KIND.get(m["phase"], "accident")
            if prev is None and d < self.BANNER_M and m.get("kind") != "breakdown" \
                    and self.banner.show("accident", f"{m.get('label', '')}  ·  {self._far(d)}".strip(" ·"),
                                         m["status"], what, now=now):
                continue
            self.feed.push(kind, m["status"].capitalize(), f"{what}  ·  {self._far(d)}", life=7.0,
                           key=f"inc{m['id']}", now=now)
        for i in [i for i in self.inc if i not in live]:
            self.inc.pop(i)
            self.feed.push("good", "Road cleared", "the accident's vehicles are gone", life=5.0, key=f"inc{i}", now=now)
        new = [e for e in events if self.ev_t is not None and e["t"] > self.ev_t]
        if events:
            self.ev_t = max([e["t"] for e in events] + [self.ev_t or -1e18])
        elif self.ev_t is None:
            self.ev_t = -1e18
        for e in new:
            kind = self.AGENT_KINDS.get(e["kind"])
            c = centres.get(e["tls"])
            if kind is None or c is None:
                continue
            d = math.dist((loc.x, loc.y), (c[0], c[1]))
            if d > self.AGENT_M:
                continue
            head, _, rest = e["text"].partition(":")
            line = f"Junction {e['tls']}  ·  {self._far(d)}"
            if rest.strip():
                line = rest.strip()[:1].upper() + rest.strip()[1:] + "  ·  " + line
            self.feed.push(kind, head[:1].upper() + head[1:], line, life=5.0, key=f"{e['tls']}-{kind}", now=now)
        fresh = [a for a in alerts if self.alert_t is not None and a["t"] > self.alert_t]
        if alerts:
            self.alert_t = max([a["t"] for a in alerts] + [self.alert_t or -1e18])
        elif self.alert_t is None:
            self.alert_t = -1e18
        for a in fresh:
            t = title_of(a) if title_of else "Long queue likely"
            p = a.get("p")
            line = f"Approach {a['node']}" + (f"  ·  {p * 100:.0f} % chance" if p is not None else "")
            self.feed.push("forecast", t[:1].upper() + t[1:].lower(), line, life=7.0, key=f"fc{a['node']}", now=now)

    def busy(self, now: float) -> bool:
        return self.feed.busy(now) or self.banner.busy(now)

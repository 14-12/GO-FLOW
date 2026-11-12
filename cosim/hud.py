"""The on-screen clock of the live viewer (cosim/live.py), drawn with pygame.

Top left: a glass card with a 24-hour sun dial (midnight at the bottom, noon at the top, the sky's
colour round the ring, the part of the day already gone lit up, a sun or a moon riding on it),
the time in large digits, the weekday, what the traffic is doing now (a coloured pill), and the
day's demand profile as 24 small bars with the present hour lit. Below it a thin status strip.
Shapes are drawn at 3x and scaled down (smooth edges); text is drawn at full size.

Fonts: Barlow and Barlow Condensed (SIL Open Font License, assets/fonts/OFL.txt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pygame

from cosim import daily, ui

FONTS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
SS = 3  # supersampling of the shapes

# Sky colour through the day: (hour, rgb). Night blue, dawn orange, day blue, golden hour, dusk.
SKY = ((0.0, (24, 32, 74)), (4.0, (34, 38, 92)), (5.2, (120, 78, 140)), (6.0, (255, 150, 96)),
       (7.5, (130, 190, 255)), (12.0, (88, 172, 255)), (17.0, (120, 186, 255)), (19.5, (255, 186, 102)),
       (21.0, (236, 98, 112)), (22.0, (70, 52, 120)), (24.0, (24, 32, 74)))

TRAFFIC = ((0.0, (86, 214, 160)), (0.45, (120, 200, 255)), (0.7, (255, 196, 84)), (0.9, (255, 110, 92)),
           (1.01, (255, 84, 84)))


def _lerp(a, b, f):
    return tuple(int(round(x + (y - x) * f)) for x, y in zip(a, b))


def _ramp(table, x):
    for (x0, c0), (x1, c1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            return _lerp(c0, c1, (x - x0) / (x1 - x0) if x1 > x0 else 0.0)
    return table[-1][1]


def sky_colour(hour: float):
    return _ramp(SKY, hour % 24.0)


def traffic_colour(demand: float):
    return _ramp(TRAFFIC, max(0.0, min(demand, 1.0)))


def time_of_day(hour: float, sun_elevation: float) -> str:
    """The time of day in words, from the clock and the sun (degrees above the horizon): the dawn,
    sunrise, sunset and dusk words follow the sun itself, so they move with the season."""
    h = hour % 24.0
    morning = h < 12.0
    if -6.0 <= sun_elevation < 0.0:
        return "DAWN" if morning else "DUSK"
    if 0.0 <= sun_elevation < 6.0:
        return "SUNRISE" if morning else "SUNSET"
    if sun_elevation < -6.0:
        if h >= 23.5 or h < 0.5:
            return "MIDNIGHT"
        return "LATE NIGHT" if h < 4.0 else ("EARLY MORNING" if morning else "NIGHT")
    if h < 11.0:
        return "MORNING"
    if h < 14.0:
        return "MIDDAY"
    if h < 17.5:
        return "AFTERNOON"
    return "EVENING"


def traffic_label(hour: float) -> str:
    """What the traffic is doing (the pill), in traffic words: the time of day has its own word in the dial."""
    h = hour % 24.0
    if 7.0 <= h < 9.0:
        return "MORNING RUSH"
    if 16.0 <= h < 19.0:
        return "EVENING RUSH"
    d = daily.demand_factor(h)
    if d < 0.2:
        return "LIGHT TRAFFIC"
    if d < 0.5:
        return "MODERATE TRAFFIC"
    return "BUSY TRAFFIC"


def dial_angle(hour: float) -> float:
    """Screen angle (radians, y down) of a time on the dial: midnight at the bottom, 06:00 on
    the left, noon at the top, 18:00 on the right - the way the sun goes for someone facing south."""
    a = math.pi / 2 + (hour % 24.0) / 24.0 * 2 * math.pi  # 0 h: pi/2 = straight down (y down)
    return a


def _font(name: str, size: int) -> pygame.font.Font:
    path = FONTS / name
    return pygame.font.Font(str(path) if path.is_file() else None, size)


def speed_label(speed: float) -> str:
    """The card's line under the weekday: how fast the day goes ("1 MIN = 1 HOUR" at 60, "2 MIN = 1 HOUR" at 30)."""
    if speed <= 1.0001:
        return "REAL TIME"
    m = 60.0 / speed
    return f"{m:g} MIN = 1 HOUR" if abs(m - round(m, 1)) < 1e-6 else f"{speed:g} X REAL TIME"


SIGNALS = {"coordinated": "fixed-cycle signals", "actuated": "actuated signals", "agents_observed": "junction agents",
           "agents": "junction agents (exact counts)", "default": "network's own signals"}


def setup_text(controller: str, demand_scale: float = 1.0) -> str:
    """(9 Nov) What runs the town, for the HUD strip: who runs the signals (--controller) and, when it is not the
    town's own, the traffic level (--demand-scale), so a picture says what it was taken under."""
    import re

    base = re.split(r"[-+@]", controller, maxsplit=1)[0]
    goflow = base.endswith("_goflow")
    base = base.removesuffix("_goflow")
    text = SIGNALS.get(base, controller)
    if goflow:
        text += " + GO-FLOW priority"
    if "+forecast" in controller:
        text += " + forecasts"
    if "+bus" in controller:
        text += " + bus priority"
    if abs(demand_scale - 1.0) > 1e-9:
        text += f"   ·   traffic x{demand_scale:g}"
    return text


@dataclass
class HudState:
    game_s: float
    vehicles: int = 0
    target: int = 0
    paused: bool = False
    fps: float = 0.0
    sun_elevation: float = 0.0
    camera: str = ""
    status: str = ""        # replaces LIVE / PAUSED in the strip (e.g. LOADING)
    note: str = ""          # a short message (e.g. "Screenshot saved"), shown in the strip
    setup: str = ""         # (9 Nov) who runs the signals and the traffic level (setup_text), shown in the strip
    help_alpha: int = 0     # 0..255: the key help at the bottom
    speed: float = daily.GAME_SPEED  # game seconds per real second (--clock-speed): the card's "2 MIN = 1 HOUR"

    @property
    def hour(self) -> float:
        return (self.game_s % daily.DAY_S) / 3600.0

    @property
    def day(self) -> int:
        return int(self.game_s // daily.DAY_S) + 1

    @property
    def weekday(self) -> str:
        return daily.WEEKDAYS[(self.day - 1) % 7]

    @property
    def hhmm(self) -> str:
        m = int((self.game_s % daily.DAY_S) // 60)
        return f"{m // 60:02d}:{m % 60:02d}"


class Hud:
    W, H = 432, 168  # the card, at 1x

    def __init__(self):
        if not pygame.font.get_init():
            pygame.font.init()
        self.f_time = _font("BarlowCondensed-SemiBold.ttf", 76)
        self.f_day_n = _font("BarlowCondensed-SemiBold.ttf", 30)
        self.f_small = _font("Barlow-SemiBold.ttf", 12)
        self.f_label = _font("Barlow-SemiBold.ttf", 13)
        self.f_week = _font("BarlowCondensed-Medium.ttf", 21)
        self.f_strip = _font("Barlow-Medium.ttf", 13)
        self.f_title = _font("BarlowCondensed-SemiBold.ttf", 54)
        self.f_sub = _font("Barlow-Medium.ttf", 16)
        self._cache_key = None
        self._card = None
        # the weekday column stands right of the widest time, so it never moves or runs off the card
        self.col_x = 182 + max(self.f_time.size(f"{d}{d}:{d}{d}")[0] for d in "0123456789") + 10
        widest = max(self._spaced(self.f_small, t, (0, 0, 0), 1.6).get_width() for t in ("1 MIN = 1 HOUR", "2.5 MIN = 1 HOUR", "PAUSED"))
        self.W = max(self.W, self.col_x + widest + 18)

    # ------------------------------------------------------------ text helpers
    @staticmethod
    def _spaced(font, text, colour, spacing=1.5):
        """Text with letter spacing (pygame has none)."""
        glyphs = [font.render(ch, True, colour) for ch in text]
        w = int(sum(g.get_width() for g in glyphs) + spacing * (len(glyphs) - 1))
        surf = pygame.Surface((max(w, 1), font.get_height()), pygame.SRCALPHA)
        x = 0.0
        for g in glyphs:
            surf.blit(g, (int(x), 0))
            x += g.get_width() + spacing
        return surf

    @staticmethod
    def _shadowed(font, text, colour, shadow=(0, 0, 0, 110), off=2):
        main = font.render(text, True, colour)
        sh = font.render(text, True, shadow[:3])
        sh.set_alpha(shadow[3])
        surf = pygame.Surface((main.get_width() + off, main.get_height() + off), pygame.SRCALPHA)
        surf.blit(sh, (off, off))
        surf.blit(main, (0, 0))
        return surf

    # ------------------------------------------------------------ the card
    def _shapes(self, st: HudState) -> pygame.Surface:
        """Everything but the text, drawn at SS x and scaled down."""
        W, H = self.W * SS, self.H * SS
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        r = 20 * SS
        sky = sky_colour(st.hour)
        # glass: dark, slightly lighter at the top, tinted by the sky at the dial's side
        body = pygame.Surface((W, H), pygame.SRCALPHA)
        for y in range(0, H, SS):
            f = y / H
            c = _lerp((22, 27, 40), (10, 12, 20), f)
            pygame.draw.rect(body, (*c, int(196 - 20 * f)), (0, y, W, SS))
        glow = pygame.Surface((W, H), pygame.SRCALPHA)
        cx, cy = 84 * SS, H // 2
        for k in range(28, 0, -1):
            rad = int(k * 6 * SS)
            pygame.draw.circle(glow, (*sky, int(5 + (28 - k) * 0.9)), (cx, cy), rad)
        body.blit(glow, (0, 0))
        mask = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, W, H), border_radius=r)
        body.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        s.blit(body, (0, 0))
        # pygame's draw functions replace pixels instead of blending, so every shape below goes
        # on its own transparent layer, which is then blended onto the glass
        ov = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(ov, (255, 255, 255, 38), (0, 0, W, H), width=SS, border_radius=r)
        pygame.draw.line(ov, (255, 255, 255, 22), (166 * SS, 18 * SS), (166 * SS, H - 18 * SS), SS)

        # the 24-hour dial
        R, T = 58 * SS, 8 * SS
        steps = 240
        now_a = st.hour / 24.0
        for i in range(steps):
            f0, f1 = i / steps, (i + 1) / steps
            col = sky_colour(f0 * 24)
            alpha = 235 if f0 <= now_a else 64
            a0, a1 = dial_angle(f0 * 24), dial_angle(f1 * 24) + 0.004
            pts = [(cx + (R + T / 2) * math.cos(a0), cy + (R + T / 2) * math.sin(a0)),
                   (cx + (R + T / 2) * math.cos(a1), cy + (R + T / 2) * math.sin(a1)),
                   (cx + (R - T / 2) * math.cos(a1), cy + (R - T / 2) * math.sin(a1)),
                   (cx + (R - T / 2) * math.cos(a0), cy + (R - T / 2) * math.sin(a0))]
            pygame.draw.polygon(ov, (*col, alpha), pts)
        # hour ticks every 3 h, longer at 0/6/12/18
        for h in range(0, 24, 3):
            a = dial_angle(h)
            l0 = R - T / 2 - (8 if h % 6 == 0 else 5) * SS
            l1 = R - T / 2 - 3 * SS
            pygame.draw.line(ov, (255, 255, 255, 130 if h % 6 == 0 else 60),
                             (cx + l0 * math.cos(a), cy + l0 * math.sin(a)),
                             (cx + l1 * math.cos(a), cy + l1 * math.sin(a)), SS)
        # the sun or the moon on the ring
        a = dial_angle(st.hour)
        mx, my = cx + R * math.cos(a), cy + R * math.sin(a)
        g = 34 * SS  # half size of the body's own layer
        body_l = pygame.Surface((2 * g, 2 * g), pygame.SRCALPHA)
        if st.sun_elevation > -3.0:
            for k in range(10, 0, -1):  # glow: larger and fainter outwards
                pygame.draw.circle(body_l, (255, 206, 92, int(10 + (10 - k) * 9)), (g, g), (10 + k * 2.2) * SS)
            pygame.draw.circle(body_l, (255, 222, 120), (g, g), 10 * SS)
            pygame.draw.circle(body_l, (255, 244, 205), (g - 2 * SS, g - 2 * SS), 5 * SS)
        else:
            for k in range(8, 0, -1):
                pygame.draw.circle(body_l, (180, 200, 255, int(8 + (8 - k) * 7)), (g, g), (10 + k * 2.0) * SS)
            pygame.draw.circle(body_l, (20, 24, 40), (g, g), 10 * SS)  # the dark disc under the crescent
            moon = pygame.Surface((2 * g, 2 * g), pygame.SRCALPHA)
            pygame.draw.circle(moon, (234, 240, 255), (g, g), 10 * SS)
            pygame.draw.circle(moon, (0, 0, 0, 0), (g + 5 * SS, g - 4 * SS), 9 * SS)  # cut: a crescent
            body_l.blit(moon, (0, 0))
        ov.blit(body_l, (mx - g, my - g))
        # demand bars
        bx, by, bw, bh = 184 * SS, 120 * SS, (self.W - 204) * SS, 26 * SS
        n = 24
        gap = 2 * SS
        w = (bw - gap * (n - 1)) / n
        cur = int(st.hour) % 24
        for h in range(n):
            v = daily.PROFILE[h]
            hh = max(2 * SS, v * bh)
            x = bx + h * (w + gap)
            col = (*traffic_colour(v), 255) if h == cur else (255, 255, 255, 50 if h > cur else 105)
            pygame.draw.rect(ov, col, (x, by + bh - hh, w, hh), border_radius=int(SS * 1.5))
        # cursor under the bars, at the exact time
        x = bx + (st.hour / 24.0) * bw
        pygame.draw.polygon(ov, (255, 255, 255, 230), [(x, by + bh + 3 * SS), (x - 4 * SS, by + bh + 8 * SS),
                                                       (x + 4 * SS, by + bh + 8 * SS)])
        s.blit(ov, (0, 0))
        return pygame.transform.smoothscale(s, (self.W, self.H))

    def card(self, st: HudState) -> pygame.Surface:
        key = (int(st.game_s // 60), st.paused, st.sun_elevation > -3.0, st.speed)
        if key == self._cache_key and self._card is not None:
            return self._card
        card = self._shapes(st)
        # inside the dial
        cx, cy = 84, self.H // 2
        word = time_of_day(st.hour, st.sun_elevation)
        lab = self._spaced(self.f_label, word, (240, 244, 252), 1.4)
        if lab.get_width() > 96:  # the longest words, a little tighter
            lab = self._spaced(self.f_small, word, (240, 244, 252), 1.1)
        card.blit(lab, (cx - lab.get_width() // 2, cy - lab.get_height() // 2))
        # time
        t = self._shadowed(self.f_time, st.hhmm, (255, 255, 255))
        card.blit(t, (182, 4))
        wk = self._spaced(self.f_week, st.weekday[:3], (226, 232, 245), 2.2)
        tx = self.col_x
        card.blit(wk, (tx, 24))
        sub = self._spaced(self.f_small, "PAUSED" if st.paused else speed_label(st.speed), (255, 196, 84) if st.paused
                           else (150, 160, 182), 1.6)
        card.blit(sub, (tx, 50))
        # traffic pill
        dem = daily.demand_factor(st.hour)
        col = traffic_colour(dem)
        label = self._spaced(self.f_label, traffic_label(st.hour), (255, 255, 255), 1.8)
        pw, ph = label.get_width() + 34, 24
        pill = pygame.Surface((pw * SS, ph * SS), pygame.SRCALPHA)
        pygame.draw.rect(pill, (*col, 62), (0, 0, pw * SS, ph * SS), border_radius=ph * SS // 2)
        pygame.draw.rect(pill, (*col, 170), (0, 0, pw * SS, ph * SS), width=SS, border_radius=ph * SS // 2)
        pygame.draw.circle(pill, col, (13 * SS, ph * SS // 2), 4 * SS)
        pill = pygame.transform.smoothscale(pill, (pw, ph))
        pill.blit(label, (24, (ph - label.get_height()) // 2))
        card.blit(pill, (184, 86))
        pct = self.f_small.render(f"{dem * 100:.0f}% of peak", True, (170, 180, 200))
        card.blit(pct, (184 + pw + 10, 86 + (ph - pct.get_height()) // 2))
        # hour labels under the bars
        for h in (0, 6, 12, 18):
            lab = self.f_small.render(f"{h:02d}", True, (120, 130, 150))
            card.blit(lab, (184 + h / 24 * (self.W - 204) - (0 if h == 0 else lab.get_width() // 2), 154 - 2))
        self._card, self._cache_key = card, key
        return card

    def strip(self, st: HudState) -> pygame.Surface:
        dot = (255, 196, 84) if st.paused else (255, 84, 84)
        parts = [st.status or ("PAUSED" if st.paused else "LIVE"), f"{st.vehicles:,} vehicles in the city".replace(",", " ")]
        if st.setup:
            parts.append(st.setup)
        if st.camera:
            parts.append(st.camera)
        if st.fps:
            parts.append(f"{st.fps:.1f} fps")
        if st.note:
            parts.append(st.note)
        text = self.f_strip.render("   ·   ".join(parts), True, (226, 232, 245))
        w, h = text.get_width() + 40, 28
        s = pygame.Surface((w * SS, h * SS), pygame.SRCALPHA)
        pygame.draw.rect(s, (12, 15, 24, 170), (0, 0, w * SS, h * SS), border_radius=h * SS // 2)
        pygame.draw.rect(s, (255, 255, 255, 30), (0, 0, w * SS, h * SS), width=SS, border_radius=h * SS // 2)
        pygame.draw.circle(s, dot, (16 * SS, h * SS // 2), 4 * SS)
        s = pygame.transform.smoothscale(s, (w, h))
        s.blit(text, (28, (h - text.get_height()) // 2))
        return s

    def help(self, alpha: int) -> pygame.Surface | None:
        if alpha <= 0:
            return None
        text = self.f_strip.render("P pause    Left / Right  -1 h / +1 h    C next camera    H hide clock    "
                                   "F12 screenshot    Esc quit (the time is saved)", True, (226, 232, 245))
        w, h = text.get_width() + 36, 30
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(s, (12, 15, 24, 165), (0, 0, w, h), border_radius=h // 2)
        s.blit(text, (18, (h - text.get_height()) // 2))
        s.set_alpha(alpha)
        return s

    def draw(self, screen: pygame.Surface, st: HudState, margin: int = 22) -> None:
        if ui.ALPHA:  # the card's soft shadow on the town
            screen.blit(ui._shadow(self.W, self.H, 20, 130), (margin - ui.MARGIN, margin - ui.MARGIN))
        screen.blit(self.card(st), (margin, margin))
        screen.blit(self.strip(st), (margin, margin + self.H + 10))
        hp = self.help(st.help_alpha)
        if hp is not None:
            screen.blit(hp, ((screen.get_width() - hp.get_width()) // 2, screen.get_height() - hp.get_height() - 18))

    # ------------------------------------------------------------ loading screen
    def loading(self, screen: pygame.Surface, st: HudState, stage: str, progress: float, t: float,
                title: str = "TOWN 12", subtitle: str = "") -> None:
        W, H = screen.get_size()
        top, bottom = _lerp(sky_colour(st.hour), (8, 10, 18), 0.72), (6, 8, 14)
        for y in range(0, H, 4):
            pygame.draw.rect(screen, _lerp(top, bottom, y / H), (0, y, W, 4))
        tt = self._spaced(self.f_title, title, (255, 255, 255), 6.0)
        screen.blit(tt, ((W - tt.get_width()) // 2, int(H * 0.40)))
        if subtitle:
            sb = self.f_sub.render(subtitle, True, (170, 182, 205))
            screen.blit(sb, ((W - sb.get_width()) // 2, int(H * 0.40) + tt.get_height() + 6))
        bw, bh = min(520, W - 80), 4
        bx, by = (W - bw) // 2, int(H * 0.62)
        pygame.draw.rect(screen, (58, 64, 80), (bx, by, bw, bh), border_radius=2)
        if progress >= 0:
            pygame.draw.rect(screen, (255, 214, 102), (bx, by, max(6, int(bw * min(progress, 1.0))), bh), border_radius=2)
        else:  # unknown length: a sliding segment
            seg = bw // 5
            x = bx + int((math.sin(t * 1.6) * 0.5 + 0.5) * (bw - seg))
            pygame.draw.rect(screen, (255, 214, 102), (x, by, seg, bh), border_radius=2)
        sg = self.f_strip.render(stage, True, (210, 218, 235))
        screen.blit(sg, ((W - sg.get_width()) // 2, by + 14))
        self.draw(screen, st)

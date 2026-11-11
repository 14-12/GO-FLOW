"""The GO-FLOW forecasters on the free-fly overlay (--goflow-prediction), drawn as the junction agents' decisions are
(cosim/agent_hud.py, whose pills, fonts and projection it reuses):

* bottom right, whenever the forecasters run, a panel (7 Nov; before, one small line):
  "TRAFFIC FORECAST" and the forecasters' state ("DSO4 ok · GO-FLOW ok", "warming up: 4 min", "OFFLINE");
  the three approaches DSO4 sees closest to full in 2 minutes, with their queue now and then and GO-FLOW's state;
  the last alert and how long ago; "F10 fly there";
* with --goflow-hud, on the road, near you: a chip over each approach GO-FLOW calls congested or severe, or DSO4
  forecasts at least half full ("CONGESTED  ·  QUEUE 4 -> 9 IN 2 MIN"), coloured by her state;
* with --goflow-hud, over the junction, for ALERT_SHOW_S simulated seconds after an alert, a pill:
  "QUEUE FILLING UP IN 120 S" with the approach, the forecast fill and, for a propagation, the full approach downstream.

Drawn into the overlay window (cosim/overlay.py), whose see-through colour is a colour key: shapes have no blended
edges and text sits on its own solid background.
"""
from __future__ import annotations

import math

import pygame

from cosim import ui
from cosim.agent_hud import BG, EDGE, NEAR_M, AgentHud
from cosim.hud import FONTS

COLOURS = {"spillback_soon": (251, 146, 60), "propagation": (244, 63, 94), "long_queue_soon": (249, 115, 22)}
STATE_COLOURS = {"free_flow": (52, 211, 153), "moderate": (234, 179, 8), "congested": (249, 115, 22),
                 "severe": (239, 68, 68)}
STATE_TEXT = {"free_flow": "free", "moderate": "moderate", "congested": "congested", "severe": "severe"}
NO_STATE = (148, 163, 184)
CHIP_M = 250.0        # approach chips within this distance
CHIP_RISK = 0.5       # ... and for DSO4: forecast at least this share of full, or this chance of a long queue
CHIP_MIN_QUEUE = 3.0  # ... and a queue of at least this many vehicles now or then (else not worth a chip)
MAX_CHIPS = 4         # the most at risk only: a junction has up to 4 approaches each way (CARLA, 7 Nov: 7 chips)
PANEL_W = 330


def _cars(x) -> str:
    return "?" if x is None else f"{x:.0f}"


def _full(risk: float) -> str:
    """DSO4's `full` is per approach: the queue it reaches only 1 % of the time in the training data."""
    return "full" if risk >= 1.0 else f"{risk * 100:.0f} % full"


class ForecastHud(AgentHud):
    def __init__(self, fov_deg: float = 90.0):
        super().__init__(fov_deg=fov_deg, mode="forecast")
        self.f_panel = pygame.font.Font(str(FONTS / "BarlowCondensed-SemiBold.ttf"), 20)
        self.f_row = pygame.font.Font(str(FONTS / "Barlow-Medium.ttf"), 14)
        # rendered once, reused while their text stays the same: the overlay draws every picture, and rendering the
        # panel's text each time cost frames (CARLA, 7 Nov: 27 fps with the forecasters, 28-32 without)
        self._panel_key, self._panel = None, None
        self._surfaces: dict = {}

    @staticmethod
    def title(a: dict) -> str:
        if a["kind"] == "long_queue_soon":
            return f"LONG QUEUE LIKELY WITHIN {a['horizon_s'] // 60} MIN"
        if a["kind"] == "propagation":
            return f"QUEUE SPILLING BACK IN {a['horizon_s']} S"
        return f"QUEUE FILLING UP IN {a['horizon_s']} S"

    @staticmethod
    def subtitle(a: dict, now: float) -> str:
        if a["kind"] == "long_queue_soon":
            gf = f"  ·  GO-FLOW: {STATE_TEXT.get(a['goflow_state'], a['goflow_state'])}" if a.get("goflow_state") else ""
            return (f"DSO4  ·  {a['node']}  ·  {a['p'] * 100:.0f} % chance  ·  queue {_cars(a.get('queue_now'))} cars"
                    f"{gf}  ·  {max(0, int(now - a['t']))} s ago")
        src = f"  ·  from {a['source']}" if a.get("source") else ""
        gf = ""
        if a.get("confirmed_by_goflow") is not None:
            gf = "  ·  GO-FLOW agrees" if a["confirmed_by_goflow"] else ""
        return (f"DSO4 forecast  ·  {a['node']}  ·  fill {a['fill_now']:.2f} -> {a['pred_fill']:.2f}{src}{gf}  ·  "
                f"{max(0, int(now - a['t']))} s ago")

    # ---------------------------------------------------------------- over the junctions and on the roads
    def draw_alerts(self, screen, size, view_tf, centres: dict, alerts: list) -> list:
        """centres: tls -> CARLA (x, y, z); alerts: the ones to draw as pills (empty without --goflow-hud)."""
        rects: list = []
        w, h = size
        loc = view_tf.location
        latest: dict[str, dict] = {}
        for a in alerts:                        # newest alert per junction
            latest[a["tls"]] = a
        for tls, a in latest.items():
            c = centres.get(tls)
            if c is None or math.dist((loc.x, loc.y, loc.z), c) > NEAR_M:
                continue
            p = self.project(view_tf, size, c[0], c[1], c[2] + 14.0)   # above the agents' pill (z + 9)
            if p is None or not (EDGE <= p[0] <= w - EDGE and EDGE <= p[1] <= h - EDGE):
                continue
            title, sub = self.title(a), self.subtitle(a, a.get("now", a["t"]))
            pill = self._cached(("pill", title, sub), lambda: self._pill(title, sub, COLOURS.get(a["kind"],
                                                                                              COLOURS["spillback_soon"])))
            px = min(max(p[0] - pill.get_width() / 2, 8), w - pill.get_width() - 8)
            py = min(max(p[1] - pill.get_height(), 8), h - pill.get_height() - 8)
            rects.append(screen.blit(pill, (px, py)))
        return rects

    def _cached(self, key, make) -> pygame.Surface:
        surf = self._surfaces.get(key)
        if surf is None:
            if len(self._surfaces) > 300:
                self._surfaces.clear()
            surf = self._surfaces[key] = make()
        return surf

    def chip(self, r: dict) -> pygame.Surface:
        state = r.get("state")
        colour = STATE_COLOURS.get(state, NO_STATE)
        head = (STATE_TEXT.get(state) or "forecast").upper()
        if r.get("long_now"):              # its queue is long already (the classifier gives 0 for it)
            text = f"{head}  ·  QUEUE {_cars(r.get('queue_now'))}  ·  LONG QUEUE NOW"
        elif r.get("p_long") is not None and r.get("long_s"):
            text = (f"{head}  ·  QUEUE {_cars(r.get('queue_now'))}  ·  LONG QUEUE IN {r['long_s'] // 60} MIN: "
                    f"{r['p_long'] * 100:.0f} %")
        else:
            mins = r.get("horizon_s", 120) / 60.0
            text = f"{head}  ·  QUEUE {_cars(r.get('queue_now'))} -> {_cars(r.get('queue_then'))} IN {mins:g} MIN"
        t = self.f_title.render(text, True, (245, 246, 248), BG)
        surf = pygame.Surface((t.get_width() + 20, t.get_height() + 6))
        surf.fill(BG)
        pygame.draw.rect(surf, colour, (0, 0, 5, surf.get_height()))
        surf.blit(t, (12, 3))
        return ui.glassify(surf, radius=6)

    def draw_chips(self, screen, size, view_tf, anchors: dict, rows: list) -> list:
        """anchors: approach -> CARLA (x, y, z) on its lanes before the stop line; rows: GoflowPrediction.at_risk for
        every approach. A chip for those congested or severe (GO-FLOW) or at least CHIP_RISK of full (DSO4)."""
        rects: list = []
        w, h = size
        loc = view_tf.location
        shown = 0

        def rank(r):
            return -(1.0 if r.get("long_now") else (r.get("p_long") or 0.0)), -r.get("risk", 0.0)

        for r in sorted(rows, key=rank):
            if shown >= MAX_CHIPS:
                break
            risk = 1.0 if r.get("long_now") else (r["p_long"] if r.get("p_long") is not None else r.get("risk", 0.0))
            if not (r.get("state") in ("congested", "severe") or risk >= CHIP_RISK):
                continue
            q = max(r.get("queue_now") or 0.0, r.get("queue_then") or 0.0)
            if q < CHIP_MIN_QUEUE and r.get("state") != "severe":
                continue
            a = anchors.get(r["node"])
            if a is None or math.dist((loc.x, loc.y, loc.z), a) > CHIP_M:
                continue
            p = self.project(view_tf, size, a[0], a[1], a[2] + 4.0)
            if p is None or not (EDGE <= p[0] <= w - EDGE and EDGE <= p[1] <= h - EDGE):
                continue
            c = self._cached(("chip", r["node"], r.get("state"), bool(r.get("long_now")), _cars(r.get("queue_now")),
                              _cars(r.get("queue_then")),
                              None if r.get("p_long") is None else round(r["p_long"] * 100), r.get("long_s"),
                              r.get("horizon_s")), lambda r=r: self.chip(r))
            px = min(max(p[0] - c.get_width() / 2, 8), w - c.get_width() - 8)
            rects.append(screen.blit(c, (px, p[1] - c.get_height())))
            shown += 1
        return rects

    # ---------------------------------------------------------------- the panel, bottom right
    def panel(self, status: str, warm_left_s, rows: list, last_alert: dict | None, now: float,
              alerts_last_minute: int) -> pygame.Surface:
        lines: list[tuple] = []          # (colour or None, surface)
        head = self.f_panel.render("TRAFFIC FORECAST", True, (245, 246, 248), BG)
        st = status if warm_left_s is None else f"{status}  ·  first forecast in about {max(1, round(warm_left_s / 60))} min"
        lines.append((None, self.f_small.render(st, True, (170, 178, 192), BG)))
        if rows:
            long_s = rows[0].get("long_s") if rows[0].get("p_long") is not None else None
            head2 = (f"most likely to get a long queue within {long_s // 60} minutes:" if long_s
                     else "most likely to fill up in 2 minutes:")
            lines.append((None, self.f_small.render(head2, True, (170, 178, 192), BG)))
            for r in rows:
                state = STATE_TEXT.get(r.get("state"), "")
                what = (f"{r['p_long'] * 100:.0f} % chance" if long_s and r.get("p_long") is not None
                        else _full(r["risk"]))
                txt = (f"{r['node']}  ·  queue {_cars(r.get('queue_now'))} -> {_cars(r.get('queue_then'))} cars"
                       f"  ·  {what}" + (f"  ·  {state}" if state else ""))
                lines.append((STATE_COLOURS.get(r.get("state"), NO_STATE),
                              self.f_row.render(txt, True, (226, 230, 238), BG)))
        if last_alert is not None:
            ago = max(0, int(now - last_alert["t"]))
            ago_t = f"{ago} s ago" if ago < 120 else f"{ago // 60} min ago"
            lines.append((COLOURS.get(last_alert["kind"], COLOURS["spillback_soon"]),
                          self.f_row.render(f"last alert: {self.title(last_alert).lower()}  ·  {last_alert['node']}"
                                            f"  ·  {ago_t}", True, (245, 246, 248), BG)))
        else:
            lines.append((None, self.f_small.render("no alert yet", True, (170, 178, 192), BG)))
        lines.append((None, self.f_small.render(f"alerts in the last minute: {alerts_last_minute}  ·  F10: fly to the top "
                                                "road", True, (170, 178, 192), BG)))
        pw = max([head.get_width()] + [s.get_width() + (18 if c else 0) for c, s in lines]) + 26
        pw = max(pw, PANEL_W)
        ph = 8 + head.get_height() + sum(s.get_height() + 3 for _, s in lines) + 8
        surf = pygame.Surface((pw, ph))
        surf.fill(BG)
        yy = 6
        surf.blit(head, (13, yy))
        yy += head.get_height()
        for colour, s in lines:
            x = 13
            if colour is not None:
                pygame.draw.rect(surf, colour, (13, yy + s.get_height() // 2 - 4, 9, 9))
                x = 31
            surf.blit(s, (x, yy))
            yy += s.get_height() + 3
        return ui.glassify(surf, radius=12)

    def draw(self, screen, size, view_tf, centres: dict, anchors: dict, fc, now: float, pills: bool,
             panel: bool = True, bottom: int = 16) -> list:
        """Everything for one picture. fc: the GoflowPrediction; pills: --goflow-hud (the chips and the alert pills;
        the panel unless panel is False: the route agent's mini-map takes its corner while you follow a car)."""
        w, h = size
        rects: list = []
        recent = fc.recent_alerts(now)
        if pills:
            rects += self.draw_chips(screen, size, view_tf, anchors, fc.at_risk(n=len(anchors) or 60))
            rects += self.draw_alerts(screen, size, view_tf, centres, [dict(a, now=now) for a in recent])
        if not panel:
            return rects
        alerts = fc.recent_alerts(now, window_s=1e9)
        last = alerts[-1] if alerts else None
        status, warm, top, n_recent = fc.status(), fc.warm_up_left_s(), fc.at_risk(3), len(recent)
        key = (status, None if warm is None else max(1, round(warm / 60)), n_recent,
               None if last is None else (last["t"], last["node"], int(now - last["t"]) if now - last["t"] < 120
                                          else int(now - last["t"]) // 60 * 60),
               tuple((r["node"], r.get("state"), _cars(r.get("queue_now")), _cars(r.get("queue_then")),
                      round(r["risk"], 2), None if r.get("p_long") is None else round(r["p_long"], 2)) for r in top))
        if key != self._panel_key:          # rebuilt only when what it says changes
            self._panel_key, self._panel = key, self.panel(status, warm, top, last, now, n_recent)
        p = self._panel
        rects.append(screen.blit(p, (w - p.get_width() - (8 if ui.ALPHA else 16), h - p.get_height() - bottom)))
        return rects

"""DSO4's spill-back alerts on the free-fly overlay (--goflow-prediction), drawn as the junction agents' decisions are
(cosim/agent_hud.py, whose pills and projection it reuses):

* over the junction, for ALERT_SHOW_S simulated seconds after an alert, a pill: "QUEUE SPILLING BACK IN 120 S" with
  the approach, the forecast fill and, for a propagation, the full approach downstream;
* top right, under the agents' list, one line with the forecasters' state ("DSO4 ok · GO-FLOW ok", "warming up",
  "OFFLINE") and the number of alerts of the last minute.
"""
from __future__ import annotations

import math

from cosim.agent_hud import BG, EDGE, NEAR_M, AgentHud

COLOURS = {"spillback_soon": (251, 146, 60), "propagation": (244, 63, 94)}


class ForecastHud(AgentHud):
    def __init__(self, fov_deg: float = 90.0):
        super().__init__(fov_deg=fov_deg, mode="forecast")

    @staticmethod
    def title(a: dict) -> str:
        if a["kind"] == "propagation":
            return f"QUEUE SPILLING BACK IN {a['horizon_s']} S"
        return f"QUEUE FILLING UP IN {a['horizon_s']} S"

    @staticmethod
    def subtitle(a: dict, now: float) -> str:
        src = f"  ·  from {a['source']}" if a.get("source") else ""
        gf = ""
        if a.get("confirmed_by_goflow") is not None:
            gf = "  ·  GO-FLOW agrees" if a["confirmed_by_goflow"] else ""
        return (f"DSO4 forecast  ·  {a['node']}  ·  fill {a['fill_now']:.2f} -> {a['pred_fill']:.2f}{src}{gf}  ·  "
                f"{max(0, int(now - a['t']))} s ago")

    def draw_alerts(self, screen, size, view_tf, centres: dict, alerts: list, status: str, now: float) -> list:
        """centres: tls -> CARLA (x, y, z); alerts: GoflowPrediction.recent_alerts(now). -> rectangles drawn."""
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
            pill = self._pill(self.title(a), self.subtitle(a, now), COLOURS.get(a["kind"], COLOURS["spillback_soon"]))
            px = min(max(p[0] - pill.get_width() / 2, 8), w - pill.get_width() - 8)
            py = min(max(p[1] - pill.get_height(), 8), h - pill.get_height() - 8)
            rects.append(screen.blit(pill, (px, py)))
        line = self.f_small.render(f"FORECAST  ·  {status}  ·  last minute: {len(alerts)} alerts", True,
                                   (226, 230, 238), BG)
        rects.append(screen.blit(line, (w - line.get_width() - 16, h - line.get_height() - 16)))
        return rects

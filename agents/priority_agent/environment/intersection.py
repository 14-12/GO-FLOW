"""Géométrie d'une intersection, lue dans SUMO au démarrage.

Les approches sont toujours rangées dans l'ordre N, E, S, W (par l'orientation du nœud
d'origine). Ainsi un même modèle PPO peut contrôler n'importe quel carrefour 4 branches
du projet (carrefour isolé d'entraînement ou carrefours du corridor).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

APPROACHES = ("N", "E", "S", "W")


def _direction(dx: float, dy: float) -> str:
    ang = math.degrees(math.atan2(dy, dx)) % 360
    if 45 <= ang < 135:
        return "N"
    if 135 <= ang < 225:
        return "W"
    if 225 <= ang < 315:
        return "S"
    return "E"


@dataclass
class Intersection:
    tls_id: str
    incoming: dict[str, str]                 # approche -> edge entrant
    incoming_lanes: dict[str, list[str]]     # approche -> voies (triées)
    outgoing: dict[str, str]                 # direction -> edge sortant
    lane_length: dict[str, float]
    green_phases: list[int]                  # indices des phases vertes du programme
    yellow_after: dict[int, int]             # phase verte -> phase jaune qui la suit
    approach_phase: dict[str, int]           # approche -> phase verte qui la sert le mieux
    program_states: set[str] = field(default_factory=set)
    program_durations: dict[int, float] = field(default_factory=dict)

    @property
    def edge_to_approach(self) -> dict[str, str]:
        return {e: a for a, e in self.incoming.items()}

    @classmethod
    def from_sumo(cls, conn, tls_id: str) -> "Intersection":
        x0, y0 = conn.junction.getPosition(tls_id)
        lanes = list(dict.fromkeys(conn.trafficlight.getControlledLanes(tls_id)))
        incoming, inc_lanes, lane_len = {}, {}, {}
        for ln in lanes:
            edge = conn.lane.getEdgeID(ln)
            shape = conn.lane.getShape(ln)
            sx, sy = shape[0]
            a = _direction(sx - x0, sy - y0)
            incoming[a] = edge
            inc_lanes.setdefault(a, []).append(ln)
            lane_len[ln] = conn.lane.getLength(ln)
        for a in inc_lanes:
            inc_lanes[a].sort()
        outgoing = {}
        for link_group in conn.trafficlight.getControlledLinks(tls_id):
            for _, out_lane, _ in link_group:
                edge = conn.lane.getEdgeID(out_lane)
                shape = conn.lane.getShape(out_lane)
                ex, ey = shape[-1]
                outgoing[_direction(ex - x0, ey - y0)] = edge

        prog = conn.trafficlight.getProgram(tls_id)
        logic = next(l for l in conn.trafficlight.getAllProgramLogics(tls_id) if l.programID == prog)
        states = [ph.state for ph in logic.phases]
        greens = [i for i, s in enumerate(states) if "y" not in s.lower() and any(c in "Gg" for c in s)]
        yellow_after = {g: (g + 1) % len(states) for g in greens}

        links = conn.trafficlight.getControlledLinks(tls_id)
        approach_phase = {}
        for a, edge in incoming.items():
            idx = [i for i, l in enumerate(links) if l and conn.lane.getEdgeID(l[0][0]) == edge]
            best = max(greens, key=lambda p: sum(2 if states[p][i] == "G" else 1 if states[p][i] == "g" else 0
                                                  for i in idx))
            approach_phase[a] = best
        return cls(tls_id, incoming, inc_lanes, outgoing, lane_len, greens, yellow_after, approach_phase,
                   set(states), {i: ph.duration for i, ph in enumerate(logic.phases)})

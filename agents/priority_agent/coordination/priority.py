"""Suivi des véhicules prioritaires à l'échelle du réseau (messages V2I simulés).

Pour chaque véhicule prioritaire présent dans le réseau, on calcule les intersections à feux
restant sur son itinéraire, avec pour chacune : l'approche utilisée, la distance, l'ETA et le
rang (0 = prochaine intersection). C'est la donnée de base des coordinateurs (règles et LLM).

Paramètres SUMO utilisés (vType ou véhicule) :
  priority_class : EMERGENCY_MEDICAL | FIRE | POLICE | TRANSIT
  siren          : on | off   (les urgences ne sont prioritaires que sirène active)
  confirmed      : false si la demande n'est pas (encore) confirmée par le centre
  confirm_at     : instant (s) où la demande devient confirmée
  late_s         : retard du bus (s) ; un bus est "en retard" au-delà de 60 s
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRIORITY_LEVEL = {"EMERGENCY_MEDICAL": 3, "FIRE": 3, "POLICE": 2, "TRANSIT": 1}
SIREN_REQUIRED = {"EMERGENCY_MEDICAL", "FIRE", "POLICE"}
LATE_BUS_S = 60.0


@dataclass
class TLSStop:
    tls_id: str
    approach: str
    edge: str
    distance_m: float
    eta_s: float
    rank: int


@dataclass
class PriorityVehicle:
    vehicle_id: str
    priority_class: str
    priority_level: int
    siren_on: bool
    confirmed: bool
    late_s: float
    first_seen: float
    speed: float
    edge: str
    upcoming: list[TLSStop] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def is_emergency(self) -> bool:
        return self.priority_class in SIREN_REQUIRED

    @property
    def is_active(self) -> bool:
        """Demande prioritaire réellement valable."""
        if self.is_emergency:
            return self.siren_on and self.confirmed
        return self.priority_class == "TRANSIT" and self.late_s >= LATE_BUS_S

    def as_dict(self) -> dict:
        return {
            "vehicle_id": self.vehicle_id, "priority_class": self.priority_class,
            "priority_level": self.priority_level, "siren_on": self.siren_on, "confirmed": self.confirmed,
            "late_s": round(self.late_s), "speed_mps": round(self.speed, 1), "current_edge": self.edge,
            "first_seen_s": self.first_seen, "passed_intersections": self.passed,
            "upcoming": [{"tls_id": s.tls_id, "approach": s.approach, "eta_s": round(s.eta_s, 1),
                          "distance_m": round(s.distance_m), "rank": s.rank} for s in self.upcoming],
        }


class PriorityTracker:
    def __init__(self, conn, intersections: dict):
        self.c = conn
        self.inters = intersections
        self.edge_tls = {}  # edge entrant -> (tls, approche)
        for tls, inter in intersections.items():
            for a, e in inter.incoming.items():
                self.edge_tls[e] = (tls, a)
        self._edge_len: dict[str, float] = {}
        self._first_seen: dict[str, float] = {}
        self.vehicles: dict[str, PriorityVehicle] = {}
        self._classes: dict[str, str] = {}

    def _len(self, edge: str) -> float:
        if edge not in self._edge_len:
            self._edge_len[edge] = self.c.lane.getLength(f"{edge}_0")
        return self._edge_len[edge]

    def _param(self, vid: str, key: str) -> str:
        c = self.c
        return c.vehicle.getParameter(vid, key) or c.vehicletype.getParameter(c.vehicle.getTypeID(vid), key)

    def _classify(self, vid: str) -> str | None:
        cls = self._param(vid, "priority_class")
        if not cls:
            vc = self.c.vehicle.getVehicleClass(vid)
            cls = {"emergency": "EMERGENCY_MEDICAL", "authority": "POLICE", "bus": "TRANSIT"}.get(vc)
        return cls if cls in PRIORITY_LEVEL else None

    def update(self, t: float) -> dict[str, PriorityVehicle]:
        """À appeler à CHAQUE pas (les départs ne sont visibles que pendant le pas courant)."""
        c = self.c
        for vid in c.simulation.getDepartedIDList():
            cls = self._classify(vid)
            if cls:
                self._classes[vid] = cls
        for vid in c.simulation.getArrivedIDList():
            self._classes.pop(vid, None)
        out = {}
        for vid, cls in list(self._classes.items()):
            siren = self._param(vid, "siren").lower() in ("on", "true", "1")
            conf_raw = self._param(vid, "confirmed").lower()
            confirmed = conf_raw not in ("false", "0", "no")
            confirm_at = self._param(vid, "confirm_at")
            if not confirmed and confirm_at and t >= float(confirm_at):
                confirmed = True
            late = float(self._param(vid, "late_s") or 0.0)
            self._first_seen.setdefault(vid, t)
            edge = c.vehicle.getRoadID(vid)
            speed = c.vehicle.getSpeed(vid)
            pv = PriorityVehicle(vid, cls, PRIORITY_LEVEL[cls], siren, confirmed, late,
                                 self._first_seen[vid], speed, edge)
            route = c.vehicle.getRoute(vid)
            ridx = c.vehicle.getRouteIndex(vid)
            internal = edge.startswith(":")
            start = ridx + 1 if internal else ridx
            v_ref = max(speed, 8.0)
            dist, rank = 0.0, 0
            for i in range(start, len(route)):
                e = route[i]
                if i == ridx and not internal:
                    dist = self._len(e) - c.vehicle.getLanePosition(vid)
                else:
                    dist += self._len(e)
                if e in self.edge_tls:
                    tls, a = self.edge_tls[e]
                    pv.upcoming.append(TLSStop(tls, a, e, dist, dist / v_ref, rank))
                    rank += 1
            for i in range(0, start):
                if route[i] in self.edge_tls:
                    pv.passed.append(self.edge_tls[route[i]][0])
            out[vid] = pv
        self.vehicles = out
        return out

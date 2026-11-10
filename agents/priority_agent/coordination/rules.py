"""Coordinateurs déterministes.

LocalCoordinator (B3L) : chaque intersection ne voit que les véhicules prioritaires déjà sur une
    de ses approches, à moins de 300 m (V2I local). Aucune anticipation, aucun corridor.

RuleCoordinator (B4) : coordinateur stratégique à règles explicites (baseline du LLM) :
    1. classe : urgence médicale = pompiers > police > bus en retard > trafic ordinaire ;
    2. à classe égale : premier arrivé premier servi, puis ETA la plus courte ;
    3. corridor : chaque intersection de l'itinéraire est préparée dès que l'ETA du véhicule
       passe sous l'horizon (60 s), dans l'ordre des ETA ; réévaluation à chaque décision ;
    4. fin : après le passage, RECOVERY pendant `recovery_s` (10 s) puis NORMAL ;
    5. demande non confirmée : aucune préemption tant qu'elle n'est pas confirmée.
"""
from __future__ import annotations

from environment.runtime import Context

V2I_RANGE_M = 300.0


def _key(pv, stop):
    return (-pv.priority_level, pv.first_seen, stop.eta_s)


class _Base:
    def __init__(self, recovery_s: float = 10.0, mode_safety=None, known_tls=None, inters=None):
        self.recovery_s = recovery_s
        self.mode_safety = mode_safety
        self.inters = inters or {}
        self.ctx: dict[str, Context] = {}
        self.decisions: list[dict] = []

    def _finish(self, t, tls, chosen: dict, vehicles) -> Context:
        prev = self.ctx.get(tls, Context())
        if chosen.get(tls):
            pv, stop, n_conflict = chosen[tls]
            if pv.is_emergency:
                mode = "CONFLICT_RESOLUTION" if n_conflict else "EMERGENCY_PREEMPTION"
            else:
                mode = "TRANSIT_PRIORITY"
            ok = True
            if self.mode_safety is not None and (prev.vehicle_id != pv.vehicle_id or prev.mode != mode):
                inter = self.inters[tls]
                ok, reason = self.mode_safety.validate(t, tls, mode, set(self.inters), stop.approach,
                                                       set(inter.incoming), pv.as_dict())
                self.decisions.append({"time": t, "tls_id": tls, "mode": mode, "vehicle": pv.vehicle_id,
                                       "accepted": ok, "reason": reason})
            if ok:
                return Context(mode, pv.vehicle_id, stop.approach, stop.eta_s, stop.rank)
        # pas (ou plus) de cible
        if prev.has_target and prev.vehicle_id:
            pv = vehicles.get(prev.vehicle_id)
            if pv is None or tls in pv.passed or not pv.is_active:
                return Context("RECOVERY", until=t + self.recovery_s)
        if prev.mode == "RECOVERY" and prev.until and t < prev.until:
            return prev
        return Context()

    def contexts(self) -> dict[str, Context]:
        return self.ctx


class LocalCoordinator(_Base):
    def update(self, t: float, vehicles: dict) -> dict[str, Context]:
        chosen: dict = {}
        for pv in vehicles.values():
            if not pv.is_active or not pv.upcoming:
                continue
            stop = pv.upcoming[0]
            if stop.rank != 0 or stop.distance_m > V2I_RANGE_M:
                continue
            cur = chosen.get(stop.tls_id)
            if cur is None or _key(pv, stop) < _key(cur[0], cur[1]):
                chosen[stop.tls_id] = (pv, stop, 0)
        for tls in self.inters:
            self.ctx[tls] = self._finish(t, tls, chosen, vehicles)
        return self.ctx


class RuleCoordinator(_Base):
    def __init__(self, horizon_s: float = 60.0, scheduled: list[dict] | None = None, edge_tls=None, **kw):
        super().__init__(**kw)
        self.horizon_s = horizon_s
        self.scheduled = scheduled or []   # pré-annonce (entraînement) : départs connus à l'avance
        self.edge_tls = edge_tls or {}
        self.bonus: dict[str, float] = {}  # priorité relevée par l'agent LLM (B5 seulement)

    def candidates(self, t: float, vehicles: dict) -> dict[str, list]:
        cands: dict[str, list] = {}
        for pv in vehicles.values():
            if not pv.is_active:
                continue
            for stop in pv.upcoming:
                prev = self.ctx.get(stop.tls_id)
                kept = prev is not None and prev.vehicle_id == pv.vehicle_id  # hystérésis : pas de va-et-vient
                if stop.eta_s <= self.horizon_s or kept:
                    cands.setdefault(stop.tls_id, []).append((pv, stop))
        return cands

    def update(self, t: float, vehicles: dict) -> dict[str, Context]:
        cands = self.candidates(t, vehicles)
        self._add_scheduled(t, vehicles, cands)
        chosen = {}
        for tls, lst in cands.items():
            lst.sort(key=lambda x: (-(x[0].priority_level + self.bonus.get(x[0].vehicle_id, 0.0)),
                                    x[0].first_seen, x[1].eta_s))
            best_pv, best_stop = lst[0]
            phase = self.inters[tls].approach_phase[best_stop.approach]
            n_conflict = sum(1 for pv, st in lst[1:] if pv.is_emergency and
                             self.inters[tls].approach_phase[st.approach] != phase)
            chosen[tls] = (best_pv, best_stop, n_conflict)
        for tls in self.inters:
            self.ctx[tls] = self._finish(t, tls, chosen, vehicles)
        return self.ctx

    def _add_scheduled(self, t, vehicles, cands):
        from coordination.priority import PRIORITY_LEVEL, PriorityVehicle, TLSStop
        types = {"ambulance": "EMERGENCY_MEDICAL", "fire_truck": "FIRE", "police": "POLICE"}
        for s in self.scheduled:
            if s["id"] in vehicles or s.get("_gone"):
                if s["id"] in vehicles:
                    s["_gone"] = True
                continue
            lead = s["depart"] - t
            if not (0 < lead <= 30) or s["from"] not in self.edge_tls or s.get("params", {}).get("confirmed"):
                continue
            cls = types.get(s["type"])
            if cls is None:
                continue
            tls, a = self.edge_tls[s["from"]]
            pv = PriorityVehicle(s["id"], cls, PRIORITY_LEVEL[cls], True, True, 0, s["depart"], 0, "")
            stop = TLSStop(tls, a, s["from"], 400.0, lead + 400 / 13.0, 1)
            pv.upcoming = [stop]
            vehicles[s["id"]] = pv
            if stop.eta_s <= self.horizon_s:
                cands.setdefault(tls, []).append((pv, stop))

"""Exécution des décisions de feux sous contrôle du Safety Controller.

Toute décision (PPO, préemption, coordinateur) passe par `propose()` :
    proposée -> validée (ou rejetée) -> exécutée -> vérifiée
Les trois états sont journalisés séparément (décision 6.2 / section 9 du document).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from safety.validator import SignalSafety

LOCK = 100000.0  # durée "infinie" : sous contrôle externe, le programme n'avance plus seul


@dataclass
class ControlStats:
    proposed: int = 0
    approved: int = 0
    executed_switches: int = 0
    records: list[dict] = field(default_factory=list)


class SignalController:
    def __init__(self, conn, inter, safety: SignalSafety, keep_records: bool = False):
        self.c = conn
        self.inter = inter
        self.safety = safety
        self.keep_records = keep_records
        self.stats = ControlStats()
        self.external = False
        self.current_green: int | None = None
        self.green_start = 0.0
        self.target: int | None = None
        self.yellow_end = 0.0
        self.priority_phase: int | None = None  # phase qui sert l'urgence en cours (extension du vert max)
        # Un programme actionné continue de basculer seul (détecteurs) même sous contrôle externe :
        # on prépare une copie statique des mêmes phases, utilisée uniquement pendant le contrôle externe.
        tls = inter.tls_id
        self.program = conn.trafficlight.getProgram(tls)
        logic = next(l for l in conn.trafficlight.getAllProgramLogics(tls) if l.programID == self.program)
        self.static_copy = None
        if int(logic.type) != 0:
            logic.programID, logic.type = "ext_static", 0
            conn.trafficlight.setProgramLogic(tls, logic)
            conn.trafficlight.setProgram(tls, self.program)  # on reste sur le programme d'origine
            self.static_copy = "ext_static"
        self._sync_from_program(conn.simulation.getTime())

    # ------------------------------------------------------------------ état
    @property
    def in_transition(self) -> bool:
        return self.target is not None

    def time_in_green(self, t: float) -> float:
        return 0.0 if self.in_transition else t - self.green_start

    def _sync_from_program(self, t: float) -> None:
        tls = self.inter.tls_id
        ph = self.c.trafficlight.getPhase(tls)
        if ph in self.inter.green_phases:
            elapsed = self.c.trafficlight.getPhaseDuration(tls) - (self.c.trafficlight.getNextSwitch(tls) - t)
            # marge d'1 s : à la reprise de contrôle, on sous-estime toujours la durée du vert écoulé.
            # Si le début du vert a été observé (programme actionné : durée variable), on retient
            # l'estimation la plus prudente des deux.
            start = t - max(0.0, elapsed - 1.0)
            if ph == self.current_green and getattr(self, "_observed", False):
                start = max(start, self.green_start)
            self.current_green, self.green_start = ph, start
            self.target = None
        else:  # jaune en cours : on laisse finir vers la phase verte suivante
            nxt = (ph + 1) % len(self.inter.program_durations)
            prev = (ph - 1) % len(self.inter.program_durations)
            self.current_green = prev
            self.target = nxt
            self.yellow_end = self.c.trafficlight.getNextSwitch(tls)

    # ------------------------------------------------------------------ contrôle
    def take_control(self, t: float) -> None:
        if self.external:
            return
        self._sync_from_program(t)
        self.external = True
        if self.static_copy:
            ph = self.c.trafficlight.getPhase(self.inter.tls_id)
            self.c.trafficlight.setProgram(self.inter.tls_id, self.static_copy)
            self.c.trafficlight.setPhase(self.inter.tls_id, ph)
            if self.in_transition:
                self.c.trafficlight.setPhaseDuration(self.inter.tls_id, max(0.5, self.yellow_end - t))
        if not self.in_transition:
            self.c.trafficlight.setPhaseDuration(self.inter.tls_id, LOCK)

    def release(self, t: float) -> None:
        """Rend la main au programme SUMO (repli sûr) en conservant un état valide."""
        if not self.external:
            return
        tls = self.inter.tls_id
        if self.static_copy:  # retour au programme actionné, sur la même phase
            ph = self.c.trafficlight.getPhase(tls)
            self.c.trafficlight.setProgram(tls, self.program)
            self.c.trafficlight.setPhase(tls, ph)
            self.external, self.target = False, None
            return
        if self.in_transition:
            self.c.trafficlight.setPhaseDuration(tls, max(0.5, self.yellow_end - t))
        else:
            normal = self.inter.program_durations[self.current_green]
            self.c.trafficlight.setPhaseDuration(tls, max(1.0, normal - self.time_in_green(t)))
        self.external = False
        self.target = None

    def mask(self, t: float):
        return self.safety.action_mask(self.inter, self.current_green, self.time_in_green(t), self.in_transition,
                                       self.priority_phase)

    def propose(self, target_phase: int, t: float, source: str, obs_age: float = 0.0) -> bool:
        self.stats.proposed += 1
        ok, reason = self.safety.validate(self.inter, target_phase, self.current_green, self.time_in_green(t),
                                          self.in_transition, obs_age, t, self.priority_phase)
        rec = {"time": t, "tls_id": self.inter.tls_id, "source": source, "proposed": target_phase,
               "approved": ok, "reason": reason, "executed": None}
        if ok:
            self.stats.approved += 1
            if target_phase != self.current_green and not self.in_transition:
                tls = self.inter.tls_id
                self.c.trafficlight.setPhase(tls, self.inter.yellow_after[self.current_green])
                self.c.trafficlight.setPhaseDuration(tls, LOCK)
                self.target = target_phase
                self.yellow_end = t + self.safety.cfg.yellow_s
                self.stats.executed_switches += 1
                rec["executed"] = f"jaune->{target_phase}"
            else:
                rec["executed"] = f"maintien {self.current_green}"
        if self.keep_records:
            self.stats.records.append(rec)
        return ok

    def tick(self, t: float) -> None:
        """À appeler après chaque pas de simulation."""
        tls = self.inter.tls_id
        if self.external and self.in_transition and t >= self.yellow_end:
            self.c.trafficlight.setPhase(tls, self.target)
            self.c.trafficlight.setPhaseDuration(tls, LOCK)
            self.current_green, self.green_start, self.target = self.target, t, None
        elif not self.external:
            ph = self.c.trafficlight.getPhase(tls)
            if ph in self.inter.green_phases and ph != self.current_green:
                self.current_green, self.green_start = ph, t
                self._observed = True
        self.safety.verify_state(self.inter, self.c.trafficlight.getRedYellowGreenState(tls), t)

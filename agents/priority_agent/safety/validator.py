"""Safety Controller : couche explicite entre les décisions (PPO, coordinateurs, LLM) et les feux.

Deux niveaux de validation :
1. SignalSafety  — actions de feux (phase demandée par PPO ou par une préemption) :
   phase verte existante, durée minimale / maximale de vert, transition par le jaune,
   fraîcheur de l'observation, vérification de l'état réellement appliqué.
   Fournit aussi le MASQUE d'actions utilisé pendant l'entraînement ET l'exécution :
   PPO n'apprend jamais sur des actions qui seraient rejetées.
2. ModeSafety    — demandes de mode stratégique (coordinateur à règles ou LLM) :
   mode connu, intersection et approche valides, véhicule prioritaire réel avec sirène,
   limitation de fréquence des préemptions par intersection.

Rien dans ce module n'appelle setRedYellowGreenState : seules les phases du programme SUMO
peuvent être activées.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

STRATEGIC_MODES = ("NORMAL", "EMERGENCY_PREEMPTION", "TRANSIT_PRIORITY", "CONFLICT_RESOLUTION",
                   "RECOVERY", "DEGRADED")


@dataclass
class SafetyConfig:
    min_green_s: float = 10.0       # un vert dure au moins 10 s
    max_green_s: float = 60.0       # au-delà, la bascule est obligatoire
    max_green_emergency_s: float = 90.0  # extension autorisée si le vert sert l'urgence en cours
    yellow_s: float = 3.0           # imposé par le programme SUMO
    max_obs_age_s: float = 2.0      # observation plus vieille -> repli
    max_preemptions: int = 3        # préemptions max par intersection...
    preemption_window_s: float = 300.0  # ...sur cette fenêtre glissante
    limit_exempt_level: int = 3     # niveau >= 3 (ambulance, pompiers) : jamais bloqué par cette limite


@dataclass
class SafetyLog:
    rejected_actions: int = 0
    rejected_modes: int = 0
    state_violations: int = 0
    min_green_violations: int = 0
    fallbacks: int = 0
    reasons: list[dict] = field(default_factory=list)

    def add(self, kind: str, t: float, tls: str, reason: str) -> None:
        self.reasons.append({"time": t, "kind": kind, "tls_id": tls, "reason": reason})


class SignalSafety:
    def __init__(self, cfg: SafetyConfig | None = None, log: SafetyLog | None = None):
        self.cfg = cfg or SafetyConfig()
        self.log = log or SafetyLog()

    def max_green(self, current_green: int, priority_phase: int | None) -> float:
        return self.cfg.max_green_emergency_s if priority_phase == current_green else self.cfg.max_green_s

    def action_mask(self, inter, current_green: int, time_in_green: float, in_transition: bool,
                    priority_phase: int | None = None) -> np.ndarray:
        """Masque sur les actions = indices dans inter.green_phases."""
        n = len(inter.green_phases)
        cur = inter.green_phases.index(current_green)
        mask = np.ones(n, dtype=bool)
        if in_transition or time_in_green < self.cfg.min_green_s:
            mask[:] = False
            mask[cur] = True
        elif time_in_green >= self.max_green(current_green, priority_phase):
            mask[cur] = False
        return mask

    def validate(self, inter, target_phase: int, current_green: int, time_in_green: float,
                 in_transition: bool, obs_age: float, t: float, priority_phase: int | None = None) -> tuple[bool, str]:
        if target_phase not in inter.green_phases:
            reason = f"phase {target_phase} absente du programme ou non verte"
        elif obs_age > self.cfg.max_obs_age_s:
            reason = f"observation trop ancienne ({obs_age:.1f} s)"
        elif in_transition and target_phase != current_green:
            reason = "transition jaune en cours"
        elif target_phase != current_green and time_in_green < self.cfg.min_green_s:
            reason = f"vert minimal non atteint ({time_in_green:.0f} s < {self.cfg.min_green_s:.0f} s)"
        elif target_phase == current_green and not in_transition and \
                time_in_green >= self.max_green(current_green, priority_phase):
            reason = f"vert maximal atteint ({time_in_green:.0f} s)"
        else:
            return True, "ok"
        self.log.rejected_actions += 1
        self.log.add("action", t, inter.tls_id, reason)
        return False, reason

    def verify_state(self, inter, state: str, t: float) -> bool:
        """L'état électrique observé doit être l'une des phases du programme (aucun conflit)."""
        if state not in inter.program_states:
            self.log.state_violations += 1
            self.log.add("state", t, inter.tls_id, f"état inconnu {state}")
            return False
        return True


class ModeSafety:
    """Valide les demandes de mode du coordinateur (règles ou LLM) avant application."""

    def __init__(self, cfg: SafetyConfig | None = None, log: SafetyLog | None = None):
        self.cfg = cfg or SafetyConfig()
        self.log = log or SafetyLog()
        self._preempt_times: dict[str, deque] = defaultdict(deque)

    def validate(self, t: float, tls_id: str, mode: str, known_tls: set[str], approach: str | None,
                 valid_approaches: set[str], vehicle: dict | None) -> tuple[bool, str]:
        reason = None
        if mode not in STRATEGIC_MODES:
            reason = f"mode inconnu {mode}"
        elif tls_id not in known_tls:
            reason = f"intersection inconnue {tls_id}"
        elif mode in ("EMERGENCY_PREEMPTION", "TRANSIT_PRIORITY", "CONFLICT_RESOLUTION"):
            if vehicle is None:
                reason = "aucun véhicule prioritaire actif associé à la demande"
            elif approach not in valid_approaches:
                reason = f"approche {approach} invalide pour {tls_id}"
            elif mode != "TRANSIT_PRIORITY" and not (vehicle.get("siren_on") and vehicle.get("confirmed")):
                reason = "véhicule d'urgence sans sirène active ou demande non confirmée"
            elif mode == "TRANSIT_PRIORITY" and vehicle.get("priority_class") != "TRANSIT":
                reason = "TRANSIT_PRIORITY réservé au transport public"
            elif mode != "TRANSIT_PRIORITY":
                q = self._preempt_times[tls_id]
                while q and t - q[0][0] > self.cfg.preemption_window_s:
                    q.popleft()
                if vehicle.get("vehicle_id") not in {v for _, v in q} and len(q) >= self.cfg.max_preemptions \
                        and vehicle.get("priority_level", 0) < self.cfg.limit_exempt_level:
                    reason = (f"trop de préemptions sur {tls_id} ({len(q)} en "
                              f"{self.cfg.preemption_window_s:.0f} s)")
        if reason:
            self.log.rejected_modes += 1
            self.log.add("mode", t, tls_id, reason)
            return False, reason
        if mode in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION"):
            q = self._preempt_times[tls_id]
            if vehicle["vehicle_id"] not in {v for _, v in q}:
                q.append((t, vehicle["vehicle_id"]))
        return True, "ok"

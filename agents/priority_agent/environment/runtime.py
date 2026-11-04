"""Socle commun à l'entraînement et à l'évaluation.

- Context      : objectif transmis par le coordinateur à une intersection (mode + cible).
- observe()    : vecteur d'observation PPO (35 valeurs) pour une intersection.
- reward()     : récompense dense conditionnée par le mode.
- Runtime      : simulation + intersections + contrôleurs de feux + suivi des priorités + événements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coordination.priority import PriorityTracker
from environment.intersection import APPROACHES, Intersection
from environment.signal_controller import SignalController
from safety.validator import ModeSafety, SafetyConfig, SafetyLog, SignalSafety
from tools.sumo_interface import SumoSession

# Les 6 modes stratégiques sont gérés par le coordinateur. Localement, PPO ne voit que 4 modes :
# CONFLICT_RESOLUTION sert localement le véhicule retenu (= EMERGENCY) et DEGRADED n'utilise pas
# PPO (repli sur le programme de feux fixe).
LOCAL_MODES = ("NORMAL", "EMERGENCY", "TRANSIT", "RECOVERY")
STRATEGIC_TO_LOCAL = {"NORMAL": "NORMAL", "EMERGENCY_PREEMPTION": "EMERGENCY",
                      "CONFLICT_RESOLUTION": "EMERGENCY", "TRANSIT_PRIORITY": "TRANSIT",
                      "RECOVERY": "RECOVERY", "DEGRADED": "NORMAL"}
OBS_DIM = 35
DECISION_S = 5


@dataclass
class Context:
    mode: str = "NORMAL"              # mode stratégique
    vehicle_id: str | None = None
    approach: str | None = None
    eta_s: float = 0.0
    rank: int = 0
    until: float | None = None        # fin du mode (RECOVERY)

    @property
    def local_mode(self) -> str:
        return STRATEGIC_TO_LOCAL[self.mode]

    @property
    def has_target(self) -> bool:
        return self.local_mode in ("EMERGENCY", "TRANSIT") and self.approach is not None

    def encode(self) -> np.ndarray:
        v = np.zeros(11, dtype=np.float32)
        v[LOCAL_MODES.index(self.local_mode)] = 1
        if self.has_target:
            v[4 + APPROACHES.index(self.approach)] = 1
            v[8] = min(max(self.eta_s, 0), 60) / 60
            v[9] = min(self.rank, 2) / 2
            v[10] = 1
        return v


def observe(conn, inter: Intersection, ctl: SignalController, ctx: Context, t: float) -> np.ndarray:
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[inter.green_phases.index(ctl.current_green)] = 1
    o[2] = min(ctl.time_in_green(t), 60) / 60
    o[3] = float(ctl.in_transition)
    k = 4
    for a in APPROACHES:
        lanes = inter.incoming_lanes.get(a, [])
        for j in range(2):
            if j < len(lanes):
                ln = lanes[j]
                cap = inter.lane_length[ln] / 7.5
                o[k] = min(conn.lane.getLastStepVehicleNumber(ln) / cap, 1)
                o[k + 1] = min(conn.lane.getLastStepHaltingNumber(ln) / cap, 1)
            k += 2
    for a in APPROACHES:
        e = inter.outgoing.get(a)
        o[k] = conn.edge.getLastStepOccupancy(e) / 100 if e else 0.0
        k += 1
    o[k:k + 11] = ctx.encode()
    return o


# poids (w_e, w_g, w_q) par mode local
REWARD_WEIGHTS = {"NORMAL": (0.0, 1.0, 0.5), "EMERGENCY": (3.0, 0.3, 0.1),
                  "TRANSIT": (1.0, 1.0, 0.3), "RECOVERY": (0.0, 1.0, 1.0)}


def reward_terms(conn, inter: Intersection, ctx: Context) -> tuple[float, float, float]:
    """(E, G, Q) mesurés à l'instant courant."""
    halts = {ln: conn.lane.getLastStepHaltingNumber(ln) for lns in inter.incoming_lanes.values() for ln in lns}
    G = sum(halts.values()) / 20
    Q = max(halts.values()) / 10 if halts else 0.0
    E = 0.0
    if ctx.has_target:
        approach_q = sum(halts[ln] for ln in inter.incoming_lanes[ctx.approach]) / 10
        E = 0.5 * approach_q
        vid = ctx.vehicle_id
        if vid:
            try:
                if conn.vehicle.getRoadID(vid) == inter.incoming[ctx.approach]:
                    E += 1.0 if conn.vehicle.getSpeed(vid) < 2.0 else 0.0
            except Exception:  # véhicule pas encore parti ou déjà arrivé
                pass
    return E, G, Q


def reward(conn, inter, ctx: Context) -> float:
    we, wg, wq = REWARD_WEIGHTS[ctx.local_mode]
    E, G, Q = reward_terms(conn, inter, ctx)
    return -(we * E + wg * G + wq * Q)


class Runtime:
    """Une simulation avec ses intersections, ses contrôleurs et son suivi de priorités."""

    def __init__(self, sumocfg, seed: int, backend: str = "libsumo", gui: bool = False,
                 events: list[dict] | None = None, safety_cfg: SafetyConfig | None = None,
                 keep_records: bool = False, label: str = "rt", extra_args: list[str] | None = None):
        self.session = SumoSession(sumocfg, label=label, seed=seed, gui=gui, backend=backend, extra_args=extra_args)
        self.events = sorted(events or [], key=lambda e: e["time"])
        self.safety_cfg = safety_cfg or SafetyConfig()
        self.safety_log = SafetyLog()
        self.keep_records = keep_records
        self.dropout: dict[str, tuple[float, float]] = {}
        self.event_log: list[dict] = []

    def start(self) -> "Runtime":
        self.session.start()
        self.c = self.session.conn
        self.inters = {tls: Intersection.from_sumo(self.c, tls) for tls in self.c.trafficlight.getIDList()}
        sig = SignalSafety(self.safety_cfg, self.safety_log)
        self.mode_safety = ModeSafety(self.safety_cfg, self.safety_log)
        self.ctls = {tls: SignalController(self.c, inter, sig, self.keep_records) for tls, inter in self.inters.items()}
        self.tracker = PriorityTracker(self.c, self.inters)
        self._obs_cache: dict[str, tuple[float, np.ndarray]] = {}
        return self

    def close(self) -> None:
        self.session.close()

    @property
    def t(self) -> float:
        return self.c.simulation.getTime()

    def step(self) -> None:
        self.c.simulationStep()
        t = self.t
        for ctl in self.ctls.values():
            ctl.tick(t)
        self.tracker.update(t)
        self._apply_events(t)

    def _apply_events(self, t: float) -> None:
        while self.events and self.events[0]["time"] <= t:
            ev = self.events[0]
            if ev["type"] == "siren_off":
                if ev["vehicle"] not in self.c.vehicle.getIDList():
                    if ev.get("_tries", 0) > 600:
                        self.events.pop(0)
                        continue
                    ev["_tries"] = ev.get("_tries", 0) + 1
                    break  # véhicule pas encore parti : on réessaie au pas suivant
                self.c.vehicle.setParameter(ev["vehicle"], "siren", "off")
            elif ev["type"] == "sensor_dropout":
                self.dropout[ev["tls"]] = (ev["time"], ev["until"])
            self.event_log.append({**{k: v for k, v in ev.items() if not k.startswith("_")}, "applied_at": t})
            self.events.pop(0)

    def set_priority(self, tls: str, ctx: Context) -> None:
        """Informe le contrôleur de la phase qui sert l'urgence (extension du vert max à 90 s)."""
        inter = self.inters[tls]
        self.ctls[tls].priority_phase = inter.approach_phase[ctx.approach] \
            if ctx.local_mode == "EMERGENCY" and ctx.approach else None

    def observation(self, tls: str, ctx: Context) -> tuple[np.ndarray, float]:
        """(observation, âge en s). Pendant une perte de capteurs, l'observation reste figée."""
        t = self.t
        d = self.dropout.get(tls)
        if d and d[0] <= t < d[1] and tls in self._obs_cache:
            ts, o = self._obs_cache[tls]
            o = o.copy()
            o[-11:] = ctx.encode()
            return o, t - ts
        o = observe(self.c, self.inters[tls], self.ctls[tls], ctx, t)
        self._obs_cache[tls] = (t, o)
        return o, 0.0

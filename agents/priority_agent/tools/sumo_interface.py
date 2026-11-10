"""Interface TraCI minimale pour l'étape 1 du projet ADSP.

Ce module sépare explicitement :
  - les fonctions d'OBSERVATION (lecture seule) ;
  - la seule fonction de COMMANDE (`request_green_phase`), qui n'accepte que des
    indices de phases vertes du programme SUMO et n'appelle jamais
    `setRedYellowGreenState`.

La validation complète (durées min/max, fréquence de préemption, fraîcheur)
sera ajoutée à l'étape 2 dans `safety/validator.py`.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import traci

try:  # libsumo : même API que TraCI, ~15x plus rapide, sans GUI, une simulation par processus
    import libsumo as _libsumo
except ImportError:  # pragma: no cover
    _libsumo = None

try:
    import sumo as _sumo_pkg  # paquet pip eclipse-sumo

    SUMO_HOME = _sumo_pkg.SUMO_HOME
except ImportError:  # installation système
    SUMO_HOME = os.environ.get("SUMO_HOME", "")


def sumo_tool(name: str) -> str:
    """Chemin d'un exécutable SUMO (sumo, sumo-gui, netconvert...), Windows ou Linux."""
    for cand in (Path(SUMO_HOME) / "bin" / name, Path(SUMO_HOME) / "bin" / f"{name}.exe"):
        if SUMO_HOME and cand.exists():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} introuvable : installez SUMO ou `pip install eclipse-sumo`.")
    return found


def sumo_binary(gui: bool = False) -> str:
    return sumo_tool("sumo-gui" if gui else "sumo")


# --------------------------------------------------------------------------- priorités
# Classe de priorité lue dans le paramètre `priority_class` du vType (fichier .rou.xml).
# Repli sur la vClass SUMO si le paramètre est absent.
VCLASS_FALLBACK = {"emergency": "EMERGENCY_MEDICAL", "authority": "POLICE", "bus": "TRANSIT"}

# Niveau de priorité (plus grand = plus prioritaire). Ambulance et pompiers sont à égalité :
# le départage se fait ensuite par ordre d'arrivée puis par ETA (règles du coordinateur B4).
PRIORITY_LEVEL = {
    "EMERGENCY_MEDICAL": 3,
    "FIRE": 3,
    "POLICE": 2,
    "TRANSIT": 1,
}

# Classes qui ne sont prioritaires QUE si la sirène / gyrophare est active.
SIREN_REQUIRED = {"EMERGENCY_MEDICAL", "FIRE", "POLICE"}


@dataclass
class PriorityRequest:
    """Message V2I simulé (cf. décision 7)."""

    request_id: str
    vehicle_id: str
    priority_class: str
    priority_level: int
    siren_on: bool
    tls_id: str
    approach_edge: str
    distance_m: float
    eta_s: float
    timestamp: float
    confirmed: bool = True


@dataclass
class VehicleTrack:
    vehicle_id: str
    priority_class: str
    depart: float
    arrival: float | None = None
    stops: int = 0
    waiting_time: float = 0.0
    siren_ever_on: bool = False
    _was_stopped: bool = field(default=False, repr=False)


class SumoSession:
    """Une simulation SUMO pilotée par TraCI, isolée par un label."""

    def __init__(self, sumocfg: str | Path, label: str = "default", seed: int | None = None,
                 gui: bool = False, v2i_range_m: float = 300.0,
                 extra_args: list[str] | None = None, backend: str = "traci"):
        self.sumocfg = str(sumocfg)
        self.label = label
        self.seed = seed
        self.gui = gui
        self.v2i_range_m = v2i_range_m
        self.extra_args = extra_args or []
        # libsumo impossible avec la GUI ; repli automatique sur TraCI s'il n'est pas installé
        self.backend = "traci" if (gui or _libsumo is None) else backend
        self._pending_target: dict[str, int] = {}
        self.conn: traci.Connection | None = None
        self._requests: dict[str, PriorityRequest] = {}
        self.priority_tracks: dict[str, VehicleTrack] = {}

    # ------------------------------------------------------------------ cycle de vie
    def start(self) -> "SumoSession":
        cmd = [sumo_binary(self.gui), "-c", self.sumocfg, "--no-step-log", "true"]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        cmd += self.extra_args + ["--no-warnings", "true"]
        if self.backend == "libsumo":
            _libsumo.start(cmd)
            self.conn = _libsumo
        else:
            traci.start(cmd, label=self.label)
            self.conn = traci.getConnection(self.label)
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def step(self) -> None:
        self.conn.simulationStep()
        self._update_priority_tracks()

    @property
    def time(self) -> float:
        return self.conn.simulation.getTime()

    def is_running(self) -> bool:
        return self.conn.simulation.getMinExpectedNumber() > 0

    # ------------------------------------------------------------------ observation
    def get_green_phases(self, tls_id: str) -> list[int]:
        """Indices des phases vertes du programme actif (pas de jaune, au moins un vert)."""
        logic = self._active_logic(tls_id)
        return [i for i, ph in enumerate(logic.phases)
                if "y" not in ph.state.lower() and any(c in "Gg" for c in ph.state)]

    def get_intersection_state(self, tls_id: str) -> dict:
        c = self.conn
        lanes = list(dict.fromkeys(c.trafficlight.getControlledLanes(tls_id)))
        per_lane = {
            ln: {
                "vehicles": c.lane.getLastStepVehicleNumber(ln),
                "queue": c.lane.getLastStepHaltingNumber(ln),
                "mean_speed": round(c.lane.getLastStepMeanSpeed(ln), 2),
                "occupancy": round(c.lane.getLastStepOccupancy(ln), 2),
                "waiting_time": round(c.lane.getWaitingTime(ln), 1),
            }
            for ln in lanes
        }
        phase = c.trafficlight.getPhase(tls_id)
        return {
            "tls_id": tls_id,
            "time": self.time,
            "phase_index": phase,
            "phase_state": c.trafficlight.getRedYellowGreenState(tls_id),
            "is_green_phase": phase in self.get_green_phases(tls_id),
            "time_in_phase": round(self.time - (c.trafficlight.getNextSwitch(tls_id)
                                                - c.trafficlight.getPhaseDuration(tls_id)), 1),
            "lanes": per_lane,
            "total_queue": sum(v["queue"] for v in per_lane.values()),
        }

    def get_approach_green_phase(self, tls_id: str, incoming_edge: str) -> int | None:
        """Phase verte qui donne le vert (G/g) à au moins une voie de l'approche donnée."""
        links = self.conn.trafficlight.getControlledLinks(tls_id)
        idx = [i for i, l in enumerate(links) if l and self.conn.lane.getEdgeID(l[0][0]) == incoming_edge]
        logic = self._active_logic(tls_id)
        best, best_score = None, 0
        for p in self.get_green_phases(tls_id):
            score = sum(1 for i in idx if logic.phases[p].state[i] == "G") * 2 + \
                    sum(1 for i in idx if logic.phases[p].state[i] == "g")
            if score > best_score:
                best, best_score = p, score
        return best

    def get_priority_class(self, vehicle_id: str) -> str | None:
        """Classe de priorité : paramètre `priority_class` du vType, sinon la vClass."""
        c = self.conn
        cls = c.vehicle.getParameter(vehicle_id, "priority_class") or \
            c.vehicletype.getParameter(c.vehicle.getTypeID(vehicle_id), "priority_class")
        return cls or VCLASS_FALLBACK.get(c.vehicle.getVehicleClass(vehicle_id))

    def is_siren_on(self, vehicle_id: str) -> bool:
        """Sirène active ? Paramètre véhicule `siren` (modifiable en cours de route),
        sinon valeur par défaut du vType, sinon éteinte."""
        c = self.conn
        val = c.vehicle.getParameter(vehicle_id, "siren") or \
            c.vehicletype.getParameter(c.vehicle.getTypeID(vehicle_id), "siren")
        return val.lower() in ("on", "true", "1")

    def set_siren(self, vehicle_id: str, on: bool) -> None:
        """Allume / éteint la sirène (ex. fin d'intervention, patrouille qui part en urgence)."""
        self.conn.vehicle.setParameter(vehicle_id, "siren", "on" if on else "off")

    def get_priority_requests(self, tls_id: str) -> list[PriorityRequest]:
        """Génère / met à jour les messages V2I des véhicules prioritaires à portée.

        - Véhicules d'urgence (ambulance, pompiers, police) : demande émise uniquement
          si la sirène est active. Si elle s'éteint, la demande disparaît.
        - Transport public : demande émise sans sirène (niveau de priorité plus bas).
        Les demandes sont triées par niveau décroissant, puis heure de demande, puis ETA.
        """
        c = self.conn
        incoming = {c.lane.getEdgeID(ln) for ln in c.trafficlight.getControlledLanes(tls_id)}
        active: list[PriorityRequest] = []
        for vid in c.vehicle.getIDList():
            cls = self.get_priority_class(vid)
            if cls not in PRIORITY_LEVEL:
                continue
            siren = self.is_siren_on(vid)
            if cls in SIREN_REQUIRED and not siren:
                self._requests.pop(vid, None)  # sirène éteinte : demande annulée
                continue
            edge = c.vehicle.getRoadID(vid)
            if edge not in incoming:
                continue  # véhicule déjà passé ou pas encore sur une approche
            lane = c.vehicle.getLaneID(vid)
            dist = c.lane.getLength(lane) - c.vehicle.getLanePosition(vid)
            if dist > self.v2i_range_m:
                continue
            speed = max(c.vehicle.getSpeed(vid), 1.0)
            req = self._requests.get(vid)
            if req is None:
                req = PriorityRequest(f"req_{vid}", vid, cls, PRIORITY_LEVEL[cls], siren, tls_id,
                                      edge, dist, dist / speed, self.time)
                self._requests[vid] = req
            req.siren_on = siren
            req.distance_m, req.eta_s = round(dist, 1), round(dist / speed, 1)
            active.append(req)
        active.sort(key=lambda r: (-r.priority_level, r.timestamp, r.eta_s))
        return active

    def has_bluelight_device(self, vehicle_id: str) -> bool:
        return self.conn.vehicle.getParameter(vehicle_id, "has.bluelight.device") == "true"

    # ------------------------------------------------------------------ commande
    def request_green_phase(self, tls_id: str, phase_index: int) -> dict:
        """Seule commande exposée. Refuse toute phase non verte du programme SUMO.

        Si la phase demandée est déjà active, ne fait rien (maintien).
        Sinon, passe par la phase jaune qui suit la phase courante :
        la transition est gérée par le programme SUMO, jamais par une chaîne brute.
        """
        greens = self.get_green_phases(tls_id)
        if phase_index not in greens:
            return {"status": "rejected", "reason": f"phase {phase_index} non verte ou inexistante"}
        c = self.conn
        current = c.trafficlight.getPhase(tls_id)
        if current == phase_index:
            return {"status": "kept", "phase_index": current}
        if current in greens:
            # Démarrer immédiatement le jaune suivant la phase courante.
            n = len(self._active_logic(tls_id).phases)
            c.trafficlight.setPhase(tls_id, (current + 1) % n)
            self._pending_target[tls_id] = phase_index
            return {"status": "transition", "from": current, "to": phase_index}
        return {"status": "deferred", "reason": "transition jaune déjà en cours"}

    def apply_pending_targets(self) -> None:
        """À appeler à chaque pas : à la fin du jaune, saute sur la phase verte cible."""
        pend = self._pending_target
        for tls_id, target in list(pend.items()):
            c = self.conn
            if c.trafficlight.getPhase(tls_id) in self.get_green_phases(tls_id):
                if c.trafficlight.getPhase(tls_id) != target:
                    c.trafficlight.setPhase(tls_id, target)
                del pend[tls_id]

    def verify_signal_state(self, tls_id: str, expected_phase: int) -> bool:
        return self.conn.trafficlight.getPhase(tls_id) == expected_phase

    # ------------------------------------------------------------------ métriques
    def _update_priority_tracks(self) -> None:
        c = self.conn
        for vid in c.simulation.getDepartedIDList():
            cls = self.get_priority_class(vid)
            if cls in PRIORITY_LEVEL:
                self.priority_tracks[vid] = VehicleTrack(vid, cls, self.time)
        for vid in c.simulation.getArrivedIDList():
            if vid in self.priority_tracks:
                self.priority_tracks[vid].arrival = self.time
        for vid, tr in self.priority_tracks.items():
            if tr.arrival is not None:
                continue
            if tr.priority_class in SIREN_REQUIRED and self.is_siren_on(vid):
                tr.siren_ever_on = True
            stopped = c.vehicle.getSpeed(vid) < 0.1
            if stopped:
                tr.waiting_time += c.simulation.getDeltaT()
                if not tr._was_stopped:
                    tr.stops += 1
            tr._was_stopped = stopped

    def _active_logic(self, tls_id: str):
        prog = self.conn.trafficlight.getProgram(tls_id)
        for logic in self.conn.trafficlight.getAllProgramLogics(tls_id):
            if logic.programID == prog:
                return logic
        raise RuntimeError(f"Programme {prog} introuvable pour {tls_id}")

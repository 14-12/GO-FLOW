"""Coordinateur stratégique agentique (B5) : LLM + LangChain + tool calling (Ollama en local).

Rôle du LLM (le QUOI et le QUAND) :
  - planifier le corridor d'un véhicule d'urgence (quelles intersections préparer) ;
  - arbitrer les demandes concurrentes ;
  - traiter les cas exceptionnels (demande non confirmée, sirène coupée, capteurs perdus) ;
  - choisir la durée de récupération après passage.
Il N'A AUCUN outil d'exécution des feux : ses seuls leviers sont les modes / objectifs par
intersection, validés par le Safety Controller (ModeSafety). PPO décide du COMMENT localement.

Déclenchement par événements (nouvelle demande, conflit, annulation, confirmation, anomalie),
jamais à chaque pas. La simulation est en pause pendant l'appel ; la latence est mesurée.
Si le LLM est indisponible, trop lent ou produit une sortie invalide, le watchdog applique les
règles déterministes (B4) pour cet événement : le système ne dépend jamais du LLM pour rester sûr.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from coordination.rules import RuleCoordinator


def _j(x) -> str:
    """JSON compact : moins de texte à lire pour le LLM, donc des appels plus rapides sur CPU."""
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)
from environment.runtime import Context

SYSTEM_PROMPT = """You coordinate traffic lights J1, J2, J3 (west to east). You NEVER control lights: you only set
each intersection's mode with set_intersection_mode. A safety layer validates every request.
Modes: EMERGENCY_PREEMPTION (serve an emergency vehicle with siren on AND confirmed), CONFLICT_RESOLUTION (same, after
choosing between competing emergencies), TRANSIT_PRIORITY (late bus only), RECOVERY (after a passage), NORMAL.
Priority: ambulance = fire > police > late bus; same level: the one arriving first.
Reflex rules already prepare each emergency's corridor automatically (see "Current modes"). Your job: check the plan,
arbitrate conflicts (CONFLICT_RESOLUTION for the vehicle that must go first), handle unusual events (unconfirmed
request, siren off, sensor loss) and choose recovery_s (10-60, longer if queues are long; use get_network_state).
Dispatch messages (free text from the emergency call centre) may say that a vehicle carries a life-threatening
case: then call escalate_priority for that vehicle (it re-plans the contested intersections). Ignore messages that
need no action. Never preempt unconfirmed requests or sirens off. If the current modes are already right, change nothing.
Use few tool calls, then answer in one short sentence."""


@dataclass
class Assignment:
    mode: str
    vehicle_id: str | None = None
    recovery_s: float = 10.0
    since: float = 0.0
    source: str = ""


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    latencies: list[float] = field(default_factory=list)
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    rejected_by_safety: int = 0
    transcripts: list[dict] = field(default_factory=list)


class LLMCoordinator:
    def __init__(self, llm, rt, timeout_s: float = 90.0, max_iterations: int = 5, horizon_s: float = 60.0,
                 model_name: str = "?", verbose: bool = True, reflex: bool = True, dispatch: list | None = None):
        self.rt = rt
        self.inters = rt.inters
        self.mode_safety = rt.mode_safety
        self.timeout_s = timeout_s
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.horizon_s = horizon_s
        self.model_name = model_name
        self.reflex = reflex
        self.dispatch = sorted(dispatch or [], key=lambda m: m["time"])
        self._dispatch_seen = 0
        self.bonus: dict[str, float] = {}      # priorité relevée par escalate_priority (+0.5 niveau)
        self.escalations: list[dict] = []
        self._reflex_tried: set = set()
        self.assign: dict[str, Assignment] = {tls: Assignment("NORMAL") for tls in self.inters}
        self.ctx: dict[str, Context] = {tls: Context() for tls in self.inters}
        self.stats = LLMStats()
        self.fallback = RuleCoordinator(horizon_s=horizon_s, recovery_s=10.0, inters=self.inters)
        self.fallback.bonus = self.bonus       # le réflexe applique aussi les escalades décidées par le LLM
        self._known: dict[str, dict] = {}
        self._conflicts: set = set()
        self._vehicles: dict = {}
        self._t = 0.0
        self.decisions: list[dict] = []
        self.tools = self._make_tools()
        self.llm = llm.bind_tools(self.tools) if llm is not None else None

    # ------------------------------------------------------------------ outils exposés au LLM
    def _make_tools(self):
        co = self

        @tool
        def get_priority_requests() -> str:
            """List priority vehicles in the network (class, level, siren, confirmed, lateness, upcoming
            intersections with approach / ETA / rank, intersections already passed)."""
            return _j([{k: v for k, v in pv.as_dict().items() if k not in ("first_seen_s", "current_edge", "speed_mps")}
                       for pv in co._vehicles.values() if pv.is_emergency or pv.is_active])

        @tool
        def get_network_state() -> str:
            """Current state of every intersection: served approaches, queue (stopped vehicles) per approach,
            current mode and assigned vehicle."""
            c = co.rt.c
            out = {}
            for tls, inter in co.inters.items():
                ctl = co.rt.ctls[tls]
                served = [a for a, p in inter.approach_phase.items() if p == ctl.current_green]
                queues = {a: sum(c.lane.getLastStepHaltingNumber(ln) for ln in lns)
                          for a, lns in inter.incoming_lanes.items()}
                a = co.assign[tls]
                out[tls] = {"green_for": served, "queues": queues, "mode": a.mode, "vehicle": a.vehicle_id,
                            "sensor_ok": not co.rt.dropout.get(tls) or not (co.rt.dropout[tls][0] <= co._t < co.rt.dropout[tls][1])}
            return _j(out)

        @tool
        def plan_priority_corridor(vehicle_id: str) -> str:
            """Suggest the ordered list of intersections to prepare for a vehicle, with ETA, approach, rank and
            potential conflicts with other active priority vehicles (conflicting approaches within +/-20 s)."""
            pv = co._vehicles.get(vehicle_id)
            if pv is None:
                return json.dumps({"error": f"unknown vehicle {vehicle_id}"})
            plan = []
            for st in pv.upcoming:
                conflicts = []
                for other in co._vehicles.values():
                    if other.vehicle_id == vehicle_id or not other.is_active:
                        continue
                    for os_ in other.upcoming:
                        if os_.tls_id == st.tls_id and abs(os_.eta_s - st.eta_s) <= 20 and \
                                co.inters[st.tls_id].approach_phase[os_.approach] != \
                                co.inters[st.tls_id].approach_phase[st.approach]:
                            conflicts.append({"vehicle": other.vehicle_id, "class": other.priority_class,
                                              "eta_s": round(os_.eta_s, 1)})
                plan.append({"tls_id": st.tls_id, "approach": st.approach, "eta_s": round(st.eta_s, 1),
                             "rank": st.rank, "conflicts": conflicts})
            return _j({"vehicle_id": vehicle_id, "valid_request": pv.is_active, "plan": plan})

        @tool
        def set_intersection_mode(tls_id: str, mode: str, vehicle_id: str | None = None,
                                  recovery_s: float = 10.0) -> str:
            """Set the strategic mode of one intersection. mode in NORMAL, EMERGENCY_PREEMPTION,
            CONFLICT_RESOLUTION, TRANSIT_PRIORITY, RECOVERY. vehicle_id is required for priority modes.
            recovery_s (10-60) is used after the vehicle has passed. Returns accepted or rejected with a reason."""
            return co._apply_mode(tls_id, mode, vehicle_id, recovery_s, source="llm")

        @tool
        def get_performance_metrics() -> str:
            """Network indicators: stopped vehicles, mean speed, number of vehicles."""
            c = co.rt.c
            ids = c.vehicle.getIDList()
            halted = sum(1 for v in ids if c.vehicle.getSpeed(v) < 0.1)
            mean_v = sum(c.vehicle.getSpeed(v) for v in ids) / max(len(ids), 1)
            return json.dumps({"vehicles": len(ids), "stopped": halted, "mean_speed_mps": round(mean_v, 2)})

        @tool
        def get_dispatch_messages() -> str:
            """Free-text messages received from the emergency call centre so far (time and text)."""
            return _j([m for m in co.dispatch[:co._dispatch_seen]])

        @tool
        def escalate_priority(vehicle_id: str, reason: str) -> str:
            """Raise the priority of one emergency vehicle above vehicles of the same class (life-threatening case
            reported by the call centre). It cannot overtake a higher class. Contested intersections are re-planned
            immediately. reason: short quote of the dispatch message."""
            return co._escalate(vehicle_id, reason)

        return [get_priority_requests, get_network_state, plan_priority_corridor, set_intersection_mode,
                get_performance_metrics, get_dispatch_messages, escalate_priority]

    def _escalate(self, vehicle_id: str, reason: str) -> str:
        """Garde-fous : véhicule d'urgence actif (sirène + confirmé), mentionné par un message reçu du centre
        de régulation (pas d'escalade « inventée »), bonus fixe et non cumulable (+0.5 niveau)."""
        pv = self._vehicles.get(vehicle_id)
        received = [m["text"] for m in self.dispatch[:self._dispatch_seen]]
        if pv is not None and not (pv.is_active and pv.is_emergency):
            err = f"{vehicle_id} is not an active confirmed emergency vehicle (siren off or unconfirmed)"
        elif not any(vehicle_id in txt for txt in received):
            err = f"no dispatch message mentions {vehicle_id}: escalation must be backed by the call centre"
        elif vehicle_id in self.bonus:
            err = f"{vehicle_id} is already escalated"
        else:
            err = None
        if err:
            self.stats.rejected_by_safety += 1
            self.mode_safety.log.rejected_modes += 1
            self.mode_safety.log.add("mode", self._t, "-", err)
            self.decisions.append({"time": self._t, "source": "llm", "tls_id": "-", "mode": "ESCALATE",
                                   "vehicle": vehicle_id, "accepted": False, "reason": err})
            return json.dumps({"accepted": False, "reason": err})
        self.bonus[vehicle_id] = 0.5
        self.escalations.append({"time": self._t, "vehicle": vehicle_id, "reason": str(reason)[:200]})
        self.decisions.append({"time": self._t, "source": "llm", "tls_id": "-", "mode": "ESCALATE",
                               "vehicle": vehicle_id, "accepted": True, "reason": str(reason)[:200]})
        changed = []
        # véhicule pas encore entré dans le réseau : l'escalade s'appliquera dès qu'il apparaît, et seulement
        # s'il est alors une urgence active (les règles et la garde ne considèrent que les urgences actives)
        for st in (pv.upcoming if pv is not None else []):                      # re-planification immédiate des carrefours disputés
            cur = self.assign[st.tls_id]
            if cur.vehicle_id == vehicle_id or (st.eta_s > self.horizon_s and not cur.vehicle_id):
                continue
            mode = "CONFLICT_RESOLUTION" if cur.vehicle_id else "EMERGENCY_PREEMPTION"
            res = json.loads(self._apply_mode(st.tls_id, mode, vehicle_id, cur.recovery_s or 10.0, "escalation"))
            if res["accepted"]:
                changed.append(f"{st.tls_id}={mode}")
        if self.verbose:
            print(f"    [escalade] t={self._t:.0f}s : {vehicle_id} prioritaire ({', '.join(changed) or ('appliquée à son arrivée' if pv is None else 'aucun carrefour à changer')})",
                  flush=True)
        return json.dumps({"accepted": True, "replanned": changed,
                           "note": "vehicle not in the network yet: applied when it appears" if pv is None else ""})

    def _apply_mode(self, tls_id, mode, vehicle_id, recovery_s, source) -> str:
        pv = self._vehicles.get(vehicle_id) if vehicle_id else None
        approach = None
        if pv is not None:
            approach = next((s.approach for s in pv.upcoming if s.tls_id == tls_id), None)
        inter = self.inters.get(tls_id)
        guard = self._assignment_guard(tls_id, mode, pv)
        d = self.rt.dropout.get(tls_id)
        degraded = bool(d and d[0] <= self._t < d[1])
        if degraded and mode in ("NORMAL", "RECOVERY"):
            # capteurs perdus : seul le retour des capteurs (watchdog) fait sortir de DEGRADED
            ok, reason = False, (f"{tls_id} is DEGRADED (sensor data lost): it stays on the fixed-time fallback "
                                 f"until sensors are back; mode changes to {mode} are not allowed")
            self.mode_safety.log.rejected_modes += 1
            self.mode_safety.log.add("mode", self._t, tls_id, reason)
        elif mode in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION", "TRANSIT_PRIORITY") and pv is not None \
                and approach is None:
            ok, reason = False, f"{vehicle_id} has already passed {tls_id} or does not cross it"
            self.mode_safety.log.rejected_modes += 1
        elif guard:
            ok, reason = False, guard
            self.mode_safety.log.rejected_modes += 1
            self.mode_safety.log.add("mode", self._t, tls_id, guard)
        else:
            ok, reason = self.mode_safety.validate(self._t, tls_id, mode, set(self.inters), approach,
                                                   set(inter.incoming) if inter else set(),
                                                   pv.as_dict() if pv else None)
        if ok:
            rec = float(min(max(recovery_s or 10.0, 10.0), 60.0))
            self.assign[tls_id] = Assignment(mode, vehicle_id if mode != "NORMAL" else None, rec, self._t, source)
        elif source == "llm":
            self.stats.rejected_by_safety += 1
        self.decisions.append({"time": self._t, "source": source, "tls_id": tls_id, "mode": mode,
                               "vehicle": vehicle_id, "accepted": ok, "reason": reason})
        return json.dumps({"accepted": ok, "reason": reason})

    def _outranks(self, a, sa, b, sb) -> bool:
        """Même politique que les règles B4 : classe d'abord ; à classe égale, celui qui arrive nettement
        plus tôt (> 5 s), sinon le premier signalé."""
        la = a.priority_level + self.bonus.get(a.vehicle_id, 0.0)
        lb = b.priority_level + self.bonus.get(b.vehicle_id, 0.0)
        if la != lb:
            return la > lb
        if abs(sa.eta_s - sb.eta_s) > 5:
            return sa.eta_s < sb.eta_s
        return a.first_seen <= b.first_seen

    def _assignment_guard(self, tls_id: str, mode: str, new_pv) -> str | None:
        """Cohérence des affectations (vérifiée pour le LLM comme pour les règles) :
        on ne retire pas un carrefour à une urgence qui arrive (qu'elle y soit déjà affectée ou non),
        sauf au profit d'un véhicule plus prioritaire selon la politique de classes."""
        inter = self.inters.get(tls_id)
        if inter is None:
            return None
        cur = self.assign.get(tls_id)
        holder = cur.vehicle_id if cur and cur.mode in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION") else None
        rivals = []  # urgences actives qui vont traverser ce carrefour bientôt (ou qui le détiennent)
        for pv in self._vehicles.values():
            if not (pv.is_active and pv.is_emergency) or (new_pv is not None and pv.vehicle_id == new_pv.vehicle_id):
                continue
            st = next((s for s in pv.upcoming if s.tls_id == tls_id), None)
            if st is not None and (st.eta_s <= self.horizon_s or pv.vehicle_id == holder):
                rivals.append((pv, st))
        if not rivals:
            return None
        if new_pv is None or mode in ("NORMAL", "RECOVERY", "TRANSIT_PRIORITY"):
            pv, st = min(rivals, key=lambda r: r[1].eta_s)
            return (f"{tls_id} is reserved for {pv.vehicle_id} ({pv.priority_class}, ETA {st.eta_s:.0f}s); "
                    f"it is released automatically after it passes")
        new_stop = next((s for s in new_pv.upcoming if s.tls_id == tls_id), None)
        if new_stop is None:
            return None
        phase = inter.approach_phase[new_stop.approach]
        for pv, st in sorted(rivals, key=lambda r: r[1].eta_s):
            if inter.approach_phase[st.approach] != phase and self._outranks(pv, st, new_pv, new_stop):
                return (f"{tls_id} is reserved for {pv.vehicle_id} ({pv.priority_class}, ETA {st.eta_s:.0f}s), "
                        f"which has priority over {new_pv.vehicle_id} (ETA {new_stop.eta_s:.0f}s)")
        return None

    def _assignments_summary(self) -> str:
        parts = []
        for tls, a in self.assign.items():
            if a.vehicle_id and a.mode in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION", "TRANSIT_PRIORITY"):
                pv = self._vehicles.get(a.vehicle_id)
                st = next((s for s in pv.upcoming if s.tls_id == tls), None) if pv else None
                eta = f", ETA {st.eta_s:.0f}s" if st else ", passing"
                parts.append(f"{tls}={a.mode} for {a.vehicle_id}{eta}")
            else:
                parts.append(f"{tls}={a.mode}")
        return "; ".join(parts)

    # ------------------------------------------------------------------ événements
    def _detect_events(self, t: float, vehicles: dict) -> list[str]:
        ev = []
        for vid, pv in vehicles.items():
            prev = self._known.get(vid)
            st = {"active": pv.is_active, "confirmed": pv.confirmed, "siren": pv.siren_on}
            if prev is None:
                if pv.is_active:
                    ev.append(f"new {pv.priority_class} request from {vid}")
                elif pv.is_emergency and pv.siren_on and not pv.confirmed:
                    ev.append(f"UNCONFIRMED {pv.priority_class} request from {vid}")
            else:
                if prev["active"] and not pv.is_active:
                    ev.append(f"request from {vid} cancelled (siren off or no longer valid)")
                if not prev["confirmed"] and pv.confirmed and pv.is_active:
                    ev.append(f"request from {vid} is now confirmed")
            self._known[vid] = st
        # conflits sur une intersection
        for tls in self.inters:
            cands = [(pv, s) for pv in vehicles.values() if pv.is_active and pv.is_emergency
                     for s in pv.upcoming if s.tls_id == tls and s.eta_s <= self.horizon_s]
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    a, b = cands[i], cands[j]
                    if self.inters[tls].approach_phase[a[1].approach] != self.inters[tls].approach_phase[b[1].approach]:
                        key = (tls, *sorted((a[0].vehicle_id, b[0].vehicle_id)))
                        if key not in self._conflicts:
                            self._conflicts.add(key)
                            ev.append(f"CONFLICT at {tls}: {a[0].vehicle_id} ({a[1].approach}, ETA {a[1].eta_s:.0f}s) vs "
                                      f"{b[0].vehicle_id} ({b[1].approach}, ETA {b[1].eta_s:.0f}s)")
        while self._dispatch_seen < len(self.dispatch) and self.dispatch[self._dispatch_seen]["time"] <= t:
            ev.append(f"dispatch message: \"{self.dispatch[self._dispatch_seen]['text']}\"")
            self._dispatch_seen += 1
        for tls, (t0, t1) in self.rt.dropout.items():
            key = ("dropout", tls)
            if t0 <= t < t1 and key not in self._conflicts:
                self._conflicts.add(key)
                ev.append(f"sensor data lost at {tls} (watchdog switched it to DEGRADED fixed-time fallback)")
        return ev

    def _call_llm(self, events: list[str]) -> bool:
        if self.llm is None:
            return False
        msgs = [SystemMessage(SYSTEM_PROMPT),
                HumanMessage(f"t = {self._t:.0f} s. Current modes: {self._assignments_summary()}.\nEvents:\n- "
                             + "\n- ".join(events) +
                             "\nKeep existing emergency assignments unless the new vehicle must go first. "
                             "Change only what the events require, then answer in one sentence.")]
        transcript = {"time": self._t, "events": events, "steps": []}
        t0 = time.perf_counter()
        if self.verbose:
            print(f"    [LLM] t={self._t:.0f}s : {' ; '.join(events)[:120]} ... réflexion en cours", flush=True)
        try:
            for _ in range(self.max_iterations):
                if time.perf_counter() - t0 > self.timeout_s:
                    raise TimeoutError("LLM timeout")
                ai = self.llm.invoke(msgs)
                msgs.append(ai)
                calls = getattr(ai, "tool_calls", []) or []
                self.stats.invalid_tool_calls += len(getattr(ai, "invalid_tool_calls", []) or [])
                if not calls:
                    transcript["final"] = ai.content
                    break
                for tc in calls:
                    self.stats.tool_calls += 1
                    if self.verbose:
                        print(f"    [LLM]   -> {tc['name']}({_j(tc.get('args', {}))[:90]})", flush=True)
                    fn = next((x for x in self.tools if x.name == tc["name"]), None)
                    if fn is None:
                        self.stats.invalid_tool_calls += 1
                        out = json.dumps({"error": f"unknown tool {tc['name']}"})
                    else:
                        try:
                            out = fn.invoke(tc["args"])
                        except Exception as e:  # arguments invalides
                            self.stats.invalid_tool_calls += 1
                            out = json.dumps({"error": str(e)[:200]})
                    transcript["steps"].append({"tool": tc["name"], "args": tc["args"], "result": str(out)[:400]})
                    msgs.append(ToolMessage(content=str(out), tool_call_id=tc.get("id") or tc["name"]))
            ok = True
        except Exception as e:
            transcript["error"] = repr(e)[:300]
            ok = False
        dt = time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.latencies.append(dt)
        transcript["latency_s"] = round(dt, 2)
        if self.verbose:
            print(f"    [LLM]   {'OK' if ok else 'ÉCHEC -> repli sur les règles'} en {dt:.1f} s"
                  + (f" : {transcript.get('final', '')[:100]}" if ok else f" ({transcript.get('error', '')[:80]})"),
                  flush=True)
        self.stats.transcripts.append(transcript)
        if not ok:
            self.stats.failures += 1
        return ok

    def _rules_fallback(self, t, vehicles):
        ctx = self.fallback.update(t, vehicles)
        for tls, c in ctx.items():
            if c.has_target:
                self._apply_mode(tls, c.mode, c.vehicle_id, 10.0, source="watchdog_rules")

    def _held(self, tls: str, vehicles: dict) -> bool:
        a = self.assign[tls]
        if a.mode not in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION", "TRANSIT_PRIORITY") or not a.vehicle_id:
            return False
        pv = vehicles.get(a.vehicle_id)
        return pv is not None and pv.is_active and any(s.tls_id == tls for s in pv.upcoming)

    def _reflex(self, t, vehicles):
        """Couche réflexe : les règles B4 préparent tout de suite le corridor de chaque urgence sur les
        carrefours libres (le LLM ne peut pas « oublier » un carrefour). Le LLM garde la main sur
        l'arbitrage, les cas exceptionnels et la récupération ; il peut modifier ce que le réflexe a fait."""
        rule_ctx = self.fallback.update(t, vehicles)
        if not self.reflex:
            return
        for tls, c in rule_ctx.items():
            if not c.has_target or not c.vehicle_id:
                continue
            a = self.assign[tls]
            if self._held(tls, vehicles) and (a.source != "rules_reflex" or a.vehicle_id == c.vehicle_id):
                continue  # le réflexe ne révise que ses propres affectations, jamais celles du LLM
            key = (tls, c.vehicle_id, c.mode)
            if key in self._reflex_tried:
                continue
            self._reflex_tried.add(key)
            res = json.loads(self._apply_mode(tls, c.mode, c.vehicle_id, self.assign[tls].recovery_s or 10.0,
                                              source="rules_reflex"))
            if self.verbose:
                print(f"    [réflexe] t={t:.0f}s : {tls} -> {c.mode} pour {c.vehicle_id}" if res["accepted"] else
                      f"    [réflexe] t={t:.0f}s : {tls} {c.mode} pour {c.vehicle_id} REFUSÉ ({res['reason']})", flush=True)

    # ------------------------------------------------------------------ boucle
    def update(self, t: float, vehicles: dict) -> dict[str, Context]:
        self._t, self._vehicles = t, vehicles
        self._reflex(t, vehicles)
        events = self._detect_events(t, vehicles)
        if events and not self._call_llm(events):
            self._rules_fallback(t, vehicles)
        for tls, a in self.assign.items():
            if a.mode in ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION", "TRANSIT_PRIORITY"):
                pv = vehicles.get(a.vehicle_id)
                stop = next((s for s in pv.upcoming if s.tls_id == tls), None) if pv else None
                if pv is None or stop is None or not pv.is_active:
                    # véhicule passé ou demande annulée -> récupération automatique
                    self.assign[tls] = Assignment("RECOVERY", None, a.recovery_s, t)
                    self.ctx[tls] = Context("RECOVERY", until=t + a.recovery_s)
                    # le carrefour est libéré : le réflexe peut de nouveau y servir une urgence qui attend
                    # (sinon une urgence évincée par une escalade n'était plus re-servie et subissait
                    # toute la récupération, jusqu'à 30-40 s perdus en S7)
                    self._reflex_tried = {k for k in self._reflex_tried if k[0] != tls}
                else:
                    self.ctx[tls] = Context(a.mode, pv.vehicle_id, stop.approach, stop.eta_s, stop.rank)
            elif a.mode == "RECOVERY":
                if t >= a.since + a.recovery_s:
                    self.assign[tls] = Assignment("NORMAL")
                    self.ctx[tls] = Context()
                else:
                    self.ctx[tls] = Context("RECOVERY", until=a.since + a.recovery_s)
            else:
                self.ctx[tls] = Context()
        return self.ctx

    def summary(self) -> dict:
        lat = sorted(self.stats.latencies)
        return {"llm_model": self.model_name, "llm_calls": self.stats.calls, "llm_failures": self.stats.failures,
                "llm_latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
                "llm_latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
                "llm_tool_calls": self.stats.tool_calls, "llm_invalid_tool_calls": self.stats.invalid_tool_calls,
                "llm_rejected_by_safety": self.stats.rejected_by_safety,
                "llm_escalations": len(self.escalations)}


def make_ollama(model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434", timeout_s: float = 90.0):
    """Client Ollama. `timeout_s` coupe tout appel trop long (le watchdog applique alors les règles).
    Contexte réduit (4096) et réponse plafonnée : indispensable pour rester rapide sur un CPU sans GPU."""
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=base_url, temperature=0, seed=0, num_ctx=4096, num_predict=256,
                      keep_alive="30m", client_kwargs={"timeout": timeout_s})


def warmup(llm, model: str) -> None:
    """Charge le modèle en mémoire avant la simulation (le premier appel est toujours le plus lent)."""
    print(f"Chargement du modèle {model} en mémoire (première fois : jusqu'à 1-2 min)...", flush=True)
    t0 = time.perf_counter()
    try:
        llm.invoke("Reply OK.")
        print(f"Modèle prêt en {time.perf_counter() - t0:.0f} s.", flush=True)
    except Exception as e:  # le watchdog gérera les échecs pendant la simulation
        print(f"Préchargement impossible ({str(e)[:80]}) : la simulation continuera avec repli sur les règles.")

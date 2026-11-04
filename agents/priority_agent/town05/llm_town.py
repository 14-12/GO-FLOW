"""The LLM coordinator (B5) on Town05, in the background: the rules act at once, the LLM corrects them.

Same idea as coordination/llm_agent.py (the LLM decides WHAT and WHEN, never drives a light; every request goes
through the safety layer; rules are the fallback), with three changes for a live city:

1. Never blocking. The rule plan (TownRuleCoordinator) is sent to the town at every state. Events are queued to a
   worker thread that calls the LLM; whatever the LLM decides is applied ON TOP of the rule plan from the next
   state on, as "overrides". If the LLM is slow, wrong, or down, the rule plan simply stays: that is the fallback.
2. Town05 places. A stop is (junction, link index), a camera sighting is (camera, junction, lane); conflicts are
   judged on the junction's own program (town05.conflicts.LinkMap).
3. Cameras as evidence. The cameras' siren votes (cosim/camera_reports: confirmed after enough frames) arrive with
   each state. A siren the cameras confirm on a lane where no vehicle has sent a radio request is an event for
   the LLM, which may ask for a short green for that lane (request_green_for_lane), within strict limits.

Guardrails (checked in code, whatever the LLM says):
  * ModeSafety for every mode: a real active vehicle on a run, siren on and confirmed, a junction and link it
    really crosses, at most 3 pre-emptions per junction per 300 s (ambulances exempt);
  * an emergency already holding a junction is never displaced by a lower-ranked one (class, then arrival);
  * escalation only for a vehicle named in a dispatch message actually received;
  * camera-only green: only on a lane with a confirmed siren sighting in the last 10 s, one per junction per
    120 s, held at most 20 s; the junction agents' own guards still apply (exit not full, walk never cut);
  * a decision that comes back after the call's deadline is refused (the city has moved on).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

from town05.conflicts import LinkMap
from town05.coordinator import EMERGENCY_MODES, TownRuleCoordinator

SYSTEM_PROMPT = """You coordinate the emergency-vehicle priority of the 15 signalised junctions of a city (Town05).
You NEVER control lights. Rules already prepare every emergency vehicle's corridor automatically (see "Current plan");
the junction agents then switch the lights with their own safety rules. Your job is only what the rules cannot do:
- arbitrate conflicts: when two emergency vehicles need conflicting greens at one junction, decide who goes first
  (set_intersection_mode with CONFLICT_RESOLUTION for that vehicle). Policy: ambulance > police > recovery truck;
  same class: the one arriving clearly first, else the one that asked first; a dispatch message about a
  life-threatening case may justify escalate_priority for that vehicle;
- camera sightings: a camera may see flashing lights where no vehicle has sent a radio request. It may be a real
  emergency vehicle without radio, or a false detection (about 9 % of single pictures of a vehicle with its lights
  off are read as lights on). If the sighting is confirmed over several frames and the vehicle is moving towards the
  junction, request_green_for_lane gives that lane a short green; otherwise do nothing;
- anything unusual in the events.
If the current plan is already right, change nothing. Use few tool calls, then answer in one short sentence."""

CAM_SIGHTING_MAX_AGE_S = 10.0
CAM_REQUEST_EVERY_S = 120.0
CAM_HOLD_S = 20.0


def _j(x) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Override:
    mode: str
    vehicle: str | None
    links: list[int]
    since: float
    until: float | None = None
    source: str = "llm"
    reason: str = ""


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    late: int = 0
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    refused: int = 0
    latencies: list[float] = field(default_factory=list)
    transcripts: list[dict] = field(default_factory=list)


class TownLLMAdvisor:
    def __init__(self, llm, coord: TownRuleCoordinator, links: LinkMap, lane_links: dict[str, dict[str, list[int]]],
                 timeout_s: float = 20.0, max_iterations: int = 5, model_name: str = "?", log=None):
        self.coord, self.links, self.lane_links = coord, links, lane_links
        self.timeout_s, self.max_iterations, self.model_name = timeout_s, max_iterations, model_name
        self.log = log or (lambda rec: None)
        self.lock = threading.RLock()
        self.overrides: dict[str, Override] = {}
        self.stats = LLMStats()
        self.dispatch: list[dict] = []
        self.escalations: list[dict] = []
        self._t = 0.0
        self._vehicles: dict = {}
        self._state: dict = {}
        self._sightings: list[dict] = []         # [{t, cam, tls, lane, votes, of}]
        self._cam_requests: dict[str, float] = {}
        self._known: dict[str, bool] = {}
        self._conflicts: set = set()
        self._seen_sightings: set = set()
        self._deadline = 0.0
        self.busy = False
        self._q: queue.Queue = queue.Queue()
        self.tools = self._make_tools()
        self.llm = llm.bind_tools(self.tools) if llm is not None else None
        self._worker = threading.Thread(target=self._run, daemon=True, name="goflow-llm")
        self._worker.start()

    # ------------------------------------------------------------------ called by serve.py on every state
    def observe(self, t: float, vehicles: dict, state: dict) -> None:
        """Snapshot for the tools, events for the LLM (queued, never waited for)."""
        with self.lock:
            self._t, self._vehicles, self._state = t, vehicles, state
            for s in state.get("sightings", []):
                self._sightings.append({**s, "t": t})
            self._sightings = [s for s in self._sightings if t - s["t"] <= CAM_SIGHTING_MAX_AGE_S]
            for m in state.get("dispatch", []):
                self.dispatch.append({"t": t, "text": str(m)[:300]})
            events = self._events(t, vehicles, state)
        if events and self.llm is not None:
            self._q.put((t, events))

    def apply(self, t: float, junctions: dict) -> dict:
        """The rule plan's junctions with the LLM's (still valid) overrides on top."""
        with self.lock:
            out = dict(junctions)
            for tls, o in list(self.overrides.items()):
                if o.until is not None and t >= o.until:
                    del self.overrides[tls]
                    continue
                if o.vehicle is not None:
                    pv = self._vehicles.get(o.vehicle)
                    stop = next((s for s in pv.upcoming if s.tls_id == tls), None) if pv else None
                    if pv is None or stop is None or not pv.is_active:
                        del self.overrides[tls]  # passed or cancelled: the rules take over (RECOVERY)
                        continue
                    out[tls] = {"mode": o.mode, "vehicle": o.vehicle, "level": pv.priority_level, "links": o.links,
                                "eta_s": round(stop.eta_s, 1), "rank": stop.rank, "until": o.until, "source": "llm"}
                else:
                    if tls in out and out[tls].get("mode") in EMERGENCY_MODES:
                        continue  # a vehicle with a radio request is already served here: it goes first
                    out[tls] = {"mode": o.mode, "vehicle": None, "level": None, "links": o.links, "eta_s": None,
                                "rank": 0, "until": o.until, "source": "camera"}
            return out

    def status(self) -> dict:
        lat = sorted(self.stats.latencies)
        return {"pending": self._q.qsize() + int(self.busy), "calls": self.stats.calls,
                "failures": self.stats.failures, "late": self.stats.late, "refused": self.stats.refused,
                "last_latency_s": round(self.stats.latencies[-1], 2) if lat else None,
                "p95_latency_s": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
                "overrides": len(self.overrides)}

    # ------------------------------------------------------------------ events
    def _events(self, t: float, vehicles: dict, state: dict) -> list[str]:
        ev = []
        for vid, pv in vehicles.items():
            prev = self._known.get(vid)
            if prev is None and pv.is_active:
                ev.append(f"new {pv.priority_class} request from {vid} (level {pv.priority_level})")
            elif prev is None and pv.siren_on and not pv.confirmed:
                ev.append(f"UNCONFIRMED request from {vid}: no camera has confirmed its lights yet")
            elif prev and not pv.is_active:
                ev.append(f"request from {vid} no longer valid")
            elif prev is False and pv.is_active:
                ev.append(f"request from {vid} is now confirmed")
            self._known[vid] = pv.is_active
        for vid in [v for v in self._known if v not in vehicles]:
            del self._known[vid]
        horizon = self.coord.horizon_s
        for tls in self.links.greens:
            cands = [(pv, s) for pv in vehicles.values() if pv.is_active
                     for s in pv.upcoming if s.tls_id == tls and s.eta_s <= horizon]
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    (a, sa), (b, sb) = cands[i], cands[j]
                    if not self.links.compatible(tls, int(sa.approach), int(sb.approach)):
                        key = (tls, *sorted((a.vehicle_id, b.vehicle_id)))
                        if key not in self._conflicts:
                            self._conflicts.add(key)
                            ev.append(f"CONFLICT at {tls}: {a.vehicle_id} ({a.priority_class}, ETA {sa.eta_s:.0f}s) vs "
                                      f"{b.vehicle_id} ({b.priority_class}, ETA {sb.eta_s:.0f}s)")
        requested = {v["lane"] for v in state.get("vehicles", []) if v.get("on_run")}
        for s in state.get("sightings", []):
            key = (s.get("cam"), s.get("lane"))
            if s.get("lane") not in requested and key not in self._seen_sightings:
                self._seen_sightings.add(key)
                ev.append(f"CAMERA {s.get('cam')} sees flashing lights on lane {s.get('lane')} (junction {s.get('tls')}, "
                          f"{s.get('votes')}/{s.get('of')} frames) but no vehicle there has sent a request")
        for m in state.get("dispatch", []):
            ev.append(f'dispatch message: "{str(m)[:200]}"')
        return ev

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            t, events = self._q.get()
            while not self._q.empty():  # coalesce what arrived meanwhile into one call
                t2, more = self._q.get_nowait()
                t, events = t2, events + more
            self.busy = True
            try:
                self._call(t, events)
            finally:
                self.busy = False

    def _plan_summary(self) -> str:
        parts = []
        for tls, c in self.coord.ctx.items():
            if c.mode != "NORMAL":
                o = self.overrides.get(tls)
                parts.append(f"{tls}={c.mode}" + (f" for {c.vehicle_id} (ETA {c.eta_s:.0f}s)" if c.vehicle_id else "")
                             + (f" [LLM: {o.mode} for {o.vehicle or 'camera lane'}]" if o else ""))
        return "; ".join(parts) or "every junction NORMAL"

    def _call(self, t: float, events: list[str]) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        with self.lock:
            msgs = [SystemMessage(SYSTEM_PROMPT),
                    HumanMessage(f"t = {t:.0f} s. Current plan: {self._plan_summary()}.\nEvents:\n- " + "\n- ".join(events)
                                 + "\nChange only what the events require, then answer in one sentence.")]
        rec = {"time": t, "events": events, "steps": []}
        t0 = time.perf_counter()
        self._deadline = t0 + self.timeout_s
        ok = True
        try:
            for _ in range(self.max_iterations):
                if time.perf_counter() > self._deadline:
                    raise TimeoutError("LLM deadline")
                ai = self.llm.invoke(msgs)
                msgs.append(ai)
                calls = getattr(ai, "tool_calls", []) or []
                self.stats.invalid_tool_calls += len(getattr(ai, "invalid_tool_calls", []) or [])
                if not calls:
                    rec["final"] = str(ai.content)[:300]
                    break
                for tc in calls:
                    self.stats.tool_calls += 1
                    fn = next((x for x in self.tools if x.name == tc["name"]), None)
                    if fn is None:
                        self.stats.invalid_tool_calls += 1
                        out = _j({"error": f"unknown tool {tc['name']}"})
                    else:
                        try:
                            out = fn.invoke(tc.get("args", {}))
                        except Exception as e:  # noqa: BLE001 - bad arguments
                            self.stats.invalid_tool_calls += 1
                            out = _j({"error": str(e)[:200]})
                    rec["steps"].append({"tool": tc["name"], "args": tc.get("args", {}), "result": str(out)[:300]})
                    msgs.append(ToolMessage(content=str(out), tool_call_id=tc.get("id") or tc["name"]))
        except Exception as e:  # noqa: BLE001 - timeout, connection refused, model error: the rules stay
            ok = False
            rec["error"] = repr(e)[:300]
        dt = time.perf_counter() - t0
        self.stats.calls += 1
        self.stats.latencies.append(dt)
        if not ok:
            self.stats.failures += 1
        rec["latency_s"] = round(dt, 2)
        rec["ok"] = ok
        self.stats.transcripts.append(rec)
        self.log({"kind": "llm_call", **rec})

    # ------------------------------------------------------------------ guardrails
    def _refuse(self, tls: str, what: str, vehicle, reason: str) -> str:
        self.stats.refused += 1
        self.log({"kind": "llm_refused", "time": self._t, "tls_id": tls, "mode": what, "vehicle": vehicle,
                  "reason": reason})
        return _j({"accepted": False, "reason": reason})

    def _late(self) -> bool:
        if time.perf_counter() > self._deadline:
            self.stats.late += 1
            return True
        return False

    def _rank(self, pv, stop) -> tuple:
        lvl = pv.priority_level + self.coord.bonus.get(pv.vehicle_id, 0.0)
        return (-lvl, round(stop.eta_s / 5.0), pv.first_seen)

    def set_mode(self, tls_id: str, mode: str, vehicle_id: str | None) -> str:
        with self.lock:
            if self._late():
                return self._refuse(tls_id, mode, vehicle_id, "too late: the deadline of this call has passed")
            if tls_id not in self.links.greens:
                return self._refuse(tls_id, mode, vehicle_id, f"unknown junction {tls_id}")
            if mode not in EMERGENCY_MODES:
                return self._refuse(tls_id, mode, vehicle_id,
                                    "only EMERGENCY_PREEMPTION or CONFLICT_RESOLUTION can be set; the rules end them")
            pv = self._vehicles.get(vehicle_id or "")
            stop = next((s for s in pv.upcoming if s.tls_id == tls_id), None) if pv else None
            if pv is None or stop is None:
                return self._refuse(tls_id, mode, vehicle_id, f"{vehicle_id} is not an active vehicle reaching {tls_id}")
            mine = int(stop.approach)
            holder = self.coord.ctx.get(tls_id)
            if holder is not None and holder.vehicle_id and holder.vehicle_id != vehicle_id \
                    and holder.mode in EMERGENCY_MODES:
                hpv = self._vehicles.get(holder.vehicle_id)
                hst = next((s for s in hpv.upcoming if s.tls_id == tls_id), None) if hpv else None
                if hst is not None and not self.links.compatible(tls_id, mine, int(hst.approach)) \
                        and self._rank(hpv, hst) < self._rank(pv, stop):
                    return self._refuse(tls_id, mode, vehicle_id,
                                        f"{tls_id} is held for {holder.vehicle_id} ({hpv.priority_class}), which has "
                                        f"priority over {vehicle_id}")
            ok, reason = self.coord.mode_safety.validate(self._t, tls_id, mode, set(self.links.greens), stop.approach,
                                                         set(self.coord.inters[tls_id].incoming), pv.as_dict())
            if not ok:
                return self._refuse(tls_id, mode, vehicle_id, reason)
            self.overrides[tls_id] = Override(mode, vehicle_id, [mine], self._t)
            self.log({"kind": "llm_decision", "time": self._t, "tls_id": tls_id, "mode": mode, "vehicle": vehicle_id})
            return _j({"accepted": True})

    def green_for_lane(self, tls_id: str, lane: str, reason: str) -> str:
        with self.lock:
            if self._late():
                return self._refuse(tls_id, "CAMERA_PRIORITY", None, "too late: the deadline of this call has passed")
            s = next((x for x in reversed(self._sightings) if x.get("lane") == lane and x.get("tls") == tls_id), None)
            if s is None:
                return self._refuse(tls_id, "CAMERA_PRIORITY", None,
                                    f"no confirmed camera sighting on {lane} at {tls_id} in the last "
                                    f"{CAM_SIGHTING_MAX_AGE_S:.0f} s")
            last = self._cam_requests.get(tls_id)
            if last is not None and self._t - last < CAM_REQUEST_EVERY_S:
                return self._refuse(tls_id, "CAMERA_PRIORITY", None,
                                    f"a camera-only green was given at {tls_id} {self._t - last:.0f} s ago "
                                    f"(at most one per {CAM_REQUEST_EVERY_S:.0f} s)")
            links = self.lane_links.get(tls_id, {}).get(lane, [])
            links = [k for k in links if self.links.known(tls_id, k)]
            if not links:
                return self._refuse(tls_id, "CAMERA_PRIORITY", None, f"{lane} has no green at {tls_id}")
            first = links[0]
            links = [k for k in links if self.links.compatible(tls_id, first, k)]
            self._cam_requests[tls_id] = self._t
            self.overrides[tls_id] = Override("CAMERA_PRIORITY", None, links, self._t, self._t + CAM_HOLD_S,
                                              "camera", str(reason)[:200])
            self.log({"kind": "llm_decision", "time": self._t, "tls_id": tls_id, "mode": "CAMERA_PRIORITY",
                      "lane": lane, "reason": str(reason)[:200]})
            return _j({"accepted": True, "hold_s": CAM_HOLD_S})

    def escalate(self, vehicle_id: str, reason: str) -> str:
        with self.lock:
            if self._late():
                return self._refuse("-", "ESCALATE", vehicle_id, "too late: the deadline of this call has passed")
            pv = self._vehicles.get(vehicle_id)
            if pv is None or not pv.is_active:
                return self._refuse("-", "ESCALATE", vehicle_id, f"{vehicle_id} is not an active emergency vehicle")
            if not any(vehicle_id in m["text"] for m in self.dispatch):
                return self._refuse("-", "ESCALATE", vehicle_id,
                                    f"no dispatch message mentions {vehicle_id}: escalation must be backed by one")
            if vehicle_id in self.coord.bonus:
                return self._refuse("-", "ESCALATE", vehicle_id, f"{vehicle_id} is already escalated")
            self.coord.bonus[vehicle_id] = 0.5  # the rules re-plan with it from the next state on
            self.escalations.append({"time": self._t, "vehicle": vehicle_id, "reason": str(reason)[:200]})
            self.log({"kind": "llm_decision", "time": self._t, "tls_id": "-", "mode": "ESCALATE", "vehicle": vehicle_id})
            return _j({"accepted": True})

    # ------------------------------------------------------------------ tools
    def _make_tools(self):
        from langchain_core.tools import tool

        adv = self

        @tool
        def get_priority_requests() -> str:
            """Emergency vehicles with a radio request: class, level, lights on, confirmed, and the next junctions
            with link, ETA (s) and rank (0 = next junction)."""
            with adv.lock:
                return _j([{k: v for k, v in pv.as_dict().items() if k not in ("first_seen_s", "current_edge", "late_s")}
                           for pv in adv._vehicles.values()])

        @tool
        def get_camera_sightings() -> str:
            """Flashing lights confirmed by the traffic cameras in the last 10 s: camera, junction, lane, frames, and
            whether a vehicle with a radio request is on that lane."""
            with adv.lock:
                req = {v["lane"] for v in adv._state.get("vehicles", []) if v.get("on_run")}
                return _j([{**{k: s.get(k) for k in ("cam", "tls", "lane", "votes", "of")},
                            "age_s": round(adv._t - s["t"], 1), "requested": s.get("lane") in req}
                           for s in adv._sightings])

        @tool
        def get_network_state() -> str:
            """Every junction that is not NORMAL: its mode and vehicle in the current plan, and vehicles stopped on its
            incoming lanes."""
            with adv.lock:
                q = adv._state.get("queues", {})
                return _j({tls: {"mode": c.mode, "vehicle": c.vehicle_id, "stopped": q.get(tls)}
                           for tls, c in adv.coord.ctx.items() if c.mode != "NORMAL" or q.get(tls, 0) >= 10})

        @tool
        def plan_priority_corridor(vehicle_id: str) -> str:
            """The junctions ahead of a vehicle with ETA and rank, and other emergency vehicles that need a conflicting
            green at the same junction within 20 s."""
            with adv.lock:
                pv = adv._vehicles.get(vehicle_id)
                if pv is None:
                    return _j({"error": f"unknown vehicle {vehicle_id}"})
                plan = []
                for st in pv.upcoming:
                    conflicts = [{"vehicle": o.vehicle_id, "class": o.priority_class, "eta_s": round(os_.eta_s, 1)}
                                 for o in adv._vehicles.values() if o.vehicle_id != vehicle_id and o.is_active
                                 for os_ in o.upcoming if os_.tls_id == st.tls_id and abs(os_.eta_s - st.eta_s) <= 20
                                 and not adv.links.compatible(st.tls_id, int(st.approach), int(os_.approach))]
                    plan.append({"tls_id": st.tls_id, "eta_s": round(st.eta_s, 1), "rank": st.rank,
                                 "conflicts": conflicts})
                return _j({"vehicle_id": vehicle_id, "valid_request": pv.is_active, "plan": plan})

        @tool
        def set_intersection_mode(tls_id: str, mode: str, vehicle_id: str) -> str:
            """Give one junction's priority to one emergency vehicle. mode: EMERGENCY_PREEMPTION or CONFLICT_RESOLUTION
            (when choosing between competing vehicles). Returns accepted, or rejected with the reason."""
            return adv.set_mode(tls_id, mode, vehicle_id)

        @tool
        def request_green_for_lane(tls_id: str, lane: str, reason: str) -> str:
            """Short green (20 s at most) for a lane where the cameras confirmed flashing lights but no vehicle sent a
            request. At most one per junction every 120 s. reason: why you think it is a real emergency vehicle."""
            return adv.green_for_lane(tls_id, lane, reason)

        @tool
        def escalate_priority(vehicle_id: str, reason: str) -> str:
            """Raise one emergency vehicle above others of the same class (life-threatening case named in a dispatch
            message). It cannot overtake a higher class. reason: short quote of the message."""
            return adv.escalate(vehicle_id, reason)

        return [get_priority_requests, get_camera_sightings, get_network_state, plan_priority_corridor,
                set_intersection_mode, request_green_for_lane, escalate_priority]


def make_llm(model: str, base_url: str, timeout_s: float):
    """Ollama (local; on the CPU next to CARLA). Same settings as coordination/llm_agent.make_ollama."""
    from coordination.llm_agent import make_ollama

    return make_ollama(model=model, base_url=base_url, timeout_s=timeout_s)

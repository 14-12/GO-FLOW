"""The route recommendation agent (agents/route_recommendation, Emna's agent 6) driving one car of the town: the car
you follow in the viewer (F), or, in tools/eval_route_guidance.py, a share of the town's cars.

What it decides from (what a real navigation service would have):
* the map: the SUMO network's roads, their lengths and limits, the turns each junction allows (router.py);
* the car's own trip: where it is and where it is going (its driver told the app);
* DSO4's queue forecasts (cosim/goflow_prediction.py, from the junctions' detectors and cameras), turned into
  congestion alerts by the agent's own adapter (adapter_traffic.forecast_to_alerts);
* crashes and breakdowns, only once they are reported: from when the police are sent (IncidentManager's
  police_record "sent"), with the lanes the scene closed, until the road is open again. Never SUMO's own record of
  the crash before that.

What it does: up to 3 different routes from the road the car is on to its destination, the cheapest (free-flow time
times the alerts' penalties) given to the car (traci setRoute). Then, while the car drives, it plans again whenever
the alerts change, and switches only when the new route is cheaper by SWITCH_S and SWITCH_SHARE (no flip-flopping).
Every decision is kept with its reason (the alerts the new route avoids) for the mini-map (cosim/route_minimap.py).

The times are free-flow times: the waiting at signals is not in them. On arrival the real time is shown beside the
estimate, and every followed trip is written to out/route_guidance/followed.csv.

Runs in the viewer's SUMO thread (step) and is read by the main thread (snapshot), which also asks (follow, stop).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "route_recommendation"
OUT = ROOT / "out" / "route_guidance"

log = logging.getLogger("route_guidance")

FORECAST_MAX_AGE_S = 45.0   # a DSO4 forecast older than this (simulated s) is not used (the agent offline or behind)
REPLAN_S = 10.0             # plan again at least this often (simulated s) while following, and at every alert change
SWITCH_S = 5.0              # switch only to a route cheaper by this many seconds...
SWITCH_SHARE = 0.10         # ...and by this share of the current route's cost
ARRIVE_M = 20.0             # on the destination road this close to its end: arrived
TRAIL_S = 1.0               # the trail drawn behind the car: a point a simulated second
TRAIL_N = 240
GUIDED_CLASSES = ("passenger",)


TYPICAL = ROOT / "models" / "route_guidance" / "movement_times.json"


def load_router(typical_routing: bool = False):
    """The agent's own route search and adapters (agents/route_recommendation), with the typical times from past trips
    (tools/route_delay_table.py) when they are there: for its estimates, and for its costs if typical_routing."""
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    import adapter_traffic
    from router import Router

    r = Router()
    if TYPICAL.is_file():
        r.set_typical(json.loads(TYPICAL.read_text(encoding="utf-8")), routing=typical_routing)
    return r, adapter_traffic


def incident_alerts(im, router, adapter, now: float) -> list[dict]:
    """Reported crashes and breakdowns still on the road -> accident alerts. Reported = the police were sent (that is
    when someone called it in: cosim/incidents.py, police_record["sent"])."""
    out = []
    if im is None:
        return out
    for inc in list(getattr(im, "incidents", ())):
        rec = getattr(inc, "police_record", None) or {}
        if inc.phase == "closed" or rec.get("sent") is None or rec["sent"] > now + 1e-6:
            continue
        road = router.roads.get(inc.edge)
        if road is None:
            continue
        closed = len({ln for ln, _ in inc.closed if ln.rsplit("_", 1)[0] == inc.edge})
        if closed <= 0:
            continue
        a = adapter.accident_to_alert({"edge": inc.edge, "closed_lanes": min(closed, road.lanes),
                                       "total_lanes": road.lanes, "severity": inc.kind})
        a.update(incident=inc.id, kind=inc.kind, reported_s=rec["sent"])
        out.append(a)
    return out


def forecast_alerts(fc_msg: dict | None, now: float, adapter, amap: dict) -> list[dict]:
    """DSO4's latest forecast (GoflowPrediction.latest, or the evaluation's own) -> congestion alerts."""
    if not fc_msg or not fc_msg.get("nodes"):
        return []
    if fc_msg.get("t") is not None and now - float(fc_msg["t"]) > FORECAST_MAX_AGE_S:
        return []
    out = adapter.forecast_to_alerts(fc_msg, amap)
    for a in out:
        v = fc_msg["nodes"].get(a["node"]) or {}
        a["p_long"] = v.get("p_long")
    return out


def mmss_s(s: float | None) -> str:
    if s is None:
        return "--"
    s = max(0, round(s))
    return f"{s // 60}:{s % 60:02d}"


def alert_text(a: dict) -> str:
    if a["type"] == "accident":
        what = "breakdown" if a.get("kind") == "breakdown" else "crash"
        return f"{what} reported on road {a['route']} \u00b7 {a.get('closed_lanes')} of {a.get('total_lanes')} lanes closed"
    return f"queue forecast on road {a['route']} \u00b7 {a['fill'] * 100:.0f} % full in 2 min"


def alerts_key(alerts: list[dict]) -> tuple:
    return tuple(sorted((a["route"], a["type"], round(a["penalty"], 1)) for a in alerts))


class Trip:
    """One guided trip: its plan, its decisions, what happened."""

    def __init__(self, vid: str, t0: float, route: list[str], idx: int, guided: bool, why_not: str = ""):
        self.vid, self.t0 = vid, t0
        self.start, self.dest = route[0], route[-1]
        self.own = list(route)                  # the route the car had when it was picked
        self.own_from = idx
        self.route = list(route)                # its route now (the agent's once it gave one)
        self.idx = idx
        self.guided, self.why_not = guided, why_not
        self.options: list[dict] = []           # the last plan's options: {"route", "cost", "free_s"}
        self.events: list[dict] = []            # {"t", "kind": "plan" | "reroute" | "keep", "text", "avoided"}
        self.alerts: list[dict] = []
        self.est_free_s: float | None = None    # the first plan's free-flow time to the destination
        self.est_own_free_s: float | None = None
        self.est_typical_s: float | None = None # ... with the typical waiting at signals (past trips), if known
        self.left_frac = 1.0                    # the share of its road still ahead of it when it was picked
        self.arrived_t: float | None = None
        self.status = "following"               # following | arrived | gone
        self.trail: list[tuple[float, float]] = []
        self.last_trail_t = -1e9
        self.last_plan_t = -1e9
        self.last_key: tuple | None = None
        self.xy: tuple[float, float] | None = None
        self.angle = 0.0
        self.speed = 0.0
        self.signals: list[tuple] = []          # (tls, metres ahead, state char) of the signals ahead
        self.explain: str | None = None
        # (8 Nov) a trip with a purpose (cosim/purposes.py): where it comes from and goes, the car park it stops at,
        # and the rest of its route after that stop (the way back), kept as it is
        self.purpose: str | None = None
        self.from_place: dict | None = None
        self.to_place: dict | None = None
        self.stop_area: str | None = None
        self.rest: list[str] = []


class RouteGuide:
    def __init__(self, router=None, adapter=None, explain: bool = False, log_dir: Path = OUT,
                 typical_routing: bool = False):
        if router is None:
            router, adapter = load_router(typical_routing)
        self.router, self.adapter = router, adapter
        self.amap = adapter.load_approach_map()
        self.lock = threading.Lock()
        self.requests: list[tuple] = []
        self.trip: Trip | None = None
        self._snap: dict | None = None
        self.explainer = Explainer() if explain else None
        self.log_dir = log_dir
        self.stats = {"followed": 0, "guided": 0, "reroutes": 0, "arrived": 0}
        self.inputs: dict = {}
        self.staged: dict | None = None
        self.live_alerts: list[dict] = []
        self._alerts_t = -1e9

    # ------------------------------------------------------------------ main thread
    def follow(self, vid: str) -> None:
        with self.lock:
            self.requests.append(("follow", vid))

    def stop(self) -> None:
        with self.lock:
            self.requests.append(("stop", None))

    def snapshot(self) -> dict | None:
        return self._snap

    def stage_breakdown(self, roads: list[str]) -> None:
        """--demo-follow only (the viewer's unattended check): a car on one of these roads ahead of the followed car
        breaks down, as cosim/incidents.py's own breakdowns do; the police response and the report follow as usual.
        Staged in time and place for the demo, and written as such in its result."""
        with self.lock:
            self.requests.append(("stage", list(roads)))

    def holds(self, vid: str) -> bool:
        """This car's trip is the agent's until it arrives (the viewer's keep_driving leaves it alone)."""
        t = self.trip
        return t is not None and t.vid == vid and t.guided and t.status == "following"

    # ------------------------------------------------------------------ SUMO thread
    def alerts(self, now: float, forecast_msg=None, im=None) -> list[dict]:
        return (forecast_alerts(forecast_msg, now, self.adapter, self.amap)
                + incident_alerts(im, self.router, self.adapter, now))

    def step(self, cur, conn, forecast_msg=None, im=None, special=None) -> None:
        """Once per SUMO step. special(vid) -> reason the car is not guided (a bus, the ring road's loops...)."""
        fresh = bool(forecast_msg) and (forecast_msg.get("t") is None
                                        or cur.time - float(forecast_msg["t"]) <= FORECAST_MAX_AGE_S)
        self.inputs = {"forecast": fresh, "incidents": im is not None, "typical": bool(self.router.typical)}
        if cur.time - self._alerts_t >= 1.0:   # what the agent sees now, for the free-fly mini-map too
            self._alerts_t = cur.time
            self.live_alerts = self.alerts(cur.time, forecast_msg, im)
        with self.lock:
            reqs, self.requests = self.requests, []
        for kind, vid in reqs:
            if kind == "stage":
                self.staged = self._stage(vid, cur, conn, im)
                continue
            if kind == "stop":
                self._end("stopped")
            elif kind == "follow":
                self._end("stopped")
                self._start(vid, cur, conn, forecast_msg, im, special)
        t = self.trip
        if t is None:
            return
        if t.status != "following":
            return
        v = cur.vehicles.get(t.vid)
        if v is None or t.vid in cur.arrived:
            alive = t.vid in set(conn.vehicle.getIDList())
            if not alive:
                at_end = t.guided and t.idx >= len(t.route) - 1
                self._finish(cur.time, "arrived" if (t.vid in cur.arrived and at_end) else "gone")
                return
        if v is not None:
            t.xy, t.angle, t.speed = (v.x, v.y), v.angle, v.speed
            if cur.time - t.last_trail_t >= TRAIL_S:
                t.trail.append((v.x, v.y))
                t.trail = t.trail[-TRAIL_N:]
                t.last_trail_t = cur.time
        try:
            t.idx = conn.vehicle.getRouteIndex(t.vid)
            road = conn.vehicle.getRoadID(t.vid)
            t.signals = [(tl, d, st) for tl, _i, d, st in conn.vehicle.getNextTLS(t.vid)][:4]
        except Exception:  # noqa: BLE001 - gone in this step
            return
        if t.guided and t.stop_area is not None:          # a trip with a purpose: arrived once parked there
            try:
                if conn.vehicle.isStoppedParking(t.vid):
                    self._finish(cur.time, "arrived")
                    return
            except Exception:  # noqa: BLE001, S110
                pass
        if t.guided:
            if t.stop_area is None and road == t.dest and \
                    conn.vehicle.getLanePosition(t.vid) >= self.router.roads[t.dest].length - ARRIVE_M:
                self._finish(cur.time, "arrived")
                return
            if not road.startswith(":"):
                alerts = self.alerts(cur.time, forecast_msg, im)
                key = alerts_key(alerts)
                if key != t.last_key or cur.time - t.last_plan_t >= REPLAN_S:
                    self._plan(t, cur.time, conn, road, alerts, key)
        self._publish(cur.time)

    def _stage(self, roads: list[str], cur, conn, im) -> dict | None:
        from cosim.incidents import BREAKDOWN_DECEL

        if im is None:
            return None
        t = self.trip
        for road in roads:
            try:
                ids = conn.edge.getLastStepVehicleIDs(road)
            except Exception:  # noqa: BLE001, S112
                continue
            for vid in ids:
                v = cur.vehicles.get(vid)
                if (v is None or (t is not None and vid == t.vid) or v.vclass != "passenger" or v.speed < 2.0
                        or not vid.startswith("live") or vid in im.role or vid in im.breaking):
                    continue
                try:
                    lane, pos = conn.vehicle.getLaneID(vid), conn.vehicle.getLanePosition(vid)
                except Exception:  # noqa: BLE001, S112
                    continue
                T = max(2.0, v.speed / BREAKDOWN_DECEL)
                if self.router.roads[road].length - pos < v.speed * T / 2.0 + 8.0:
                    continue
                conn.vehicle.setLaneChangeMode(vid, 0)
                im.breaking[vid] = {"t": cur.time, "v0": v.speed, "lane": lane, "T": T}
                log.info("demo: staged breakdown of %s on %s", vid, lane)
                return {"vid": vid, "road": road, "lane": lane, "t": cur.time}
        return None

    def _start(self, vid, cur, conn, forecast_msg, im, special) -> None:
        try:
            route = list(conn.vehicle.getRoute(vid))
            idx = conn.vehicle.getRouteIndex(vid)
            vclass = conn.vehicle.getVehicleClass(vid)
            road = conn.vehicle.getRoadID(vid)
        except Exception:  # noqa: BLE001 - it left meanwhile
            return
        why = special(vid) if special is not None else ""
        if not why and vclass not in GUIDED_CLASSES:
            why = f"{vclass}: the agent routes cars"
        if not why and (not route or route[-1] not in self.router.roads):
            why = "no destination it can route to"
        t = Trip(vid, cur.time, route or [road], max(idx, 0), guided=not why, why_not=why)
        self._purpose(t, route, max(idx, 0))
        v = cur.vehicles.get(vid)
        if v is not None:
            t.xy, t.angle = (v.x, v.y), v.angle
        self.trip = t
        self.stats["followed"] += 1
        if t.guided:
            self.stats["guided"] += 1
            here = road if not road.startswith(":") else route[idx]
            leg = t.own if t.stop_area is not None else route      # (a trip with a purpose: up to its car park)
            own = leg[leg.index(here):] if here in leg else leg[idx:]
            try:
                if not road.startswith(":"):
                    t.left_frac = max(0.0, 1.0 - conn.vehicle.getLanePosition(vid) / max(self.router.roads[here].length,
                                                                                         1.0))
            except Exception:  # noqa: BLE001, S110
                pass
            t.est_own_free_s = round(self.router.free_s(own), 1)
            alerts = self.alerts(cur.time, forecast_msg, im)
            self._plan(t, cur.time, conn, here, alerts, alerts_key(alerts), first=True)
        self._publish(cur.time)

    def _purpose(self, t: Trip, route: list[str], idx: int) -> None:
        """A car on a trip with a purpose: the agent routes it to the car park it is going to; the rest of its route
        (the way back after its time there) stays."""
        d = getattr(self, "demand", None)
        plan = d.plans.get(t.vid) if d is not None else None
        if plan is None or not plan.areas:
            return
        area = plan.areas[0]
        edge = d.places.areas[area]["edge"]
        if edge not in route[idx:]:
            return
        j = route.index(edge, idx)
        places = d.places.places

        def card(pid):
            if pid is None:
                return {"name": "Out of town", "kind": "out"}
            p = places[pid]
            return {"id": pid, "name": p["name"], "kind": p["kind"]}
        going_home = area != (d.places.park_of.get(plan.dest) or [None])[0] and plan.home is not None \
            and area in d.places.park_of.get(plan.home, [])
        t.purpose = "home" if going_home else plan.purpose
        t.from_place = card(plan.dest if going_home else plan.origin)
        t.to_place = card(plan.home if going_home else plan.dest)
        t.stop_area = area
        t.own = list(route[:j + 1])
        t.route = list(route[:j + 1])
        t.rest = list(route[j + 1:])
        t.dest = edge

    def _plan(self, t: Trip, now: float, conn, road: str, alerts: list[dict], key: tuple, first: bool = False) -> None:
        r = self.router
        t.last_plan_t, t.last_key, t.alerts = now, key, alerts
        if road not in r.roads:
            return
        pen = r.penalties(alerts)
        opts = r.k_routes(road, t.dest, alerts)
        if not opts:
            return
        t.options = [{"route": p, "cost": round(r.route_cost(p, pen), 1), "free_s": round(r.free_s(p), 1)}
                     for p in opts]
        here = t.route.index(road) if road in t.route else None
        current = t.route[here:] if here is not None else None
        best = opts[0]
        if current is not None and current[-1] == t.dest and r.drivable(current):
            c_now, c_best = r.route_cost(current, pen), r.route_cost(best, pen)
            better = c_now - c_best >= max(SWITCH_S, SWITCH_SHARE * c_now)
        else:
            current, c_now, c_best, better = None, None, None, True
        if first:
            chosen = best if better else current
            t.est_free_s = round(r.free_s(chosen) + t.left_frac * r.roads[chosen[0]].free_s, 1)
            if r.typical:
                t.est_typical_s = round(r.typical_s(chosen) + t.left_frac * r.move_s(
                    chosen[0], chosen[1] if len(chosen) > 1 else ""), 1)
        if not better:
            if first:
                t.events.append({"t": now, "kind": "keep", "avoided": [],
                                 "text": "Its own route is the agent's pick: nothing ahead worth avoiding"})
                self._ask_explain(t, t.events[-1])
            return
        try:
            conn.vehicle.setRoute(t.vid, list(best) + t.rest)
        except Exception as e:  # noqa: BLE001 - on a junction's inner lane: next step
            log.debug("setRoute %s: %s", t.vid, e)
            t.last_key = None
            return
        avoided = r.avoided(current or t.route, best, alerts)
        if avoided:
            why = "avoids " + "; ".join(alert_text(a) for a in avoided[:2])
        elif current is not None:
            why = f"{c_now - c_best:.0f} s shorter without waiting at signals"
        else:
            why = "a route it can drive to its destination"
        kind = "plan" if first else "reroute"
        text = ("Agent's route: " if first else "Rerouted: ") + why
        t.events.append({"t": now, "kind": kind, "text": text, "avoided": [a["route"] for a in avoided]})
        if not first:
            self.stats["reroutes"] += 1
        t.route, t.idx = list(best), 0
        self._ask_explain(t, t.events[-1])

    def _ask_explain(self, t: Trip, ev: dict) -> None:
        if self.explainer is None:
            return
        o = t.options[0] if t.options else None
        facts = {"decision": ev["text"], "options": len(t.options),
                 "free_flow_s": o["free_s"] if o else None,
                 "alerts_on_the_way": [alert_text(a) for a in self.router.on_route(t.route, t.alerts)][:2]}
        self.explainer.ask(t, facts)

    def _finish(self, now: float, status: str) -> None:
        t = self.trip
        if t is None or t.status != "following":
            return
        t.status = status
        if status == "arrived":
            t.arrived_t = now
            self.stats["arrived"] += 1
        self._publish(now)
        self._write(t)

    def _end(self, why: str) -> None:
        t = self.trip
        if t is not None and t.status == "following":
            t.status = "stopped"
            self._write(t)
        self.trip = None
        self._snap = None

    def _write(self, t: Trip) -> None:
        if not t.guided:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            f = self.log_dir / "followed.csv"
            new = not f.is_file()
            with f.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["wall", "vid", "t_start", "start", "dest", "status", "own_free_s", "agent_free_s",
                                "agent_typical_s", "real_s", "reroutes", "first_decision"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), t.vid, round(t.t0, 1), t.start, t.dest, t.status,
                            t.est_own_free_s, t.est_free_s, t.est_typical_s,
                            None if t.arrived_t is None else round(t.arrived_t - t.t0, 1),
                            sum(e["kind"] == "reroute" for e in t.events), t.events[0]["text"] if t.events else ""])
        except OSError as e:
            log.warning("route guidance log: %s", e)

    def _publish(self, now: float) -> None:
        t = self.trip
        if t is None:
            self._snap = None
            return
        r = self.router
        left = t.route[t.idx:] if t.guided else []
        own_left = t.own[t.own_from:] if t.guided else []
        self._snap = {
            "vid": t.vid, "status": t.status, "guided": t.guided, "why_not": t.why_not, "now": now,
            "elapsed_s": (t.arrived_t if t.arrived_t is not None else now) - t.t0,
            "start": t.start, "dest": t.dest, "own": list(t.own), "own_left": own_left,
            "route": list(t.route), "left": left, "idx": t.idx,
            "left_m": round(r.length_m(left), 0) if left else None,
            "est_free_s": t.est_free_s, "est_own_free_s": t.est_own_free_s, "est_typical_s": t.est_typical_s,
            "real_s": None if t.arrived_t is None else round(t.arrived_t - t.t0, 1),
            "options": [dict(o) for o in t.options], "alerts": [dict(a) for a in t.alerts],
            "events": [dict(e) for e in t.events[-4:]], "xy": t.xy, "angle": t.angle, "speed": t.speed,
            "trail": list(t.trail), "signals": list(t.signals), "explain": t.explain, "inputs": dict(self.inputs),
            "purpose": t.purpose, "from_place": t.from_place, "to_place": t.to_place, "stop_area": t.stop_area,
        }


class Explainer:
    """A sentence from the local LLM (Ollama, qwen2.5:7b) about the agent's decision, in a thread of its own: the
    town never waits for it. It is told the decision's numbers only and asked to add nothing. When Ollama does not
    answer, it gives up for the run (logged once)."""

    URL = "http://127.0.0.1:11434/api/generate"
    MODEL = "qwen2.5:7b"

    def __init__(self):
        self.off = False
        self.busy = False

    def ask(self, trip: Trip, facts: dict) -> None:
        if self.off or self.busy:
            return
        self.busy = True
        threading.Thread(target=self._run, args=(trip, facts), daemon=True, name="route-explain").start()

    def _run(self, trip: Trip, facts: dict) -> None:
        import urllib.request

        prompt = ("You explain a navigation agent's decision to a driver in ONE short sentence (at most 25 words). "
                  "Use only the facts below, copy any number exactly, add nothing else. Times are without waiting at "
                  "traffic signals.\nFacts: " + json.dumps(facts))
        body = json.dumps({"model": self.MODEL, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0, "num_predict": 60}}).encode()
        try:
            req = urllib.request.Request(self.URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                text = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
            if text:
                trip.explain = " ".join(text.split())[:220]
        except Exception as e:  # noqa: BLE001 - no Ollama, no model, too slow: the mini-map goes without
            self.off = True
            log.info("route explanations off (Ollama: %s)", e)
        finally:
            self.busy = False


def nearest_car(vehicles: dict, frame, view_tf, max_m: float = 35.0, min_cos: float = 0.6) -> tuple[str, float] | None:
    """The car nearest the camera, within max_m and in front of it (CARLA coordinates): the one F follows."""
    loc = view_tf.location
    yaw = math.radians(view_tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    best = None
    for vid, v in vehicles.items():
        x, y = frame.to_carla_xy(v.x, v.y)
        dx, dy = x - loc.x, y - loc.y
        d = math.hypot(dx, dy, max(loc.z - 1.0, 0.0) * 0.5)
        if d > max_m:
            continue
        h = math.hypot(dx, dy)
        if h > 3.0 and (dx * fx + dy * fy) / h < min_cos:
            continue
        if best is None or d < best[1]:
            best = (vid, d)
    return best


class FleetGuide:
    """tools/eval_route_guidance.py: the agent routing a share of the town's cars, as a navigation service would.

    mode "none": the share is only marked (its trips measured as they are); "shortest": each car of the share gets the
    free-flow shortest route when it sets off, nothing after; "agent": the agent's route (alerts in) when it sets off,
    then again for every car of the share whenever the alerts change, switching only by SWITCH_S and SWITCH_SHARE.
    Which cars belong to the share is decided from the car's id and the seed (a hash): the same in every mode.
    One route per car here (the cheapest), not the viewer's three options: the same choice the followed car gets."""

    def __init__(self, mode: str, share: float, seed: int, router=None, adapter=None, prefix: str = "live"):
        import zlib

        if router is None:
            router, adapter = load_router(typical_routing=(mode == "agent_typical"))
        self.router, self.adapter = router, adapter
        self.amap = adapter.load_approach_map()
        self.mode, self.share, self.seed, self.prefix = mode, share, seed, prefix
        self._crc = zlib.crc32
        self.seen: set[str] = set()
        self.flag: set[str] = set()
        self.dest: dict[str, str] = {}           # guided car -> its destination road
        self.est: dict[str, tuple] = {}          # marked car -> (free-flow, typical) time of the route it set off on
        self.key: tuple | None = None
        self.stats = {"flagged": 0, "routed": 0, "changed_at_start": 0, "reroutes": 0, "set_failed": 0,
                      "plans": 0, "alert_changes": 0}

    def flagged(self, vid: str) -> bool:
        return vid.startswith(self.prefix) and (self._crc(f"{self.seed}:{vid}".encode()) / 2 ** 32) < self.share

    def alerts(self, now: float, forecast_msg=None, im=None) -> list[dict]:
        if not self.mode.startswith("agent"):
            return []
        return (forecast_alerts(forecast_msg, now, self.adapter, self.amap)
                + incident_alerts(im, self.router, self.adapter, now))

    def _route(self, conn, vid: str, alerts: list[dict], first: bool) -> None:
        r = self.router
        try:
            road = conn.vehicle.getRoadID(vid)
            route = list(conn.vehicle.getRoute(vid))
            idx = conn.vehicle.getRouteIndex(vid)
        except Exception:  # noqa: BLE001 - gone
            self.dest.pop(vid, None)
            return
        if road.startswith(":") or road not in r.roads:
            return
        dest = self.dest[vid]
        best = r.best(road, dest, alerts)
        self.stats["plans"] += 1
        if best is None:
            return
        cur = route[idx:] if idx < len(route) and route[idx] == road else None
        pen = r.penalties(alerts)
        if cur is not None and cur[-1] == dest and r.drivable(cur):
            c_now, c_best = r.route_cost(cur, pen), r.route_cost(best, pen)
            if first:
                if c_best >= c_now - 1e-6:
                    return
            elif c_now - c_best < max(SWITCH_S, SWITCH_SHARE * c_now):
                return
        try:
            conn.vehicle.setRoute(vid, best)
            self.stats["changed_at_start" if first else "reroutes"] += 1
        except Exception:  # noqa: BLE001
            self.stats["set_failed"] += 1

    def step(self, cur, conn, forecast_msg=None, im=None) -> None:
        for vid in cur.arrived:
            self.dest.pop(vid, None)
        alerts = self.alerts(cur.time, forecast_msg, im)
        new = [v for v in cur.vehicles if v not in self.seen]
        if new:
            alive = set(conn.vehicle.getIDList())
            new = [v for v in new if v in alive]
        for vid in new:
            self.seen.add(vid)
            if not self.flagged(vid) or getattr(cur.vehicles[vid], "vclass", "passenger") not in GUIDED_CLASSES:
                continue
            if im is not None and im.protected(vid):
                continue
            self.flag.add(vid)
            self.stats["flagged"] += 1
            try:
                route = conn.vehicle.getRoute(vid)
            except Exception:  # noqa: BLE001, S112
                continue
            if not route or route[-1] not in self.router.roads:
                continue
            if self.mode != "none":
                self.dest[vid] = route[-1]
                self.stats["routed"] += 1
                self._route(conn, vid, alerts, first=True)
                try:
                    route = conn.vehicle.getRoute(vid)
                except Exception:  # noqa: BLE001, S112
                    continue
            if all(e in self.router.roads for e in route):
                self.est[vid] = (round(self.router.free_s(route, skip_first=False), 1),
                                 round(self.router.typical_s(route, skip_first=False), 1))
        if self.mode.startswith("agent"):
            key = alerts_key(alerts)
            if key != self.key:
                self.key = key
                self.stats["alert_changes"] += 1
                for vid in list(self.dest):
                    if im is not None and im.protected(vid):
                        self.dest.pop(vid, None)
                        continue
                    self._route(conn, vid, alerts, first=False)

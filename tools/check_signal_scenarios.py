"""Stress scenarios for the signals at one Town05 junction, checked on what the simulation actually did.

    python tools/check_signal_scenarios.py                         # every scenario, actuated / agents / agents_observed
    python tools/check_signal_scenarios.py --scenarios heavy_ns,peds_each_crossing --controllers agents --seconds 900

Each scenario runs the town as tools/eval_town05_baseline.py does (scenario routes at a scale, the main project's
pedestrians and emergency vehicles, no teleporting) and adds, at the junction under test (--junction, 1126: four
approaches of two lanes, each with a shared right/straight and a shared straight/left lane, four 14 m crossings),
vehicles on chosen movements and people on chosen crossings. cosim/signal_monitor.SignalMonitor watches every step
(conflicting greens, intergreens, yellow, the pedestrians' clearance, people on a crossing when conflicting traffic is
released, people starting on don't walk, the CARLA poles) and SUMO reports collisions (--collision.check-junctions,
action warn, vehicles and pedestrians). Also: the longest any vehicle stood at a red and anyone waited at a crossing,
and how many injected vehicles and people got through.

Scenarios (veh/h per movement are added to the scenario's own traffic):
  heavy_ns            N and S approaches 450 veh/h each (70 % straight, 15 % right, 15 % left)
  heavy_ew            the same on E and W
  all_heavy           all four approaches 450 veh/h each
  left_turners        150 veh/h turning left from every approach against 300 veh/h straight on from the opposite one
  right_turn_peds     150 veh/h turning right from every approach, 150 people/h on every crossing
  peds_each_crossing  100 people/h on each of the four crossings, light traffic (x0.3)
  ped_transitions     people put at a crossing 2 s before and just after each change of its signal
  slow_ped_walk_end   people at 0.9 m/s (slower than the 1.07 m/s design speed) arriving 1 s before a walk ends
  low_traffic_peds    x0.2 traffic, 60 people/h on each crossing
  heavy_no_peds       x1.0 traffic, nobody walking
  ambulance           x0.8 traffic, an emergency vehicle about every 30 s
  uneven              W 700 veh/h, N 40 veh/h, one person every 4 min on the crossing over the N road

Writes out/signal_checks/<junction>/<scenario>_<controller>.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOWN05 = ROOT / "scenarios" / "town05"
OUT = ROOT / "out" / "signal_checks"
SCENARIOS = ("heavy_ns", "heavy_ew", "all_heavy", "left_turners", "right_turn_peds", "peds_each_crossing",
             "ped_transitions", "slow_ped_walk_end", "low_traffic_peds", "heavy_no_peds", "ambulance", "uneven")
# "town_x0.5", "town_x0.8", ...: nothing added, the town as tools/eval_town05_baseline.py runs it at that demand
BASE = {"peds_each_crossing": 0.3, "low_traffic_peds": 0.2, "heavy_no_peds": 1.0, "ambulance": 0.8}
SLOW_TYPE = "check_slow_walker"


def geometry(junction: str) -> dict:
    """Approaches (side -> incoming edge, {dir: outgoing edge}) and crossings (id -> (from edge, to edge) of a walk
    over it) of a signalised junction, from the network."""
    from cosim.signal_audit import load

    jn = load(TOWN05 / "town05.net.xml")[junction]
    root = ET.parse(TOWN05 / "town05.net.xml").getroot()
    edge_to = {e.get("id"): e.get("to") for e in root.iter("edge") if not e.get("function")}
    app: dict[str, dict] = {}
    for lk in jn.links:
        if lk.kind != "vehicle":
            continue
        a = app.setdefault(lk.approach, {"in": lk.from_lane.rsplit("_", 1)[0], "out": {}})
        a["out"].setdefault(lk.dir, lk.to_lane.rsplit("_", 1)[0])
    cross = {}
    for lk in jn.links:
        if lk.kind != "crossing":
            continue
        e1, e2 = lk.crosses
        inc, outg = (e1, e2) if edge_to.get(e1) == junction else (e2, e1)
        cross[lk.crossing] = (inc, outg, lk.index, lk.length)
    return {"approaches": app, "crossings": cross}


def flows(scenario: str, geo: dict) -> tuple[list[tuple[list[str], float]], list[tuple[str, float]]]:
    """([(route edges, veh/h)], [(crossing id, people/h)])."""
    app, cross = geo["approaches"], geo["crossings"]
    veh: list[tuple[list[str], float]] = []
    peds: list[tuple[str, float]] = []

    def approach(side: str, rate: float, split=(("s", 0.7), ("r", 0.15), ("l", 0.15))):
        a = app.get(side)
        if a is None:
            return
        for d, share in split:
            if d in a["out"]:
                veh.append(([a["in"], a["out"][d]], rate * share))

    if scenario == "heavy_ns":
        approach("N", 450)
        approach("S", 450)
    elif scenario == "heavy_ew":
        approach("E", 450)
        approach("W", 450)
    elif scenario == "all_heavy":
        for s in "NESW":
            approach(s, 450)
    elif scenario == "left_turners":
        opp = {"N": "S", "S": "N", "E": "W", "W": "E"}
        for s in "NESW":
            approach(s, 150, (("l", 1.0),))
            approach(opp[s], 300, (("s", 1.0),))
    elif scenario == "right_turn_peds":
        for s in "NESW":
            approach(s, 150, (("r", 1.0),))
        peds = [(c, 150.0) for c in cross]
    elif scenario == "peds_each_crossing":
        peds = [(c, 100.0) for c in cross]
    elif scenario == "low_traffic_peds":
        peds = [(c, 60.0) for c in cross]
    elif scenario == "uneven":
        approach("W", 700)
        approach("N", 40)
        n_in = app.get("N", {}).get("in")
        peds = [(c, 15.0) for c, v in cross.items() if n_in in v[:2]]
    return veh, peds


def run(scenario: str, controller: str, junction: str, seed: int, seconds: float, port: int, out_dir: Path) -> dict:
    from cosim.agents_actuated import make as make_agents
    from cosim.controllers import _Conn
    from cosim.signal_monitor import SignalMonitor
    from cosim.sumo_side import SumoSide

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{scenario}_{controller}"
    coll = out_dir / f"{name}.collisions.xml"
    scale = float(scenario.split("_x")[1]) if scenario.startswith("town_x") else BASE.get(scenario, 0.5)
    policy = make_agents(controller) if controller.startswith("agents") else None
    s = SumoSide(TOWN05, controller if policy is None else "agents", seed=seed, port=port, label=f"chk-{name}",
                 policy=policy, extras=True, emergency_every=30.0 if scenario == "ambulance" else 120.0,
                 extra_args=["--scale", f"{scale:g}", "--end", f"{seconds:g}",
                             "--collision.check-junctions", "true", "--collision.action", "warn",
                             "--intermodal-collision.action", "warn", "--collision-output", str(coll)])
    ex = s.extras
    ex.config.pedestrian_trips = True
    if scenario == "heavy_no_peds":
        ex.config.pedestrians_per_km = 0.0
    if policy is not None:
        policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    cams = None
    if policy is not None and policy.uses_cameras:
        from cosim.camera_reports import Layout, SimulatedCameras

        cdir = ROOT / "out" / "town05_cameras"
        parked = json.loads((cdir / "parked_static.json").read_text(encoding="utf-8")) \
            if (cdir / "parked_static.json").is_file() else None
        cams = SimulatedCameras(Layout.load(cdir), seed=seed, parked=parked)
        policy.visible = cams.layout.visible
    c = s.conn
    geo = geometry(junction)
    veh, peds = flows(scenario, geo)
    rng = random.Random(seed * 31 + 7)
    for k, (edges, _) in enumerate(veh):
        c.route.add(f"chk_r{k}", edges)
    c.vehicletype.copy("DEFAULT_PEDTYPE", SLOW_TYPE)
    c.vehicletype.setMaxSpeed(SLOW_TYPE, 0.9)
    mon = SignalMonitor(TOWN05 / "town05.net.xml")
    dt = s.step_length
    added_v, added_p = [], []
    n_v = n_p = 0
    prev_state = ""
    t0 = time.time()

    def add_person(crossing: str, slow: bool = False) -> None:
        nonlocal n_p
        inc, outg, _, _ = geo["crossings"][crossing]
        n_p += 1
        pid = f"chkp{n_p}"
        ln = c.lane.getLength(f"{inc}_0")
        try:
            c.person.add(pid, inc, max(0.0, ln - 3.0), typeID=SLOW_TYPE if slow else "twin_pedestrian")
            c.person.appendWalkingStage(pid, [inc, outg], arrivalPos=3.0)
            added_p.append(pid)
        except Exception:  # noqa: BLE001 - no room / no route
            pass

    try:
        while s.time < seconds:
            if cams is None:
                r = s.step()
            else:
                r = s.step(run_policy=False)
                reps = cams.step(s.conn, s.time, {v: rec.on_run for v, rec in ex.emergency.items()})
                if reps is not None:
                    policy.observe(reps)
                policy.step(_Conn(s.conn))
                r.tls.update({t: c.trafficlight.getRedYellowGreenState(t) for t in r.tls})
            now = r.time
            for k, (edges, rate) in enumerate(veh):
                if rng.random() < rate * dt / 3600.0:
                    n_v += 1
                    vid = f"chkv{n_v}"
                    c.vehicle.add(vid, f"chk_r{k}", typeID="pas_passenger", depart="now", departLane="best",
                                  departSpeed="max")
                    added_v.append(vid)
            for cr, rate in peds:
                if rng.random() < rate * dt / 3600.0:
                    add_person(cr)
            st = r.tls.get(junction, "")
            if scenario in ("ped_transitions", "slow_ped_walk_end"):
                rem = c.trafficlight.getNextSwitch(junction) - now
                for cr, (_, _, idx, _) in geo["crossings"].items():
                    walking = idx < len(st) and st[idx] in "Gg"
                    if scenario == "ped_transitions":
                        if abs(rem - 2.0) < dt / 2 or (prev_state and prev_state[idx] != st[idx]):
                            add_person(cr)
                    elif walking and abs(rem - 1.0) < dt / 2:
                        add_person(cr, slow=True)
            prev_state = st
            mon.step(c, now, r.tls)
            if int(now / dt) % 10 == 0:
                mon.vehicle_waits(c, {junction: st})
        arrived_v = sum(1 for v in added_v if v not in set(c.vehicle.getIDList()))
        live_p = set(c.person.getIDList())
        done_p = sum(1 for p in added_p if p not in live_p)
        pending = len(c.simulation.getPendingVehicles())
    finally:
        s.close()
    collisions = {"vehicle": 0, "pedestrian": 0}
    coll_detail = []
    if coll.is_file():
        for e in ET.parse(coll).getroot().iter("collision"):
            ped = e.get("type") in ("walkingarea", "crossing", "sharedLane") or "ped" in (
                e.get("victimType", "") + e.get("colliderType", "")).lower()
            collisions["pedestrian" if ped else "vehicle"] += 1
            if len(coll_detail) < 20:
                coll_detail.append(dict(e.attrib))
        coll.unlink()
    rep = mon.report()
    jw = {k: v for k, v in mon.max_wait.items() if k.startswith(f"{junction}:") or k.startswith(f":{junction}_")}
    out = {"scenario": scenario, "controller": controller, "junction": junction, "seed": seed, "seconds": seconds,
           "scale": scale, "injected_vehicles": n_v, "injected_arrived": arrived_v, "injected_people": len(added_p),
           "injected_people_done": done_p, "pending_insertion": pending, "collisions": collisions, "collision_detail": coll_detail,
           "violations": rep["violations"], "examples": rep["examples"],
           "town_max_vehicle_wait_s": rep["max_vehicle_wait_s"], "town_max_ped_wait_s": rep["max_ped_wait_s"],
           "junction_max_wait_s": {k: round(v, 1) for k, v in sorted(jw.items())},
           "wall_s": round(time.time() - t0)}
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def summary(out_dir: Path) -> str:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    kinds = ["conflicting_greens", "short_intergreen", "green_to_red", "ped_short_clearance",
             "ped_on_crossing_at_release", "ped_started_on_red", "carla_pole_disagree"]
    md = [f"# Signal checks at junction {rows[0]['junction'] if rows else '?'}", "",
          "Counts over the whole town (every signal), from what the simulation showed each step; collisions from SUMO "
          "(junctions checked, vehicles and pedestrians). Longest waits: at the junction under test.", "",
          "| scenario | controller | " + " | ".join(kinds) + " | collisions veh/ped | injected veh arrived | "
          "injected people done | longest vehicle wait (s) | longest pedestrian wait (s) |",
          "|---|---|" + "---|" * len(kinds) + "---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (SCENARIOS.index(r["scenario"]) if r["scenario"] in SCENARIOS else 99,
                                         r["controller"])):
        jw = r["junction_max_wait_s"]
        vw = max((v for k, v in jw.items() if not k.startswith(":")), default=0.0)
        pw = max((v for k, v in jw.items() if k.startswith(":")), default=0.0)
        md.append(f"| {r['scenario']} | {r['controller']} | " + " | ".join(str(r["violations"].get(k, 0)) for k in kinds)
                  + f" | {r['collisions']['vehicle']}/{r['collisions']['pedestrian']} | "
                  f"{r['injected_arrived']}/{r['injected_vehicles']} | {r['injected_people_done']}/{r['injected_people']} | "
                  f"{vw:g} | {pw:g} |")
    text = "\n".join(md) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--controllers", default="actuated,agents,agents_observed")
    ap.add_argument("--junction", default="1126")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else OUT / a.junction
    todo = [(sc, ct) for sc in a.scenarios.split(",") for ct in a.controllers.split(",")]
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        futs = {pool.submit(run, sc, ct, a.junction, a.seed, a.seconds, a.port + i, out_dir): (sc, ct)
                for i, (sc, ct) in enumerate(todo)}
        for f in as_completed(futs):
            sc, ct = futs[f]
            try:
                r = f.result()
                print(f"{sc} {ct}: violations {r['violations']} collisions {r['collisions']} wall {r['wall_s']} s",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{sc} {ct}: FAILED {e!r}", flush=True)
    print(summary(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Step 1 of the multi-agent plan: what every signal controller does on Town05 today (SUMO alone).

    python tools/eval_town05_baseline.py                      # every controller, both demands, seeds 1-3
    python tools/eval_town05_baseline.py --controllers actuated,max_pressure_walk --scales 0.6 --seeds 1
    python tools/eval_town05_baseline.py --aggregate-only     # only rebuild the tables from the runs written

Every run has the same Town05 network (signalised crosswalks, tools/build_town05_peds.py), the
scenario's demand scaled by --scale, the main project's pedestrians (routed trips over the
crosswalks) and emergency vehicles (one every ~--emergency-every s, 60 % on a run), and no
teleporting. Within a seed, every controller meets the same vehicles and the same emergency
vehicles (cosim/extras.py draws them from their own generator).

Controllers:
  default                     the network's own fixed-time programs (walk phase skipped when nobody waits)
  webster, actuated           programs from scenarios/town05/town05.<name>.add.xml (same walk skip)
  max_pressure_walk           max-pressure on exact counts, pedestrians waiting count in the walk phase's
                              pressure (cosim/controllers.WalkAware)
  max_pressure_walk_priority  the same, plus pre-emption for emergency vehicles on a run (within 150 m)
  max_pressure_walk_observed  max_pressure_walk on the cameras' reports instead of exact counts
                              (cosim/camera_reports.SimulatedCameras: visible lane stretches only, the
                              detector's measured misses and false boxes; needs out/town05_cameras from
                              tools/town05_cameras.py plan). Not in the default list.

Metrics (SUMO tripinfo, unfinished trips included, so a gridlock cannot hide):
  vehicles     trips, completed, throughput per hour, mean waiting time, mean time loss
  emergency    vehicles on a run: mean time loss, waiting, trip time
  pedestrians  mean time loss per walk (time lost to waiting at crossings and crowding)
  gridlock     vehicles standing still for over 300 s at the end, vehicles that never got in

Writes out/baseline_town05/runs/<controller>_x<scale>_seed<seed>.json, then
out/baseline_town05/baseline.json and baseline.md (mean and spread over the seeds).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "baseline_town05"
RUNS = OUT / "runs"
CONTROLLERS = "default,webster,actuated,max_pressure_walk,max_pressure_walk_priority"
STUCK_S = 300.0


def tag(controller: str, scale: float, seed: int) -> str:
    return f"{controller}_x{scale:g}_seed{seed}"


CAMERAS = ROOT / "out" / "town05_cameras"


def run(controller: str, scale: float, seed: int, end: float, every: float, peds_km: float, port: int,
        runs: Path = RUNS, probe=None, blocker_s: float | None = None) -> dict:
    """probe: called after every step with the SumoSide (tools/diag_town05_agents.py watches a run with it)."""
    from cosim.controllers import ObservedMaxPressure, PriorityMaxPressure, _Conn
    from cosim.sumo_side import SumoSide

    runs.mkdir(parents=True, exist_ok=True)
    name = tag(controller, scale, seed)
    trip = runs / f"{name}.tripinfo.xml"
    s = SumoSide(ROOT / "scenarios" / "town05", controller, seed=seed, port=port, label=name, extras=True,
                 emergency_every=every,
                 extra_args=["--scale", f"{scale:g}", "--end", f"{end:g}",
                             "--tripinfo-output", str(trip), "--tripinfo-output.write-unfinished", "true"]
                 + (["--ignore-junction-blocker", f"{blocker_s:g}"] if blocker_s is not None else []))
    ex = s.extras
    ex.config.pedestrians_per_km = peds_km
    ex.config.pedestrian_trips = True
    if isinstance(s.policy, PriorityMaxPressure) or hasattr(s.policy, "emergency"):
        s.policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    cams = sample = None
    if isinstance(s.policy, ObservedMaxPressure) or hasattr(s.policy, "observed") or getattr(s.policy, "uses_cameras", False):
        from cosim.camera_reports import Layout, SimulatedCameras, observed

        parked_file = CAMERAS / "parked_static.json"
        parked = json.loads(parked_file.read_text(encoding="utf-8")) if parked_file.is_file() else None
        cams = SimulatedCameras(Layout.load(CAMERAS), seed=seed, parked=parked)
        if hasattr(s.policy, "visible"):  # the step 3 agents' storage is the stretch the cameras see
            s.policy.visible = cams.layout.visible
        sample = (runs / f"{name}.reports.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    try:
        r = None
        while s.time < end:
            if cams is None:
                r = s.step()
                if probe is not None:
                    probe(s)
                continue
            r = s.step(run_policy=False)
            reps = cams.step(s.conn, s.time, {v: rec.on_run for v, rec in ex.emergency.items()})
            if reps is not None:
                if hasattr(s.policy, "observe"):  # the step 3 agents read vehicles and queued per lane
                    s.policy.observe(reps)
                else:
                    s.policy.observed = observed(reps)
                if s.time <= 600.0:  # the first 10 minutes of reports, to read and to feed the report agent
                    sample.write(json.dumps({"t": s.time, "reports": [x for x in reps if x["lanes"] or x["ems"]]}) + "\n")
            s.policy.step(_Conn(s.conn))
            if probe is not None:
                probe(s)
        c = s.conn
        stuck = sum(1 for v in (r.vehicles if r else {}) if c.vehicle.getWaitingTime(v) > STUCK_S)
        pending = len(c.simulation.getPendingVehicles())
        on_run = {v for v, rec in ex.emergency.items() if rec.on_run}
        emergency = set(ex.emergency)
        walk = [s.walk.served, s.walk.skipped] if s.walk is not None and s.policy is None else None
        policy = s.policy
    finally:
        s.close()
        if sample is not None:
            sample.close()
    root = ET.parse(trip).getroot()
    trips = root.findall("tripinfo")
    walks = [w for p in root.findall("personinfo") for w in p.findall("walk")]

    def veh(sel) -> dict | None:
        if not sel:
            return None
        done = [t for t in sel if float(t.get("arrival", -1)) >= 0]
        return {"n": len(sel), "completed": len(done),
                "mean_waiting_s": round(st.mean(float(t.get("waitingTime")) for t in sel), 2),
                "mean_time_loss_s": round(st.mean(float(t.get("timeLoss")) for t in sel), 2),
                "mean_trip_s": round(st.mean(float(t.get("duration")) for t in sel), 2)}

    others = [t for t in trips if t.get("id") not in emergency]
    out = {"controller": controller, "scale": scale, "seed": seed, "simulated_s": end,
           "emergency_every_s": every, "pedestrians_per_km": peds_km,
           "vehicles": veh(others),
           "throughput_per_h": round(sum(float(t.get("arrival", -1)) >= 0 for t in others) * 3600.0 / end, 1),
           "emergency_on_run": veh([t for t in trips if t.get("id") in on_run]),
           "pedestrians": ({"walks": len(walks),
                            "completed": sum(float(w.get("arrival", -1)) >= 0 for w in walks),
                            "mean_time_loss_s": round(st.mean(float(w.get("timeLoss", 0)) for w in walks), 2),
                            "mean_walk_s": round(st.mean(float(w.get("duration", 0)) for w in walks), 2)}
                           if walks else None),
           "stuck_over_300s_at_end": stuck, "never_entered": pending,
           "walk_phases_served_skipped": walk, "wall_s": round(time.time() - t0),
           "ignore_junction_blocker_s": blocker_s}
    if policy is not None and hasattr(policy, "junctions"):
        out["switches"] = sum(j.switches for j in policy.junctions.values())
    if isinstance(policy, PriorityMaxPressure):
        out.update(preemptions=policy.preemptions, held_steps=policy.held, dropped=policy.dropped)
    if hasattr(policy, "handed_back"):
        out.update(handed_back=policy.handed_back, corridor_requests=policy.corridor_requests)
    if hasattr(policy, "stats"):  # step 3, version 2 (cosim/agents_actuated.py)
        out.update(agent_stats=policy.stats, features_on=sorted(policy.on), fallback=sorted(policy.fallback))
    (runs / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    trip.unlink(missing_ok=True)  # 5-20 MB each; the summary above is what is kept
    return out


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), 0.0)
    return (round(st.mean(xs), 1), round(st.stdev(xs), 1) if len(xs) > 1 else 0.0)


def aggregate(controllers: list[str], scales: list[float], seeds: list[int], out_dir: Path = OUT,
              title: str = "Town05 baseline (SUMO alone)") -> dict:
    rows = []
    for scale in scales:
        for ctl in controllers:
            rs = [json.loads(p.read_text(encoding="utf-8")) for seed in seeds
                  if (p := out_dir / "runs" / f"{tag(ctl, scale, seed)}.json").is_file()]
            if not rs:
                continue

            def g(path, rs=rs):
                vals = []
                for r in rs:
                    v = r
                    for k in path:
                        v = v.get(k) if isinstance(v, dict) else None
                    vals.append(v)
                return _mean_sd(vals)

            rows.append({"scale": scale, "controller": ctl, "seeds": sorted(r["seed"] for r in rs),
                         "veh_waiting_s": g(["vehicles", "mean_waiting_s"]),
                         "veh_time_loss_s": g(["vehicles", "mean_time_loss_s"]),
                         "throughput_per_h": g(["throughput_per_h"]),
                         "ems_time_loss_s": g(["emergency_on_run", "mean_time_loss_s"]),
                         "ems_trip_s": g(["emergency_on_run", "mean_trip_s"]),
                         "ped_time_loss_s": g(["pedestrians", "mean_time_loss_s"]),
                         "stuck": g(["stuck_over_300s_at_end"]),
                         "never_entered": g(["never_entered"])})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def f(ms):
        m, sd = ms
        return "n/a" if m != m else (f"{m:g} ± {sd:g}" if sd else f"{m:g}")

    md = [f"# {title}", "",
          "Mean ± standard deviation over the seeds. Waiting and time loss are per trip (seconds), unfinished "
          "trips included. Stuck: vehicles standing still > 300 s at the end (gridlock). Never entered: vehicles "
          "that could not be inserted.", ""]
    for scale in scales:
        sel = [r for r in rows if r["scale"] == scale]
        if not sel:
            continue
        md += [f"## Demand x{scale:g} (scenario routes, SUMO --scale)", "",
               "| controller | seeds | vehicles: waiting | vehicles: time loss | throughput /h | ambulance on a run: "
               "time loss | ambulance: trip | pedestrians: time loss | stuck | never entered |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sel:
            md.append(f"| {r['controller']} | {','.join(map(str, r['seeds']))} | {f(r['veh_waiting_s'])} | "
                      f"{f(r['veh_time_loss_s'])} | {f(r['throughput_per_h'])} | {f(r['ems_time_loss_s'])} | "
                      f"{f(r['ems_trip_s'])} | {f(r['ped_time_loss_s'])} | {f(r['stuck'])} | {f(r['never_entered'])} |")
        md.append("")
    (out_dir / "baseline.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--controllers", default=CONTROLLERS)
    ap.add_argument("--scales", default="0.5,0.8",
                    help="demand levels, as SUMO --scale on the scenario's routes (1.0 = the scenario's design "
                         "demand, ~3000 vehicles/h)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--end", type=float, default=3600.0, help="simulated seconds per run")
    ap.add_argument("--emergency-every", type=float, default=120.0)
    ap.add_argument("--pedestrians-per-km", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=2, help="runs at the same time (one SUMO each)")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-done", action="store_true", help="keep runs already written")
    ap.add_argument("--out", default=str(OUT), help="output folder (runs/ and the tables go there)")
    ap.add_argument("--title", default="Town05 baseline (SUMO alone)")
    ap.add_argument("--open-test", default=None, metavar="REASON",
                    help="these are held-out test seeds: open them once (OUT/TEST_OPENED.lock; refused if it exists)")
    ap.add_argument("--ignore-junction-blocker", type=float, default=None, metavar="S",
                    help="SUMO's deadlock breaker: a vehicle standing in a junction for S seconds is no longer waited for "
                         "by the others (no vehicle is removed). Off by default, as in every run before 24 Oct")
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    runs = out_dir / "runs"
    controllers = a.controllers.split(",")
    scales = [float(x) for x in a.scales.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.open_test and not a.aggregate_only:
        lock = out_dir / "TEST_OPENED.lock"
        if lock.is_file():
            print(f"test seeds already opened here: {lock.read_text(encoding='utf-8')}")
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"seeds": seeds, "reason": a.open_test, "when": time.strftime("%Y-%m-%d %H:%M"),
                                    "args": vars(a)}), encoding="utf-8")
    if not a.aggregate_only:
        todo = [(c, sc, sd) for sc in scales for sd in seeds for c in controllers
                if not (a.skip_done and (runs / f"{tag(c, sc, sd)}.json").is_file())]
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
            futs = {pool.submit(run, c, sc, sd, a.end, a.emergency_every, a.pedestrians_per_km, a.port + i, runs,
                               None, a.ignore_junction_blocker): (c, sc, sd)
                    for i, (c, sc, sd) in enumerate(todo)}
            for fut in as_completed(futs):
                c, sc, sd = futs[fut]
                try:
                    r = fut.result()
                    v = r["vehicles"] or {}
                    print(f"{tag(c, sc, sd)}: waiting {v.get('mean_waiting_s')} s, time loss {v.get('mean_time_loss_s')} s, "
                          f"stuck {r['stuck_over_300s_at_end']}, wall {r['wall_s']} s", flush=True)
                except Exception as e:  # noqa: BLE001 - one failed run must not lose the others
                    print(f"{tag(c, sc, sd)}: FAILED {e!r}", flush=True)
    # the tables cover every run written so far, not only this invocation's
    done = {(r.stem.split("_x")[0], float(r.stem.split("_x")[1].split("_seed")[0]), int(r.stem.split("_seed")[1]))
            for r in runs.glob("*_seed*.json")}
    order = CONTROLLERS.split(",") + ["max_pressure_walk_observed", "agents_v1", "agents_v1_observed", "agents",
                                      "agents_observed"]
    ctls = sorted({c for c, _, _ in done}, key=lambda c: (order.index(c) if c in order else 99, c))
    aggregate(ctls, sorted({sc for _, sc, _ in done}), sorted({sd for _, _, sd in done}), out_dir, a.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())

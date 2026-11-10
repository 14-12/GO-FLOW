"""Town05 datasets for agents/goflow_prediction and agents/goflow_traffic (plan, section 6), SUMO alone, no CARLA.

    .venv\\Scripts\\python tools\\goflow_prediction_dataset.py --seeds 101-104 --minutes 30 --out out\\goflow_prediction\\datasets\\D2_try

The viewer's traffic as tools/check_accidents.py builds it (the day's curve held by cosim/live.Population, the bus
lines, the ring road's cars), the viewer's SUMO options, and the clock MOVING: town time = start hour + SUMO seconds x
--clock-speed (the viewer's, cosim/daily.GAME_SPEED: 30 since 7 Nov). Signals: --controller (default coordinated, the viewer's). Accidents off unless
--accidents. Demand levels: --scale multiplies the day's curve (the viewer has no scale; decision of the plan).

Every second each approach (tools/goflow_prediction_graph.py: one per camera) is sampled through TraCI lane
subscriptions: vehicles, halting vehicles, mean speed (m/s), occupancy (fraction, as SUMO gives it), and the share of its signal links
that are green. Each --dt (30 s) window is averaged into one row. The first --warmup minutes are dropped.

Output, a NEW folder: runs/seed<seed>.npz (one run each), dataset.npz (all runs, goflow_prediction.data format),
nodes.json and distance.csv (copied from the graph), runs.json (per run: wall time, vehicles, gridlock flag).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIEWER_SUMO_ARGS = ["--lateral-resolution", "0.8", "--collision.action", "warn", "--collision.check-junctions", "true",
                    "--collision.mingap-factor", "0"]
HORIZONS_S = (60, 120, 300)
STUCK_S = 300.0


def seeds_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def one_run(seed: int, a: dict, nodes: list[dict], out_dir: str, port: int, probe=None, hook=None) -> dict:
    """probe(cur, conn, seconds_of_day): called after every step (tools/goflow_prediction_live_check.py
    plugs the live connector in through it, on exactly this traffic)."""
    from cosim import daily, vmodels
    from cosim import scenario as scenario_mod
    from cosim.blueprints import Candidate
    from cosim.freefly import BusLines
    from cosim.lanegeo import LaneGeometry
    from cosim.live import Population
    from cosim.sumo_side import SumoSide

    t_wall = time.time()
    scen = scenario_mod.load("town05", ROOT)
    pool = daily.load_pool(scen.folder)
    models = vmodels.load()
    pool.vtypes = {t: vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS) for t, vt in pool.vtypes.items()}
    work = Path(out_dir) / "work"
    work.mkdir(parents=True, exist_ok=True)
    pf = work / f"routes_{seed}.rou.xml"
    rids = daily.write_pool_routes(pool, pf)
    et = vmodels.extras_types_file(models, work / f"extras_types_{seed}.add.xml", vmodels.LANE_CHANGE_ATTRS)
    curve = [v * a["scale"] for v in daily.population_curve(None, pool, mean_trip_min=140 / 60.0)]
    extra = ["--route-files", str(pf), "--end", "100000000"] + VIEWER_SUMO_ARGS + list(a.get("sumo_extra", []))
    if a["junction_blocker"]:
        extra += ["--ignore-junction-blocker", str(a["junction_blocker"])]
    purpose = a.get("trips", "random") == "purpose"   # (8 Nov) trips between the town's places (cosim/purposes.py)
    from cosim.purposes import ADD as PLACES_ADD

    sumo = SumoSide(scen.folder, a["controller"], seed=seed, port=port, label=f"gfp{port}", extras=True,
                    additional=(BusLines.additional(scen.folder, "town05") if a["buses"] else [])
                    + ([str(PLACES_ADD)] if purpose else []),
                    emergency_every=45, extras_types=et, lane_geometry=LaneGeometry(scen.net_file), extra_args=extra)
    c = sumo.conn
    if purpose:
        from cosim.purposes import PurposeDemand, purpose_population

        pop = purpose_population(Population)(c, pool, rids, curve, seed=seed, mean_trip_s=140.0)
        pop.demand = PurposeDemand(c, a["clock_speed"], seed=seed)
        pop.sources = tuple(sorted({r.split()[0] for routes in pool.routes.values() for r in routes}))
        pop.sinks = tuple(sorted({r.split()[-1] for routes in pool.routes.values() for r in routes}))
    else:
        pop = Population(c, pool, rids, curve, seed=seed, mean_trip_s=140.0)
    cands = [Candidate(m, i["base_type"].lower(), i["special_type"].lower(), int(i["generation"] or 1), ("0",),
                       i["has_lights"]) for m, i in models.items()]
    base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
    base.update({t: vc for t, vc in vmodels.EMERGENCY_TYPES.items()})
    typer = vmodels.Typer(c, cands, models, base)
    # The incident manager always runs, as in the viewer: it is also what undoes knots of cars out of sight
    # (IncidentManager.unjam). Without --accidents it makes no crash, breakdown, pile-up or look-alike.
    from cosim.incidents import IncidentManager

    sinks = sorted({r.split()[-1] for routes in pool.routes.values() for r in routes})
    kw = {}
    if not a["accidents"]:
        kw = {"share": 0.0, "breakdown_mean_s": 0.0, "speeder_share": 0.0, "pileups": False, "lookalikes": False}
    elif a["careless_share"] is not None:
        kw = {"share": a["careless_share"]}
    im = IncidentManager(c, scen.net_file, typer, seed=seed, walk=sumo.walk, sinks=sinks, **kw)
    pop.type_for = im.type_for
    if purpose:
        im.parked = pop.demand.parked       # (8 Nov) a car parked at a place is not a knot to undo
    # the junction agents connected as the viewer connects them: the cameras' reports and the emergency requests
    # (cosim/agents_actuated.wire; before 8 Nov agents_observed ran here on no counts and no requests)
    from cosim.agents_actuated import wire

    wired = wire(sumo.policy, sumo, seed, lambda: [v for v in list(im.role) if im.on_run(v)])
    if hook is not None:  # tools/eval_agents_forecast.py: the forecasters in the loop, its own measurements
        hook(sumo, im)
    bus = ring = None
    if a["buses"]:
        bus_types = [t for t, vt in pool.vtypes.items() if 'vClass="bus"' in vt]
        bus = BusLines(c, scen.folder, "town05", bus_types[0], type_for=typer.type_for)
        for i, t in enumerate(pop.types):
            if t in bus_types:
                pop.weights[i] = 0
    if a["ring"]:
        import sumolib

        from cosim.highway import HighwayTraffic, find_loops

        net = sumolib.net.readNet(str(scen.net_file))
        loops = find_loops(net)
        top = max(curve) or 1.0
        ring = HighwayTraffic(c, loops, lambda s_: daily.target_at(curve, s_) / top,
                              types=[(t, n) for t, n in pool.per_hour.items() if base.get(t) != "bus"],
                              type_for=pop.type_for, seed=seed + 3,
                              lengths={e: net.getEdge(e).getLength() for lp in loops for e in lp}, policy=sumo.policy)
        ring.protected = im.protected
        ring.setup()

    # what is measured: the live connector's own Aggregator (cosim/goflow_prediction.py), so the agents are
    # trained on exactly what the town will send them
    from cosim.goflow_prediction import Aggregator

    agg = Aggregator(c, nodes, a["dt"])

    N = len(nodes)
    sod0 = a["start_hour"] * 3600.0
    t_start = sumo.time

    def sod_at(t):
        return (sod0 + (t - t_start) * a["clock_speed"]) % 86400.0

    pop.add(int(pop.target(sod_at(t_start))), anywhere=True)
    if purpose and a.get("prefill", True):   # the car parks and homes as a measured day has them at this hour
        pop.prefill(sod_at(t_start) / 3600.0)
    end = t_start + 60.0 * (a["minutes"] + a["warmup"])
    keep_from = t_start + 60.0 * a["warmup"]
    rows = {k: [] for k in ("t", "tod", "dem", "dem_at", "halt", "veh", "speed", "occ", "green")}
    k = 0
    cur = None
    while sumo.time < end:
        cur = sumo.step()
        k += 1
        sod = sod_at(cur.time)
        if purpose:
            pop.demand.step(cur.arrived)
        pop.step(cur.vehicles, sod, lambda vid, v: not im.protected(vid) and not vid.startswith("pt_"))
        if ring is not None and k % 4 == 0:
            ring.step(cur.vehicles, sod, set(c.simulation.getPendingVehicles()))
        if bus is not None and k % 2 == 0:
            bus.step(cur.time, sod / 3600.0)
            for vid in [v for v in cur.vehicles if v.startswith("pt_")]:
                try:
                    route, idx = c.vehicle.getRoute(vid), c.vehicle.getRouteIndex(vid)
                except Exception:  # noqa: BLE001, S112 - a bus that just left
                    continue
                if idx >= len(route) - 3:
                    bus.next_lap(vid, idx, route)
        im.step(cur.time, cur.vehicles, cur.tls)
        if int(cur.time * 2) % 10 == 0:  # every 5 s, as the viewer
            im.unjam(cur.time, cur.vehicles)
        if probe is not None:
            probe(cur, c, sod)
        if cur.time < keep_from:
            continue
        row = agg.sample(c, cur.time, cur.tls)
        if row is None:
            continue
        rows["t"].append(row["t"])
        rows["tod"].append(sod)
        rows["dem"].append(daily.demand_factor(sod / 3600.0) * a["scale"])
        rows["dem_at"].append([daily.demand_factor(((sod + h * a["clock_speed"]) % 86400) / 3600.0) * a["scale"]
                               for h in HORIZONS_S])
        for name in ("halt", "veh", "speed", "occ", "green"):
            rows[name].append(np.asarray(row[name], dtype=np.float64))

    def stuck_one(v):  # waiting at a red or in a jam, not at a planned stop (deliveries, bus stops)
        try:
            return (not v.startswith("pt_") and c.vehicle.getWaitingTime(v) > STUCK_S
                    and not c.vehicle.isStopped(v))
        except Exception:  # noqa: BLE001 - already gone
            return False

    stuck_ids = [v for v in (cur.vehicles if cur else {}) if stuck_one(v)]
    stuck = len(stuck_ids)
    where = {}
    for v in stuck_ids:
        try:
            ln = c.vehicle.getLaneID(v)
            where[ln] = where.get(ln, 0) + 1
        except Exception:  # noqa: BLE001, S110 - already gone
            pass
    vehicles_end = len(cur.vehicles) if cur else 0
    sumo.close()
    T = len(rows["t"])
    arr = {k_: np.asarray(v, dtype=np.float32) for k_, v in rows.items()}
    np.savez_compressed(Path(out_dir) / "runs" / f"seed{seed}.npz", t=arr["t"], tod_s=arr["tod"], demand=arr["dem"],
                        demand_at=arr["dem_at"].reshape(T, len(HORIZONS_S)),
                        **{n_: arr[n_].reshape(T, N) for n_ in ("halt", "veh", "speed", "occ", "green")})
    return {"seed": seed, "steps": T, "wall_s": round(time.time() - t_wall), "vehicles_end": vehicles_end,
                "stuck_over_300s": stuck, "stuck_lanes": dict(sorted(where.items(), key=lambda kv: -kv[1])[:6]),
                "gridlock": stuck > 50, "incidents": len(im.done) + len(im.incidents), "agents_wired": wired,
                # (9 Nov) ordinary cars taken off the road during the run, by why: a result that needed it says so
                "removed": {**im.removals(), "population_excess": pop.removed, "trips_not_started": pop.not_added,
                            "ring_excess": ring.removed if ring is not None else 0},
                **({"agent_stats": dict(sumo.policy.stats)} if wired else {}),
                **({"purposes": dict(pop.demand.stats, parked_end=len(pop.demand.parked),
                                     occupancy_end=pop.demand.occupancy())} if purpose else {})}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT,
                              timeout=10, check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--minutes", type=float, default=30.0, help="SUMO minutes kept per run (after the warm-up)")
    ap.add_argument("--warmup", type=float, default=5.0, help="SUMO minutes dropped at the start")
    ap.add_argument("--start-hour", type=float, default=6.0)
    from cosim.daily import GAME_SPEED

    ap.add_argument("--clock-speed", type=float, default=GAME_SPEED)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--controller", default="coordinated")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the day's vehicle curve")
    ap.add_argument("--accidents", action="store_true")
    ap.add_argument("--careless-share", type=float, default=None)
    ap.add_argument("--no-buses", dest="buses", action="store_false")
    ap.add_argument("--no-ring", dest="ring", action="store_false")
    ap.add_argument("--trips", choices=("purpose", "random"), default="purpose",
                    help="(10 Nov) trips between the town's places, as the viewer (cosim/purposes.py; before: random)")
    ap.add_argument("--junction-blocker", type=float, default=60.0,
                    help="SUMO --ignore-junction-blocker (60 as agents_final_jb; 0 = off)")
    ap.add_argument("--graph", default=str(ROOT / "out" / "goflow_prediction" / "graph"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--port", type=int, default=8930)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="(10 Nov) the folder exists: keep the runs its runs.json lists as finished, run the seeds missing, "
                         "make dataset.npz again (nothing to do when every seed is there)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.resume:
        print(f"{out} exists: results are never overwritten")
        return 1
    g = Path(args.graph)
    nodes = json.loads((g / "nodes.json").read_text(encoding="utf-8"))
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for f in ("nodes.json", "distance.csv"):
        if not (out / f).exists():
            shutil.copy(g / f, out / f)
    a = vars(args).copy()
    seeds = seeds_arg(args.seeds)
    results = []
    if args.resume and (out / "runs.json").is_file():
        results = [r for r in json.loads((out / "runs.json").read_text(encoding="utf-8"))
                   if "error" not in r and r.get("seed") in seeds and (out / "runs" / f"seed{r['seed']}.npz").is_file()]
    todo = [s for s in seeds if s not in {r["seed"] for r in results}]
    if args.resume:
        print(f"resume {out}: {len(results)} runs kept, {len(todo)} to run", flush=True)
        if not todo and (out / "dataset.npz").is_file():
            return 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one_run, s, a, nodes, str(out), args.port + i): s for i, s in enumerate(todo)}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - one failed run is reported, the others go on
                r = {"seed": futs[f], "error": f"{type(e).__name__}: {e}"}
            results.append(r)
            print(json.dumps(r), flush=True)
    good = sorted((r for r in results if "error" not in r), key=lambda r: seeds.index(r["seed"]))
    (out / "runs.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if not good:
        print("no run finished")
        return 2
    parts = [np.load(out / "runs" / f"seed{r['seed']}.npz") for r in good]
    T = [len(p["t"]) for p in parts]
    meta = {"dt_s": args.dt, "horizons_s": list(HORIZONS_S), "source": "exact", "controller": args.controller,
                "clock_speed": args.clock_speed, "start_hour": args.start_hour, "scale": args.scale, "minutes": args.minutes,
                "warmup": args.warmup, "accidents": args.accidents, "careless_share": args.careless_share, "buses": args.buses,
                "ring": args.ring, "junction_blocker": args.junction_blocker, "trips": args.trips, "git_commit": git_commit(),
                "tool": "tools/goflow_prediction_dataset.py", "runs": good}
    np.savez_compressed(
        out / "dataset.npz", nodes=np.array([n["node"] for n in nodes]),
        storage=np.array([n["storage"] for n in nodes], dtype=np.float64),
        t=np.concatenate([p["t"] for p in parts]).astype(np.float64),
        run=np.concatenate([np.full(n_, i) for i, n_ in enumerate(T)]),
        seed=np.concatenate([np.full(n_, r["seed"]) for n_, r in zip(T, good)]),
        tod_s=np.concatenate([p["tod_s"] for p in parts]), demand=np.concatenate([p["demand"] for p in parts]),
        demand_at=np.concatenate([p["demand_at"] for p in parts]), meta=json.dumps(meta),
        **{c_: np.concatenate([p[c_] for p in parts]) for c_ in ("halt", "veh", "speed", "occ", "green")})
    print(f"[ok] {out / 'dataset.npz'}: {len(good)} runs, {sum(T)} steps x {len(nodes)} approaches")
    return 0


if __name__ == "__main__":
    sys.exit(main())

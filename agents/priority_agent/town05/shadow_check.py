"""Step 2 of the integration plan: my UNCHANGED code (environment.intersection, coordination.priority,
coordination.rules) watching a Town05 SUMO-only run, without touching any signal.

    python -m town05.shadow_check --seed 11 --scale 0.8 --end 1800

Run from agents/priority_agent with its venv (needs eclipse-sumo/traci 1.27.1 and the project's cosim on the path,
which this script adds). The town runs as under "--controller agents" (SUMO's actuated programs + the junction agents,
their own pre-emption). For each vehicle on a run, this script sets the SUMO parameters my scenarios used for the
V2I message (priority_class, siren=on, confirmed=true): that is the only thing it writes, and it changes no signal
and no driving. Writes out/goflow_priority/shadow_<time>/report.md and report.json:
  * what Intersection.from_sumo finds on the 15 signals (greens: the PPO needs exactly 2; approaches that collapse);
  * on-run vehicles tracked by my PriorityTracker against the extras' count;
  * whether my RuleCoordinator runs, and how often its targets agree with the agents' own requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # agents/priority_agent
ROOT = HERE.parents[1]                               # the Town05 project
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from coordination.priority import PriorityTracker  # noqa: E402  (unchanged)
from coordination.rules import RuleCoordinator  # noqa: E402  (unchanged)
from environment.intersection import Intersection  # noqa: E402  (unchanged)

CLASS = {"emergency": "EMERGENCY_MEDICAL", "authority": "POLICE"}


def run(seed: int, scale: float, end: float, port: int, shadow: bool, out: Path | None) -> dict:
    from cosim.sumo_side import SumoSide

    s = SumoSide(ROOT / "scenarios" / "town05", "agents", seed=seed, port=port, label=f"shadow{seed}{int(shadow)}",
                 extras=True, emergency_every=120.0,
                 extra_args=["--scale", f"{scale:g}", "--end", f"{end:g}"])
    ex = s.extras
    s.policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    c = s.conn
    h = hashlib.sha256()
    rep: dict = {"seed": seed, "scale": scale, "end": end}
    inters, from_sumo = {}, {}
    if shadow:
        for tls in c.trafficlight.getIDList():
            try:
                it = Intersection.from_sumo(c, tls)
                inters[tls] = it
                lanes_by_dir = Counter()
                for ln in dict.fromkeys(c.trafficlight.getControlledLanes(tls)):
                    if not ln.startswith(":"):  # road lanes only (not walking areas or crossings)
                        lanes_by_dir[c.lane.getEdgeID(ln)] += 1
                from_sumo[tls] = {"greens": len(it.green_phases), "approaches": sorted(it.incoming),
                                  "incoming_edges": len(lanes_by_dir),
                                  "collapsed": len(lanes_by_dir) - len(it.incoming),
                                  "lanes_per_approach": {a: len(v) for a, v in it.incoming_lanes.items()}}
            except Exception as e:  # noqa: BLE001
                from_sumo[tls] = {"error": f"{type(e).__name__}: {e}"}
        tracker = PriorityTracker(c, inters)
        coord = RuleCoordinator(horizon_s=30.0, recovery_s=10.0, inters=inters)
    tagged: set[str] = set()
    on_run_seen: set[str] = set()
    tracked_seen: set[str] = set()
    coord_errors = Counter()
    agree = disagree = only_mine = only_agents = 0
    while s.time < end:
        s.step()
        t = s.time
        for tls in s.tls_ids:
            h.update(c.trafficlight.getRedYellowGreenState(tls).encode())
        if not shadow:
            continue
        alive = set(c.vehicle.getIDList())
        runs = [v for v, r in ex.emergency.items() if r.on_run and v in alive]
        on_run_seen.update(runs)
        for vid in runs:
            if vid not in tagged:  # the V2I message of my scenarios: the only write
                vc = c.vehicle.getVehicleClass(vid)
                c.vehicle.setParameter(vid, "priority_class", CLASS.get(vc, "POLICE"))
                c.vehicle.setParameter(vid, "siren", "on")
                c.vehicle.setParameter(vid, "confirmed", "true")
                tagged.add(vid)
        try:
            vehicles = tracker.update(t)
            tracked_seen.update(v for v, pv in vehicles.items() if pv.is_active)
        except Exception as e:  # noqa: BLE001
            coord_errors[f"tracker {type(e).__name__}: {e}"] += 1
            continue
        try:
            ctx = coord.update(t, vehicles)
        except Exception as e:  # noqa: BLE001
            coord_errors[f"coordinator {type(e).__name__}: {e}"] += 1
            continue
        mine = {tls: cx.vehicle_id for tls, cx in ctx.items() if cx.local_mode == "EMERGENCY"}
        theirs = dict(s.policy.request_by)
        for tls in set(mine) | set(theirs):
            if tls in mine and tls in theirs:
                agree += mine[tls] == theirs[tls]
                disagree += mine[tls] != theirs[tls]
            elif tls in mine:
                only_mine += 1
            else:
                only_agents += 1
    s.close()
    rep["signal_states_sha256"] = h.hexdigest()
    if shadow:
        rep.update({"from_sumo": from_sumo, "on_run_vehicles": len(on_run_seen), "tracked_active": len(tracked_seen),
                    "coordinator_errors": dict(coord_errors.most_common(10)),
                    "steps_both_request_same_vehicle": agree, "steps_both_request_other_vehicle": disagree,
                    "steps_only_mine": only_mine, "steps_only_agents": only_agents})
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--end", type=float, default=1800.0)
    ap.add_argument("--port", type=int, default=8950)
    ap.add_argument("--no-reference", action="store_true", help="skip the second run without the script")
    a = ap.parse_args(argv)
    out = ROOT / "out" / "goflow_priority" / f"shadow_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        rep = run(a.seed, a.scale, a.end, a.port, True, out)
        if not a.no_reference:
            ref = run(a.seed, a.scale, a.end, a.port + 1, False, None)
            rep["reference_signal_states_sha256"] = ref["signal_states_sha256"]
            rep["signals_untouched"] = ref["signal_states_sha256"] == rep["signal_states_sha256"]
    except Exception:  # noqa: BLE001
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    fs = rep["from_sumo"]
    lines = [f"# Shadow check: my unchanged code on Town05 (seed {a.seed}, x{a.scale:g}, {a.end:g} s)", "",
             "| signal | greens found (PPO needs 2) | incoming roads | approaches N/E/S/W kept | collapsed | error |",
             "|---|---|---|---|---|---|"]
    for tls, f in sorted(fs.items()):
        lines.append(f"| {tls} | {f.get('greens', '')} | {f.get('incoming_edges', '')} | "
                     f"{' '.join(f.get('approaches', []))} | {f.get('collapsed', '')} | {f.get('error', '')} |")
    lines += ["", f"* on-run vehicles: {rep['on_run_vehicles']}, active in my tracker: {rep['tracked_active']}",
              f"* coordinator errors: {rep['coordinator_errors'] or 'none'}",
              f"* steps where both ask for the same vehicle: {rep['steps_both_request_same_vehicle']}; other vehicle: "
              f"{rep['steps_both_request_other_vehicle']}; only mine: {rep['steps_only_mine']}; only the agents': "
              f"{rep['steps_only_agents']}",
              f"* signals untouched (same signal states as a run without this script): {rep.get('signals_untouched')}"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

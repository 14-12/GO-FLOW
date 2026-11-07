"""The live connector checked end to end, SUMO alone (no CARLA): the plan's step S6 checks that do not need the picture.

    .venv\\Scripts\\python tools\\goflow_prediction_live_check.py [--minutes 12] [--kill-at 6] [--goflow]

On the training data's own traffic (tools/goflow_prediction_dataset.one_run), cosim/goflow_prediction.GoflowPrediction
is called after every SUMO step exactly as FreeFly.produce() calls it, with the real agent processes in their own
venvs. SUMO runs as fast as it can (about 25x the viewer), so the agents get more traffic per second than live.

Run 1 (normal): the agents start and answer; the extra time the connector costs the SUMO thread per step; the
forecasts written live equal an offline replay of the same messages through goflow_prediction.serve.
Run 2 (--kill-at): DSO4's process is killed at that simulated minute: the town must go on, the process must be
restarted, and forecasts must come back.

Writes out/goflow_prediction/live_check_<stamp>/report.json and prints it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def run(name, a, out, kill_at_s=None):
    import goflow_prediction_dataset as gen

    from cosim.goflow_prediction import GoflowPrediction

    nodes = json.loads((ROOT / "out" / "goflow_prediction" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    models = [str(ROOT / "out" / "goflow_prediction" / "models" / "lgbm_real_v2")]
    gf = None
    if a.goflow:
        models.append(str(ROOT / "out" / "goflow_prediction" / "models" / "lgbm_gf_v2"))
        gf = str(ROOT / "out" / "goflow_traffic" / "models" / "xgb_120s_v1")
    st = {"fp": None, "costs": [], "t0": None, "killed": None, "restarted": None, "steps": 0}

    def probe(cur, conn, sod):
        if st["fp"] is None:
            st["fp"] = GoflowPrediction(conn, model_dirs=models, goflow_model=gf, clock_speed=a.clock_speed,
                                        out_dir=out / name)
            st["t0"] = cur.time
            time.sleep(8.0)            # let the children load (the viewer takes longer than that to start anyway)
        fp = st["fp"]
        t = time.perf_counter()
        fp.on_step(cur, conn, sod)
        st["costs"].append(time.perf_counter() - t)
        st["steps"] += 1
        if kill_at_s is not None and st["killed"] is None and cur.time - st["t0"] >= kill_at_s:
            fp.dso4.proc.kill()
            st["killed"] = {"sim_t": cur.time, "wall": time.time(), "forecasts_before": _count(out / name, "forecasts")}
        if st["killed"] and st["restarted"] is None and fp.dso4.state == "ready" and fp.dso4.failures:
            st["restarted"] = {"sim_t": cur.time, "after_wall_s": round(time.time() - st["killed"]["wall"], 1)}
        if a.slow:
            time.sleep(a.slow)

    gargs = {"minutes": a.minutes, "warmup": 1.0, "start_hour": a.start_hour, "clock_speed": a.clock_speed, "dt": 30.0,
                 "controller": "coordinated", "scale": a.scale, "accidents": False, "careless_share": None, "buses": True,
                 "ring": True, "junction_blocker": 60.0}
    (out / name / "runs").mkdir(parents=True, exist_ok=True)
    t_wall = time.time()
    res = gen.one_run(a.seed, gargs, nodes, str(out / name), a.port, probe=probe)
    wall = time.time() - t_wall
    fp = st["fp"]
    time.sleep(3.0)                    # the last answers
    states = {"dso4": fp.dso4.state, "dso4_failures": fp.dso4.failures,
              "goflow": fp.goflow.state if fp.goflow else None}
    fp.close()
    costs = sorted(st["costs"])
    pick = lambda q: round(1000 * costs[min(len(costs) - 1, int(q * len(costs)))], 3)
    return {"run": name, "sumo_steps": st["steps"], "wall_s": round(wall, 1), "town": res, "children": states,
                "obs": _count(out / name, "obs"), "forecasts": _count(out / name, "forecasts"),
                "alerts": _count(out / name, "alerts"), "goflow_answers": _count(out / name, "goflow"),
                "obs_dropped": fp.dropped, "connector_ms_per_step": {"median": pick(0.5), "p99": pick(0.99),
                                                              "max": pick(1.0)},
                "killed": st["killed"], "restarted": st["restarted"]}


def _count(folder, kind) -> int:
    p = Path(folder) / f"{kind}.jsonl"
    if not p.is_file():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def replay_equal(folder: Path, models: list[str]) -> dict:
    """The logged messages fed again to goflow_prediction.serve (offline): same forecasts as live?"""
    with open(folder / "obs.jsonl", encoding="utf-8") as f:
        obs = [json.loads(x) for x in f]
    sent = [{k: v for k, v in m.items() if k != "delivered"} for m in obs if m.get("delivered")]
    nodes = json.loads((ROOT / "out" / "goflow_prediction" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    hello = {"type": "hello", "v": 1, "dt_s": 30.0, "source": "exact", "nodes": [n["node"] for n in nodes]}
    py = ROOT / "agents" / "goflow_prediction" / ".venv" / "Scripts" / "python.exe"
    args = [str(py), "-m", "goflow_prediction.serve"]
    for m in models:
        args += ["--model", m]
    p = subprocess.run(args, input="\n".join(json.dumps(m) for m in [hello, *sent]) + "\n", capture_output=True,
                       text=True, cwd=str(ROOT / "agents" / "goflow_prediction"), timeout=600, check=False)
    offline = {m["seq"]: m for m in map(json.loads, p.stdout.splitlines()) if m.get("type") == "forecast"}
    with open(folder / "forecasts.jsonl", encoding="utf-8") as f:
        live = {m["seq"]: m for m in map(json.loads, f)}
    common = sorted(set(offline) & set(live))
    same = sum(1 for s in common if offline[s]["nodes"] == live[s]["nodes"] and offline[s]["chosen"] == live[s]["chosen"])
    return {"live_forecasts": len(live), "offline_forecasts": len(offline), "compared": len(common), "identical": same}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--kill-at", type=float, default=6.0, help="simulated minute at which run 2 kills DSO4's process")
    ap.add_argument("--goflow", action="store_true", help="GO-FLOW's agent too")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--clock-speed", type=float, default=60.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--slow", type=float, default=0.0, help="seconds of sleep per SUMO step (closer to the viewer)")
    ap.add_argument("--port", type=int, default=8960)
    a = ap.parse_args(argv)
    out = ROOT / "out" / "goflow_prediction" / f"live_check_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    models = [str(ROOT / "out" / "goflow_prediction" / "models" / "lgbm_real_v2")]
    if a.goflow:
        models.append(str(ROOT / "out" / "goflow_prediction" / "models" / "lgbm_gf_v2"))
    r1 = run("normal", a, out)
    r1["replay"] = replay_equal(out / "normal", models)
    a.port += 1
    r2 = run("kill", a, out, kill_at_s=60.0 * a.kill_at)
    report = {"args": vars(a), "normal": r1, "kill": r2}
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

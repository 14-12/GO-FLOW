"""Town05 in sumo-gui (2D, no CARLA) with the GO-FLOW priority agent: a demo of the emergency-vehicle priority.

    python tools/goflow_priority_demo.py                 # the junction agents + GO-FLOW (the agent plans the requests)
    python tools/goflow_priority_demo.py --without       # the same city, the junction agents alone (before / after)
    python tools/goflow_priority_demo.py --llm qwen2.5:3b   # + the LLM in the background (Ollama running)

The city runs fast while there are only cars; as soon as an ambulance or a police car is on a run, it slows down,
the view follows that vehicle (ringed in yellow: ambulance, blue: police), and the terminal prints, live, what the
agents decide at each junction (GO-FLOW: green for ambulance, getting ready for ..., two emergencies here, ...).
At the end: how the emergency vehicles fared, and how often GO-FLOW was in charge.

Needs SUMO (eclipse-sumo) and traci in the Python that runs it; the GO-FLOW child process uses GOFLOW_PRIORITY_PY,
else agents/priority_agent/.venv, else this same Python. Launcher: tools/goflow_priority_demo.bat.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIEW = "View #0"
COLORS = {"emergency": (255, 210, 0, 255), "authority": (40, 120, 255, 255)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--without", action="store_true", help="the junction agents alone (no GO-FLOW)")
    ap.add_argument("--observed", action="store_true", help="the agents on the cameras' reports (agents_observed)")
    ap.add_argument("--llm", default=None, metavar="MODEL", help="the LLM in the background (Ollama model)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scale", type=float, default=0.8, help="demand (0.8 = 80 %%)")
    ap.add_argument("--every", type=float, default=45.0, help="mean seconds between emergency vehicles")
    ap.add_argument("--end", type=float, default=1800.0, help="simulated seconds")
    ap.add_argument("--delay", type=float, default=0.08, help="real seconds per step while an emergency is on a run")
    ap.add_argument("--zoom", type=float, default=900.0)
    ap.add_argument("--no-gui", action="store_true", help="sumo instead of sumo-gui (tests)")
    a = ap.parse_args(argv)

    if not a.without and not os.environ.get("GOFLOW_PRIORITY_PY"):
        venv = ROOT / "agents" / "priority_agent" / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        os.environ["GOFLOW_PRIORITY_PY"] = str(venv) if venv.is_file() else sys.executable
    if a.llm:
        os.environ["GOFLOW_PRIORITY_ARGS"] = f"--llm --llm-model {a.llm}"
    os.environ.setdefault("GOFLOW_PRIORITY_MODE", "lockstep")  # the demo may wait for it: no CARLA to keep fed

    from cosim.sumo_side import SumoSide

    base = "agents_observed" if a.observed else "agents"
    controller = base if a.without else base + "_goflow"
    trip = ROOT / "out" / "goflow_priority" / f"demo_{time.strftime('%Y%m%d-%H%M%S')}_{controller}.tripinfo.xml"
    trip.parent.mkdir(parents=True, exist_ok=True)
    extra = ["--scale", f"{a.scale:g}", "--end", f"{a.end:g}", "--tripinfo-output", str(trip),
             "--tripinfo-output.write-unfinished", "true"]
    if not a.no_gui:
        extra += ["--start", "true", "--quit-on-end", "true", "--delay", "0"]
    print(f"Town05, {controller}, seed {a.seed}, demand x{a.scale:g}, an emergency vehicle every ~{a.every:g} s")
    s = SumoSide(ROOT / "scenarios" / "town05", controller, gui=not a.no_gui, seed=a.seed, port=8890,
                 label="goflow_demo", extras=True, emergency_every=a.every, extra_args=extra)
    ex = s.extras
    ex.config.pedestrians_per_km = 20.0
    ex.config.pedestrian_trips = True
    pol = s.policy
    pol.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    c = s.conn
    cams = None
    if pol.uses_cameras:  # as tools/eval_town05_baseline.py: the cameras' reports every 2 s, then the agents
        import json

        from cosim.camera_reports import Layout, SimulatedCameras
        cam_dir = ROOT / "out" / "town05_cameras"
        parked_file = cam_dir / "parked_static.json"
        parked = json.loads(parked_file.read_text(encoding="utf-8")) if parked_file.is_file() else None
        cams = SimulatedCameras(Layout.load(cam_dir), seed=a.seed, parked=parked)
        pol.visible = cams.layout.visible
    from cosim.controllers import _Conn
    if not a.no_gui:
        try:
            c.gui.setSchema(VIEW, "real world")
        except Exception:  # noqa: BLE001 - not every SUMO has it
            pass
    printed = 0
    followed = None
    marked: set[str] = set()
    seen_run: dict[str, str] = {}
    try:
        while s.time < a.end and c.simulation.getMinExpectedNumber() > 0:
            if cams is None:
                s.step()
            else:
                s.step(run_policy=False)
                reps = cams.step(c, s.time, {v: rec.on_run for v, rec in ex.emergency.items()})
                if reps is not None:
                    pol.observe(reps)
                pol.step(_Conn(c))
            t = s.time
            runs = [v for v in pol.emergency() if v in set(c.vehicle.getIDList())]
            for vid in runs:
                if vid not in seen_run:
                    vc = c.vehicle.getVehicleClass(vid)
                    seen_run[vid] = vc
                    print(f"[{t:6.0f} s] {('AMBULANCE' if vc == 'emergency' else 'POLICE')} {vid} starts a run")
                if vid not in marked and not a.no_gui:
                    marked.add(vid)
                    try:
                        c.vehicle.highlight(vid, COLORS.get(seen_run[vid], (255, 255, 255, 255)), 12, 255, -1, 0)
                    except Exception:  # noqa: BLE001
                        pass
            if not a.no_gui:
                want = runs[0] if runs else None
                if want != followed:
                    try:
                        c.gui.trackVehicle(VIEW, want or "")
                        if want:
                            c.gui.setZoom(VIEW, a.zoom)
                    except Exception:  # noqa: BLE001
                        pass
                    followed = want
                if runs:
                    time.sleep(a.delay)
            ev = list(pol.events)
            for e in ev[printed:]:
                if e.get("kind") in ("ambulance", "corridor"):
                    print(f"[{e['t']:6.0f} s] junction {e['tls']}: {e['text']}")
            printed = len(ev)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        stats = dict(pol.stats)
        s.close()
    summary(trip, set(seen_run), stats, controller)
    return 0


def summary(trip: Path, ems: set[str], stats: dict, controller: str) -> None:
    import xml.etree.ElementTree as ET

    if not trip.is_file():
        return
    loss, wait_cars, n_cars = [], 0.0, 0
    for el in ET.parse(trip).getroot().iter("tripinfo"):
        if el.get("id") in ems:
            loss.append(float(el.get("timeLoss", 0)))
        else:
            wait_cars += float(el.get("waitingTime", 0))
            n_cars += 1
    print("\n" + "=" * 60)
    print(f"{controller}: {len(loss)} emergency trips, mean time loss "
          f"{(sum(loss) / len(loss)) if loss else 0:.1f} s; other vehicles waited {wait_cars / max(n_cars, 1):.1f} s on average")
    if "external_plan_steps" in stats:
        tot = stats["external_plan_steps"] + stats.get("external_fallback_steps", 0)
        print(f"GO-FLOW in charge on {stats['external_plan_steps']} of {tot} steps; "
              f"{stats.get('external_requests', 0)} requests sent to the junction agents")
    if stats.get("goflow_llm_calls") is not None:
        print(f"LLM: {stats['goflow_llm_calls']} calls, {stats.get('goflow_llm_failures')} failed, "
              f"p95 {stats.get('goflow_llm_p95_latency_s')} s")
    print(f"trip details: {trip}")


if __name__ == "__main__":
    sys.exit(main())

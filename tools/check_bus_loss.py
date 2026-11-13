"""(13 Nov) Where the buses and the emergency vehicles lose their time, road by road, and how many cars park on each
side of the kerb streets: SUMO alone, the evaluation's run (tools/goflow_prediction_dataset.one_run, accidents on),
cosim/purposes.PARK_SIDE "random" or "route" (the parking side by route failed its test with accidents: the buses
lost 8 % more time a km, 12 Nov):

    python tools/check_bus_loss.py --seeds 11 --park-side route --out out/park_side/bus_dev

Every step (0.5 s), each line bus (pt_*) and each emergency vehicle on a run: time lost on its road = dt x (1 - its
speed / its allowed speed), a bus's time standing at its stop left out. Each car parking: the road it parks on.
Writes OUT/<side>_seed<k>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def one(seed: int, side: str, out: Path, port: int, minutes: float, controller: str) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    a = {"minutes": minutes, "warmup": 5.0, "start_hour": 6.0, "clock_speed": GAME_SPEED, "dt": 30.0,
         "controller": controller, "scale": 1.0, "accidents": True, "careless_share": None, "buses": True,
         "ring": True, "junction_blocker": 60.0, "trips": "purpose", "park_side": side}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    st = {"bus": {}, "ems": {}, "parks": {}, "unparks": {}, "k": 0, "ems_ids": set(), "t0": None}

    def hook(sumo, im) -> None:
        st.update(sumo=sumo, im=im)

    def probe(cur, conn, sod) -> None:
        st["k"] += 1
        t = conn.simulation.getTime()
        if st["t0"] is None:
            st["t0"] = t
        if t < st["t0"] + 60.0 * a["warmup"]:
            return
        dt = conn.simulation.getDeltaT()
        if st["k"] % 10 == 0:
            sumo, im = st["sumo"], st["im"]
            ex = sumo.extras
            st["ems_ids"] = {v for v in (ex.emergency if ex is not None else ()) if sumo.on_run(v)} | \
                {v for v in list(im.role) if im.on_run(v)}
        for vid in conn.simulation.getParkingStartingVehiclesIDList():
            r = conn.vehicle.getRoadID(vid)
            st["parks"][r] = st["parks"].get(r, 0) + 1
        for vid in conn.simulation.getParkingEndingVehiclesIDList():
            try:
                r = conn.vehicle.getRoadID(vid)
            except Exception:  # noqa: BLE001
                continue
            st["unparks"][r] = st["unparks"].get(r, 0) + 1
        for vid in cur.vehicles:
            kind = "bus" if vid.startswith("pt_") else "ems" if vid in st["ems_ids"] else None
            if kind is None:
                continue
            try:
                if kind == "bus" and conn.vehicle.getStopState(vid) & 1:
                    continue                          # (at its stop: its own time, not lost)
                r = conn.vehicle.getRoadID(vid)
                vmax = conn.vehicle.getAllowedSpeed(vid)
                v = conn.vehicle.getSpeed(vid)
            except Exception:  # noqa: BLE001
                continue
            if vmax > 0.1:
                d = st[kind]
                d[r] = d.get(r, 0.0) + dt * max(0.0, 1.0 - v / vmax)

    w = out / "work" / f"{side}_s{seed}"
    (w / "runs").mkdir(parents=True, exist_ok=True)
    t = time.time()
    res = gen.one_run(seed, a, nodes, str(w), port, probe=probe, hook=hook)
    r = {"seed": seed, "side": side, "bus": st["bus"], "ems": st["ems"], "parks": st["parks"],
         "unparks": st["unparks"], "incidents": res.get("incidents"), "wall_s": round(time.time() - t)}
    (out / f"{side}_seed{seed}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--park-side", required=True, choices=("random", "route"))
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--controller", default="agents_observed+bus@bus_cut=1")
    ap.add_argument("--port", type=int, default=9960)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_agents_forecast import seeds_arg

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(seeds_arg(a.seeds)):
        if (out / f"{a.park_side}_seed{seed}.json").is_file():
            continue
        r = one(seed, a.park_side, out, a.port + i, a.minutes, a.controller)
        print(seed, a.park_side, r["wall_s"], "s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A town day with trips that have a purpose (cosim/purposes.py), SUMO alone: what the cars do hour by hour and how
the car parks fill, against the same day on the old random routes.

    python tools/check_purpose_day.py --seed 11 --scale 1 --hours 24 --out out/purpose_day

Each run is the forecasters' recipe (tools/goflow_prediction_dataset.one_run: crashes, buses, the ring road, the clock
at 30, from --start), --hours town hours long (2 real minutes each), under --controller, once with trips="purpose" and
once with "random". Writes OUT/<trips>_seed<seed>.json and OUT/day.png (the purposes by hour, the car parks' fill).
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def run(trips: str, a, port: int) -> dict:
    import goflow_prediction_dataset as gen

    from cosim.daily import GAME_SPEED

    out = Path(a.out)
    work = out / "work" / f"{trips}_{a.seed}"
    (work / "runs").mkdir(parents=True, exist_ok=True)
    trip = work / "tripinfo.xml"
    minutes = a.hours * 3600.0 / GAME_SPEED / 60.0
    args = {"minutes": minutes, "warmup": 1.0, "start_hour": a.start, "clock_speed": GAME_SPEED, "dt": 30.0,
            "controller": a.controller, "scale": a.scale, "accidents": True, "careless_share": None, "buses": True,
            "ring": True, "junction_blocker": 60.0, "trips": trips,
            "sumo_extra": ["--tripinfo-output", str(trip), "--tripinfo-output.write-unfinished", "true"]}
    nodes = json.loads((ROOT / "models" / "goflow" / "graph" / "nodes.json").read_text(encoding="utf-8"))
    st_: dict = {"series": [], "k": 0}

    def hook(sumo, im) -> None:
        st_["sumo"] = sumo

    def probe(cur, conn, sod) -> None:
        st_["k"] += 1
        if st_["k"] % 40 == 0:                       # every 20 s of SUMO (10 town minutes)
            pop = st_.get("pop")
            occ, stale = {}, 0
            if pop is not None and getattr(pop, "demand", None) is not None:
                occ = pop.demand.occupancy()
                gone = [v for v in pop.demand.parked if v not in cur.vehicles]
                stale = len(gone)

            parked = pop.demand.parked if pop is not None and getattr(pop, "demand", None) is not None else {}
            standing: dict = {}                      # where the cars stand (not parked): the knots, if any
            for vid, v in cur.vehicles.items():
                if v.speed < 0.1 and vid not in parked:
                    try:
                        e = conn.vehicle.getRoadID(vid)
                    except Exception:  # noqa: BLE001, S112
                        continue
                    standing[e] = standing.get(e, 0) + 1
            st_["series"].append({"hour": round(sod / 3600.0, 3), "moving": len(cur.vehicles) - sum(occ.values()),
                                  "occ": occ, "stale": stale, "standing": standing})

    # the Population is made inside one_run: catch it as it is built
    from cosim import live

    orig_init = live.Population.__init__

    def init(self, *aa, **kw):
        orig_init(self, *aa, **kw)
        st_["pop"] = self
    live.Population.__init__ = init
    t0 = time.time()
    try:
        res = gen.one_run(a.seed, args, nodes, str(work), port, probe=probe, hook=hook)
    finally:
        live.Population.__init__ = orig_init
    root = ET.parse(trip).getroot()
    town = [x for x in root.findall("tripinfo") if x.get("id", "").startswith("live")]
    done = [x for x in town if float(x.get("arrival", -1)) >= 0]
    km = sum(float(x.get("routeLength", 0)) for x in town) / 1000.0
    legs = len(done) + ((res.get("purposes") or {}).get("arrived_parked", 0))
    r = {"trips": trips, "seed": a.seed, "scale": a.scale, "hours": a.hours, "start": a.start,
         "town_trips": len(town), "finished": len(done), "km": round(km, 1), "legs": legs,
         # (a car on trips with a purpose is one vehicle for all its legs: compare per km driven, per leg)
         "waiting_per_km": round(sum(float(x.get("waitingTime")) for x in town) / km, 2) if km else None,
         "time_loss_per_km": round(sum(float(x.get("timeLoss")) for x in town) / km, 2) if km else None,
         "km_per_leg": round(km / legs, 3) if legs else None,
         "moving_mean": round(st.mean(p["moving"] for p in st_["series"]), 1) if st_["series"] else None,
         "waiting": round(st.mean(float(x.get("waitingTime")) for x in done), 2) if done else None,
         "time_loss": round(st.mean(float(x.get("timeLoss")) for x in done), 2) if done else None,
         "stuck_over_300s": res["stuck_over_300s"], "gridlock": res["gridlock"], "incidents": res["incidents"],
         "purposes": res.get("purposes"), "series": st_["series"], "wall_s": round(time.time() - t0)}
    (out / f"{trips}_seed{a.seed}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    trip.unlink(missing_ok=True)
    return r


def table(r: dict, path: Path) -> None:
    """The cars parked in each area, hour of the day by hour (the mean of the samples in that hour; the run's first
    hour left out: the town starts empty), for cosim/purposes.py's prefill."""
    acc: dict = {}
    for i, p in enumerate(r["series"]):
        if i < 6:                                  # (a sample every 10 town minutes: the first hour)
            continue
        h = str(int(p["hour"]) % 24)
        acc.setdefault(h, []).append(p["occ"])
    hours = {}
    for h, samples in sorted(acc.items(), key=lambda kv: int(kv[0])):
        areas = sorted({a for o in samples for a in o})
        hours[h] = {a: round(st.mean(o.get(a, 0) for o in samples), 1) for a in areas}
    card = {"what": "cars parked in each parking area by hour of the day, from a day of trips with a purpose "
                    "(tools/check_purpose_day.py --table): cosim/purposes.py puts them there when a run starts",
            "seed": r["seed"], "start": r["start"], "hours_run": r["hours"], "hours": hours}
    path.write_text(json.dumps(card, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--controller", default="agents_observed")
    ap.add_argument("--trips", default="purpose,random")
    ap.add_argument("--port", type=int, default=9950)
    ap.add_argument("--out", default=str(ROOT / "out" / "purpose_day"))
    ap.add_argument("--table", default=None, help="write the parked-by-hour table here (from the purpose run): "
                                                  "use a training seed")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(a.trips.split(",")):
        r = run(t, a, a.port + 3 * i)
        if t == "purpose" and a.table:
            table(r, Path(a.table))
        print(t, {k: r[k] for k in ("town_trips", "legs", "km", "km_per_leg", "waiting_per_km", "time_loss_per_km",
                                    "moving_mean", "stuck_over_300s", "gridlock", "wall_s")}, (r["purposes"] or {}).get("trips"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

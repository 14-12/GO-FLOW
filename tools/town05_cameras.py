"""Step 2 of the multi-agent plan: the Town05 traffic-light cameras as reporters.

    python tools/town05_cameras.py plan            # no CARLA server needed (a few minutes)
    python tools/town05_cameras.py live            # CARLA server running (start_carla.bat); GPU detector

plan  Lays out the 53 cameras exactly as the inspector does (cosim/inspector.py: one per road at
      each signal, 10 m into the junction, 9 m up, pitch -22, 960x540, 75 deg), draws the lanes
      max-pressure reads into each, and every drivable lane in view (parked vs queued, see
      cosim/camera_reports.py). Writes out/town05_cameras/: layout.json, masks.npz,
      coverage.json and coverage.md (per junction: metres of its incoming and outgoing lanes in a
      camera's picture), and with --previews one coloured mask per camera. Ground heights come
      from the map file through CARLA's client library (no server); without it, from SUMO.

live  Co-simulation (SUMO master, CARLA renders, as tools/eval_perception.py) at a few junctions:
      every 1/--fps simulated seconds, those junctions' cameras take a picture, the detector runs,
      each camera's CameraReporter tracks the vehicles; every --period s each camera reports.
      Written to out/town05_cameras/live/: reports.jsonl (every report, with SUMO's truth for
      the same lanes next to it), accuracy.json (count and queue errors, emergency confirmations
      right and wrong, parked cars per camera), parked_static.json (median parked cars per
      camera, which SimulatedCameras then reports). Default junctions: 1126, 1882, 2296, the
      detector's held-out test junctions, so nothing here was trained on.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import scenario as scenario_mod  # noqa: E402

OUT = ROOT / "out" / "town05_cameras"
TEST_JUNCTIONS = "1126,1882,2296"
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def _policy(scen, port: int):
    """A walk-aware max-pressure controller set up on Town05 (it knows which lanes it reads)."""
    from cosim.controllers import WalkMaxPressure
    from cosim.sumo_side import SumoSide

    ctl = WalkMaxPressure()
    s = SumoSide(scen.folder, "max_pressure_walk", seed=1, port=port, label="cams-plan", policy=ctl)
    s.close()
    return ctl


def _ground(scen):
    try:
        import carla

        cmap = carla.Map(scen.carla_map, (scen.folder / f"{scen.name}.xodr").read_text(encoding="utf-8"))

        def ground_z(x, y):
            wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
            return float(wp.transform.location.z) if wp is not None else 0.0

        return ground_z, "map file (CARLA client library)"
    except Exception as e:  # noqa: BLE001 - no carla module, or no map file
        return (lambda x, y: 0.0), f"flat z = 0 (no CARLA client library: {e.__class__.__name__})"


def plan(a) -> int:
    import sumolib

    from cosim.camera_reports import build_layout, coverage

    scen = scenario_mod.load("town05", ROOT)
    t0 = time.time()
    policy = _policy(scen, a.port)
    net = sumolib.net.readNet(str(scen.net_file), withInternal=True)
    ground_z, ground_src = _ground(scen)
    lay = build_layout(net, scen.frame, policy, ground_z)
    lay.save(OUT)
    cov = coverage(lay, policy)
    cov["ground"] = ground_src
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1), encoding="utf-8")
    md = ["# Town05 camera coverage", "",
          f"{cov['cameras']} cameras, {cov['lanes_counted']} vehicle lanes read by max-pressure, "
          f"{cov['lanes_owned']} of them owned by a camera. Ground heights: {ground_src}.", "",
          f"Of the last {cov['reach_m']:g} m before every stop line (where queues stand) and the first "
          f"{cov['reach_m']:g} m after each junction, the share inside a camera's picture: "
          f"**incoming {cov['in_share']:.0%}**, outgoing {cov['out_share']:.0%}. Geometry only: buildings and "
          "trees can hide more, so these are upper bounds.", "",
          "| junction | cameras | incoming seen | outgoing seen | incoming lanes under half seen |", "|---|---|---|---|---|"]
    for tls, v in sorted(cov["junctions"].items(), key=lambda kv: kv[1]["in_share"] or 0):
        md.append(f"| {tls} | {v['cameras']} | {v['in_seen_m']} / {v['in_total_m']} m ({(v['in_share'] or 0):.0%}) | "
                  f"{v['out_seen_m']} / {v['out_total_m']} m ({(v['out_share'] or 0):.0%}) | "
                  f"{len(v['incoming_lanes_under_half_seen'])} |")
    (OUT / "coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if a.previews:
        import numpy as np
        from PIL import Image

        prev = OUT / "previews"
        prev.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(lay.cams):
            rgb = np.zeros((*c.mask.shape, 3), dtype=np.uint8)
            rgb[lay.road[i]] = (70, 70, 70)  # road the camera does not count
            for v in np.unique(c.mask[c.mask > 0]):
                rgb[c.mask == v] = PALETTE[int(v) % len(PALETTE)]
            Image.fromarray(rgb).save(prev / f"{lay.names[i].replace(' ', '_')}.png")
    print(f"written to {OUT} in {time.time() - t0:.0f} s")
    return 0


# ----------------------------------------------------------------------------- live


def live(a) -> int:  # pragma: no cover - needs the CARLA server and the GPU
    import carla
    import numpy as np
    from cosim.carla_side import CarlaSide
    from cosim.convert import light_state
    from cosim.run import Poser, load_world
    from ultralytics import YOLO

    from cosim.camera_reports import CameraReporter, Layout, ReportRules, in_view
    from cosim.sumo_side import SumoSide

    lay = Layout.load(OUT)
    tls_set = set(a.tls.split(","))
    idx = [i for i, c in enumerate(lay.cams) if c.spec.tls in tls_set]
    if not idx:
        sys.exit(f"no camera at {a.tls}")
    scen = scenario_mod.load("town05", ROOT)
    out = OUT / "live"
    out.mkdir(parents=True, exist_ok=True)
    rules = ReportRules(period_s=a.period)
    reporters = {i: CameraReporter(lay, i, rules) for i in idx}

    sumo = SumoSide(scen.folder, a.controller, seed=a.seed, port=a.sumo_port, label="cams-live", extras=True,
                    emergency_every=a.emergency_every, extra_args=["--scale", f"{a.scale:g}"])
    sumo.extras.config.pedestrians_per_km = 0.0  # not counted here; spares the GPU
    client = carla.Client(a.host, a.port)
    client.set_timeout(300.0)
    world = load_world(client, scen)
    links = scen.links([lm.id for lm in world.get_map().get_all_landmarks_of_type("1000001")])
    side = CarlaSide(client, world, scen.frame, fps=20.0, links=links)
    side.poser = Poser(scen.frame, side.ground)
    lib = world.get_blueprint_library()
    sensors, qs = [], []
    w, h = lay.cams[0].width, lay.cams[0].height
    for i in idx:
        bp = lib.find("sensor.camera.rgb")
        for k, v in (("image_size_x", str(w)), ("image_size_y", str(h)), ("fov", str(lay.cams[0].fov)),
                     ("motion_blur_intensity", "0.0")):
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        s = lay.cams[i].spec
        sensor = world.spawn_actor(bp, carla.Transform(carla.Location(s.x, s.y, s.z), carla.Rotation(s.pitch, s.yaw, 0.0)))
        q: queue.Queue = queue.Queue()
        sensor.listen(q.put)
        sensors.append(sensor)
        qs.append(q)
    model = YOLO(str(ROOT / a.model))

    def render(cur):
        vehicles = cur.vehicles
        side.destroy([vid for vid in side.actors if vid not in vehicles])
        side.poser.begin_step(cur, cur)
        poses = side.poser.poses(1.0, sumo.step_length)
        side.spawn([(vid, v.vclass, poses[vid]) for vid, v in vehicles.items()
                    if vid not in side.actors and vid not in side.failed and vid in poses])
        side.set_traffic_lights(cur.tls)
        side.move(poses, {vid: light_state(v.signals, False, sumo.on_run(vid)) for vid, v in vehicles.items()})
        side.tick()
        fid = side.tick()  # moved actors settle in the frame that is read
        imgs = []
        for q in qs:
            while True:
                im = q.get(timeout=120)
                if im.frame >= fid:
                    imgs.append(np.frombuffer(im.raw_data, dtype=np.uint8).reshape(im.height, im.width, 4)[:, :, :3])
                    break
        return imgs

    def truth(c):
        """SUMO, for the camera's own lanes: vehicles at least partly on their visible stretches
        (motorcycles apart: the detector has no class for them), queued among them, emergency
        vehicles there on a run and not on a run."""
        tc = sumo.conn
        n = qd = moto = 0
        ems, ems_off = [], []
        for ln in lay.owned(c):
            for vid in tc.lane.getLastStepVehicleIDs(ln):
                if not in_view(lay, ln, tc.vehicle.getLanePosition(vid), tc.vehicle.getLength(vid)):
                    continue
                if tc.vehicle.getVehicleClass(vid) == "motorcycle":
                    moto += 1
                    continue
                n += 1
                qd += tc.vehicle.getSpeed(vid) < 0.1
                if sumo.on_run(vid):
                    ems.append(vid)
                elif sumo.extras is not None and vid in sumo.extras.emergency:
                    ems_off.append(vid)
        return n, qd, ems, ems_off, moto

    begin = sumo.time
    end = begin + a.duration
    last_pic = last_rep = -1e9
    rows = []
    t0 = time.time()
    log = (out / "reports.jsonl").open("w", encoding="utf-8")
    try:
        while sumo.time < end:
            cur = sumo.step()
            t = cur.time
            if t - last_pic >= 1.0 / a.fps - 1e-6:
                last_pic = t
                imgs = render(cur)
                res = model.predict(imgs, imgsz=a.imgsz, conf=a.conf, verbose=False, device=0)
                for i, r in zip(idx, res, strict=True):
                    boxes = r.boxes.xyxy.cpu().numpy()
                    names = [model.names[int(k)] for k in r.boxes.cls.cpu().numpy()]
                    reporters[i].observe(t, boxes, names)
            if t - last_rep >= a.period - 1e-6 and t - begin >= a.warmup:
                last_rep = t
                for i in idx:
                    rep = reporters[i].report(t)
                    n, qd, ems, ems_off, moto = truth(i)
                    rep["truth"] = {"vehicles": n, "queued": qd, "ems_on_run": len(ems), "ems_siren_off": len(ems_off),
                                    "motorcycles": moto}
                    rows.append(rep)
                    log.write(json.dumps(rep) + "\n")
            k = int(round(t - begin))
            if k % 60 == 0 and abs(t - begin - k) < 1e-6:
                print(f"t={k} s  vehicles={len(cur.vehicles)}  reports={len(rows)}  wall={time.time() - t0:.0f} s", flush=True)
    finally:
        log.close()
        sumo.close()
        for s in sensors:
            s.stop()
            s.destroy()
        side.close()

    def seen(r):
        return sum(v[0] for v in r["lanes"].values())

    def queued(r):
        return sum(v[1] for v in r["lanes"].values())

    per_cam: dict[str, list[dict]] = {}
    for r in rows:
        per_cam.setdefault(r["cam"], []).append(r)
    parked = {cam: int(st.median(x["parked"] for x in rs)) for cam, rs in per_cam.items()}
    ems_true = sum(1 for r in rows if r["truth"]["ems_on_run"] and any(e["confirmed"] for e in r["ems"]))
    ems_missed = sum(1 for r in rows if r["truth"]["ems_on_run"] and not any(e["confirmed"] for e in r["ems"]))
    ems_false = sum(1 for r in rows if not r["truth"]["ems_on_run"] and any(e["confirmed"] for e in r["ems"]))
    ems_false_off = sum(1 for r in rows if not r["truth"]["ems_on_run"] and r["truth"]["ems_siren_off"]
                        and any(e["confirmed"] for e in r["ems"]))
    tot_true = sum(r["truth"]["vehicles"] for r in rows)
    acc = {"junctions": sorted(tls_set), "cameras": len(idx), "reports": len(rows), "period_s": a.period,
           "pictures_per_s": a.fps, "simulated_s": a.duration, "scale": a.scale, "seed": a.seed,
           "vehicles_true_per_report": round(tot_true / max(len(rows), 1), 2),
           "vehicles_reported_per_report": round(sum(seen(r) for r in rows) / max(len(rows), 1), 2),
           "share_reported": round(sum(seen(r) for r in rows) / tot_true, 3) if tot_true else None,
           "count_mae_per_camera": round(st.mean(abs(seen(r) - r["truth"]["vehicles"]) for r in rows), 2) if rows else None,
           "queue_mae_per_camera": round(st.mean(abs(queued(r) - r["truth"]["queued"]) for r in rows), 2) if rows else None,
           "emergency_reports": {"confirmed_and_true": ems_true, "missed": ems_missed, "confirmed_but_false": ems_false,
                                 "of_which_a_siren_off_emergency_vehicle_in_view": ems_false_off},
           "parked_median_per_camera": parked, "wall_s": round(time.time() - t0)}
    (out / "accuracy.json").write_text(json.dumps(acc, indent=1), encoding="utf-8")
    (OUT / "parked_static.json").write_text(json.dumps(parked, indent=1), encoding="utf-8")
    print(json.dumps(acc, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--previews", action="store_true", help="save one coloured mask per camera")
    p.add_argument("--port", type=int, default=8879)
    lv = sub.add_parser("live")
    lv.add_argument("--tls", default=TEST_JUNCTIONS)
    lv.add_argument("--duration", type=float, default=900.0, help="simulated seconds")
    lv.add_argument("--warmup", type=float, default=120.0, help="no reports before this (the town fills)")
    lv.add_argument("--period", type=float, default=2.0, help="seconds between reports")
    lv.add_argument("--fps", type=float, default=2.0, help="pictures per simulated second")
    lv.add_argument("--scale", type=float, default=0.5)
    lv.add_argument("--seed", type=int, default=1)
    lv.add_argument("--controller", default="actuated")
    lv.add_argument("--emergency-every", type=float, default=60.0)
    lv.add_argument("--model", default="models/carla-yolo26n-t05.pt")
    lv.add_argument("--imgsz", type=int, default=960)
    lv.add_argument("--conf", type=float, default=0.25)
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=2000)
    lv.add_argument("--sumo-port", type=int, default=8871)
    a = ap.parse_args(argv)
    return plan(a) if a.cmd == "plan" else live(a)


if __name__ == "__main__":
    sys.exit(main())

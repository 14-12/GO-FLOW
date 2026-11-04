"""Free-fly over a CARLA town with SUMO's traffic: you fly the CARLA window, the city lives.

    python -m cosim.freefly                          # Town05, actuated signals, clock continues
    python -m cosim.freefly --set-time 17:30         # start the clock at 17:30 (the day is kept)
    python -m cosim.freefly --demo 60                # unattended check: a scripted flight, FPS to JSON

Started by town05_freefly.bat, which restarts CARLA in a window with the "medium" render profile
(tools/fps_test/engine_medium.ini; measured on the GTX 1650 Max-Q: 50 FPS flying over Town05, 112 at
street level, against 46 / 72 at plain Epic, out/town05_fps/summary.md).

How it differs from cosim/live.py (the Town12 viewer, 2-3 FPS):
* CARLA runs ASYNCHRONOUSLY and draws straight to its own window: no camera sensor, no image copied
  to Python, nobody waits for a tick. The frame rate is whatever the GPU gives.
* You move the camera yourself in the CARLA window: hold the right mouse button to look,
  W A S D to move, Q / E down / up. The vehicles drawn are those within --radius of where you are.
* SUMO runs in real time beside it (0.5 s steps); every vehicle is moved --rate times a second along a
  smooth curve between two SUMO states (cosim/interp.py), traffic lights follow SUMO.
* The clock (1 real minute = 1 game hour, cosim/daily.py) is drawn over the top-left corner of the
  CARLA picture (cosim/overlay.py). It drives the sun (street lights and headlights at night) and how
  many vehicles the town carries (the day's demand curve; Little's law with the town's mean trip).

Keys, pressed while flying in the CARLA window: arrows turn the view (left / right, up / down), F5 pause,
F6 / F7 one game hour back / forward, F8 fly to the next signalised junction, F9 hide or show the clock,
H open the traffic camera you are close to, full screen as a live street camera (cosim/cctv.py; H again
closes it), Esc quit (the time is saved).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import queue
import re
import logging
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import traci.constants as tc

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("freefly")
INSPECT_M = 25.0  # how close to a traffic camera "H inspect" is offered (cosim/inspector.py)
TURN_DEG_S, PITCH_DEG_S = 90.0, 60.0  # the arrow keys turn the view this fast
PEDESTRIAN_RADIUS_M = 110.0  # pedestrians are drawn this close to you (beyond, ~15 px tall full screen)
UI_RATE = 30.0  # the main loop (keys, sun, overlay); the poses have their own thread (Mover) at --rate
PREEMPT_MIN_GREEN_S = 5.0  # a car green cut for an emergency vehicle has had at least this (the program's minDur)
PEDESTRIANS_DRAWN = 28  # and at most this many, the nearest (demo, full screen: 47 drawn 28 FPS, 32 drawn 29.6)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="town05")
    ap.add_argument("--controller", default=None,
                    help="who runs the signals (default: coordinated on town05, else actuated): coordinated (one cycle "
                         "for every signal, green waves: cosim/coordination.py), actuated (SUMO's actuated program), "
                         "agents_observed (the junction "
                         "agents on top of it, from the cameras' reports; cosim/agents_actuated.py), agents (the same "
                         "on exact counts)")
    ap.add_argument("--agent-hud", action=argparse.BooleanOptionalAction, default=False,
                    help="with the agents: their decisions on the overlay, over each junction and top right "
                         "(cosim/agent_hud.py). Off by default since 24 Oct (too many notices); the accident markers stay")
    ap.add_argument("--goflow-priority", action="store_true",
                    help="with --controller agents / agents_observed: the GO-FLOW priority agent plans the emergency "
                         "vehicles' requests (cosim/goflow_priority.py, agents/priority_agent); off by default")
    ap.add_argument("--goflow-confirm", choices=("v2i", "camera"), default=None,
                    help="with --goflow-priority: a request counts on the vehicle's own radio request (v2i) or once a "
                         "camera has confirmed its siren (camera; default with agents_observed)")
    ap.add_argument("--radius", type=float, default=500.0, help="draw the vehicles within this many metres of you")
    ap.add_argument("--rate", type=float, default=60.0,
                    help="vehicle updates per second sent to CARLA (the picture's own rate is the GPU's)")
    ap.add_argument("--pedestrians-per-km", type=float, default=20.0,
                    help="pedestrians per km of pavement (cosim/extras.py; Town05 has 7 km): 0 for none. They walk "
                         "trips over the signalised crosswalks (tools/build_town05_peds.py). Walkers are costly to "
                         "draw (70 of them took the frame rate from 34 to 21, full screen): only those within "
                         "110 m are drawn")
    ap.add_argument("--ped-lights", action=argparse.BooleanOptionalAction, default=False,
                    help="draw a walk (green) / wait (red) lamp at each crosswalk end, as SUMO shows it "
                         "(cosim/ped_lights.py; off by default since 23 Oct: the lamps are debug points on the pole, "
                         "right behind CARLA's lower signal head, and in daylight their colour washed out to two "
                         "little white squares beside every pedestrian light). CARLA's own lamps on the poles "
                         "cannot be set from Python")
    ap.add_argument("--signal-heads", action=argparse.BooleanOptionalAction, default=False,
                    help="draw each head's direction arrows over the pole and switch CARLA's lamps off "
                         "(cosim/signal_heads.py, the 27 Oct look). Off by default since 29 Oct: CARLA's own lamps "
                         "light, in the lens, the colour SUMO gives the pole's approach; every approach of the town "
                         "now goes as one (cosim/signal_plan.py, SIGNAL_UNIT approach), so one colour per pole is "
                         "exactly what each driver may do, as a full-circle signal says in real life")
    ap.add_argument("--accidents", action=argparse.BooleanOptionalAction, default=True,
                    help="accidents happen (careless drivers, cosim/incidents.py), the police and recovery trucks "
                         "come (cosim/incident_scene.py); a marker on screen points to each (default on)")
    ap.add_argument("--drivers", action=argparse.BooleanOptionalAction, default=True,
                    help="a person at the wheel of the cars near the camera (cosim/drivers.py; --poses frame; "
                         "default on)")
    ap.add_argument("--plates", action=argparse.BooleanOptionalAction, default=True,
                    help="every car its own Tunisian number plates, front and back (cosim/plate_proc.py; default on)")
    ap.add_argument("--arrows", choices=["fly", "mouse", "proc", "tick", "mouse-loop"], default="fly",
                    help="how the arrow keys turn the view: mouse = CARLA's own mouse look, driven from a thread "
                         "(cosim/arrow_mouse.py, default); proc = set_transform from CARLA's tick callback in a "
                         "process of its own (cosim/arrow_proc.py); tick = the same in the viewer's process; "
                         "mouse-loop = the first mouse-look version, from the main loop")
    ap.add_argument("--poses", choices=["frame", "thread"], default="frame",
                    help="how the vehicles and pedestrians are moved: frame = once per CARLA picture, for the moment "
                         "it is shown, by a process of its own (cosim/pose_proc.py); thread = --rate times a second "
                         "from a thread of the viewer (freefly.Mover)")
    ap.add_argument("--measure-judder", action="store_true",
                    help="--poses thread: measure the judder too (the frame mode always does), for comparison")
    ap.add_argument("--cctv", choices=["native", "sensor"], default="native",
                    help="the traffic camera you open with H, full screen: native = CARLA's own view is put at the "
                         "camera, so the feed runs at CARLA's frame rate, and the street-camera labels and the "
                         "detector's boxes are drawn over it; sensor = the camera sensor's picture, copied and "
                         "drawn by the viewer (it changes only when a detection comes back)")
    ap.add_argument("--cctv-tick", type=float, default=None,
                    help="native camera: seconds between the detector sensor's pictures (default cosim/inspector.py)")
    ap.add_argument("--cctv-width", type=int, default=None,
                    help="native camera: the detector sensor's width in pixels (default cosim/inspector.py)")
    ap.add_argument("--cctv-fov", type=float, default=90.0,
                    help="native: CARLA's view's horizontal field of view, so the detector's boxes land on the vehicles")
    ap.add_argument("--windowed", action="store_true",
                    help="keep CARLA in its window (default: borderless full screen on its monitor)")
    ap.add_argument("--no-cameras", dest="cameras", action="store_false",
                    help="no traffic cameras / YOLO inspector (cosim/inspector.py)")
    ap.add_argument("--depots", action=argparse.BooleanOptionalAction, default=True,
                    help="Town05: a hospital, a police station and a recovery yard (cosim/depots.py) the ambulances, "
                         "police cars and recovery trucks leave from and go back to; --no-depots: they start on the "
                         "roads out of sight and cross the town (before 25 Oct)")
    ap.add_argument("--highway-cars", type=int, default=None,
                    help="cars driving round each loop of the ring road at the evening peak (cosim/highway.py, "
                         "default PEAK_PER_LOOP; 0: none, the town's trips only)")
    ap.add_argument("--highway-kmh", type=float, default=80.0, help="the ring road's limit (the town's: 50)")
    ap.add_argument("--mean-trip-s", type=float, default=140.0,
                    help="mean trip of the town (Town05, actuated, SUMO alone: 138 s) for the vehicle curve")
    ap.add_argument("--emergency-every", type=float, default=45.0,
                    help="mean seconds between emergency vehicles (ambulances and police, all on a run: beacons "
                         "on, above the limit, through red, and the signals ahead turn green; cosim/extras.py); 0: none")
    ap.add_argument("--no-bus-lines", dest="bus_lines", action="store_false",
                    help="no public transport (scenarios/<scenario>/<scenario>.pt.json, tools/build_town05_pt.py)")
    ap.add_argument("--clock-file", default=None, help="default out/<scenario>_clock.json")
    ap.add_argument("--set-time", default=None, help="HH:MM: start the clock here (the day is kept)")
    ap.add_argument("--start", default="08:00", help="HH:MM of day 1 when there is no saved clock")
    ap.add_argument("--fill-s", type=float, default=30.0, help="simulated seconds run before the first picture")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--sumo-port", type=int, default=8814)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", type=float, default=0.0,
                    help="unattended check: fly a scripted path for this many seconds (half high, half at street "
                         "level), write out/<scenario>_freefly_demo.json, quit")
    ap.add_argument("--demo-camera", type=float, default=0.0,
                    help="--demo: instead of the flight, open a traffic camera's native feed for this many seconds after "
                         "the warm-up, measure it (out/<scenario>_freefly_camera.json) and quit")
    ap.add_argument("--demo-incident", type=float, default=0.0,
                    help="--demo: instead of the flight, wait up to this many seconds for an accident, then watch it "
                         "from the road behind until it is cleared (screenshots at each stage, close-ups of number "
                         "plates first), out/<scenario>_freefly_incident.json")
    ap.add_argument("--demo-agents", type=float, default=0.0,
                    help="--demo: instead of the flight, for this many seconds go to junctions where the agents have just "
                         "decided something (emergency vehicles first) and take a screenshot of each "
                         "(out/freefly/<scenario>_agents_*.png, out/<scenario>_freefly_agents.json)")
    ap.add_argument("--demo-highway", type=int, default=0,
                    help="--demo: instead of the flight, this many views over the ring road (cosim/highway.py), a "
                         "screenshot of each (out/freefly/<scenario>_highway_*.png, out/<scenario>_freefly_highway.json)")
    ap.add_argument("--demo-signals", type=int, default=0,
                    help="--demo: instead of the flight, this many signalised junctions from 30 m up, each once some "
                         "poles are green (out/freefly/<scenario>_signals_*.png, out/<scenario>_freefly_signals.json)")
    ap.add_argument("--demo-signals-close", default=None, metavar="JUNCTION",
                    help="with --demo-signals: instead, each CARLA pole of this junction close up (8 m), once green "
                         "and once red, with its light boxes (heads) as CARLA reports them and the junction's SUMO "
                         "state (out/freefly/<scenario>_poles_*.png, out/<scenario>_freefly_poles.json)")
    ap.add_argument("--start-junction", default=None, metavar="JUNCTION",
                    help="start the view over this signalised junction (e.g. 1126), 22 m up at a corner, instead of the "
                         "chosen first view; fly on from there as usual")
    ap.add_argument("--demo-drivers", type=int, default=0,
                    help="--demo: instead of the flight, this many cars standing in traffic seen close from the "
                         "driver's side and the front (cosim/drivers.py), a screenshot of each "
                         "(out/freefly/<scenario>_drivers_*.png, out/<scenario>_freefly_drivers.json)")
    ap.add_argument("--demo-depots", type=int, default=0,
                    help="--demo: instead of the flight, this many rounds of the hospital, the police station and the "
                         "recovery yard (cosim/depots.py), a screenshot of each (out/freefly/<scenario>_depot_*.png, "
                         "out/<scenario>_freefly_depots.json)")
    ap.add_argument("--record-clips", type=float, default=0.0, metavar="SECONDS",
                    help="with --demo: record --record-cams to MP4 (cosim/clip_recorder.py) until there is a stretch "
                         "of normal traffic and a crash in a recorded picture, or this many seconds; then cut "
                         "normal_traffic.mp4 and crash.mp4 with their labels into out/incident_clips/<time>/")
    ap.add_argument("--record-cams", default="", help="--record-clips: camera names (layout.json), ';' between")
    ap.add_argument("--record-fps", type=float, default=10.0, help="--record-clips: pictures per second")
    ap.add_argument("--record-specs", default="", help="--record-clips: a JSON file (from the project folder) of "
                                                        "cameras of their own: name, tls, node, road_to, x, y, z, yaw, "
                                                        "pitch (CARLA)")
    ap.add_argument("--record-crash-match", default="",
                    help="--record-clips: only cameras whose name holds this text give the crash clip (\"overview\": "
                         "the city cameras of --record-specs)")
    ap.add_argument("--record-dynamic", type=int, default=2,
                    help="--record-clips: at most this many cameras started after a crash none of the recorded "
                         "ones sees (0: only the planned cameras)")
    ap.add_argument("--clock-speed", type=float, default=None,
                    help="game seconds per real second (default cosim/daily.GAME_SPEED, 60: a day in 24 min); "
                         "1 = real time, the light stays as it is during a recording")
    ap.add_argument("--max-incidents", type=int, default=None,
                    help="incidents open at a time at most (cosim/incidents.py MAX_ACTIVE, 3); the scenes' people, "
                         "police cars and recovery trucks are made ready for that many")
    ap.add_argument("--careless-share", type=float, default=None,
                    help="share of new cars whose drivers are careless (cosim/incidents.py CARELESS_SHARE, 0.10): "
                         "more, accidents come sooner")
    # 31 Oct: more kinds of incident (cosim/incidents.py, module doc)
    ap.add_argument("--breakdowns", action=argparse.BooleanOptionalAction, default=True,
                    help="now and then a car breaks down and stops in its lane, hazards on; the police and a recovery "
                         "truck come (default on)")
    ap.add_argument("--breakdown-every", type=float, default=None,
                    help="simulated seconds between breakdowns on average (cosim/incidents.py BREAKDOWN_MEAN_S, 2400)")
    ap.add_argument("--speeders", type=float, default=None,
                    help="share of new cars driven 25-50 %% over the limit, three times as often careless "
                         "(cosim/incidents.py SPEEDER_SHARE, 0.08; 0: none)")
    ap.add_argument("--pileups", action=argparse.BooleanOptionalAction, default=True,
                    help="on the ring road, cars behind a crash or a hard braking may react late and run into it "
                         "(default on)")
    ap.add_argument("--police-delay", default=None, metavar="MEDIAN[:SPREAD]",
                    help="seconds from a crash to the police car being sent: the median, and the spread (sigma of a "
                         "log-normal; 0: always the median). Default 240:0.5 (cosim/incidents.py POLICE_MEDIAN_S, "
                         "POLICE_SPREAD, 3 Nov); 6:0 is the response before 3 Nov (on scene 15-45 s after the crash)")
    ap.add_argument("--red-runners", type=float, default=None,
                    help="share of private cars whose drivers go on through the end of the amber when stopping would "
                         "mean braking hard, and so sometimes reach the line on red (cosim/red_runners.py "
                         "GAMBLER_SHARE, 0.10; 0: none)")
    ap.add_argument("--enforce", default=None,
                    help="SIGNAL:ROAD, e.g. 1292:6: a red-light enforcement camera on that road into that junction "
                         "(cosim/enforcement.py), its controller log and the simulation's own record of who entered "
                         "on red, written to out/enforcement/<site>_<time>/")
    ap.add_argument("--enforce-agent", action=argparse.BooleanOptionalAction, default=True,
                    help="with --enforce: the red-light agent watches the site's camera (cosim/redlight.py, in a "
                         "process of its own) and writes its cases to out/enforcement/cases/ for the police office "
                         "(tools/police_office.bat); the town's vehicle register to out/enforcement/register.json "
                         "(default on)")
    ap.add_argument("--enforce-record", type=float, default=0.0,
                    help="with --enforce: record the site's camera (video and the picture times) for this many "
                         "seconds, then quit (the data to build and grade the agent)")
    ap.add_argument("--ring-brake-every", type=float, default=None,
                    help="simulated seconds between hard brakings on the ring road on average, where pile-ups start "
                         "(cosim/incidents.py RING_BRAKE_MEAN_S, 400)")
    ap.add_argument("--lookalikes", action=argparse.BooleanOptionalAction, default=True,
                    help="things that look like incidents and are not: delivery vans stopping in their lane with "
                         "hazards on, buses standing long at a stop (logged, nobody responds; default on)")
    ap.add_argument("--demo-flight-only", action="store_true",
                    help="--demo: quit after the scripted flight (no crosswalk wait, close-ups or camera check)")
    ap.add_argument("--profile", action="store_true",
                    help="log, every 10 s, where each thread of the viewer spends its time (cosim/sampler.py)")
    args = ap.parse_args(argv)
    if args.controller is None:
        # Town05: since 27 Oct the coordinated signals (one cycle, green waves: cosim/coordination.py), when the signal
        # set has them; else the junction agents (they need Town05's cameras, out/town05_cameras)
        coord = ROOT / "scenarios" / args.scenario / f"{args.scenario}.coordinated.add.xml"
        if args.scenario == "town05":
            args.controller = "coordinated" if coord.is_file() else "agents_observed"
        else:
            args.controller = "actuated"
    if args.goflow_priority:
        base = re.split(r"[-+@]", args.controller, maxsplit=1)[0]
        if base in ("agents", "agents_observed"):
            args.controller = base + "_goflow" + args.controller[len(base):]
            os.environ["GOFLOW_PRIORITY_MODE"] = "live"  # the SUMO thread never waits for it
            if args.goflow_confirm:
                os.environ["GOFLOW_PRIORITY_CONFIRM"] = args.goflow_confirm
        elif not base.endswith("_goflow"):
            log.warning("--goflow-priority needs the junction agents (--controller agents or agents_observed); ignored")
    return args


def hhmm_to_s(text: str) -> float:
    h, m = (int(v) for v in text.split(":"))
    return h * 3600.0 + m * 60.0


def population_class():
    """Population (cosim/live.py) that starts new trips where the camera does not look: a vehicle
    inserted at the start of an edge in view popped up out of nowhere."""
    from cosim.live import Population

    class ViewAwarePopulation(Population):
        start_ok = None  # edge id -> True if a vehicle may appear at its start

        def __init__(self, conn, *a, **kw):
            super().__init__(conn, *a, **kw)
            remove = conn.vehicle.remove

            def unsubscribe_and_remove(vid, *ra, **rkw):
                # a removed vehicle still subscribed made traci print "Error! Vehicle ... is not known"
                # for a few steps; the subscription goes first
                try:
                    conn.vehicle.unsubscribe(vid)
                except Exception:  # noqa: BLE001 - not subscribed yet
                    pass
                return remove(vid, *ra, **rkw)

            conn.vehicle.remove = unsubscribe_and_remove

        def add(self, k: int, anywhere: bool = False) -> None:
            if self.start_ok is None:
                return super().add(k, anywhere)
            c = self.conn
            for _ in range(k):
                for _try in range(8):
                    t = self.rng.choices(self.types, self.weights)[0]
                    i = self.rng.randrange(len(self.route_ids[t]))
                    edges = self.pool.routes[t][i].split()
                    j = self.rng.randrange(max(1, len(edges) - 1)) if anywhere else 0
                    if self.start_ok(edges[j]):
                        break
                vid = f"live{self.n}"
                self.n += 1
                try:
                    if anywhere:
                        rid = f"livep{self.n}"
                        c.route.add(rid, edges[j:])
                        c.vehicle.add(vid, rid, typeID=self.typed(vid, t), depart="now", departLane="best",
                                      departPos="random_free", departSpeed="max")
                    else:
                        c.vehicle.add(vid, self.route_ids[t][i], typeID=self.typed(vid, t), depart="now",
                                      departLane="best", departSpeed="max")
                    self.added += 1
                except Exception as e:  # noqa: BLE001 - one bad insertion must not stop the town
                    log.debug("add %s failed: %s", vid, e)

    return ViewAwarePopulation


def pooled_carla_side():
    """CarlaSide that parks the vehicles that leave out of sight and hands them to the next vehicle
    of the same class, instead of destroying them and spawning new ones. Spawning an actor stalls
    the server for a frame or more; at the evening rush on Town05 a vehicle enters about every
    second (measured, first free-fly demo: 1 % of frames over 50 ms when flying high)."""
    import carla

    from cosim.carla_side import CarlaSide

    park = carla.Transform(carla.Location(0.0, 0.0, -1000.0))

    class PooledCarlaSide(CarlaSide):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # --poses frame: another process moves the actors (cosim/pose_proc.py), parking included;
            # this side only lists who is to be parked
            self.external_moves = False
            self.parked_v: list[int] = []
            self.parked_w: list[int] = []
            # The pool is kept per CARLA model (cosim/vmodels.py: each SUMO vehicle is sized as the model that draws
            # it); a key that is a vehicle class (no model known) is drawn with blueprints.choose as before.
            self.idle: dict[str, list[int]] = {}   # model (or vclass) -> parked actor ids
            self.vclass_of: dict[str, str] = {}    # SUMO id -> pool key of its actor
            self.reused = 0
            self.new_cars: list[tuple[int, str]] = []  # (actor, model) made since asked: they need plates

        def spawn(self, new):
            batch, rest = [], []
            for vid, vclass, tf in new:
                pool = self.idle.get(vclass)
                if pool:
                    aid = pool.pop()
                    self.actors[vid] = aid
                    self.vclass_of[vid] = vclass
                    self.lights.pop(vid, None)
                    batch.append(carla.command.ApplyTransform(aid, tf))
                    self.reused += 1
                else:
                    rest.append((vid, vclass, tf))
            if batch:
                self.client.apply_batch(batch)
            if rest:  # more vehicles of a class than parked ones: spawned without holding the picture up
                if not hasattr(self, "_vq"):
                    self._vq: queue.Queue = queue.Queue()
                    self._vpending: set[str] = set()
                    threading.Thread(target=self._vehicle_spawner, daemon=True).start()
                rest = [r for r in rest if r[0] not in self._vpending]
                self._vpending.update(r[0] for r in rest)
                self.short = getattr(self, "short", 0) + len(rest)
                if rest:
                    self._vq.put(rest)

        def _vehicle_spawner(self):
            """Spawns parked vehicles for the classes that ran short, in a thread; the main thread picks
            them up from the pool on a later step."""
            from cosim.blueprints import choose

            while True:
                items = self._vq.get()
                if items is None:
                    return
                batch, keys = [], []
                for vid, vclass, _ in items:
                    bp = self.library.find(vclass if vclass.startswith("vehicle.") else choose(self.cands, vclass, vid)[0])
                    if bp.has_attribute("color"):
                        cols = bp.get_attribute("color").recommended_values
                        if cols:
                            bp.set_attribute("color", cols[(len(batch) * 7 + len(vid)) % len(cols)])
                    loc = carla.Location(-2000.0 - 12.0 * len(batch), 2000.0, -500.0)
                    batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)).then(
                        carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
                    keys.append(vclass)
                t = time.perf_counter()
                try:
                    for vc, r in zip(keys, self.client.apply_batch_sync(batch, False), strict=True):
                        if not r.error:
                            self.idle.setdefault(vc, []).append(r.actor_id)
                            if vc.startswith("vehicle."):
                                self.new_cars.append((r.actor_id, vc))
                except Exception as e:  # noqa: BLE001
                    log.debug("vehicle spawn: %s", e)
                log.info("spawned %d more parked vehicles (%s) in %.1f s", len(batch), ", ".join(sorted(set(keys))),
                         time.perf_counter() - t)
                self._vpending.difference_update(vid for vid, _, _ in items)

        def destroy(self, vids):
            batch = []
            for v in vids:
                aid = self.actors.pop(v, None)
                if aid is None:
                    continue
                self.lights.pop(v, None)
                self.idle.setdefault(self.vclass_of.pop(v, "passenger"), []).append(aid)
                if self.external_moves:
                    self.parked_v.append(aid)
                else:
                    batch.append(carla.command.ApplyTransform(aid, park))
                batch.append(carla.command.SetVehicleLightState(aid, carla.VehicleLightState.NONE))
            if batch:
                self.client.apply_batch(batch)

        # -- pedestrians: a pool of walkers spawned once while loading, parked and reused. Spawning a
        # walker held the caller ~2.5 s per call (measured: 8.3 s of every 10 s of the main loop for 3
        # walkers), and doing it from a second thread left a vehicle spawn waiting 83 s behind it.
        def prewarm_vehicles(self, counts: dict[str, int]) -> int:
            """Parked vehicles spawned while loading, `counts` per vehicle class, every model a class may use
            among them, handed out as vehicles enter (spawn). A spawn while the picture runs stalled
            everything: the first police car 12.6 s, the first bus 4.7 s, and later buses and ambulances
            77-98 s, model already loaded or not (measured, 19 Oct)."""
            from cosim.blueprints import choose

            batch, keys = [], []
            for vc, n in counts.items():
                models = ([vc] if vc.startswith("vehicle.") else
                          sorted({choose(self.cands, vc, f"warm{k}")[0] for k in range(400)}))
                for j in range(max(n, len(models))):
                    i = len(batch)
                    loc = carla.Location(2000.0 + 12.0 * (i % 25), 2000.0 + 12.0 * (i // 25), -500.0)
                    bp = self.library.find(models[j % len(models)])
                    if bp.has_attribute("color"):
                        cols = bp.get_attribute("color").recommended_values
                        if cols:
                            bp.set_attribute("color", cols[(j * 7) % len(cols)])
                    batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)).then(
                        carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
                    keys.append(vc)
            n = 0
            for vc, r in zip(keys, self.client.apply_batch_sync(batch, False), strict=True):
                if not r.error:
                    self.idle.setdefault(vc, []).append(r.actor_id)
                    if vc.startswith("vehicle."):
                        self.new_cars.append((r.actor_id, vc))
                    n += 1
            self.short = 0
            return n

        def prespawn_scene_walkers(self, scenes: int = 2) -> dict[str, list[int]]:
            """The people an accident scene needs (cosim/incident_scene.py): police officers, the recovery trucks'
            crew, drivers; parked out of sight, physics off (posed by the pose process). `scenes`: accidents open at a
            time (one officer, one of the crew, three drivers each)."""
            from cosim.incident_scene import CIVIL_BP, POLICE_BP, WORKER_BP

            out: dict[str, list[int]] = {}
            batch, roles = [], []
            for role, bps, n in (("police", POLICE_BP, scenes), ("worker", WORKER_BP, scenes),
                                 ("civil", CIVIL_BP, 3 * scenes)):
                for i in range(n):
                    try:
                        bp = self.library.find(bps[i % len(bps)])
                    except Exception:  # noqa: BLE001 - not in this CARLA
                        continue
                    if bp.has_attribute("is_invincible"):  # never knocked over (a car's door, a wreck beside them)
                        bp.set_attribute("is_invincible", "true")
                    loc = carla.Location(-2100.0 - 3.0 * len(batch), -2100.0, -500.0)
                    batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)).then(
                        carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
                    roles.append(role)
            for role, r in zip(roles, self.client.apply_batch_sync(batch, False), strict=True):
                if not r.error:
                    out.setdefault(role, []).append(r.actor_id)
            return out

        def prespawn_drivers(self, n: int) -> list[int]:
            """Walkers for the drivers (cosim/drivers.py): grown-ups in everyday clothes (not the police or the
            recovery crew's), parked out of sight, physics off; seated and posed by the pose process."""
            from cosim.incident_scene import POLICE_BP, WORKER_BP

            bps = []
            for bp in sorted(self.library.filter("walker.pedestrian.*"), key=lambda b: b.id):
                if bp.id in POLICE_BP or bp.id in WORKER_BP:
                    continue
                if bp.has_attribute("age") and bp.get_attribute("age").as_str() != "adult":
                    continue
                bps.append(bp)
            batch = []
            for i in range(n if bps else 0):
                bp = bps[(i * 7) % len(bps)]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "true")
                loc = carla.Location(-2200.0 - 3.0 * i, -2200.0, -500.0)
                batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)).then(
                    carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
            return [r.actor_id for r in self.client.apply_batch_sync(batch, False) if not r.error]

        def trim(self, keep: int = 3) -> int:
            """Destroys parked vehicles beyond `keep` per class."""
            batch = []
            for vc, pool in self.idle.items():
                while len(pool) > keep:
                    batch.append(carla.command.DestroyActor(pool.pop(0)))
            if batch:
                self.client.apply_batch(batch)
            return len(batch)

        def prespawn_walkers(self, n: int) -> int:
            if not self._walker_bps:
                self._walker_bps = sorted(self.library.filter("walker.pedestrian.*"), key=lambda b: b.id)
            batch = []
            for i in range(n):
                bp = self._walker_bps[i % len(self._walker_bps)]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                loc = carla.Location(-2000.0 + 3.0 * (i % 40), -2000.0 + 3.0 * (i // 40), -500.0)
                # no physics: the pose comes from SUMO every frame, and parked walkers kept falling
                batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)).then(
                    carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
            self.idle_walkers = [r.actor_id for r in self.client.apply_batch_sync(batch, False) if not r.error]
            self.walkers_short = 0
            return len(self.idle_walkers)

        def spawn_walkers(self, new):
            batch = []
            for pid, tf in new:
                if not getattr(self, "idle_walkers", None):
                    self.walkers_short = getattr(self, "walkers_short", 0) + 1  # more pedestrians than walkers
                    continue
                aid = self.idle_walkers.pop()
                self.walkers[pid] = aid
                if not self.external_moves:  # else the pose process puts it there with its step's poses
                    batch.append(carla.command.ApplyWalkerState(aid, tf, 0.0))
            if batch:
                self.client.apply_batch(batch)

        def destroy_walkers(self, pids):
            batch = []
            for p in pids:
                aid = self.walkers.pop(p, None)
                if aid is None:
                    continue
                if not hasattr(self, "idle_walkers"):
                    self.idle_walkers = []
                self.idle_walkers.append(aid)
                if self.external_moves:
                    self.parked_w.append(aid)
                else:
                    batch.append(carla.command.ApplyWalkerState(aid, park, 0.0))
            if batch:
                self.client.apply_batch(batch)

        def close(self):
            ids = ([a for pool in self.idle.values() for a in pool] + list(getattr(self, "idle_walkers", []))
                   + list(self.walkers.values()) + list(self.actors.values()))
            self.idle, self.idle_walkers, self.walkers, self.actors = {}, [], {}, {}
            if ids:
                self.client.apply_batch([carla.command.DestroyActor(a) for a in ids])
            super().close()

    return PooledCarlaSide


class BusLines:
    """Public transport: buses running the closed loops of scenarios/<name>/<name>.pt.json
    (tools/build_town05_pt.py), stopping DWELL_S at each SUMO busStop, lap after lap: near the end
    of a lap the next one is appended to the route with its stops. Two buses a line from 06:00 to
    22:00, one at night, spaced half a lap apart. The random bus trips of the demand are left out."""

    DWELL_S = 20.0
    SPEED_MS = 9.0  # rough average with junctions, to space the buses

    def __init__(self, conn, folder: Path, name: str, bus_type: str, type_for=None):
        """type_for(vid, type) -> the type a bus is inserted with (cosim/vmodels.Typer: sized as its CARLA model)."""
        self.type_for = type_for
        f = folder / f"{name}.pt.json"
        self.lines = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        self.conn, self.type = conn, bus_type
        self.n = 0
        self.buses: dict[str, list[str]] = {line: [] for line in self.lines}
        self.next_t = {line: 0.0 for line in self.lines}
        for line, d in self.lines.items():
            conn.route.add(f"pt_{line}", d["edges"])

    @staticmethod
    def additional(folder: Path, name: str) -> list[str]:
        f = folder / f"{name}.pt.add.xml"
        return [str(f)] if f.is_file() else []

    @staticmethod
    def wanted(hour: float) -> int:
        return 2 if 6.0 <= hour % 24.0 < 22.0 else 1

    def lap_s(self, line: str) -> float:
        d = self.lines[line]
        return d["length_m"] / self.SPEED_MS + len(d["stops"]) * self.DWELL_S

    def _add_stops(self, vid: str, line: str, skip_edge: str | None = None) -> None:
        for st in self.lines[line]["stops"]:
            if st["edge"] == skip_edge:
                continue
            try:
                self.conn.vehicle.setBusStop(vid, st["id"], duration=self.DWELL_S)
            except Exception as e:  # noqa: BLE001 - a stop already passed: the next lap has it
                log.debug("stop %s for %s: %s", st["id"], vid, e)

    def step(self, now: float, hour: float) -> None:
        c = self.conn
        alive = set(c.vehicle.getIDList()) | set(c.simulation.getPendingVehicles())
        for line in self.lines:
            self.buses[line] = [v for v in self.buses[line] if v in alive]
            want = self.wanted(hour)
            if len(self.buses[line]) < want and now >= self.next_t[line]:
                vid = f"pt_{line}_{self.n}"
                self.n += 1
                try:
                    c.vehicle.add(vid, f"pt_{line}", depart="now", departLane="best",
                                  typeID=self.type_for(vid, self.type) if self.type_for else self.type)
                    self._add_stops(vid, line)
                    self.buses[line].append(vid)
                except Exception as e:  # noqa: BLE001 - try again later
                    log.debug("bus %s not added: %s", vid, e)
                self.next_t[line] = now + self.lap_s(line) / want

    def is_bus(self, vid: str) -> bool:
        return vid.startswith("pt_")

    def next_lap(self, vid: str, idx: int, route: list[str]) -> None:
        line = vid.split("_")[1]
        loop = self.lines[line]["edges"]
        try:
            self.conn.vehicle.setRoute(vid, list(route[idx:]) + loop[1:])
        except Exception as e:  # noqa: BLE001
            log.debug("next lap for %s: %s", vid, e)
            return
        self._add_stops(vid, line, skip_edge=loop[0])


@dataclass
class Packet:
    """One SUMO step, ready to show: produced by the SUMO thread, shown by the main thread."""
    cur: object            # StepResult, only what is near you
    arrived: list
    poser: object          # Poser holding this step's start and end poses
    lights: dict
    walk: dict             # person id -> (start transform, end transform, speed)
    incidents: list = dataclasses.field(default_factory=list)  # IncidentManager.snapshot() of this step


class SumoThread(threading.Thread):
    """Runs SUMO one or two steps ahead of the picture (a queue of 2): the main thread only
    interpolates and moves actors, at an even rate, and is never held up by a SUMO step."""

    def __init__(self, ff: "FreeFly"):
        super().__init__(daemon=True, name="sumo")
        self.ff = ff
        self.q: queue.Queue = queue.Queue(maxsize=2)
        self.stop = False
        self.error: str | None = None

    def run(self) -> None:
        while not self.stop:
            try:
                pk = self.ff.produce()
            except Exception:  # noqa: BLE001 - shown in the clock and logged
                self.error = traceback.format_exc()
                log.error("SUMO thread failed:\n%s", self.error)
                return
            while not self.stop:
                try:
                    self.q.put(pk, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def halt(self) -> None:
        self.stop = True
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.join(timeout=5.0)


class Mover(threading.Thread):
    """Sends the vehicles' and pedestrians' poses to CARLA at an even `rate`, in a thread of its own.

    They used to be sent from the main loop, between the clock overlay (20-77 ms a redraw, 5 times a
    second), the camera markers and CARLA round trips: the log showed 13-47 updates a second while CARLA
    drew 45-118 frames, so cars moved in visible jerks. Here nothing else runs in between; the main loop
    only handles keys, the sun and the overlay. Also takes the next SUMO step from the SUMO thread when its
    time comes (FreeFly.apply). `lock` is held while CARLA's actors are changed, for others that touch
    them (CarlaSide.trim)."""

    def __init__(self, ff: "FreeFly", worker: "SumoThread", clock, rate: float, pk: "Packet", demo: bool):
        super().__init__(daemon=True, name="mover")
        self.ff, self.worker, self.clock = ff, worker, clock
        self.period = 1.0 / max(rate, 1.0)
        self.pk = pk
        self.demo = demo
        self.stop = False
        self.lock = threading.Lock()
        self.updates = 0
        self.held = 0                    # SUMO was late: the last pose was held
        self.timing = {"move": 0.0, "apply": 0.0}
        self.steep = 0                   # poses with a pitch over 30 deg (a car tipping over)
        self.reversals = 0               # --demo: the turn reversing between two updates
        self.gaps: list[float] = []      # --demo: time between two updates
        self._yaw_last: dict = {}
        self.error: str | None = None

    def run(self) -> None:
        step = self.ff.sumo.step_length
        t_in, last = 0.0, time.perf_counter()
        due = last
        while not self.stop:
            now = time.perf_counter()
            dt = min(now - last, 0.25)
            last = now
            try:
                with self.lock:
                    if not self.clock.paused:
                        t_in = self._advance(t_in + dt, step)
                        tm = time.perf_counter()
                        poses = self.ff.move(self.pk, min(t_in / step, 1.0))
                        self.timing["move"] += time.perf_counter() - tm
                        self._measure(poses, dt)
                self.updates += 1
            except Exception:  # noqa: BLE001 - logged once, the thread goes on
                if self.error is None:
                    self.error = traceback.format_exc()
                    log.error("pose thread:\n%s", self.error)
            due += self.period
            wait = due - time.perf_counter()
            if wait < -self.period:  # far behind (a long apply): start again from now, no burst
                due, wait = time.perf_counter(), 0.0
            if wait > 0:
                time.sleep(wait)  # Python 3.11+ on Windows: a high-resolution timer, ~1 ms

    def _advance(self, t_in: float, step: float) -> float:
        while t_in >= step:  # the next SUMO step, already worked out by the SUMO thread
            try:
                nxt = self.worker.q.get_nowait()
            except queue.Empty:
                self.held += 1
                return step  # SUMO is late: hold the last pose rather than jump
            t_in -= step
            self.pk = nxt
            tm = time.perf_counter()
            self.ff.apply(nxt)
            self.timing["apply"] += time.perf_counter() - tm
            mq = getattr(self, "measure_q", None)
            if mq is not None:  # --measure-judder: who moves, for cosim/pose_proc.py's measurement
                veh = [(aid, vid) + (None,) * 10 for vid, aid in self.ff.cs.actors.items()]
                mq.put(("seg", {"start": time.time(), "dt": step, "veh": veh, "walk": []}))
        return t_in

    def _measure(self, poses: dict, dt: float) -> None:
        self.steep += sum(1 for tf in poses.values() if abs(tf.rotation.pitch) > 30.0)
        if not self.demo:
            return
        self.gaps.append(dt)
        for vid, tf in poses.items():
            y0, d0 = self._yaw_last.get(vid, (None, 0.0))
            d = 0.0 if y0 is None else (tf.rotation.yaw - y0 + 180.0) % 360.0 - 180.0
            if abs(d) > 0.5 and abs(d0) > 0.5 and d * d0 < 0:
                self.reversals += 1
            self._yaw_last[vid] = (tf.rotation.yaw, d)

    def take(self) -> tuple[int, dict]:
        """Updates and timing since the last call (for the 10 s log line)."""
        n, t = self.updates, self.timing
        self.updates, self.timing = 0, {"move": 0.0, "apply": 0.0}
        return n, t


class FrameFeeder(threading.Thread):
    """--poses frame: the poses are worked out once per CARLA picture by a process of its own
    (cosim/pose_proc.py, as the free-fly camera's are); this thread only starts each SUMO step on time
    (FreeFly.apply: actors spawned, reused or parked, lights) and sends it to that process as a segment:
    when it starts, each vehicle's curve points and heights, each pedestrian's start and end, the actors to
    park. Pause freezes the process's poses where they are."""

    def __init__(self, ff: "FreeFly", worker: "SumoThread", clock, pk: "Packet", q):
        super().__init__(daemon=True, name="feeder")
        self.ff, self.worker, self.clock, self.pk, self.q = ff, worker, clock, pk, q
        self.stop = False
        self.lock = threading.Lock()  # held while CARLA's actors are changed (CarlaSide.trim takes it too)
        self.held = 0
        self.segments = 0
        self.timing = {"move": 0.0, "apply": 0.0}
        self.steep = self.reversals = 0  # measured by the pose process in this mode
        self.gaps: list[float] = []
        self.error: str | None = None

    def segment(self, pk: "Packet", start: float) -> dict:
        cs, poser = self.ff.cs, pk.poser
        prev = poser.prev.vehicles if poser.prev else {}
        pre, nxt = poser.pre or {}, poser.nxt or {}
        veh = []
        empty = []  # no one at the wheel: wrecks (their drivers got out), vehicles parked at their depot
        scene = getattr(self.ff, "scene", None)
        dep = getattr(self.ff, "depots", None)
        at_depot = {pv for d in dep.depots.values() for pv in d.parked.values()} if dep is not None else set()
        for vid, v in poser.cur.vehicles.items():
            aid = cs.actors.get(vid)
            if aid is None:
                continue
            (z0, pt0), (z1, pt1) = poser._from[vid], poser._g[vid]
            p = prev.get(vid)
            # the lane's direction and the speed at both ends of the step (the body kept along its lane,
            # interp.Trails), and a wreck's damage (pose_proc.lay_wreck)
            dmg = scene.damage_for(vid) if scene is not None else None
            if dmg or vid in at_depot:
                empty.append(aid)
            if p is None:
                veh.append((aid, vid, None, None, (v.x, v.y), None, v.length, v.angle, z0, z1, pt0, pt1,
                            None, getattr(v, "lane_angle", None), None, getattr(v, "speed", None), dmg))
                continue
            p1, p2 = (p.x, p.y), (v.x, v.y)
            q0, q3 = pre.get(vid), nxt.get(vid)
            p0 = (q0.x, q0.y) if q0 is not None else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
            p3 = (q3.x, q3.y) if q3 is not None else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])
            veh.append((aid, vid, p0, p1, p2, p3, v.length, v.angle, z0, z1, pt0, pt1,
                        getattr(p, "lane_angle", None), getattr(v, "lane_angle", None), getattr(p, "speed", None),
                        getattr(v, "speed", None), dmg))

        def xyzw(tf):
            return (tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw)

        walk = [(cs.walkers[pid], xyzw(a), xyzw(b), speed) for pid, (a, b, speed) in pk.walk.items()
                if pid in cs.walkers]
        park_v, park_w = cs.parked_v, cs.parked_w
        cs.parked_v, cs.parked_w = [], []
        return {"start": start, "dt": self.ff.sumo.step_length, "veh": veh, "walk": walk,
                "park_v": park_v, "park_w": park_w, "empty": empty}

    LEAD_S = 0.2  # each step goes to the pose process this long before it starts

    def send(self, pk: "Packet", start: float, now: bool = False) -> dict:
        """now: take over at once (the first step); else at `start`. Returns the segment sent."""
        seg = self.segment(pk, start)
        # the lights that changed go with the poses, in the pose process's one call a picture: a call of their own
        # each step put the poses a picture late (the server handles about one batch a picture)
        seg["lights"] = self.ff.cs.light_changes(pk.lights)
        t_k = getattr(getattr(pk, "cur", None), "time", None)
        if t_k is not None:  # (31 Oct) the time each picture shows: FreeFly.depicted_at
            seg["t"] = t_k
            if getattr(self.ff, "shown_log", None) is not None:
                self.ff.shown_log.append((start, t_k))
        self.q.put(("seg", seg, now))
        self.segments += 1
        return seg

    def resend(self, seg: dict, start: float, now: bool) -> None:
        """A segment sent before, again with a new start (after a pause; its parks went the first time)."""
        if "t" in seg and getattr(self.ff, "shown_log", None) is not None:
            self.ff.shown_log.append((start, seg["t"]))
        self.q.put(("seg", dict(seg, start=start, park_v=[], park_w=[]), now))

    def run(self) -> None:
        step = self.ff.sumo.step_length
        start = time.time()
        segs: list[dict] = []  # the last two segments sent: the one shown, and maybe the next, sent ahead
        paused = None          # (segment shown, its fraction, the next one or None)
        late = False
        try:
            with self.lock:
                segs = [self.send(self.pk, start, now=True)]
        except Exception:  # noqa: BLE001
            self.error = traceback.format_exc()
            log.error("pose feeder:\n%s", self.error)
        while not self.stop:
            try:
                now = time.time()
                if self.clock.paused:
                    if paused is None and segs:
                        cur = [g for g in segs if g["start"] <= now]
                        shown = cur[-1] if cur else segs[0]
                        ahead = segs[-1] if segs[-1] is not shown else None
                        frac = min(max((now - shown["start"]) / step, 0.0), 1.0)
                        paused = (shown, frac, ahead)
                        self.q.put(("pause", frac))
                    time.sleep(0.01)
                    continue
                if paused is not None:  # on again from where it stopped
                    shown, frac, ahead = paused
                    paused = None
                    t0 = now - frac * step
                    self.resend(shown, t0, now=True)
                    segs = [dict(shown, start=t0)]
                    start = t0
                    if ahead is not None:
                        self.resend(ahead, t0 + step, now=False)
                        segs.append(dict(ahead, start=t0 + step))
                        start = t0 + step
                if now >= start + step - self.LEAD_S:  # the next step, sent ahead of its start
                    try:
                        nxt = self.worker.q.get_nowait()
                    except queue.Empty:
                        if not late and now >= start + step:
                            self.held += 1  # SUMO is late: the pose process holds the end of the step
                            late = True
                        time.sleep(0.002)
                        continue
                    late = False
                    with self.lock:
                        tm = time.perf_counter()
                        self.ff.apply(nxt)
                        self.timing["apply"] += time.perf_counter() - tm
                        # on time: the next step starts where this one ends (even motion); late: from now
                        start = start + step if time.time() - (start + step) < 0.25 * step else time.time()
                        self.pk = nxt
                        segs = (segs + [self.send(nxt, start)])[-2:]
                time.sleep(0.002)
            except Exception:  # noqa: BLE001 - logged once, the thread goes on
                if self.error is None:
                    self.error = traceback.format_exc()
                    log.error("pose feeder:\n%s", self.error)
                time.sleep(0.05)

    def take(self) -> tuple[int, dict]:
        n, t = self.segments, self.timing
        self.segments, self.timing = 0, {"move": 0.0, "apply": 0.0}
        return n, t


class FreeFly:
    def __init__(self, args, clock):
        self.args = args
        self.clock = clock
        self.stage, self.progress, self.error = "Starting", -1.0, None
        self.ready = False
        self.city_vehicles = 0
        self.target = 0.0
        self.night = False
        self.prev = None
        self.lights = {}
        self.carla_dt: list[float] = []     # every CARLA frame time (async: real seconds)
        self._dt_seen = 0
        self.junction_i = -1
        self._view = None                   # camera (x, y, z, forward x, forward y), CARLA coordinates
        self.extended = 0                   # trips given a new destination so they don't end in view
        self.vanished_in_view = 0           # drawn vehicles that ended their trip in view (should stay ~0)
        self.emergency_seen: set[str] = set()
        self.buses_seen: set[str] = set()
        self._k = 0
        self.drawn: set[str] = set()        # vehicles CARLA draws (a copy the SUMO thread may read)
        self.preempted = 0                  # signals turned towards an emergency vehicle's green
        self.walkers_max = 0
        self.walks_extended = 0
        self.shown_tls: dict = {}
        self.ped_lights = None
        self.signal_heads = None
        self.first_tf = None
        self.incidents = self.scene = None
        self.models: dict = {}

    # ------------------------------------------------------------ loading (a thread)
    def load(self) -> None:
        try:
            self._load()
            self.ready = True
        except Exception:  # noqa: BLE001 - shown in the clock window and logged
            self.error = traceback.format_exc()
            log.error("loading failed:\n%s", self.error)

    def _load(self) -> None:
        import carla

        from cosim import daily
        from cosim import scenario as scenario_mod
        from cosim.run import Poser, ground_z
        from cosim.sumo_side import SumoSide

        a = self.args
        self.stage = "Reading the road network"
        scen = scenario_mod.load(a.scenario, ROOT)
        self.scen = scen
        self.centres = scenario_mod.tls_centres(scen.net_file, scen.frame)

        self.stage = "Connecting to CARLA"
        client = carla.Client(a.host, a.port)
        client.set_timeout(120.0)
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != scen.carla_map:
            self.stage = f"Loading {scen.carla_map}"
            world = client.load_world(scen.carla_map)
        self.client, self.world = client, world
        self.cmap = world.get_map()
        from cosim import vmodels
        from cosim.blueprints import candidates_from_library

        self.models = vmodels.load()
        self.cands = candidates_from_library(world.get_blueprint_library())
        self.zs = {t: ground_z(self.cmap, x, y) for t, (x, y, _) in self.centres.items()}
        landmark_ids = [lm.id for lm in self.cmap.get_all_landmarks_of_type("1000001")]

        self.stage = "Starting SUMO"
        pool = daily.load_pool(scen.folder)
        # every vehicle type: no sideways slide while standing, sideways at most 0.3 x the forward speed, 1 m beside a
        # neighbour (SUMO's sublane model, below); the emergency types sized as their CARLA models (cosim/vmodels.py)
        pool.vtypes = {t: vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS) for t, vt in pool.vtypes.items()}
        extras_types = vmodels.extras_types_file(self.models, ROOT / "out" / "freefly" / "extras_types.add.xml",
                                                 vmodels.LANE_CHANGE_ATTRS)
        pool_file = ROOT / "out" / "freefly" / f"{a.scenario}.pool.rou.xml"
        route_ids = daily.write_pool_routes(pool, pool_file)
        curve = daily.population_curve(None, pool, mean_trip_min=a.mean_trip_s / 60.0)
        # Lane changes: SUMO's sublane model (--lateral-resolution 0.8), every car sideways at most 0.3 x its forward
        # speed and never while standing (vmodels.LANE_CHANGE_ATTRS). The continuous lane change used before
        # (--lanechange.duration 2) slid cars standing at a stop line 3.5 m sideways, and the bodies stood diagonally
        # across two lanes: 100 sideways slides and 16 cars inside another in 120 s of rush hour, now 12 and 4 (the
        # deepest 0.58 m, briefly, in a junction; tools/check_overlaps.py --variant viewer). Collisions are reported,
        # not acted on by SUMO (cosim/incidents.py makes accidents of them).
        from cosim.lanegeo import LaneGeometry

        self.sumo = SumoSide(scen.folder, a.controller, seed=a.seed, port=a.sumo_port, label="freefly",
                             extras=a.emergency_every > 0 or a.pedestrians_per_km > 0,
                             emergency_every=a.emergency_every or 1e9,
                             additional=BusLines.additional(scen.folder, a.scenario) if a.bus_lines else [],
                             extra_args=["--route-files", str(pool_file), "--end", "100000000",
                                         "--lateral-resolution", "0.8", "--collision.action", "warn",
                                         "--collision.check-junctions", "true", "--collision.mingap-factor", "0"],
                             vehicle_vars=[tc.VAR_ROUTE_INDEX], person_vars=[tc.VAR_STAGES_REMAINING],
                             extras_types=extras_types, lane_geometry=LaneGeometry(scen.net_file))
        if self.sumo.extras is not None:
            self.sumo.extras.config.pedestrians_per_km = a.pedestrians_per_km
            self.sumo.extras.config.on_run_share = 1.0  # every emergency vehicle is on a run, beacons on
            # pedestrians walk routed trips over the crosswalks, and appear where you are not looking
            self.sumo.extras.config.pedestrian_trips = True
            self.sumo.extras.config.near_share = 0.75  # most start within 260 m of you (out of sight)
            self.sumo.extras.spawn_ok = lambda edge, pos: not self.in_view(
                *self.scen.frame.to_carla_xy(*self.sumo.conn.simulation.convert2D(edge, pos)))
        self.pop = population_class()(self.sumo.conn, pool, route_ids, curve, seed=a.seed)
        base = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
        base.update(vmodels.EMERGENCY_TYPES)
        self.typer = vmodels.Typer(self.sumo.conn, self.cands, self.models, base)
        self.pop.type_for = self.typer.type_for
        bus_types = [t for t, vt in pool.vtypes.items() if 'vClass="bus"' in vt]
        self.bus = None
        if a.bus_lines and bus_types and BusLines.additional(scen.folder, a.scenario):
            self.bus = BusLines(self.sumo.conn, scen.folder, a.scenario, bus_types[0], type_for=self.typer.type_for)
            for i, t in enumerate(self.pop.types):  # buses run the lines, not random trips
                if t in bus_types:
                    self.pop.weights[i] = 0
        import random

        import sumolib

        net = sumolib.net.readNet(str(scen.net_file))
        self.edge_start = {}
        for e in net.getEdges():
            x, y = e.getLanes()[0].getShape()[0]
            self.edge_start[e.getID()] = scen.frame.to_carla_xy(x, y)
        self.sinks = sorted({r.split()[-1] for routes in pool.routes.values() for r in routes})
        self.rng = random.Random(a.seed + 1)
        self.agents = None
        self.agents_xy: dict = {}
        self.sim_now = 0.0
        self.shown_log: collections.deque = collections.deque(maxlen=64)  # (segment start, its SUMO time): depicted_at
        self.incidents = None
        self._wire_agents(a)
        if a.accidents:
            from cosim import incidents as incidents_mod
            from cosim.incidents import IncidentManager

            frame_ = scen.frame
            self.incidents = IncidentManager(self.sumo.conn, scen.net_file, self.typer, seed=a.seed + 2,
                                             walk=self.sumo.walk, sinks=self.sinks,
                                             visible=lambda x, y: self.in_view(*frame_.to_carla_xy(x, y)),
                                             breakdown_mean_s=(0.0 if not a.breakdowns else
                                                               a.breakdown_every or incidents_mod.BREAKDOWN_MEAN_S),
                                             speeder_share=(incidents_mod.SPEEDER_SHARE if a.speeders is None
                                                            else a.speeders),
                                             pileups=a.pileups, lookalikes=a.lookalikes,
                                             ring_brake_mean_s=(a.ring_brake_every or incidents_mod.RING_BRAKE_MEAN_S),
                                             **incidents_mod.police_delay_args(a.police_delay),
                                             **({} if a.max_incidents is None else {"max_active": a.max_incidents}),
                                             **({} if a.careless_share is None else {"share": a.careless_share}))
            self.pop.type_for = self.incidents.type_for
        self.pop.start_ok = lambda edge: not self.in_view(*self.edge_start.get(edge, (1e9, 1e9)))
        # (31 Oct) drivers who go on through the end of the amber (cosim/red_runners.py), and a red-light camera
        self.red_runners = None
        from cosim import red_runners as rr_mod

        rr_share = rr_mod.GAMBLER_SHARE if a.red_runners is None else a.red_runners
        if rr_share > 0:
            im_ = self.incidents
            self.red_runners = rr_mod.RedRunners(
                self.sumo.conn, share=rr_share, seed=a.seed + 7,
                busy=lambda vid: im_ is not None and (vid in im_.role or vid in im_.distracted or vid in im_.breaking))
        self.enf = None
        if a.enforce:
            from cosim import enforcement as enf_mod

            tls_, edge_ = a.enforce.split(":", 1)
            site = enf_mod.plan_site(net, scen.frame, tls_, edge_)
            stamp_ = time.strftime("%Y%m%d_%H%M%S")
            self.enf = enf_mod.SiteRun(site, self.sumo.conn, ROOT / "out" / "enforcement" / f"{site.name}_{stamp_}",
                                       record=a.enforce_record > 0, on_run=self.sumo.on_run)
        self.highway = None
        if a.highway_cars != 0:
            self._wire_highway(a, net, pool, base, curve)
        self.depots = self.emcalls = None
        self.calls = None  # (--profile: the counter of calls to the server)
        if a.depots and self.incidents is not None and a.scenario == "town05":
            self._wire_depots(a, net)
        sod = self.clock.seconds_of_day
        target = self.pop.target(sod)
        self.stage = f"Filling the town for {self.clock.hhmm()}: {target:.0f} vehicles"
        self.pop.add(int(target), anywhere=True)
        n_fill = int(a.fill_s / self.sumo.step_length)
        for k in range(n_fill):
            cur = self.sumo.step()
            self.pop.step(self._town(cur.vehicles), sod, lambda vid, v: True)
            if self.highway is not None and k % 4 == 0:
                self.highway.step(cur.vehicles, sod, set(self.sumo.conn.simulation.getPendingVehicles()))
            self.city_vehicles = len(cur.vehicles)
            self.progress = (k + 1) / n_fill
        self.progress = -1.0
        log.info("town filled: %d vehicles for %s (target %.0f)", len(cur.vehicles), self.clock.hhmm(), target)

        self.stage = "Handing the traffic to CARLA"
        if getattr(a, "profile", False):  # count the calls to the server (cosim/sampler.py)
            from cosim.sampler import CallCounter, CountingClient

            self.calls = CallCounter()
            client = CountingClient(client, self.calls)
            self.client = client
        self.cs = pooled_carla_side()(client, world, scen.frame, fps=a.rate, links=scen.links(landmark_ids),
                                      synchronous=False)
        self.cs.use_road_levels()  # Town05's ring road runs 10 m above other roads
        self.cs.ground_server = (a.host, a.port)  # the walkers' ground heights: asked by a process (cosim/ground_proc.py)
        self.poser = Poser(scen.frame, self.cs.ground, heading="trail")  # tangent and even turn both shook
        from cosim.interp import Trails

        self.trails = Trails()  # the main thread's: body headings from each vehicle's path
        self.spectator = world.get_spectator()
        self.shelters = self.spawn_shelters(net) if self.bus is not None else []
        self.stage = "Parking the vehicles the day needs"
        # what the town carries now, by class from the demand's mix, plus room; more are spawned in a thread
        # when needed, and surplus parked ones are destroyed (parked actors still cost CARLA frames)
        peak = self.city_vehicles * 1.1 + 6
        tot = sum(self.pop.weights) or 1
        vclass_of = {t: (re.search(r'vClass="([^"]+)"', vt) or [None, "passenger"])[1] for t, vt in pool.vtypes.items()}
        counts: dict[str, int] = {}
        for t, w in zip(self.pop.types, self.pop.weights, strict=True):
            vc = vclass_of.get(t, "passenger")
            counts[vc] = counts.get(vc, 0) + int(math.ceil(peak * w / tot)) + 2
        counts["bus"] = max(counts.get("bus", 0), 7)  # the line buses
        counts["emergency"], counts["authority"] = 2, 2
        scenes = self.incidents.max_active if self.incidents is not None else 0
        if self.incidents is not None:  # police cars and recovery trucks of the accidents open at a time
            counts["authority"] += scenes
            counts["truck"] = counts.get("truck", 0) + scenes
        # per CARLA model, as the vehicles will be drawn (each is sized as its model, cosim/vmodels.py)
        per_model: dict[str, int] = {}
        for vc, n in counts.items():
            picks = [self.typer.model_for(f"warm{k}", vc) for k in range(400)]
            for m in sorted(set(picks)):
                per_model[m] = per_model.get(m, 0) + int(math.ceil(n * picks.count(m) / len(picks))) + 1
        if self.incidents is not None:  # the recovery trucks (cosim/towing.py): the HGV, whatever the mix says
            from cosim import towing as _tw

            per_model[_tw.TRUCK_MODEL] = per_model.get(_tw.TRUCK_MODEL, 0) + scenes
        log.info("%d parked vehicles %s", self.cs.prewarm_vehicles(per_model), per_model)
        self.scene = None
        if self.incidents is not None:
            from cosim.incident_scene import IncidentScene, SceneWalkers

            roles = self.cs.prespawn_scene_walkers(self.incidents.max_active)
            log.info("accident scenes: walkers %s", {k: len(v) for k, v in roles.items()})
            self.scene = IncidentScene(scen.frame, self.cs.ground, self.incidents.net, self.scene_send,
                                       SceneWalkers(roles), self.cs, self.models)
            if self.depots is not None and "tow" in self.depots.depots:  # the yard's trucks carry their decks too
                yard = self.depots.depots["tow"]
                self.scene.extra_tows = lambda: list(yard.members)
                self.incidents.tow_free = lambda vid: not self.scene.carrying(vid)
        self.drivers_on = bool(a.drivers) and a.poses == "frame"
        if self.drivers_on:  # people at the wheel of the cars near the camera (cosim/drivers.py)
            from cosim import drivers as _drv

            ids = self.cs.prespawn_drivers(_drv.POOL)
            self.scene_send(("drivers", ids))
            log.info("drivers: %d walkers", len(ids))
        self.plate_proc = self.plate_in = self.plate_out = None
        self.plate_text: dict[int, str] = {}
        self.plates_made = self.plates_asked = 0
        if a.plates and a.poses == "frame":  # every car its own plates (cosim/plate_proc.py), made beside the viewer
            import multiprocessing as mp

            from cosim import plate_proc as _plp

            cfg = {m: i["plates"] for m, i in self.models.items() if i.get("plates")}
            pctx = mp.get_context("spawn")
            self.plate_in, self.plate_out = pctx.Queue(), pctx.Queue()
            self.plate_proc = pctx.Process(target=_plp.run, args=(a.host, a.port, self.plate_in, self.plate_out, cfg,
                                                                  a.seed), daemon=True)
            self.plate_proc.start()
            self.plate_cfg = cfg
            self.ask_plates()
            if self.scene is not None:  # the recovery trucks' decks (cosim/towing.py), made by the same process
                yard = self.depots.depots.get("tow") if self.depots is not None else None
                self.plate_in.put(("decks", 2 * self.incidents.max_active
                                   + (len(yard.slots) if yard is not None else 0)))
            if self.depots is not None:  # the depots' signs (cosim/depots.py)
                self.plate_in.put(("signs", self._depot_signs()))
        elif self.drivers_on:
            self.ask_plates()  # (the seats)
        if self.sumo.extras is not None and a.pedestrians_per_km > 0:
            self.stage = "Putting the pedestrians on the pavements"
            # only those within PEDESTRIAN_RADIUS_M are drawn: a pool of 40 covers them
            n = self.cs.prespawn_walkers(PEDESTRIANS_DRAWN + 4)
            log.info("%d walkers ready for %d pedestrians", n, self.sumo.extras.pedestrian_target)
        self.cams = None
        if self.sumo.walk is not None and a.ped_lights:  # signalised crosswalks: their pedestrian lights
            from cosim.ped_lights import PedLights

            self.stage = "Putting up the pedestrian lights"
            self.ped_lights = PedLights(world, scen.net_file, scen.frame, self.cmap)
            log.info("%d pedestrian lights", len(self.ped_lights.heads))
        self.signal_heads = None
        if a.signal_heads and self.cs.traffic_light_count:  # every head its own direction (cosim/signal_heads.py)
            from cosim.ped_lights import PedLights, clearance_states
            from cosim.signal_heads import SignalHeads

            self.stage = "Lighting the signal heads"
            crossings = self.ped_lights or PedLights(world, scen.net_file, scen.frame, self.cmap)
            self.signal_heads = SignalHeads(world, self.cs._tl_links, scen.net_file, crossings.heads,
                                            clearance_states(scen.net_file))
            self.cs.lights_off = True
            log.info("signal heads: %d drawn (%d poles), %d lens faces found by ray", len(self.signal_heads.lamps),
                     self.cs.traffic_light_count, self.signal_heads.faces_found)
        if a.cameras:
            from cosim.inspector import TrafficCameras

            self.stage = "Placing the traffic cameras"
            self.cams = TrafficCameras(world, net, scen.frame, self.cmap, host=a.host, port=a.port)
            if a.cctv_tick is not None:
                self.cams.native_tick = a.cctv_tick
            if a.cctv_width is not None:
                self.cams.native_width = a.cctv_width
            self.cams.view_fov = a.cctv_fov
            if getattr(self, "enf", None) is not None:   # the enforcement camera sees no inspector markers
                c_ = self.enf.site.cam
                self.cams.hide_near.append((float(c_["x"]), float(c_["y"]), 120.0))
        self.stage = "Choosing the camera views"
        self.choose_views(net)
        self._spectator_id = self.spectator.id
        self._tick_cb = world.on_tick(self.note_tick)
        self.first_view(net)
        if a.start_junction and a.start_junction in self.centres:  # --start-junction: start over that junction
            import carla

            x, y, _ = self.centres[a.start_junction]
            z = self.zs.get(a.start_junction, 0.0)
            px, py, pz = x - 26.0, y + 26.0, z + 22.0  # a corner view: poles, crossings and both directions in it
            yaw = math.degrees(math.atan2(y - py, x - px))
            pitch = math.degrees(math.atan2(z - pz, math.hypot(x - px, y - py)))
            self.first_tf = carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0))
            self.spectator.set_transform(self.first_tf)
            self.junction_i = list(self.centres).index(a.start_junction)
        if self.plate_proc is not None:  # the plates go on while the town is still behind the loading screen
            self.stage = "Fitting the number plates"
            t = time.time()
            while self.plates_made < self.plates_asked and time.time() - t < 60.0 and self.plate_proc.is_alive():
                self.pump_plates()
                self.progress = self.plates_made / max(self.plates_asked, 1)
                time.sleep(0.1)
            self.progress = -1.0
            log.info("number plates: %d of %d cars in %.0f s while loading", self.plates_made, self.plates_asked,
                     time.time() - t)
        self.stage = "Ready"

    def _wire_agents(self, a) -> None:
        """--controller agents / agents_observed (cosim/agents_actuated.py): the junction agents run the signals on
        top of the actuated program, as in the evaluation (tools/eval_town05_baseline.py): the emergency vehicles on
        a run (the extras' and the accidents' police cars) ask for green; agents_observed reads the cameras' reports
        (camera_reports.SimulatedCameras: the 53 cameras' visible lane stretches with the detector's measured misses
        and false boxes, every 2 s; the YOLO detector itself stays on the camera you inspect with H). Their decisions
        are drawn on the overlay (cosim/agent_hud.py)."""
        pol = self.sumo.policy
        if pol is None or not hasattr(pol, "events"):
            return
        self.agents = pol
        sumo = self.sumo

        def on_run() -> list:
            ex = sumo.extras
            out = [v for v in (ex.emergency if ex is not None else ()) if sumo.on_run(v)]
            im = self.incidents
            if im is not None:
                out += [v for v in list(im.role) if im.on_run(v)]
            return out

        pol.emergency = on_run
        if hasattr(pol, "goflow_role"):  # GO-FLOW: an accident's police car or recovery truck (cosim/goflow_priority.py)
            def role(vid: str):
                r = self.incidents.role.get(vid) if self.incidents is not None else None
                return r[1] if r and r[1] in ("police", "tow") else None

            pol.goflow_role = role
        if getattr(pol, "uses_cameras", False):
            from cosim.camera_reports import Layout, SimulatedCameras

            cam_dir = ROOT / "out" / "town05_cameras"
            parked_file = cam_dir / "parked_static.json"
            parked = json.loads(parked_file.read_text(encoding="utf-8")) if parked_file.is_file() else None
            cams = SimulatedCameras(Layout.load(cam_dir), seed=a.seed, parked=parked)
            pol.visible = cams.layout.visible

            def feed(c) -> None:
                ex = sumo.extras
                reps = cams.step(c, c.simulation.getTime(),
                                 {v: sumo.on_run(v) for v in (ex.emergency if ex is not None else ())})
                if reps is not None:
                    pol.observe(reps)

            sumo.pre_policy = feed
        frame = self.scen.frame
        for tls, ag in pol.junctions.items():  # where to draw: each green's stop lines, in CARLA's frame
            z = self.zs.get(tls, 0.0)
            self.agents_xy[tls] = {k: [(*frame.to_carla_xy(x, y), z, (h - 90.0) % 360.0) for x, y, h in pts]
                                   for k, pts in ag.stop_lines.items()}
        log.info("junction agents: %s on %d signals (%d on the program alone: too little camera view)",
                 pol.name, len(pol.junctions), len(pol.fallback))

    def agent_events(self, window_s: float = 60.0) -> list:
        """The agents' decisions of the last `window_s` simulated seconds (read from the main thread)."""
        if self.agents is None:
            return []
        try:
            evs = list(self.agents.events)
        except RuntimeError:  # appended to while copied (SUMO thread): next time
            return []
        return [e for e in evs if self.sim_now - e["t"] <= window_s]

    def spawn_shelters(self, net) -> list:
        """A bus shelter (CARLA's static.prop.busstop) on the pavement beside each stop."""
        import carla
        from sumolib.geomhelper import positionAtShapeOffset

        bp = self.world.get_blueprint_library().filter("static.prop.busstop")
        if not bp:
            log.info("no bus shelter model in this CARLA")
            return []
        out = []
        for line in self.bus.lines.values():
            for st in line["stops"]:
                lane = net.getLane(st["lane"])
                x, y = positionAtShapeOffset(lane.getShape(), st["pos"])
                cx, cy = self.scen.frame.to_carla_xy(x, y)
                wp = self.cmap.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True,
                                            lane_type=carla.LaneType.Driving)
                side = wp
                for _ in range(6):  # walk right to the pavement
                    nxt = side.get_right_lane() if side is not None else None
                    if nxt is None:
                        break
                    side = nxt
                    if side.lane_type == carla.LaneType.Sidewalk:
                        break
                if side is None or side.lane_type != carla.LaneType.Sidewalk:
                    continue
                tf = side.transform
                r = tf.get_right_vector()
                loc = carla.Location(tf.location.x + r.x * 0.6, tf.location.y + r.y * 0.6, tf.location.z)
                actor = self.world.try_spawn_actor(bp[0], carla.Transform(loc, carla.Rotation(0.0, tf.rotation.yaw, 0.0)))
                if actor is not None:
                    out.append(actor)
        log.info("bus lines: %s; %d shelters", ", ".join(f"{k} ({len(v['stops'])} stops)" for k, v in
                                                         self.bus.lines.items()), len(out))
        return out

    def ask_plates(self) -> None:
        """Cars made since the last call (the pool's) get plates (cosim/plate_proc.py) and their driver's place
        (cosim/drivers.py, for the pose process)."""
        if self.plate_in is None and not getattr(self, "drivers_on", False):
            return
        new, self.cs.new_cars = self.cs.new_cars, []
        if getattr(self, "drivers_on", False):
            from cosim.drivers import seat_rel

            seats = {aid: rel for aid, m in new if m in self.models and (rel := seat_rel(m, self.models[m])) is not None}
            if seats:
                self.scene_send(("seats", seats))
        if self.plate_in is None:
            return
        new = [(aid, m) for aid, m in new if m in self.plate_cfg]
        if new:
            self.plate_in.put(("cars", new))
            self.plates_asked += len(new)

    def pump_plates(self) -> None:
        """Plates made: to the pose process (it moves them with their cars), their numbers for the markers."""
        if self.plate_out is None:
            return
        try:
            while True:
                m = self.plate_out.get_nowait()
                if m[0] == "plates":
                    self.scene_send(("plates", m[1]))
                    self.plate_text.update(m[2])
                    self.plates_made += len(m[1])
                elif m[0] == "timing":
                    log.info("number plates: %d cars, textures %d ms, server %d ms", m[1], m[2], m[3])
                elif m[0] == "decks":
                    if self.scene is not None:
                        self.scene.add_decks(m[1])
                    log.info("recovery truck decks: %d", len(m[1]))
                elif m[0] == "signs":
                    log.info("depot signs: %s", m[1])
                elif m[0] == "fatal":
                    log.error("plate process failed:\n%s", m[1])
        except Exception as e:  # noqa: BLE001 - queue.Empty
            if type(e).__name__ != "Empty":
                log.warning("plates: %r", e)

    def scene_send(self, msg) -> None:
        """A message for the pose process from the accident scene (cosim/incident_scene.py); held until it runs."""
        q = getattr(self, "pose_q", None)
        if q is None:
            self._scene_backlog = getattr(self, "_scene_backlog", []) + [msg]
            return
        for m in getattr(self, "_scene_backlog", []):
            q.put(m)
        self._scene_backlog = []
        q.put(msg)

    # ------------------------------------------------------------ camera
    VIEW_BACK_M, VIEW_UP_M = 70.0, 28.0   # a junction's view: this far back along an approach, this high
    # what a view may see through: the ground, the roads and their markings, the sky, the traffic; anything else
    # (buildings, trees, walls, and what CARLA labels None or Static) blocks it
    _SEE_THROUGH = {"Roads", "RoadLines", "Sidewalks", "Ground", "Terrain", "Water", "Sky", "GuardRail",
                    "TrafficSigns", "TrafficLight", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians",
                    "Rider", "Train", "RailTrack", "Dynamic"}

    @classmethod
    def _blocks(cls, hit) -> bool:
        return getattr(hit.label, "name", str(hit.label)) not in cls._SEE_THROUGH

    def junction_pose(self, tid: str):
        """The view of junction `tid` (F8, and the first view): chosen once by `choose_views`; before that, the
        old fallback, 60 m south-west and 35 m up."""
        views = getattr(self, "views", {})
        if tid in views:
            return views[tid]
        import carla

        x, y, _ = self.centres[tid]
        z = self.zs.get(tid, 0.0)
        px, py, pz = x - 42.0, y + 42.0, z + 35.0
        yaw = math.degrees(math.atan2(y - py, x - px))
        pitch = math.degrees(math.atan2(z - pz, math.hypot(x - px, y - py)))
        return carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0))

    def approach_views(self, net, tid: str) -> list:
        """Candidate views of junction `tid`, best first: above one of its approach roads, VIEW_BACK_M back
        from the junction and VIEW_UP_M up, looking down the road at it (the road in front, the junction and
        its traffic in the middle, the street beyond). The longest and widest approaches first. [(transform,
        target location)]"""
        import carla

        frame = self.scen.frame
        cx, cy, _ = self.centres[tid]
        z = self.zs.get(tid, 0.0)
        try:
            nodes = {c[0].getEdge().getToNode() for c in net.getTLS(tid).getConnections()}
        except KeyError:
            return []
        cands = []
        for node in nodes:
            for e in node.getIncoming():
                if e.getFunction() == "internal" or not e.allows("passenger"):
                    continue
                shape = e.getShape()
                pts = [frame.to_carla_xy(px, py) for px, py in shape[::-1]]  # from the junction back
                walked, at = 0.0, pts[0]
                for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
                    seg = math.hypot(bx - ax, by - ay)
                    if walked + seg >= self.VIEW_BACK_M:
                        k = (self.VIEW_BACK_M - walked) / max(seg, 1e-6)
                        at = (ax + (bx - ax) * k, ay + (by - ay) * k)
                        walked = self.VIEW_BACK_M
                        break
                    walked += seg
                    at = (bx, by)
                if walked < 30.0:
                    continue
                score = walked + 6.0 * e.getLaneNumber()
                px, py, pz = at[0], at[1], z + self.VIEW_UP_M
                tx, ty, tz = cx, cy, z + 2.0
                yaw = math.degrees(math.atan2(ty - py, tx - px))
                pitch = math.degrees(math.atan2(tz - pz, math.hypot(tx - px, ty - py)))
                cands.append((score, carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0)),
                              carla.Location(tx, ty, tz)))
        return [(tf, tgt) for _, tf, tgt in sorted(cands, key=lambda c: -c[0])]

    def _clear(self, tf, target) -> bool:
        """Nothing solid (a building, a tree, a wall) between the view and the junction (CARLA's ray cast)."""
        try:
            hits = self.world.cast_ray(tf.location, target)
        except Exception:  # noqa: BLE001 - an older CARLA: take the view as it is
            return True
        total = tf.location.distance(target)
        for h in hits:
            if self._blocks(h) and tf.location.distance(h.location) < total - 10.0:
                return False
        return True


    @staticmethod
    def near_m(tf) -> float:
        """What counts as close in a view: 80 % of the distance to the ground along the bottom of its picture
        (90 degrees across, 16:9: 29.4 degrees below the middle)."""
        down = math.radians(min(85.0, abs(tf.rotation.pitch) + 29.4))
        return 0.8 * max(tf.location.z, 5.0) / max(math.sin(down), 0.2)

    def _depth_probe(self, views: list):
        """For each view, the share of its picture close up (closer than near_m), as CARLA renders it: a small depth camera
        put at each view in turn (CARLA's ray cast passed through some towers' meshes, and a tower beside the
        view filled a third of the first picture). None if the camera could not be used."""
        import queue

        import carla
        import numpy as np

        try:
            bp = self.world.get_blueprint_library().find("sensor.camera.depth")
            for k, v in (("image_size_x", 192), ("image_size_y", 108), ("fov", 90)):
                bp.set_attribute(k, str(v))
            cam = self.world.spawn_actor(bp, carla.Transform(carla.Location(0.0, 0.0, 500.0)))
        except Exception as e:  # noqa: BLE001 - no depth camera: the ray cast alone
            log.info("views: no depth camera (%s)", e)
            return None
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        out = []
        try:
            for tf in views:
                cam.set_transform(tf)
                img = None
                for _ in range(3):  # the first pictures may still be from the last place
                    try:
                        img = q.get(timeout=1.0)
                    except queue.Empty:
                        break
                if img is None:
                    out.append(1.0)
                    continue
                a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4).astype(np.float32)
                depth = (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / 16777215.0 * 1000.0
                out.append(float((depth < self.near_m(tf)).mean()))
                while not q.empty():
                    q.get_nowait()
        finally:
            try:
                cam.stop()
                cam.destroy()
            except Exception:  # noqa: BLE001
                pass
        return out

    def choose_views(self, net) -> None:
        """For every signal, the approach view (see approach_views) whose picture has the least close up: rendered
        by a depth camera (_depth_probe), or, without one, the first with a clear line to the junction. The
        viewer used to start where CARLA puts its spectator, and F8 looked from a fixed diagonal: both could face
        a wall."""
        self.views = {}
        cands = {tid: self.approach_views(net, tid)[:4] for tid in self.centres}
        flat = [(tid, tf) for tid, cs in cands.items() for tf, _ in cs]
        near = self._depth_probe([tf for _, tf in flat])
        self.view_near = {}
        if near is not None:
            for (tid, tf), n in zip(flat, near, strict=True):
                if tid not in self.view_near or n < self.view_near[tid]:
                    self.views[tid], self.view_near[tid] = tf, n
            worst = max(self.view_near.values(), default=0.0)
            log.info("junction views: %d by depth, the closest up %.0f %% of a picture", len(self.views), 100 * worst)
            return
        clear = 0
        for tid, cs in cands.items():
            if not cs:
                continue
            pick = next(((tf, tgt) for tf, tgt in cs if self._clear(tf, tgt)), None)
            clear += pick is not None
            self.views[tid] = (pick or cs[0])[0]
        log.info("junction views: %d of %d with a clear line", clear, len(self.views))

    FIRST_SHOTS = ((85.0, 34.0), (110.0, 55.0))  # the first view's (back, up): wider shots than F8's

    def first_view(self, net=None) -> None:
        """The view the viewer starts with: of the five busiest junctions' approach views, higher and further back
        than F8's, the one with the least close up in its picture (_depth_probe). The first version took the
        busiest junction's longest approach, and a parking deck filled a third of the picture; a tower did in
        the second, which looked for walls with CARLA's ray cast."""
        ids = list(self.centres)
        self.junction_i = 0
        best = None
        if net is not None:
            back, up = self.VIEW_BACK_M, self.VIEW_UP_M
            flat = []
            try:
                for fb, fu in self.FIRST_SHOTS:
                    self.VIEW_BACK_M, self.VIEW_UP_M = fb, fu
                    flat += [(k, tf) for k, tid in enumerate(ids[:5]) for tf, _ in self.approach_views(net, tid)[:4]]
            finally:
                self.VIEW_BACK_M, self.VIEW_UP_M = back, up
            near = self._depth_probe([tf for _, tf in flat]) if flat else None
            if near is not None:
                i = min(range(len(flat)), key=near.__getitem__)
                self.junction_i, best = flat[i][0], flat[i][1]
                log.info("first view: junction %s, %.0f m up, %.0f %% of the picture close up (of %d views; the worst "
                         "%.0f %%)", ids[self.junction_i], best.location.z, 100 * near[i], len(flat), 100 * max(near))
        if best is None:
            best = self.junction_pose(ids[0])
        self.first_tf = best
        self.spectator.set_transform(best)

    def next_junction(self) -> str:
        ids = list(self.centres)
        self.junction_i = (self.junction_i + 1) % len(ids)
        tid = ids[self.junction_i]
        self.spectator.set_transform(self.junction_pose(tid))
        return tid

    def note_tick(self, snap) -> None:
        """Every CARLA picture (its snapshot, no call to the server): its time, and where the view is."""
        self.carla_dt.append(snap.timestamp.delta_seconds)
        a = snap.find(self._spectator_id)
        if a is not None:
            self.view_tf = a.get_transform()

    def depicted_at(self, wall: float) -> float | None:
        """The SUMO time a picture taken at wall time `wall` shows (cosim/enforcement.py module doc): the step shown
        then (FrameFeeder: started at `start`) moves its vehicles from step k-1 to step k over one step."""
        log_ = list(self.shown_log)
        cur = None
        for start, t in log_:
            if start <= wall:
                cur = (start, t)
        if cur is None:
            return None
        start, t = cur
        step = self.sumo.step_length
        return t - step + min(max(wall - start, 0.0), step)

    def view_transform(self):
        """Where CARLA's view is, from the last picture. spectator.get_transform() is a call that waits for
        the server's next picture: made on every SUMO step and every pass of the main loop, it held both up
        whenever CARLA was slow (a camera open: the main loop ran 6 times a second)."""
        tf = getattr(self, "view_tf", None)
        return tf if tf is not None else self.spectator.get_transform()

    def focus(self) -> tuple[float, float]:
        tf = self.view_transform()
        f = tf.get_forward_vector()
        self._view = (tf.location.x, tf.location.y, tf.location.z, f.x, f.y)
        return tf.location.x, tf.location.y

    def _wire_highway(self, a, net, pool, base: dict, curve: list) -> None:
        """Town05's ring road: its limit and cars of its own (cosim/highway.py), started and taken off out of sight."""
        from cosim import daily
        from cosim.highway import PEAK_PER_LOOP, HighwayTraffic, find_loops

        loops = find_loops(net)
        if not loops:
            return
        conn, frame = self.sumo.conn, self.scen.frame

        def free(edge: str, pos: float) -> bool:
            try:
                return not self.in_view(*frame.to_carla_xy(*conn.simulation.convert2D(edge, pos)))
            except Exception:  # noqa: BLE001
                return False

        top = max(curve) or 1.0
        types = [(t, n) for t, n in pool.per_hour.items() if base.get(t) != "bus"]
        self.highway = HighwayTraffic(conn, loops, lambda s: daily.target_at(curve, s) / top, free=free, types=types,
                                      type_for=self.pop.type_for, seed=a.seed + 3,
                                      peak=PEAK_PER_LOOP if a.highway_cars is None else a.highway_cars,
                                      speed=a.highway_kmh / 3.6,
                                      lengths={e: net.getEdge(e).getLength() for lp in loops for e in lp},
                                      policy=self.sumo.policy)
        if self.incidents is not None:
            self.highway.protected = self.incidents.protected
        self.highway.setup()

    def _wire_depots(self, a, net) -> None:
        """The hospital, the police station and the recovery yard (cosim/depots.py): their vehicles parked (the
        police cars and trucks the accidents', the ambulances the emergency calls'), their signs put up later
        (_depot_signs, by the number plates' process)."""
        import random

        from cosim.depots import TOWN05, Depots, EmergencyCalls

        conn = self.sumo.conn
        self.depots = Depots(conn, net, TOWN05)
        self.depots.setup()
        self.incidents.stock_depots(self.depots)
        ex = self.sumo.extras
        if ex is not None and a.emergency_every > 0:
            from cosim.depots import CALL_EVERY_S

            ex.config.emergency_every = max(ex.config.emergency_every, CALL_EVERY_S)
            self.emcalls = EmergencyCalls(conn, net, self.depots, ex, random.Random(a.seed + 11))
            self.emcalls.stock()
            self.emcalls.avoid = self.incidents.blocked_roads
            ex.spawn_hook = self.emcalls.spawn
        self.depot_net = net

    def _depot_signs(self) -> list:
        """The depots' boards for the number plates' process: (name, image, CARLA pose, scale, mesh)."""
        from cosim import depots, towing
        from cosim.run import ground_z

        out = []
        frame = self.scen.frame
        for name, d in (self.depots.depots.items() if self.depots is not None else ()):
            x, y, facing = depots.sign_spot(self.depot_net, d.plan)
            cx, cy = frame.to_carla_xy(x, y)
            fx, fy = frame.to_carla_xy(x + math.cos(math.radians(facing)), y + math.sin(math.radians(facing)))
            yaw = math.degrees(math.atan2(fy - cy, fx - cx))
            z = ground_z(self.cmap, cx, cy) + depots.SIGN_UP_M + 2.36 * depots.SIGN_SCALE / 2.0
            out.append((name, depots.sign_texture(d.plan), (cx, cy, z, 0.0, yaw + depots.SIGN_YAW, depots.SIGN_ROLL),
                        depots.SIGN_SCALE, towing.DECK_MESH))
        return out

    def _unload_at_yard(self, vehicles: dict) -> None:
        """A recovery truck back at the yard with a wreck on its deck: the wreck is taken off where nobody looks."""
        if self.depots is None or self.scene is None:
            return
        frame = self.scen.frame
        for tv in self.depots.parked("tow"):
            v = vehicles.get(tv)
            if self.scene.carrying(tv) and (v is None or not self.in_view(*frame.to_carla_xy(v.x, v.y))):
                self.scene.unload(tv)

    def _town(self, vehicles: dict) -> dict:
        """The town's vehicles: all but the ring road's own (the day's curve is the town's)."""
        hw = getattr(self, "highway", None)
        dep = getattr(self, "depots", None)
        if hw is None and dep is None:
            return vehicles
        return {v: x for v, x in vehicles.items()
                if not (hw is not None and hw.is_ring_car(v)) and not (dep is not None and dep.owns(v))}

    def in_view(self, x: float, y: float) -> bool:
        """Whether CARLA point (x, y) may be on screen: near the camera, or ahead of it within
        60 deg of where it looks (wider than the 90 deg picture) and 500 m. Occlusion ignored."""
        if self._view is None:
            return False
        cx, cy, _, fx, fy = self._view
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d < 120.0:
            return True
        if d > 500.0:
            return False
        h = math.hypot(fx, fy)
        if h < 0.3:  # looking (nearly) straight down
            return False
        return (dx * fx + dy * fy) / (d * h) > 0.5

    def keep_walking(self, persons: dict) -> None:
        """A pedestrian you can see who is about to reach the end of the trip walks on to a new one
        (otherwise SUMO takes them off the pavement and they vanish in front of you)."""
        ex, c = self.sumo.extras, self.sumo.conn
        if ex is None or not ex.config.pedestrian_trips:
            return
        frame = self.scen.frame
        subs = self.sumo.person_subs
        # A person's last stage, asked once (it changes only when we add one, below); the remaining stages
        # and the road come with the step (SumoSide person_vars). These were three questions to SUMO per
        # person in view every second, ~36 round trips a step (tools/count_traci.py).
        stages = self._last_stage = getattr(self, "_last_stage", {})
        for pid in [p for p in stages if p not in persons]:
            del stages[pid]
        for pid, p in persons.items():
            r = subs.get(pid)
            if r is None or r.get(tc.VAR_STAGES_REMAINING) != 1 or not self.in_view(*frame.to_carla_xy(p.x, p.y)):
                continue
            try:
                st = stages.get(pid)
                if st is None:
                    st = stages[pid] = c.person.getStage(pid)
                road = p.road
                if not st.edges or road != st.edges[-1] or abs(c.person.getLanePosition(pid) - st.arrivalPos) > 15.0:
                    continue
                trip = ex._trip(c, road)
                if trip is not None:
                    c.person.appendWalkingStage(pid, trip[0], arrivalPos=trip[1])
                    stages.pop(pid, None)
                    self.walks_extended += 1
            except Exception as e:  # noqa: BLE001 - gone meanwhile
                log.debug("keep walking %s: %s", pid, e)

    def keep_driving(self, vehicles: dict) -> None:
        """Every second: a vehicle on the last three edges of its trip gets a new destination if it is in
        view (or the town has no more than it should), so trips don't end in front of the camera;
        SUMO ends a trip at the end of an edge, often a stop line, and the car vanished there.
        Emergency vehicles finish their run once out of sight."""
        c = self.sumo.conn
        over = len(vehicles) > self.target + 2
        ems = self.sumo.extras.emergency if self.sumo.extras is not None else {}
        alive = set(c.vehicle.getIDList())  # after this step's removals
        subs = self.sumo.vehicle_subs
        # The route index comes with the step (SumoSide vehicle_vars); a vehicle's route is asked once and
        # again when it nears its end (the only time it matters), and forgotten when we change it. Asking both
        # for every vehicle every second was ~72 round trips a step (tools/count_traci.py).
        routes = self._routes = getattr(self, "_routes", {})
        for vid in [v for v in routes if v not in alive]:
            del routes[vid]
        im = self.incidents
        for vid, v in vehicles.items():
            if vid not in alive or (im is not None and im.protected(vid)):  # an accident's vehicles keep their routes
                continue
            try:
                idx = subs[vid][tc.VAR_ROUTE_INDEX]
                route = routes.get(vid)
                if route is None:
                    route = routes[vid] = c.vehicle.getRoute(vid)
                if idx < len(route) - 3:
                    continue
                route = routes[vid] = c.vehicle.getRoute(vid)  # it may have changed: the real one
            except Exception:  # noqa: BLE001 - left meanwhile
                continue
            if idx < len(route) - 3:
                continue
            if self.bus is not None and self.bus.is_bus(vid):
                self.bus.next_lap(vid, idx, route)
                routes.pop(vid, None)
                continue
            if not self.in_view(*self.scen.frame.to_carla_xy(v.x, v.y)) and (over or vid in ems):
                continue
            for _ in range(5):
                dest = self.rng.choice(self.sinks)
                if dest in route:
                    continue
                try:
                    c.vehicle.changeTarget(vid, dest)
                    routes.pop(vid, None)
                    self.extended += 1
                    break
                except Exception:  # noqa: BLE001 - not reachable from here, or not for this vehicle class
                    continue

    # ------------------------------------------------------------ one SUMO step (the SUMO thread)
    def produce(self) -> "Packet":
        """One SUMO step and everything worked out from it that does not touch CARLA's actors: the
        vehicles and pedestrians near you, their start and end poses, lights. Runs in the SUMO thread,
        a step or two ahead of what is shown, so a slow step never stalls the picture."""
        from cosim.convert import light_state
        from cosim.run import Poser, within

        tt = self.produce_t = getattr(self, "produce_t", {})
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            tt[name] = tt.get(name, 0.0) + now - t
            t = now

        cur = self.sumo.step()
        self.sim_now = cur.time
        lap("sumo.step")
        fx, fy = self.focus()
        R = self.args.radius
        frame = self.scen.frame

        # vehicles SUMO still has (a vehicle taken off the road can linger in the subscription results)
        alive = set(self.sumo.conn.vehicle.getIDList())

        im_ = self.incidents

        def may_remove(vid, v):  # taken off the road to follow the day's curve: only out of sight, never an accident's
            return (vid in alive and not vid.startswith("pt_") and not self.in_view(*frame.to_carla_xy(v.x, v.y))
                    and (im_ is None or not im_.protected(vid)))

        lap("idlist")
        self.target = self.pop.step(self._town(cur.vehicles), self.clock.seconds_of_day, may_remove)
        if self.highway is not None and self._k % 4 == 0:  # every 2 s: the ring road's own cars
            self.highway.step(cur.vehicles, self.clock.seconds_of_day,
                              set(self.sumo.conn.simulation.getPendingVehicles()))
        lap("pop.step")
        im = self.incidents
        if self.red_runners is not None:
            self.red_runners.step(cur.time, cur.vehicles, cur.tls)
        if self.enf is not None:
            self.enf.sumo_step(cur)
        self._unhit = {}
        if im is not None:
            im.step(cur.time, cur.vehicles, cur.tls)
            # (3 Nov) cars that crashed in this step are drawn stopping where they touched (IncidentManager._to_contact),
            # not inside each other where SUMO found them; the step before still runs on towards where SUMO had them
            for vid, (x, y, ang) in im.hit_poses(cur.time).items():
                v = cur.vehicles.get(vid)
                if v is not None:
                    self._unhit[vid] = v
                    cur.vehicles[vid] = dataclasses.replace(v, x=x, y=y, angle=ang)
            if self._k % 10 == 0:  # every 5 s: a knot out of sight undone (IncidentManager.unjam)
                im.unjam(cur.time, cur.vehicles)
            lap("incidents")
        if self.depots is not None and self._k % 4 == 2:  # every 2 s: who got back to a depot, the calls
            self.depots.step(cur.vehicles)
            if self.emcalls is not None:
                self.emcalls.step(cur.time)
            lap("depots")
        self.city_vehicles = len(cur.vehicles)
        self._k += 1
        if self._k % 2 == 0:  # every second
            self.keep_driving(cur.vehicles)
            lap("keep_driving")
            if self.bus is not None:
                self.bus.step(cur.time, self.clock.hour)
            lap("bus")
            if self.agents is None or "preempt" not in self.agents.on:  # else the agents do it
                self.preempt(cur.vehicles)
            lap("preempt")
            self.keep_walking(cur.persons)
            lap("keep_walking")
            if self.sumo.extras is not None:
                self.sumo.extras.near_xy = frame.to_sumo_xy(fx, fy)
        drawn = self.drawn
        if self.prev is not None:
            for vid in cur.arrived:
                p = self.prev.vehicles.get(vid)
                if p is not None and vid in drawn and self.in_view(*frame.to_carla_xy(p.x, p.y)):
                    self.vanished_in_view += 1
                    if self.vanished_in_view <= 20:
                        log.info("vanished in view: %s at SUMO (%.0f, %.0f)", vid, p.x, p.y)
        arrived = list(cur.arrived)
        lap("arrived")
        full = cur
        cur = within(cur, frame, fx, fy, R, drawn)
        if im is not None and im.role:  # an accident's cars are drawn wherever you are: it never vanishes
            keep = {vid: v for vid, v in full.vehicles.items() if im.protected(vid)}
            if keep:
                cur = dataclasses.replace(cur, vehicles={**cur.vehicles, **keep})
        lap("within")
        # pedestrians only near you: a walker costs CARLA much more than a car (35 of them: 53 -> 35 FPS
        # at street level), and beyond ~150 m they are a few pixels
        rp = PEDESTRIAN_RADIUS_M
        d2 = {pid: (lambda c: (c[0] - fx) ** 2 + (c[1] - fy) ** 2)(frame.to_carla_xy(p.x, p.y))
              for pid, p in cur.persons.items()}
        keep = sorted((pid for pid in d2 if d2[pid] <= rp * rp), key=d2.get)[:PEDESTRIANS_DRAWN]
        near_p = {pid: cur.persons[pid] for pid in keep}
        cur = dataclasses.replace(cur, persons=near_p)
        prev = within(self.prev, frame, fx, fy, R + 200.0, set()) if self.prev else None
        if prev is not None and im is not None:
            keep = {vid: v for vid, v in self.prev.vehicles.items() if vid in cur.vehicles and vid not in prev.vehicles}
            if keep:
                prev = dataclasses.replace(prev, vehicles={**prev.vehicles, **keep})
        lap("persons+prev")
        self.poser.begin_step(prev, cur)
        lap("begin_step")
        snap = Poser(frame, None, heading=self.poser.heading)  # this step's poses, for the main thread
        snap.prev, snap.cur, snap._g, snap._from = self.poser.prev, self.poser.cur, self.poser._g, self.poser._from
        snap.pre = self._before.vehicles if getattr(self, "_before", None) is not None else None
        self._before = prev
        on_run = self.sumo.on_run
        lights = {vid: light_state(v.signals, self.night, emergency_run=on_run(vid)) for vid, v in cur.vehicles.items()}
        snapshot = []
        if im is not None:
            import carla

            for vid in cur.vehicles:
                bits = im.light_bits(vid)
                if bits is not None:  # hazards on the wrecks and the trucks, beacons on the police car
                    lights[vid] = carla.VehicleLightState(int(lights[vid]) | bits)
            snapshot = im.snapshot()
        dep = getattr(self, "depots", None)
        if dep is not None:  # parked at its depot: switched off (25 Oct: their headlights were on all night)
            import carla

            for d in dep.depots.values():
                for vid in d.parked.values():
                    if vid in lights:
                        lights[vid] = carla.VehicleLightState.NONE
        lap("lights")
        walk = {}
        last = getattr(self, "_walk_end", {})
        for pid, pp in cur.persons.items():
            # once per step and person (the pavement offset glides, cosim/carla_side.walker_pose); the
            # start of this step is where the last one ended
            end = self.cs.walker_pose(pp.x, pp.y, pp.angle, pp.road, pid=pid)
            walk[pid] = (last.get(pid, end), end, pp.speed)
        self._walk_end = {pid: w[1] for pid, w in walk.items()}
        self.cs.forget_walkers(walk)
        lap("walker_pose")
        self.emergency_seen.update(vid for vid, v in cur.vehicles.items() if v.vclass in ("emergency", "authority"))
        self.buses_seen.update(vid for vid in cur.vehicles if vid.startswith("pt_"))
        self.prev = cur
        pk = Packet(cur, arrived, snap, lights, walk, snapshot)
        lap("rest")
        tt["steps"] = tt.get("steps", 0) + 1
        # shown one step late: the curve between two steps also uses the step after (Poser "trail")
        held, self._held = getattr(self, "_held", None), pk
        if held is None:
            return self.produce()
        held.poser.nxt = {**cur.vehicles, **self._unhit} if self._unhit else cur.vehicles
        return held

    def preempt(self, vehicles: dict) -> None:
        """Signal pre-emption for emergency vehicles on a run: within 120 m of its next signal, if its
        movement has red, the current car green is cut to 1 s once it has had its 5 s minimum, so the
        signal's own program moves on (through its yellow) towards the phase that serves it; when it has
        green, that green is held until it is through (at most 15 s).

        The pedestrian walk and the all-red phases are never cut. They used to be: the walk was ended
        after 1 s (an emergency vehicle every ~45 s, 218 pre-emptions in 3 min of flight), so the walk
        signal hardly ever showed while the people who had set off finished crossing on red."""
        if self.sumo.extras is None:
            return
        c = self.sumo.conn
        now = c.simulation.getTime()
        im = self.incidents
        for vid, v in vehicles.items():
            if not (self.sumo.on_run(vid) or (im is not None and im.on_run(vid))):  # a police car on its way too
                continue
            try:
                nxt = c.vehicle.getNextTLS(vid)
            except Exception:  # noqa: BLE001 - gone
                continue
            if not nxt:
                continue
            tls, _, dist, state = nxt[0]
            if dist > 120.0:
                continue
            try:
                left = c.trafficlight.getNextSwitch(tls) - now
                if state in "Gg":
                    need = dist / max(v.speed, 5.0) + 2.0
                    if left < need:
                        c.trafficlight.setPhaseDuration(tls, min(need, 15.0))
                        self.preempted += 1
                elif state in "rRs":
                    cur_state = c.trafficlight.getRedYellowGreenState(tls)
                    walk = self.sumo.walk is not None and self.sumo.walk.is_walk(tls, cur_state)
                    car_green = "y" not in cur_state.lower() and any(ch in "Gg" for ch in cur_state) and not walk
                    if car_green and left > 1.0 and c.trafficlight.getSpentDuration(tls) >= PREEMPT_MIN_GREEN_S:
                        c.trafficlight.setPhaseDuration(tls, 1.0)
                        self.preempted += 1
            except Exception as e:  # noqa: BLE001 - a signal SUMO does not let us change
                log.debug("pre-emption at %s: %s", tls, e)

    # ------------------------------------------------------------ CARLA's side of a step (main thread)
    def apply(self, pk: "Packet") -> None:
        cs, dt = self.cs, self.sumo.step_length
        tt = self.apply_t = getattr(self, "apply_t", {})
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            tt[name] = tt.get(name, 0.0) + now - t
            t = now

        cs.destroy(pk.arrived)
        cs.destroy([vid for vid in list(cs.actors) if vid not in pk.cur.vehicles])
        lap("destroy")
        first = pk.poser.poses(0.0, dt)
        taken = self.scene.taken if self.scene is not None else ()  # wrecks a recovery truck has (pose process)
        new = [(vid, self.pool_key(vid, v), first[vid]) for vid, v in pk.cur.vehicles.items()
               if vid not in cs.actors and vid not in cs.failed and vid not in taken
               and not vid.startswith("incblock")]  # a closed lane's end in SUMO (cosim/incidents.py): not drawn
        t_sp = time.perf_counter()
        cs.spawn(new)
        if time.perf_counter() - t_sp > 1.0:
            log.warning("spawning %s took %.1f s", [(vid, vc) for vid, vc, _ in new], time.perf_counter() - t_sp)
        tt["spawned"] = tt.get("spawned", 0) + len(new)
        tt["reused_total"] = getattr(cs, "reused", 0)
        tt["short_total"] = getattr(cs, "short", 0)
        tt["parked"] = sum(len(p) for p in getattr(cs, "idle", {}).values())
        lap("spawn")
        cs.set_traffic_lights(pk.cur.tls)
        self.shown_tls = pk.cur.tls
        self.shown_next = (pk.cur.time, pk.cur.tls_next, time.time())  # for the lamps' lives (remaining())
        lap("lights")
        cs.destroy_walkers(list(pk.cur.persons_arrived) + [p for p in list(cs.walkers) if p not in pk.walk])
        lap("walker_destroy")
        wnew = [(pid, w[0]) for pid, w in pk.walk.items() if pid not in cs.walkers and pid not in cs.failed]
        cs.spawn_walkers(wnew)
        tt["walkers_spawned"] = tt.get("walkers_spawned", 0) + len(wnew)
        lap("walker_spawn")
        self.walkers_max = max(self.walkers_max, len(cs.walkers))
        if self.scene is not None:
            self.scene.update(pk.incidents, pk.cur.vehicles)
            if self.depots is not None and int(pk.cur.time * 2) % 4 == 0:
                self._unload_at_yard(pk.cur.vehicles)
            lap("incident scene")
        self.drawn = set(cs.actors)

    def pool_key(self, vid: str, v) -> str:
        """The CARLA model that draws SUMO vehicle `vid` (its type says it, cosim/vmodels.py), the pool's key."""
        from cosim import vmodels

        return vmodels.model_of(v.type_id) or vmodels.FIXED.get(v.vclass) or self.typer.model_for(vid, v.vclass)

    def move(self, pk: "Packet", s: float) -> dict:
        """Vehicles at fraction s of the step; pedestrians every other call (they walk slowly)."""
        import carla

        poses = pk.poser.poses(s, self.sumo.step_length, self.trails)
        self.cs.move(poses, pk.lights)
        self._mv = getattr(self, "_mv", 0) + 1
        if self._mv % 600 == 0:  # paths of vehicles no longer drawn
            self.trails.forget(set(poses))
        if pk.walk and self._mv % 2 == 0:
            walk = {}
            for pid, (a, b, speed) in pk.walk.items():
                la, lb = a.location, b.location
                loc = carla.Location(la.x + (lb.x - la.x) * s, la.y + (lb.y - la.y) * s, la.z + (lb.z - la.z) * s)
                ya, yb = a.rotation.yaw, b.rotation.yaw  # turn the short way round, not snap
                yaw = ya + ((yb - ya + 180.0) % 360.0 - 180.0) * s
                walk[pid] = (carla.Transform(loc, carla.Rotation(0.0, yaw, 0.0)), speed)
            self.cs.move_walkers(walk)
        return poses

    def remaining(self, paused: bool = False) -> dict[str, float]:
        """Seconds until each signal's next switch, as shown now (SUMO's step and the time since it is shown)."""
        shown = getattr(self, "shown_next", None)
        if not shown or not shown[1]:
            return {}
        t_step, nxt, at = shown
        since = 0.0 if paused else min(time.time() - at, self.sumo.step_length)
        return {tid: t - t_step - since for tid, t in nxt.items()}

    def carla_fps(self) -> float:
        """CARLA's frame rate over the frames since the last call."""
        k = len(self.carla_dt)
        d = self.carla_dt[self._dt_seen:k]
        self._dt_seen = k
        d = [x for x in d if x > 0]
        return len(d) / sum(d) if d else 0.0

    def close(self) -> None:
        if getattr(self, "cams", None) is not None:
            self.cams.shutdown()
        for a in getattr(self, "shelters", []):
            try:
                a.destroy()
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        for f in (lambda: self.world.remove_on_tick(self._tick_cb), lambda: self.cs.close(),
                  lambda: self.sumo.close()):
            try:
                f()
            except Exception:  # noqa: BLE001 - closing anyway
                pass


# ---------------------------------------------------------------- the clock window
def compose_overlay(screen, size, dirty_prev, full: bool, key, hud=None, keys_pill=None, prompt=None, panel=None,
                    native=None, markers=None) -> list:
    """Draws the overlay's parts on `screen` (see-through `key` elsewhere) and returns the rectangles drawn,
    to be sent to the window with those of last time. hud: (draw(screen), width, height) of the clock card;
    keys_pill, prompt: surfaces; panel: the camera panel, or the sensor feed's full-screen picture; native:
    draw(screen) -> rectangles for the native camera feed (CARLA's window is the picture)."""
    import pygame

    w, h = size
    rects = []
    for r in dirty_prev:
        screen.fill(key, r)  # see-through again
    if full:
        screen.fill(key)
    # a camera feed covers everything: its own picture (sensor), or CARLA's window under its labels (native)
    cctv_full = native is not None or (panel is not None and panel.get_size() == (w, h))
    if native is not None:
        rects += native(screen)
    if hud is not None and not cctv_full:
        hud[0](screen)
        rects.append(pygame.Rect(0, 0, max(hud[1], 560) + 40, hud[2] + 70))
    if keys_pill is not None and not cctv_full:
        rects.append(screen.blit(keys_pill, (16, h - 42)))
    if panel is not None:
        if cctv_full:  # the sensor feed's full-screen picture
            rects.append(screen.blit(panel, (0, 0)))
        else:
            rects.append(screen.blit(panel, (w - panel.get_width() - 16, h - panel.get_height() - 16)))
    if prompt is not None and not cctv_full:
        rects.append(screen.blit(prompt, ((w - prompt.get_width()) // 2, h - 96)))
    if markers is not None and not cctv_full:  # the accidents (cosim/incident_hud.py)
        rects += markers(screen)
    return rects


def lower_priority() -> None:
    """Below-normal priority for this process (Windows), so CARLA's game thread is served first
    on a laptop CPU; SUMO runs in its own process and keeps normal priority."""
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
    except Exception:  # noqa: BLE001 - not Windows
        pass


DEMO_WARMUP_S = 10.0  # --demo: seconds flown before measuring (as the 15 s of tools/town05_fps_test.py)


def demo_pose(ff, t: float, seconds: float):
    """The scripted flight of --demo: after DEMO_WARMUP_S, half the rest circling the town 60 m up,
    half driving along a road 3 m up (as tools/town05_fps_test.py, so the numbers compare)."""
    import carla

    if not hasattr(ff, "_demo"):
        xs = [v[0] for v in ff.centres.values()]
        ys = [v[1] for v in ff.centres.values()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        wp = ff.cmap.get_waypoint(carla.Location(cx, cy, 0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
        route = [wp]
        for _ in range(600):
            nxt = wp.next(1.0)
            if not nxt:
                break
            wp = nxt[0]
            route.append(wp)
        ff._demo = (cx, cy, min(max(0.3 * span, 80.0), 220.0), route)
    cx, cy, r, route = ff._demo
    warm = DEMO_WARMUP_S
    half = warm + (seconds - warm) / 2
    if t < half:
        a = 2 * math.pi * t / 30.0
        px, py, pz = cx + r * math.cos(a), cy + r * math.sin(a), 60.0
        yaw = math.degrees(math.atan2(cy - py, cx - px))
        pitch = math.degrees(math.atan2(-pz, r))
        return ("high" if t >= warm else None), carla.Transform(carla.Location(px, py, pz),
                                                               carla.Rotation(pitch, yaw, 0.0))
    w = route[int((t - half) * 12.0) % len(route)].transform
    return "street", carla.Transform(carla.Location(w.location.x, w.location.y, w.location.z + 3.0),
                                     carla.Rotation(-5.0, w.rotation.yaw, 0.0))


def crosswalk_view(ff):
    """--demo: a raised corner view of the junction whose pedestrian phase was served last (someone was
    waiting there), as (transform, tls id, link indices of its crossings), or None."""
    import carla

    if ff.ped_lights is None or ff.sumo.walk is None:
        return None
    tid = ff.sumo.walk.last_served or next(iter(ff.centres))
    heads = [h for h in ff.ped_lights.heads if h[3] == tid]
    if not heads or tid not in ff.centres:
        return None
    cx, cy, _ = ff.centres[tid]
    hx, hy, z = heads[0][:3]
    d = math.hypot(hx - cx, hy - cy) or 1.0
    px, py, pz = hx + (hx - cx) / d * 9.0, hy + (hy - cy) / d * 9.0, z + 4.5
    yaw = math.degrees(math.atan2(cy - py, cx - px)) + 20.0
    pitch = -14.0
    return (carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0)), tid,
            sorted({h[4] for h in heads}))


def demo_close_ups(ff, shooter, shots: dict, scenario: str, draw=None, inspector_panel=None, native_feed=None,
                   screen_size=None) -> None:
    """--demo: close views of a bus shelter, a line bus and an emergency vehicle (if one is drawn),
    after the measuring (the traffic keeps moving in between); then the traffic camera as H opens it, the
    sensor way and the native way (native_feed(i) -> the overlay's draw for camera i; screen_size() -> the
    window's size)."""
    import carla

    def look(target: carla.Location, back: float, up: float, yaw: float) -> carla.Transform:
        a = math.radians(yaw)
        px, py, pz = target.x - back * math.cos(a), target.y - back * math.sin(a), target.z + up
        pitch = math.degrees(math.atan2(target.z + 1.0 - pz, back))
        return carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0))

    views = []
    if getattr(ff, "shelters", None):
        tf = ff.shelters[0].get_transform()
        views.append(("shelter", look(tf.location, 14.0, 4.0, tf.rotation.yaw + 90.0)))
    for pid, aid in list(ff.cs.walkers.items())[:1]:
        actor = ff.world.get_actor(aid)
        if actor is not None:
            tf = actor.get_transform()
            views.append(("pedestrian", look(tf.location, 6.0, 1.5, tf.rotation.yaw + 60.0)))
    for name, pick in (("bus", lambda vid, v: vid.startswith("pt_")),
                       ("emergency", lambda vid, v: v.vclass in ("emergency", "authority"))):
        for vid, v in (ff.prev.vehicles.items() if ff.prev else []):
            aid = ff.cs.actors.get(vid) if pick(vid, v) else None  # .get: the pose thread may let it go meanwhile
            if aid is not None:
                actor = ff.world.get_actor(aid)
                if actor is not None:
                    tf = actor.get_transform()
                    views.append((name, look(tf.location, 12.0, 4.0, tf.rotation.yaw + 35.0)))
                break
    # the crosswalks: SUMO's crossings against the zebras painted in CARLA's map, and a view of the
    # crossing with the most pedestrians near it, from a raised corner
    try:
        import sumolib

        net = sumolib.net.readNet(str(ff.scen.net_file), withInternal=True, withPedestrianConnections=True)
        centres = []
        for e in net.getEdges():
            if e.getFunction() == "crossing":
                sh = e.getLanes()[0].getShape()
                centres.append(ff.scen.frame.to_carla_xy(sum(p[0] for p in sh) / len(sh), sum(p[1] for p in sh) / len(sh)))
        pts = ff.cmap.get_crosswalks()
        polys, cur_poly = [], []
        for loc in pts:
            cur_poly.append((loc.x, loc.y))
            if len(cur_poly) > 2 and math.hypot(loc.x - cur_poly[0][0], loc.y - cur_poly[0][1]) < 0.01:
                polys.append(cur_poly)
                cur_poly = []
        zebra = [(sum(p[0] for p in q) / len(q), sum(p[1] for p in q) / len(q)) for q in polys]
        offs = sorted(min(math.hypot(x - zx, y - zy) for zx, zy in zebra) for x, y in centres) if zebra else []
        shots["crossings"] = len(centres)
        shots["carla_zebras"] = len(zebra)
        if offs:
            shots["crossing_to_zebra_m"] = {"median": round(offs[len(offs) // 2], 1), "max": round(offs[-1], 1),
                                            "within_3m": sum(1 for o in offs if o <= 3.0)}
        wl = [ff.scen.frame.to_carla_xy(p.x, p.y) for p in (ff.prev.persons.values() if ff.prev else [])]
        if centres and wl:
            best = max(centres, key=lambda c: sum(1 for w in wl if math.hypot(w[0] - c[0], w[1] - c[1]) < 15.0))
            from cosim.run import ground_z

            z = ground_z(ff.cmap, best[0], best[1])
            views.append(("crosswalk", look(carla.Location(best[0], best[1], z), 16.0, 9.0, 35.0)))
            views.append(("crosswalk_wide", look(carla.Location(best[0], best[1], z), 35.0, 22.0, 215.0)))
    except Exception as e:  # noqa: BLE001 - the demo goes on without these
        log.warning("crosswalk check: %s", e)
    for name, tf in views:
        ff.spectator.set_transform(tf)
        time.sleep(1.5)
        p = ROOT / "out" / "freefly" / f"{scenario}_demo_{name}.png"
        shots[name] = p.name if shooter.grab_window(p) else None
    shots["arrow_probe"] = probe_arrows(ff, shooter)
    cams = ff.cams
    if cams is not None and draw is not None and cams.cams:
        # the inspector at the busiest junction's first camera, as H would open it
        tid = next(iter(ff.centres))
        i = next((k for k, c in enumerate(cams.cams) if c.tls == tid), 0)
        c = cams.cams[i]
        ff.spectator.set_transform(look(carla.Location(c.x, c.y, c.z - 4.0), 12.0, 4.0, c.yaw))
        cams.open(i)
        t = time.time()
        while time.time() - t < 40.0 and (cams.view is None or (cams.model is None and "not" not in cams.model_state)):
            draw(panel=inspector_panel(cams.view, cams.names[i], cams.model_state))
            time.sleep(0.2)
        for _ in range(15):
            draw(panel=inspector_panel(cams.view, cams.names[i], cams.model_state))
            time.sleep(0.15)
        p = ROOT / "out" / "freefly" / f"{scenario}_demo_inspector.png"
        shots["inspector"] = p.name if shooter.grab_window(p) else None
        if cams.view is not None:
            shots["inspector_counts"] = cams.view[1]
            shots["detector_fps"] = round(cams.view[2], 1)
        cams.close()
        if native_feed is not None:
            shots["cctv_native"] = demo_native_camera(ff, shooter, scenario, draw, native_feed, screen_size, i)


AGENT_SHOT_ORDER = ("ambulance", "corridor", "noflow", "spill", "trickle", "balance", "skip")


def agent_view(ff, e: dict):
    """A view of an agent's decision: above the approach road of the green it was about, 42 m back from its stop line
    and 34 m up, looking at the junction (the road keeps buildings out of the way); else the junction's own view."""
    import carla

    pts = [p for k in e.get("phases", []) for p in ff.agents_xy.get(e["tls"], {}).get(k, [])]
    if not pts or e["tls"] not in ff.centres:
        return ff.junction_pose(e["tls"])
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    z = pts[0][2]
    a = math.radians(pts[0][3])
    px, py, pz = x - 42.0 * math.cos(a), y - 42.0 * math.sin(a), z + 34.0
    cx, cy, _ = ff.centres[e["tls"]]
    yaw = math.degrees(math.atan2(cy - py, cx - px))
    pitch = math.degrees(math.atan2(z - pz, math.hypot(cx - px, cy - py)))
    return carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0))


def demo_agents(ff, shooter, scenario: str, st: dict, t_el: float, limit_s: float, redraw=None) -> bool:
    """--demo-agents, once per pass of the main loop: every ~12 s, a junction where an agent decided something in the
    last 6 s (each kind of decision in turn: emergency vehicles, a full exit, a trickle, a short road filling up, an
    empty green skipped; a junction not shown in the last 2 minutes), seen from above the road of that green
    (agent_view), and a screenshot 3 s later with the overlay. True when done (out/<scenario>_freefly_agents.json)."""
    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    if t_el < DEMO_WARMUP_S + 20.0 or ff.agents is None:
        return t_el >= limit_s
    pend = st.get("pending")
    if pend is not None:
        if time.time() - pend["at"] < 3.0:
            return False
        if redraw is not None:
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        name = f"{scenario}_agents_{len(shots) + 1:02d}_{pend['kind']}_{pend['tls']}.png"
        ok = shooter.grab_window(out / name)
        shots.append({"file": name if ok else None, "tls": pend["tls"], "kind": pend["kind"], "text": pend["text"],
                      "clock": ff.clock.hhmm()})
        st["pending"] = None
        st["next"] = time.time() + 9.0
    if t_el >= limit_s or len(shots) >= 16:
        res = {"shots": shots, "stats": dict(ff.agents.stats), "controller": ff.agents.name,
               "fallback": sorted(ff.agents.fallback)}
        evs = ff.agent_events(window_s=1e9)
        res["decisions_kept"] = len(evs)
        res["by_kind"] = {k: sum(1 for e in evs if e["kind"] == k) for k in sorted({e["kind"] for e in evs})}
        (ROOT / "out" / f"{scenario}_freefly_agents.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return True
    if time.time() < st.get("next", 0.0):
        return False
    seen = st.setdefault("seen", {})
    recent = [e for e in ff.agent_events(window_s=6.0) if e["kind"] != "walk" and e["tls"] in ff.centres
              and time.time() - seen.get(e["tls"], -1e9) > 120.0]
    if not recent:
        return False
    rank = {k: i for i, k in enumerate(AGENT_SHOT_ORDER)}
    taken = {k: sum(1 for x in shots if x["kind"] == k) for k in AGENT_SHOT_ORDER}  # every kind in turn
    e = min(recent, key=lambda e: (taken.get(e["kind"], 0), rank.get(e["kind"], 99), -e["t"]))
    seen[e["tls"]] = time.time()
    ff.spectator.set_transform(agent_view(ff, e))
    st["pending"] = {"at": time.time(), "tls": e["tls"], "kind": e["kind"], "text": e["text"]}
    return False


def demo_highway(ff, shooter, scenario: str, st: dict, t_el: float, views: int, redraw=None) -> bool:
    """--demo-highway, once per pass of the main loop: `views` views over the ring road (cosim/highway.py), spread
    along its loops, 16 m up and 32 m back along the road, a screenshot of each 6 s after the camera got there; then
    out/<scenario>_freefly_highway.json (the cars on each loop, on the road in each view). True when done."""
    import carla

    hw = getattr(ff, "highway", None)
    if t_el < DEMO_WARMUP_S + 20.0:
        return False
    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    conn, frame = ff.sumo.conn, ff.scen.frame
    pend = st.get("pending")
    if pend is not None:
        if time.time() - pend["at"] < 6.0:
            return False
        if redraw is not None:
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        name = f"{scenario}_highway_{len(shots) + 1:02d}_{pend['edge']}.png"
        ok = shooter.grab_window(out / name)
        try:
            on = conn.edge.getLastStepVehicleNumber(pend["edge"])
            kmh = conn.edge.getLastStepMeanSpeed(pend["edge"]) * 3.6
        except Exception:  # noqa: BLE001
            on, kmh = None, None
        shots.append({"file": name if ok else None, "edge": pend["edge"], "cars_on_edge": on,
                      "mean_kmh": None if kmh is None else round(kmh, 1), "clock": ff.clock.hhmm(),
                      "ring_cars": [len(m) for m in hw.cars] if hw else None})
        st["pending"] = None
    if len(shots) >= views:
        res = {"shots": shots, "loops": hw.loops if hw else None, "speed_kmh": round(hw.speed * 3.6) if hw else None,
               "target_per_loop": hw.target(ff.clock.seconds_of_day) if hw else None,
               "added": hw.added if hw else 0, "removed": hw.removed if hw else 0}
        (ROOT / "out" / f"{scenario}_freefly_highway.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return True
    if hw is None or not hw.loops:
        (ROOT / "out" / f"{scenario}_freefly_highway.json").write_text(json.dumps({"result": "no ring road"}),
                                                                         encoding="utf-8")
        return True
    edges = [e for lp in hw.loops for e in lp]
    i = len(shots)
    e = edges[i % len(edges)]
    frac = (0.3, 0.7)[(i // len(edges)) % 2]
    from cosim.run import ground_z

    try:
        ln = f"{e}_{max(0, conn.edge.getLaneNumber(e) // 2)}"
        L = conn.lane.getLength(ln)
        x0, y0 = frame.to_carla_xy(*conn.simulation.convert2D(e, L * frac))
        x1, y1 = frame.to_carla_xy(*conn.simulation.convert2D(e, min(L, L * frac + 5.0)))
    except Exception:  # noqa: BLE001
        shots.append({"file": None, "edge": e, "error": "no position"})
        return False
    yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
    a = math.radians(yaw)
    z = ground_z(ff.cmap, x0, y0)
    px, py, pz = x0 - 32.0 * math.cos(a), y0 - 32.0 * math.sin(a), z + 16.0
    pitch = math.degrees(math.atan2(-16.0, 32.0))
    ff.spectator.set_transform(carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0)))
    st["pending"] = {"at": time.time(), "edge": e}
    return False


def demo_signals(ff, shooter, scenario: str, st: dict, t_el: float, views: int, redraw=None) -> bool:
    """--demo-signals, once per pass of the main loop: `views` signalised junctions seen from 30 m up, close, each
    photographed once some of its poles are green (up to 45 s), with which of its CARLA poles were green, yellow
    and red at that moment (since 25 Oct the two opposite approaches go together: tools/build_town05_signals.py);
    then out/<scenario>_freefly_signals.json. True when done."""
    import carla

    from cosim.run import ground_z

    if t_el < DEMO_WARMUP_S + 20.0:
        return False
    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    ids = sorted(ff.centres)
    cs = ff.cs
    if getattr(ff, "demo_close", None):
        return _demo_poles_close(ff, shooter, scenario, st, ff.demo_close, redraw)

    def poles(tid: str) -> dict:
        n = {"green": 0, "yellow": 0, "red": 0, "other": 0}
        for tl_id, (_tl, links) in cs._tl_links.items():
            if not any(t == tid for t, _ in links):
                continue
            want = cs._tl_set.get(tl_id)
            key = {carla.TrafficLightState.Green: "green", carla.TrafficLightState.Yellow: "yellow",
                   carla.TrafficLightState.Red: "red"}.get(want, "other")
            n[key] += 1
        return n

    pend = st.get("pending")
    if pend is not None:
        now = time.time()
        p = poles(pend["tid"])
        if now - pend["at"] < 6.0 or (p["green"] == 0 and now - pend["at"] < 45.0):
            return False
        if redraw is not None:
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        name = f"{scenario}_signals_{len(shots) + 1:02d}_{pend['tid']}.png"
        ok = shooter.grab_window(out / name)
        shots.append({"file": name if ok else None, "junction": pend["tid"], "poles": poles(pend["tid"]),
                      "clock": ff.clock.hhmm()})
        st["pending"] = None
    if len(shots) >= min(views, len(ids)):
        n = len(shots)
        res = {"shots": shots, "junctions": n,
               "with_two_or_more_green": sum(1 for s_ in shots if s_["poles"]["green"] >= 2)}
        (ROOT / "out" / f"{scenario}_freefly_signals.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return True
    tid = ids[len(shots)]
    x, y, _ = ff.centres[tid]
    z = ground_z(ff.cmap, x, y)
    px, py, pz = x - 20.0, y + 20.0, z + 30.0
    yaw = math.degrees(math.atan2(y - py, x - px))
    pitch = math.degrees(math.atan2(z - pz, math.hypot(x - px, y - py)))
    ff.spectator.set_transform(carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0)))
    st["pending"] = {"at": time.time(), "tid": tid}
    return False


def _mast_heads(tl) -> list[tuple[float, float, float, float]]:
    """CARLA's own lamps (29 Oct): the heads over the road of pole `tl` as (x, y, z, drivers' yaw), from its light
    boxes (the highest ones: the mast; the low one is on the post) and the way its lanes run at their stop lines.
    Empty when CARLA does not say."""
    try:
        boxes = list(tl.get_light_boxes())
        wps = list(tl.get_stop_waypoints())
    except Exception:  # noqa: BLE001 - an older API
        return []
    if not boxes or not wps:
        return []
    top = max(b.location.z for b in boxes)
    mast = [b for b in boxes if b.location.z > top - 1.0]
    sx = sum(math.cos(math.radians(w.transform.rotation.yaw)) for w in wps)
    sy = sum(math.sin(math.radians(w.transform.rotation.yaw)) for w in wps)
    yaw = math.degrees(math.atan2(sy, sx))
    return [(b.location.x, b.location.y, b.location.z, yaw) for b in mast]


def _demo_poles_close(ff, shooter, scenario: str, st: dict, tid: str, redraw=None) -> bool:
    """--demo-signals-close: each CARLA pole of junction `tid` from 8 m (towards the junction's middle, 3 m aside,
    at the heads' height), photographed once while it shows green and once red (up to 150 s each), with the light
    boxes CARLA reports for it (its heads: TrafficLight.get_light_boxes) and the junction's SUMO state then, so what
    the pole's own lamps show can be compared with SUMO's walk on the crossings beside it. True when done."""
    import carla

    from cosim.ped_lights import clearance_states, ped_aspect
    from cosim.run import ground_z

    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    cs = ff.cs
    heads = getattr(ff, "signal_heads", None)
    if "todo" not in st:
        todo = []
        for tl_id, (tl, links) in sorted(cs._tl_links.items()):
            if any(t == tid for t, _ in links):
                if heads is not None:  # each head its own direction: straight green, left green, all red
                    for want in ("s", "l", "red"):
                        todo.append((tl_id, want))
                else:
                    for want in (carla.TrafficLightState.Green, carla.TrafficLightState.Yellow,
                                 carla.TrafficLightState.Red):
                        todo.append((tl_id, want))
        st["todo"] = todo
        st["fdw"] = clearance_states(ff.scen.net_file) if hasattr(ff, "scen") else {}
    if not st["todo"]:
        res = {"junction": tid, "shots": shots}
        (ROOT / "out" / f"{scenario}_freefly_poles.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return True
    tl_id, want = st["todo"][0]
    tl, links = cs._tl_links[tl_id]
    pend = st.get("pending")
    if pend is None:
        loc = tl.get_transform().location
        cx, cy, _ = ff.centres[tid]
        d = math.hypot(cx - loc.x, cy - loc.y) or 1.0
        ux, uy = (cx - loc.x) / d, (cy - loc.y) / d
        z = ground_z(ff.cmap, loc.x, loc.y)
        px, py, pz = loc.x + ux * 8.0 - uy * 3.0, loc.y + uy * 8.0 + ux * 3.0, z + 4.0
        tx, ty, tz = loc.x, loc.y, z + 3.0
        mast = [lp for k, lp in enumerate(heads.lamps) if heads.pole_of[k] == tl_id and lp[6] != "ped"] \
            if heads is not None else _mast_heads(tl)
        if mast:  # as the drivers coming up to it see its heads: 16 m back along their lane, a little below them
            hx = sum(lp[0] for lp in mast) / len(mast)
            hy = sum(lp[1] for lp in mast) / len(mast)
            hz = sum(lp[2] for lp in mast) / len(mast)
            a = math.radians(mast[0][3])
            fx, fy = math.cos(a), math.sin(a)
            back = 16.0 if heads is not None else 9.0  # CARLA's lamps (29 Oct): close enough to see the lenses
            px, py, pz = hx - fx * back, hy - fy * back, hz - 1.5
            tx, ty, tz = hx, hy, hz
        yaw = math.degrees(math.atan2(ty - py, tx - px))
        if mast:
            yaw += 12.0  # the heads left of the middle (the right of the window may be covered)
        pitch = math.degrees(math.atan2(tz - pz, math.hypot(tx - px, ty - py)))
        ff.spectator.set_transform(carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yaw, 0.0)))
        st["pending"] = {"at": time.time(), "since": None}
        return False
    now = time.time()
    shown = cs._tl_set.get(tl_id)
    if heads is not None:  # from SUMO: the pole's directions as its heads show them
        state_now = ff.shown_tls.get(tid, "")
        mine = [k for k in range(len(heads.lamps)) if heads.pole_of[k] == tl_id]
        cols = [a for k in mine for a in heads.arrows(k, {tid: state_now})]
        ok = (any(d == want and c == "G" for d, c in cols) if want in ("s", "l")
              else bool(cols) and all(c == "r" for _, c in cols))
        shown = want if ok else None
    if shown != want:
        pend["since"] = None
        if now - pend["at"] < 150.0:
            return False
    elif pend["since"] is None:
        pend["since"] = now
        return False
    elif now - pend["since"] < (0.6 if want == carla.TrafficLightState.Yellow else 3.0):  # amber lasts 3.5 s
        return False
    if redraw is not None:
        for _ in range(3):
            redraw()
            time.sleep(0.12)
    colour = {carla.TrafficLightState.Green: "green", carla.TrafficLightState.Yellow: "yellow",
              carla.TrafficLightState.Red: "red", "s": "straight_green",
              "l": "left_green", "red": "all_red"}.get(want, "other")
    name = f"{scenario}_poles_{tid}_{tl_id}_{colour}.png"
    ok = shooter.grab_window(out / name) if shown == want else False
    state = ff.shown_tls.get(tid, "") if isinstance(getattr(ff, "shown_tls", None), dict) else ""
    try:
        boxes = [{"x": round(b.location.x, 2), "y": round(b.location.y, 2), "z": round(b.location.z, 2),
                  "ext": [round(b.extent.x, 2), round(b.extent.y, 2), round(b.extent.z, 2)]}
                 for b in tl.get_light_boxes()]
    except Exception:  # noqa: BLE001 - an older API
        boxes = None
    fdw = st.get("fdw", {}).get(tid, {})
    walks = {str(i): ped_aspect(state, i, fdw.get(i)) for i in sorted(fdw)} if state else {}
    if heads is not None:
        lamp_info = [{"what": heads.lamps[k][6], "aspect": heads.aspect(k, ff.shown_tls),
                      "arrows": heads.arrows(k, ff.shown_tls),
                      "xyz": [round(v, 2) for v in heads.lamps[k][:3]]} for k in range(len(heads.lamps))
                     if heads.pole_of[k] == tl_id]
    else:
        lamp_info = None
    shots.append({"file": name if ok else None, "pole": tl_id, "wanted": colour, "heads": lamp_info,
                  "shown": str(shown), "sumo_links": [i for t, i in links if t == tid], "sumo_state": state,
                  "crossings": walks, "light_boxes": boxes, "pole_index": getattr(tl, "get_pole_index", lambda: None)()})
    st["todo"].pop(0)
    st["pending"] = None
    return False


def demo_drivers(ff, shooter, scenario: str, st: dict, t_el: float, cars: int, redraw=None) -> bool:
    """--demo-drivers, once per pass of the main loop: `cars` vehicles standing in traffic (a red light, a queue; not
    a wreck), different models first, each seen close from its driver's side and from the front (the camera kept
    with the car if it moves off), a screenshot of each once the pose process has seated a driver in it (cosim/drivers.py, pose_proc.Driver.seat_drivers); then
    out/<scenario>_freefly_drivers.json. True when done."""
    import carla

    from cosim.drivers import seat_rel

    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    out_json = ROOT / "out" / f"{scenario}_freefly_drivers.json"
    if t_el < DEMO_WARMUP_S + 15.0:
        return False
    def place(c: dict) -> bool:
        """The camera beside the car (or in front of it), where the car is now: it may have moved off."""
        a = ff.world.get_actor(c["aid"])
        if a is None or ff.cs.actors.get(c["vid"]) != c["aid"]:
            return False
        tf = a.get_transform()
        x, y, z = seat_rel(c["model"], ff.models[c["model"]])[:3]
        hip = tf.transform(carla.Location(x, y, z + 0.35))
        # from above the neighbours' roofs (in a queue a car stands beside it and another 2-3 m ahead)
        eye = (tf.transform(carla.Location(x + 1.2, y - 3.4, z + 2.2)) if c["view"] == "side"
               else tf.transform(carla.Location(ff.models[c["model"]]["extent"][0] + 1.3, y - 0.6, z + 2.5)))
        d = hip - eye
        ff.spectator.set_transform(carla.Transform(eye, carla.Rotation(
            math.degrees(math.atan2(d.z, math.hypot(d.x, d.y))), math.degrees(math.atan2(d.y, d.x)), 0.0)))
        return True

    pend = st.get("pending")
    if pend is not None:
        if not place(pend):  # the car has gone (arrived): another one
            st["pending"] = None
            return False
        if time.time() - pend["at"] < 2.5:
            return False
        if redraw is not None:
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        name = f"{scenario}_drivers_{len(shots) + 1:02d}_{pend['model'].split('.', 1)[1]}_{pend['view']}.png"
        ok = shooter.grab_window(out / name)
        shots.append({"file": name if ok else None, "model": pend["model"], "view": pend["view"],
                      "vehicle": pend["vid"], "clock": ff.clock.hhmm()})
        st["pending"] = None
        if pend["view"] == "side":  # the same car from the front next
            st["next"] = dict(pend, view="front")
    if len(shots) >= 2 * cars or t_el > DEMO_WARMUP_S + 15.0 + 60.0 * cars:
        out_json.write_text(json.dumps({"shots": shots, "cars": len({s_["vehicle"] for s_ in shots})}, indent=2),
                            encoding="utf-8")
        return True
    nxt = st.pop("next", None)
    if nxt is None:
        seen = {s_["model"] for s_ in shots}
        cands = []
        scene = getattr(ff, "scene", None)
        for vid, v in (ff.prev.vehicles.items() if ff.prev else []):
            aid = ff.cs.actors.get(vid)
            m = ff.cs.vclass_of.get(vid) if hasattr(ff.cs, "vclass_of") else None
            if aid is None or v.speed >= 0.2 or not m or m not in ff.models or seat_rel(m, ff.models[m]) is None:
                continue
            if scene is not None and scene.damage_for(vid):  # a wreck: its driver got out
                continue
            cands.append((m in seen, vid in {s_["vehicle"] for s_ in shots}, vid, aid, m))
        if not cands:
            return False
        cands.sort()
        _, _, vid, aid, m = cands[0]
        nxt = {"vid": vid, "aid": aid, "model": m, "view": "side"}
    if place(nxt):
        st["pending"] = dict(nxt, at=time.time())
    return False


def demo_depots(ff, shooter, scenario: str, st: dict, t_el: float, rounds: int, redraw=None) -> bool:
    """--demo-depots, once per pass of the main loop: `rounds` rounds of the hospital, the police station and the
    recovery yard (cosim/depots.py), each seen from its road 22 m before the sign, 3 m up, a screenshot 6 s after the camera got
    there, a round every 60 s; then out/<scenario>_freefly_depots.json (who was parked and out at each shot, the
    depots' and the emergency calls' counts). True when done."""
    import carla

    from cosim import depots as dmod
    from cosim.run import ground_z

    dep = getattr(ff, "depots", None)
    out_json = ROOT / "out" / f"{scenario}_freefly_depots.json"
    if dep is None:
        out_json.write_text(json.dumps({"result": "no depots"}), encoding="utf-8")
        return True
    if t_el < DEMO_WARMUP_S + 20.0:
        return False
    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", [])
    names = list(dep.depots)
    pend = st.get("pending")
    if pend is not None:
        if time.time() - pend["at"] < 6.0:
            return False
        if redraw is not None:
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        name = f"{scenario}_depot_{len(shots) + 1:02d}_{pend['depot']}.png"
        ok = shooter.grab_window(out / name)
        d = dep.depots[pend["depot"]]
        shots.append({"file": name if ok else None, "depot": pend["depot"], "clock": ff.clock.hhmm(),
                      "parked": sorted(d.parked.values()), "coming": sorted(d.coming), "out": sorted(d.out)})
        st["pending"] = None
        st["last_t"] = time.time()
    if len(shots) >= rounds * len(names):
        em = getattr(ff, "emcalls", None)
        res = {"shots": shots, "stats": dep.stats, "calls": em.stats if em is not None else None}
        out_json.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        return True
    i = len(shots)
    if i % len(names) == 0 and i > 0 and time.time() - st.get("last_t", 0.0) < 60.0:
        return False  # the next round a minute later: some out, some back
    d = dep.depots[names[i % len(names)]]
    frame = ff.scen.frame
    try:
        x, y, facing = dmod.sign_spot(ff.depot_net, d.plan)
        cx, cy = frame.to_carla_xy(x, y)
        e = ff.depot_net.getEdge(d.plan.edge)
        far = e.getLanes()[-1]  # (as tools/probe_depots.py's road view: the far lane, 22 m before the sign)
        fx, fy, _ = dmod.lane_point(far.getShape(), far.getLength(), far.getLength() * d.plan.sign_at - 22.0, -2.0)
        ex, ey = frame.to_carla_xy(fx, fy)
    except Exception:  # noqa: BLE001
        shots.append({"file": None, "depot": names[i % len(names)], "error": "no position"})
        return False
    yaw = math.degrees(math.atan2(cy - ey, cx - ex))
    z = ground_z(ff.cmap, ex, ey)
    dist = math.hypot(cx - ex, cy - ey)
    pitch = math.degrees(math.atan2(-2.0, max(dist, 1.0)))
    ff.spectator.set_transform(carla.Transform(carla.Location(ex, ey, z + 3.0), carla.Rotation(pitch, yaw, 0.0)))
    st["pending"] = {"at": time.time(), "depot": names[i % len(names)]}
    return False


RECORD_PRE_S = 60.0     # --record-clips: the crash clip starts this long before the crash
RECORD_POST_S = 180.0   # ... and runs this long after it
RECORD_NORMAL_S = 240.0  # the normal-traffic clip
RECORD_SETTLE_S = 20.0  # a recording's first seconds are not used (the sensor and the encoder settling)
RECORD_CRASH_MAX_M = 60.0  # the crash clip's wrecks at most this far from the camera (a crash 98 m off, at the next
                           # junction behind a truck, was hard to make out: 30 Oct, second run)


def enforce_lookups(ff):
    """(plate_of(vid), car_of(vid)) for the site's records: the plate the car carries (cosim/plate_proc.py), its
    CARLA model and paint colour."""
    def plate_of(vid):
        aid = ff.cs.actors.get(vid)
        return ff.plate_text.get(aid) if aid is not None else None

    def car_of(vid):
        aid = ff.cs.actors.get(vid)
        if aid is None:
            return None
        cache = ff.__dict__.setdefault("_car_info", {})
        if aid not in cache:
            try:
                a = ff.world.get_actor(aid)
                cache[aid] = {"model": a.type_id, "color": a.attributes.get("color")} if a is not None else None
            except Exception:  # noqa: BLE001
                cache[aid] = None
        return cache[aid]

    return plate_of, car_of


def enforce_step(ff, st: dict, args) -> bool:
    """--enforce, once per pass of the main loop: the site's camera started once the town is loaded and the first
    steps shown, its records written every second; True when --enforce-record's time is up."""
    run = ff.enf
    now = time.time()
    if "t0" not in st:
        if not ff.shown_log or now - ff.shown_log[0][0] < 3.0:
            return False
        cam = run.site.cam
        z, _ = ff.cs.ground(cam["x"], cam["y"], cam["yaw"])
        consumer = plate_consumer = None
        if args.enforce_agent:
            consumer, plate_consumer = start_agent(ff, run, z, st)
        run.start(ff.world, ff.depicted_at, z, consumer=consumer, plate_consumer=plate_consumer)
        st.update(t0=now, last=0.0)
        # the cars it sees posed every picture, with their plates (the pose process), and nobody appearing, vanishing
        # or ending a trip in its view (as in the viewer's own)
        q = getattr(ff, "pose_q", None)
        if q is not None:
            q.put(("eyes", [(cam["x"] + 30.0 * math.cos(math.radians(cam["yaw"])),
                             cam["y"] + 30.0 * math.sin(math.radians(cam["yaw"])))]))
        from cosim.clip_recorder import near_view
        from types import SimpleNamespace

        spec = SimpleNamespace(x=cam["x"], y=cam["y"], yaw=cam["yaw"])
        orig = ff.in_view
        ff.in_view = lambda x, y: orig(x, y) or near_view(spec, cam["fov"], x, y, reach_m=160.0)
        print(f"red-light camera at {run.site.name}: {run.out}", flush=True)
        return False
    if now - st["last"] >= 1.0:
        st["last"] = now
        run.tick(*enforce_lookups(ff))
        agent_news(ff, st)
        register_fleet(ff, st)
    return args.enforce_record > 0 and now - st["t0"] >= args.enforce_record


def start_agent(ff, run, ground_z: float, st: dict):
    """The red-light agent's process (cosim/redlight_proc.py) and the cameras' hand-over to it: each picture as JPEG
    on a short queue (dropped when the agent is busy), the plate camera's on a queue of its own (the agent only keeps
    them until it needs one). Returns the two cameras' consumers (overview, plate)."""
    import multiprocessing as mp

    import cv2

    from cosim import redlight_proc

    ctx = mp.get_context("spawn")
    q_ctl, q_frames, q_out, q_plates = ctx.Queue(), ctx.Queue(maxsize=6), ctx.Queue(), ctx.Queue(maxsize=40)
    site_json = dict(run.site.to_json())
    for key in ("cam", "plate_cam"):
        if site_json.get(key):
            site_json[key] = dict(site_json[key], z=ground_z + site_json[key].get("height_m", 6.0), ground_z=ground_z)
    cases = ROOT / "out" / "enforcement" / "cases"
    model = ROOT / "models" / "carla-yolo26n-t05.pt"
    proc = ctx.Process(target=redlight_proc.run, args=(q_ctl, q_frames, q_out, site_json, str(model), str(cases),
                                                      q_plates), daemon=True)
    proc.start()
    with run._lock:
        for t, s_ in zip(run.signals.times, run.signals.states, strict=False):  # the log so far
            q_ctl.put(("signal", t, s_))
        run.agent_q = q_ctl
    st.update(agent=proc, agent_out=q_out, agent_ctl=q_ctl, sent=0, dropped=0, cases=0)

    def consumer(k, t, rgb):
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return
        try:
            q_frames.put_nowait(("frame", k, t, buf.tobytes()))
            st["sent"] += 1
        except Exception:  # noqa: BLE001 - queue full: the agent is busy, this picture is skipped
            st["dropped"] += 1

    def plate_consumer(k, t, rgb):
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
        if not ok:
            return
        try:
            q_plates.put_nowait(("plate", k, t, buf.tobytes()))
        except Exception:  # noqa: BLE001 - full
            st["plate_dropped"] = st.get("plate_dropped", 0) + 1

    log.info("red-light agent started for %s (cases in %s)", run.site.name, cases)
    return consumer, plate_consumer


def agent_news(ff, st: dict) -> None:
    """The agent's messages: ready, a new case (a note on screen), an error (logged); and the town's clock for its
    evidence."""
    q = st.get("agent_out")
    if q is None:
        return
    clock = getattr(ff, "clock", None)
    if clock is not None and st.get("agent_ctl") is not None:
        try:
            st["agent_ctl"].put(("clock", ff.sim_now, clock.seconds_of_day, getattr(clock, "speed", 60.0),
                                 getattr(clock, "weekday", "")))
        except Exception:  # noqa: BLE001
            pass
    while True:
        try:
            m = q.get_nowait()
        except Exception:  # noqa: BLE001 - empty
            break
        if m[0] == "ready":
            log.info("red-light agent ready (%s)", m[1])
        elif m[0] == "error":
            log.error("red-light agent failed:\n%s", m[1])
        elif m[0] == "case":
            c = m[1]
            st["cases"] += 1
            log.info("red-light case %s: plate %s (%.2f), %.2f s into red", c["id"], c["plate"], c["conf"], c["red_s"])
            ff.enf_note = (f"RED LIGHT at {ff.enf.site.name}: {c['plate'] or 'plate not read'} -> police office",
                           time.time() + 6.0)
    if st.get("sent", 0) and int(time.time()) % 30 == 0:
        log.info("red-light agent: %d pictures sent, %d skipped (busy), %d cases", st["sent"], st["dropped"],
                 st["cases"])


def register_fleet(ff, st: dict) -> None:
    """The town's vehicle register (cosim/register.py): every car with plates, its model and colour, a fictional
    keeper; written to out/enforcement/register.json (the police office reads it)."""
    from cosim.register import Register

    reg = st.get("register")
    if reg is None:
        reg = st["register"] = Register(ROOT / "out" / "enforcement" / "register.json")
        st["registered"] = set()
    _, car_of = enforce_lookups(ff)
    by_actor = {aid: vid for vid, aid in ff.cs.actors.items()}
    n = 0
    for aid, text in list(getattr(ff, "plate_text", {}).items()):
        if aid in st["registered"] or n >= 25:
            continue
        cache = ff.__dict__.setdefault("_car_info", {})
        if aid not in cache:
            try:
                a = ff.world.get_actor(aid)
                cache[aid] = {"model": a.type_id, "color": a.attributes.get("color")} if a is not None else None
            except Exception:  # noqa: BLE001
                cache[aid] = None
        info = cache.get(aid)
        if info is not None:
            reg.add(text, info["model"], info["color"])
        st["registered"].add(aid)
        n += 1
    if n:
        reg.save()
    del by_actor


def record_eyes(ff, rec) -> None:
    """(3 Nov) Every recorded camera is an eye for the pose process: the vehicles it films are posed every picture.
    Before, only those within NEAR_M of the viewer's own eye (the first camera) were; beyond, every 2nd picture, and
    beyond FAR_M every 4th (cosim/pose_proc.py): on the other cameras' film a car moved 2-4 times a second, and a crash
    there showed in steps of up to 0.45 s (CARLA trace of a crash 300 m off, 3 Nov). As for the red-light camera: a
    point 30 m ahead of each camera."""
    q = getattr(ff, "pose_q", None)
    if q is None:
        return
    pts = []
    for r in rec.recs.values():
        if getattr(r, "stopped", False):
            continue
        sp = r.spec
        pts.append((sp.x + 30.0 * math.cos(math.radians(sp.yaw)), sp.y + 30.0 * math.sin(math.radians(sp.yaw))))
    q.put(("eyes", pts))


def demo_record_clips(ff, st: dict, args, clock) -> bool:
    """--record-clips, once per pass of the main loop: record --record-cams to MP4 (cosim/clip_recorder.py) until
    there is a normal-traffic stretch and a crash in a recorded picture (with RECORD_PRE_S before and RECORD_POST_S
    after it), or --record-clips seconds have gone by; then cut the two clips and write their labels. True when
    done."""
    from cosim import clip_recorder as cr
    from cosim.camera_reports import Layout

    now = time.time()
    if "rec" not in st:
        if ff.incidents is None:
            raise RuntimeError("--record-clips needs the accidents (--accidents)")
        lay = Layout.load(ROOT / "out" / f"{args.scenario}_cameras")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = ROOT / "out" / "incident_clips" / stamp
        rec = cr.ClipRecorder(ff.world, lay, out, lambda: ff.sim_now, fps=args.record_fps)
        names = [n.strip() for n in args.record_cams.split(";") if n.strip()]
        if args.record_specs:  # cameras of their own (tools/record_city_cams.json): name + CamSpec fields
            from cosim.perception import CamSpec

            for c in json.loads((ROOT / args.record_specs).read_text(encoding="utf-8")):
                fields = {k: c[k] for k in ("tls", "node", "road_to", "x", "y", "z", "yaw", "pitch")}
                rec.add_camera(c["name"], CamSpec(**fields))
                names.append(c["name"])
        if not names:
            raise RuntimeError("--record-clips: no camera (--record-cams, --record-specs)")
        for n in names:
            rec.start(rec.index_of(n))
        first = rec.recs[rec.index_of(names[0])].spec
        import carla

        ff.spectator.set_transform(carla.Transform(carla.Location(first.x, first.y, first.z),
                                                   carla.Rotation(first.pitch, first.yaw, 0.0)))
        orig = ff.in_view
        ff.in_view = lambda x, y: orig(x, y) or rec.near_recorded(x, y)  # nobody appears or vanishes on film
        record_eyes(ff, rec)
        st.update(rec=rec, out=out, t0=now, last=0.0, incs={}, seen={}, dynamic=0, stamp=stamp,
                  sim0=ff.sim_now, game0=clock.game_s)
        cr.write_json(out / "run.json", {
            "made": stamp, "args": sys.argv[1:], "scenario": args.scenario, "seed": args.seed,
            "controller": getattr(ff, "controller_name", args.controller), "careless_share": args.careless_share,
            "clock_start": clock.hhmm(), "clock_weekday": clock.weekday, "clock_speed": clock.speed,
            "sim_s_at_start": round(ff.sim_now, 2), "fps": args.record_fps,
            "cameras": {n: dict(vars(lay.cams[lay.names.index(n)].spec), width=lay.cams[0].width,
                                height=lay.cams[0].height, fov=lay.cams[0].fov) for n in lay.names}})
        print(f"recording {len(names)} cameras to {out}", flush=True)
        return False
    rec, out = st["rec"], st["out"]
    if now - st["last"] < 0.5:
        return False
    st["last"] = now
    sim = ff.sim_now
    frame = ff.scen.frame
    im = ff.incidents
    for inc in im.done + im.incidents:
        info = st["incs"].get(inc.id)
        if info is None:
            cx, cy = frame.to_carla_xy(inc.x, inc.y)
            in_rec = rec.seen_by(cx, cy, cams=[i for i, r in rec.recs.items() if not r.stopped])
            info = {"id": inc.id, "t0": round(inc.t0, 2), "kind": inc.kind, "severity_ms": inc.severity,
                    "severity_kmh": round(inc.severity * 3.6), "label": inc.label, "edge": inc.edge, "lane": inc.lane,
                    "sumo_xy": [round(inc.x, 1), round(inc.y, 1)], "carla_xy": [round(cx, 1), round(cy, 1)],
                    "seen_by_at_crash": [rec.lay.names[i] for i, _ in in_rec]}
            if not in_rec and st["dynamic"] < args.record_dynamic:
                cand = [(i, d) for i, d in rec.seen_by(cx, cy) if i not in rec.recs]
                if cand:
                    r = rec.start(cand[0][0], reason=f"after crash {inc.id}")
                    record_eyes(ff, rec)
                    st["dynamic"] += 1
                    info["dynamic_camera"] = r.name
            st["incs"][inc.id] = info
            log.info("clips: crash %d (%s, %s) seen by %s", inc.id, inc.kind, inc.label,
                     info["seen_by_at_crash"] or info.get("dynamic_camera") or "no camera")
        info["events"] = [[round(inc.t0 + t, 1), e] for t, e in inc.events]
        info["phase"] = inc.phase
        # (31 Oct) a breakdown hit by a car becomes a crash, a crash with a third car a pile-up
        info.update(kind=inc.kind, label=inc.label, severity_ms=inc.severity, severity_kmh=round(inc.severity * 3.6),
                    cause=getattr(inc, "cause", None), cars=len(inc.wrecks),
                    speeding=[w.vid for w in inc.wrecks if w.speeding],
                    # (3 Nov) the moment the cars touched, as drawn (t0 is the SUMO step that saw it, up to 0.5 s later),
                    # and the police: when they were due, sent, on scene (None: not yet / never), and how they got there
                    t_contact=inc.t_contact, police=dict(inc.police_record))
        for w in inc.wrecks:  # the seconds each recorded camera has a wreck in its picture
            if w.removed:
                continue
            wx, wy = frame.to_carla_xy(w.x, w.y)
            for i, d_ in rec.seen_by(wx, wy, cams=[i for i, r in rec.recs.items() if not r.stopped]):
                span = st["seen"].setdefault((rec.lay.names[i], inc.id), [sim, sim])
                span[1] = sim
                key = (rec.lay.names[i], inc.id)
                st.setdefault("dist", {})[key] = min(st.get("dist", {}).get(key, 1e9), d_)
    # (31 Oct) the look-alikes: a van stopped in its lane, a bus standing long at its stop; not incidents, the seconds
    # each recorded camera has one in its picture
    live = [i for i, r in rec.recs.items() if not r.stopped]
    for lk in im.lookalikes:
        if lk.t0 is None:
            continue
        lx, ly = frame.to_carla_xy(lk.x, lk.y)
        for i, _d in rec.seen_by(lx, ly, cams=live):
            span = st.setdefault("lseen", {}).setdefault((rec.lay.names[i], lk.id), [sim, sim])
            span[1] = sim
    looks = [dict(lk, carla_xy=[round(v, 1) for v in frame.to_carla_xy(lk["x"], lk["y"])])
             for lk in im.lookalike_snapshot() if lk["t0"] is not None and lk["t0"] >= st["sim0"] - 5.0]
    cr.write_json(out / "incidents.json", {"incidents": list(st["incs"].values()),
                                           "visible": [{"camera": c, "incident": k, "from_sim_s": round(a, 1),
                                                        "to_sim_s": round(b, 1)} for (c, k), (a, b) in st["seen"].items()],
                                           "lookalikes": looks,
                                           "lookalikes_visible": [{"camera": c, "lookalike": k, "from_sim_s": round(a, 1),
                                                                   "to_sim_s": round(b, 1)}
                                                                  for (c, k), (a, b) in st.get("lseen", {}).items()]})

    def normal_span(r):
        """The first RECORD_NORMAL_S stretch of recording r with no wreck in its picture (30 s margin)."""
        if r.started_sim is None or r.last_sim is None:
            return None
        busy = sorted((a - 30.0, b + 30.0) for (c, _k), (a, b) in st["seen"].items() if c == r.name)
        a0 = r.started_sim + RECORD_SETTLE_S
        for a, b in busy + [(1e18, 1e18)]:
            if min(a, r.last_sim) - a0 >= RECORD_NORMAL_S:
                return a0
            a0 = max(a0, b)
        return None

    def crash_span(r):
        """(incident id, clip start) of a crash in recording r's picture with its stretch before and after."""
        if r.started_sim is None or r.last_sim is None:
            return None
        if args.record_crash_match and args.record_crash_match not in r.name:
            return None
        for (c, k), (a, _b) in st["seen"].items():
            if c != r.name:
                continue
            t0 = st["incs"][k]["t0"]
            if a - t0 > 5.0:
                continue  # in the picture only later (moved, or the camera started after the crash)
            if st.get("dist", {}).get((c, k), 1e9) > RECORD_CRASH_MAX_M:
                continue
            # the stretch before the crash shows ordinary traffic: no earlier accident's wrecks in this picture
            # (30 Oct, first run: the clip opened on the last minute of another crash's recovery at the same spot)
            before = [b2 for (c2, k2), (_a2, b2) in st["seen"].items() if c2 == r.name and k2 != k and b2 < t0]
            start = max([r.started_sim + RECORD_SETTLE_S, t0 - RECORD_PRE_S] + [b2 + 15.0 for b2 in before])
            if t0 - start >= 0.75 * RECORD_PRE_S and r.last_sim - t0 >= RECORD_POST_S:
                return k, start
        return None

    def dynamic_span(r):
        for (c, k), (a, _b) in st["seen"].items():
            if c == r.name and r.reason.startswith("after crash") and r.started_sim is not None \
                    and r.last_sim - r.started_sim >= RECORD_SETTLE_S + RECORD_POST_S:
                return k, r.started_sim + 2.0
        return None

    normal = next(((r, s) for r in rec.recs.values() if (s := normal_span(r)) is not None), None)
    crash = next(((r, c) for r in rec.recs.values() if (c := crash_span(r)) is not None), None)
    late = crash is None and next(((r, c) for r in rec.recs.values() if (c := dynamic_span(r)) is not None), None)
    timed_out = now - st["t0"] >= args.record_clips
    if not ((normal and crash) or timed_out):
        k = int(now - st["t0"])
        if k // 60 != st.get("said", -1):
            st["said"] = k // 60
            print(f"recording: {k // 60} min, crashes {len(st['incs'])}, normal clip {'yes' if normal else 'no'}, "
                  f"crash clip {'yes' if crash else 'no'}", flush=True)
        return False
    if not crash and late:
        crash = late
    rec.close()
    clips = {}

    def clip_of(r, start_sim, seconds, name):
        rows = cr.frames_index(out / f"{cr.slug(r.name)}.frames.csv")
        v0 = cr.video_s_at(rows, start_sim)
        seconds = min(seconds, rows[-1][1] - v0)
        cr.cut(r.path, out / f"{name}.mp4", v0, seconds)
        spec = r.spec
        gs = st["game0"] + (start_sim - st["sim0"]) * clock.speed
        return {"file": f"{name}.mp4", "camera_id": r.name, "junction": spec.tls, "looking_along_road_to": spec.road_to,
                "camera_pose_carla": {"x": round(spec.x, 2), "y": round(spec.y, 2), "z": round(spec.z, 2),
                                      "yaw_deg": round(spec.yaw, 2), "pitch_deg": spec.pitch},
                "resolution": [rec.w, rec.h], "fov_deg": rec.fov, "fps": args.record_fps,
                "duration_s": round(seconds, 1), "sim_start_s": round(start_sim, 2),
                "sim_end_s": round(start_sim + seconds, 2),
                "clock_start": f"{int(gs % 86400 // 3600):02d}:{int(gs % 3600 // 60):02d}:{int(gs % 60):02d}",
                "source_recording": r.path.name, "source_video_start_s": round(v0, 2), "recording_reason": r.reason}

    if normal:
        r, s = normal
        clips["normal_traffic"] = clip_of(r, s, RECORD_NORMAL_S, "normal_traffic")
        clips["normal_traffic"]["incidents_in_picture"] = []
        clips["normal_traffic"]["incidents_elsewhere_in_town"] = [
            {"id": i["id"], "sim_s": i["t0"]} for i in st["incs"].values()
            if s <= i["t0"] <= s + RECORD_NORMAL_S]
    if crash:
        r, (k, s) = crash
        c = clip_of(r, s, RECORD_PRE_S + RECORD_POST_S, "crash")
        inc = st["incs"][k]
        a, b = st["seen"][(r.name, k)]
        c["incident"] = dict(inc, crash_at_clip_s=round(inc["t0"] - s, 1),
                             contact_at_clip_s=(None if inc.get("t_contact") is None
                                                else round(inc["t_contact"] - s, 2)),
                             wreck_distance_m=round(st.get("dist", {}).get((r.name, k), -1.0), 1),
                             wreck_in_picture_clip_s=[round(max(a, s) - s, 1), round(min(b, s + c["duration_s"]) - s, 1)],
                             events_clip_s=[[round(t - s, 1), e] for t, e in inc.get("events", [])
                                            if s <= t <= s + c["duration_s"]])
        c["other_incidents_in_picture"] = [
            {"id": k2, "from_clip_s": round(max(a2, s) - s, 1), "to_clip_s": round(min(b2, s + c["duration_s"]) - s, 1)}
            for (c2, k2), (a2, b2) in st["seen"].items()
            if c2 == r.name and k2 != k and a2 <= s + c["duration_s"] and b2 >= s]
        clips["crash"] = c
    res = {"made": st["stamp"], "folder": str(out.relative_to(ROOT)), "clips": clips,
           "complete": bool(normal and crash), "timed_out": timed_out,
           "note": "times: sim_s = SUMO seconds of this run; clip_s = seconds from the clip's first picture; "
                   "clock = the town's time of day (clock_speed 1: real time)"}
    cr.write_json(out / "clips.json", res)
    print(json.dumps(res, indent=2), flush=True)
    return True


def demo_incident(ff, shooter, scenario: str, st: dict, t_el: float, limit_s: float, redraw=None) -> bool:
    """--demo-incident, once per pass of the main loop: plates close-ups of cars standing at a red light, then wait for
    an accident (the marker seen from where the camera is), then watch it from the road behind and from the side,
    a screenshot at each stage and every 15 s, until it is cleared. True when done (out/<scenario>_freefly_incident.json
    written)."""
    import carla

    out = ROOT / "out" / "freefly"
    shots = st.setdefault("shots", {})

    def shot(name):
        if redraw is not None:  # the overlay (the markers) for this view first
            for _ in range(3):
                redraw()
                time.sleep(0.12)
        p = out / f"{scenario}_incident_{name}.png"
        shots[name] = p.name if shooter.grab_window(p) else None

    def look(x, y, z, back, up, yaw, side=0.0):
        a = math.radians(yaw)
        px = x - back * math.cos(a) - side * math.sin(a)
        py = y - back * math.sin(a) + side * math.cos(a)
        pz = z + up
        pitch = math.degrees(math.atan2(z + 0.5 - pz, math.hypot(px - x, py - y)))
        yw = math.degrees(math.atan2(y - py, x - px))
        return carla.Transform(carla.Location(px, py, pz), carla.Rotation(pitch, yw, 0.0))

    if t_el < DEMO_WARMUP_S:
        return False
    # 1. number plates: cars standing near the view, from behind and in front
    if st.get("plates_done") is None and t_el > DEMO_WARMUP_S + 12.0:
        n = 0
        texts = getattr(ff, "plate_text", {})
        stopped = [vid for vid, v in (ff.prev.vehicles.items() if ff.prev else []) if v.speed < 0.2
                   and ff.cs.actors.get(vid) in texts]
        for vid in stopped[:6]:
            aid = ff.cs.actors.get(vid)  # the pose thread may have let the car go since
            a = ff.world.get_actor(aid) if aid is not None else None
            if a is None:
                continue
            tf = a.get_transform()
            for name, back in (("rear", 6.5), ("front", -6.5)):
                ff.spectator.set_transform(look(tf.location.x, tf.location.y, tf.location.z, back, 1.4,
                                                tf.rotation.yaw, 1.2))
                time.sleep(0.6)
                shot(f"plate{n}_{name}")
            st.setdefault("plates", []).append(ff.plate_text.get(aid))
            n += 1
            if n >= 3:
                break
        st["plates_done"] = n
        if ff.first_tf is not None:
            ff.spectator.set_transform(ff.first_tf)
    im, scene = ff.incidents, ff.scene
    if im is None or scene is None:
        return True
    if st.get("inc") is None:
        if scene.markers:
            m = scene.markers[0]
            st["inc"], st["t_found"] = m["id"], t_el
            time.sleep(1.0)
            shot("marker_from_afar")
        elif t_el > limit_s:
            st["result"] = "no accident in time"
            return True
        return False
    m = next((m for m in scene.markers if m["id"] == st["inc"]), None)
    if m is None:  # cleared
        time.sleep(1.0)
        shot("cleared")
        return True
    inc = next((i for i in im.incidents if i.id == st["inc"]), None)
    yaw = st.get("yaw")
    if yaw is None and inc is not None and inc.wrecks:
        w = inc.wrecks[0]
        from cosim.mapping import sumo_angle_to_carla_yaw

        yaw = st["yaw"] = sumo_angle_to_carla_yaw(w.angle)
    yaw = yaw or 0.0
    tx, ty = m["x"], m["y"]
    if m["status"] == "LOADING THE WRECK":  # the loading, from the side and from behind, the truck in the picture
        views = (("load_side", 6.0, 6.0, 14.0), ("load_behind", 20.0, 7.0, 3.0), ("load_close", 4.0, 3.5, 8.0))
        tx, ty = tx + 6.0 * math.cos(math.radians(yaw)), ty + 6.0 * math.sin(math.radians(yaw))
        every = 5.0
    else:
        views = (("behind", 30.0, 12.0, 0.0), ("side", 12.0, 5.0, 9.0), ("close", 9.0, 3.0, -4.0))
        every = 6.0
    k = int((t_el - st["t_found"]) // every) % len(views)
    name, back, up, side = views[k]
    ff.spectator.set_transform(look(tx, ty, m["z"] - 1.8, back, up, yaw, side))
    gap = 6.0 if m["status"] == "LOADING THE WRECK" else 15.0
    if st.get("status") != m["status"] or t_el - st.get("t_shot", 0.0) > gap:
        time.sleep(1.5)
        st["n"] = st.get("n", 0) + 1
        shot(f"{st['n']:02d}_{m['status'].lower().replace(' ', '_')}_{name}")
        st["status"], st["t_shot"] = m["status"], t_el
    if t_el > limit_s + 400.0:
        st["result"] = "not cleared in time"
        return True
    return False


def pose_figures(ff, wait_s: float = 2.0) -> dict | None:
    """The pose process's figures since the last call: asks it to report now, and merges that report with the
    regular ones (every 10 s) that came in between (None without a pose process)."""
    q, sq = getattr(ff, "pose_q", None), getattr(ff, "stats_q", None)
    if q is None or sq is None:
        return None
    q.put(("report",))
    t = time.time()
    got = []
    while time.time() - t < wait_s:
        try:
            ps = sq.get(timeout=0.1)
        except Exception:  # noqa: BLE001 - queue.Empty
            continue
        if "fatal" in ps:
            log.error("pose process failed:\n%s", ps["fatal"])
            continue
        got.append(ps)
        if ps.get("forced"):
            break
    return merge_pose_figures(got) if got else None


def merge_pose_figures(reports: list[dict]) -> dict:
    """Several reports of the pose process as one (shares weighted by what they count)."""
    span = sum(r.get("span_s") or 0.0 for r in reports)
    pics = sum(r.get("pictures") or 0 for r in reports)
    moving = sum(r.get("moving_vehicle_pictures") or 0 for r in reports)
    out = {"span_s": round(span, 1), "pictures_per_s": round(pics / span, 1) if span else None,
           "moving_vehicle_pictures": moving,
           "judder_share": round(sum((r.get("judder_share") or 0.0) * (r.get("moving_vehicle_pictures") or 0)
                                     for r in reports) / moving, 4) if moving else None,
           "uneven_pictures": round(sum((r.get("uneven_pictures") or 0.0) * (r.get("pictures") or 0)
                                        for r in reports) / pics, 4) if pics else None,
           "pictures_waited_for_server": sum(r.get("pictures_waited_for_server") or 0 for r in reports),
           "held": sum(r.get("held") or 0 for r in reports)}
    lags: dict = {}
    for r in reports:
        for k, v in (r.get("shown_lag") or {}).items():
            if k != "median_cm":
                lags[k] = lags.get(k, 0.0) + v * (r.get("pictures") or 0)
    if pics:
        out["shown_lag"] = {k: round(v / pics, 3) for k, v in sorted(lags.items())}
    return out


def demo_native_camera(ff, shooter, scenario: str, draw, native_feed, screen_size, i: int,
                       seconds: float = 10.0) -> dict:
    """--demo: the native camera feed (H, --cctv native) for 10 s, measured: CARLA's frame rate while it is
    open, the detector's rate, how many boxes are drawn, whether CARLA's view sits at the camera, and whether
    the view goes back where it was when the camera closes."""
    cams = ff.cams
    back = ff.spectator.get_transform()
    out: dict = {}
    try:
        cams.open(i, native=True, screen=tuple(screen_size()))
        t = time.time()
        while time.time() - t < 40.0 and not cams.det and "not" not in cams.model_state:
            draw(native=native_feed(i))
            time.sleep(1 / 30)
        out["detections_after_s"] = round(time.time() - t, 1)
        pose_figures(ff)  # what came before the camera opened: not counted
        n0, t0 = len(ff.carla_dt), time.time()
        boxes, closed = [], 0
        while time.time() - t0 < seconds:
            draw(native=native_feed(i))
            boxes.append(len(cams.boxes_now(time.time(), tuple(screen_size()))))
            if cams.flown_away(ff.view_transform()):
                closed += 1
            time.sleep(1 / 30)
        span = time.time() - t0
        d = ff.carla_dt[n0:]
        out["carla_fps"] = round(len(d) / sum(d), 1) if d and sum(d) > 0 else None
        out["overlay_per_s"] = round(len(boxes) / span, 1)
        out["boxes_mean"] = round(sum(boxes) / max(len(boxes), 1), 1)
        out["detector_per_s"] = round(cams.view[2], 1) if cams.view is not None else None
        out["counts"] = cams.view[1] if cams.view is not None else None
        loc = ff.view_transform().location
        out["view_off_camera_m"] = round(math.dist((loc.x, loc.y, loc.z), cams.pose), 2)
        out["would_have_closed"] = closed
        ps = pose_figures(ff)
        if ps is not None:  # how the traffic moved in CARLA's window while the camera was open
            out["poses_while_open"] = ps
        p = ROOT / "out" / "freefly" / f"{scenario}_demo_cctv_native.png"
        out["screenshot"] = p.name if shooter.grab_window(p) else None
    except Exception as e:  # noqa: BLE001 - recorded, the demo goes on
        out["error"] = f"{e!r}"
        log.exception("native camera check")
    finally:
        cams.close()
        time.sleep(0.5)
        loc, b = ff.spectator.get_transform().location, back.location
        out["view_back_m"] = round(math.dist((loc.x, loc.y, loc.z), (b.x, b.y, b.z)), 2)
    log.info("native camera check: %s", out)
    return out


def probe_arrows(ff, shooter) -> dict:
    """--demo: what CARLA's own spectator does with the arrow keys (the viewer's arrow turning is off in
    the demo): each key held 1 s with keybd_event, the camera's move and turn measured."""
    try:
        import ctypes

        u = ctypes.windll.user32
    except Exception:  # noqa: BLE001 - not Windows
        return {}
    import carla

    out = {}
    shooter.grab_window(ROOT / "out" / "freefly" / "arrow_probe.png")  # brings CARLA to the front
    for name, vk in (("UP", 0x26), ("LEFT", 0x25)):
        ff.spectator.set_transform(carla.Transform(carla.Location(0.0, 0.0, 30.0), carla.Rotation(-10.0, 0.0, 0.0)))
        time.sleep(0.5)
        a = ff.spectator.get_transform()
        u.keybd_event(vk, 0, 0x0001, 0)  # extended key, down
        time.sleep(1.0)
        u.keybd_event(vk, 0, 0x0001 | 0x0002, 0)  # up
        time.sleep(0.3)
        b = ff.spectator.get_transform()
        out[name] = {"moved_m": round(a.location.distance(b.location), 2),
                     "yaw_deg": round((b.rotation.yaw - a.rotation.yaw + 180) % 360 - 180, 1),
                     "pitch_deg": round(b.rotation.pitch - a.rotation.pitch, 1)}
    log.info("arrow keys in CARLA's own spectator: %s", out)
    return out


def stats(d: list[float]) -> dict | None:
    d = sorted(x for x in d if x > 0)
    if not d:
        return None
    n, tot = len(d), sum(d)
    return {"frames": n, "fps_avg": round(n / tot, 1), "fps_1pct_low": round(1.0 / d[min(n - 1, int(0.99 * n))], 1),
            "ms_worst": round(1000 * d[-1], 1)}


def main(argv=None) -> int:
    args = parse_args(argv)
    (ROOT / "tools" / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "tools" / "logs" / f"freefly_{args.scenario}.log", "w",
                                                      "utf-8"), logging.StreamHandler()])
    from cosim.overlay import KEY_COLOUR, Overlay, dpi_aware

    dpi_aware()  # before any window: the overlay is placed in CARLA's own pixels
    # The pose thread (Mover) must get the interpreter within a millisecond or so of its time, not after the
    # default 5 ms slice of the SUMO thread or the overlay; and Windows' timers at 1 ms, not 15.6 ms.
    sys.setswitchinterval(0.001)
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:  # noqa: BLE001 - not Windows
        pass
    shooter = None
    if args.demo > 0:  # screenshots of the CARLA window (with the overlay) during the demo
        import importlib.util

        spec = importlib.util.spec_from_file_location("fps_test", ROOT / "tools" / "town05_fps_test.py")
        shooter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shooter)
    os.environ["SDL_VIDEO_WINDOW_POS"] = "-4000,-4000"  # off screen until laid over CARLA
    import pygame

    from cosim import daily
    from cosim.hud import Hud, HudState
    from cosim.scenario import INGOLSTADT
    from cosim.solar import carla_sun_azimuth, scenario_sun

    lower_priority()
    clock_file = ROOT / (args.clock_file or f"out/{args.scenario}_clock.json")
    clock = daily.GameClock.load(clock_file, default_s=hhmm_to_s(args.start),
                                 **({} if args.clock_speed is None else {"speed": args.clock_speed}))
    if args.set_time:
        clock.set((clock.day - 1) * daily.DAY_S + hhmm_to_s(args.set_time))
    clock.pause(True)
    clock.save()
    log.info("clock: %s %s (%s)", clock.weekday, clock.hhmm(), clock_file)

    ff = FreeFly(args, clock)
    pygame.display.init()
    pygame.font.init()
    hud = Hud()
    size = [max(hud.W, 560) + 32, hud.H + 120]
    screen = pygame.display.set_mode(tuple(size), pygame.NOFRAME)
    pygame.display.set_caption(f"{args.scenario.capitalize()} overlay")
    ov = Overlay(pygame.display.get_wm_info()["window"])
    font = pygame.font.Font(str(ROOT / "assets" / "fonts" / "Barlow-Medium.ttf"), 13)
    font_b = pygame.font.Font(str(ROOT / "assets" / "fonts" / "Barlow-SemiBold.ttf"), 15)

    def pill(text: str, f=font, bg=(16, 19, 28), fg=(226, 232, 245), pad=14, h=26) -> pygame.Surface:
        t = f.render(text, True, fg)
        surf = pygame.Surface((t.get_width() + 2 * pad, h))
        surf.fill(KEY_COLOUR)
        pygame.draw.rect(surf, bg, surf.get_rect(), border_radius=h // 2)
        surf.blit(t, (pad, (h - t.get_height()) // 2))
        return surf

    keys_pill = pill("F5 pause    F6 / F7  -1 h / +1 h    F8 next junction    F9 hide clock    H inspect a camera    "
                     "Esc quit")
    lat, lon = INGOLSTADT
    ticker = pygame.time.Clock()
    ui = {"show": True, "follow_t": 0.0, "front": False}  # front: hide when CARLA is not in front (once loaded)

    def state(**kw) -> HudState:
        _, el = scenario_sun(clock.seconds_of_day, lat, lon)
        return HudState(game_s=clock.game_s, vehicles=ff.city_vehicles, target=int(ff.target), paused=clock.paused,
                        sun_elevation=el, **kw)

    from cosim.cctv import CCTV, stamp_of

    cctv = CCTV(town=args.scenario.capitalize())

    def inspector_panel(view, name: str, model_state: str) -> pygame.Surface:
        """The inspected camera, full screen, as a live street camera feed (cosim/cctv.py): its picture
        with the detector's boxes, REC, the camera, the game clock and the counts."""
        return cctv.frame(tuple(size), view, name, model_state, stamp_of(state()))

    dirty_prev: list = []

    from cosim.incident_hud import MarkerHud

    marker_hud = MarkerHud()

    from cosim.agent_hud import AgentHud

    agent_hud = AgentHud(mode="cameras" if args.controller.startswith("agents_observed") else "exact counts")

    def markers_draw():
        """The accident markers and the junction agents' decisions (cosim/agent_hud.py), or None when neither."""
        scene = getattr(ff, "scene", None)
        ms = list(scene.markers) if scene is not None and scene.markers else []
        evs = ff.agent_events() if args.agent_hud and getattr(ff, "agents", None) is not None else []
        if not ms and not evs:
            return None
        tf = ff.view_transform()
        plates = getattr(ff, "plate_text", {})
        centres = {t: (x, y, ff.zs.get(t, 0.0)) for t, (x, y, _) in ff.centres.items()} if evs else {}
        now = ff.sim_now

        def paint(scr):
            rects = agent_hud.draw(scr, tuple(size), tf, centres, ff.agents_xy, evs, now) if evs else []
            if ms:
                rects += marker_hud.draw(scr, tuple(size), tf, ms, time.time(), plates)
            return rects

        return paint

    def draw(st: HudState, keys_help: bool = False, prompt: str | None = None, panel=None, native=None) -> None:
        """The overlay. Only the parts drawn now and the parts drawn last time are sent to the window:
        sending the whole full-screen picture each time cost the main loop."""
        nonlocal screen, dirty_prev
        full = False
        want = ov.client_size()
        if want and list(want) != size:  # the overlay covers the whole CARLA picture
            size[:] = list(want)
            screen = pygame.display.set_mode(tuple(size), pygame.NOFRAME)
            ov.restyle(pygame.display.get_wm_info()["window"])
            ui["follow_t"] = 0.0
            full = True
        rects = compose_overlay(screen, tuple(size), dirty_prev, full, KEY_COLOUR,
                                hud=(lambda scr: hud.draw(scr, st, margin=16), hud.W, hud.H) if ui["show"] else None,
                                keys_pill=keys_pill if keys_help else None,
                                prompt=(pill(prompt, font_b, bg=(255, 196, 84), fg=(20, 20, 20), h=34, pad=18)
                                        if prompt else None),
                                panel=panel, native=native, markers=markers_draw())
        if full:
            pygame.display.flip()
        else:
            pygame.display.update(rects + dirty_prev)
        dirty_prev = rects
        if time.time() - ui["follow_t"] > 0.2:  # lay it over the CARLA picture (or hide it)
            ov.follow(True, need_front=ui["front"])
            ui["follow_t"] = time.time()

    th = threading.Thread(target=ff.load, daemon=True)
    th.start()
    full = args.windowed
    # the loading screen (cosim/splash.py), started by tools/town05_freefly.bat before CARLA: it shows the load
    # from what is written here, and this window keeps out of its way until it has faded
    from cosim import splash as splash_mod
    from cosim.hud import traffic_label

    splash_on = os.environ.get("FREEFLY_SPLASH", "0") == "1"
    last_status = 0.0

    def status(ready: bool = False) -> None:
        counts = {}
        if ff.city_vehicles:
            counts["vehicles"] = int(ff.city_vehicles)
        if getattr(ff, "centres", None):
            counts["signals"] = len(ff.centres)
        if getattr(ff, "cams", None) is not None:
            counts["cameras"] = len(ff.cams.cams)
        ex = getattr(getattr(ff, "sumo", None), "extras", None)
        if ex is not None and getattr(ex, "pedestrian_target", None):
            counts["pedestrians"] = int(ex.pedestrian_target)
        hour = clock.seconds_of_day / 3600.0
        try:
            splash_mod.write_status(ff.stage, ff.progress, counts,
                                    f"{clock.weekday.capitalize()}  {clock.hhmm()}   ·   {traffic_label(hour).title()}",
                                    ready=ready, error=(ff.error.strip().splitlines()[-1][:120] if ff.error else None))
        except Exception as e:  # noqa: BLE001 - the loading screen is not essential
            log.debug("loading status: %s", e)

    while th.is_alive() or ff.error:
        pygame.event.pump()
        if not full:
            full = ov.fullscreen_carla()
        if "ESC" in ov.pressed():
            clock.save()
            pygame.quit()
            os._exit(0)  # the loader may be blocked inside CARLA or SUMO
        if time.time() - last_status > 0.2:
            status()
            last_status = time.time()
        if not splash_on:
            msg = ff.stage if not ff.error else "ERROR: " + ff.error.strip().splitlines()[-1][:60]
            if ff.progress >= 0:
                msg += f" {ff.progress * 100:.0f}%"
            draw(state(status="LOADING", note=msg))
        ticker.tick(20)
        if ff.error and not th.is_alive():
            continue
    if not ff.ready:
        pygame.quit()
        return 1
    status(ready=True)
    if splash_on:  # let the loading screen fade out (0.6 s, then 0.8 s) before this window shows
        t_end = time.time() + 1.6
        while time.time() < t_end:
            pygame.event.pump()
            ticker.tick(30)
            status(ready=True)  # again: one lost write must not leave the loading screen up (cosim/splash.py)
    if args.demo > 0 and shooter is not None:  # the view the viewer starts with (FreeFly.first_view)
        time.sleep(3.0)  # the traffic in place
        shooter.grab_window(ROOT / "out" / "freefly" / f"{args.scenario}_first_view.png")

    import carla

    from cosim.run import WEATHER

    world, cs = ff.world, ff.cs
    cams = ff.cams
    ui["front"] = args.demo <= 0  # the demo's screenshots want the overlay over CARLA whatever is in front
    weather = carla.WeatherParameters(**WEATHER["clear"])
    last_sun, last_sun_t = None, 0.0
    note, note_until = "", 0.0
    fps_shown, fps_t = 0.0, time.time()
    last_save = last_hud = last_log = time.time()
    loops = 0
    timing = {"overlay": 0.0}
    sampler = None
    if args.profile:
        from cosim.sampler import StackSampler

        threading.current_thread().name = "main"
        sampler = StackSampler()
        sampler.start()
    demo = {"high": [], "street": []} if args.demo > 0 else None
    shots: dict[str, str | None] = {}
    if demo is not None:
        world.remove_on_tick(ff._tick_cb)

        def on_tick(snap):
            if demo_seg_ref[0]:
                demo[demo_seg_ref[0]].append(snap.timestamp.delta_seconds)
            ff.note_tick(snap)
        demo_seg_ref = [None]
        ff._tick_cb = world.on_tick(on_tick)
        print(f"demo: {args.demo:.0f} s scripted flight with traffic", flush=True)

    worker = SumoThread(ff)
    worker.start()
    frame_mode = args.poses == "frame"
    ff.cs.external_moves = frame_mode  # the pose process parks and moves; the viewer only lists the parks
    pk = worker.q.get()
    ff.apply(pk)
    clock.pause(False)
    pose_proc = pose_q = stats_q = None
    if frame_mode or args.measure_judder:  # the poses (or only the measurement) once per CARLA picture
        import multiprocessing as mp

        from cosim import pose_proc as _pp

        ctx = mp.get_context("spawn")
        pose_q, stats_q = ctx.Queue(), ctx.Queue()
        pose_proc = ctx.Process(target=_pp.run, args=(args.host, args.port, pose_q, stats_q, ff.scen.frame.cx,
                                                      ff.scen.frame.cy, not frame_mode), daemon=True)
        pose_proc.start()
        ff.pose_q, ff.stats_q = pose_q, stats_q
    plate_proc, plate_in = ff.plate_proc, ff.plate_in
    if pose_q is not None:
        ff.scene_send(("noop",))  # the accident scene's messages held while loading go now
    if frame_mode:
        mover = FrameFeeder(ff, worker, clock, pk, pose_q)
    else:
        mover = Mover(ff, worker, clock, args.rate, pk, demo is not None)  # the poses, at an even rate
        if pose_q is not None:  # measurement only: it needs the segments to know who moves
            mover.measure_q = pose_q
    mover.start()
    pose_stats: list[dict] = []
    last = time.perf_counter()
    # the arrows (--arrows): mouse (default) = CARLA's own mouse look from a thread (cosim/arrow_mouse.py):
    # CARLA alone moves and turns the camera. proc / tick = set_transform from CARLA's tick callback in a
    # process of its own / in this one: the place it sends is late while Z Q S D move the camera, and the
    # camera shook (tick: backwards in 12 of 86 frames with Z and an arrow; proc: still shook, Firas).
    arrow_mouse = None
    if demo is None and args.arrows == "mouse":
        from cosim.arrow_mouse import ArrowMouse

        arrow_mouse = ArrowMouse(ov)
        arrow_mouse.start()
    dt_loop = 1.0 / args.rate
    arrow_turn = None  # time of the last arrow turn while the arrows are held
    fly_cb = None
    arrow_proc = None
    if demo is None and args.arrows in ("proc", "fly"):
        import multiprocessing as mp

        if args.arrows == "fly":
            from cosim import fly_proc as _ap
        else:
            from cosim import arrow_proc as _ap

        arrow_proc = mp.get_context("spawn").Process(target=_ap.run, args=(args.host, args.port), daemon=True)
        arrow_proc.start()
    if demo is None and args.arrows == "tick":
        from cosim.flycam import ArrowTurn

        fly = ArrowTurn(ff.spectator)

        def fly_tick(snap):
            try:
                fly.tick(snap, ov.held())
            except Exception as e:  # noqa: BLE001 - never kill CARLA's callback thread
                log.debug("arrow turn: %s", e)

        fly_cb = ff.world.on_tick(fly_tick)
    t0 = time.time()
    inc_demo: dict = {}
    clip_st: dict = {}
    enf_st: dict = {}
    agents_demo: dict = {}
    hw_demo: dict = {}
    dep_demo: dict = {}
    sig_demo: dict = {}
    ff.demo_close = args.demo_signals_close  # --demo-signals-close: one junction's poles close up
    drv_demo: dict = {}
    try:
        quit_ = False
        while not quit_:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    quit_ = True
            near_i, near_d = (None, 1e9)
            if cams is not None:
                sp_tf = ff.view_transform()
                loc = sp_tf.location
                near_i, near_d = cams.nearest(loc.x, loc.y, loc.z)
                cams.draw_markers(loc.x, loc.y)
                if cams.flown_away(sp_tf):  # native feed: you flew or turned the view off the camera
                    cams.close(restore=False)
                    note, note_until = "camera closed", time.time() + 2
            # the lamps: CARLA's own window shows them in their colours (native feed too); a sensor's picture
            # shows them as white squares
            if ff.ped_lights is not None:  # drawn by a thread of its own (cosim/ped_lights.py)
                if cams is None:
                    loc = ff.view_transform().location
                ff.ped_lights.update(loc.x, loc.y, ff.shown_tls,
                                     visible=not (cams is not None and cams.active is not None and not cams.native),
                                     remaining=ff.remaining(paused=clock.paused))
            if ff.signal_heads is not None:  # drawn by a thread of its own (cosim/signal_heads.py)
                if cams is None:
                    loc = ff.view_transform().location
                ff.signal_heads.update(loc.x, loc.y, ff.shown_tls,
                                       visible=not (cams is not None and cams.active is not None and not cams.native),
                                       remaining=ff.remaining(paused=clock.paused))
            if plate_proc is not None or ff.drivers_on:  # cars that need plates (and drivers' seats), plates made
                ff.ask_plates()
                ff.pump_plates()
            for k in ov.pressed():
                if k == "ESC":
                    quit_ = True
                elif k == "F5":
                    clock.pause()
                elif k in ("F6", "F7"):
                    clock.shift(3600.0 if k == "F7" else -3600.0)
                    note, note_until = f"jumped to {clock.hhmm()}", time.time() + 3
                elif k == "F8":
                    tid = ff.next_junction()
                    note, note_until = f"junction {tid}", time.time() + 3
                elif k == "F9":
                    ui["show"] = not ui["show"]
                elif k == "H" and cams is not None:
                    if cams.active is not None:
                        cams.close()
                    elif near_i is not None and near_d <= INSPECT_M:
                        cams.open(near_i, native=args.cctv == "native", screen=tuple(size))
            arrows = ov.held() if demo is None and args.arrows == "mouse-loop" else set()
            if arrows:  # the arrows turn the view through CARLA's own mouse look (Overlay.look)
                now_l = time.perf_counter()
                dt_l = min(now_l - arrow_turn, 0.05) if arrow_turn is not None else dt_loop
                arrow_turn = now_l
                ov.look((("RIGHT" in arrows) - ("LEFT" in arrows)) * TURN_DEG_S * dt_l,
                        (("UP" in arrows) - ("DOWN" in arrows)) * PITCH_DEG_S * dt_l)
            elif arrow_turn is not None:
                ov.look_end()
                arrow_turn = None
            if worker.error:
                note, note_until = "SUMO stopped: see tools/logs/freefly_town05.log", time.time() + 5
            if mover.error and time.time() > note_until:
                note, note_until = "pose thread failed: see tools/logs/freefly_town05.log", time.time() + 5
            now = time.perf_counter()
            dt_loop = min(now - last, 0.25)
            last = now
            if not clock.paused:
                # the sun of the game clock, at most twice a second
                if time.time() - last_sun_t > 0.5:
                    az, el = scenario_sun(clock.seconds_of_day, lat, lon)
                    if last_sun is None or abs(el - last_sun[1]) > 0.3 or abs(az - last_sun[0]) > 0.5:
                        weather.sun_altitude_angle = el
                        weather.sun_azimuth_angle = carla_sun_azimuth(az)
                        world.set_weather(weather)
                        last_sun = (az, el)
                        ff.night = el < 3.0
                    last_sun_t = time.time()
            if demo is not None and args.demo_agents > 0:
                demo_seg_ref[0] = None
                try:
                    done_a = demo_agents(ff, shooter, args.scenario, agents_demo, time.time() - t0, args.demo_agents,
                                         redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("agents demo failed")
                    done_a = True
                if done_a:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_highway > 0:
                demo_seg_ref[0] = None
                try:
                    done_h = demo_highway(ff, shooter, args.scenario, hw_demo, time.time() - t0, args.demo_highway,
                                          redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("highway demo failed")
                    done_h = True
                if done_h:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_signals > 0:
                demo_seg_ref[0] = None
                try:
                    done_s = demo_signals(ff, shooter, args.scenario, sig_demo, time.time() - t0, args.demo_signals,
                                          redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("signals demo failed")
                    done_s = True
                if done_s:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_drivers > 0:
                demo_seg_ref[0] = None
                try:
                    done_v = demo_drivers(ff, shooter, args.scenario, drv_demo, time.time() - t0, args.demo_drivers,
                                          redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("drivers demo failed")
                    done_v = True
                if done_v:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_depots > 0:
                demo_seg_ref[0] = None
                try:
                    done_d = demo_depots(ff, shooter, args.scenario, dep_demo, time.time() - t0, args.demo_depots,
                                         redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("depots demo failed")
                    done_d = True
                if done_d:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_incident > 0:
                demo_seg_ref[0] = None
                try:
                    demo_done = demo_incident(ff, shooter, args.scenario, inc_demo, time.time() - t0,
                                              args.demo_incident, redraw=lambda: draw(state(fps=fps_shown)))
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("incident demo failed")
                    inc_demo["result"] = "demo error (see the log)"
                    demo_done = True
                if demo_done:
                    im_ = ff.incidents
                    res = {"result": inc_demo.get("result", "cleared"), "clock": clock.hhmm(), "args": sys.argv[1:],
                           "plates": inc_demo.get("plates"), "screenshots": inc_demo.get("shots"),
                           "stats": im_.stats if im_ else None,
                           "incidents": [{"id": i.id, "kind": i.kind, "label": i.label, "kmh": round(i.severity * 3.6),
                                          "wrecks": [w.vid for w in i.wrecks], "events": i.events}
                                         for i in ((im_.done + im_.incidents) if im_ else [])]}
                    (ROOT / "out" / f"{args.scenario}_freefly_incident.json").write_text(json.dumps(res, indent=2),
                                                                                       encoding="utf-8")
                    demo = None
                    quit_ = True
            if ff.enf is not None:  # (31 Oct) the red-light camera site (cosim/enforcement.py)
                try:
                    if enforce_step(ff, enf_st, args):
                        quit_ = True
                    n_ = ff.__dict__.pop("enf_note", None)
                    if n_:
                        note, note_until = n_
                except Exception:  # noqa: BLE001 - reported, the viewer goes on
                    log.exception("enforcement site failed")
                    ff.enf.stop()
                    ff.enf = None
            if demo is not None and args.record_clips > 0:
                demo_seg_ref[0] = None
                try:
                    done_r = demo_record_clips(ff, clip_st, args, clock)
                except Exception:  # noqa: BLE001 - a recording's failure is reported, it never stops the viewer
                    log.exception("clip recording failed")
                    if clip_st.get("rec") is not None:
                        clip_st["rec"].close()
                    done_r = True
                if done_r:
                    demo = None
                    quit_ = True
            if demo is not None and args.demo_camera > 0:
                if time.time() - t0 >= DEMO_WARMUP_S and cams is not None:
                    tid = next(iter(ff.centres))
                    i_cam = next((k for k, c in enumerate(cams.cams) if c.tls == tid), 0)

                    def feed(scr, i_cam=i_cam):
                        sz = tuple(size)
                        return cctv.draw_native(scr, sz, cams.view, cams.names[i_cam], cams.model_state,
                                                stamp_of(state()), cams.boxes_now(time.time(), sz))

                    res = demo_native_camera(ff, shooter, args.scenario, lambda **kw: draw(state(), **kw),
                                             lambda i: feed, lambda: tuple(size), i_cam, seconds=args.demo_camera)
                    res.update(camera=cams.names[i_cam], clock=clock.hhmm(), args=sys.argv[1:])
                    (ROOT / "out" / f"{args.scenario}_freefly_camera.json").write_text(json.dumps(res, indent=2),
                                                                                       encoding="utf-8")
                    demo = None  # no flight figures to write
                    quit_ = True
                ticker.tick(UI_RATE)
                continue
            if demo is not None and args.demo_incident <= 0 and args.demo_agents <= 0 and args.demo_highway <= 0 \
                    and args.demo_depots <= 0 and args.demo_signals <= 0 and args.demo_drivers <= 0 \
                    and args.record_clips <= 0:
                el_s = time.time() - t0
                seg, tf = demo_pose(ff, el_s, args.demo)
                demo_seg_ref[0] = seg
                if el_s < args.demo:
                    ff.spectator.set_transform(tf)
                    if getattr(ff, "calls", None) is not None:
                        ff.calls.count("spectator.set_transform (demo)")
                for frac, name in ((0.5, "high"), (0.95, "street")):
                    if name not in shots and el_s >= frac * args.demo:
                        demo_seg_ref[0] = None  # the screenshot stalls the window: not measured
                        p = ROOT / "out" / "freefly" / f"{args.scenario}_demo_{name}.png"
                        shots[name] = p.name if shooter.grab_window(p) else None
                        time.sleep(0.3)
                if el_s >= args.demo and args.demo_flight_only:
                    quit_ = True
                if el_s >= args.demo and getattr(ff, "_cw", None) is None and not quit_:
                    ff._cw = crosswalk_view(ff) or False  # then: hover over a crosswalk until it shows walk
                    ff._cw_t0 = time.time()
                if el_s >= args.demo and getattr(ff, "_cw", None):
                    demo_seg_ref[0] = None
                    tf, tid, idx = ff._cw
                    ff.spectator.set_transform(tf)
                    st = ff.shown_tls.get(tid, "")
                    walking = bool(idx) and all(i < len(st) and st[i] == "G" for i in idx)
                    waited = time.time() - ff._cw_t0
                    if (walking and waited > 6.0) or waited > 150.0:
                        time.sleep(2.0 if walking else 0.0)
                        p = ROOT / "out" / "freefly" / f"{args.scenario}_demo_crosswalk_live.png"
                        shots["crosswalk_live"] = p.name if shooter.grab_window(p) else None
                        shots["crosswalk_live_walk"] = walking
                        at = ff.spectator.get_transform().location
                        shots["crosswalk_live_at"] = {"tls": tid, "want": [round(tf.location.x, 1), round(tf.location.y, 1),
                                                                         round(tf.location.z, 1)],
                                                      "spectator": [round(at.x, 1), round(at.y, 1), round(at.z, 1)]}
                        ff._cw = False
                if el_s >= args.demo and getattr(ff, "_cw", None) is False:
                    quit_ = True
                    demo_seg_ref[0] = None
                    def native_feed(i):
                        def feed(scr):
                            sz = tuple(size)
                            return cctv.draw_native(scr, sz, cams.view, cams.names[i], cams.model_state,
                                                    stamp_of(state()), cams.boxes_now(time.time(), sz))
                        return feed

                    demo_close_ups(ff, shooter, shots, args.scenario, draw=lambda **kw: draw(state(), **kw),
                                   native_feed=native_feed, screen_size=lambda: tuple(size),
                                   inspector_panel=inspector_panel)
            loops += 1
            if time.time() - fps_t >= 1.0:
                fps_shown, fps_t = ff.carla_fps(), time.time()
            inspecting = cams is not None and cams.active is not None
            native_feed = inspecting and cams.native
            # the overlay: 5 times a second; 15 with a camera open, 30 for the native feed (its boxes move)
            marking = getattr(ff, "scene", None) is not None and bool(ff.scene.markers)
            if time.time() - last_hud >= (1 / 30 if native_feed or marking else 1 / 15 if inspecting else 0.2):
                prompt = None
                if cams is not None and not inspecting and near_i is not None and near_d <= INSPECT_M:
                    prompt = f"H   Inspect traffic camera {cams.names[near_i]}"
                tm = time.perf_counter()
                panel = native = None
                if native_feed:
                    i_act = cams.active

                    def native(scr, i_act=i_act):
                        sz = tuple(size)
                        return cctv.draw_native(scr, sz, cams.view, cams.names[i_act], cams.model_state,
                                                stamp_of(state()), cams.boxes_now(time.time(), sz))
                elif inspecting:
                    panel = inspector_panel(cams.view, cams.names[cams.active], cams.model_state)
                draw(state(fps=fps_shown, note=note if time.time() < note_until else ""),
                     keys_help=time.time() - t0 < 25.0 or clock.paused, prompt=prompt, panel=panel, native=native)
                timing["overlay"] += time.perf_counter() - tm
                last_hud = time.time()
            if time.time() - last_save > 2.0:
                clock.save()
                last_save = time.time()
            if time.time() - last_log > 10.0:
                span = time.time() - last_log
                updates, mt = mover.take()
                if stats_q is not None:  # the pose process's last 10 s (cosim/pose_proc.py)
                    got = 0
                    try:
                        while True:
                            ps = stats_q.get_nowait()
                            got += 1
                            if "fatal" in ps:
                                log.error("pose process failed:\n%s", ps["fatal"])
                                continue
                            pose_stats.append(ps)
                            log.info("poses %s: %s", "measured" if ps["measure_only"] else "per picture",
                                     {k: v for k, v in ps.items() if k not in ("error", "measure_only")})
                            if ps.get("error"):
                                log.error("pose process: %s", ps["error"])
                    except Exception as e:  # noqa: BLE001 - nothing more (queue.Empty), or not readable
                        if type(e).__name__ != "Empty":
                            log.warning("pose statistics not readable: %r", e)
                    if not got and pose_proc is not None and time.time() - t0 > 25.0:
                        log.warning("no statistics from the pose process in 10 s (alive %s, exit code %s)",
                                    pose_proc.is_alive(), pose_proc.exitcode)
                    if frame_mode and pose_stats:
                        updates = pose_stats[-1]["posed_per_s"] * span
                log.info("%s  CARLA %.1f fps  updates %.0f/s  vehicles %d in town, %d drawn, %d pedestrians, "
                         "SUMO late %d; ms/s: move %.0f, apply %.0f (pose thread), overlay %.0f (main loop, %.0f/s)",
                         clock.hhmm(), fps_shown, updates / span, ff.city_vehicles, len(cs.actors), len(cs.walkers),
                         mover.held, 1000 * mt["move"] / span, 1000 * mt["apply"] / span,
                         1000 * timing["overlay"] / span, loops / span)
                with mover.lock:
                    trimmed = cs.trim()
                if trimmed:
                    log.info("destroyed %d surplus parked vehicles", trimmed)
                hw = getattr(ff, "highway", None)
                if hw is not None:
                    log.info("ring road: %s cars on its loops (target %d each), %d started, %d taken off",
                             [len(m) for m in hw.cars], hw.target(ff.clock.seconds_of_day), hw.added, hw.removed)
                if ff.incidents is not None:
                    log.info("accidents: %s; now %s", ff.incidents.stats,
                             [(i.id, i.kind, i.label, i.phase) for i in ff.incidents.incidents])
                log.info("apply, ms in the last 10 s: %s", {k: (round(1000 * v) if isinstance(v, float) else v)
                                                            for k, v in getattr(ff, "apply_t", {}).items()})
                ff.apply_t = {}
                log.info("SUMO thread, ms in the last 10 s: %s",
                         {k: (round(1000 * v) if isinstance(v, float) else v)
                          for k, v in sorted(getattr(ff, "produce_t", {}).items(), key=lambda kv: -kv[1])})
                ff.produce_t = {}
                if getattr(ff, "calls", None) is not None:
                    tl_calls, ff.cs.tl_set_calls = getattr(ff.cs, "tl_set_calls", 0), 0
                    lamp_calls = 0
                    if ff.ped_lights is not None:
                        lamp_calls, ff.ped_lights.calls = getattr(ff.ped_lights, "calls", 0), 0
                    log.info("calls to the server in the last %.1f s (calls, commands): %s; traffic light set_state %d, "
                             "lamp draws %d", span, ff.calls.take(), tl_calls, lamp_calls)
                if sampler is not None:
                    log.info("CPU s in the last %.1f s: %s", span,
                             dict(sorted(sampler.cpu().items(), key=lambda kv: -kv[1])))
                    for name, r in sorted(sampler.report(top=10).items()):
                        log.info("profile %s (%d samples): inner %s", name, r["samples"],
                                 ", ".join(f"{w} {100 * f:.0f}%" for w, f in r["exclusive"]))
                        log.info("profile %s: on stack %s", name,
                                 ", ".join(f"{w} {100 * f:.0f}%" for w, f in r["inclusive"]))
                loops, last_log = 0, time.time()
                timing = {"overlay": 0.0}
            ticker.tick(UI_RATE)
    finally:
        if clip_st.get("rec") is not None:  # --record-clips stopped early (Esc, an error): the MP4s closed properly
            clip_st["rec"].close()
        if getattr(ff, "enf", None) is not None:  # the site's video closed properly, its last records written
            try:
                ff.enf.stop()
                ff.enf.tick(*enforce_lookups(ff))
                if enf_st.get("agent_ctl") is not None:  # the agent writes its last cases and stops
                    enf_st["agent_ctl"].put(("stop",))
                    enf_st["agent"].join(timeout=10.0)
                    if enf_st["agent"].is_alive():
                        enf_st["agent"].terminate()
            except Exception:  # noqa: BLE001
                log.exception("closing the enforcement site")
        mover.stop = True
        mover.join(timeout=2.0)
        if plate_proc is not None:
            try:
                plate_in.put(("stop",))
                plate_proc.join(timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
            if plate_proc.is_alive():
                plate_proc.terminate()
            plate_in.cancel_join_thread()
        if pose_proc is not None:
            try:
                pose_q.put(("stop",))
                pose_proc.join(timeout=3.0)
            except Exception:  # noqa: BLE001
                pass
            if pose_proc.is_alive():
                pose_proc.terminate()
            pose_q.cancel_join_thread()  # never wait at exit to write out what the process did not read
        ov.look_end()  # never leave the right mouse button held
        try:
            ff.cs.stop_ground()
        except Exception:  # noqa: BLE001 - never loaded
            pass
        if arrow_mouse is not None:
            log.info("arrow mouse: %d presses, %d releases, longest gap %.0f ms", arrow_mouse.presses,
                     arrow_mouse.releases, 1000 * arrow_mouse.max_gap_s)
            arrow_mouse.stop = True
            arrow_mouse.join(timeout=1.0)
            arrow_mouse.release()
        if arrow_proc is not None:
            arrow_proc.terminate()
        if fly_cb is not None:
            ff.world.remove_on_tick(fly_cb)
        clock.save()
        worker.halt()
        if demo is not None:
            g = sorted(mover.gaps[60:]) or [0.0]
            res = {"scenario": args.scenario, "seconds": args.demo, "rate": args.rate, "steep_poses": mover.steep,
                   "turn_reversals": mover.reversals, "vanished_in_view": ff.vanished_in_view,
                   "update_gap_ms_median": round(1000 * g[len(g) // 2], 1),
                   "update_gap_ms_p99": round(1000 * g[min(len(g) - 1, int(0.99 * len(g)))], 1),
                   "sumo_late": mover.held, "pedestrians_drawn_max": ff.walkers_max, "walks_extended": ff.walks_extended, "walk_phases": [ff.sumo.walk.served, ff.sumo.walk.skipped] if ff.sumo.walk else None, "preemptions": ff.preempted,
                   "trips_extended": ff.extended, "emergency_vehicles_seen": len(ff.emergency_seen),
                   "line_buses_seen": len(ff.buses_seen), "bus_shelters": len(getattr(ff, "shelters", [])),
                   "traffic_cameras": len(cams.cams) if cams else 0,
                   "actors_reused": getattr(cs, "reused", 0), "vehicles_in_town": ff.city_vehicles,
                   "vehicles_drawn": len(cs.actors), "clock": clock.hhmm(),
                   "poses": args.poses, "pose_stats": pose_stats,
                   "judder_share": (round(sum(p["judder_share"] * p["moving_vehicle_pictures"] for p in pose_stats
                                              if p["judder_share"] is not None)
                                          / max(sum(p["moving_vehicle_pictures"] for p in pose_stats), 1), 4)
                                    if pose_stats else None),
                   "high": stats(demo["high"]), "street": stats(demo["street"]),
                   "all": stats(demo["high"] + demo["street"]), "screenshots": shots}
            out = ROOT / "out" / (f"{args.scenario}_freefly_flight.json" if args.demo_flight_only else f"{args.scenario}_freefly_demo.json")
            out.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(json.dumps(res), flush=True)
        log.info("quit at %s; vehicles added %d, CARLA actors reused %d, poses over 30 deg pitch %d, trips extended %d, "
                 "vanished in view %d, emergency vehicles %d, pre-emptions %d", clock.hhmm(), ff.pop.added,
                 getattr(ff.cs, "reused", 0), mover.steep, ff.extended, ff.vanished_in_view, len(ff.emergency_seen),
                 ff.preempted)
        ff.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    try:
        sys.exit(main())
    except Exception:
        # into the log as well: the terminal closes when town05_freefly.bat goes on to stop CARLA
        log.exception("free-fly failed")
        raise

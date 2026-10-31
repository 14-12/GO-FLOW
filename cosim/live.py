"""Town12 live: watch the city in a window, with a clock that runs the day.

    python -m cosim.live                      # the clock continues from where it was left
    python -m cosim.live --set-time 06:30     # start the clock at 06:30 (day kept)

GTA time: 1 real minute = 1 game hour (cosim/daily.GAME_SPEED). The clock drives the sun (and
with it CARLA's street lights, which switch themselves on in the dark, and the cars' headlights)
and how much traffic the city carries. The cars themselves drive at normal speed: SUMO runs in
real time (one 0.5 s step per 0.5 s of wall time, if the PC keeps up), as in every run here.

How the traffic follows a clock 60x faster than the cars: the number of vehicles on the road is
kept on the curve measured by the 24-hour SUMO study (tools/day_study_town12.py; at every minute
of the day, how many vehicles a day of the PROFILE demand puts on Town12's roads). Below the curve,
new trips are added (from the benchmark's own routes); above it, vehicles out of the camera's sight
(more than --draw-radius + 150 m away) are taken off the road. So what the camera sees at 08:00 or
at 17:30 is the density the full simulation has then; the day itself is studied, at real speed,
by the 24-hour run.

The clock is written to --clock-file every 2 s and when the window closes; the next start
continues from there.

Keys: P pause, Left / Right (or [ ] - +) one game hour back / forward, C next camera, H hide the clock, F12
screenshot, Esc or closing the window: quit (the time is saved).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("live")

# Default cameras: the junctions with the most vehicles within 150 m in the benchmark (peak) hour
# (SUMO alone, actuated, minutes 10-25: 4450 25.5, 13615 16.4, 14055 16.2; downtown 10577 only 4.6).
KNOWN_ARCS = {  # clear orbit arcs measured by cosim/run.py camera_arcs (logs of 15 and 16 Oct)
    "10577": (190.0, 10.0),
    "14055": (340.0, 70.0),
    "4450": (0.0, 360.0),
    "13615": (170.0, 65.0),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default="town12")
    ap.add_argument("--controller", default="actuated")
    ap.add_argument("--junctions", default="4450:Country crossroads,14055:Boulevard,13615:Grand junction,10577:Downtown",
                    help="cameras, id:name,... (C switches between them)")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fov", type=float, default=75.0)
    ap.add_argument("--draw-radius", type=float, default=250.0)
    ap.add_argument("--clock-file", default="out/town12_clock.json")
    ap.add_argument("--set-time", default=None, help="HH:MM: start the clock here (the day is kept)")
    ap.add_argument("--start", default="07:00", help="HH:MM of day 1 when there is no saved clock")
    ap.add_argument("--study", default="out/large_map/day/town12_day.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--sumo-port", type=int, default=8813)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill-s", type=float, default=60.0, help="simulated seconds run before the first picture")
    ap.add_argument("--shot-every", type=float, default=0.0, help="real seconds between automatic screenshots (0: off)")
    ap.add_argument("--quit-after", type=float, default=0.0, help="quit after this many real seconds of live view")
    ap.add_argument("--shots", default="out/large_map/live")
    return ap.parse_args(argv)


def hhmm_to_s(text: str) -> float:
    h, m = (int(v) for v in text.split(":"))
    return h * 3600.0 + m * 60.0


class Population:
    """Keeps the number of vehicles on SUMO's roads on the day's curve (see the module doc)."""

    def __init__(self, conn, pool, route_ids: dict[str, list[str]], curve: list[float], seed: int = 1,
                 max_add: int = 25, max_add_anywhere: int = 100, max_remove: int = 40):
        from cosim import daily

        self.daily = daily
        self.conn = conn
        self.pool = pool
        self.route_ids = route_ids
        self.curve = curve
        self.rng = random.Random(seed)
        self.types = list(pool.per_hour)
        self.weights = [pool.per_hour[t] for t in self.types]
        self.max_add, self.max_add_anywhere, self.max_remove = max_add, max_add_anywhere, max_remove
        self.n = 0
        self.pending = 0
        self.added = self.removed = 0
        self._k = 0
        self.type_for = None  # (vehicle id, demand type) -> the type it is inserted with (cosim/vmodels.Typer)

    def typed(self, vid: str, t: str) -> str:
        return self.type_for(vid, t) if self.type_for is not None else t

    def target(self, seconds_of_day: float) -> float:
        return self.daily.target_at(self.curve, seconds_of_day)

    def add(self, k: int, anywhere: bool = False) -> None:
        """k new vehicles. anywhere=False: at the start of a benchmark route (a new trip);
        True: somewhere along one (to fill the city at once: start, or a jump in time)."""
        c = self.conn
        for _ in range(k):
            t = self.rng.choices(self.types, self.weights)[0]
            i = self.rng.randrange(len(self.route_ids[t]))
            vid = f"live{self.n}"
            self.n += 1
            try:
                if anywhere:
                    edges = self.pool.routes[t][i].split()
                    j = self.rng.randrange(max(1, len(edges) - 1))
                    rid = f"livep{self.n}"
                    c.route.add(rid, edges[j:])
                    c.vehicle.add(vid, rid, typeID=self.typed(vid, t), depart="now", departLane="best",
                                  departPos="random_free", departSpeed="max")
                else:
                    c.vehicle.add(vid, self.route_ids[t][i], typeID=self.typed(vid, t), depart="now", departLane="best",
                                  departSpeed="max")
                self.added += 1
            except Exception as e:  # noqa: BLE001 - one bad insertion must not stop the city
                log.debug("add %s failed: %s", vid, e)

    def step(self, running: dict, seconds_of_day: float, may_remove) -> float:
        """After a SUMO step. running: SUMO id -> Vehicle; may_remove(vid, v): out of sight."""
        target = self.target(seconds_of_day)
        self._k += 1
        if self._k % 4 == 1:
            self.pending = len(self.conn.simulation.getPendingVehicles())
        deficit = target - len(running) - self.pending
        if deficit > 0:
            anywhere = deficit > 0.25 * target  # far below (a jump in time): fill along the routes
            k = int(min(deficit, self.max_add_anywhere if anywhere else self.max_add))
            self.add(k, anywhere=anywhere)
            self.pending += k
        excess = len(running) - target
        if excess > max(5.0, 0.02 * target):
            k = int(min(excess, self.max_remove))
            for vid, v in running.items():
                if k <= 0:
                    break
                if may_remove(vid, v):
                    try:
                        self.conn.vehicle.remove(vid)
                        self.removed += 1
                        k -= 1
                    except Exception:  # noqa: BLE001 - already gone
                        pass
        return target


class Live:
    def __init__(self, args):
        self.args = args
        self.stage, self.progress = "Starting", -1.0
        self.error: str | None = None
        self.ready = False
        self.quit = False
        self.cams = []
        for part in args.junctions.split(","):
            tid, _, name = part.partition(":")
            self.cams.append((tid.strip(), name.strip() or tid.strip()))
        self.cam_i = 0

    # ---------------------------------------------------------------- loading (a thread)
    def load(self) -> None:
        try:
            self._load()
            self.ready = True
        except Exception:  # noqa: BLE001 - shown on screen and logged
            self.error = traceback.format_exc()
            log.error("loading failed:\n%s", self.error)

    def _load(self) -> None:
        import carla

        from cosim import daily
        from cosim import scenario as scenario_mod
        from cosim.carla_side import CarlaSide, is_large_map
        from cosim.run import Poser, camera_arcs, ground_z, settle
        from cosim.sumo_side import SumoSide

        a = self.args
        self.stage = "Reading the Town12 road network"
        scen = scenario_mod.load(a.scenario, ROOT)
        self.scen = scen
        self.centres = scenario_mod.tls_centres(scen.net_file, scen.frame)
        for tid, _ in self.cams:
            if tid not in self.centres:
                raise SystemExit(f"--junctions: {tid} is not a signalised junction of {a.scenario}")

        self.stage = "Connecting to CARLA"
        client = carla.Client(a.host, a.port)
        client.set_timeout(900.0)
        self.client = client
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != scen.carla_map:
            self.stage = f"Loading {scen.carla_map} (about a minute and a half)"
            world = client.load_world(scen.carla_map)
        self.world = world
        self.large = is_large_map(world.get_map())
        landmark_ids = [lm.id for lm in world.get_map().get_all_landmarks_of_type("1000001")]

        self.stage = "Starting SUMO"
        pool = daily.load_pool(scen.folder)
        pool_file = ROOT / "out" / "large_map" / "live" / f"{a.scenario}.pool.rou.xml"
        route_ids = daily.write_pool_routes(pool, pool_file)
        study = None
        sp = ROOT / a.study
        if sp.is_file():
            study = json.loads(sp.read_text(encoding="utf-8"))
        self.curve_source = "24 h study" if study and len(study.get("running_per_minute", [])) >= 1440 else "estimate"
        curve = daily.population_curve(study, pool)
        self.sumo = SumoSide(scen.folder, a.controller, seed=a.seed, port=a.sumo_port,
                             extra_args=["--route-files", str(pool_file), "--end", "100000000"])
        self.all_tls = list(self.sumo.tls_ids)
        self.pop = Population(self.sumo.conn, pool, route_ids, curve, seed=a.seed)
        sod = self.clock.seconds_of_day
        target = self.pop.target(sod)
        self.stage = f"Filling the city for {self.clock.hhmm()}: {target:.0f} vehicles"
        self.pop.add(int(target), anywhere=True)
        n_fill = int(a.fill_s / self.sumo.step_length)
        for k in range(n_fill):
            cur = self.sumo.step()
            self.pop.step(cur.vehicles, sod, lambda vid, v: True)
            self.city_vehicles = len(cur.vehicles)
            self.progress = (k + 1) / n_fill
        self.prev = None
        self.progress = -1.0
        log.info("city filled: %d vehicles for %s (target %.0f, curve: %s)", len(cur.vehicles), self.clock.hhmm(),
                 target, self.curve_source)

        self.stage = "Large-map settings (the server takes a minute or two)"
        fps_nominal = 10.0  # CARLA's fixed step; the picture's own rate is whatever the GPU gives
        self.carla_side = CarlaSide(client, world, scen.frame, fps=fps_nominal,
                                    links=scen.links(landmark_ids))
        if self.large:
            settle(self.carla_side.tick)
        self.poser = Poser(scen.frame, self.carla_side.ground)
        self.cmap = world.get_map()

        arc_file = ROOT / "out" / "large_map" / "live" / "arcs.json"
        arcs = dict(KNOWN_ARCS)
        try:
            arcs.update({k: tuple(v) for k, v in json.loads(arc_file.read_text(encoding="utf-8")).items()})
        except (OSError, ValueError):
            pass
        todo = [tid for tid, _ in self.cams if tid not in arcs]
        if todo:
            self.stage = f"Finding a clear camera view at {len(todo)} junction(s)"
            arcs.update(camera_arcs(world, self.centres, "tour", ",".join(todo), tick=self.carla_side.tick,
                                    fov=a.fov))
            arc_file.write_text(json.dumps({k: list(v) if v else None for k, v in arcs.items()}, indent=1),
                                encoding="utf-8")
        self.arcs = arcs
        self.zs = {tid: ground_z(self.cmap, *self.centres[tid][:2]) for tid, _ in self.cams}
        self.stage = f"Streaming the city around {self.cams[0][1]}"
        self.move_camera(0, ui=None)
        from cosim.camera import Recorder

        w, h = (int(v) for v in a.res.lower().split("x"))
        self.rec = Recorder(world, w, h, fov=a.fov)
        self.stage = "Ready"

    # ---------------------------------------------------------------- camera
    def focus(self) -> tuple[float, float]:
        cx, cy, _ = self.centres[self.cams[self.cam_i][0]]
        return cx, cy

    def move_camera(self, i: int, ui) -> None:
        """Point the camera at junction i: stream the tiles round it in (large maps), and from
        now on draw the vehicles near it and send it its signals' states."""
        from cosim.run import stream_in

        self.cam_i = i % len(self.cams)
        tid = self.cams[self.cam_i][0]
        cx, cy = self.focus()
        r = self.args.draw_radius + 150.0
        near = []
        for t, (x, y, _) in self.centres.items():
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                near.append(t)
        self.sumo.tls_ids = [t for t in self.all_tls if t in near]  # only these are read each step
        if self.large:
            tick = self.carla_side.tick if ui is None else (lambda: (ui(), self.carla_side.tick())[1])
            stream_in(self.world, tick, cx, cy, self.zs[tid])
        log.info("camera at %s (%s), %d signals within %.0f m", tid, self.cams[self.cam_i][1], len(near), r)

    def camera_pose(self, t: float):
        from cosim.camera import arc_pose
        from cosim.run import lift

        tid = self.cams[self.cam_i][0]
        cx, cy = self.focus()
        return lift(arc_pose(cx, cy, t, self.arcs.get(tid)), self.zs[tid])

    # ---------------------------------------------------------------- one SUMO step
    def sumo_step(self) -> None:
        from cosim.convert import light_state
        from cosim.run import within

        cur = self.sumo.step()
        fx, fy = self.focus()
        R = self.args.draw_radius
        frame = self.scen.frame
        drawn = self.carla_side.actors

        def may_remove(vid, v):
            if vid in drawn:
                return False
            x, y = frame.to_carla_xy(v.x, v.y)
            return (x - fx) ** 2 + (y - fy) ** 2 > (R + 150.0) ** 2

        self.target = self.pop.step(cur.vehicles, self.clock.seconds_of_day, may_remove)
        self.city_vehicles = len(cur.vehicles)
        cs = self.carla_side
        cs.destroy(cur.arrived)
        cur = within(cur, frame, fx, fy, R, set(cs.actors))
        prev = within(self.prev, frame, fx, fy, R + 200.0, set()) if self.prev else None
        cs.destroy([vid for vid in cs.actors if vid not in cur.vehicles])
        self.poser.begin_step(prev, cur)
        first = self.poser.poses(0.0, self.sumo.step_length)
        cs.spawn([(vid, v.vclass, first[vid]) for vid, v in cur.vehicles.items()
                  if vid not in cs.actors and vid not in cs.failed])
        cs.set_traffic_lights(cur.tls)
        self.lights = {vid: light_state(v.signals, self.night) for vid, v in cur.vehicles.items()}
        self.prev = cur


def main(argv=None) -> int:
    args = parse_args(argv)
    (ROOT / "tools" / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "tools" / "logs" / "live_town12.log", "w", "utf-8"),
                                  logging.StreamHandler()])
    import pygame

    from cosim import daily
    from cosim.hud import Hud, HudState
    from cosim.solar import carla_sun_azimuth, scenario_sun

    clock = daily.GameClock.load(ROOT / args.clock_file, default_s=hhmm_to_s(args.start))
    if args.set_time:
        clock.set((clock.day - 1) * daily.DAY_S + hhmm_to_s(args.set_time))
    clock.pause(True)  # stands still while loading
    clock.save()
    log.info("clock: day %d %s %s (%s)", clock.day, clock.weekday, clock.hhmm(), args.clock_file)

    live = Live(args)
    live.clock = clock
    live.night = False
    live.target = 0.0
    live.city_vehicles = 0
    live.lights = {}

    pygame.display.init()
    pygame.font.init()
    w, h = (int(v) for v in args.res.lower().split("x"))
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Town12 live - SUMO traffic, CARLA pictures")
    hud = Hud()
    ticker = pygame.time.Clock()
    lat, lon = live_lat_lon(args)

    def state(**kw) -> HudState:
        _, el = scenario_sun(clock.seconds_of_day, lat, lon)
        return HudState(game_s=clock.game_s, vehicles=live.city_vehicles, target=int(live.target),
                        paused=clock.paused, sun_elevation=el, **kw)

    # ---- loading screen while the thread loads
    th = threading.Thread(target=live.load, daemon=True)
    th.start()
    t0 = time.time()
    last_save = time.time()
    while th.is_alive() or live.error:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                clock.save()
                log.info("closed while loading; clock saved at %s", clock.hhmm())
                pygame.quit()
                os._exit(0)  # the loader thread may be blocked inside CARLA or SUMO
        stage = live.stage if not live.error else "Could not start: " + live.error.strip().splitlines()[-1][:120]
        hud.loading(screen, state(status="LOADING"), stage, live.progress, time.time() - t0,
                    subtitle="Ingolstadt clock  ·  320 signals  ·  SUMO traffic, CARLA pictures")
        pygame.display.flip()
        ticker.tick(30)
        if live.error and not th.is_alive():
            continue  # stay on the error until the window is closed
    if not live.ready:
        pygame.quit()
        return 1

    # ---- live
    world, cs, rec, sumo = live.world, live.carla_side, live.rec, live.sumo
    import carla

    from cosim.run import WEATHER

    weather = carla.WeatherParameters(**WEATHER["clear"])
    last_sun = None
    clock.pause(False)
    show_hud = True
    acc = 0.0
    cam_t = 0.0
    last = time.perf_counter()
    live_t0 = time.time()
    next_shot = time.time() + args.shot_every if args.shot_every > 0 else None
    frame_ema = None
    sim_ema = 1.0
    note, note_until = "", 0.0
    surf = None
    shots = ROOT / args.shots
    shots.mkdir(parents=True, exist_ok=True)

    def shot(tag: str = "") -> Path:
        p = shots / f"live_day{clock.day}_{clock.hhmm().replace(':', '')}{tag}.png"
        pygame.image.save(screen, str(p))
        log.info("screenshot %s", p)
        return p

    def pump():  # keeps the window alive during a camera move
        pygame.event.pump()

    try:
        while not live.quit:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    live.quit = True
                elif ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k == pygame.K_ESCAPE:
                        live.quit = True
                    elif k == pygame.K_p:
                        clock.pause()
                    elif k in (pygame.K_RIGHT, pygame.K_LEFT) or ev.unicode in ("]", "[", "+", "-"):
                        # arrows work on every keyboard layout; [ ] + - where the layout has them
                        fwd = k == pygame.K_RIGHT or ev.unicode in ("]", "+")
                        clock.shift(3600.0 if fwd else -3600.0)
                        note, note_until = f"jumped to {clock.hhmm()}", time.time() + 3
                    elif k == pygame.K_h:
                        show_hud = not show_hud
                    elif k == pygame.K_F12:
                        p = shot()
                        note, note_until = f"saved {p.name}", time.time() + 3
                    elif k == pygame.K_c and len(live.cams) > 1:
                        was = clock.paused
                        clock.pause(True)
                        if surf is not None:
                            screen.blit(surf, (0, 0))
                        hud.draw(screen, state(note=f"moving to {live.cams[(live.cam_i + 1) % len(live.cams)][1]}..."))
                        pygame.display.flip()
                        live.move_camera(live.cam_i + 1, ui=pump)
                        live.prev = None
                        clock.pause(was)
                        cam_t = 0.0
                        last = time.perf_counter()
            now = time.perf_counter()
            dt = min(now - last, 0.5)
            last = now

            if not clock.paused:
                # traffic in real time: one SUMO step per step length of wall time (at most two
                # per picture; if the PC is slower, the traffic is too, and the strip says so)
                acc += dt
                cam_t += dt
                steps = 0
                while acc >= sumo.step_length and steps < 2:
                    live.sumo_step()
                    acc -= sumo.step_length
                    steps += 1
                if acc >= sumo.step_length:
                    acc = sumo.step_length * 0.999
                if steps:
                    sim_ema = 0.9 * sim_ema + 0.1 * min(1.0, steps * sumo.step_length / max(dt, 1e-3))
                # the sun (and street lights) of the game clock
                az, el = scenario_sun(clock.seconds_of_day, lat, lon)
                if last_sun is None or abs(el - last_sun[1]) > 0.1 or abs(az - last_sun[0]) > 0.2:
                    weather.sun_altitude_angle = el
                    weather.sun_azimuth_angle = carla_sun_azimuth(az)
                    world.set_weather(weather)
                    last_sun = (az, el)
                    live.night = el < 3.0
                if live.prev is not None:
                    s = acc / sumo.step_length
                    cs.move(live.poser.poses(s, sumo.step_length), live.lights)
                rec.place(live.camera_pose(cam_t))
                fid = cs.tick()
                img = grab(rec, fid)
                surf = pygame.image.frombuffer(bytes(img.raw_data), (img.width, img.height), "BGRA").convert()
                if surf.get_size() != screen.get_size():
                    surf = pygame.transform.smoothscale(surf, screen.get_size())
                ft = time.perf_counter() - now
                frame_ema = ft if frame_ema is None else 0.9 * frame_ema + 0.1 * ft
            if surf is not None:
                screen.blit(surf, (0, 0))
            else:
                screen.fill((10, 12, 18))
            if show_hud:
                extra = []
                if sim_ema < 0.9 and not clock.paused:
                    extra.append(f"traffic at {sim_ema:.1f}x real speed")
                if time.time() < note_until:
                    extra.append(note)
                help_a = int(max(0.0, min(1.0, (18.0 - (time.time() - live_t0)) / 3.0)) * 255)
                hud.draw(screen, state(fps=(1.0 / frame_ema) if frame_ema else 0.0,
                                       camera=live.cams[live.cam_i][1], note="   ·   ".join(extra),
                                       help_alpha=help_a))
            pygame.display.flip()
            if time.time() - last_save > 2.0:
                clock.save()
                last_save = time.time()
            if next_shot and time.time() >= next_shot:
                shot()
                next_shot += args.shot_every
            if args.quit_after and time.time() - live_t0 > args.quit_after:
                live.quit = True
    finally:
        clock.save()
        log.info("quit at day %d %s; clock saved; vehicles added %d, removed out of sight %d",
                 clock.day, clock.hhmm(), live.pop.added, live.pop.removed)
        for closer in (lambda: rec.close(), lambda: cs.close(), lambda: sumo.close()):
            try:
                closer()
            except Exception:  # noqa: BLE001 - closing anyway
                pass
        pygame.quit()
    return 0


def live_lat_lon(args) -> tuple[float, float]:
    from cosim.scenario import INGOLSTADT

    return INGOLSTADT


def grab(rec, fid, timeout: float = 30.0):
    """This tick's image from the Recorder's camera."""
    while True:
        img = rec.frames.get(timeout=timeout)
        if img.frame >= fid:
            return img


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())

"""cosim/route_guidance.py (the route agent driving the followed car), cosim/route_minimap.py (its mini-map) and the
follow camera (cosim/pose_proc.ChaseCam): SUMO on the Town05 network, no CARLA."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim import route_guidance as rg  # noqa: E402


@dataclass
class V:
    x: float
    y: float
    angle: float
    speed: float
    vclass: str = "passenger"


@dataclass
class Cur:
    time: float
    vehicles: dict
    arrived: list = field(default_factory=list)


@dataclass
class Inc:
    id: int
    kind: str
    edge: str
    closed: list
    phase: str = "secured"
    police_record: dict = field(default_factory=dict)


class IM:
    def __init__(self):
        self.incidents = []


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    return rg.RouteGuide(log_dir=tmp_path_factory.mktemp("rg"))


def test_incident_alerts_only_once_reported(guide):
    im = IM()
    im.incidents.append(Inc(1, "rear-end", "2", [("2_0", ""), ("2_1", "")], police_record={"sent": None}))
    assert rg.incident_alerts(im, guide.router, guide.adapter, 100.0) == []          # not reported yet
    im.incidents[0].police_record["sent"] = 90.0
    al = rg.incident_alerts(im, guide.router, guide.adapter, 100.0)
    assert len(al) == 1 and al[0]["route"] == "2" and al[0]["closed_lanes"] == 2
    assert al[0]["penalty"] == 50.0 if guide.router.roads["2"].lanes == 2 else al[0]["penalty"] > 1
    im.incidents[0].phase = "closed"
    assert rg.incident_alerts(im, guide.router, guide.adapter, 100.0) == []          # road open again


def test_old_forecasts_are_not_used(guide):
    msg = {"t": 100.0, "nodes": {"1427-1": {"fill": [0.6, 0.8, 0.9], "p_long": 0.7}}}
    assert len(rg.forecast_alerts(msg, 120.0, guide.adapter, guide.amap)) == 1
    assert rg.forecast_alerts(msg, 100.0 + rg.FORECAST_MAX_AGE_S + 1, guide.adapter, guide.amap) == []


def _sumo(port_label="rgtest"):
    import sumolib
    import traci

    scen = ROOT / "scenarios" / "town05"
    cmd = [sumolib.checkBinary("sumo"), "-n", str(scen / "town05.net.xml"), "-r", str(scen / "town05.passenger.rou.xml"),
           "-a", str(scen / "town05.coordinated.add.xml"), "--step-length", "0.5", "--no-step-log", "--no-warnings",
           "--seed", "11", "--time-to-teleport", "-1"]
    traci.start(cmd, label=port_label)
    return traci.getConnection(port_label)


def _cur(c, t, arrived):
    vs = {}
    for vid in c.vehicle.getIDList():
        x, y = c.vehicle.getPosition(vid)
        vs[vid] = V(x, y, c.vehicle.getAngle(vid), c.vehicle.getSpeed(vid), c.vehicle.getVehicleClass(vid))
    return Cur(t, vs, arrived)


@pytest.mark.slow
def test_the_followed_car_takes_the_agents_route_avoids_a_reported_crash_and_arrives(tmp_path):
    guide = rg.RouteGuide(log_dir=tmp_path)
    c = _sumo()
    try:
        for _ in range(240):
            c.simulationStep()
        r = guide.router
        pick = None
        for vid in c.vehicle.getIDList():      # a car with a long way to go, whose route has an alternative
            route, idx = list(c.vehicle.getRoute(vid)), c.vehicle.getRouteIndex(vid)
            if len(route) - idx >= 6 and c.vehicle.getRoadID(vid) == route[idx]:
                mid = route[idx + 3]
                alt = r.best(route[idx], route[-1], [{"route": mid, "penalty": 50.0}])
                if alt and mid not in alt:
                    pick = (vid, route, idx, mid)
                    break
        assert pick is not None
        vid, route, idx, mid = pick
        im = IM()
        cur = _cur(c, c.simulation.getTime(), [])
        guide.follow(vid)
        guide.step(cur, c, None, im)
        snap = guide.snapshot()
        assert snap["guided"] and snap["status"] == "following" and snap["events"]
        # a crash on its way is reported: the agent plans again and the car's route no longer goes there
        lanes = [f"{mid}_{i}" for i in range(c.edge.getLaneNumber(mid))]
        im.incidents.append(Inc(7, "rear-end", mid, [(ln, "") for ln in lanes],
                                police_record={"sent": c.simulation.getTime()}))
        rerouted = False
        for _ in range(4000):
            c.simulationStep()
            arrived = list(c.simulation.getArrivedIDList())
            guide.step(_cur(c, c.simulation.getTime(), arrived), c, None, im)
            snap = guide.snapshot()
            if snap["status"] == "following" and vid in c.vehicle.getIDList():
                rt = list(c.vehicle.getRoute(vid))
                assert r.drivable(rt[c.vehicle.getRouteIndex(vid):])
                if mid not in rt[c.vehicle.getRouteIndex(vid):]:
                    rerouted = True
            if snap["status"] != "following":
                break
        assert rerouted
        assert any(e["kind"] in ("plan", "reroute") and mid in e["avoided"] for e in snap["events"])
        assert snap["status"] == "arrived" and snap["real_s"] > 0
        assert snap["est_free_s"] is not None and snap["real_s"] >= 0.8 * snap["est_free_s"]
        rows = (tmp_path / "followed.csv").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 2 and vid in rows[1]
    finally:
        c.close()


@pytest.mark.slow
def test_the_way_left_is_the_way_sumo_drives_after_the_town_changes_it(tmp_path):
    """(10 Nov) An accident's detour (cosim/incidents._divert) sets the followed car's route in SUMO: the panel's way
    left and its length follow it (before: SUMO's route index read against the agent's old list, "0 m left")."""
    guide = rg.RouteGuide(log_dir=tmp_path)
    c = _sumo()
    try:
        for _ in range(240):
            c.simulationStep()
        r = guide.router
        pick = None
        for vid in c.vehicle.getIDList():
            route, idx = list(c.vehicle.getRoute(vid)), c.vehicle.getRouteIndex(vid)
            if len(route) - idx >= 6 and c.vehicle.getRoadID(vid) == route[idx]:
                mid = route[idx + 3]
                alt = r.best(route[idx], route[-1], [{"route": mid, "penalty": 500.0}])
                if alt and mid not in alt and alt != route[idx:]:
                    pick = (vid, alt)
                    break
        assert pick is not None
        vid, alt = pick
        guide.follow(vid)
        guide.step(_cur(c, c.simulation.getTime(), []), c, None, IM())
        c.vehicle.setRoute(vid, alt)                         # the town's detour
        for _ in range(4):
            c.simulationStep()
            guide.step(_cur(c, c.simulation.getTime(), list(c.simulation.getArrivedIDList())), c, None, IM())
        snap = guide.snapshot()
        sroute, i = list(c.vehicle.getRoute(vid)), c.vehicle.getRouteIndex(vid)
        assert snap["left"] == sroute[i:sroute.index(snap["dest"], i) + 1]
        assert snap["left_m"] == round(r.length_m(snap["left"]), 0) and snap["left_m"] > 0
    finally:
        c.close()


def test_a_bus_is_followed_but_not_routed(guide):
    class Conn:
        class vehicle:  # noqa: N801
            getRoute = staticmethod(lambda v: ("11", "3", "2"))
            getRouteIndex = staticmethod(lambda v: 0)
            getVehicleClass = staticmethod(lambda v: "bus")
            getRoadID = staticmethod(lambda v: "11")
            getIDList = staticmethod(lambda: ["pt_1"])
            getNextTLS = staticmethod(lambda v: [])

    guide.follow("pt_1")
    guide.step(Cur(5.0, {"pt_1": V(1.0, 2.0, 0.0, 3.0, "bus")}), Conn, None, None,
               special=lambda v: "a line bus keeps its line")
    snap = guide.snapshot()
    assert not snap["guided"] and snap["why_not"] == "a line bus keeps its line" and not guide.holds("pt_1")
    guide.stop()
    guide.step(Cur(5.5, {}), Conn, None, None)
    assert guide.snapshot() is None


def test_the_mini_map_draws_an_opaque_panel(guide):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    pygame.font.init()
    from cosim.overlay import KEY_COLOUR
    from cosim.route_minimap import MiniMap

    r = guide.router
    route = r.best("11", "-48")
    snap = {"vid": "live1", "status": "following", "guided": True, "why_not": "", "now": 100.0, "elapsed_s": 30.0,
            "start": "11", "dest": "-48", "own": route, "own_left": route, "route": route, "left": route, "idx": 0,
            "left_m": 300.0, "est_free_s": 19.0, "est_own_free_s": 19.0, "real_s": None,
            "options": [{"route": route, "cost": 19.0, "free_s": 19.0}],
            "alerts": [{"route": route[2], "type": "congestion", "fill": 0.8, "penalty": 2.6}],
            "events": [{"t": 70.0, "kind": "plan", "text": "Agent's route: x", "avoided": []}],
            "xy": r.roads["11"].shape[0], "angle": 90.0, "speed": 5.0, "trail": [], "signals": [("1427", 40.0, "r")],
            "explain": None}
    mm = MiniMap(r)
    x, y = r.roads["11"].shape[0]
    free = {"cam": (x - 50, y + 30, 120.0), "cam0": (x - 50, y + 30, 120.0), "cand": ("live1", x, y),
            "cars": [("live1", x, y, 3.0), ("live2", x + 20, y, 0.0)], "alerts": []}
    flying = mm.render(None, 0.0, free=free)
    first = mm.render(snap, 0.1, free=free, intro=0.05)          # following starts: text folded, map growing
    for i in range(30):
        s = mm.render(snap, 0.2 + 0.1 * i, free=free, intro=min(1.0, 0.1 * i))
    assert first.get_height() < s.get_height() and flying.get_height() < s.get_height()
    # (10 Nov) the car's own trip, estimated when it set off: its rows above the agent's
    snap["trip"] = {"t0": 40.0, "elapsed_s": 60.0, "est_s": 75.0, "free_s": 40.0, "table_share": 0.9,
                    "start": "Résidence Bellevue", "dest": "Marché Central", "dest_edge": "-48",
                    "changes": [(20.0, "new way")], "dest_changed": False, "ended": None, "real_s": None}
    with_trip = mm.render(snap, 3.5, free=free, intro=1.0)
    assert with_trip.get_height() >= s.get_height()
    if with_trip.get_size() == s.get_size():   # (the panel at its tallest: the map gives the rows room)
        assert (pygame.surfarray.array3d(with_trip) != pygame.surfarray.array3d(s)).any()
    w, h = s.get_size()
    assert w > 400 and h > 400
    for surf in (flying, first, s):
        px = pygame.surfarray.array3d(surf).reshape(-1, 3)
        assert not ((px[:, 0] == KEY_COLOUR[0]) & (px[:, 1] == KEY_COLOUR[1]) & (px[:, 2] == KEY_COLOUR[2])).any()


def test_the_camera_flies_in_smoothly_from_where_it_was():
    carla = pytest.importorskip("carla")
    import math

    from cosim.pose_proc import ChaseCam

    cam = ChaseCam()
    start = carla.Transform(carla.Location(-60.0, 40.0, 30.0), carla.Rotation(-30.0, 45.0, 0.0))
    cam.begin(start, ChaseCam.FLY_IN_S)
    car = carla.Transform(carla.Location(0.0, 0.0, 0.5), carla.Rotation(0.0, 0.0, 0.0))
    path = []
    for _ in range(int(2.5 * 60)):
        car = carla.Transform(carla.Location(car.location.x + 10 / 60, 0.0, 0.5), car.rotation)
        path.append(cam.update(car, 1 / 60, "chase", (2.3, 1.0, 0.75)))
    first = path[0].location
    assert math.dist((first.x, first.y, first.z), (-60.0, 40.0, 30.0)) < 0.1      # starts where the camera was
    steps = [math.dist((a.location.x, a.location.y, a.location.z), (b.location.x, b.location.y, b.location.z))
             for a, b in zip(path, path[1:])]
    assert max(steps) < 2.0 and steps[0] < 0.05                                    # no jump, eased from rest
    end = path[-1]
    assert end.location.x < car.location.x and abs(end.rotation.yaw) < 1.0        # behind the car, looking ahead


def test_chase_camera_sits_behind_the_car_and_eases_its_turn():
    carla = pytest.importorskip("carla")
    from cosim.pose_proc import ChaseCam

    cam = ChaseCam()
    car = carla.Transform(carla.Location(10.0, 20.0, 0.5), carla.Rotation(0.0, 0.0, 0.0))
    tf = cam.update(car, 1 / 60, "chase", (2.3, 1.0, 0.75))
    assert tf.location.x < 10.0 and abs(tf.location.y - 20.0) < 1e-6 and tf.location.z > 2.0
    car90 = carla.Transform(car.location, carla.Rotation(0.0, 90.0, 0.0))
    tf = cam.update(car90, 1 / 60, "chase", (2.3, 1.0, 0.75))
    assert 0.0 < tf.rotation.yaw < 10.0                    # eases towards the car's new heading
    for _ in range(120):
        tf = cam.update(car90, 1 / 60, "chase", (2.3, 1.0, 0.75))
    assert abs(tf.rotation.yaw - 90.0) < 0.5 and tf.location.y < 20.0
    roof = cam.update(car90, 1 / 60, "roof", (2.3, 1.0, 0.75))
    assert roof.location.z > 0.5 + 1.5 and abs(roof.location.x - 10.0) < 0.5


def test_the_pose_process_follows_by_sumo_id():
    pytest.importorskip("carla")
    from cosim.pose_proc import Driver

    d = Driver(0.0, 0.0)
    seg = {"start": 0.0, "dt": 0.5, "t": 0.5, "walk": [],
           "veh": [(41, "live7", None, None, (5.0, 5.0), None, 4.5, 0.0, 0.3, 0.3, 0.0, 0.0)]}
    d.receive(("seg", seg, True))
    d.receive(("follow", "live7", "roof"))
    d.picture(0.25, (0.0, 0.0), 0)
    assert d.follow_view == "roof" and d.follow_tf is not None
    d.receive(("follow", None))
    d.picture(0.3, (0.0, 0.0), 1)
    assert d.follow_tf is None

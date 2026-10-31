"""Accident pieces that need no SUMO run: severity, damage (cosim/incidents.py), the vehicle types (cosim/vmodels.py),
the lanes' directions (cosim/lanegeo.py), the bodies kept along their lane (cosim/interp.Trails), and the wreck,
part and crane poses (cosim/pose_proc.py). The whole accident, crash to clearance, is tools/check_incident_scene.py."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cosim import vmodels
from cosim.incidents import IncidentManager, severity_label
from cosim.interp import Trails

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ severity and damage
def test_severity_labels():
    assert severity_label(2.0) == "minor"
    assert severity_label(5.0) == "moderate"
    assert severity_label(9.5) == "severe"


def _damage(kind, role, sev, n=1):
    im = IncidentManager.__new__(IncidentManager)
    out = []
    for i in range(n):
        w = NS(vid=f"v{i}", role=role, model="vehicle.audi.tt")
        out.append(im._damage(sev, kind, w, NS(id=i)))
    return out


def test_damage_grows_with_the_crash():
    for kind, role in (("rear-end", "collider"), ("side", "victim")):
        soft = _damage(kind, role, 2.0, 40)
        hard = _damage(kind, role, 13.0, 40)
        mean = lambda ds, k: statistics.mean(abs(d[k]) for d in ds)  # noqa: E731
        assert mean(hard, "yaw") > 1.5 * mean(soft, "yaw"), kind
        assert mean(hard, "roll") > 1.5 * mean(soft, "roll"), kind
        assert all(h["dz"] < s["dz"] < 0 for h, s in zip(hard, soft))     # sits lower, more so when harder
        assert all(h["ramp_s"] > s["ramp_s"] for h, s in zip(hard, soft))  # a longer slide
        assert all(0.0 <= d["severity"] <= 1.0 for d in soft + hard)
    # a rear-ender's nose dips, the car it hit is pushed on
    assert _damage("rear-end", "collider", 8.0)[0]["pitch"] < 0
    assert _damage("rear-end", "victim", 8.0)[0]["dx"] > 0.25


def test_damage_is_the_same_every_run():
    # chosen from the incident and car, not from Python's salted hash(): a recording can be repeated
    a, b = _damage("side", "victim", 9.0, 5), _damage("side", "victim", 9.0, 5)
    assert a == b


def test_only_a_hard_side_crash_rolls_a_car_over():
    assert not any(d["rollover"] for d in _damage("side", "victim", 9.0, 60))
    assert not any(d["rollover"] for d in _damage("rear-end", "victim", 20.0, 60))
    over = sum(d["rollover"] for d in _damage("side", "victim", 16.0, 200))
    assert 30 < over < 110  # about one in three


# ------------------------------------------------------------------ the vehicle types
def test_type_ids_carry_their_carla_model():
    assert vmodels.model_of("pas_passenger@vehicle.audi.tt") == "vehicle.audi.tt"
    assert vmodels.model_of("pas_passenger@vehicle.audi.tt~2") == "vehicle.audi.tt"
    assert vmodels.model_of("pas_passenger") is None


def test_with_attrs_adds_and_replaces():
    vt = '<vType id="a" minGapLat="0.6" vClass="passenger"/>'
    out = vmodels.with_attrs(vt, vmodels.LANE_CHANGE_ATTRS)
    assert 'minGapLat="1.0"' in out and 'minGapLat="0.6"' not in out
    assert 'lcMaxSpeedLatStanding="0"' in out and out.count("<vType ") == 1


def test_sumo_sizes_follow_the_carla_model():
    models = vmodels.load()
    for m, info in models.items():
        length, width, height = vmodels.sumo_dims(info)
        assert length == pytest.approx(2 * info["extent"][0], abs=0.01)
        assert width <= 2.55 and width <= 2 * info["extent"][1] + 1e-6   # mirrors are not body
        assert height > 0.5, m


def test_typer_makes_one_sized_type_per_model():
    calls = []
    vt = NS(copy=lambda a, b: calls.append(("copy", a, b)), setLength=lambda t, v: calls.append(("L", t, v)),
            setWidth=lambda t, v: calls.append(("W", t, v)), setHeight=lambda t, v: calls.append(("H", t, v)))
    models = vmodels.load()
    t = vmodels.Typer(NS(vehicletype=vt), [], models, {"pol": "authority"})
    a = t.type_for("x1", "pol")
    b = t.type_for("x2", "pol")
    assert a == b == f"pol@{vmodels.FIXED['authority']}"
    assert [c[0] for c in calls] == ["copy", "L", "W", "H"]   # made once
    assert calls[1][2] == vmodels.sumo_dims(models[vmodels.FIXED["authority"]])[0]


# ------------------------------------------------------------------ lanes and bodies
def test_lane_direction_along_a_lane():
    pytest.importorskip("sumolib")
    from cosim.lanegeo import LaneGeometry

    net = ROOT / "scenarios" / "town05" / "town05.net.xml"
    if not net.exists():
        pytest.skip("no Town05 network")
    g = LaneGeometry(net)
    assert g.angle("no such lane", 3.0) is None
    lane, (cum, ang) = next((k, v) for k, v in g.lanes.items() if not k.startswith(":") and len(v[1]) == 1)
    assert g.angle(lane, 0.0) == g.angle(lane, cum[-1]) == ang[0]
    assert all(0.0 <= a < 360.0 for _, (_, angs) in list(g.lanes.items())[:200] for a in angs)


def _drive(tr, pts, lane_angle, speed, length=4.5):
    h = None
    for x, y in pts:
        h = tr.update("c", x, y, length, 0.0, lane_angle=lane_angle, speed=speed)
    return h


def test_a_body_slid_sideways_stays_along_its_lane():
    # SUMO moves a creeping car 3.5 m sideways over 4 m: the body may not stand across the lane
    tr = Trails()
    _drive(tr, [(0.0, y) for y in range(0, 30)], 0.0, 8.0)
    pts = [(3.5 * k / 40, 30 + 4.0 * k / 40) for k in range(1, 41)]
    for x, y in pts:
        h = tr.update("c", x, y, 4.5, 0.0, lane_angle=0.0, speed=1.0)
        off = (h + 180.0) % 360.0 - 180.0
        assert abs(off) <= Trails.MAX_SLANT_SLOW_DEG + Trails.MAX_STEP_DEG + 1e-6


def test_a_standing_body_never_turns():
    tr = Trails()
    _drive(tr, [(0.0, y) for y in range(0, 20)], 0.0, 8.0)
    h0 = tr.heading["c"]
    for k in range(30):  # nudged sideways in place, no forward roll
        h = tr.update("c", 0.1 * k, 19.0, 4.5, 0.0, lane_angle=0.0, speed=0.0)
    assert h == pytest.approx(h0)


def test_a_real_turn_is_followed():
    import math

    tr = Trails()
    _drive(tr, [(0.0, y * 0.5) for y in range(0, 40)], 0.0, 8.0)
    r = 12.0
    hs = [tr.update("c", r - r * math.cos(a), 20.0 + r * math.sin(a), 4.5, 0.0, lane_angle=math.degrees(a), speed=6.0)
          for a in [k * math.pi / 2 / 80 for k in range(1, 81)]]
    assert 70.0 < hs[-1] < 95.0   # a right turn, about 90 deg, with the lane's direction in the band


# ------------------------------------------------------------------ wreck and part poses (CARLA's maths)

def _carla():
    return pytest.importorskip("carla")


def test_a_wreck_comes_to_rest_over_its_ramp():
    c = _carla()
    from cosim.pose_proc import lay_wreck

    tf = c.Transform(c.Location(10.0, 5.0, 0.3), c.Rotation(0.0, 90.0, 0.0))
    dmg = {"yaw": 20.0, "roll": 3.0, "pitch": -2.0, "dx": 1.0, "dz": -0.05, "rollover": False, "ramp_s": 0.5, "t0": 100.0}
    start = lay_wreck(tf, dmg, 100.0)
    assert start.rotation.yaw == pytest.approx(90.0) and start.location.y == pytest.approx(5.0)
    end = lay_wreck(tf, dmg, 101.0)
    assert end.rotation.yaw == pytest.approx(110.0)
    assert end.location.y == pytest.approx(6.0, abs=1e-4)      # pushed 1 m along its heading (+y)
    assert end.location.z == pytest.approx(0.25)
    over = lay_wreck(tf, dict(dmg, rollover=True), 101.0)
    assert abs(over.rotation.roll) == pytest.approx(88.0) and over.location.z > 1.0


def test_a_part_moves_and_turns_with_its_body():
    c = _carla()
    from cosim.pose_proc import compose

    body = c.Transform(c.Location(1.0, 2.0, 0.0), c.Rotation(0.0, 90.0, 0.0))
    rear = compose(body, (-2.0, 0.0, 0.5, 0.0, 0.0))
    assert (rear.location.x, rear.location.y, rear.location.z) == pytest.approx((1.0, 0.0, 0.5), abs=1e-5)
    assert rear.rotation.yaw == pytest.approx(90.0)
    front = compose(body, (2.0, 0.0, 0.5, 5.0, 180.0))
    assert front.rotation.yaw % 360 == pytest.approx(270.0)
    # on a body pitched up, a part's pitch adds up and a part at the front rises
    up = c.Transform(c.Location(0, 0, 0), c.Rotation(10.0, 0.0, 0.0))
    p = compose(up, (2.0, 0.0, 0.0, 5.0, 0.0))
    assert p.rotation.pitch == pytest.approx(15.0, abs=1e-4) and p.location.z > 0.3


def test_the_deck_rests_on_the_chassis_and_tilts_to_the_road():
    from cosim import towing

    level = towing.deck_level()
    assert level[3] == 0.0 and level[2] - towing.DECK_T == pytest.approx(towing.CHASSIS_Z)
    assert level[0] + towing.DECK_L == pytest.approx(towing.DECK_FRONT_X)       # its front just behind the cab
    tilted = towing.deck_pose(1.0, 1.0)
    assert tilted[3] == pytest.approx(towing.TILT_DEG)
    # its rear end's underside on the road, its top edge a few centimetres up
    ex, ez = towing.deck_end(tilted)
    bottom = ez - 2 * towing.DECK_T * math.cos(math.radians(towing.TILT_DEG))
    assert bottom == pytest.approx(0.0, abs=0.02) and 0.05 < ez < 0.2
    assert ex < -towing.TRUCK_HALF - 3.0                                         # well behind the truck
    # it turns about the chassis' end: that point of its underside stays put while it tilts
    for k in (0.0, 0.3, 0.7, 1.0):
        d = towing.deck_pose(1.0, k)
        r = math.radians(d[3])
        a = towing._pivot_local()
        px = d[0] + a * math.cos(r) + towing.DECK_T * math.sin(r)
        pz = d[2] + a * math.sin(r) - towing.DECK_T * math.cos(r)
        assert (px, pz) == pytest.approx((towing.PIVOT_X, towing.CHASSIS_Z), abs=1e-6)


def test_the_winch_brings_the_car_up_onto_the_deck():
    from cosim import towing

    p = towing.Path()
    L = 4.6
    x0, z0, pitch0 = p.car(p.s_start(L), L)
    assert z0 == 0.0 and pitch0 == 0.0                                            # on the road, level
    front0 = x0 + L / 2
    assert front0 == pytest.approx(towing.deck_end(towing.deck_pose(1, 1))[0] - towing.GAP_M, abs=1e-6)
    xm, zm, pm = p.car((p.s_start(L) + p.s_end(L)) / 2, L)
    assert 0.0 < pm <= towing.TILT_DEG and zm > 0.0                               # half way: climbing
    x1, z1, p1 = p.car(p.s_end(L), L)
    d = towing.deck_pose(1, 1)
    want = towing._on_deck(d, towing.CAR_X, towing.DECK_T)
    assert (x1, z1, p1) == pytest.approx((want[0], want[1], towing.TILT_DEG), abs=1e-6)
    # each step of the winch moves it a little: no jump
    ss = [p.s_start(L) + (p.s_end(L) - p.s_start(L)) * k / 200 for k in range(201)]
    pts = [p.car(s_, L) for s_ in ss]
    assert max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) < 0.06


def test_the_loading_timeline_is_in_order_and_takes_under_a_minute():
    from cosim import towing

    tl = towing.timeline(4.6)
    names = ["reverse", "slide", "tilt", "align", "winch", "tilt_back", "slide_back", "strap"]
    assert [tl[n] for n in names] == sorted(tl[n] for n in names)
    assert 35.0 < tl["total"] < 60.0
    assert towing.phase_at(tl, 0.1)[0] == "reverse" and towing.phase_at(tl, tl["total"] + 1)[0] == "done"
    assert towing.deck_at(tl, tl["winch"] - 0.1) == towing.deck_pose(1, 1)
    assert towing.deck_at(tl, tl["total"]) == towing.deck_pose(0, 0)


def test_the_truck_backs_in_smoothly_from_where_it_stopped():
    from cosim import towing

    p0, p1 = (100.0, 50.0, 0.0, 0.0), (85.0, 53.5, 0.0, 0.0)  # stopped ahead in the next lane, backs into its place
    pts = [towing.manoeuvre(p0, p1, k / 100) for k in range(101)]
    assert pts[0][:2] == pytest.approx(p0[:2]) and pts[-1][:2] == pytest.approx(p1[:2])
    assert pts[0][3] == pytest.approx(0.0) and pts[-1][3] == pytest.approx(0.0, abs=1e-6)
    assert all(b[0] <= a[0] + 1e-9 for a, b in zip(pts, pts[1:]))              # always backwards
    assert max(abs(p[3]) for p in pts) < 35.0                                    # a plausible S, nose out
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    assert max(steps) < 0.4


def test_a_tow_plan_hands_the_truck_back_where_sumo_has_it():
    c = _carla()
    from cosim import towing
    from cosim.pose_proc import TowPlan, compose

    tl = towing.timeline(4.6)
    truck0 = c.Transform(c.Location(100.0, 50.0, 0.0), c.Rotation(0.0, 0.0, 0.0))
    car0 = c.Transform(c.Location(80.0, 53.6, 0.0), c.Rotation(0.0, 12.0, 1.5))
    plan = TowPlan(7, 8, {"t0": 0.0, "load": (91.0, 53.5, 0.0, 0.0), "car_len": 4.6, "tl": tl}, truck0, car0)
    assert plan.truck(0.0, None).location.x == pytest.approx(100.0)
    load = plan.truck(tl["reverse"] + 0.1, None)
    assert (load.location.x, load.location.y) == pytest.approx((91.0, 53.5))
    assert plan.truck(tl["total"] + 0.1, None) is None                           # SUMO's again
    assert plan.car(1.0, load, plan.deck(1.0)).location.x == pytest.approx(80.0)  # the wreck waits where it lies
    end = plan.car(tl["total"], load, plan.deck(tl["total"]))
    want = compose(compose(load, towing.deck_level()), (towing.CAR_X, 0.0, towing.DECK_T, 0.0, 0.0))
    assert end.location.distance(want.location) < 1e-6 and end.rotation.pitch == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ the recovery truck while you watch (24 Oct)
class _Lane:
    def __init__(self, lid, length, idx=0, edge=None):
        self.lid, self.length, self.idx, self.edge = lid, length, idx, edge

    def getID(self):
        return self.lid

    def getLength(self):
        return self.length

    def getIndex(self):
        return self.idx

    def getEdge(self):
        return self.edge

    def allows(self, vc):
        return True


def _im_with_lanes(lanes, via=None):
    im = IncidentManager.__new__(IncidentManager)
    im.net = NS(getLane=lambda lid: lanes[lid])
    im._via = via or {}
    return im


def test_where_the_place_is_along_the_trucks_lane():
    """The truck backs from where it stands to the place: on its road, in the junction past its lane's end, or in
    the junction before it (negative): the stretch it sweeps, where nobody may stand (IncidentManager._back_in)."""
    lanes = {"6_3": _Lane("6_3", 54.0), ":1126_4_0": _Lane(":1126_4_0", 20.0), ":979_12_0": _Lane(":979_12_0", 25.0)}
    im = _im_with_lanes(lanes, {":1126_4_0": ("6_4", "-16_4"), ":979_12_0": ("-5_3", "6_3")})
    assert im._along("6_3", "6_4", 30.0) == 30.0
    assert im._along("6_3", ":1126_4_0", 2.0) == 56.0
    assert im._along("6_3", ":979_12_0", 8.0) == -17.0
    assert im._along("6_3", "9_1", 8.0) is None


def test_cars_behind_the_truck_are_waved_round_it():
    """A car standing where the truck would back is waved onto the next lane (not the closed one) after MAKE_WAY_S;
    before 24 Oct the truck waited for good while someone watched."""
    calls = []
    edge = NS()
    lanes = [_Lane("e_0", 50, 0), _Lane("e_1", 50, 1), _Lane("e_2", 50, 2)]
    for ln in lanes:
        ln.edge = edge
    edge.getLanes = lambda: lanes
    im = _im_with_lanes({ln.getID(): ln for ln in lanes})
    im.squeezed, im.stats, im.now = {}, {}, 0.0
    veh = NS(getLaneID=lambda o: "e_1", getVehicleClass=lambda o: "passenger",
             setParameter=lambda *a: calls.append(("param",) + a), setLaneChangeMode=lambda *a: None,
             changeLane=lambda *a: calls.append(("change",) + a), getMinGap=lambda o: 2.5, getMinGapLat=lambda o: 0.6,
             getParameter=lambda o, k: "", setMinGap=lambda *a: None, setMinGapLat=lambda *a: None,
             getRoadID=lambda o: "e", getLanePosition=lambda o: 20.0)
    im.conn = NS(vehicle=veh)
    inc = NS(closed=[("e_2", None)], log=lambda t, m: calls.append(("log", m)))
    t = NS(vid="tow")
    im._out_of_the_way(inc, t, ["car"], 0.0)
    assert not [c for c in calls if c[0] == "change"]  # not yet: it may go by itself
    im._out_of_the_way(inc, t, ["car"], IncidentManager.MAKE_WAY_S + 0.5)
    assert ("change", "car", 0, 30.0) in calls  # lane 2 is closed: onto lane 0
    assert im.stats["waved_round"] == 1 and "car" in im.squeezed


def test_the_stand_ins_want_no_other_lane():
    """The cone line's stand-in and a truck's want no other lane (IncidentManager._inert): cooperative drivers on the
    open lane stopped behind it to let it in, for good (seed 7, watched)."""
    params = {}
    veh = NS(getLateralLanePosition=lambda v: 0.0, setParameter=lambda v, k, x: params.__setitem__((v, k), x))
    im = IncidentManager.__new__(IncidentManager)
    im.conn = NS(vehicle=veh)
    inc = NS(blocker="incblock1", tows=[NS(ghost="incblock1_tow0"), NS()])
    im._inert(inc)
    for vid in ("incblock1", "incblock1_tow0"):
        assert params[(vid, "laneChangeModel.lcStrategic")] == "-1"
        assert params[(vid, "laneChangeModel.lcCooperative")] == "0"
    assert inc.inert == {"incblock1", "incblock1_tow0"}

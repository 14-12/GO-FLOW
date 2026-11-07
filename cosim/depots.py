"""A hospital, a police station and a recovery yard on Town05 (25 Oct): where the ambulances, the police cars and the
recovery trucks start from and go back to.

Town05 has no such buildings, so three stretches of the avenue that runs north-south through the town (x 136-153 in
SUMO), the only roads with a parking lane 3 m wide, are made into them:
* the parked vehicles stand in that lane, each in its own slot (a SUMO stop that lasts until it is sent out), far
  enough apart to pull out (`SLOT_GAP_M`);
* a sign stands on the pavement beside them (a board made from the recovery truck's deck mesh, SM_TnDeck, standing,
  with a texture drawn here: `sign_texture`), put up by the number plates' process (cosim/plate_proc.py).

What uses them (cosim/freefly.py wires it): the emergency calls (cosim/extras.py: an ambulance or a police car leaves
on a run, attends a call at a parking spot elsewhere in town, comes back without its beacons and parks), and the
accidents (cosim/incidents.py: the police car and the recovery trucks come from here, and go back after the job).
When a depot has nobody left, the old way is used (a vehicle starts out of sight on the roads leading there).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger("depots")

SLOT_GAP_M = 8.0          # free road between two parked vehicles: room to pull out
PARK_LANE_SPEED = 8.33    # the parking lane at a depot: 30 km/h (netconvert gives parking lanes 5 km/h)
SIGN_SCALE = 0.55         # the deck mesh (6.4 x 2.36 m) at this scale: a 3.5 x 1.3 m board
SIGN_UP_M = 0.25          # the board's lower edge above the pavement
SIGN_ROLL = 90.0          # the deck mesh stood up, its face to the road, the text upright (variant A of
SIGN_YAW = 180.0          # tools/probe_depots.py): yaw = the way it faces + 180, roll +90. In the viewer (25 Oct),
                          # yaw + 0 showed the road the board's plain back (roll -90 or +90), and yaw + 180 with
                          # roll -90 its face the wrong way round


@dataclass
class DepotPlan:
    name: str                 # hospital | police | tow
    title: str                # the sign's big text
    subtitle: str
    edge: str                 # SUMO edge with the parking lane
    lane: int                 # the parking lane's index
    vehicle_len: float        # the longest vehicle kept there
    fleet: int                # how many it keeps
    sign_at: float = 0.5      # where the sign stands along the stretch (share of the edge's length)
    colours: tuple = ((255, 255, 255), (20, 24, 32), (200, 30, 40))  # board, text, accent
    keep_clear: tuple = ()    # stretches of the parking lane nobody parks on ((from, to) m): a bus stop
    gap: float = SLOT_GAP_M   # room between two parked vehicles
    also: tuple = ()          # classes allowed on the parking lane besides the depots' (the buses of that stop)


TOWN05 = (
    DepotPlan("hospital", "HÔPITAL", "URGENCES 24h/24", "49", 2, 6.9, 4, 0.5,
              ((246, 248, 250), (24, 70, 160), (210, 30, 40))),
    # -52_2 is also bus line L2's first stop (town05.pt.add.xml, L2_S1: 20.5-40.5 m, the shelter): the cars park
    # either side of it, 5 m apart, and the buses may use the lane (25 Oct: two cars stood at the stop, and the
    # lane was closed to the buses)
    DepotPlan("police", "POLICE", "COMMISSARIAT", "-52", 2, 4.9, 4, 0.15,
              ((22, 54, 120), (255, 255, 255), (230, 230, 230)), keep_clear=((18.0, 43.0),), gap=5.0,
              also=("bus",)),
    DepotPlan("tow", "DÉPANNAGE", "REMORQUAGE 24h/24", "-50", 2, 7.94, 3, 0.12,  # (before the trucks, not behind)
              ((250, 200, 20), (20, 20, 20), (20, 20, 20))),
)


def slots(edge_len: float, vehicle_len: float, n: int, margin: float = 3.0, gap: float = SLOT_GAP_M,
          keep_clear=()) -> list[float]:
    """Stop positions (a vehicle's front, m from the edge's start) for `n` vehicles, the first the furthest along
    (it drives straight out), each `gap` behind the one ahead, none over a `keep_clear` stretch; as many as fit."""
    out, pos = [], edge_len - margin
    while len(out) < n and pos - vehicle_len >= margin:
        over = next(((a, b) for a, b in keep_clear if pos > a and pos - vehicle_len < b), None)
        if over is not None:
            pos = over[0]  # on before the stretch
            continue
        out.append(round(pos, 1))
        pos -= vehicle_len + gap
    return out


def lane_point(shape, length: float, pos: float, lateral: float = 0.0):
    """(x, y, heading deg (SUMO's x/y frame, counter-clockwise from +x)) at `pos` along a lane `shape` of `length`,
    `lateral` m to its left."""
    from sumolib.geomhelper import positionAtShapeOffset

    geo = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(shape, shape[1:], strict=False))
    p = min(max(pos, 0.0), length) * geo / max(length, 0.01)
    x, y = positionAtShapeOffset(shape, p)
    x2, y2 = positionAtShapeOffset(shape, min(p + 0.5, geo))
    if math.hypot(x2 - x, y2 - y) < 1e-3:
        x2, y2 = x, y
        x, y = positionAtShapeOffset(shape, max(p - 0.5, 0.0))
    dx, dy = x2 - x, y2 - y
    n = math.hypot(dx, dy) or 1.0
    return x - dy / n * lateral, y + dx / n * lateral, math.degrees(math.atan2(dy, dx))


PARK_STRAIGHT_M = 0.15    # off its lane's middle by more than this when it stops: set straight in the middle


def park_straight(conn, vid: str) -> float | None:
    """A vehicle that just stopped in a parking lane (a depot's slot, a call's place), set straight in the lane's middle.
    It turns in while it slows down, and it may not move sideways once standing (cosim/vmodels.py,
    lcMaxSpeedLatStanding=0): a police car back from a call stood 1 m off the middle, 0.4 m into the next lane, and the
    traffic there waited behind it, up to 22 minutes (7 Nov, SUMO with the depots and the calls). -> how far it was
    off (m), or None when it was not moved."""
    try:
        ln = conn.vehicle.getLaneID(vid)
        lat = conn.vehicle.getLateralLanePosition(vid)
        if not ln or ln.startswith(":") or abs(lat) <= PARK_STRAIGHT_M:
            return None
        pos = conn.vehicle.getLanePosition(vid)
        x, y, hdg = lane_point(conn.lane.getShape(ln), conn.lane.getLength(ln), pos)
        edge, idx = ln.rsplit("_", 1)
        conn.vehicle.moveToXY(vid, edge, int(idx), x, y, (90.0 - hdg) % 360.0, keepRoute=1)
        return lat
    except Exception as e:  # noqa: BLE001
        log.debug("park straight %s: %s", vid, e)
        return None


TURN_IN_LAT_MS = 0.5      # sideways m/s a vehicle may still move while standing, on the last road to its parking spot
TURN_IN_STUCK_S = 15.0    # standing that long beside its spot, not in its lane: it is put in


def turn_in(conn, vid: str, edge: str, lane_index: int, pos: float, set_lat: set) -> str | None:
    """A vehicle on the road of its stop (a depot's slot, a call's place, a bus stop), not stopped there yet: it may finish
    moving over into the parking lane while it slows down (SUMO's sublane model, lcMaxSpeedLatStanding, which is 0 for
    every type: cosim/vmodels.py), and one standing beside its spot for TURN_IN_STUCK_S is put in. 7 Nov (SUMO with the
    depots): a police car on the police station's 61 m road reached its slot still in the traffic lane, stopped, could
    not move sideways, and stood there 0.9 m across, the traffic behind it waiting up to 34 minutes; bus line L2 did the
    same at its first stop, on that road (up to 7 minutes).
    set_lat: the vehicles already allowed (once each). -> "put in" when moved there, else None."""
    try:
        if conn.vehicle.getRoadID(vid) != edge or conn.vehicle.isStopped(vid):
            return None
        if vid not in set_lat:
            set_lat.add(vid)
            try:
                conn.vehicle.setParameter(vid, "laneChangeModel.lcMaxSpeedLatStanding", str(TURN_IN_LAT_MS))
            except Exception as e:  # noqa: BLE001 - an older SUMO: the fallback below still does it
                log.debug("turn in %s: %s", vid, e)
        if conn.vehicle.getLaneIndex(vid) == lane_index or conn.vehicle.getWaitingTime(vid) < TURN_IN_STUCK_S:
            return None
        ln = f"{edge}_{lane_index}"
        x, y, hdg = lane_point(conn.lane.getShape(ln), conn.lane.getLength(ln), pos)
        conn.vehicle.moveToXY(vid, edge, lane_index, x, y, (90.0 - hdg) % 360.0, keepRoute=1)
        return "put in"
    except Exception as e:  # noqa: BLE001
        log.debug("turn in %s: %s", vid, e)
        return None


def turned_in(conn, vid: str, set_lat: set) -> None:
    """Stopped at its spot: back to the town's rule (no sideways move while standing, cosim/vmodels.py)."""
    if vid in set_lat:
        set_lat.discard(vid)
        try:
            conn.vehicle.setParameter(vid, "laneChangeModel.lcMaxSpeedLatStanding", "0")
        except Exception as e:  # noqa: BLE001
            log.debug("turned in %s: %s", vid, e)


def sign_spot(net, plan: DepotPlan):
    """Where the sign stands (SUMO x, y) and the direction it faces (towards the road, SUMO heading in degrees): on
    the pavement right of the parking lane, past its shoulder."""
    e = net.getEdge(plan.edge)
    lanes = e.getLanes()
    park = lanes[plan.lane]
    off = park.getWidth() / 2.0
    for ln in lanes[:plan.lane][::-1]:  # the shoulder, then the pavement (lower indices: further right)
        if ln.allows("pedestrian") and not ln.allows("passenger"):
            off += ln.getWidth() * 0.55
            break
        off += ln.getWidth()
    x, y, hdg = lane_point(park.getShape(), park.getLength(), park.getLength() * plan.sign_at, -off)
    return x, y, (hdg + 90.0) % 360.0  # facing left of the lane's direction: the road


def sign_texture(plan: DepotPlan, w: int = 768, h: int = 288):
    """The sign's picture (Pillow RGB): u along the board (the deck mesh's length), v across it; the top 8 % of the
    rows is the deck's side band (cosim/towing.deck_texture), not on the face."""
    from PIL import Image, ImageDraw, ImageFont

    from cosim.hud import FONTS

    board, text, accent = plan.colours
    im = Image.new("RGB", (w, h), board)
    d = ImageDraw.Draw(im)
    v0 = int(round(0.08 * h))
    d.rectangle([0, 0, w - 1, v0 - 1], fill=board)
    face_h = h - v0
    pad = int(face_h * 0.08)
    d.rectangle([pad // 2, v0 + pad // 2, w - 1 - pad // 2, h - 1 - pad // 2], outline=text, width=max(3, pad // 4))
    x = pad * 2
    if plan.name == "hospital":  # a blue square with a white H
        s = face_h - 3 * pad
        d.rectangle([x, v0 + int(1.5 * pad), x + s, v0 + int(1.5 * pad) + s], fill=text)
        fh = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), int(s * 0.95))
        d.text((x + s / 2, v0 + 1.5 * pad + s / 2), "H", font=fh, fill=board, anchor="mm")
        x += s + pad
    elif plan.name == "tow":  # black and yellow chevrons
        cw = int(face_h * 0.18)
        for i in range(3):
            x0 = x + i * cw
            d.polygon([(x0, v0 + 1.5 * pad), (x0 + cw * 0.6, v0 + face_h / 2), (x0, h - 1.5 * pad),
                       (x0 + cw * 0.35, h - 1.5 * pad), (x0 + cw * 0.95, v0 + face_h / 2),
                       (x0 + cw * 0.35, v0 + 1.5 * pad)], fill=text)
        x += 3 * cw + pad
    elif plan.name == "police":  # a white star badge
        r = (face_h - 3 * pad) / 2
        cx, cy = x + r, v0 + face_h / 2
        pts = [(cx + (r if k % 2 == 0 else r * 0.45) * math.cos(math.radians(-90 + 36 * k)),
                cy + (r if k % 2 == 0 else r * 0.45) * math.sin(math.radians(-90 + 36 * k))) for k in range(10)]
        d.polygon(pts, fill=accent)
        x += int(2 * r) + pad
    cx = (x + w - 2 * pad) / 2
    big = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), int(face_h * 0.52))
    small = ImageFont.truetype(str(FONTS / "BarlowCondensed-Medium.ttf"), int(face_h * 0.2))
    for f in (big,):  # shrink the title until it fits
        size = f.size
        while d.textlength(plan.title, font=f) > (w - 2 * pad - x) and size > 20:
            size -= 4
            f = ImageFont.truetype(str(FONTS / "BarlowCondensed-SemiBold.ttf"), size)
        big = f
    d.text((cx, v0 + face_h * 0.42), plan.title, font=big, fill=text, anchor="mm")
    d.text((cx, v0 + face_h * 0.80), plan.subtitle, font=small, fill=text if plan.name != "hospital" else accent,
           anchor="mm")
    return im


@dataclass
class Depot:
    plan: DepotPlan
    edge_len: float
    slots: list[float]
    parked: dict = field(default_factory=dict)    # slot -> vid standing there
    coming: dict = field(default_factory=dict)    # vid -> slot it was sent back to
    members: set = field(default_factory=set)     # every vehicle of this depot
    out: dict = field(default_factory=dict)       # vid -> the slot it was taken from (until it has left)


BIG_S = 1e7  # a parked vehicle's stop: until it is sent out
DEPOT_CLASSES = ("emergency", "authority")  # the ambulances and recovery trucks, the police cars


def pull_out(conn, vid: str) -> None:
    """A vehicle in a parking lane (a depot's slot, a call's place) about to leave: it moves over into the lane
    beside, at a crawl if it must (every type has no sideways move standing, cosim/vmodels.LANE_CHANGE_ATTRS). Left
    to SUMO, a truck drove along the parking lane to its end, the queue beside it stood, and it stood there for good
    (seed 11)."""
    try:
        lane = conn.vehicle.getLaneID(vid)
        if not lane or lane.startswith(":"):
            return
        idx = int(lane.rsplit("_", 1)[1])
        if conn.lane.getMaxSpeed(lane) > PARK_LANE_SPEED + 0.1:
            return  # not a parking lane
        conn.vehicle.setParameter(vid, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
        conn.vehicle.changeLane(vid, idx + 1, 20.0)
    except Exception as e:  # noqa: BLE001
        log.debug("pull out %s: %s", vid, e)


class Depots:
    """The depots at run time. conn: TraCI; net: sumolib net; plans: DepotPlan list. Vehicles are put in with
    add_parked (by whoever owns their type: cosim/incidents.py for police cars and recovery trucks, EmergencyCalls
    for ambulances), taken out with take + leave, and sent back with send_home; step() notices who got back."""

    def __init__(self, conn, net, plans=TOWN05):
        self.conn = conn
        self.net = net
        self.turn_lat: set = set()     # vehicles allowed to move sideways while standing, on their last road (turn_in)
        self.depots: dict[str, Depot] = {}
        for p in plans:
            if not net.hasEdge(p.edge):
                log.info("depot %s: no edge %s", p.name, p.edge)
                continue
            ln = net.getEdge(p.edge).getLanes()[p.lane]
            self.depots[p.name] = Depot(p, ln.getLength(), slots(ln.getLength(), p.vehicle_len, p.fleet, gap=p.gap,
                                                                 keep_clear=p.keep_clear))
        self.home: dict[str, str] = {}   # vid -> depot name
        self.stats = {"sent": 0, "back": 0, "added": 0, "no_slot": 0}

    def setup(self) -> None:
        """The depots' parking lanes: for their vehicles only (the town's cars drove into them and stood behind the
        parked ones for good, unable to move over standing), at PARK_LANE_SPEED."""
        for d in self.depots.values():
            ln = f"{d.plan.edge}_{d.plan.lane}"
            try:
                self.conn.lane.setAllowed(ln, list(DEPOT_CLASSES) + list(d.plan.also))
                self.conn.lane.setMaxSpeed(ln, PARK_LANE_SPEED)
            except Exception as e:  # noqa: BLE001
                log.info("depot %s parking lane: %s", d.plan.name, e)
        log.info("depots: %s", {n: (d.plan.edge, d.slots) for n, d in self.depots.items()})

    # ---------------------------------------------------------------- who is where
    def owns(self, vid: str) -> bool:
        return vid in self.home

    def parked(self, name: str) -> list[str]:
        d = self.depots.get(name)
        return [] if d is None else [v for _, v in sorted(d.parked.items(), reverse=True)]

    def _free_slot(self, d: Depot) -> float | None:
        taken = set(d.parked) | set(d.coming.values()) | set(d.out.values())
        return next((s for s in d.slots if s not in taken), None)

    # ---------------------------------------------------------------- in and out
    def add_parked(self, name: str, vid: str, vtype: str) -> bool:
        """A new vehicle parked at once in the depot's next free slot (the viewer's start)."""
        d = self.depots.get(name)
        if d is None:
            return False
        slot = self._free_slot(d)
        if slot is None:
            return False
        c, p = self.conn, d.plan
        rid = f"depot_{name}_r"
        try:
            if rid not in c.route.getIDList():
                c.route.add(rid, [p.edge])
            vlen = c.vehicletype.getLength(vtype)
            c.vehicle.add(vid, rid, typeID=vtype, depart="now", departLane=str(p.lane),
                          departPos=f"{max(slot - 0.1, vlen + 0.1):.1f}", departSpeed="0")
            c.vehicle.setStop(vid, p.edge, pos=slot, laneIndex=p.lane, duration=BIG_S)
        except Exception as e:  # noqa: BLE001
            log.info("depot %s: %s not parked: %s", name, vid, e)
            return False
        d.parked[slot] = vid
        d.members.add(vid)
        self.home[vid] = name
        self.stats["added"] += 1
        return True

    def take(self, name: str, ok=None, keep: int = 0) -> str | None:
        """A parked vehicle that can go (the front-most first: it drives straight out), or None. ok(vid): it may go
        (a recovery truck with a wreck still on its deck may not); keep: leave this many behind."""
        d = self.depots.get(name)
        if d is None or len(d.parked) <= keep:
            return None
        for slot in sorted(d.parked, reverse=True):
            vid = d.parked[slot]
            if ok is not None and not ok(vid):
                continue
            del d.parked[slot]
            d.out[vid] = slot
            return vid
        return None

    def put_back(self, vid: str) -> None:
        """Taken but not sent after all: it stays in its slot."""
        for d in self.depots.values():
            slot = d.out.pop(vid, None)
            if slot is not None:
                d.parked[slot] = vid

    def leave(self, vid: str, edges: list) -> bool:
        """Out of its slot along `edges` (from the depot's edge), straight over into the next lane (pull_out)."""
        c = self.conn
        try:
            c.vehicle.setRoute(vid, edges)
            if c.vehicle.isStopped(vid):
                c.vehicle.resume(vid)
            c.vehicle.setSpeed(vid, -1)
            pull_out(c, vid)
        except Exception as e:  # noqa: BLE001
            log.info("%s does not leave its depot: %s", vid, e)
            return False
        for d in self.depots.values():
            d.out.pop(vid, None)
        self.stats["sent"] += 1
        return True

    def route_from(self, name: str, to_edge: str, vtype: str):
        """SUMO's route from the depot `name` to `to_edge` (findRoute: its edges, length), or None."""
        d = self.depots.get(name)
        if d is None:
            return None
        try:
            r = self.conn.simulation.findRoute(d.plan.edge, to_edge, vType=vtype.split("@")[0])
        except Exception:  # noqa: BLE001
            return None
        return r if r.edges else None

    def send_home(self, vid: str) -> bool:
        """Back to its depot, into a free slot (where it stays, stopped). False when there is no way or no room."""
        name = self.home.get(vid)
        d = self.depots.get(name) if name else None
        if d is None:
            return False
        slot = self._free_slot(d)
        if slot is None:
            self.stats["no_slot"] += 1
            return False
        c, p = self.conn, d.plan
        try:
            pull_out(c, vid)
            c.vehicle.changeTarget(vid, p.edge)
            c.vehicle.setStop(vid, p.edge, pos=slot, laneIndex=p.lane, duration=BIG_S)
        except Exception as e:  # noqa: BLE001
            log.info("%s not sent home: %s", vid, e)
            return False
        d.coming[vid] = slot
        return True

    def step(self, vehicles: dict) -> None:
        """Every couple of seconds: who got back (stopped in its slot) is parked; who is gone is forgotten."""
        c = self.conn
        for d in self.depots.values():
            for vid, slot in list(d.coming.items()):
                if vid not in vehicles:
                    try:
                        alive = vid in c.vehicle.getIDList()
                    except Exception:  # noqa: BLE001
                        alive = False
                    if not alive:
                        d.coming.pop(vid, None)
                        self._forget(vid)
                    continue
                if turn_in(c, vid, d.plan.edge, d.plan.lane, slot, self.turn_lat) is not None:
                    self.stats["put_in"] = self.stats.get("put_in", 0) + 1
                    log.info("depot %s: %s put into its slot (it stood beside it in the traffic lane)", d.plan.name, vid)
                try:
                    there = c.vehicle.isStopped(vid) and c.vehicle.getRoadID(vid) == d.plan.edge
                except Exception:  # noqa: BLE001
                    there = False
                if there:
                    d.coming.pop(vid, None)
                    d.parked[slot] = vid
                    self.stats["back"] += 1
                    turned_in(c, vid, self.turn_lat)
                    if park_straight(c, vid) is not None:
                        self.stats["straightened"] = self.stats.get("straightened", 0) + 1
            for slot, vid in list(d.parked.items()):
                if vid not in vehicles:
                    try:
                        alive = vid in c.vehicle.getIDList()
                    except Exception:  # noqa: BLE001
                        alive = False
                    if not alive:
                        d.parked.pop(slot, None)
                        self._forget(vid)

    def adopt(self, vid: str, name: str) -> None:
        """A vehicle that did not start here becomes this depot's (if sent home and there is room, it stays)."""
        if name in self.depots:
            self.home[vid] = name
            self.depots[name].members.add(vid)

    def disown(self, vid: str) -> None:
        """Not this depot's after all (no room or no way back): it drives on as any vehicle."""
        for d in self.depots.values():
            d.coming.pop(vid, None)
            d.out.pop(vid, None)
            for slot in [k for k, v in d.parked.items() if v == vid]:
                d.parked.pop(slot)
        self._forget(vid)

    def _forget(self, vid: str) -> None:
        name = self.home.pop(vid, None)
        if name and name in self.depots:
            self.depots[name].members.discard(vid)


CALL_S = (20.0, 45.0)     # an emergency call: this long at the place it was called to
CALL_EVERY_S = 75.0       # an emergency call every this many seconds at least (one every 45 s left the depots empty)
CALL_MIN_M = 250.0        # a call this far from the depot at least (on the road)
POLICE_KEEP = 1           # police cars the calls leave at the station (for the accidents)


class EmergencyCalls:
    """The extras' emergency vehicles (cosim/extras.py) from the depots: when one is due, an ambulance (or a police
    car) leaves its depot on a run (beacons, through red: the "_run" type, the extras' record on a run, so the
    junction agents give it green), stops at the place it was called to (a parking spot elsewhere in town) for
    CALL_S, then drives back to its depot without the beacons and parks. extras.spawn_hook = calls.spawn."""

    TYPES = {"ambulance": ("twin_ambulance_run", "twin_ambulance"), "police": ("twin_police_run", "twin_police")}
    DEPOT = {"ambulance": "hospital", "police": "police"}

    def __init__(self, conn, net, depots: Depots, extras, rng, type_for=None):
        self.conn, self.net, self.depots, self.extras, self.rng = conn, net, depots, extras, rng
        self.type_for = type_for or (lambda vid, t: t)
        self.calls: dict[str, dict] = {}   # vid -> {"kind", "edge", "lane", "pos", "until", "t_out"}
        self.avoid = lambda: set()          # roads not to call to or through (an accident's: IncidentManager.blocked_roads)
        self.spots = self._spots()
        self.stats = {"calls": 0, "attended": 0, "back": 0, "no_vehicle": 0}

    def _spots(self) -> list[tuple[str, int, float]]:
        """Where calls are attended: the middle of every parking lane but the depots' own."""
        own = {d.plan.edge for d in self.depots.depots.values()}
        out = []
        for e in self.net.getEdges():
            if e.getFunction() or e.getID() in own:
                continue
            for ln in e.getLanes():
                if ln.allows("passenger") and ln.getSpeed() < 2.0 and ln.getLength() > 18.0:  # a parking lane
                    out.append((e.getID(), ln.getIndex(), round(ln.getLength() * 0.6, 1)))
        return sorted(out)

    def stock(self, n: dict | None = None) -> None:
        """The ambulances parked at the hospital (the police cars are the accidents' to put: cosim/incidents.py)."""
        d = self.depots.depots.get("hospital")
        for k in range(len(d.slots) if d else 0):
            vid = f"ambulance_h{k + 1}"
            self.depots.add_parked("hospital", vid, self.type_for(vid, self.TYPES["ambulance"][1]))

    def spawn(self, tc, now: float, kind: str, on_run: bool) -> bool:
        """extras.spawn_hook: True when a depot vehicle took the call (else the extras' own way: none, here)."""
        depot = self.DEPOT[kind]
        vid = self.depots.take(depot, keep=POLICE_KEEP if kind == "police" else 0)
        if vid is None:
            self.stats["no_vehicle"] += 1
            return True  # nobody free: the call waits for the next (no vehicle out of nowhere)
        run_t = self.type_for(vid, self.TYPES[kind][0])
        spots = list(self.spots)
        self.rng.shuffle(spots)
        avoid = self.avoid()
        for edge, lane, pos in spots[:12]:
            r = self.depots.route_from(depot, edge, run_t)
            if r is not None and r.length >= CALL_MIN_M and not avoid & set(r.edges):
                break
        else:
            self.depots.put_back(vid)
            return True
        c = self.conn
        try:
            c.vehicle.setType(vid, run_t)
            if not self.depots.leave(vid, list(r.edges)):
                raise RuntimeError("did not leave")
            c.vehicle.setStop(vid, edge, pos=pos, laneIndex=lane, duration=BIG_S)
        except Exception as e:  # noqa: BLE001
            log.info("call for %s: %s", vid, e)
            self.depots.put_back(vid)
            return True
        from cosim.extras import EmergencyRecord

        self.extras.emergency[vid] = EmergencyRecord(vid, kind, True, now, r.length)
        self.calls[vid] = {"kind": kind, "edge": edge, "until": None, "t_out": now, "lane": lane, "pos": pos}
        self.stats["calls"] += 1
        log.info("%s leaves the %s on a run to %s (%.0f m)", vid, depot, edge, r.length)
        return True

    def step(self, now: float) -> None:
        """Every couple of seconds: at the call's place -> wait there; done -> back home, beacons off."""
        c = self.conn
        for vid, call in list(self.calls.items()):
            try:
                stopped = c.vehicle.isStopped(vid)
                road = c.vehicle.getRoadID(vid)
            except Exception:  # noqa: BLE001 - gone
                self.calls.pop(vid, None)
                continue
            if call["until"] is None and call["edge"] in self.avoid():
                # an accident on the road it was called to (it would stand behind the cone line): the call is
                # taken by someone else, it goes back
                call["until"] = now
                self.stats["called_off"] = self.stats.get("called_off", 0) + 1
            if call["until"] is None:
                if not stopped and "lane" in call and turn_in(c, vid, call["edge"], call["lane"], call["pos"],
                                                              self.depots.turn_lat) is not None:
                    self.stats["put_in"] = self.stats.get("put_in", 0) + 1
                if stopped and road == call["edge"]:
                    call["until"] = now + self.rng.uniform(*CALL_S)
                    turned_in(c, vid, self.depots.turn_lat)
                    if park_straight(c, vid) is not None:
                        self.stats["straightened"] = self.stats.get("straightened", 0) + 1
                    self.stats["attended"] += 1
                    rec = self.extras.emergency.get(vid)
                    if rec is not None:
                        rec.on_run = False  # beacons off while it attends
                continue
            if now < call["until"]:
                continue
            kind = call["kind"]
            try:
                c.vehicle.setType(vid, self.type_for(vid, self.TYPES[kind][1]))
                c.vehicle.resume(vid)
            except Exception:  # noqa: BLE001
                pass
            rec = self.extras.emergency.get(vid)
            if rec is not None:
                rec.on_run = False
            if self.depots.send_home(vid):
                self.stats["back"] += 1
            self.calls.pop(vid, None)

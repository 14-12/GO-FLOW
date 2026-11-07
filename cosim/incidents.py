"""Accidents in the free-fly viewer: they happen on their own, the police come, the wrecks are taken away (23 Oct).

SUMO's drivers never crash: Krauss keeps a safe gap, the junction model a safe slot (measured: 0 collisions in
10 minutes of Town05's evening rush even with 5 % of drivers given a short reaction time, a high sigma and
SUMO's "ignore foe" junction parameters, tools/check_accidents.py --careless-only). People crash because they
look away. So a few drivers (CARELESS_SHARE of the cars) are careless: now and then, at speed, one of them looks
away for 1.5-3 s (`Distraction`):
* "phone": it keeps its speed whatever happens ahead (SUMO's speed checks off). If the car ahead brakes or
  queues, it runs into it: a rear-end crash, at the speed difference it had.
* "red light": near a signal that is red for it (not during the pedestrians' walk), it drives on through the red,
  still keeping its distance to the car ahead but ignoring who has the right of way: a side impact in the
  junction if crossing traffic is there, nothing if not. Once in the junction it goes on at its speed and brakes for
  nobody until it is out (3 Nov, RED_THROUGH_MAX_S).
Whether anything happens depends only on the traffic around: nothing is placed or timed. SUMO detects the
collision (--collision.action warn --collision.check-junctions); then:

1. crash: both cars stop where they are, for good (speed and lane-change control taken over), hazard lights on;
   severity from the impact speed (rear-end: the speed difference; side: both speeds combined). The CARLA side
   shows the damage (cosim/incident_scene.py). Later cars that hit the wrecks join the same incident.
2. after a dispatch delay (3 Nov: minutes, drawn per incident, `police_delay_s` / `police_spread`; before, a fixed
   6 s): a police car is sent from 250-700 m away on a run (beacons, through red lights, signals turned for it),
   to stop 8 m behind the wrecks in their lane. Each incident records when the police were due, sent and on scene,
   and how they got there (`Incident.police_record`).
3. police there: cones and a warning sign go out (CARLA side), the officer guides the traffic past on the open lane.
   The wrecks' lane is closed from the crash on: SUMO lets only emergency and authority vehicles use it where that
   strands nobody, nobody may change into it near the scene, and drivers still on it behind the wrecks are told to
   merge well before them; the open lanes past the scene are slowed to 30 km/h. The wrecks stand straight in the
   middle of their lane (as SUMO has them, so no passing car touches one).
4. the drivers get out on the side of the pavement and wait there; when that means crossing an open lane, the
   traffic on it is held for them (it stops before them, as drivers would).
5. +10 s: a recovery truck (cosim/towing.py: a rollback, the front-most wreck first) passes the scene on the open
   lane, stops ahead of the wreck and backs into the closed lane in front of it; there it tilts its deck to the road,
   winches the wreck up, levels the deck and drives off (CARLA side; SUMO holds it for that time, then takes the
   wreck off the road). The next wreck's truck comes when the zone ahead of it is free.
6. all wrecks gone: the lane reopens, the officer picks up the cones and drives off.

At most MAX_ACTIVE incidents at a time and one new every COOLDOWN_S at most: while that is reached nobody looks
away (and a collision then is a near miss: the car that hit brakes and the two part). An accident is never removed for being far away:
its cars are always drawn (FreeFly), and its vehicles are never taken off the road or given new trips.

More kinds (31 Oct), so that an incident is not always two cars that met:
* breakdown: now and then a car (not one in a junction, on a road of MIN_CRASH_ROAD_M and more) puts its hazard
  lights on and rolls to a stop in its lane (BREAKDOWN_DECEL); the same response follows (police, the lane
  closed, a recovery truck), with no damage and no debris. Incident.kind "breakdown", severity 0.
* speeding drivers: SPEEDER_SHARE of the new cars drive 25-50 % over the limit (SUMO's speed factor) and are
  SPEEDER_CARELESS times as likely to be careless: their crashes are harder. Wreck / snapshot field "speeding".
* pile-up: after a crash on the ring road (or in a junction it leads into), each of the next few cars behind it in its lane may react late
  (PILEUP_LATE_P, PILEUP_LATE_S: it keeps its speed that long, then brakes as hard as it can); whoever cannot stop
  in time runs into the wrecks and joins the incident (up to PILEUP_MAX_WRECKS cars). From three cars on, its kind
  is "pile-up". Whether anyone hits is again SUMO's traffic alone.
  Pile-ups start from any crash there, and from a car that brakes hard all of a sudden (RING_BRAKE_MEAN_S: something
  on the road, a near miss, a driver who changed their mind): the cars behind it may react late in the same way.
* look-alikes, not incidents (`Lookalike`): a delivery van stopping in its lane with its hazards on for 1-3 minutes
  ("double-parked"), a line bus standing at its stop well past its dwell ("bus-long-stop"). Nobody responds; they are
  logged (lookalike_snapshot) so that an incident detector can be shown what it must not report.
"""

from __future__ import annotations

import logging
import math
import random
import zlib
from dataclasses import dataclass, field

log = logging.getLogger("incidents")

CARELESS_SUFFIX = "~careless"
TOW_SUFFIX = "~tow"
CARELESS_SHARE = 0.10       # of new cars
DISTRACT_PER_S = 1 / 8.0    # a careless driver looks away about every 8 s of driving at speed
# 3 Nov: a red-light runner braked inside the junction (for who crossed it, its mode keeping a safe gap; or back to
# SUMO's careful driver when its lapse ran out there), at 4-4.5 m/s2 for 2-3.5 s, and was then hit nearly standing: on
# camera a car braking to a stop, then a soft crash (tools/measure_impact.py: 4 of 5 junction crashes). Now, once in the
# junction, it goes on at the speed it entered with, and its lapse lasts until it is out, at most this much longer: a
# crash there happens at the speeds the two had.
RED_THROUGH_MAX_S = 4.0
# 6 Nov, option A of the GO-FLOW crash delivery (docs/WORKLOG.md): while cameras record, a car heading into a place
# one of them sees clearly (`focus`, cosim/clip_recorder.Focus) is made careless with chance `focus_p` (once per car;
# the town-wide share stays CARELESS_SHARE), and a careless driver there looks away FOCUS_BOOST times as often. Whether
# a crash follows, and how, is still the traffic's. Each incident says whether the focus had a hand in it
# (crash_source "camera focus"), so a crash rate in front of the cameras is never read as a real one.
# Where it would hit: the car ahead, if there is one within FOCUS_LEAD_M, else where its middle will be in
# FOCUS_AHEAD_S at its speed; only a car whose hit would be in the clear view counts. A car made careless this way
# looks away only there (elsewhere it drives as the others): a first version left it careless for its whole trip,
# and 2 crashes of 3 it caused were out of view (tools/check_accidents.py --focus-cams, 6 Nov).
FOCUS_AHEAD_S = (1.5, 3.0)
FOCUS_LEAD_M = 40.0
FOCUS_BOOST = 6.0
FOCUS_MIN_MS = 3.0
FOCUS_NEAR_M = 80.0          # cars further than this from every camera are not looked at
LAPSE_AHEAD = 10.0          # ... ten times as often when the car ahead is slower and close, or the light is red
MAX_ACTIVE = 3                # (3 Nov: was 2) with the police minutes away an incident stays open 7-10 min, and 2 at a
                              # time held the crashes back: careless share 0.5, ring road and buses, 3 seeds x 60 min
                              # (tools/check_accidents.py): 13 crashes with the police at 6 s, 8 at 4 min with 2 open,
                              # 14 with 3 (and 14 with 4), cars standing over 2 min 35 / 52 / 57 / 50
COOLDOWN_S = 60.0           # simulated seconds between two new incidents at least
POLICE_DELAY_S = 6.0          # (before 3 Nov: every police car was sent this soon; `police_delay_s=6, police_spread=0`)
# 3 Nov: the police are sent minutes after the crash, not seconds. With 6 s they were on scene 15-45 s after it, and the
# GO-FLOW team's accident detector (v2) learnt "stopped police lights = crash": slow on crashes, false alarms on any
# stopped police car. The dispatch delay is drawn per incident from a log-normal: median POLICE_MEDIAN_S, spread
# POLICE_SPREAD (sigma of its log), kept within POLICE_BOUNDS x the median. The drive there comes on top: the car
# starts out of sight close by (_send_police), so it is usually on scene within a minute of being sent.
POLICE_MEDIAN_S = 240.0
POLICE_SPREAD = 0.5           # 5-95 %: about 105-545 s
POLICE_BOUNDS = (0.25, 3.75)  # x the median: 60-900 s by default
STEP_S = 0.5                  # SUMO's step in the viewer (the per-step rates here assume it too)
HAZARDS_AFTER_S = 2.0         # a wreck's hazard lights go on this long after the hit (its driver switches them on)
TOW_DELAY_S = 10.0
LEAVE_AFTER_LOAD_S = 1.5     # the wreck strapped down -> the truck drives off
PASS_KMH = 30.0              # the open lanes beside an accident
MERGE_BEFORE_M = 150.0       # drivers on the closed lane this far behind the wrecks are told to merge
HOLD_BEFORE_M = 12.0         # traffic held for people crossing stops from this far before them
HOLD_CLEAR_M = 9.0           # ... or, if it can stop gently, no closer than this (where they cross)
CLEAR_S = 12.0               # last wreck gone -> lane reopened (the officer has walked back to his car)
UNJAM_S = 150.0              # a car standing this long out of sight is taken off the road (IncidentManager.unjam)
BLOCK_BEHIND_M = 36.0        # the closed lane is shut this far behind the wrecks (where the cone taper begins)
BOX_STUCK_S = 40.0           # standing this long inside a junction: it edges through (IncidentManager.unjam)
NUDGE_S = 8.0
GREEN_STUCK_S = 15.0        # first at a green light, standing this long: it stops giving way (IncidentManager.unjam)
LEAVE_NUDGE_S = 4.0         # a loaded truck leaving from a junction, standing this long: it edges out
SEVERITY = ((3.5, "minor"), (8.0, "moderate"), (1e9, "severe"))
CLOSED_TO = ["passenger", "private", "taxi", "bus", "coach", "delivery", "truck", "trailer", "motorcycle", "moped",
             "evehicle", "army", "vip", "hov", "bicycle"]
POLICE_TYPE = "twin_police_run"
POLICE_PARKED = "twin_police"  # its type at the police station and on its way back (no beacons: cosim/depots.py)
BLOCKER_PREFIX = "incblock"   # a closed lane's end SUMO knows of (Incident._block): never drawn
TRUCK_BASE = "tru_truck"
DEPOT_NAMES = {"police": "police station", "tow": "recovery yard", "hospital": "hospital"}
ADJ_WAIT_S = 40.0            # a truck backing across open lanes waits this long at most for who is where it sweeps
ON_FOOT_AFTER_S = 30.0       # (3 Nov) a police car standing this long in a queue within ON_FOOT_NEAR_M of the scene, on a
ON_FOOT_NEAR_M = 60.0        # road it may not stop on: the officer walks from there (IncidentManager._advance)
EN_ROUTE_MAX_S = 240.0       # a police car (or, a minute more, a truck) this long on its way: another comes
DEPOT_REACH_M = 500.0        # an accident further than this by road gets a vehicle put on the road near it instead
JAMMED_SHARE = 0.6           # a road with most lanes standing this full counts as jammed (no depot vehicle sent through)
MIN_CRASH_ROAD_M = 20.0      # no accident on a road shorter than this (a near miss instead): road 13 into the ring
                             # junction 2014 is 14 m, its wrecks stood in the junction, the ring stood behind them and
                             # neither the police nor a truck got there (seeds 53 and 61, watched, with and without
                             # the depots)
NEAR_OTHER_M = 60.0          # nor this close to one under way (the drivers there go slowly, looking): three crashes
                             # in junction 1722 at once had trucks and officers among each other's wrecks (seed 41)
KEEP_LANE_S = 40.0           # a car standing this long on a lane that does not lead on: a road its lane leads to
UNBLOCK_S = 20.0             # a truck waiting this long for its way back to clear: who waits for its lane goes on
DEPOT_STUCK_S = 60.0         # a depot's vehicle standing this long on its way, out of sight: it goes back, one put on
                             # the road comes instead
NEVER_CARELESS = ("emergency", "authority", "bus", "coach")
# 31 Oct: more kinds of incident (module doc)
BREAKDOWN_MEAN_S = 2400.0    # one car breaks down about this often (simulated s) while a new incident may start
BREAKDOWN_DECEL = 2.5        # it rolls to a stop at this (m/s2)
BREAKDOWN_RING_P = 0.3       # on the ring road this often (its long roads would otherwise take nearly all of them)
SPEEDER_SHARE = 0.08         # of new cars: drive SPEEDER_FACTOR times the limit
SPEEDER_FACTOR = (1.25, 1.5)
SPEEDER_CARELESS = 3.0       # a speeder is this many times as likely to be careless
RING_CARELESS = 3.0          # the ring road's own cars (cosim/highway.py) are this many times as often careless
RING_BRAKE_MEAN_S = 400.0    # a car on the ring road brakes hard for no reason the others see, about this often
RING_BRAKE_TO_MS = 0.0       # ... down to this, over RING_BRAKE_S (about 9 m/s2: an emergency stop)
RING_BRAKE_S = 2.4
RING_BRAKE_GAP_M = 15.0      # ... only with a car close behind it (tailgating), that then reacts late
RING_BRAKE_LATE_S = (1.2, 2.2)
PILEUP_MAX_WRECKS = 6        # an incident on the ring road takes up to this many cars (elsewhere 3)
PILEUP_LATE_P = (0.7, 0.6, 0.45, 0.3)  # the 1st, 2nd ... car behind a crash on the ring: chance it reacts late
PILEUP_LATE_S = (0.8, 1.8)   # ... this late (it keeps its speed, then brakes)
PILEUP_REACH_M = 90.0        # only cars this close behind, and faster than PILEUP_MIN_MS
PILEUP_MIN_MS = 8.0
LOOKALIKE_MAX = 2            # at a time
DOUBLE_PARK_MEAN_S = 240.0   # a delivery van stops in its lane about this often (simulated s)
DOUBLE_PARK_S = (60.0, 180.0)
BUS_LONG_P = 0.03            # of a line bus's stops: it stands there BUS_LONG_S instead of its dwell
BUS_LONG_S = (70.0, 150.0)
TOW_GIVE_UP_S = 240.0        # a recovery truck waiting this long ahead of its wreck goes back (_tow_gives_up)
LC_DEFAULT = 1621            # SUMO's default lane change mode


def severity_label(v: float) -> str:
    return next(name for limit, name in SEVERITY if v < limit)


def police_delay(rng: random.Random, median_s: float, spread: float) -> float:
    """Seconds from a crash to the police car being sent (module doc): log-normal around median_s, within
    POLICE_BOUNDS x median_s. spread 0: always median_s."""
    if spread <= 0.0 or median_s <= 0.0:
        return max(median_s, 0.0)
    lo, hi = POLICE_BOUNDS
    return min(max(median_s * math.exp(rng.gauss(0.0, spread)), lo * median_s), hi * median_s)


def police_delay_args(spec: str | None) -> dict:
    """--police-delay "MEDIAN[:SPREAD]" (cosim/freefly.py, tools/check_accidents.py) -> IncidentManager keywords."""
    if spec is None or str(spec).strip() == "":
        return {}
    parts = str(spec).split(":")
    out = {"police_delay_s": float(parts[0])}
    if len(parts) > 1 and parts[1] != "":
        out["police_spread"] = float(parts[1])
    if out["police_delay_s"] < 0 or out.get("police_spread", 0.0) < 0:
        raise ValueError(f"--police-delay {spec}: negative")
    return out


def body_box(x: float, y: float, angle: float, length: float, width: float) -> tuple:
    """A vehicle's body (centre x, y, heading unit vector, half length, half width) from SUMO's front position and
    angle (degrees, 0 north, clockwise)."""
    hx, hy = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    return (x - hx * length / 2.0, y - hy * length / 2.0, hx, hy, length / 2.0, width / 2.0)


def boxes_touch(a: tuple, b: tuple) -> bool:
    """Two body_box()es overlap (separating axis test)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    for ux, uy in ((a[2], a[3]), (-a[3], a[2]), (b[2], b[3]), (-b[3], b[2])):
        ra = a[4] * abs(a[2] * ux + a[3] * uy) + a[5] * abs(-a[3] * ux + a[2] * uy)
        rb = b[4] * abs(b[2] * ux + b[3] * uy) + b[5] * abs(-b[3] * ux + b[2] * uy)
        if abs(dx * ux + dy * uy) > ra + rb:
            return False
    return True


def contact_fraction(a0: tuple, a1: tuple, b0: tuple, b1: tuple, size_a: tuple, size_b: tuple, n: int = 50) -> float:
    """When in a step two vehicles first touched: a0, a1 (b0, b1) their (x, y, angle) at the step's start and end
    (SUMO front position), size_* (length, width). -> fraction of the step, 0..1 (1: they never overlap on the way, as
    far as these boxes go; 0: already at its start).

    SUMO sees a collision at the end of the step it happened in, the cars already inside each other by up to a step's
    travel (3 Nov, tools/measure_impact.py: up to 5.5 m). The cars are put back to where they touched (_to_contact),
    and the viewer draws them stopping there, at that moment (cosim/pose_proc.Poser.vehicles)."""
    def at(p, q, f):
        d = (q[2] - p[2] + 180.0) % 360.0 - 180.0
        return p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, p[2] + d * f

    for i in range(n + 1):
        f = i / n
        if boxes_touch(body_box(*at(a0, a1, f), *size_a), body_box(*at(b0, b1, f), *size_b)):
            return f
    return 1.0


@dataclass
class Wreck:
    vid: str
    model: str | None
    role: str                       # collider | victim
    lane: str
    pos: float
    x: float
    y: float
    angle: float
    length: float
    speed: float                    # at impact
    damage: dict = field(default_factory=dict)  # yaw, roll, pitch, dz, dx (deg, m), rollover
    removed: bool = False           # off the road (loaded onto a recovery truck)
    tow: str | None = None
    exit: dict | None = None        # its driver getting out: when, to where on the pavement (SUMO x, y), crossing?
    speeding: bool = False          # a speeder (31 Oct)
    t_hit: float | None = None      # (3 Nov) SUMO time of the hit, within the step SUMO saw it (contact_fraction)
    hit_pose: tuple | None = None   # (3 Nov) SUMO (x, y, angle) where it touched, before it is centred in its lane


MIN_ENTRY_M = 10.0  # room left at the start of a short closed lane: who still comes onto it moves over there


def block_at(rear: float) -> float:
    """Where on the closed lane its stand-in stands (its front; m from the lane's start), the wrecks' rear at `rear`:
    where the cone taper begins, BLOCK_BEHIND_M behind them. On a short road that would be at its very start, and a
    car already in the junction before it, bound for the closed lane, stood there for good (seed 7, watched): it is
    then MIN_ENTRY_M on, or just behind the wrecks, so such a car gets onto the road and moves over (the cones begin
    there too, cosim/incident_scene.py)."""
    pos = rear - BLOCK_BEHIND_M
    if pos >= MIN_ENTRY_M:
        return pos
    return max(1.5, min(rear - 1.5, MIN_ENTRY_M))


@dataclass
class Tow:
    vid: str
    wreck: str
    state: str = "en route"         # en route | there | backing | loading | loaded | leaving | gone
    t_there: float | None = None
    t_load: float | None = None     # when it started backing in: cosim/towing.timeline's time 0
    place: tuple | None = None
    stuck_s: float = 0.0
    load: tuple | None = None       # where it loads (SUMO front x, y, angle, length): in the closed lane ahead
    tl: dict | None = None          # cosim/towing.timeline for its wreck
    rejoin: bool = False            # loads where SUMO does not hold it (a wreck in a junction): drives back after
    moved: bool = False             # SUMO moved it into the closed lane


@dataclass
class Incident:
    id: int
    t0: float
    kind: str                       # rear-end | side | junction | frontal
    severity: float                 # m/s
    label: str
    edge: str
    lane: str
    x: float                        # SUMO, where it happened (between the cars)
    y: float
    wrecks: list = field(default_factory=list)
    phase: str = "crash"            # crash | police en route | secured | recovery | clearing | closed
    police: str | None = None
    police_t: float | None = None
    police_stuck_s: float = 0.0
    police_due: float | None = None  # (3 Nov) when the police are (next) to be sent
    police_record: dict = field(default_factory=dict)  # due, sent, on_scene (SUMO s), how, cars sent
    t_contact: float | None = None   # (3 Nov) the first hit, within the step SUMO saw it (t0 is that step)
    crash_source: str = "natural"    # (6 Nov) "camera focus": a driver made careless or distracted by option A
    tows: list = field(default_factory=list)
    closed: list = field(default_factory=list)  # (lane, disallowed before)
    open_lane: str | None = None    # the lane traffic passes on (the officer points to it)
    clear_t: float | None = None
    events: list = field(default_factory=list)
    holds: list = field(default_factory=list)   # traffic held for people crossing: (t0, t1, lanes, before pos)
    held: set = field(default_factory=set)
    merged: set = field(default_factory=set)
    restore: list = field(default_factory=list)  # (kind, lane, value): lane settings to put back at reopening

    def log(self, t: float, text: str) -> None:
        self.events.append((round(t - self.t0, 1), text))
        log.info("incident %d at %.0f s (+%.0f s): %s", self.id, self.t0, t - self.t0, text)


@dataclass
class Lookalike:
    """Something that looks like an incident to a camera and is not one (module doc): logged, nobody responds."""
    id: int
    kind: str                       # double-parked | bus-long-stop
    vid: str
    lane: str
    t_told: float
    seconds: float                  # how long it was told to stand
    t0: float | None = None         # standing from
    t1: float | None = None         # moving again
    x: float = 0.0                  # SUMO, where it stands
    y: float = 0.0
    stop: str | None = None         # the bus stop


class Distraction:
    """A careless driver looking away (see the module doc)."""

    def __init__(self, vid: str, kind: str, until: float, speed: float):
        self.vid, self.kind, self.until, self.speed = vid, kind, until, speed
        self.focus = False  # (6 Nov) looked away as it came into a camera's clear view (option A)
        self.blind = False  # (red light) in the junction, going on without braking for who crosses (3 Nov)


class IncidentManager:
    """Runs in the SUMO thread, after each step (it owns no CARLA state). `snapshot()` is what the CARLA side
    draws."""

    def __init__(self, conn, net_file, typer=None, seed: int = 11, walk=None, share: float = CARELESS_SHARE,
                 max_active: int = MAX_ACTIVE, cooldown_s: float = COOLDOWN_S, visible=None, sinks=None,
                 breakdown_mean_s: float = BREAKDOWN_MEAN_S, speeder_share: float = SPEEDER_SHARE,
                 pileups: bool = True, lookalikes: bool = True, ring_brake_mean_s: float = RING_BRAKE_MEAN_S,
                 police_delay_s: float = POLICE_MEDIAN_S, police_spread: float = POLICE_SPREAD, focus_p: float = 0.0):
        import sumolib

        self.conn = conn
        self.net = sumolib.net.readNet(str(net_file), withInternal=True)
        self.typer = typer
        self.rng = random.Random(seed)
        self.walk = walk              # sumo_side.WalkSkip: never run a red during the pedestrians' walk
        self.share = share
        self.max_active = max_active
        self.cooldown_s = cooldown_s
        self.visible = visible        # (SUMO x, y) -> bool: in sight of the camera (police and trucks appear elsewhere)
        self.sinks = list(sinks or [])
        self.incidents: list[Incident] = []
        self.done: list[Incident] = []
        self.n = 0
        self.careless: dict[str, str] = {}  # vid -> its normal type
        self.distracted: dict[str, Distraction] = {}
        self.last_t = -1e9
        self.role: dict[str, tuple[int, str]] = {}  # vid -> (incident id, wreck | police | tow)
        self.stats = {"distractions": 0, "collisions": 0, "incidents": 0, "near_misses": 0, "person_hits": 0,
                      "joined": 0, "careless_made": 0}
        self._via = {}                # internal (via) lane -> (from lane, to lane)
        for e in self.net.getEdges():
            for c in e.getOutgoing().values() if isinstance(e.getOutgoing(), dict) else []:
                for conn in c:
                    via = conn.getViaLaneID()
                    if via:
                        self._via[via] = (conn.getFromLane().getID(), conn.getToLane().getID())
        self.now = 0.0
        self._pairs: set = set()
        self.inserted_back: dict = {}  # vid -> (metres back, lane) where a police car or a recovery truck started
        self.depots = None             # cosim/depots.Depots: police cars and recovery trucks come from there (stock_depots)
        self.tow_free = None           # vid -> True: that recovery truck has nothing on its deck (the viewer's scene)
        self.depot_dispatch = True     # False: the accidents' police cars and trucks are put on the road near them (they
                                       # still go to the depots after the job)
        self.squeezed: dict = {}  # vid -> (road, position, when, its own min gaps) while it pulls out
        self.standing: dict[str, float] = {}
        self.halted: dict[str, float] = {}
        self.jammed: dict[str, float] = {}
        # 31 Oct: breakdowns, speeding drivers, pile-ups on the ring road, look-alikes (module doc)
        self.breakdown_mean_s = breakdown_mean_s
        self.speeder_share = speeder_share
        self.pileups = pileups
        self.ring_brake_mean_s = ring_brake_mean_s
        self.lookalikes_on = lookalikes
        # 3 Nov: the police dispatch delay (module doc), from its own generator (the town's draws stay as they were)
        self.police_delay_s = police_delay_s
        self.police_spread = police_spread
        self.police_rng = random.Random(seed * 7919 + 17)
        # 6 Nov, option A: the cameras' clear views (set by the viewer while recording), from their own generator
        self.focus = None             # callable (SUMO x, y, angle) -> bool (cosim/clip_recorder.Focus)
        self.focus_p = focus_p
        self.focus_rng = random.Random(seed * 104729 + 3)
        self.focus_seen: set = set()  # cars already decided
        self.focus_made: set = set()  # cars made careless by it
        self._approaching: set = set()
        self._before: dict = {}   # the step before's vehicles (where the cars that crash came from: _to_contact)
        self._last: dict = {}
        self._moved: dict = {}    # vid -> SUMO (x, y, angle) once put back to where it touched
        self.speeders: dict[str, float] = {}       # vid -> its speed factor
        self._speed_pending: dict[str, tuple] = {}  # vid -> (factor, made at): set once it is on the road
        self.breaking: dict[str, dict] = {}        # vid -> {t, v0}: rolling to a stop (a breakdown)
        self.lookalikes: list[Lookalike] = []
        self.lookalikes_done: list[Lookalike] = []
        self.n_look = 0
        self._bus_at: dict[str, str] = {}          # bus -> the stop it was last seen standing at
        self.late_for: dict[str, int | None] = {}  # vid -> the incident it reacts late to (_late_chain; None: a car
                                                   # braking hard ahead of it)
        self._late_t: dict[str, float] = {}
        self._hard: dict[str, float] = {}           # vid -> until when it brakes past its normal deceleration
        self.near_why: dict[str, int] = {}         # near misses by reason
        try:
            from cosim.highway import find_loops

            from cosim.highway import PREFIX

            self.ring = {e for lp in find_loops(self.net) for e in lp}
            self._ring_prefix = PREFIX
        except Exception:  # noqa: BLE001
            self.ring, self._ring_prefix = set(), None
        for k in ("breakdowns", "speeders_made", "pileup_late", "pileups", "double_parked", "bus_long_stops",
                  "sudden_brakes"):
            self.stats[k] = 0

    # ------------------------------------------------------------------ who may be careless
    def allowed(self) -> bool:
        active = [i for i in self.incidents if i.phase != "closed"]
        return len(active) < self.max_active and self.now - self.last_t >= self.cooldown_s

    def type_for(self, vid: str, base: str) -> str:
        """The type a new vehicle is inserted with (Population.type_for): its model's (cosim/vmodels.Typer), careless
        for a share of the cars while new incidents are allowed."""
        careful = self.typer.type_for(vid, base) if self.typer is not None else base
        vclass = self.typer.base.get(base, "passenger") if self.typer is not None else "passenger"
        share = self.share
        if vclass == "passenger" and self.speeder_share > 0 and self.rng.random() < self.speeder_share:
            f = round(self.rng.uniform(*SPEEDER_FACTOR), 2)
            self.speeders[vid] = f
            self._speed_pending[vid] = (f, self.now)
            self.stats["speeders_made"] += 1
            share = min(share * SPEEDER_CARELESS, 0.6)
        if self.pileups and self._ring_prefix and vid.startswith(self._ring_prefix):
            share = min(share * RING_CARELESS, 0.6)
        if vclass == "passenger" and self.allowed() and self.rng.random() < share:
            self.careless[vid] = careful
            self.stats["careless_made"] += 1
        return careful

    # ------------------------------------------------------------------ one step
    def step(self, now: float, vehicles: dict, tls: dict | None = None) -> None:
        """After a SUMO step. vehicles: id -> sumo_side.Vehicle (all of them); tls: signal states."""
        self.now = now
        self._before, self._last = self._last, vehicles
        c = self.conn
        for vid in [v for v in self.careless if v not in vehicles]:
            self.careless.pop(vid, None)
            self.distracted.pop(vid, None)
        self._approaching = self._focus_step(vehicles) if self.focus is not None else set()
        for vid in [v for v in self.late_for if v not in vehicles or now - self._late_t.get(v, now) > 10.0
                    or v not in self.distracted and v not in self.halted and vehicles[v].speed < 0.1]:
            self.late_for.pop(vid, None)  # (gone, stopped in time, or long past)
            self._late_t.pop(vid, None)
        for vid, t in list(self.halted.items()):
            if now >= t:
                self.halted.pop(vid, None)
                if vid not in self.role:
                    try:
                        c.vehicle.setSpeed(vid, -1)
                    except Exception:  # noqa: BLE001
                        pass
        self._speeding(now, vehicles)
        self._distractions(now, vehicles, tls or {})
        self._breakdowns(now, vehicles)
        if self.pileups:
            self._ring_brake(now, vehicles)
        if self.lookalikes_on:
            self._lookalikes(now, vehicles)
        if int(now * 2) % 4 == 1:  # every 2 s
            self._unsqueeze(now, vehicles)
        try:
            cols = c.simulation.getCollisions()
        except Exception:  # noqa: BLE001 - older SUMO
            cols = []
        for col in cols:
            self._collision(now, col, vehicles)
        for inc in list(self.incidents):
            try:
                self._advance(inc, now, vehicles)
            except Exception as e:  # noqa: BLE001 - one incident's trouble must not stop the town
                log.warning("incident %d: %r", inc.id, e)
        for inc in [i for i in self.incidents if i.phase == "closed"]:
            self.incidents.remove(inc)
            self.done.append(inc)
            for vid in [v for v, r in self.role.items() if r[0] == inc.id]:
                self.role.pop(vid, None)
        if not self.allowed():  # nobody looks away while no new incident may start
            for vid, d in list(self.distracted.items()):
                if d.kind != "late":  # (the cars behind a crash on the ring: they react late to it, see _pileup)
                    self._calm(vid)

    def unjam(self, now: float, vehicles: dict, after_s: float = UNJAM_S) -> int:
        """Knots undone, whether you watch or not (24 Oct: before, a knot in sight stayed until you looked away).
        * A car standing inside a junction for BOX_STUCK_S edges through as a driver would: for NUDGE_S it no longer
          waits for vehicles already in the junction (SUMO speed mode bit 5; it still gives way to those coming up to
          it), then drives as before. Cars standing in a junction each waiting for another are a deadlock of SUMO's
          junction model (seed 26 of the evaluation: three cars for 10 minutes, their exit empty).
        * A car that has not moved for `after_s` gets a new way (rerouted on travel times) if it is in sight, and is
          taken off the road if it is not (its driver gave up and turned off; the day's curve brings another
          elsewhere). SUMO runs without teleporting, so a knot never undoes itself: after an incident at the evening
          rush the town stayed locked for the rest of the run (115 cars standing over 2 minutes, 28 min,
          tools/check_accidents.py). Incident vehicles and buses are never taken."""
        c = self.conn
        n = 0
        nudged = self.nudged = getattr(self, "nudged", {})
        for vid, until in list(nudged.items()):  # back to the normal right of way
            if now >= until:
                try:
                    c.vehicle.setSpeedMode(vid, 31)
                except Exception:  # noqa: BLE001 - gone
                    pass
                del nudged[vid]
        for vid, v in vehicles.items():
            depot_car = self.depots is not None and self.depots.owns(vid) and vid not in self.role
            if v.speed > 0.3 or vid in self.role or vid.startswith("pt_") or (self.protected(vid) and not depot_car):
                self.jammed.pop(vid, None)
                getattr(self, "green_wait", {}).pop(vid, None)
                getattr(self, "line_seen", {}).pop(vid, None)
                continue
            t0 = self.jammed.setdefault(vid, now)
            if vid not in nudged and now - t0 >= BOX_STUCK_S:
                try:
                    lane = c.vehicle.getLaneID(vid)
                except Exception:  # noqa: BLE001
                    lane = ""
                if lane.startswith(":"):
                    try:
                        c.vehicle.setSpeedMode(vid, 31 | 32)
                        nudged[vid] = now + NUDGE_S
                        self.stats["box_nudged"] = self.stats.get("box_nudged", 0) + 1
                    except Exception:  # noqa: BLE001
                        pass
                    continue
            # first at the stop line, its light green, and still it stands: it gives way to cars that are themselves
            # stuck (a knot of SUMO's right of way: seed 7, watched, an ambulance through four greens). After
            # GREEN_STUCK_S of that it goes, as a driver would, for NUDGE_S not giving way (it still keeps its distance
            # to the car ahead and stops at red)
            gw = self.green_wait = getattr(self, "green_wait", {})
            seen = self.line_seen = getattr(self, "line_seen", {})  # vid -> when last found not first at a line
            if vid not in nudged and now - t0 >= 30.0 and (vid in gw or now - seen.get(vid, -1e9) >= 10.0):
                try:
                    lane = c.vehicle.getLaneID(vid)
                    end = self.net.getLane(lane).getLength() - c.vehicle.getLanePosition(vid) \
                        if lane and not lane.startswith(":") else 1e9
                    tl = c.vehicle.getNextTLS(vid)[:1] if end < 3.0 else []
                except Exception:  # noqa: BLE001
                    tl = []
                if tl and tl[0][3] in "gG" and tl[0][2] < 3.0:
                    g0 = gw.setdefault(vid, now)
                    if now - g0 >= GREEN_STUCK_S:
                        try:
                            c.vehicle.setSpeedMode(vid, (31 | 32) & ~8)
                            nudged[vid] = now + NUDGE_S
                            self.stats["entry_nudged"] = self.stats.get("entry_nudged", 0) + 1
                        except Exception:  # noqa: BLE001
                            pass
                        gw.pop(vid, None)
                        continue
                else:
                    gw.pop(vid, None)
                    seen[vid] = now
            # at the end of a lane that does not lead to its next road, waiting to move over (two cars side by side
            # each wanting the other's lane stood for good at a green: 15_4 and 15_5, seed 23 watched): it takes a
            # road its lane leads to
            if now - t0 >= KEEP_LANE_S and vid not in getattr(self, "kept_lane", set()):
                if self._keep_lane(vid, near_end=12.0):
                    self.kept_lane = getattr(self, "kept_lane", set()) | {vid}
                    self.jammed[vid] = now
                    continue
            if depot_car:
                continue  # (a depot's vehicle is never taken off: its depot counts on it)
            if now - t0 < after_s:
                continue
            if self.visible is not None and self.visible(v.x, v.y):
                try:
                    c.vehicle.rerouteTraveltime(vid, currentTravelTimes=True)
                    self.stats["rerouted_in_sight"] = self.stats.get("rerouted_in_sight", 0) + 1
                except Exception:  # noqa: BLE001
                    pass
                # (3 Nov) the quickest way by travel times may need a lane it cannot reach from where it stands (the
                # one _keep_lane took it off): a car at the end of 8_4, its turn only from 8_3 behind two wrecks, was
                # put back on that turn every 150 s and stood at a green for 20 minutes, the queue behind it too, and
                # with it the police car and the recovery trucks (tools/check_incident_scene.py --share 0.5, seed 11)
                if self._keep_lane(vid, near_end=12.0):
                    self.stats["kept_lane_after_reroute"] = self.stats.get("kept_lane_after_reroute", 0) + 1
                self.jammed[vid] = now  # again after another after_s, if still standing
                continue
            try:
                self._remove(vid)
                n += 1
            except Exception:  # noqa: BLE001
                pass
            self.jammed.pop(vid, None)
        self.stats["unjammed"] = self.stats.get("unjammed", 0) + n
        return n

    # ------------------------------------------------------------------ 31 Oct: speeders, breakdowns, look-alikes
    def _speeding(self, now: float, vehicles: dict) -> None:
        """The speeders' factor, set once they are on the road (type_for runs before they are added)."""
        for vid, (f, t) in list(self._speed_pending.items()):
            if vid in vehicles:
                try:
                    self.conn.vehicle.setSpeedFactor(vid, f)
                except Exception:  # noqa: BLE001
                    pass
                self._speed_pending.pop(vid, None)
            elif now - t > 120.0:  # never came
                self._speed_pending.pop(vid, None)
                self.speeders.pop(vid, None)
        if int(now * 2) % 120 == 3:  # every minute: those gone
            for vid in [v for v in self.speeders if v not in vehicles and v not in self._speed_pending]:
                self.speeders.pop(vid, None)

    def _place_ok(self, lane: str, x: float, y: float) -> bool:
        """Where an incident may start: a road (not a junction) of MIN_CRASH_ROAD_M or more, not at a depot, not
        within NEAR_OTHER_M of another incident under way."""
        if not lane or lane.startswith(":"):
            return False
        try:
            if self.net.getLane(lane).getLength() < MIN_CRASH_ROAD_M:
                return False
        except Exception:  # noqa: BLE001
            return False
        if lane.rsplit("_", 1)[0] in getattr(self, "depot_edges", set()):
            return False
        return not any(i.phase != "closed" and math.hypot(i.x - x, i.y - y) < NEAR_OTHER_M for i in self.incidents)

    def _look_vids(self) -> set:
        return {lk.vid for lk in self.lookalikes}

    def _breakdowns(self, now: float, vehicles: dict) -> None:
        """A car breaks down (module doc): hazards on, it rolls to a stop in its lane, then it is an incident."""
        c = self.conn
        for vid, b in list(self.breaking.items()):
            v = vehicles.get(vid)
            if v is None:
                self.breaking.pop(vid, None)
                continue
            left = 1.0 - (now - b["t"]) / b["T"]
            if left > 0.0 and v.speed > 0.1:
                try:
                    c.vehicle.setSpeed(vid, max(0.0, b["v0"] * left))
                except Exception:  # noqa: BLE001
                    self.breaking.pop(vid, None)
                continue
            try:
                c.vehicle.setSpeed(vid, 0.0)
            except Exception:  # noqa: BLE001
                pass
            if v.speed > 0.1 and now - b["t"] < b["T"] + 6.0:
                continue  # (still rolling: a car ahead slowed it less than asked)
            self.breaking.pop(vid, None)
            self._broken_down(now, vid, v)
        if self.breaking or self.breakdown_mean_s <= 0 or not self.allowed():
            return
        if self.rng.random() >= 0.5 / self.breakdown_mean_s:
            return
        busy = set(self.role) | set(self.distracted) | self._look_vids()
        cands = sorted(vid for vid, v in vehicles.items() if v.vclass == "passenger" and v.speed > 5.0
                       and vid not in busy and not vid.startswith("pt_"))
        self.rng.shuffle(cands)
        on_ring = self.rng.random() < BREAKDOWN_RING_P
        for vid in cands[:40]:
            v = vehicles[vid]
            try:
                lane = c.vehicle.getLaneID(vid)
                pos = c.vehicle.getLanePosition(vid)
            except Exception:  # noqa: BLE001
                continue
            if (lane.rsplit("_", 1)[0] in self.ring) != on_ring or not self._place_ok(lane, v.x, v.y):
                continue
            # it stops on this road, short of its end: the rest of the road at least its rolling distance and 15 m
            T = max(2.0, v.speed / BREAKDOWN_DECEL)
            if self.net.getLane(lane).getLength() - pos < v.speed * T / 2.0 + 15.0:
                continue
            try:
                c.vehicle.setLaneChangeMode(vid, 0)
            except Exception:  # noqa: BLE001
                continue
            self.breaking[vid] = {"t": now, "v0": v.speed, "lane": lane, "T": T}
            self.careless.pop(vid, None)
            log.info("breakdown: %s slowing to a stop on %s (%.0f km/h)", vid, lane, v.speed * 3.6)
            return

    def _broken_down(self, now: float, vid: str, v) -> None:
        c = self.conn
        try:
            lane = c.vehicle.getLaneID(vid)
        except Exception:  # noqa: BLE001
            return
        if not self.allowed() or not self._place_ok(lane, v.x, v.y):  # (rolled into a junction, or another came)
            try:
                c.vehicle.setSpeed(vid, -1)
                c.vehicle.setLaneChangeMode(vid, LC_DEFAULT)
            except Exception:  # noqa: BLE001
                pass
            log.info("breakdown of %s called off (%s)", vid, lane)
            return
        self.n += 1
        inc = Incident(self.n, now, "breakdown", 0.0, "breakdown", lane.rsplit("_", 1)[0], lane, v.x, v.y)
        self._freeze(vid)
        w = self._wreck(vid, v, "stalled")
        w.damage = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0, "dz": 0.0, "dx": 0.0, "rollover": False, "ramp_s": 0.25,
                    "severity": 0.0, "breakdown": True}
        inc.wrecks.append(w)
        self.role[vid] = (inc.id, "wreck")
        self._fit_in_lane(w)
        self._close_lanes(inc)
        self._divert(inc)
        self._skip_stops(inc)
        self._plan_exits(inc, now)
        self.incidents.append(inc)
        self.last_t = now
        self.stats["incidents"] += 1
        self.stats["breakdowns"] += 1
        inc.log(now, f"breakdown: {vid} stalled on {lane}" + (" (speeding)" if vid in self.speeders else ""))
        self.careless.pop(vid, None)
        self.distracted.pop(vid, None)

    def _lookalikes(self, now: float, vehicles: dict) -> None:
        """Delivery vans stopping in their lane, buses standing long at their stop (module doc)."""
        c = self.conn
        for lk in list(self.lookalikes):
            v = vehicles.get(lk.vid)
            if v is None:
                lk.t1 = lk.t1 or now
            elif lk.t0 is None:
                if v.speed < 0.1:
                    try:
                        stopped = c.vehicle.isStopped(lk.vid)
                    except Exception:  # noqa: BLE001
                        stopped = False
                    if stopped:
                        lk.t0, lk.x, lk.y = now, v.x, v.y
                        try:
                            lk.lane = c.vehicle.getLaneID(lk.vid)
                        except Exception:  # noqa: BLE001
                            pass
                        log.info("look-alike %d: %s %s standing on %s", lk.id, lk.kind, lk.vid, lk.lane)
                elif now - lk.t_told > 60.0:  # never got to its stop
                    lk.t1 = now
            elif v.speed > 0.5 and lk.t1 is None:
                lk.t1 = now
            if lk.t1 is not None:
                if lk.kind == "double-parked":
                    try:
                        c.vehicle.setLaneChangeMode(lk.vid, LC_DEFAULT)
                    except Exception:  # noqa: BLE001
                        pass
                self.lookalikes.remove(lk)
                if lk.t0 is not None:
                    self.lookalikes_done.append(lk)
        # line buses: one stop now and then far longer than its dwell
        if int(now * 2) % 2 == 0:  # every second
            for vid in [v for v in vehicles if v.startswith("pt_")]:
                try:
                    at = c.vehicle.isAtBusStop(vid)
                except Exception:  # noqa: BLE001
                    continue
                if not at:
                    self._bus_at.pop(vid, None)
                    continue
                try:
                    stop = c.vehicle.getStops(vid, 1)[0]
                except Exception:  # noqa: BLE001
                    continue
                key = f"{stop.stoppingPlaceID}@{stop.arrival:.0f}"
                if self._bus_at.get(vid) == key:
                    continue
                self._bus_at[vid] = key
                if len(self.lookalikes) >= LOOKALIKE_MAX or self.rng.random() >= BUS_LONG_P:
                    continue
                s = round(self.rng.uniform(*BUS_LONG_S))
                try:
                    c.vehicle.setStopParameter(vid, 0, "duration", str(s))
                except Exception as e:  # noqa: BLE001
                    log.debug("bus stop %s: %s", vid, e)
                    continue
                self.n_look += 1
                v = vehicles[vid]
                lk = Lookalike(self.n_look, "bus-long-stop", vid, stop.lane, now, s, t0=now, x=v.x, y=v.y,
                               stop=stop.stoppingPlaceID)
                self.lookalikes.append(lk)
                self.stats["bus_long_stops"] += 1
                log.info("look-alike %d: %s stands %d s at %s", lk.id, vid, s, lk.stop)
        if len(self.lookalikes) >= LOOKALIKE_MAX or self.rng.random() >= 0.5 / DOUBLE_PARK_MEAN_S:
            return
        busy = set(self.role) | set(self.distracted) | set(self.breaking) | self._look_vids()
        cands = sorted(vid for vid, v in vehicles.items() if v.vclass == "delivery" and v.speed > 3.0 and vid not in busy)
        self.rng.shuffle(cands)
        for vid in cands[:10]:
            v = vehicles[vid]
            try:
                lane = c.vehicle.getLaneID(vid)
                pos = c.vehicle.getLanePosition(vid)
            except Exception:  # noqa: BLE001
                continue
            if not self._place_ok(lane, v.x, v.y) or lane.rsplit("_", 1)[0] in self.ring:
                continue
            ln = self.net.getLane(lane)
            if sum(1 for x in ln.getEdge().getLanes() if x.allows("passenger")) < 2:
                continue  # (a single lane: nobody could pass)
            at = pos + v.speed * v.speed / 6.0 + 4.0  # where it can stop gently
            if at > ln.getLength() - 25.0:
                continue
            s = round(self.rng.uniform(*DOUBLE_PARK_S))
            try:
                c.vehicle.setLaneChangeMode(vid, 0)
                c.vehicle.setStop(vid, ln.getEdge().getID(), pos=at, laneIndex=ln.getIndex(), duration=s)
            except Exception as e:  # noqa: BLE001
                log.debug("double-park %s: %s", vid, e)
                try:
                    c.vehicle.setLaneChangeMode(vid, LC_DEFAULT)
                except Exception:  # noqa: BLE001
                    pass
                continue
            self.n_look += 1
            self.lookalikes.append(Lookalike(self.n_look, "double-parked", vid, lane, now, s))
            self.stats["double_parked"] += 1
            log.info("look-alike %d: %s to stop on %s for %d s", self.n_look, vid, lane, s)
            return

    def _ring_brake(self, now: float, vehicles: dict) -> None:
        """A car on the ring road brakes hard all of a sudden; the cars behind it may react late (module doc)."""
        for vid, until in list(self._hard.items()):
            if now >= until or vid not in vehicles or vid in self.role:
                self._hard.pop(vid, None)
                if vid in vehicles and vid not in self.role:
                    try:
                        self.conn.vehicle.setSpeedMode(vid, 31)
                        self.conn.vehicle.setSpeed(vid, -1)
                    except Exception:  # noqa: BLE001
                        pass
        if not self.ring or self.ring_brake_mean_s <= 0 or not self.allowed() or \
                self.rng.random() >= 0.5 / self.ring_brake_mean_s:
            return
        c = self.conn
        busy = set(self.role) | set(self.distracted) | set(self.breaking) | self._look_vids()
        cands = sorted(vid for vid, v in vehicles.items() if v.speed > 18.0 and vid not in busy
                       and v.vclass not in NEVER_CARELESS and not vid.startswith("pt_"))
        self.rng.shuffle(cands)
        for vid in cands[:40]:
            try:
                lane = c.vehicle.getLaneID(vid)
                f = c.vehicle.getFollower(vid, RING_BRAKE_GAP_M)
            except Exception:  # noqa: BLE001
                continue
            if lane.startswith(":") or lane.rsplit("_", 1)[0] not in self.ring or not f or not f[0] \
                    or f[0] not in vehicles or vehicles[f[0]].speed < 18.0:
                continue
            try:
                c.vehicle.setSpeedMode(vid, 31 & ~4)  # past its normal deceleration: an emergency stop
                c.vehicle.slowDown(vid, RING_BRAKE_TO_MS, RING_BRAKE_S)
            except Exception:  # noqa: BLE001
                continue
            self._hard[vid] = now + RING_BRAKE_S + 2.0
            self.stats["sudden_brakes"] += 1
            log.info("ring: %s brakes hard on %s (%.0f km/h), %s behind it", vid, lane, vehicles[vid].speed * 3.6, f[0])
            self._late_chain(vid, vehicles, None, 0, f"{vid} braking hard", late_s=RING_BRAKE_LATE_S)
            return

    def _late_chain(self, start: str, vehicles: dict, inc_id: int | None, k0: int, why: str,
                    inc: Incident | None = None, late_s: tuple = PILEUP_LATE_S) -> None:
        """The cars behind `start` in its lane: the k-th may react late (PILEUP_LATE_P[k], from k0 on)."""
        c = self.conn
        cur, back = start, 0.0
        for k, p in list(enumerate(PILEUP_LATE_P))[k0:]:
            try:
                f = c.vehicle.getFollower(cur, PILEUP_REACH_M)
            except Exception:  # noqa: BLE001
                return
            if not f or not f[0] or f[0] not in vehicles:
                return
            fid, gap = f
            back += gap
            v = vehicles[fid]
            if back > PILEUP_REACH_M or fid in self.role or v.vclass in NEVER_CARELESS or fid.startswith("pt_"):
                return
            cur = fid
            if fid in self.distracted or v.speed < PILEUP_MIN_MS or self.rng.random() >= p:
                continue
            try:
                c.vehicle.setSpeedMode(fid, 0)
                c.vehicle.setSpeed(fid, v.speed)
            except Exception:  # noqa: BLE001
                continue
            self.distracted[fid] = Distraction(fid, "late", self.now + self.rng.uniform(*late_s), v.speed)
            self.late_for[fid] = inc_id
            self._late_t[fid] = self.now
            self.stats["pileup_late"] += 1
            text = f"{fid} ({k + 1}. behind, {v.speed * 3.6:.0f} km/h) late to react to {why}"
            if inc is not None:
                inc.log(self.now, text)
            else:
                log.info(text)

    def _pileup(self, inc: Incident, vehicles: dict, start: str | None = None, k0: int = 0) -> None:
        """A crash on the ring road: the cars coming up behind it in its lane may react late (module doc). From the
        rear-most wreck in its lane, or (start) from a car that has just run into the wrecks: the chance for the k-th
        car behind the first crash is PILEUP_LATE_P[k]."""
        if start is None:
            same = [w for w in inc.wrecks if w.lane == inc.lane] or inc.wrecks
            start = min(same, key=lambda w: w.pos).vid
        self._late_chain(start, vehicles, inc.id, k0, "the crash", inc)

    def _on_ring(self, lane: str) -> bool:
        """On the ring road: one of its roads, or a junction a ring road leads into (where its queues stand)."""
        if not lane:
            return False
        if lane.startswith(":"):
            frm = self._via.get(lane, (None, None))[0]
            return bool(frm) and frm.rsplit("_", 1)[0] in self.ring
        return lane.rsplit("_", 1)[0] in self.ring

    def _behind(self, vid: str, n: int = 4) -> list:
        out, cur = [], vid
        for _ in range(n):
            try:
                f = self.conn.vehicle.getFollower(cur, PILEUP_REACH_M)
            except Exception:  # noqa: BLE001
                break
            if not f or not f[0]:
                break
            out.append(f[0])
            cur = f[0]
        return out

    def _veh(self, vid: str):
        """A small stand-in for sumo_side.Vehicle (vclass, speed) of a vehicle on the road now, or None."""
        c = self.conn
        try:
            from types import SimpleNamespace

            return SimpleNamespace(vclass=c.vehicle.getVehicleClass(vid), speed=c.vehicle.getSpeed(vid))
        except Exception:  # noqa: BLE001
            return None

    def lookalike_snapshot(self) -> list[dict]:
        return [{"id": lk.id, "kind": lk.kind, "vid": lk.vid, "lane": lk.lane, "x": lk.x, "y": lk.y, "t_told": lk.t_told,
                 "t0": lk.t0, "t1": lk.t1, "seconds": lk.seconds, "stop": lk.stop}
                for lk in self.lookalikes_done + self.lookalikes if lk.t0 is not None]

    def _focus_step(self, vehicles: dict) -> set:
        """Option A (FOCUS_*): the cars heading into a camera's clear view this step; those seen there for the first
        time are made careless with chance focus_p. -> the set of those heading in (their lapses are FOCUS_BOOST times
        as likely, _distractions)."""
        out = set()
        near = getattr(self.focus, "near", None)
        for vid, v in vehicles.items():
            if v.speed < FOCUS_MIN_MS or vid in self.role or vid.startswith("pt_") or v.vclass != "passenger":
                continue
            if near is not None and not near(v.x, v.y, FOCUS_NEAR_M):
                continue
            lead = None
            try:
                ld = self.conn.vehicle.getLeader(vid, FOCUS_LEAD_M) if self.conn is not None else None
                lead = vehicles.get(ld[0]) if ld else None
            except Exception:  # noqa: BLE001
                lead = None
            if lead is not None:  # it would run into the car ahead: where that one is
                la = math.radians(lead.angle)
                hit = [(lead.x - math.sin(la) * lead.length / 2, lead.y - math.cos(la) * lead.length / 2, lead.angle)]
            else:
                a = math.radians(v.angle)
                ux, uy = math.sin(a), math.cos(a)
                half = v.length / 2.0
                hit = [(v.x + ux * (v.speed * t - half), v.y + uy * (v.speed * t - half), v.angle) for t in FOCUS_AHEAD_S]
            if not any(self.focus(*h) for h in hit):
                continue
            out.add(vid)
            if vid in self.focus_seen or not self.allowed():  # (decided when a new incident may start)
                continue
            self.focus_seen.add(vid)
            if vid not in self.careless and self.focus_p > 0 and self.focus_rng.random() < self.focus_p:
                self.careless[vid] = v.type_id
                self.focus_made.add(vid)
                self.stats["focus_careless"] = self.stats.get("focus_careless", 0) + 1
        if len(self.focus_seen) > 20000:  # (long runs: forget the cars long gone)
            self.focus_seen &= set(vehicles)
            self.focus_made &= set(vehicles) | set(self.role)
        return out

    def _in_junction(self, vid: str) -> bool:
        try:
            return self.conn.vehicle.getLaneID(vid).startswith(":")
        except Exception:  # noqa: BLE001 - gone
            return False

    def _calm(self, vid: str) -> None:
        if self.distracted.pop(vid, None) is not None:
            try:
                self.conn.vehicle.setSpeedMode(vid, 31)
                self.conn.vehicle.setSpeed(vid, -1)
            except Exception:  # noqa: BLE001 - gone
                pass

    def _distractions(self, now: float, vehicles: dict, tls: dict) -> None:
        c = self.conn
        for vid, d in list(self.distracted.items()):
            if d.kind == "red light" and not getattr(d, "blind", False) and vid in vehicles and self._in_junction(vid):
                # (3 Nov) in the junction it does not brake for who crosses (its safe-gap mode did: 4.5 m/s2 for 2-3 s
                # before the hit, tools/measure_impact.py): it goes on as it entered
                try:
                    c.vehicle.setSpeedMode(vid, 0)
                    c.vehicle.setSpeed(vid, vehicles[vid].speed)
                    d.blind = True
                except Exception:  # noqa: BLE001 - gone
                    pass
            if now >= d.until:
                if d.kind == "red light" and now < d.until + RED_THROUGH_MAX_S and self._in_junction(vid):
                    continue  # (3 Nov) looks up only once through the junction: see RED_THROUGH_MAX_S
                self._calm(vid)
        if not self.allowed():
            return
        p = DISTRACT_PER_S * 0.5  # per 0.5 s step
        for vid in self.careless:
            if vid in self.distracted or vid in self.role:
                continue
            v = vehicles.get(vid)
            if v is None or v.speed < 5.0:
                continue
            # A lapse of attention matters when something happens ahead: the car in front slower and close, or a red
            # light coming up. Lapses are sampled more often then (LAPSE_AHEAD x), so that the few minutes someone
            # watches have their accidents; whether a lapse ends in a crash is still SUMO's traffic alone.
            kind, boost = "phone", 1.0
            try:
                lead = c.vehicle.getLeader(vid, 40.0)
            except Exception:  # noqa: BLE001
                lead = None
            if lead and lead[0] in vehicles and vehicles[lead[0]].speed < v.speed - 2.0:
                boost = LAPSE_AHEAD
            try:
                nxt = c.vehicle.getNextTLS(vid)
            except Exception:  # noqa: BLE001
                nxt = []
            if nxt:
                tid, _, dist, state = nxt[0]
                walk = self.walk is not None and tls.get(tid) and self.walk.is_walk(tid, tls[tid])
                if state in "rR" and 8.0 < dist < 45.0 and not walk:
                    kind, boost = "red light", LAPSE_AHEAD
            focus = vid in self._approaching
            if focus:  # (6 Nov, option A) heading into a camera's clear view
                boost = max(boost, FOCUS_BOOST)
            elif vid in self.focus_made:  # made careless there: only there
                continue
            if self.rng.random() >= p * boost:
                continue
            try:
                if kind == "phone":
                    c.vehicle.setSpeedMode(vid, 0)       # no safe speed, no limits: keeps going as it is
                    c.vehicle.setSpeed(vid, v.speed)
                else:
                    c.vehicle.setSpeedMode(vid, 0b00111)  # safe gap to the leader kept; red and right of way ignored
            except Exception:  # noqa: BLE001
                continue
            self.distracted[vid] = Distraction(vid, kind, now + self.rng.uniform(1.5, 3.0), v.speed)
            self.distracted[vid].focus = focus
            self.stats["distractions"] += 1

    # ------------------------------------------------------------------ collisions
    def _collision(self, now: float, col, vehicles: dict) -> None:
        a, b = col.collider, col.victim
        if (a, b) in self._pairs:
            return
        self._pairs.add((a, b))
        self.stats["collisions"] += 1
        if a not in vehicles or b not in vehicles:  # a pedestrian
            self.stats["person_hits"] += 1
            self._halt(a if a in vehicles else b, 2.0)
            return
        ra, rb = self.role.get(a), self.role.get(b)
        if ra and rb:
            return  # already part of an incident (the wrecks touch)
        late = next((self.late_for[x] for x in (a, b) if self.late_for.get(x) is not None), None)
        if ra or rb or late is not None:
            # someone ran into a wreck (or the wreck into nobody), or a car late to react behind a crash on the ring
            # hit the car ahead of it: a pile-up, if they are ordinary cars
            inc_id = (ra or rb)[0] if (ra or rb) else late
            inc = next((i for i in self.incidents if i.id == inc_id), None)
            new = [x for x in (a, b) if x not in self.role]
            cap = PILEUP_MAX_WRECKS if self.pileups and inc is not None and self._on_ring(inc.lane) else 3
            if (inc is None or inc.phase not in ("crash", "police en route") or len(inc.wrecks) + len(new) > cap
                    or any(vehicles[x].vclass in NEVER_CARELESS or x.startswith("pt_") for x in new)):
                for x in new:
                    self._halt(x, 2.5)
                self.stats["near_misses"] += 1
                self.near_why['join refused'] = self.near_why.get('join refused', 0) + 1
                return
            sev_j = abs(col.colliderSpeed - col.victimSpeed)
            if sev_j < 1.0:  # a touch at walking pace (a car creeping up to the one ahead): no crash
                for x in new:
                    self._halt(x, 2.5)
                self.stats["near_misses"] += 1
                self.near_why['touch'] = self.near_why.get('touch', 0) + 1
                return
            f = self._to_contact(a, b, vehicles)
            for x in new:
                self._join(inc, x, vehicles[x], "collider" if x == a else "victim", sev_j, now,
                           t_hit=round(now - (1.0 - f) * STEP_S, 3))
            return
        va, vb = vehicles[a], vehicles[b]
        if (va.vclass in NEVER_CARELESS or vb.vclass in NEVER_CARELESS or a.startswith("pt_") or b.startswith("pt_")
                or not self.allowed()):
            self._halt(a, 2.0)
            self.stats["near_misses"] += 1
            self.near_why['not allowed'] = self.near_why.get('not allowed', 0) + 1
            return
        kind = {"collision": "rear-end", "frontal": "frontal", "junction": "junction", "side": "side"}.get(col.type,
                                                                                                             col.type)
        sa, sb = col.colliderSpeed, col.victimSpeed
        if kind == "rear-end":
            sev = abs(sa - sb)
        elif kind == "frontal":
            sev = sa + sb
        else:
            sev = math.hypot(sa, sb)
        lane = col.lane
        edge = lane.rsplit("_", 1)[0]
        short = False
        for ln in (lane, *self._lanes_of(a, b)):  # (a junction crash: where the two stand, too)
            try:
                short = short or (not ln.startswith(":") and self.net.getLane(ln).getLength() < MIN_CRASH_ROAD_M)
            except Exception:  # noqa: BLE001
                pass
        mx, my = (va.x + vb.x) / 2, (va.y + vb.y) / 2
        near = any(i.phase != "closed" and math.hypot(i.x - mx, i.y - my) < NEAR_OTHER_M for i in self.incidents)
        dep = getattr(self, "depot_edges", set())
        djs = getattr(self, "depot_junctions", set())
        at_depot = any(ln.rsplit("_", 1)[0] in dep or (ln.startswith(":") and ln[1:].rsplit("_", 2)[0] in djs)
                       for ln in (lane, *self._lanes_of(a, b)))
        if at_depot or short or near:
            self._halt(a, 2.0)
            self.stats["near_misses"] += 1
            self.near_why['place'] = self.near_why.get('place', 0) + 1
            return
        self.n += 1
        f = self._to_contact(a, b, vehicles)
        inc = Incident(self.n, now, kind, round(sev, 1), severity_label(sev), edge, lane,
                       (va.x + vb.x) / 2, (va.y + vb.y) / 2)
        inc.t_contact = round(now - (1.0 - f) * STEP_S, 3)
        inc.police_due = now + police_delay(self.police_rng, self.police_delay_s, self.police_spread)
        inc.police_record = {"due": round(inc.police_due, 1), "sent": None, "on_scene": None, "how": None, "cars": 0}
        for vid, v, role in ((a, va, "collider"), (b, vb, "victim")):
            self._freeze(vid)
            w = self._wreck(vid, v, role)
            w.t_hit = inc.t_contact
            w.hit_pose = (w.x, w.y, w.angle)
            inc.wrecks.append(w)
            self.role[vid] = (inc.id, "wreck")
        if kind == "rear-end":
            self._order_rear_end(inc)
            self._touching(inc)
        for w in inc.wrecks:
            w.damage = self._damage(sev, kind, w, inc)
            self._fit_in_lane(w)
        # drivers coming up see the wrecks and merge into the open lane at once (SUMO: the lane is closed to them);
        # left open, the queue behind the wrecks filled the junction before and locked the town
        self._close_lanes(inc)
        self._divert(inc)
        self._skip_stops(inc)
        self._plan_exits(inc, now)
        self.incidents.append(inc)
        self.last_t = now
        self.stats["incidents"] += 1
        d = self.distracted.get(a)
        inc.cause = d.kind if d else None
        if a in self.focus_made or b in self.focus_made or (d is not None and d.focus):
            inc.crash_source = "camera focus"
            self.stats["focus_incidents"] = self.stats.get("focus_incidents", 0) + 1
        if a in self.late_for or b in self.late_for:
            inc.cause = "late to react to a car braking hard"
        fast = [x for x in (a, b) if x in self.speeders]
        hitter = next((w.vid for w in inc.wrecks if w.role == "collider"), a)
        inc.log(now, f"{kind} crash, {inc.label} ({sev * 3.6:.0f} km/h): {hitter} into {b if hitter == a else a}"
                     + (f" ({d.kind})" if d else "") + (f" (speeding: {', '.join(fast)})" if fast else ""))
        for vid in (a, b):
            self.careless.pop(vid, None)
            self.distracted.pop(vid, None)
        if self.pileups and self._on_ring(lane):
            self._pileup(inc, vehicles)

    def _join(self, inc: Incident, vid: str, v, role: str, sev: float, now: float, t_hit: float | None = None) -> None:
        """A car that ran into an incident's wrecks (or, late to react, into the car ahead of it) becomes one of its
        wrecks: a pile-up from three cars on."""
        self._freeze(vid)
        w = self._wreck(vid, v, role)
        w.t_hit = now if t_hit is None else t_hit
        w.hit_pose = (w.x, w.y, w.angle)
        w.damage = self._damage(sev, "rear-end", w, inc)
        self._fit_in_lane(w)
        inc.wrecks.append(w)
        self._plan_exits(inc, now, [w])
        self.role[vid] = (inc.id, "wreck")
        self.distracted.pop(vid, None)
        self.careless.pop(vid, None)
        late = self.late_for.pop(vid, None) is not None
        inc.log(now, f"{vid} ran into the wreck ({sev * 3.6:.0f} km/h"
                     + (", late to react" if late else "")
                     + (", speeding" if vid in self.speeders else "") + ")")
        self.stats["joined"] += 1
        if inc.kind == "breakdown":  # someone ran into the stalled car: now a crash
            inc.kind, inc.severity, inc.label = "rear-end", round(sev, 1), severity_label(sev)
            inc.cause = "ran into a stalled car"
        elif len(inc.wrecks) >= 3 and inc.kind != "pile-up":
            inc.kind = "pile-up"
            self.stats["pileups"] += 1
        if sev > inc.severity:
            inc.severity, inc.label = round(sev, 1), severity_label(sev)
        if inc.closed and w.lane and not w.lane.startswith(":") and w.lane not in {cl for cl, _ in inc.closed}:
            inc.log(now, f"{vid} stands on {w.lane}, beside the closed lane")
        if self.pileups and self._on_ring(inc.lane) and len(inc.wrecks) < PILEUP_MAX_WRECKS:
            vehicles = {x: v for x, v in ((x, self._veh(x)) for x in self._behind(vid)) if v is not None}
            self._pileup(inc, vehicles, start=vid, k0=len(inc.wrecks) - 1)  # the car behind it sees it stop at once

    def _keep_lane(self, vid: str, near_end: float | None = None) -> bool:
        """A vehicle standing on a lane that does not lead to the next road of its route takes a road its lane leads to
        (on to where it was going if there is a way, else elsewhere; a depot's vehicle round a loop and back)."""
        c = self.conn
        try:
            if c.vehicle.getSpeed(vid) > 0.3 or c.vehicle.getStopState(vid):
                return False
            ln = c.vehicle.getLaneID(vid)
            if not ln or ln.startswith(":"):
                return False
            route = list(c.vehicle.getRoute(vid))
            i = c.vehicle.getRouteIndex(vid)
            vt = c.vehicle.getTypeID(vid)
        except Exception:  # noqa: BLE001
            return False
        lane = self.net.getLane(ln)
        if near_end is not None:
            try:
                if lane.getLength() - c.vehicle.getLanePosition(vid) > near_end:
                    return False  # (in a queue it may still move over before the end)
            except Exception:  # noqa: BLE001
                return False
        eid = lane.getEdge().getID()
        reach = sorted({x.getTo().getID() for x in lane.getOutgoing() if x.getTo().allows("passenger")})
        if i + 1 >= len(route) or route[i] != eid or route[i + 1] in reach or not reach:
            return False
        depot_car = self.depots is not None and self.depots.owns(vid)
        best = None
        for alt in reach:
            try:
                r = self._route(alt, route[-1], vt, vid) if alt != route[-1] else None
            except Exception:  # noqa: BLE001
                continue
            if r is not None and not r.edges:
                continue
            cost = r.travelTime if r is not None else 0.0
            if best is None or cost < best[0]:
                best = (cost, [alt] if r is None else list(r.edges))
        if best is None and depot_car:
            best = self._detour_home(vid, reach, route[-1], eid)
        if best is None and not depot_car:
            for alt in reach:
                on = self._onwards(alt, vt.split("@" + vid)[0])
                if on and on[0] == alt:
                    best = (0.0, on)
                    break
        if best is None:
            return False
        try:
            c.vehicle.setRoute(vid, [eid] + best[1])
            self.stats["turns_rerouted"] = self.stats.get("turns_rerouted", 0) + 1
            return True
        except Exception as e:  # noqa: BLE001
            log.debug("keep lane %s: %s", vid, e)
            return False

    def _lanes_of(self, *vids: str) -> list:
        out = []
        for vid in vids:
            try:
                out.append(self.conn.vehicle.getLaneID(vid))
            except Exception:  # noqa: BLE001
                pass
        return out

    def _halt(self, vid: str, s: float) -> None:
        """A near miss: the car that hit brakes (hard) and stands `s` seconds, then drives on (they part)."""
        try:
            self.conn.vehicle.setSpeedMode(vid, 31)
            self.conn.vehicle.setSpeed(vid, 0.0)
            self.halted[vid] = self.now + s
        except Exception:  # noqa: BLE001
            pass
        self.distracted.pop(vid, None)

    def _freeze(self, vid: str) -> None:
        c = self.conn
        try:
            c.vehicle.setSpeedMode(vid, 0)
            c.vehicle.setSpeed(vid, 0.0)
            c.vehicle.setLaneChangeMode(vid, 0)
            c.vehicle.setRoutingMode(vid, 0)
        except Exception as e:  # noqa: BLE001
            log.debug("freeze %s: %s", vid, e)

    def _wreck(self, vid: str, v, role: str) -> Wreck:
        c = self.conn
        try:
            lane, pos = c.vehicle.getLaneID(vid), c.vehicle.getLanePosition(vid)
        except Exception:  # noqa: BLE001
            lane, pos = "", 0.0
        from cosim import vmodels

        x, y, ang, pos = self._moved.pop(vid, (v.x, v.y, v.angle, pos))  # put back to where it touched (_to_contact)
        return Wreck(vid, vmodels.model_of(v.type_id), role, lane, pos, x, y, ang, v.length, v.speed,
                     speeding=vid in self.speeders)

    WIDTH_M = {"bus": 2.5, "coach": 2.5, "truck": 2.5, "trailer": 2.5, "delivery": 2.1, "motorcycle": 0.9}

    def _to_contact(self, a: str, b: str, vehicles: dict) -> float:
        """Two cars SUMO found inside each other at the end of this step: when in the step they touched
        (contact_fraction, from where they were at the step before), and each put back along its lane by the way it
        went after that. -> that fraction of the step. (Before 3 Nov they stood inside each other, the viewer drew them
        so, and a rear-end's follower was pulled back out over the next step: tools/measure_impact.py.)"""
        c = self.conn
        va, vb = vehicles.get(a), vehicles.get(b)
        pa, pb = self._before.get(a), self._before.get(b)
        if va is None or vb is None:
            return 1.0
        pa = pa or va
        pb = pb or vb
        size = {}
        for vid, v in ((a, va), (b, vb)):
            try:
                size[vid] = (v.length, c.vehicle.getWidth(vid))
            except Exception:  # noqa: BLE001
                size[vid] = (v.length, self.WIDTH_M.get(v.vclass, 1.85))
        f = contact_fraction((pa.x, pa.y, pa.angle), (va.x, va.y, va.angle), (pb.x, pb.y, pb.angle),
                             (vb.x, vb.y, vb.angle), size[a], size[b])
        if f >= 1.0:
            return 1.0
        for vid, v, p in ((a, va, pa), (b, vb, pb)):
            went = math.hypot(v.x - p.x, v.y - p.y)
            back = (1.0 - f) * went
            if back < 0.02:
                continue
            try:
                lane, pos = c.vehicle.getLaneID(vid), c.vehicle.getLanePosition(vid)
            except Exception as e:  # noqa: BLE001
                log.debug("to contact %s: %s", vid, e)
                continue
            # where it touched, as drawn: back along the way it went in the step, all the way (8 Nov: a car doing
            # 86 km/h that had come onto this lane in the step went back only to the lane's start; it was drawn
            # hitting from inside the other car, centres 0.4 m apart)
            k = back / went
            d = (v.angle - p.angle + 180.0) % 360.0 - 180.0
            x, y, ang = v.x + (p.x - v.x) * k, v.y + (p.y - v.y) * k, (v.angle - d * k) % 360.0
            self._moved[vid] = (x, y, ang, pos)
            back = min(back, max(pos - 0.1, 0.0))  # SUMO: on this lane, as far back as it goes (_touching ends it)
            if back < 0.02:
                continue
            try:
                if lane.startswith(":"):  # in a junction, where it is across its lane: the very place
                    edge, idx = lane.rsplit("_", 1)
                    c.vehicle.moveToXY(vid, edge, int(idx), x, y, ang, keepRoute=1)
                else:  # (on a road it is then centred in its lane: _fit_in_lane)
                    c.vehicle.moveTo(vid, lane, pos - back)
            except Exception as e:  # noqa: BLE001
                log.debug("to contact %s: %s", vid, e)
                continue
            # (SUMO reports the new place only after the next step)
            self._moved[vid] = (x, y, ang, pos - back)
        return f

    def _touching(self, inc: Incident) -> None:
        """Rear-end: SUMO noticed the crash a step late, the follower partly inside the leader (or past it); they are
        set bumper to bumper, the follower (the collider, _order_rear_end) behind. The follower goes back; when its
        lane has no room behind (it came onto it in the step), the leader goes on instead."""
        c = self.conn
        ws = [w for w in inc.wrecks if w.lane == inc.wrecks[0].lane and w.lane]
        if len(ws) != 2:
            return
        roles = {w.role: w for w in ws}
        if set(roles) == {"collider", "victim"}:
            lead, foll = roles["victim"], roles["collider"]
        else:
            lead, foll = sorted(ws, key=lambda w: -w.pos)
        gap = (lead.pos - lead.length) - foll.pos
        if gap >= -0.02:
            return
        want = lead.pos - lead.length - 0.05
        try:
            if want >= 0.1:
                c.vehicle.moveTo(foll.vid, foll.lane, want)
                inc.log(self.now, f"{foll.vid} put back {foll.pos - want:.2f} m (bumper to bumper)")
                foll.pos = want
            else:
                on = min(foll.pos + lead.length + 0.05, self.net.getLane(lead.lane).getLength() - 0.1)
                c.vehicle.moveTo(lead.vid, lead.lane, on)
                inc.log(self.now, f"{lead.vid} put on {on - lead.pos:.2f} m (bumper to bumper, no room behind)")
                lead.pos = on
        except Exception as e:  # noqa: BLE001
            log.debug("touching: %s", e)

    def _order_rear_end(self, inc: Incident) -> None:
        """Rear-end: the car behind is the one that hit (the collider), the car ahead is shoved on (the victim, _damage).
        SUMO's collider / victim do not always say so: on the ring road (7 Nov, CARLA) it named a car doing 27 km/h the
        collider of the car doing 72 km/h right behind it; the shove went to the car behind, and it was drawn a metre
        inside the one ahead. Who is ahead: along the lane when both are on it, else along the hit car's heading."""
        ws = [w for w in inc.wrecks if w.role in ("collider", "victim")]
        if len(ws) != 2:
            return
        a, b = ws
        # where they touched (_to_contact: x, y), along the heading (SUMO's angle: degrees clockwise from north); not
        # the lane positions, which SUMO took with the cars already inside each other, up to a step's travel deep
        # (8 Nov: 86 km/h into a standing car, the follower's front past the other car's middle)
        h = math.radians(b.angle)
        a_ahead = (a.x - b.x) * math.sin(h) + (a.y - b.y) * math.cos(h) > 0.0
        lead, foll = (a, b) if a_ahead else (b, a)
        if lead.role == "collider":
            lead.role, foll.role = "victim", "collider"
            self.stats["rear_end_roles_swapped"] = self.stats.get("rear_end_roles_swapped", 0) + 1
            inc.log(self.now, f"{foll.vid} is behind {lead.vid}: it hit, {lead.vid} was hit (SUMO said the other way)")

    def _lane_pose(self, lane: str, pos: float) -> tuple[float, float, float]:
        """SUMO (x, y, angle) at `pos` along `lane`: its middle, its direction (degrees, 0 north, clockwise)."""
        length = self.net.getLane(lane).getLength()
        pos = min(max(pos, 0.0), length)
        a, b = (pos, pos + 0.5) if pos + 0.5 <= length else (max(pos - 0.5, 0.0), pos)
        xa, ya = self._xy(lane, a)
        xb, yb = self._xy(lane, b)
        x, y = self._xy(lane, pos)
        return x, y, math.degrees(math.atan2(xb - xa, yb - ya)) % 360.0

    def _fit_in_lane(self, w: Wreck) -> None:
        """A wreck stands straight in the middle of its lane in SUMO (one hit while changing lanes stood across two,
        and the open lane's traffic stopped for it), and as drawn it is turned no more than its lane holds: a car spun
        across its lane reached into the open one, where SUMO's cars passed through it."""
        if not w.lane or w.lane.startswith(":"):
            return
        c = self.conn
        ln = self.net.getLane(w.lane)
        try:
            x, y, ang = self._lane_pose(w.lane, w.pos)
            c.vehicle.moveToXY(w.vid, ln.getEdge().getID(), ln.getIndex(), x, y, ang, keepRoute=1)
            w.x, w.y, w.angle = x, y, ang
        except Exception as e:  # noqa: BLE001
            log.debug("centre %s: %s", w.vid, e)
        try:
            width = c.vehicle.getWidth(w.vid)
        except Exception:  # noqa: BLE001
            width = 1.8
        d = w.damage
        room = ln.getWidth() / 2.0 + 0.15

        def reach(yaw: float) -> float:
            r = math.radians(abs(yaw))
            return w.length / 2.0 * math.sin(r) + width / 2.0 * math.cos(r) + abs(d.get("dx", 0.0)) * math.sin(r)

        yaw = d.get("yaw", 0.0)
        while abs(yaw) > 0.5 and reach(yaw) > room:
            yaw *= 0.92
        if yaw != d.get("yaw", 0.0):
            d["yaw"] = round(yaw, 2)
            d["fitted"] = True

    def _plan_exits(self, inc: Incident, now: float, wrecks: list | None = None) -> None:
        """Each driver gets out on the pavement's side (the right: SUMO's lane 0 is the pavement) and waits on it,
        beside the car; when that means crossing an open lane, the traffic on it is held from 2 s before until they
        are over (Incident.holds). A car on its side keeps its driver."""
        k_st = 0
        for w in wrecks or inc.wrecks:
            if w.exit is not None or w.damage.get("rollover") or not w.lane:
                continue
            lane, pos_on = w.lane, w.pos - w.length / 2.0
            if lane.startswith(":"):
                lane = self._via.get(lane, (None, None))[0]
                if not lane:
                    continue
                pos_on = self.net.getLane(lane).getLength() - 1.0
            ln = self.net.getLane(lane)
            lanes = ln.getEdge().getLanes()
            walk = lanes[0] if lanes[0].allows("pedestrian") and not lanes[0].allows("passenger") else None
            if walk is None:
                continue
            closed = {cl for cl, _ in inc.closed}
            cross = [x.getID() for x in lanes[1:ln.getIndex()] if x.allows("passenger") and x.getID() not in closed]
            sp = pos_on * walk.getLength() / max(ln.getLength(), 0.01) - 1.5 - 2.5 * k_st
            sp = min(max(sp, 0.5), walk.getLength() - 0.5)
            tx, ty = self._xy(walk.getID(), sp)
            walk_s = 1.4 + (math.hypot(tx - w.x, ty - w.y) + 1.0) / 1.3
            t_out = now + (5.0 if cross else 3.5)
            w.exit = {"t_out": t_out, "target": (tx, ty), "cross": bool(cross), "out": False, "walk_s": walk_s,
                      "walk_lane": walk.getID(), "walk_pos": sp}
            if cross:
                inc.holds.append((t_out - 2.0, t_out + walk_s + 1.5, cross, pos_on + w.length / 2.0))
                inc.log(now, f"traffic on {', '.join(cross)} held for {w.vid}'s driver to reach the pavement")
            k_st += 1

    def _hold(self, inc: Incident, now: float) -> None:
        """Traffic held for people crossing (see _plan_exits): who comes up on those lanes stops before them."""
        if not inc.holds:
            return
        c = self.conn
        live = [h for h in inc.holds if h[0] <= now < h[1]]
        for _, _, lanes, at in live:
            for ln in lanes:
                try:
                    ids = c.lane.getLastStepVehicleIDs(ln)
                except Exception:  # noqa: BLE001
                    continue
                for o in ids:
                    if o in inc.held or o in self.role or self.protected(o):
                        continue
                    try:
                        p = c.vehicle.getLanePosition(o)
                        # and one closer that can still stop gently before them (seed 26: a car rolling up at walking
                        # pace went through a driver crossing to the pavement)
                        sp = c.vehicle.getSpeed(o) if p >= at - HOLD_BEFORE_M else 0.0
                        # (they cross from beside the wreck to the pavement a few metres behind it: nobody stops
                        # closer than HOLD_CLEAR_M before `at`)
                        if p < at - HOLD_BEFORE_M or (p < at - HOLD_CLEAR_M and sp * sp / 8.0 < at - HOLD_CLEAR_M - p):
                            c.vehicle.setSpeed(o, 0.0)
                            inc.held.add(o)
                    except Exception:  # noqa: BLE001
                        pass
        if all(now >= h[1] for h in inc.holds):
            for o in inc.held:
                try:
                    c.vehicle.setSpeed(o, -1)
                except Exception:  # noqa: BLE001
                    pass
            inc.held.clear()
            inc.holds = []

    def _merge(self, inc: Incident) -> None:
        """Drivers on a closed lane behind the wrecks are told to move over to the open lane, well before them (left
        alone they drove up to the wrecks and had to squeeze out sideways at a standstill)."""
        if not inc.open_lane or not inc.closed:
            return
        c = self.conn
        lane, rear, _ = self._anchor(inc)
        oi = int(inc.open_lane.rsplit("_", 1)[1])
        if inc.open_lane.rsplit("_", 1)[0] != lane.rsplit("_", 1)[0]:
            return
        closed = {cl for cl, _ in inc.closed}
        for cl in closed:
            try:
                ids = c.lane.getLastStepVehicleIDs(cl)
            except Exception:  # noqa: BLE001
                continue
            for o in ids:
                if o in inc.merged or o in self.role or self.protected(o):
                    continue
                try:
                    p = c.vehicle.getLanePosition(o)
                    if rear - MERGE_BEFORE_M < p < rear - 0.5:
                        c.vehicle.changeLane(o, oi, 60.0)
                        inc.merged.add(o)
                except Exception:  # noqa: BLE001
                    pass
            # the lanes of the road before that lead only onto the closed lane: drivers on them move over before the
            # junction, onto a lane of that road that leads onto an open one (else they all merge at the scene)
            target = self.net.getLane(cl)
            for frm in target.getEdge().getIncoming():
                if frm.getFunction() == "internal":
                    continue
                conns = frm.getOutgoing().get(target.getEdge(), [])
                good = sorted({k.getFromLane().getIndex() for k in conns if k.getToLane().getID() not in closed})
                if not good:
                    continue
                for fl in {k.getFromLane() for k in conns if k.getToLane().getID() in closed}:
                    if fl.getIndex() in good:
                        continue
                    alt = min(good, key=lambda g: abs(g - fl.getIndex()))
                    try:
                        ids = c.lane.getLastStepVehicleIDs(fl.getID())
                    except Exception:  # noqa: BLE001
                        continue
                    for o in ids:
                        if o in self.role or self.protected(o):
                            continue
                        if o in inc.merged:
                            # still standing at the end of a lane that leads only into the closed one: it edges over
                            # at a crawl (as _still_stuck); it waited there for good before 24 Oct (seed 11, watched)
                            try:
                                if c.vehicle.getSpeed(o) < 0.3 and c.vehicle.getWaitingTime(o) > 30.0 \
                                        and o not in self.squeezed:
                                    self._pull_out(o)
                                    rec = self.squeezed.get(o)
                                    if rec is not None:
                                        c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
                                        c.vehicle.changeLane(o, alt, 30.0)
                                        self.squeezed[o] = rec + (True,)
                                        self.stats["edged_over"] = self.stats.get("edged_over", 0) + 1
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                        try:
                            route, i = c.vehicle.getRoute(o), c.vehicle.getRouteIndex(o)
                            if route[i + 1:i + 2] == (target.getEdge().getID(),) or \
                                    list(route[i + 1:i + 2]) == [target.getEdge().getID()]:
                                c.vehicle.changeLane(o, alt, 60.0)
                                inc.merged.add(o)
                        except Exception:  # noqa: BLE001
                            pass

    RECENTRE_S = 10.0  # standing this long half over the closed lane: it moves back into its own

    def _recentre(self, inc: Incident, now: float, vehicles: dict) -> None:
        """A car on the open lane standing half over the closed one (it pulled out of it and stopped half way, or
        edged towards it to turn) behind the cone line's stand-in or a wreck: it moves back into its own lane at a
        crawl (as _still_stuck). It is on the open lane, so nothing else looked at it, and it stood there for good,
        everything behind it too (seed 23, watched: the recovery truck behind it for 20 minutes)."""
        if not inc.open_lane or not inc.closed:
            return
        c = self.conn
        ol = inc.open_lane
        oi = int(ol.rsplit("_", 1)[1])
        toward = {1 if int(cl.rsplit("_", 1)[1]) > oi else -1 for cl, _ in inc.closed
                  if cl.rsplit("_", 1)[0] == ol.rsplit("_", 1)[0]}
        if len(toward) != 1:
            return
        sgn = toward.pop()  # SUMO: + is to the left
        off = getattr(inc, "off_since", {})
        inc.off_since = off
        try:
            ids = c.lane.getLastStepVehicleIDs(ol)
        except Exception:  # noqa: BLE001
            return
        for o in ids:
            v = vehicles.get(o)
            if o in self.role or v is None:
                continue
            try:
                lat = c.vehicle.getLateralLanePosition(o)
            except Exception:  # noqa: BLE001
                continue
            if v.speed > 0.3 or lat * sgn < 0.5:
                off.pop(o, None)
                continue
            t0 = off.setdefault(o, now)
            rec_t = getattr(inc, "recentred", {})
            if now - t0 < self.RECENTRE_S:
                continue
            last = rec_t.get(o)
            sliding = last is not None and len(last) > 2
            if last is not None and (sliding or (now - last[0] >= self.RECENTRE_S and abs(lat) > abs(last[1]) - 0.1)):
                # asked to and still as far over (one beside the cone line's stand-in, touching it, never moved:
                # seed 23 watched, the recovery truck behind it for good): it slides back, half a metre a step
                try:
                    ln = self.net.getLane(ol)
                    pos = c.vehicle.getLanePosition(o)
                    x, y, ang = self._lane_pose(ol, pos)
                    new_lat = lat - math.copysign(min(0.5, abs(lat)), lat)
                    a = math.radians(ang)
                    c.vehicle.moveToXY(o, ln.getEdge().getID(), ln.getIndex(), x - math.cos(a) * new_lat,
                                       y + math.sin(a) * new_lat, ang, keepRoute=1)
                    self.stats["slid_back"] = self.stats.get("slid_back", 0) + 1
                    rec_t[o] = (last[0], last[1], True)
                except Exception as e:  # noqa: BLE001
                    log.debug("slide back %s: %s", o, e)
                continue
            if last is not None:
                continue
            try:
                if o not in self.squeezed:
                    self._pull_out(o)
                c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
                c.vehicle.changeSublane(o, -lat)
                rec = self.squeezed.get(o)
                if rec is not None and len(rec) <= 4:
                    self.squeezed[o] = rec + (True,)
                inc.recentred = {**getattr(inc, "recentred", {}), o: (now, lat)}
                self.stats["recentred"] = self.stats.get("recentred", 0) + 1
            except Exception as e:  # noqa: BLE001
                log.debug("recentre %s: %s", o, e)

    def _reroute_turns(self, inc: Incident) -> None:
        """Drivers on the incident's road whose next road can only be reached past a closed lane (a left turn from the
        closed left lane, or from a lane beyond it) take another road, one their own side of the closure leads to. Left
        alone they stopped at the end of the open lane, waiting to move into the closed one (not allowed: Incident
        change permissions), and the open lane stood behind them: the recovery truck never got through.
        Only the side they are on: a route that needed them to cross the closed lane (6_4 -> 15_5 for road 13 off
        15_3) left them standing for good, and SUMO 1.27 crashed some time later (seeds 11, 23, 31, 53)."""
        closed = {cl for cl, _ in inc.closed}
        if not closed:
            return
        c = self.conn
        edge = self.net.getLane(next(iter(closed))).getEdge()
        eid = edge.getID()
        lanes = edge.getLanes()
        shut = {int(cl.rsplit("_", 1)[1]) for cl in closed}

        def side(k: int) -> set:
            """The edges reached from the open lanes next to lane k with no closed lane between."""
            out = set()
            for step in (-1, 1):
                j = k
                while 0 <= j < len(lanes) and j not in shut:
                    if lanes[j].allows("passenger"):
                        out |= {x.getTo().getID() for x in lanes[j].getOutgoing()}
                    j += step
            return out

        done = getattr(inc, "rerouted", set())
        for vid in c.edge.getLastStepVehicleIDs(eid):
            # (ambulances and police cars on a run too: standing at the end of the open lane, waiting to move into the
            # closed one, one stood there for good, the recovery truck behind it: seed 7, watched)
            # (a depot's vehicle on its way home or to a call too, keeping where it goes: one on 23_3 waiting for the
            # closed 23_4 to turn to the police station stood there for good, the trucks behind it: seed 41)
            depot_car = self.depots is not None and self.depots.owns(vid)
            if vid in done or vid in self.role or vid.startswith(("inc", "pt_")) or \
                    (self.protected(vid) and not depot_car) or any(vid == t.vid for t in inc.tows):
                continue
            try:
                if c.vehicle.getStopState(vid):
                    continue
                k = c.vehicle.getLaneIndex(vid)
                if k in shut:
                    continue
                route = list(c.vehicle.getRoute(vid))
                i = c.vehicle.getRouteIndex(vid)
            except Exception:  # noqa: BLE001
                continue
            if route[i] != eid:
                continue
            reach = side(k)
            if i + 1 >= len(route) or route[i + 1] in reach or not reach:
                if not depot_car:
                    done.add(vid)
                continue
            best = None
            for alt in sorted(reach):
                try:
                    r = self._route(alt, route[-1], c.vehicle.getTypeID(vid), vid) if alt != route[-1] else None
                except Exception:  # noqa: BLE001
                    continue
                if r is not None and (not r.edges or (eid in r.edges and not depot_car)):
                    continue  # no way on, or back round through here (a depot's vehicle may: round the block)
                cost = r.travelTime if r is not None else 0.0
                if best is None or cost < best[0]:
                    best = (cost, [alt] if r is None else list(r.edges))
            if best is None and depot_car:
                # its depot is reached from here only past the closed lane (the police station from 23, seed 41):
                # round a wider loop, by a road 300-900 m away, and back (by then the lane may be open again)
                best = self._detour_home(vid, sorted(reach), route[-1], eid)
            if best is None and not depot_car:  # no way from its side to where it was going: it goes somewhere else
                for alt in sorted(reach):
                    on = self._onwards(alt, c.vehicle.getTypeID(vid).split("@" + vid)[0])
                    if on and on[0] == alt and eid not in on:
                        best = (0.0, on)
                        break
            if best is not None:
                try:
                    c.vehicle.setRoute(vid, [eid] + best[1])
                    if not depot_car:  # (round the block, it may need another way when it comes back)
                        done.add(vid)
                    self.stats["turns_rerouted"] = self.stats.get("turns_rerouted", 0) + 1
                except Exception as e:  # noqa: BLE001
                    log.debug("reroute %s: %s", vid, e)
        inc.rerouted = done

    def _detour_home(self, vid: str, starts: list, home: str, avoid: str):
        """(cost, edges): from one of `starts` to a road 300-900 m away not through `avoid`, then to `home`; None."""
        c = self.conn
        try:
            vt = c.vehicle.getTypeID(vid)
        except Exception:  # noqa: BLE001
            return None
        if not hasattr(self, "_far_pool"):
            self._far_rng = random.Random(len(self.net.getEdges()))  # its own: the accidents' draws stay as they were
            self._far_pool = [e.getID() for e in self.net.getEdges() if e.getFunction() != "internal"
                              and e.getLength() > 30.0 and any(ln.allows("passenger") for ln in e.getLanes())]
        for _ in range(12):
            far = self._far_rng.choice(self._far_pool)
            if far in (home, avoid) or far in starts:
                continue
            for alt in starts:
                try:
                    r1 = self._route(alt, far, vt, vid)
                    r2 = self._route(far, home, vt, vid)
                except Exception:  # noqa: BLE001
                    continue
                if not r1.edges or not r2.edges or avoid in r1.edges or not 300.0 <= r1.length <= 900.0:
                    continue
                return r1.travelTime + r2.travelTime, list(r1.edges) + list(r2.edges[1:])
        return None

    def _onwards_from(self, lane: str, vtype: str) -> list:
        """A route from `lane`'s road to a far sink that leaves it by one of `lane`'s own connections (no lane change
        needed after a stop on it)."""
        ln = self.net.getLane(lane)
        nexts = [k.getTo().getID() for k in ln.getOutgoing()]
        route = self._onwards(ln.getEdge().getID(), vtype)
        if len(route) < 2 or route[1] in nexts or not nexts:
            return route
        for nxt in nexts:
            for s_ in self.rng.sample(self.sinks, min(10, len(self.sinks))):
                try:
                    r = self._route(nxt, s_, vtype)
                except Exception:  # noqa: BLE001
                    continue
                if r.edges and r.length > 200:
                    return [ln.getEdge().getID()] + list(r.edges)
        return route

    def _skip_stops(self, inc: Incident) -> None:
        """Line buses skip a stop on a closed lane (they would drive into the cones to reach it)."""
        closed = {cl for cl, _ in inc.closed}
        if not closed:
            return
        c = self.conn
        for vid in c.vehicle.getIDList():
            if not vid.startswith("pt_"):
                continue
            for _ in range(4):
                try:
                    stops = c.vehicle.getStops(vid)
                except Exception:  # noqa: BLE001
                    break
                i = next((k for k, st in enumerate(stops) if st.lane in closed), None)
                if i is None:
                    break
                try:
                    c.vehicle.replaceStop(vid, i, "")
                    inc.log(self.now, f"{vid} skips its stop on {stops[i].lane} (closed)")
                except Exception:  # noqa: BLE001
                    break

    def _damage(self, sev: float, kind: str, w: Wreck, inc: Incident) -> dict:
        """How the wreck lies after the crash (CARLA side): turned, tilted, pushed; more for a harder crash."""
        r = random.Random(zlib.crc32(f"{inc.id}/{w.vid}".encode()))  # the same every run (hash() is salted)
        k = min(sev / 14.0, 1.0)  # 0 .. 1 (50 km/h and more)
        sign = r.choice((-1.0, 1.0))
        d = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0, "dz": 0.0, "dx": 0.0, "rollover": False, "ramp_s": 0.25 + 0.6 * k,
             "severity": round(k, 2)}
        if kind == "rear-end":
            d["yaw"] = sign * r.uniform(1.5, 4.0 + 16.0 * k)
            d["roll"] = r.uniform(-1.0, 1.0) * (0.6 + 2.4 * k)
            # the follower's nose dips (crushed front), the leader is pushed on and its rear sits low
            d["pitch"] = (-1.0 if w.role == "collider" else 0.8) * (0.6 + 2.0 * k)
            d["dx"] = (0.25 + 1.2 * k) if w.role == "victim" else 0.0
        else:  # side / junction / frontal: the struck car spins
            spin = (8.0 + 55.0 * k) if w.role == "victim" else (4.0 + 22.0 * k)
            d["yaw"] = sign * r.uniform(0.6, 1.0) * spin
            d["roll"] = sign * r.uniform(0.5, 1.0) * (1.0 + 4.0 * k)
            d["pitch"] = r.uniform(-1.0, 1.0) * (0.5 + 1.5 * k)
            d["dx"] = r.uniform(0.2, 0.6 + 1.5 * k)
            if w.role == "victim" and sev > 13.0 and (w.model or "").split(".")[1:2] != ["carlamotors"] and r.random() < 0.35:
                d["rollover"] = True  # over on its side
        d["dz"] = -0.02 - 0.05 * k     # sits lower: a broken suspension
        return d

    # ------------------------------------------------------------------ the response
    def _divert(self, inc: Incident) -> None:
        """Drivers heading for the incident's road, still two or more roads away, take another way (as a navigation app
        would tell them): routing sees the road as slow (10 min) until it reopens. Without this, the queue at the
        evening rush filled the junctions behind and locked the town (tools/check_accidents.py)."""
        c = self.conn
        edges = {w.lane.rsplit("_", 1)[0] for w in inc.wrecks if w.lane and not w.removed}
        edges |= {self._via.get(w.lane, (None, None))[1].rsplit("_", 1)[0] for w in inc.wrecks
                  if w.lane.startswith(":") and self._via.get(w.lane, (None, None))[1]}
        if not edges:
            return
        for e in edges:
            try:
                c.edge.adjustTraveltime(e, 600.0)
            except Exception:  # noqa: BLE001
                pass
        inc.diverted = sorted(edges)
        n = 0
        for vid in c.vehicle.getIDList():
            if vid in self.role or vid.startswith("pt_"):
                continue
            try:
                route = c.vehicle.getRoute(vid)
                i = c.vehicle.getRouteIndex(vid)
            except Exception:  # noqa: BLE001
                continue
            ahead = route[i + 1:i + 8]
            if any(e in edges for e in ahead):
                try:
                    c.vehicle.rerouteTraveltime(vid, currentTravelTimes=False)
                    n += 1
                except Exception:  # noqa: BLE001
                    pass
                # one already standing at the end of the road before may move sideways to its new lane (see _squeeze)
                if route[i + 1:i + 2] and route[i + 1] in edges and vid not in self.squeezed:
                    try:
                        if c.vehicle.getSpeed(vid) < 0.3:
                            self._pull_out(vid)
                    except Exception:  # noqa: BLE001
                        pass
        inc.diverted_n = getattr(inc, "diverted_n", 0) + n

    def _undivert(self, inc: Incident) -> None:
        for e in getattr(inc, "diverted", []):
            try:
                ed = self.net.getEdge(e)
                self.conn.edge.adjustTraveltime(e, ed.getLength() / max(ed.getSpeed(), 1.0))
            except Exception:  # noqa: BLE001
                pass
        inc.diverted = []

    def _squeeze(self, inc: Incident, now: float, vehicles: dict) -> None:
        """Drivers standing behind the wrecks (or the police car, the closed lane's end, a truck) pull out as a driver
        does: every vehicle type has lcMaxSpeedLatStanding=0 (no sideways slide while standing, cosim/vmodels.py), so
        a car stopped right behind a wreck could never get out, nor anyone behind it (measured: the town locked). It may
        now close up to 1 m behind it and steer out as it rolls (sideways as fast as forwards), with 60 cm to spare
        beside it: it moves over while creeping forwards, as a pulling-out car does, never sliding sideways at a
        standstill (as a first version let them: 0.6 m/s, which looked like a glitch)."""
        c = self.conn
        blockers = [w.vid for w in inc.wrecks if not w.removed] + ([inc.police] if inc.police else []) + \
                   [t.vid for t in inc.tows if t.state in ("there", "backing", "loading", "loaded")] + \
                   ([inc.blocker] if getattr(inc, "blocker", None) else [])
        for b in blockers:
            if b not in vehicles:
                continue
            try:
                lane = c.vehicle.getLaneID(b)
                pos = c.vehicle.getLanePosition(b) - c.vehicle.getLength(b)
            except Exception:  # noqa: BLE001
                continue
            for o in c.lane.getLastStepVehicleIDs(lane):
                if o in self.squeezed and o not in self.role:
                    self._still_stuck(inc, o, now, vehicles)
                    continue
                if o in self.role:
                    continue
                v = vehicles.get(o)
                try:
                    p = c.vehicle.getLanePosition(o)
                except Exception:  # noqa: BLE001
                    continue
                if v is not None and v.speed < 0.3 and pos - 25.0 < p < pos + 0.5:
                    self._pull_out(o)

    PULL_GAP_M = 1.0      # while pulling out: the gap it closes up to (the usual is about 2.5 m)
    PULL_GAP_LAT_M = 0.6  # and beside it (the usual is 1 m)

    def _pull_out(self, o: str) -> None:
        """Lets a car standing behind a blocked lane steer out as it rolls (_squeeze). Only until it is on its way:
        it gets its own gaps back 25 m further on or on its next road (_unsqueeze). It kept 30 cm gaps for the rest
        of its trip before 24 Oct: cars bunched up bumper to bumper past every accident and grazed the cones."""
        c = self.conn

        def param(k: str, default: str) -> str:
            try:
                return c.vehicle.getParameter(o, f"laneChangeModel.{k}") or default
            except Exception:  # noqa: BLE001
                return default

        try:
            own = (c.vehicle.getMinGap(o), c.vehicle.getMinGapLat(o), param("lcMaxSpeedLatFactor", "0.3"),
                   param("lcAssertive", "1"))
            self.squeezed[o] = (c.vehicle.getRoadID(o), c.vehicle.getLanePosition(o), self.now, own)
            c.vehicle.setMinGap(o, min(own[0], self.PULL_GAP_M))
            c.vehicle.setMinGapLat(o, min(own[1], self.PULL_GAP_LAT_M))
            c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatFactor", "1.0")
            c.vehicle.setParameter(o, "laneChangeModel.lcAssertive", "3")
        except Exception as e:  # noqa: BLE001
            log.debug("pull out %s: %s", o, e)

    STUCK_PULLING_S = 15.0  # a car still standing this long after it began to pull out may edge sideways

    def _still_stuck(self, inc: Incident, o: str, now: float, vehicles: dict) -> None:
        """A car that began to pull out (_pull_out) and still stands STUCK_PULLING_S later, often half in the open
        lane, its nose at the car ahead: it edges over at 0.3 m/s, as a driver would at a crawl, into the open lane.
        Left alone it stood across both lanes for good, and everything behind it on the open lane too (seed 7,
        watched: before 24 Oct only taking it off the road out of sight undid it)."""
        v = vehicles.get(o)
        rec = self.squeezed.get(o)
        if v is None or rec is None or v.speed > 0.3 or now - rec[2] < self.STUCK_PULLING_S or len(rec) > 4:
            return
        c = self.conn
        try:
            c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
            if inc.open_lane and inc.open_lane.rsplit("_", 1)[0] == c.vehicle.getRoadID(o):
                c.vehicle.changeLane(o, int(inc.open_lane.rsplit("_", 1)[1]), 30.0)
            self.squeezed[o] = rec + (True,)
            self.stats["edged_over"] = self.stats.get("edged_over", 0) + 1
        except Exception as e:  # noqa: BLE001
            log.debug("edge over %s: %s", o, e)

    def _unsqueeze(self, now: float, vehicles: dict) -> None:
        c = self.conn
        for o, rec in list(self.squeezed.items()):
            road, pos, t, own = rec[:4]
            if o not in vehicles:
                self.squeezed.pop(o, None)
                continue
            try:
                r, p = c.vehicle.getRoadID(o), c.vehicle.getLanePosition(o)
            except Exception:  # noqa: BLE001
                self.squeezed.pop(o, None)
                continue
            if (r == road and p - pos < 25.0) and now - t < 90.0:
                continue
            try:
                c.vehicle.setMinGap(o, own[0])
                c.vehicle.setMinGapLat(o, own[1])
                c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatFactor", own[2] or "0.3")
                c.vehicle.setParameter(o, "laneChangeModel.lcAssertive", own[3] or "1")
                if len(rec) > 4:
                    c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatStanding", "0")
            except Exception as e:  # noqa: BLE001
                log.debug("unsqueeze %s: %s", o, e)
            self.squeezed.pop(o, None)

    def _relieve(self, inc: Incident, now: float, vehicles: dict) -> None:
        """A queue behind an incident that does not move, out of sight, dissolves: those drivers turned off somewhere
        (they are taken off the road; the day's curve brings others elsewhere). Without it the queue backed up across
        junctions and locked the town at the evening rush, no way through for the police either (SUMO alone,
        tools/check_accidents.py). In sight the queue stays as it is."""
        c = self.conn
        edges = set(getattr(inc, "diverted", []))
        feeders = {f.getID() for e in edges for f in self.net.getEdge(e).getIncoming()}
        for vid, v in vehicles.items():
            if vid in self.role or vid.startswith("pt_") or self.protected(vid):
                self.standing.pop(vid, None)
                continue
            if v.speed > 0.3:
                self.standing.pop(vid, None)
                continue
            t0 = self.standing.setdefault(vid, now)
            if now - t0 < 25.0:
                continue
            try:
                road = c.vehicle.getRoadID(vid)
            except Exception:  # noqa: BLE001
                continue
            if road not in edges and road not in feeders:
                continue
            if self.visible is not None and self.visible(v.x, v.y):
                continue
            try:
                self._remove(vid)
                self.stats["relieved"] = self.stats.get("relieved", 0) + 1
                self.standing.pop(vid, None)
            except Exception:  # noqa: BLE001
                pass

    def _advance(self, inc: Incident, now: float, vehicles: dict) -> None:
        c = self.conn
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 10 == 5:
            self._relieve(inc, now, vehicles)
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 10 == 0:  # every 5 s
            self._divert(inc)
        if inc.phase != "closed" and int(now * 2) % 4 == 0:  # every 2 s
            self._squeeze(inc, now, vehicles)
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 2 == 0:  # every second
            self._merge(inc)
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 4 == 3:  # every 2 s
            self._reroute_turns(inc)
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 4 == 1:  # every 2 s
            self._recentre(inc, now, vehicles)
        if inc.phase not in ("clearing", "closed") and int(now * 2) % 20 == 7:  # every 10 s: buses come and go
            self._skip_stops(inc)
        self._hold(inc, now)
        for w in inc.wrecks:
            if w.exit is not None and not w.exit["out"] and now >= w.exit["t_out"]:
                w.exit["out"] = True
        for w in inc.wrecks:  # stays frozen (SUMO may reset a speed on some events)
            if not w.removed and w.vid in vehicles and vehicles[w.vid].speed > 0.05:
                self._freeze(w.vid)
        if inc.police_due is None:  # (an incident made elsewhere: a breakdown, or one from before 3 Nov)
            inc.police_due = inc.t0 + police_delay(self.police_rng, self.police_delay_s, self.police_spread)
            inc.police_record = {"due": round(inc.police_due, 1), "sent": None, "on_scene": None, "how": None,
                                 "cars": 0}
        if inc.phase == "crash" and now >= inc.police_due:
            if self._send_police(inc, late=now - inc.police_due > 24.0):
                inc.phase = "police en route"
                rec = inc.police_record
                rec["cars"] = rec.get("cars", 0) + 1
                if rec.get("sent") is None:
                    rec["sent"] = round(now, 1)
                    inc.log(now, f"police sent ({now - inc.t0:.0f} s after the crash)")
            elif now - inc.police_due > 84.0:
                inc.phase = "secured"  # no way in for a police car: the scene is handled on foot
                inc.police_t = now
                inc.police_record.update(on_scene=round(now, 1), how="on foot (no way in for a car)")
                inc.log(now, "no police car could get there: secured on foot")
        elif inc.phase == "police en route":
            pid = inc.police
            v = vehicles.get(pid)
            arrived = False
            if v is None and pid in c.simulation.getPendingVehicles():
                inc.police_stuck_s += 0.5
                if inc.police_stuck_s <= 20.0:
                    return
                try:
                    self._remove(pid)
                except Exception:  # noqa: BLE001
                    pass
            if v is None and pid not in c.vehicle.getIDList():
                inc.log(now, "police car lost; sending another")
                inc.phase = "crash"
                inc.police_due = now  # another is sent at once (before 3 Nov: t0 moved, which shifted the events' times)
                return
            if v is not None:
                try:
                    arrived = c.vehicle.isStopped(pid)
                except Exception:  # noqa: BLE001
                    arrived = False
                inc.police_stuck_s = inc.police_stuck_s + 0.5 if v.speed < 0.3 else 0.0
                if not arrived and int(now * 2) % 4 == 0:
                    self._steer(pid)
                late_ = now - getattr(inc, "police_sent_t", now) > EN_ROUTE_MAX_S
                if not arrived and late_ and not self._near(v, inc, 60.0) and \
                        (self.visible is None or not self.visible(v.x, v.y) or
                         now - getattr(inc, "police_sent_t", now) > 2 * EN_ROUTE_MAX_S):
                    # far too long on its way (round and round a knot, or a lane it cannot leave): another car is
                    # put on the road near the scene, this one goes back (or on its way)
                    try:
                        c.vehicle.replaceStop(pid, 0, "")
                    except Exception:  # noqa: BLE001
                        pass
                    self._go(pid)
                    self._home(pid, "police", POLICE_PARKED)
                    self.role.pop(pid, None)
                    inc.police, inc.no_depot = None, True
                    inc.phase = "crash"
                    inc.police_due = now
                    inc.log(now, f"{pid} far too long on its way: another car comes")
                    return
                if not arrived and self.depots is not None and self.depots.owns(pid) and \
                        inc.police_stuck_s > DEPOT_STUCK_S and not self._near(v, inc, 120.0) and \
                        (self.visible is None or not self.visible(v.x, v.y) or
                         inc.police_stuck_s > 2 * DEPOT_STUCK_S):
                    # held up on its way from the station: it goes back, one put on the road comes instead
                    try:
                        c.vehicle.replaceStop(pid, 0, "")
                    except Exception:  # noqa: BLE001
                        pass
                    self._home(pid, "police", POLICE_PARKED)
                    self.role.pop(pid, None)
                    inc.police, inc.no_depot = None, True
                    inc.phase = "crash"
                    inc.police_due = now
                    inc.log(now, f"{pid} held up on its way: back to the station, another car comes")
                    return
                try:
                    p_lane = c.vehicle.getLaneID(pid)
                except Exception:  # noqa: BLE001
                    p_lane = ""
                # it may stop short only where it blocks nobody: on a closed lane (or the lane its stop is on). Stopped
                # on the open lane beside the wrecks it shut the road: everything behind stood for good (seed 7, watched)
                closed = {cl for cl, _ in inc.closed}
                lane0, _, _ = self._anchor(inc)
                # (a crash in a junction: no lane is kept free, the queue it stands in stands anyway; never inside a
                # junction: a truck backed into it there, seed 26)
                ok_here = p_lane in closed or p_lane == lane0 or (not inc.closed and not p_lane.startswith(":"))
                if not arrived and ok_here and ((inc.police_stuck_s > 30.0 and self._near(v, inc, 35.0))
                                                or (inc.police_stuck_s > 120.0 and self._near(v, inc, 120.0))):
                    self._stop_here(pid)  # stuck in the queue close by: stops there, the officer walks
                    arrived = True
                    inc.police_record["how"] = "car stopped short in traffic, officer on foot"
                    inc.log(now, "police car stuck in traffic close by: stops there")
                elif not arrived and not ok_here and \
                        ((inc.police_stuck_s > ON_FOOT_AFTER_S and self._near(v, inc, ON_FOOT_NEAR_M))
                         or (inc.police_stuck_s > 120.0 and self._near(v, inc, 120.0))):
                    # (3 Nov) stuck in a queue close by, on a road where it may not stand (the road into the scene's,
                    # across the junction): the officer gets out and walks; the car keeps its place in the queue and
                    # comes on to its stop when the traffic moves. Before, it waited there for good: in the CARLA run
                    # of 3 Nov and tools/check_incident_scene.py --share 0.5 --watch 150 (seed 11) the police never got
                    # to two scenes in 20 minutes (one car 40 m from the scene, behind a queue that did not move)
                    arrived = True
                    inc.police_record["how"] = "officer on foot from the queue (the car comes on later)"
                    inc.log(now, f"police car {pid} stuck in the queue on {p_lane}: the officer walks")
            if arrived:
                inc.phase = "secured"
                inc.police_t = now
                inc.police_record["on_scene"] = round(now, 1)
                inc.police_record["how"] = inc.police_record.get("how") or "car at the scene"
                try:
                    inc.police_at = (c.vehicle.getLaneID(pid), c.vehicle.getLanePosition(pid))
                except Exception:  # noqa: BLE001
                    inc.police_at = None
                self._block(inc, now)
                inc.log(now, "police on scene" + (f"; {len(inc.closed)} lane(s) closed, traffic on {inc.open_lane}"
                                                   if inc.closed else ""))
        elif inc.phase in ("secured", "recovery"):
            if inc.closed and getattr(inc, "blocker", None) is None and int(now * 2) % 4 == 1:
                self._block(inc, now)  # (no room for it a moment ago)
            self._inert(inc)
            self._recovery(inc, now, vehicles)
        elif inc.phase == "clearing":
            # the last truck loaded its wreck as the phase turned to clearing: it still has to drive off (before 24 Oct
            # only the secured/recovery phases ran _recovery, and the last truck stood, loaded, until something else
            # moved it)
            self._recovery(inc, now, vehicles)
            if now - inc.clear_t >= CLEAR_S:
                self._open_lanes(inc)
                self._undivert(inc)
                inc.log(now, f"{getattr(inc, 'diverted_n', 0)} drivers took another way meanwhile")
                if inc.police and inc.police in vehicles:
                    self._go(inc.police)
                    self._home(inc.police, "police", POLICE_PARKED)
                for t in inc.tows:  # (none should be left: one standing would block the reopened lane)
                    if t.state in ("there", "backing", "loading", "loaded") and t.vid in vehicles:
                        self._go(t.vid)
                        self._home(t.vid, "tow")
                        t.state = "leaving"
                inc.phase = "closed"
                inc.log(now, "lane reopened; police leaves")

    def _near(self, v, inc: Incident, m: float) -> bool:
        return math.hypot(v.x - inc.x, v.y - inc.y) <= m

    def _stop_here(self, vid: str) -> None:
        """Stuck in traffic close to the scene: it stays where it is (its planned stop dropped)."""
        c = self.conn
        try:
            c.vehicle.replaceStop(vid, 0, "")
        except Exception:  # noqa: BLE001
            pass
        try:
            c.vehicle.setSpeedMode(vid, 31)
            c.vehicle.setSpeed(vid, 0.0)
            c.vehicle.setLaneChangeMode(vid, 0)  # it stays in its lane (its stop dropped, it drifted into the open one)
        except Exception:  # noqa: BLE001
            pass

    def _go(self, vid: str) -> None:
        """Drives on: from its stop, or from where it was held."""
        c = self.conn
        try:
            if c.vehicle.isStopped(vid):
                c.vehicle.resume(vid)
        except Exception:  # noqa: BLE001
            pass
        try:
            c.vehicle.setSpeed(vid, -1)
            c.vehicle.setLaneChangeMode(vid, 0b011001010101)
        except Exception:  # noqa: BLE001
            pass

    def _remove(self, vid: str) -> None:
        c = self.conn
        try:
            c.vehicle.unsubscribe(vid)  # a removed vehicle still subscribed: "Error! Vehicle ... is not known"
        except Exception:  # noqa: BLE001
            pass
        c.vehicle.remove(vid)

    def _anchor(self, inc: Incident) -> tuple[str, float, float]:
        """(lane, rear of the rear-most wreck, front of the front-most) on a normal lane, for stops; a wreck in a
        junction counts on the lane leading into it."""
        live = [w for w in inc.wrecks if not w.removed and w.lane]
        lane = next((w.lane for w in live if not w.lane.startswith(":")), None)
        if lane is None:
            via = next((w.lane for w in live if w.lane.startswith(":")), "")
            frm = self._via.get(via, (None, None))[0]
            if frm is None:
                return "", 0.0, 0.0
            ln = self.net.getLane(frm)
            return frm, ln.getLength() - 1.0, ln.getLength()
        on = [w for w in live if w.lane == lane]
        rear = min(w.pos - w.length for w in on)
        front = max(w.pos for w in on)
        return lane, rear, front

    def _route_from_behind(self, to_edge: str, lo: float = 250.0, hi: float = 700.0):
        """A route to `to_edge` from an edge lo-hi m before it (along the roads), its start out of sight if possible."""
        start = self.net.getEdge(to_edge)
        frontier = [(0.0, start)]
        seen = {to_edge: 0.0}
        cands = []
        while frontier:
            frontier.sort(key=lambda t: t[0])
            d, e = frontier.pop(0)
            for inc_e in e.getIncoming():
                if inc_e.getFunction() == "internal" or not inc_e.allows("passenger"):
                    continue
                nd = d + inc_e.getLength()
                if inc_e.getID() in seen or nd > hi:
                    continue
                seen[inc_e.getID()] = nd
                frontier.append((nd, inc_e))
                if nd >= lo:
                    cands.append(inc_e.getID())
        if not cands:
            cands = [k for k in seen if k != to_edge]
        self.rng.shuffle(cands)
        if self.visible is not None:
            cands.sort(key=lambda k: 1 if self.visible(*self._xy(self.net.getEdge(k).getLanes()[0].getID(), 0.0)) else 0)
        for s in cands[:12]:
            try:
                r = self._route(s, to_edge, POLICE_TYPE)
            except Exception:  # noqa: BLE001
                continue
            if r.edges and len(r.edges) >= 2:
                return list(r.edges)
        return None

    def _onwards(self, edge: str, vtype: str) -> list:
        """A route from `edge` to a far sink (the car leaves the scene and the town)."""
        sinks = list(self.sinks)
        self.rng.shuffle(sinks)
        for s in sinks[:15]:
            if s == edge:
                continue
            try:
                r = self._route(edge, s, vtype)
            except Exception:  # noqa: BLE001
                continue
            if r.edges and r.length > 300:
                return list(r.edges)
        return [edge]

    def _route(self, frm: str, to: str, vtype: str, vid: str | None = None):
        """simulation.findRoute with a type that lives on: a vehicle's own copy of its type (SUMO makes one, "type@vid",
        once its min gap or the like is set) is deleted with the vehicle, and SUMO 1.27 crashed some steps after a route
        had been looked up with one (seeds 11, 23, 31, 53)."""
        if vid and vtype.endswith("@" + vid):
            vtype = vtype[: -len(vid) - 1]
        return self.conn.simulation.findRoute(frm, to, vType=vtype)

    def _tow_type(self, vid: str) -> str:
        """A recovery truck's type: the HGV sized to its CARLA model, of the emergency class (it may use a closed
        lane)."""
        from cosim import towing

        c = self.conn
        tid = self.typer.type_for(vid, TRUCK_BASE, towing.TRUCK_MODEL) if self.typer is not None else TRUCK_BASE
        tow_t = tid + TOW_SUFFIX
        if tow_t not in c.vehicletype.getIDList():
            c.vehicletype.copy(tid, tow_t)
            c.vehicletype.setVehicleClass(tow_t, "emergency")
            c.vehicletype.setMaxSpeed(tow_t, 16.0)
        return tow_t

    # ------------------------------------------------------------------ the depots (cosim/depots.py)
    def stock_depots(self, depots) -> None:
        """The police station's cars and the recovery yard's trucks, parked there; from now on the accidents' police
        cars and trucks come from there (when one is free) and go back after the job."""
        self.depots = depots
        # no accident on a depot's road: its parking lane is the lane beside the other lanes, closed to the town, and
        # with one of those shut there was no way past (an accident on the hospital's road 49, seed 53: the trucks
        # stood behind the parked ambulances for good)
        self.depot_edges = {d.plan.edge for d in depots.depots.values()}
        # nor in the junctions at their ends (979, between the police station and the hospital: a crash there stood
        # in the way of the station's car pulling out, and a truck backed into it, seed 26 watched)
        self.depot_junctions = set()
        for e in self.depot_edges:
            try:
                ed = self.net.getEdge(e)
                self.depot_junctions |= {ed.getFromNode().getID(), ed.getToNode().getID()}
            except Exception:  # noqa: BLE001
                pass
        for name, prefix in (("police", "police_st"), ("tow", "tow_yard")):
            d = depots.depots.get(name)
            for k in range(len(d.slots) if d is not None else 0):
                vid = f"{prefix}{k + 1}"
                vt = self._type(vid, POLICE_PARKED) if name == "police" else self._tow_type(vid)
                depots.add_parked(name, vid, vt)

    def _from_depot(self, name: str, edge: str, onwards: list, run_type: str | None = None, lane: str | None = None):
        """A vehicle of depot `name` sent to `edge` and on along `onwards` (which starts at `edge`), on a run as
        `run_type` (a base type, sized to its model here); None when none is free there or there is no way.
        lane: it must come onto that lane of `edge` straight from the road before (the open lane beside an
        accident: from some roads the only way into the accident's road is its closed lane, and a truck sent that
        way stood behind the cone line's stand-in for good, seed 11)."""
        dep = self.depots
        if dep is None or name not in dep.depots or getattr(self, "_no_depot_now", False) or not self.depot_dispatch:
            return None
        blocked = self.blocked_roads()
        d_edge = dep.depots[name].plan.edge
        if d_edge in blocked or self._jammed(d_edge):
            return None  # an accident at the depot, or its road standing: it could not get out (seed 11)
        vid = dep.take(name, ok=self.tow_free if name == "tow" else None)
        if vid is None:
            return None
        c = self.conn
        try:
            vt = self._type(vid, run_type) if run_type else c.vehicle.getTypeID(vid)
        except Exception:  # noqa: BLE001
            dep.put_back(vid)
            return None
        r, tail = dep.route_from(name, edge, vt), []
        if lane is not None:
            best = None
            for fe in self._feeders(lane):
                rf = dep.route_from(name, fe, vt)
                if rf is not None and edge not in rf.edges and (best is None or rf.length < best.length):
                    best = rf
            if best is None:  # no road leads onto that lane (only its own closed neighbour): put on it, as before
                dep.put_back(vid)
                return None
            r, tail = best, [edge]
        if r is None or r.length > DEPOT_REACH_M or blocked & set(r.edges[:-1]) or \
                any(self._jammed(e) for e in r.edges[1:-1]):
            dep.put_back(vid)  # its way there goes through an accident or a queue: one put on the road comes
            return None
        edges = list(r.edges) + tail + (list(onwards[1:]) if onwards and onwards[0] == edge else [])
        try:
            if run_type:
                c.vehicle.setType(vid, vt)
        except Exception:  # noqa: BLE001
            pass
        if not dep.leave(vid, edges):
            dep.put_back(vid)
            return None
        self.inserted_back[vid] = (round(r.length), f"the {DEPOT_NAMES.get(name, name)}")
        return vid

    def _steer(self, vid: str) -> None:
        """On the road before the one it stops on, a vehicle on a lane that does not lead onto its stop's lane moves
        over to one that does (at a crawl if it stands: every type has no sideways move standing). A recovery truck
        from the yard took the lane SUMO chose, the only one into the accident's closed lane, and stood at its end
        for good (seed 11); the trucks put on the road were put on the right lane."""
        c = self.conn
        try:
            stops = c.vehicle.getStops(vid, 1)
            if not stops:
                return
            target = stops[0].lane
            lane = c.vehicle.getLaneID(vid)
            if not lane or lane.startswith(":") or lane.rsplit("_", 1)[0] == target.rsplit("_", 1)[0]:
                return
            route = c.vehicle.getRoute(vid)
            i = c.vehicle.getRouteIndex(vid)
            if i + 1 >= len(route) or route[i + 1] != target.rsplit("_", 1)[0]:
                return
            ln = self.net.getLane(lane)
            good = [x.getIndex() for x in ln.getEdge().getLanes()
                    if any(k.getToLane().getID() == target for k in x.getOutgoing())]
            if not good or ln.getIndex() in good:
                return
            k = min(good, key=lambda g: abs(g - ln.getIndex()))
            if vid not in self.squeezed:
                self._pull_out(vid)
            c.vehicle.setParameter(vid, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
            rec = self.squeezed.get(vid)
            if rec is not None and len(rec) <= 4:
                self.squeezed[vid] = rec + (True,)
            c.vehicle.changeLane(vid, k, 10.0)
            self.stats["steered"] = self.stats.get("steered", 0) + 1
        except Exception as e:  # noqa: BLE001
            log.debug("steer %s: %s", vid, e)

    def _jammed(self, edge: str) -> bool:
        """Most of a road's lanes for cars standing full (more than JAMMED_SHARE of their length)."""
        c = self.conn
        try:
            lanes = [ln for ln in self.net.getEdge(edge).getLanes() if ln.allows("passenger")]
            full = 0
            for ln in lanes:
                n = c.lane.getLastStepHaltingNumber(ln.getID())
                if n * 7.5 > JAMMED_SHARE * ln.getLength():
                    full += 1
            return bool(lanes) and full * 2 > len(lanes)
        except Exception:  # noqa: BLE001
            return False

    def _junction_lanes(self, junction: str) -> list:
        """The internal lanes of a junction (its ways through it)."""
        cache = self.__dict__.setdefault("_jl_cache", {})
        if junction not in cache:
            cache[junction] = sorted(v for v in self._via if v[1:].rsplit("_", 2)[0] == junction)
        return cache[junction]

    def _feeders(self, lane: str) -> list:
        """The roads with a lane that leads straight onto `lane`."""
        ln = self.net.getLane(lane)
        e = ln.getEdge()
        out = []
        for frm in e.getIncoming():
            if frm.getFunction() == "internal":
                continue
            if any(k.getToLane() == ln for k in frm.getOutgoing().get(e, [])):
                out.append(frm.getID())
        return out

    def _home(self, vid: str, depot: str, parked_type: str | None = None) -> None:
        """After the job: back to its depot (beacons off), or, one that started on the road, to its depot if there
        is room (it stays there), else on its way as before."""
        dep = self.depots
        if dep is None:
            return
        if not dep.owns(vid):
            dep.adopt(vid, depot)
        try:
            if parked_type:
                self.conn.vehicle.setType(vid, self._type(vid, parked_type))
        except Exception:  # noqa: BLE001
            pass
        if not dep.send_home(vid):
            dep.disown(vid)

    def _type(self, vid: str, base: str) -> str:
        return self.typer.type_for(vid, base) if self.typer is not None else base

    INSERT_FAR_M = 700.0  # how far back a police car or a recovery truck may start when the near roads are in sight

    def _insert(self, vid: str, tid: str, onwards: list, lane: str, at: float, near: float, far: float,
                late: bool, extend: bool = True) -> bool:
        """Puts `vid` on the road `near`-`far` m before position `at` of `lane` (whose edge starts `onwards`), on that
        road or the roads leading onto it, where there is room, out of sight (`visible`, SUMO x, y), the nearest
        first. When every spot that close is in sight (you are watching the scene), it starts further back, out of
        sight, up to INSERT_FAR_M along the roads leading there, and drives in: before 24 Oct it waited until you
        looked away (or 30-45 s, then appeared in sight), so nothing came while you watched. `late`: in sight too,
        if nowhere else. It then drives up to its stop."""
        c = self.conn
        target = self.net.getLane(lane)
        vlen = c.vehicletype.getLength(tid)
        spots = []  # (lane id, pos, metres before the target, edges from that lane's road to the target's, exclusive)
        reach = max(far, self.INSERT_FAR_M) if extend else far

        def scan(ln, end_pos, back0, path):
            occ = []
            for o in c.lane.getLastStepVehicleIDs(ln.getID()):
                try:
                    p = c.vehicle.getLanePosition(o)
                    occ.append((p - c.vehicle.getLength(o) - 4.0, p + vlen + 4.0))
                except Exception:  # noqa: BLE001
                    pass
            p = min(end_pos, ln.getLength() - 0.5)
            while p >= max(vlen + 0.5, 0.5):
                back = back0 + (end_pos - p)
                if back > reach:
                    break
                if back >= near and all(not (o0 <= p <= o1) for o0, o1 in occ):
                    spots.append((ln.getID(), p, back, path))
                p -= 6.0 if back < far else 20.0

        scan(target, at, 0.0, [])
        frontier = [(at, target, [])]
        seen = {target.getID()}
        while frontier:
            back0, ln, path = frontier.pop(0)
            if back0 > reach:
                continue
            e = ln.getEdge()
            for frm in e.getIncoming():  # the lanes that lead straight onto this one
                if frm.getFunction() == "internal" or not frm.allows("passenger"):
                    continue
                for fl in {k.getFromLane() for k in frm.getOutgoing().get(e, []) if k.getToLane() == ln}:
                    if fl.getID() in seen or not fl.allows("passenger"):
                        continue
                    seen.add(fl.getID())
                    sub = [e.getID()] + path
                    scan(fl, fl.getLength(), back0, sub)
                    frontier.append((back0 + fl.getLength(), fl, sub))
        if not spots:
            return False
        scored = []
        for ln_id, pos, back, path in spots:
            x, y = self._xy(ln_id, pos)
            vis = self.visible(x, y) if self.visible is not None else False
            scored.append((vis, back > far, back, ln_id, pos, path))
        scored.sort(key=lambda t: t[:3])
        vis, _, back, ln_id, pos, path = scored[0]
        if vis and not late:
            return False
        first = ln_id.rsplit("_", 1)[0]
        lead = [first] + [x for x in path if x != onwards[0]]
        route = onwards if first == onwards[0] else lead + onwards
        rid = f"{vid}_r"
        try:
            if rid in c.route.getIDList():
                rid = f"{vid}_r{len(c.route.getIDList())}"
            c.route.add(rid, route)
            c.vehicle.add(vid, rid, typeID=tid, depart="now", departLane=ln_id.rsplit("_", 1)[1],
                          departPos=f"{pos:.1f}", departSpeed="0")
        except Exception as e:  # noqa: BLE001
            log.info("%s not inserted on %s: %s", vid, ln_id, e)
            return False
        if back > far:
            log.info("%s starts %.0f m back on %s (the near roads are in sight)", vid, back, ln_id)
        self.inserted_back[vid] = (round(back), ln_id)
        return True

    def _feeder(self, lane: str):
        """The lane that leads straight onto `lane` (the longest), or None."""
        target = self.net.getLane(lane)
        best = None
        for frm in target.getEdge().getIncoming():
            if frm.getFunction() == "internal":
                continue
            for k in frm.getOutgoing().get(target.getEdge(), []):
                if k.getToLane() == target and (best is None or k.getFromLane().getLength() > best.getLength()):
                    best = k.getFromLane()
        return best

    def _xy(self, lane: str, pos: float) -> tuple[float, float]:
        from sumolib.geomhelper import positionAtShapeOffset

        ln = self.net.getLane(lane)
        shape = ln.getShape()
        k = pos * (sum(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 for (ax, ay), (bx, by) in zip(shape, shape[1:]))
                   / max(ln.getLength(), 0.01))
        return positionAtShapeOffset(shape, k)

    def _send_police(self, inc: Incident, late: bool = False) -> bool:
        """The police car comes up the closed lane behind the wrecks (the corridor the closure has emptied; the open
        lane is queuing at the rush hour): it is put on it 30-110 m behind them, out of sight if possible, and
        stops 8 m behind the last wreck."""
        c = self.conn
        lane, rear, _ = self._anchor(inc)
        if not lane:
            return False
        edge = lane.rsplit("_", 1)[0]
        inc.police_n = getattr(inc, "police_n", 0) + 1
        inc.police_stuck_s = 0.0  # a new car: its own wait (the last one's made the next look stuck at once)
        pid = f"inc{inc.id}_police" + ("" if inc.police_n == 1 else f"_{inc.police_n}")
        tid = self._type(pid, POLICE_TYPE)
        plen = c.vehicletype.getLength(tid)
        stop = rear - 8.0
        if stop < plen + 0.5:  # the wrecks are at the start of the road: the police car waits before the junction
            fl = self._feeder(lane)
            if fl is not None:
                lane, stop, edge = fl.getID(), fl.getLength() - 1.0, fl.getEdge().getID()
            else:
                stop = plen + 0.5
        route = self._onwards(edge, tid)
        self._no_depot_now = getattr(inc, "no_depot", False)
        dep_pid = self._from_depot("police", edge, route, POLICE_TYPE)
        self._no_depot_now = False
        if dep_pid is not None:
            pid = dep_pid
        elif not self._insert(pid, tid, route, lane, stop, 30.0, 140.0, late):
            path = self._route_from_behind(edge) if late else None  # far away, through the traffic
            if not path:
                return False
            try:
                c.route.add(f"{pid}_r", path + route[1:])
                c.vehicle.add(pid, f"{pid}_r", typeID=tid, depart="now", departLane="best", departSpeed="max")
            except Exception as e:  # noqa: BLE001
                log.info("police for incident %d not sent: %s", inc.id, e)
                return False
        try:
            c.vehicle.setStop(pid, edge, pos=stop, laneIndex=int(lane.rsplit("_", 1)[1]), duration=1e5)
        except Exception as e:  # noqa: BLE001
            log.info("police stop for incident %d: %s", inc.id, e)
            if dep_pid is not None:
                self._home(pid, "police", POLICE_PARKED)
                return False
            try:
                self._remove(pid)
            except Exception:  # noqa: BLE001
                pass
            return False
        inc.police = pid
        inc.police_sent_t = self.now
        self.role[pid] = (inc.id, "police")
        b = self.inserted_back.get(pid)
        inc.log(self.now, f"police car {pid} sent" + ((f" from {b[1]}, {b[0]} m away" if b[1].startswith("the ") else
                                                      f" from {b[0]} m back ({b[1]})") if b else ""))
        return True

    def _close_lanes(self, inc: Incident) -> None:
        c = self.conn
        lanes = sorted({w.lane for w in inc.wrecks if not w.removed and w.lane and not w.lane.startswith(":")})
        if not lanes:
            return
        edge = lanes[0].rsplit("_", 1)[0]
        drive = [ln.getID() for ln in self.net.getEdge(edge).getLanes() if ln.allows("passenger")]
        open_ = [ln for ln in drive if ln not in lanes]
        if not open_:
            inc.log(self.now, "every lane blocked: none closed")
            return
        # the open lane next to the closed ones: traffic passes there (the recovery trucks pass there too)
        idx = [int(ln.rsplit("_", 1)[1]) for ln in lanes]
        inc.open_lane = min(open_, key=lambda ln: min(abs(int(ln.rsplit("_", 1)[1]) - i) for i in idx))
        inc.closed = [(ln, None) for ln in lanes]  # closed on the street (cones) in any case
        # nobody changes into a closed lane (only the police and the recovery trucks), and the open lanes go slow
        for ol in open_:
            oi = int(ol.rsplit("_", 1)[1])
            for d in {1 if i > oi else -1 for i in idx if abs(i - oi) == 1}:
                try:
                    before = c.lane.getChangePermissions(ol, d)
                    c.lane.setChangePermissions(ol, ["emergency", "authority"], d)
                    inc.restore.append(("change", ol, (d, before)))
                except Exception as e:  # noqa: BLE001
                    log.debug("change permissions %s: %s", ol, e)
            try:
                before = c.lane.getMaxSpeed(ol)
                c.lane.setMaxSpeed(ol, min(before, PASS_KMH / 3.6))
                inc.restore.append(("speed", ol, before))
            except Exception as e:  # noqa: BLE001
                log.debug("speed %s: %s", ol, e)
        # In SUMO only if every lane that leads onto a closed lane also leads onto an open one: closing lane 23_4,
        # the only lane edge 43 leads onto, left the cars coming from 43 with "no valid route", standing at the
        # junction for good, and the town locked (tools/check_accidents.py). Otherwise drivers pass the wrecks by
        # changing lanes, as they would anyway.
        e = self.net.getEdge(edge)
        safe = True
        for frm in e.getIncoming():
            for conn in frm.getOutgoing().get(e, []):
                if conn.getToLane().getID() in lanes:
                    alts = [k for k in frm.getOutgoing().get(e, []) if k.getFromLane() == conn.getFromLane()
                            and k.getToLane().getID() in open_]
                    safe = safe and bool(alts)
        if not safe:
            inc.log(self.now, f"{', '.join(lanes)} coned off; left open in SUMO (the only way in from a road behind)")
            return
        inc.closed = []
        for ln in lanes:
            try:
                before = list(c.lane.getDisallowed(ln))
                c.lane.setDisallowed(ln, sorted(set(before) | set(CLOSED_TO)))
                inc.closed.append((ln, before))
            except Exception as e:  # noqa: BLE001
                log.debug("close %s: %s", ln, e)

    def _block(self, inc: Incident, now: float) -> None:
        """The closed lane is shut where its cones begin (the taper, cosim/incident_scene.py: BLOCK_BEHIND_M behind the
        wrecks) for whoever still comes up it, by a cone line SUMO knows of: a stopped vehicle never drawn
        (BLOCKER_PREFIX). Without it, in a lane left open in SUMO, drivers drove up to the wrecks and round them where
        the officer stands; shut 6 m behind the wrecks or at the police car (8 m), they drove through the taper's
        cones first (24 Oct)."""
        lane, rear, _ = self._anchor(inc)
        if not lane or not inc.closed or lane.startswith(":"):
            return
        if rear < 3.0:
            # the wrecks reach back to the lane's start (a crash in the junction before it): they shut it
            # themselves; a stand-in there stood where the recovery truck loads, and it waited for good (seed 7)
            inc.blocker = ""
            return
        c = self.conn
        bt = "inc_blocker"
        if bt not in c.vehicletype.getIDList():
            c.vehicletype.copy("DEFAULT_VEHTYPE", bt)
            c.vehicletype.setVehicleClass(bt, "authority")
            c.vehicletype.setLength(bt, 1.0)
            c.vehicletype.setWidth(bt, 2.6)  # the lane's width, nearly: nobody slips by it within the lane (sublanes)
            c.vehicletype.setMinGap(bt, 0.5)
        bid = f"{BLOCKER_PREFIX}{inc.id}"
        pos = block_at(rear)
        try:
            rid = f"{bid}_r"
            if rid not in c.route.getIDList():
                c.route.add(rid, [lane.rsplit("_", 1)[0]])
            c.vehicle.add(bid, rid, typeID=bt, depart="now", departLane=lane.rsplit("_", 1)[1],
                          departPos=f"{pos:.1f}", departSpeed="0")
            c.vehicle.setSpeedMode(bid, 31)
            c.vehicle.setSpeed(bid, 0.0)  # it never moves (a stop where it starts counts as passed)
            c.vehicle.setLaneChangeMode(bid, 0)
            self.role[bid] = (inc.id, "blocker")
            inc.blocker = bid
            inc.log(now, f"closed lane shut {rear - pos:.0f} m behind the wrecks")
        except Exception as e:  # noqa: BLE001
            log.debug("blocker for incident %d: %s", inc.id, e)

    def _inert(self, inc: Incident) -> None:
        """The stand-ins (the cone line's, a truck's: never drawn) want no other lane. Standing on a lane SUMO lets
        nobody leave by (the closed lane, its way on shut), each wanted the open lane beside it: the cars there, as
        cooperative drivers, stopped behind it to let it in, for good (seed 7, watched: a car at the start of the open
        lane for 10 minutes, the recovery truck behind it). Once each is on the road (SUMO's lane-change model)."""
        c = self.conn
        done = inc.inert = getattr(inc, "inert", set())
        ids = ([inc.blocker] if getattr(inc, "blocker", None) else []) + \
              [t.ghost for t in inc.tows if getattr(t, "ghost", None)]
        for vid in ids:
            if vid in done:
                continue
            try:
                c.vehicle.getLateralLanePosition(vid)  # on the road yet?
                for k, v in (("lcStrategic", "-1"), ("lcCooperative", "0"), ("lcSpeedGain", "0"),
                             ("lcKeepRight", "0"), ("lcSublane", "0")):
                    c.vehicle.setParameter(vid, f"laneChangeModel.{k}", v)
                done.add(vid)
            except Exception as e:  # noqa: BLE001
                log.debug("inert %s: %s", vid, e)

    def _open_lanes(self, inc: Incident) -> None:
        c = self.conn
        for t in inc.tows:
            self._drop_ghost(t)
        if getattr(inc, "blocker", None):
            try:
                self._remove(inc.blocker)
            except Exception:  # noqa: BLE001
                pass
            inc.blocker = None
        for kind, ln, before in inc.restore:
            try:
                if kind == "change":
                    c.lane.setChangePermissions(ln, list(before[1]), before[0])
                else:
                    c.lane.setMaxSpeed(ln, before)
            except Exception as e:  # noqa: BLE001
                log.debug("restore %s %s: %s", kind, ln, e)
        inc.restore = []
        for ln, before in inc.closed:
            if before is None:
                continue
            try:
                self.conn.lane.setDisallowed(ln, before)
            except Exception as e:  # noqa: BLE001
                log.debug("open %s: %s", ln, e)
        inc.closed = []

    def _recovery(self, inc: Incident, now: float, vehicles: dict) -> None:
        c = self.conn
        for t in inc.tows:
            self._guard_place(inc, t)
        for t in inc.tows:
            v = vehicles.get(t.vid)
            if t.state == "there" and v is None and t.t_load is None:
                # put on the road where it backs in from (a wreck in a junction), but SUMO has found no room for it
                # there yet: taken back, and put elsewhere in a moment (_send_tow)
                if t.vid in c.simulation.getPendingVehicles():
                    t.stuck_s += 0.5
                    if t.stuck_s > 8.0:
                        try:
                            self._remove(t.vid)
                        except Exception:  # noqa: BLE001
                            pass
                        t.state = "gone"
                        self.role.pop(t.vid, None)
                        w = next(w for w in inc.wrecks if w.vid == t.wreck)
                        w.tow = None
                        inc.direct_tries = getattr(inc, "direct_tries", 0) + 1
                        inc.next_tow_t = now + 2.0
                        inc.log(now, f"no room for {t.vid} there yet: it comes another way")
                    continue
            if t.state == "en route":
                pending = v is None and t.vid in c.simulation.getPendingVehicles()
                if pending:
                    t.stuck_s += 0.5
                    if t.stuck_s > 20.0:  # never got onto the road
                        try:
                            self._remove(t.vid)
                        except Exception:  # noqa: BLE001
                            pass
                if v is None and not (pending and t.stuck_s <= 20.0) and t.vid not in c.vehicle.getIDList():
                    t.state = "gone"
                    w = next(w for w in inc.wrecks if w.vid == t.wreck)
                    w.tow = None
                    inc.log(now, f"recovery truck {t.vid} lost")
                    continue
                stopped = False
                if v is not None and now - getattr(t, "seen_log", -1e9) >= 30.0:
                    t.seen_log = now
                    try:
                        inc.log(now, f"{t.vid} en route: {c.vehicle.getLaneID(t.vid)} at "
                                     f"{c.vehicle.getLanePosition(t.vid):.0f} m, {math.hypot(v.x - inc.x, v.y - inc.y):.0f} m "
                                     f"from the scene, {v.speed:.1f} m/s")
                    except Exception:  # noqa: BLE001
                        pass
                if v is not None:
                    try:
                        stopped = c.vehicle.isStopped(t.vid)
                    except Exception:  # noqa: BLE001
                        stopped = False
                    t.stuck_s = t.stuck_s + 0.5 if v.speed < 0.3 else 0.0
                    # held up in the traffic: it waits there, unless it has passed the wreck already (then it backs
                    # in from where it is; behind the wrecks it would have to drive through them)
                    if not stopped and t.stuck_s > 15.0 and self._near(v, inc, 60.0) and self._past(inc, t):
                        self._stop_here(t.vid)
                        stopped = True
                    if not stopped and int(now * 2) % 4 == 0:
                        self._steer(t.vid)
                    too_long = now - getattr(t, "sent_t", now) > EN_ROUTE_MAX_S + 60.0 and \
                        not self._near(v, inc, 40.0) and (self.visible is None or not self.visible(v.x, v.y) or
                                                        now - getattr(t, "sent_t", now) > 2 * EN_ROUTE_MAX_S)
                    if not stopped and (too_long or (self.depots is not None and self.depots.owns(t.vid) and
                            t.stuck_s > DEPOT_STUCK_S and not self._near(v, inc, 40.0) and \
                            (self.visible is None or not self.visible(v.x, v.y) or t.stuck_s > 2 * DEPOT_STUCK_S))):
                        try:
                            c.vehicle.replaceStop(t.vid, 0, "")
                        except Exception:  # noqa: BLE001
                            pass
                        self._go(t.vid)
                        self._home(t.vid, "tow")
                        self.role.pop(t.vid, None)
                        t.state = "gone"
                        w_ = next((w for w in inc.wrecks if w.vid == t.wreck), None)
                        if w_ is not None:
                            w_.tow = None
                        inc.no_depot = True
                        inc.log(now, f"{t.vid} held up on its way: another truck comes")
                        continue
                    if not stopped:
                        self._pre_hold(inc, t)
                if stopped:
                    t.state, t.t_there = "there", now
                    inc.phase = "recovery"
                    inc.log(now, f"recovery truck {t.vid} ahead of {t.wreck}")
            if t.state == "there" and now - (t.t_there or now) > TOW_GIVE_UP_S:
                self._tow_gives_up(inc, t, now)
                continue
            if t.state in ("there", "backing"):
                self._back_in(inc, t, now)
            elif t.state == "loading" and now - t.t_load >= t.tl["total"]:
                w = next(w for w in inc.wrecks if w.vid == t.wreck)
                try:
                    self._remove(w.vid)
                except Exception:  # noqa: BLE001
                    pass
                w.removed = True
                self.role.pop(w.vid, None)
                t.state = "loaded"
                inc.log(now, f"{t.wreck} loaded onto {t.vid}")
            elif t.state == "loaded" and now - t.t_load >= t.tl["total"] + LEAVE_AFTER_LOAD_S and t.vid in vehicles:
                self._go(t.vid)
                try:
                    c.vehicle.setLaneChangeMode(t.vid, 0b011001010101)
                except Exception:  # noqa: BLE001
                    pass
                self._home(t.vid, "tow")
                t.state = "leaving"
                t.left_t = now
                inc.log(now, f"recovery truck {t.vid} leaves with {t.wreck}")
            elif t.state == "leaving" and v is not None and not getattr(t, "rolling", False):
                if v.speed > 1.0:
                    t.rolling = True
                    inc.log(now, f"{t.vid} on its way {now - t.left_t:.1f} s after leaving")
                elif now - t.left_t >= LEAVE_NUDGE_S and not getattr(t, "nudged", False) and \
                        c.vehicle.getLaneID(t.vid).startswith(":"):
                    # it loaded in the junction and waits there to give way: it is in everyone's way, so it goes
                    # (as a driver edges out of a box), for NUDGE_S not waiting for those already in the junction
                    t.nudged = True
                    try:
                        c.vehicle.setSpeedMode(t.vid, 31 | 32)
                        self.nudged = getattr(self, "nudged", {})
                        self.nudged[t.vid] = now + NUDGE_S
                    except Exception:  # noqa: BLE001
                        pass
                    inc.log(now, f"{t.vid} edges out of the junction it loaded in")
                elif now - t.left_t >= getattr(t, "next_why", 10.0):
                    t.next_why = now - t.left_t + 10.0
                    try:
                        why = (c.vehicle.getLaneID(t.vid), round(c.vehicle.getLanePosition(t.vid), 1),
                               c.vehicle.getRoute(t.vid)[c.vehicle.getRouteIndex(t.vid):][:3], c.vehicle.getStopState(t.vid),
                               c.vehicle.getLeader(t.vid, 30.0), c.vehicle.getNextTLS(t.vid)[:1],
                               c.vehicle.getSpeedMode(t.vid), round(c.vehicle.getAllowedSpeed(t.vid), 1))
                    except Exception as e:  # noqa: BLE001
                        why = str(e)
                    inc.log(now, f"{t.vid} still standing {now - t.left_t:.0f} s after leaving: {why}")
        # the next truck once the one before has its wreck on board (one at a time: they load at the same place)
        busy = any(t.state in ("en route", "there", "backing", "loading", "loaded") for t in inc.tows) or \
            any(t.state == "leaving" and t.vid in vehicles and now - t.t_load - t.tl["total"] < 20.0
                for t in inc.tows) or \
            now < getattr(inc, "next_tow_t", 0.0)
        waiting = [w for w in inc.wrecks if not w.removed and w.tow is None]
        if waiting and not busy and now - (inc.police_t or now) >= TOW_DELAY_S:
            if not self._send_tow(inc, waiting):
                inc.next_tow_t = now + 3.0  # no room yet: again in a moment
        if all(w.removed for w in inc.wrecks) and inc.phase != "clearing":
            inc.phase = "clearing"
            inc.clear_t = now
            inc.log(now, "all wrecks gone; clearing up")

    def _tow_gives_up(self, inc: Incident, t: Tow, now: float) -> None:
        """A truck that has waited TOW_GIVE_UP_S ahead of its wreck without getting to back in (31 Oct: a second wreck
        in a junction, its truck behind it for 50 minutes, seed 41) goes back; the next truck comes another way, and
        if that one cannot either, the wreck is lifted off where it stands (a crane truck from the side)."""
        w = next(w for w in inc.wrecks if w.vid == t.wreck)
        try:
            self.conn.vehicle.replaceStop(t.vid, 0, "")
        except Exception:  # noqa: BLE001
            pass
        self._go(t.vid)
        self._home(t.vid, "tow")
        self.role.pop(t.vid, None)
        t.state = "gone"
        w.tow = None
        w.tries = getattr(w, "tries", 0) + 1
        if w.tries >= 2:
            try:
                self._remove(w.vid)
            except Exception:  # noqa: BLE001
                pass
            w.removed = True
            self.role.pop(w.vid, None)
            inc.log(now, f"{t.vid} could not get to {w.vid} either: lifted off where it stands")
        else:
            inc.direct_tries = getattr(inc, "direct_tries", 0) + 1
            inc.next_tow_t = now + 5.0
            inc.log(now, f"{t.vid} could not get to {w.vid} in {TOW_GIVE_UP_S:.0f} s: another truck comes")
        self.stats["tow_gave_up"] = self.stats.get("tow_gave_up", 0) + 1

    def _past(self, inc: Incident, t: Tow) -> bool:
        """The truck is ahead of its wreck's front (on the wreck's road, or past it)."""
        if getattr(t, "direct", False):
            return True
        c = self.conn
        w = next((w for w in inc.wrecks if w.vid == t.wreck), None)
        if w is None:
            return False
        try:
            road, p = c.vehicle.getRoadID(t.vid), c.vehicle.getLanePosition(t.vid)
        except Exception:  # noqa: BLE001
            return False
        w_road = w.lane.rsplit("_", 1)[0]
        if w.lane.startswith(":"):  # a wreck in a junction: past it is on the road it was heading for, or beyond
            ahead = self._via_to_edge(w.lane)
            return road == ahead or (not road.startswith(":") and ahead in self._route_before(t.vid, road))
        if road == w_road:
            return p >= min(w.pos - 0.5, self.net.getLane(w.lane).getLength() - 1.0)  # beside its front at least
        return not road.startswith(":") and road != w_road and w_road in self._route_before(t.vid, road)

    def _route_before(self, vid: str, road: str) -> list:
        try:
            route = list(self.conn.vehicle.getRoute(vid))
            return route[:route.index(road)] if road in route else []
        except Exception:  # noqa: BLE001
            return []

    def _load_place(self, inc: Incident, w: Wreck) -> tuple[str, float] | None:
        """(lane, position of the truck's front) where a truck loads wreck `w`: ahead of it by towing.truck_ahead()
        (the tilted deck's end just before its front, as drawn: pushed on by the crash), along the way the wreck was
        going: its lane, then on through the junction (its own way through it, or straight on). The truck may then
        stand in the junction: SUMO's traffic there waits for it, as it would."""
        from cosim import towing

        need = w.pos + max(w.damage.get("dx", 0.0), 0.0) + towing.truck_ahead()
        lane = w.lane
        for _ in range(4):
            ln = self.net.getLane(lane)
            if need <= ln.getLength() - 0.2:
                return lane, max(need, 0.5)
            need -= ln.getLength()
            if lane.startswith(":"):
                to = self._via.get(lane, (None, None))[1]
                if to is None:
                    return lane, ln.getLength() - 0.2
                lane = to
                continue
            conns = list(ln.getOutgoing())
            if not conns:
                return lane, ln.getLength() - 0.2
            k = next((k for k in conns if k.getDirection() == "s"), conns[0])
            lane = k.getViaLaneID() or k.getToLane().getID()
        return lane, self.net.getLane(lane).getLength() - 0.2

    def _road_ahead(self, lane: str) -> str:
        """The first normal lane at or after `lane` (a junction's internal lane leads to one)."""
        for _ in range(3):
            if not lane.startswith(":"):
                return lane
            lane = self._via.get(lane, (None, None))[1] or lane
            if lane.startswith(":"):
                break
        return lane

    def _back_in(self, inc: Incident, t: Tow, now: float) -> None:
        """The truck has stopped ahead of the wreck on the open lane. The traffic behind it on that lane is held back
        (as the crew and the police would) and, once the place in front of the wreck and the road behind the truck
        are clear, it backs in (cosim/towing.py draws it, REV_S; SUMO keeps it on the open lane meanwhile, then puts
        it in place and holds it while it loads)."""
        from cosim import towing

        c = self.conn
        w = next(w for w in inc.wrecks if w.vid == t.wreck)
        place = self._load_place(inc, w)
        if place is None:
            return
        lane, pos = place
        mine = {t.vid} | {x.vid for x in inc.wrecks}
        try:
            t_lane, t_pos = c.vehicle.getLaneID(t.vid), c.vehicle.getLanePosition(t.vid)
        except Exception:  # noqa: BLE001
            return
        same = lane.rsplit("_", 1)[0] == t_lane.rsplit("_", 1)[0]
        # its body sweeps from where it stands to the place: from the rearmost of the two (along its lane) on, nobody
        # may be on its lane; who is behind that waits (as the crew and the police would have them)
        if getattr(t, "hold", None) is None or t.state == "there":
            along = self._along(t_lane, lane, pos)
            sweep = min(t_pos, t_pos if along is None else along) - 2 * towing.TRUCK_HALF - 1.0
            t.hold = (t_lane, t_pos, sweep - 5.0, sweep, along)
        t_lane, t_pos, line, sweep, along = t.hold
        clear = True
        blockers = []
        try:
            for o in c.lane.getLastStepVehicleIDs(t_lane):
                if o in mine or o in self.role:
                    continue
                p = c.vehicle.getLanePosition(o)
                if p < sweep:
                    if o not in inc.held:
                        c.vehicle.setSpeed(o, 0.0)
                        inc.held.add(o)
                elif p < t_pos:
                    clear = False  # where its body would sweep: it gets out of the way first
                    blockers.append(o)
                elif along is not None and along > t_pos and p - c.vehicle.getLength(o) < along + 1.0:
                    # ahead of it, where it moves on to the place (stopped short of it in a queue): it drives on
                    # first (seed 7, watched: the truck drawn through the car ahead of it)
                    clear = False
        except Exception:  # noqa: BLE001
            pass
        if (not inc.open_lane or getattr(t, "direct", False)) and t.state in ("there", "backing"):
            # a crash in a junction: the truck backs from the road ahead to the junction across that road's lanes,
            # which stay open; who comes up behind where it sweeps waits, on every lane, and so does who comes up to
            # the junction behind it, while it backs too (seed 7: cars on the next lane drove through the truck)
            hi = max(t_pos, along if along is not None else t_pos) + 1.0
            edge = self.net.getLane(t_lane).getEdge()
            try:
                for ln in edge.getLanes():
                    if not ln.allows("passenger"):
                        continue
                    for o in c.lane.getLastStepVehicleIDs(ln.getID()):
                        if o in mine or o in self.role or o in blockers:
                            continue
                        p = c.vehicle.getLanePosition(o)
                        if ln.getID() == t_lane and p < t_pos:
                            continue  # its own lane behind it: above
                        if p <= sweep:
                            if o not in inc.held:  # coming up behind where it sweeps: waits
                                c.vehicle.setSpeed(o, 0.0)
                                inc.held.add(o)
                        elif p - c.vehicle.getLength(o) < hi and t.state == "there" and \
                                now - (t.t_there or now) < ADJ_WAIT_S:
                            # where it sweeps: goes by first, for a while (one standing in a queue may never: then
                            # the truck goes ahead, seeds 11 and 26 stood ahead of the junction for good)
                            clear = False
                if lane.startswith(":") and t.state == "there" and now - (t.t_there or now) < ADJ_WAIT_S:
                    # it backs into the junction: nobody in it meanwhile (cars crossing it drove through the truck
                    # backing in, seed 11); its other traffic is held at its stop lines by the signals, or waits
                    j = lane[1:].rsplit("_", 2)[0]
                    for via in self._junction_lanes(j):
                        if any(o not in mine and o not in self.role for o in c.lane.getLastStepVehicleIDs(via)):
                            clear = False
                            break
                if along is not None and along < 0.0:  # it backs into the junction behind: hold who comes to it
                    for frm in edge.getIncoming():
                        for k in frm.getOutgoing().get(edge, []):
                            fl = k.getFromLane()
                            for o in c.lane.getLastStepVehicleIDs(fl.getID()):
                                if o in mine or o in self.role or o in inc.held:
                                    continue
                                left = fl.getLength() - c.vehicle.getLanePosition(o)
                                sp = c.vehicle.getSpeed(o)
                                if left < 60.0 and sp * sp / 8.0 < left - 2.0:
                                    c.vehicle.setSpeed(o, 0.0)
                                    inc.held.add(o)
            except Exception:  # noqa: BLE001
                pass
        if blockers and t.state == "there":
            self._out_of_the_way(inc, t, blockers, now)
        w_front = w.pos + max(w.damage.get("dx", 0.0), 0.0) if lane == w.lane else 0.0
        try:
            for o in c.lane.getLastStepVehicleIDs(lane):
                if o in mine:
                    continue
                p = c.vehicle.getLanePosition(o)
                if p > w_front + 0.3 and p - c.vehicle.getLength(o) < pos + 1.0:  # in the place the truck needs
                    clear = False
                    if o in inc.held:  # held coming up to it, it rolled in before it stood: it goes on (it stood
                        try:           # there for good, the truck waiting for it, seed 7 watched)
                            c.vehicle.setSpeed(o, -1)
                        except Exception:  # noqa: BLE001
                            pass
                        inc.held.discard(o)
                    if now - getattr(t, "waited_log", -1e9) > 15.0:
                        t.waited_log = now
                        inc.log(now, f"{t.vid} waits: {o} at {p:.0f} m where it would load ({pos:.0f} m on {lane})")
        except Exception:  # noqa: BLE001
            pass
        if t.state == "there":
            past = self._past(inc, t)
            if not clear:
                t.blocked_s = getattr(t, "blocked_s", 0.0) + 0.5
                if t.blocked_s > UNBLOCK_S and int(now * 2) % 10 == 0:
                    # who stands in its way because the next road it wants is reached only from the lane the truck
                    # stands on takes a road its own lane leads to (they waited for each other for good: a car on
                    # 8_3 wanting 8_4, the truck on 8_4 waiting for 8_3 to clear, seed 7 watched)
                    for e_ in {lane.rsplit("_", 1)[0], t_lane.rsplit("_", 1)[0]}:
                        try:
                            ids = c.edge.getLastStepVehicleIDs(e_)
                        except Exception:  # noqa: BLE001
                            continue
                        for o in ids:
                            if o not in mine and o not in self.role and not o.startswith("pt_"):
                                self._keep_lane(o)
            else:
                t.blocked_s = 0.0
            if not clear or not past:
                if now - getattr(t, "why_log", -1e9) > 20.0:
                    t.why_log = now
                    inc.log(now, f"{t.vid} waits to back in: {'road behind it not clear' if not clear else ''}"
                                 f"{'' if past else ' not past the wreck'} ({t_lane} {t_pos:.0f} m; place {lane} "
                                 f"{pos:.0f} m)")
                return
            x, y, ang = self._lane_pose(lane, pos)
            t.load = (x, y, ang, 2 * towing.TRUCK_HALF)
            t.tl = towing.timeline(w.length)
            t.state, t.t_load = "backing", now
            t.place = (lane, pos)
            inc.log(now, f"{t.vid} backs in ahead of {w.vid} ({t.tl['total']:.0f} s to load it)")
            # in (or past) the junction: SUMO puts it there now, so the cross traffic waits from now on, before the
            # truck drawn backing in gets there
            if not same:
                self._ghost(inc, t, t_lane, t_pos)
                self._put_in_place(inc, t, w, now)
            return
        if t.state != "backing" or now - t.t_load < towing.REV_S - 0.25:
            return
        self._put_in_place(inc, t, w, now)

    def _along(self, t_lane: str, lane: str, pos: float) -> float | None:
        """Where `pos` on `lane` is along the truck's lane `t_lane` (m from its start): on the same road, in the
        junction past its end (beyond its length) or in the junction before it (negative); None elsewhere."""
        road = t_lane.rsplit("_", 1)[0]
        if lane.rsplit("_", 1)[0] == road:
            return pos
        if lane.startswith(":"):
            fr, to = self._via.get(lane, (None, None))
            try:
                if fr and fr.rsplit("_", 1)[0] == road:
                    return self.net.getLane(t_lane).getLength() + pos
                if to and to.rsplit("_", 1)[0] == road:
                    return pos - self.net.getLane(lane).getLength()
            except Exception:  # noqa: BLE001
                return None
        return None

    MAKE_WAY_S = 6.0  # a car standing where the truck backs this long: it is waved round (_out_of_the_way)

    def _out_of_the_way(self, inc: Incident, t: Tow, blockers: list, now: float) -> None:
        """Cars stopped right behind the truck, where it would back: the crew waves them round it, onto the next
        lane (at a crawl if they stand, as _still_stuck). Before 24 Oct the truck waited for them to go by
        themselves, which they never did while someone watched (out of sight they were taken off the road): the
        truck stood ahead of the wreck for good (seeds 7 and 11, watched)."""
        c = self.conn
        if getattr(t, "blocked_t", None) is None:
            t.blocked_t = now
        if now - t.blocked_t < self.MAKE_WAY_S:
            return
        closed = {cl for cl, _ in inc.closed}
        waved = getattr(t, "waved", set())
        t.waved = waved
        for o in blockers:
            if o in waved and now - getattr(t, "waved_t", {}).get(o, now) < 30.0:
                continue
            try:
                ln = c.vehicle.getLaneID(o)
                lane = self.net.getLane(ln)
                vc = c.vehicle.getVehicleClass(o)
                lanes = lane.getEdge().getLanes()
                i = lane.getIndex()
                cands = [x for x in (i + 1, i - 1) if 0 <= x < len(lanes) and lanes[x].allows(vc)
                         and lanes[x].getID() not in closed]
                if not cands:
                    continue
                if o not in self.squeezed:
                    self._pull_out(o)
                c.vehicle.setParameter(o, "laneChangeModel.lcMaxSpeedLatStanding", "0.3")
                c.vehicle.setLaneChangeMode(o, 0b011001010101)
                c.vehicle.changeLane(o, cands[0], 30.0)
                rec = self.squeezed.get(o)
                if rec is not None and len(rec) <= 4:
                    self.squeezed[o] = rec + (True,)
                waved.add(o)
                t.waved_t = {**getattr(t, "waved_t", {}), o: now}
                self.stats["waved_round"] = self.stats.get("waved_round", 0) + 1
                inc.log(now, f"{o} waved round {t.vid} onto {lanes[cands[0]].getID()}")
            except Exception as e:  # noqa: BLE001
                log.debug("wave round %s: %s", o, e)

    def _pre_hold(self, inc: Incident, t: Tow) -> None:
        """The truck pulls up to stop ahead of the wreck: who follows it on that lane stops short of where it will
        back (else the next car stops right behind it, where it backs: _out_of_the_way)."""
        from cosim import towing

        c = self.conn
        w = next((w for w in inc.wrecks if w.vid == t.wreck), None)
        if w is None:
            return
        try:
            stops = c.vehicle.getStops(t.vid, 1)
            if not stops:
                return
            s_lane, s_pos = stops[0].lane, stops[0].endPos
            if c.vehicle.getLaneID(t.vid) != s_lane:
                return
            t_pos = c.vehicle.getLanePosition(t.vid)
        except Exception:  # noqa: BLE001
            return
        place = self._load_place(inc, w)
        if place is None:
            return
        along = self._along(s_lane, *place)
        sweep = min(s_pos, s_pos if along is None else along) - 2 * towing.TRUCK_HALF - 1.0
        mine = {t.vid} | {x.vid for x in inc.wrecks}
        try:
            for o in c.lane.getLastStepVehicleIDs(s_lane):
                if o in mine or o in self.role or o in inc.held:
                    continue
                p = c.vehicle.getLanePosition(o)
                if p < min(sweep, t_pos - 2 * towing.TRUCK_HALF - 1.0):
                    c.vehicle.setSpeed(o, 0.0)
                    inc.held.add(o)
        except Exception:  # noqa: BLE001
            pass

    def _put_in_place(self, inc: Incident, t: Tow, w: Wreck, now: float) -> None:
        """SUMO puts the truck where it loads and holds it there; once it is there as drawn (REV_S after it started
        backing), the loading starts and the traffic held behind it may go."""
        from cosim import towing

        if t.moved:  # put there already (a junction): only the start of the loading
            if now - t.t_load >= towing.REV_S - 0.25:
                t.state = "loading"
                self._release_held(inc)
                inc.log(now, f"{t.vid} in place ahead of {w.vid}; loading")
            return
        c = self.conn
        lane, pos = t.place
        try:
            c.vehicle.replaceStop(t.vid, 0, "")
        except Exception:  # noqa: BLE001
            pass
        moved = False
        try:  # its route through the place first (it may load in the junction or past it, on another road)
            road = c.vehicle.getRoadID(t.vid)
            to_edge = self._via_to_edge(lane) if lane.startswith(":") else lane.rsplit("_", 1)[0]
            if to_edge and to_edge != road:
                route = list(c.vehicle.getRoute(t.vid))
                r = self._route(to_edge, route[-1], c.vehicle.getTypeID(t.vid), t.vid) \
                    if route[-1] != to_edge else None
                c.vehicle.setRoute(t.vid, [road] + (list(r.edges) if r is not None and r.edges else [to_edge]))
        except Exception as e:  # noqa: BLE001
            log.debug("route through %s for %s: %s", lane, t.vid, e)
        try:
            c.vehicle.moveTo(t.vid, lane, pos)
            moved = True
        except Exception as e:  # noqa: BLE001
            log.info("recovery truck %s not moved to %s: %s", t.vid, lane, e)
        if moved:
            self._route_on(t.vid, lane)
        try:
            c.vehicle.setLaneChangeMode(t.vid, 0)
            c.vehicle.setSpeedMode(t.vid, 31)
            c.vehicle.setSpeed(t.vid, 0.0)
        except Exception:  # noqa: BLE001
            pass
        t.moved, t.rejoin = True, not moved  # (tried: not again)
        if not moved:
            inc.log(now, f"{t.vid}: SUMO could not put it in place")
        if now - t.t_load >= towing.REV_S - 0.25:
            t.state = "loading"
            self._release_held(inc)
            inc.log(now, f"{t.vid} in place ahead of {w.vid}; loading")

    GUARD_M = 2.75  # a way through a junction this close to a loading truck's body (half its width, half a car's):
                    # held while it loads

    def _guard_place(self, inc: Incident, t: Tow) -> None:
        """A truck that loads where SUMO does not have it (a wreck in a junction: SUMO could not move it there, or
        its place is a way through the junction) is drawn standing where SUMO's cars drive: they drove through it
        (m19/m20, seeds 11, 26, 31: up to 29 frames a run). While it backs in and loads, the cars coming up to the
        ways through the junction that pass its body wait at the stop line (those already past go on); they go
        when it leaves."""
        c = self.conn
        guard = getattr(t, "guard", None)
        if t.state not in ("backing", "loading", "loaded") or t.load is None or t.place is None:
            if guard:
                for o in guard:
                    try:
                        c.vehicle.setSpeed(o, -1)
                    except Exception:  # noqa: BLE001
                        pass
                t.guard = set()
            return
        fx, fy, ang, length = t.load
        if not t.place[0].startswith(":"):
            try:
                sx, sy = c.vehicle.getPosition(t.vid)
            except Exception:  # noqa: BLE001
                sx, sy = 1e9, 1e9
            if math.hypot(sx - fx, sy - fy) < 3.0:
                return  # SUMO holds it where it is drawn: the cars stop behind it
        if getattr(t, "guard_vias", None) is None:
            a = math.radians(ang)
            hx, hy = math.sin(a), math.cos(a)
            body = [(fx - hx * k, fy - hy * k) for k in [i * 0.5 for i in range(int(length / 0.5) + 1)]]
            vias = set()
            if not hasattr(self, "_internal_lanes"):
                self._internal_lanes = [ln.getID() for e in self.net.getEdges() if e.getFunction() == "internal"
                                        for ln in e.getLanes()]
            for via in self._internal_lanes:  # (every part of every way through, not only the first)
                try:
                    shape = self.net.getLane(via).getShape()
                except Exception:  # noqa: BLE001
                    continue
                if math.hypot(shape[0][0] - fx, shape[0][1] - fy) > 80.0:
                    continue
                pts = []
                for (x0, y0), (x1, y1) in zip(shape, shape[1:], strict=False):
                    n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.5))
                    pts += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n + 1)]
                if pts and min(math.hypot(px - bx, py - by) for px, py in pts[::2] for bx, by in body[::2]) \
                        < self.GUARD_M + 0.5:
                    vias.add(via)
            t.guard_vias = vias
            t.guard = set()
        mine = {t.vid} | {x.vid for x in inc.wrecks}
        for via in t.guard_vias:
            frm, to = self._via.get(via, (None, None))
            frm = frm or via
            for _ in range(4):  # (a later part of a way through: back to the road it starts from)
                if not frm or not frm.startswith(":"):
                    break
                frm = self._from_internal(frm)
            to = to or self._to_internal(via)
            for _ in range(4):
                if not to or not to.startswith(":"):
                    break
                to = self._to_internal(to)
            if not frm or frm.startswith(":") or not to:
                continue
            to_edge = to.rsplit("_", 1)[0]
            try:
                ids = c.lane.getLastStepVehicleIDs(frm)
                flen = c.lane.getLength(frm)
            except Exception:  # noqa: BLE001
                continue
            for o in ids:
                if o in mine or o in self.role or o in t.guard or o in inc.held:
                    continue
                try:
                    route, i = c.vehicle.getRoute(o), c.vehicle.getRouteIndex(o)
                    if i + 1 >= len(route) or route[i + 1] != to_edge:
                        continue
                    left = flen - c.vehicle.getLanePosition(o)
                    sp = c.vehicle.getSpeed(o)
                    if left < 60.0 and sp * sp / 8.0 < left - 1.0:
                        c.vehicle.setSpeed(o, 0.0)
                        t.guard.add(o)
                except Exception:  # noqa: BLE001
                    continue

    def _to_internal(self, lane: str) -> str | None:
        """The lane internal lane `lane` leads onto (sumolib)."""
        try:
            out = self.net.getLane(lane).getOutgoing()
            return out[0].getToLane().getID() if out else None
        except Exception:  # noqa: BLE001
            return None

    def _from_internal(self, lane: str) -> str | None:
        """The lane that leads into internal lane `lane` (sumolib)."""
        try:
            inc = self.net.getLane(lane).getIncoming()
            return inc[0].getID() if inc else None
        except Exception:  # noqa: BLE001
            return None

    def _ghost(self, inc: Incident, t: Tow, lane: str, pos: float) -> None:
        """SUMO puts a truck that loads in (or past) a junction there at once, but it is drawn backing in from where
        it stood for REV_S more: a stopped vehicle of its size, never drawn (BLOCKER_PREFIX), stands there meanwhile,
        so who comes up behind stops behind it (in a seed-7 run a car drove through the truck drawn backing)."""
        from cosim import towing

        c = self.conn
        gt = "inc_ghost"
        try:
            if gt not in c.vehicletype.getIDList():
                c.vehicletype.copy("DEFAULT_VEHTYPE", gt)
                c.vehicletype.setVehicleClass(gt, "authority")
                c.vehicletype.setLength(gt, 2 * towing.TRUCK_HALF)
                c.vehicletype.setWidth(gt, 2.5)
                c.vehicletype.setMinGap(gt, 1.0)
            gid = f"{BLOCKER_PREFIX}{inc.id}_{t.vid}"
            rid = f"{gid}_r"
            if rid not in c.route.getIDList():
                c.route.add(rid, [lane.rsplit("_", 1)[0]])
            c.vehicle.add(gid, rid, typeID=gt, depart="now", departLane=lane.rsplit("_", 1)[1],
                          departPos=f"{max(pos, 2 * towing.TRUCK_HALF):.1f}", departSpeed="0")
            c.vehicle.setSpeedMode(gid, 31)
            c.vehicle.setSpeed(gid, 0.0)
            c.vehicle.setLaneChangeMode(gid, 0)
            self.role[gid] = (inc.id, "blocker")
            t.ghost = gid
        except Exception as e:  # noqa: BLE001
            log.debug("stand-in for %s: %s", t.vid, e)

    def _drop_ghost(self, t: Tow) -> None:
        gid = getattr(t, "ghost", None)
        if gid:
            try:
                self._remove(gid)
            except Exception:  # noqa: BLE001
                pass
            self.role.pop(gid, None)
            t.ghost = None

    def _release_held(self, inc: Incident) -> None:
        for t in inc.tows:
            if t.state != "backing":
                self._drop_ghost(t)
        for o in list(inc.held):
            try:
                self.conn.vehicle.setSpeed(o, -1)
            except Exception:  # noqa: BLE001
                pass
            inc.held.discard(o)

    def _via_to_edge(self, via: str) -> str | None:
        to = self._via.get(via, (None, None))[1]
        return to.rsplit("_", 1)[0] if to else None

    def _route_on(self, vid: str, lane: str) -> None:
        """A route onwards that starts with a road `lane` itself leads to: a truck put on the closed lane would else
        need to change lanes at a standstill to reach its next road (it cannot: it stood there for good, and the next
        truck waited behind it)."""
        c = self.conn
        ln = self.net.getLane(lane)
        if lane.startswith(":"):  # in a junction: it leaves it on the road its internal lane leads to
            e = self._via_to_edge(lane)
            nexts = [e] if e else []
        else:
            nexts = [k.getTo().getID() for k in ln.getOutgoing() if k.getTo().getFunction() != "internal"]
        if not nexts:
            return
        try:
            route = list(c.vehicle.getRoute(vid))
            i = c.vehicle.getRouteIndex(vid)
        except Exception:  # noqa: BLE001
            return
        if route[i + 1:i + 2] and route[i + 1] in nexts:
            return
        dest = route[-1]
        vtype = c.vehicle.getTypeID(vid)
        for nxt in nexts:
            try:
                r = self._route(nxt, dest, vtype, vid) if nxt != dest else None
                rest = list(r.edges) if r is not None and r.edges else [nxt]
                c.vehicle.setRoute(vid, [route[i]] + rest)
                return
            except Exception:  # noqa: BLE001
                continue

    def _send_tow(self, inc: Incident, waiting: list) -> bool:
        """A recovery truck (cosim/towing.py) for the front-most wreck still there: it comes up the open lane, passes
        the scene and stops 8 m beyond where it will load, then backs in (_back_in). With no open lane (every lane
        blocked) it stops behind the wrecks."""
        from cosim import towing

        c = self.conn
        lane, rear, front = self._anchor(inc)
        if not lane:
            return False
        on = [w for w in waiting if w.lane == lane] or waiting
        w = max(on, key=lambda w: w.pos)
        k = len(inc.tows)
        tid_v = f"inc{inc.id}_tow{k}"
        tow_t = self._tow_type(tid_v)
        tlen = c.vehicletype.getLength(tow_t)
        place = self._load_place(inc, w)
        stops = []
        direct = None
        no_way = bool(inc.open_lane) and not self._feeders(inc.open_lane)
        if place is not None and (not inc.open_lane or no_way):
            # (and when no road leads onto the open lane, only its closed neighbour does: nobody could get to the
            # wreck's front along it, seed 11)
            # a wreck in a junction (no lane closed) blocks the way into the road it was heading for: the truck is
            # put on that road a little past the place (out of sight if it can be) and backs in from there
            road = self._road_ahead(place[0])
            r_len = self.net.getLane(road).getLength()
            p = place[1] + 18.0 if road == place[0] else 2 * towing.TRUCK_HALF + 18.0
            # tried there and SUMO found no room (traffic standing or coming up behind): another lane of that road,
            # or further on (seed 23, watched: a truck waiting to get on for 20 minutes)
            n = getattr(inc, "direct_tries", 0)
            if n:
                ln_ = self.net.getLane(road)
                others = [x.getID() for x in ln_.getEdge().getLanes() if x.getID() != road and x.allows("truck")]
                lanes_ = [road] + others
                road = lanes_[n % len(lanes_)]
                r_len = self.net.getLane(road).getLength()
                p += 10.0 * (n // len(lanes_))
            direct = (road, min(r_len - 0.2, p))
        if inc.open_lane and place is not None and not no_way:
            ol = inc.open_lane
            same_edge = ol.rsplit("_", 1)[0] == place[0].rsplit("_", 1)[0]
            ol_len = self.net.getLane(ol).getLength()
            # ahead of where it will load, on the open lane (at its end when it loads in or past the junction)
            stops.append((ol, min(ol_len - 0.2, place[1] + 8.0) if same_edge else ol_len - 0.2))
        # (never behind the wrecks: it loads from ahead of them; no room on the way there yet: again in a moment)
        late = self.now - (inc.police_t or self.now) > 45.0
        if direct is not None:
            road, pos = direct
            route = self._onwards(road.rsplit("_", 1)[0], tow_t)
            if self._insert(tid_v, tow_t, route, road, pos, 0.0, 12.0, True, extend=False):
                try:
                    c.vehicle.setSpeed(tid_v, 0.0)  # it is where it backs in from
                    c.vehicle.setLaneChangeMode(tid_v, 0)
                except Exception:  # noqa: BLE001
                    pass
                inc.tows.append(Tow(tid_v, w.vid, state="there", t_there=self.now))
                inc.tows[-1].direct = True  # put on the road ahead of the place: past the wreck by construction
                w.tow = tid_v
                self.role[tid_v] = (inc.id, "tow")
                inc.log(self.now, f"recovery truck {tid_v} for {w.vid} comes along {road.rsplit('_', 1)[0]}")
                return True
            return False
        for stop_lane, pos in stops:
            s_edge = stop_lane.rsplit("_", 1)[0]
            route = self._onwards_from(stop_lane, tow_t)
            self._no_depot_now = getattr(inc, "no_depot", False)
            dep_v = self._from_depot("tow", s_edge, route, lane=stop_lane)
            self._no_depot_now = False
            if dep_v is not None:
                tid_v = dep_v
            elif not self._insert(tid_v, tow_t, route, stop_lane, pos - tlen, 25.0, 160.0, late):
                continue
            try:
                c.vehicle.setStop(tid_v, s_edge, pos=pos, laneIndex=int(stop_lane.rsplit("_", 1)[1]), duration=1e5)
            except Exception as e:  # noqa: BLE001
                log.info("recovery truck stop for incident %d: %s", inc.id, e)
                if dep_v is not None:  # back to the yard
                    self._home(dep_v, "tow")
                    tid_v = f"inc{inc.id}_tow{k}"
                    continue
                try:
                    self._remove(tid_v)
                except Exception:  # noqa: BLE001
                    pass
                continue
            inc.tows.append(Tow(tid_v, w.vid))
            inc.tows[-1].sent_t = self.now
            w.tow = tid_v
            self.role[tid_v] = (inc.id, "tow")
            b = self.inserted_back.get(tid_v)
            inc.log(self.now, f"recovery truck {tid_v} sent for {w.vid} (stops on {stop_lane})" + (
                (f", from {b[1]}, {b[0]} m away" if b[1].startswith("the ") else f", from {b[0]} m back ({b[1]})")
                if b else ""))
            return True
        return False

    # ------------------------------------------------------------------ for the viewer
    def protected(self, vid: str) -> bool:
        """Never taken off the road for the day's curve, never given a new trip."""
        return vid in self.role or any(vid == t.vid for i in self.incidents for t in i.tows) or \
            (self.depots is not None and self.depots.owns(vid)) or vid in self.breaking or \
            any(vid == lk.vid for lk in self.lookalikes)

    def blocked_roads(self) -> set:
        """The roads of the accidents under way (their wrecks' and the closed lanes')."""
        out = set()
        for i in self.incidents:
            out |= {w.lane.rsplit("_", 1)[0] for w in i.wrecks if w.lane and not w.removed}
            out |= {ln.rsplit("_", 1)[0] for ln, _ in i.closed}
        return {e for e in out if not e.startswith(":")}

    def on_run(self, vid: str) -> bool:
        """A vehicle the signals let through as an emergency vehicle: the police car on its way and at the scene, and
        a recovery truck that has just loaded until it rolls (the officer waves it off: loaded at a stop line, it
        waited for a whole cycle of the signal, standing in the junction's mouth, before 24 Oct)."""
        r = self.role.get(vid)
        if r is None or r[1] not in ("police", "tow"):
            return False
        inc = next((i for i in self.incidents if i.id == r[0]), None)
        if inc is None:
            return False
        if r[1] == "tow":
            return any(t.vid == vid and t.state == "leaving" and not getattr(t, "rolling", False) for t in inc.tows)
        return inc.phase in ("police en route", "secured", "recovery", "clearing")

    def hit_poses(self, now: float) -> dict:
        """Wrecks that crashed in the step ending at `now`: vid -> SUMO (x, y, angle) where they touched (put back to
        there, _to_contact). The viewer draws their last step ending there (cosim/freefly). Their centring in the lane
        (_fit_in_lane, up to a lane's width for a car hit while changing lanes) is drawn in the step after, with the
        push and turn of the impact: drawn in the hit's step, a standing car slid sideways into the hit (CARLA trace,
        3 Nov, tools/check_crash_trace.py: 1.5 m in the 0.5 s before it)."""
        out = {}
        for i in self.incidents:
            for w in i.wrecks:
                if w.t_hit is not None and now - STEP_S - 1e-6 <= w.t_hit <= now + 1e-6 and not w.removed:
                    out[w.vid] = w.hit_pose or (w.x, w.y, w.angle)
        return out

    def light_bits(self, vid: str) -> int | None:
        """Extra light bits (carla.VehicleLightState values) for incident vehicles: wrecks' hazards, police beacons,
        the trucks' hazards while stopped. None: not an incident vehicle."""
        r = self.role.get(vid)
        if r is None and (vid in self.breaking or any(vid == lk.vid and lk.kind == "double-parked" and lk.t1 is None
                                                       for lk in self.lookalikes)):
            return 0x10 | 0x20  # hazards: a car breaking down, a van stopped in its lane
        if r is None:
            for i in self.incidents:
                for t in i.tows:
                    if t.vid == vid and t.state in ("there", "backing", "loading", "loaded"):
                        backing = t.state == "backing"
                        return 0x10 | 0x20 | (0x40 if backing else 0)  # hazards (blinkers), reversing light
            return None
        if r[1] == "wreck":
            w = next((w for i in self.incidents if i.id == r[0] for w in i.wrecks if w.vid == vid), None)
            if w is not None and w.t_hit is not None and self.now - w.t_hit < HAZARDS_AFTER_S:
                return 0  # (3 Nov) not yet: on with the hit they were on before the cars touched
            return 0x10 | 0x20
        if r[1] == "police":
            return 0x200 | 0x1 if self.on_run(vid) else 0x1  # Special1 (beacons) | Position
        if r[1] == "tow":
            for i in self.incidents:
                for t in i.tows:
                    if t.vid == vid and t.state in ("there", "backing", "loading", "loaded"):
                        backing = t.state == "backing"
                        return 0x10 | 0x20 | (0x40 if backing else 0)
            return None
        return None

    def snapshot(self) -> list[dict]:
        out = []
        for i in self.incidents:
            out.append({
                "id": i.id, "phase": i.phase, "kind": i.kind, "label": i.label, "severity": i.severity, "t0": i.t0,
                "cause": getattr(i, "cause", None), "speeding": [w.vid for w in i.wrecks if w.speeding],
                "x": i.x, "y": i.y, "edge": i.edge, "lane": i.lane, "open_lane": i.open_lane,
                "closed": [ln for ln, _ in i.closed], "police": i.police, "police_t": i.police_t,
                "police_at": getattr(i, "police_at", None), "police_record": dict(i.police_record),
                "t_contact": i.t_contact, "crash_source": i.crash_source,
                "wrecks": [{"vid": w.vid, "model": w.model, "role": w.role, "lane": w.lane, "pos": w.pos, "x": w.x,
                            "y": w.y, "angle": w.angle, "length": w.length, "damage": w.damage, "removed": w.removed,
                            "speed": w.speed, "t_hit": w.t_hit,
                            "tow": w.tow, "exit": dict(w.exit) if w.exit else None} for w in i.wrecks],
                "tows": [{"vid": t.vid, "wreck": t.wreck, "state": t.state, "t_load": t.t_load, "load": t.load,
                          "tl": t.tl, "rejoin": t.rejoin} for t in i.tows],
                "holding": bool(i.held) or any(h[0] <= self.now < h[1] for h in i.holds),
                "events": list(i.events[-6:]), "now": self.now})
        return out

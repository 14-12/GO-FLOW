"""Pedestrians and emergency vehicles added to the running SUMO (the main project's "extras").

Ported unchanged in behaviour from the main project, backend/src/utm/twin/extras.py
(Intelligent Multi-Agent Urban Traffic Management, ADR 0009), so the CARLA view shows the
same kind of scene the twin and its detector were built on. Only the location of the types
file differs (scenarios/twin_extras.add.xml here, a byte copy). The original documentation
follows.

The twin's "extras" scenario variant: pedestrians and emergency vehicles (ADR 0009).

The RESCO scenarios contain only cars and a few buses. The traffic-light cameras and the
detector need ambulances, police cars and pedestrians to look at, so this variant adds
them to the *running SUMO*, through TraCI, while the session runs. Nothing here is drawn
by the browser on its own: every added road user is a real SUMO agent that other traffic
reacts to.

What is added, and what it means in SUMO:

* **Emergency vehicles.** Roughly one every ``emergency_every`` simulated seconds, entering
  at a network boundary edge and leaving at another. About 60 % are ambulances
  (``vClass="emergency"``) and 40 % police cars (``vClass="authority"``). About 60 % are
  *on a run* (blue lights and siren on): they drive above the limit and may enter a
  junction against red at up to 20 km/h, using SUMO's junction-model parameters
  (``sim/scenarios/twin_extras.add.xml``). The others drive like ordinary traffic.
  Measured over the full hour (ADR 0009): on Ingolstadt7, vehicles on a run average
  11.2 m/s over their trip against 7.5 m/s for the others.
  On a run or not is the ground truth the detector is scored on.

  SUMO's bluelight device (rescue lane) is not used: SUMO 1.27.1 segfaults when a
  device-equipped vehicle passes pedestrians, with either pedestrian model.
* **Pedestrians.** Kept near ``pedestrians_per_km`` per kilometre of sidewalk. Each walks
  along *one* sidewalk, turns back, and leaves after a few legs.

  Why one sidewalk: these networks have no walking areas or crossings (they were built
  for vehicle control). Without them SUMO moves a pedestrian across a junction in one
  step, 15-30 m at once (measured on Ingolstadt7, ADR 0009). Adding crossings would change
  every signal program the baselines were tuned on. So pedestrians stay on sidewalks, and
  "pedestrians crossing" is not modelled. Networks without sidewalk lanes (the Cologne
  ones) get no pedestrians, rather than pedestrians walking on the carriageway.

Everything is driven by one seeded random generator, so the same seed gives the same
arrivals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

#: vTypes and the pedestrian type used by this variant.
EXTRAS_TYPES_FILE: Path = Path(__file__).resolve().parents[1] / "scenarios" / "twin_extras.add.xml"

#: Extra SUMO options for this variant. None: it runs with exactly the options the benchmark
#: runs use, so the only difference is the added road users.
EXTRAS_SUMO_ARGS: list[str] = []

#: Prefix that keeps person ids apart from vehicle ids in the twin's single handle space.
PERSON_PREFIX = "ped:"

#: (vType, kind, on a run)
_EMERGENCY_TYPES = {
    ("ambulance", True): "twin_ambulance_run",
    ("ambulance", False): "twin_ambulance",
    ("police", True): "twin_police_run",
    ("police", False): "twin_police",
}


@dataclass
class ExtrasConfig:
    emergency_every: float = 75.0
    ambulance_share: float = 0.6
    on_run_share: float = 0.6
    pedestrians_per_km: float = 20.0
    #: Pedestrians added per step at most, so the first steps don't insert hundreds at once.
    pedestrian_spawn_per_step: int = 6
    #: Minimum route length for an emergency vehicle, metres: it should cross the network.
    min_route_m: float = 300.0
    seed: int = 1
    #: Pedestrians walk a routed trip from one sidewalk to another, over the network's crossings
    #: (needs walking areas and crossings: scenarios/town05 after tools/build_town05_peds.py),
    #: instead of legs along one sidewalk. A trip that cannot be routed falls back to the legs.
    pedestrian_trips: bool = False
    trip_m: tuple[float, float] = (100.0, 500.0)  # straight-line distance between start and end
    #: Share of new pedestrians started within near_m of `TwinExtras.near_xy` (the free-fly camera), so the
    #: town's pedestrians gather where someone is looking instead of thinning out over 7 km of pavement.
    near_share: float = 0.0
    near_m: float = 260.0


@dataclass
class EmergencyRecord:
    vid: str
    kind: str  # "ambulance" | "police"
    on_run: bool
    depart: float
    route_m: float
    arrive: float | None = None


@dataclass
class TwinExtras:
    """Adds and tracks the variant's road users during one SUMO run."""

    config: ExtrasConfig = field(default_factory=ExtrasConfig)
    emergency: dict[str, EmergencyRecord] = field(default_factory=dict)
    #: Edge ids, filled by setup().
    entries: list[str] = field(default_factory=list)
    exits: list[str] = field(default_factory=list)
    sidewalks: list[tuple[str, float]] = field(default_factory=list)
    sidewalk_km: float = 0.0
    #: (edge id, position) -> may a pedestrian appear there? None: anywhere (free-fly: not in view)
    spawn_ok: object = None
    #: sidewalk edge id -> (x, y) of its middle, SUMO coordinates
    sidewalk_mid: dict = field(default_factory=dict)
    trips_routed: int = 0
    near_xy: tuple | None = None  # SUMO (x, y) around which near_share of new pedestrians start
    trips_unrouted: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)
        # Emergency vehicles draw from their own generator: the same seed then gives the same
        # emergency vehicles whatever the pedestrians do (they depend on the signal controller),
        # so controllers compared on one seed meet the same ambulances (tools/eval_town05_baseline.py).
        self._ev_rng = random.Random(self.config.seed * 7919 + 1)
        self._next_emergency: float | None = None
        #: (tc, now, kind, on_run) -> True when it sent a vehicle itself (cosim/depots.EmergencyCalls.spawn)
        self.spawn_hook = None
        self._ev_count = 0
        self._ped_count = 0
        self._ped_live: set[str] = set()

    # -- setup -------------------------------------------------------------------

    def setup(self, net_file: Path, begin: float) -> None:
        """Find boundary edges for emergency trips and the sidewalks pedestrians use."""
        import sumolib

        net = sumolib.net.readNet(str(net_file))
        drivable = [
            e for e in net.getEdges()
            if not e.getFunction() and e.allows("emergency") and e.allows("authority")
        ]
        self.entries = sorted(e.getID() for e in drivable if not e.getIncoming())
        self.exits = sorted(e.getID() for e in drivable if not e.getOutgoing())
        # Fall back to any drivable edge when the boundary has too few.
        if len(self.entries) < 2:
            self.entries = sorted(e.getID() for e in drivable)
        if len(self.exits) < 2:
            self.exits = sorted(e.getID() for e in drivable)

        for e in net.getEdges():
            if e.getFunction():
                continue
            # A sidewalk: a lane for pedestrians only. Lanes shared with cars don't count.
            if any(
                lane.allows("pedestrian") and not lane.allows("passenger")
                for lane in e.getLanes()
            ) and e.getLength() >= 15.0:
                self.sidewalks.append((e.getID(), e.getLength()))
                shape = e.getShape()
                self.sidewalk_mid[e.getID()] = shape[len(shape) // 2]
        self.sidewalks.sort()
        self.sidewalk_km = sum(length for _, length in self.sidewalks) / 1000.0
        self._next_emergency = begin + 20.0

    @property
    def pedestrian_target(self) -> int:
        return int(round(self.sidewalk_km * self.config.pedestrians_per_km))

    # -- per step ----------------------------------------------------------------

    def step(self, tc, now: float) -> None:
        for vid in tc.simulation.getArrivedIDList():
            record = self.emergency.get(vid)
            if record is not None:
                record.arrive = now
        if self._next_emergency is not None and now >= self._next_emergency:
            self._spawn_emergency(tc, now)
            spread = self._ev_rng.uniform(0.5, 1.5)
            self._next_emergency = now + self.config.emergency_every * spread

        if self.sidewalks:
            live = set(tc.person.getIDList())
            self._ped_live &= live
            missing = self.pedestrian_target - len(self._ped_live)
            for _ in range(min(missing, self.config.pedestrian_spawn_per_step)):
                self._spawn_pedestrian(tc, now)

    def _spawn_emergency(self, tc, now: float) -> None:
        cfg = self.config
        kind = "ambulance" if self._ev_rng.random() < cfg.ambulance_share else "police"
        on_run = self._ev_rng.random() < cfg.on_run_share
        if self.spawn_hook is not None and self.spawn_hook(tc, now, kind, on_run):
            return  # the viewer's depots took the call (cosim/depots.EmergencyCalls)
        vtype = _EMERGENCY_TYPES[(kind, on_run)]
        for _ in range(40):
            a = self._ev_rng.choice(self.entries)
            b = self._ev_rng.choice(self.exits)
            if a == b:
                continue
            route = tc.simulation.findRoute(a, b, vType=vtype)
            if route.edges and route.length >= cfg.min_route_m:
                break
        else:
            return  # no usable route this time; try again at the next interval
        self._ev_count += 1
        vid = f"{kind}{self._ev_count}"
        rid = f"twin_extras_route_{self._ev_count}"
        tc.route.add(rid, list(route.edges))
        tc.vehicle.add(vid, rid, typeID=vtype, depart="now", departLane="best",
                       departSpeed="max")
        self.emergency[vid] = EmergencyRecord(vid, kind, on_run, now, route.length)

    def _pick_sidewalk(self) -> tuple[str, float]:
        if self.near_xy is not None and self._rng.random() < self.config.near_share:
            nx, ny, r2 = self.near_xy[0], self.near_xy[1], self.config.near_m ** 2
            near = [(e, n) for e, n in self.sidewalks
                    if (self.sidewalk_mid[e][0] - nx) ** 2 + (self.sidewalk_mid[e][1] - ny) ** 2 <= r2]
            if near:
                pick = self._rng.uniform(0.0, sum(n for _, n in near))
                for e, n in near:
                    pick -= n
                    if pick <= 0:
                        return e, n
                return near[-1]
        total = sum(length for _, length in self.sidewalks)
        pick = self._rng.uniform(0.0, total)
        edge, length = self.sidewalks[-1]
        for candidate in self.sidewalks:
            pick -= candidate[1]
            if pick <= 0:
                edge, length = candidate
                break
        return edge, length

    def _trip(self, tc, edge: str) -> tuple[list[str], float] | None:
        """A walking route from `edge` to a sidewalk trip_m away, over the crossings, or None."""
        lo, hi = self.config.trip_m
        x0, y0 = self.sidewalk_mid[edge]
        far = [(e, n) for e, n in self.sidewalks
               if lo <= ((self.sidewalk_mid[e][0] - x0) ** 2 + (self.sidewalk_mid[e][1] - y0) ** 2) ** 0.5 <= hi]
        for _ in range(4):
            if not far:
                return None
            dest, length = self._rng.choice(far)
            try:
                stages = tc.simulation.findIntermodalRoute(edge, dest, modes="", pType="twin_pedestrian")
            except Exception:  # noqa: BLE001 - no route
                continue
            edges = [e for st in stages for e in st.edges]
            if edges and edges[-1] == dest:
                return edges, self._rng.uniform(1.0, length - 1.0)
        return None

    def _spawn_pedestrian(self, tc, now: float) -> None:
        for _ in range(8):
            edge, length = self._pick_sidewalk()
            pos = self._rng.uniform(1.0, length - 1.0)
            if self.spawn_ok is None or self.spawn_ok(edge, pos):
                break
        else:
            return  # every pick in view: try again next step
        self._ped_count += 1
        pid = f"{PERSON_PREFIX}{self._ped_count}"
        trip = self._trip(tc, edge) if self.config.pedestrian_trips else None
        if trip is not None:
            edges, arrival = trip
            tc.person.add(pid, edge, pos, depart=now, typeID="twin_pedestrian")
            tc.person.appendWalkingStage(pid, edges, arrivalPos=arrival)
            self._ped_live.add(pid)
            self.trips_routed += 1
            return
        if self.config.pedestrian_trips:
            self.trips_unrouted += 1
        tc.person.add(pid, edge, pos, depart=now, typeID="twin_pedestrian")
        # One to four legs back and forth along this sidewalk. No pauses: SUMO draws a
        # person in a waiting stage about 3 m to the side of the sidewalk, turned 90 degrees,
        # which would show as a sideways hop (measured, ADR 0009).
        here = pos
        towards_end = self._rng.random() < 0.5
        for _ in range(self._rng.randint(1, 4)):
            # Walk at least 3 m; if there is no room ahead, turn round first. Sidewalks
            # are >= 15 m, so one of the two directions always has room.
            if towards_end and here > length - 4.0:
                towards_end = False
            elif not towards_end and here < 4.0:
                towards_end = True
            if towards_end:
                target = self._rng.uniform(here + 3.0, length - 0.5)
            else:
                target = self._rng.uniform(0.5, here - 3.0)
            tc.person.appendWalkingStage(pid, [edge], arrivalPos=target)
            here = target
            towards_end = not towards_end
        self._ped_live.add(pid)

    # -- ground truth ------------------------------------------------------------

    def on_run(self, vid: str) -> bool:
        record = self.emergency.get(vid)
        return bool(record and record.on_run)

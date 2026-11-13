"""Trips with a purpose (step 2 of giving the town's trips real destinations, 8 Nov): most cars now go from one of the
town's places to another (scenarios/town05/places.json) for a reason that follows the time of day, park there, and go
back; the rest is through traffic on the old random routes.

* Purposes (PURPOSES): work (home -> offices, the mall, the market, cafés, restaurants, the hospital...: by their jobs),
  shopping (home -> the market while it is open, the mall), eating out (home -> café or restaurant, open), the park, the
  hospital, errands, visitors from out of town, deliveries (the vans: to the shops, then on out of town).
  Each hour of the town's day weighs them (PROFILE): to work in the morning, the market in the morning, lunch, the
  evening's shops and restaurants. Where a trip goes is drawn by the place's jobs, homes or visits, among the open ones.
* A car parks: at its destination's own car park (its spaces on CARLA's painted bays, or inside the building), or the
  car park or street the place uses. A full car park is not driven to: the nearest other one with room is.
  A car that parks at home stays there until a trip from that home takes it out again; a car that parks elsewhere goes
  back after its time there (work about 8 town hours, shopping about 1), on the route the router gives.
* Cars on their way are kept to the day's curve as before (cosim/live.Population): parked cars are not on the roads,
  so they do not count. THROUGH_SHARE of the new trips, the trucks and the ring road's cars stay as they were.

Decided from what the town is (its places, their hours, the car parks' own counts), not from any agent.
Used by the viewer and the SUMO tools through `purpose_population` (a Population that makes these trips).
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACES = ROOT / "scenarios" / "town05" / "places.json"
ADD = ROOT / "scenarios" / "town05" / "town05.places.add.xml"
TABLE = ROOT / "scenarios" / "town05" / "parked_by_hour.json"   # tools/check_purpose_day.py --table
log = logging.getLogger("purposes")

THROUGH_SHARE = 0.2          # new trips that keep the old random routes (through traffic)
OPEN_STOP_S = 1e7            # parked at home: until a trip takes the car out
MIN_IDLE_S = 40.0            # a car parked at home is taken out for a trip only after this long (it has parked)
PURPOSE_CLASSES = ("passenger", "motorcycle", "delivery")
# (12 Nov) which side of a street a car parks on, where a place has parking on both (its kerbs): "route", the side (at
# the destination, and at home for the way back) that makes the trip there and back shortest; "random", drawn (before:
# 4.4 % of these trips' km was driving round a block for the other side, 15.5 of 359 km on three dev days, almost all
# on the way home: tools/check_park_side.py). The draw is made either way, so a seed's trips stay the same.
# (13 Nov) with "route": a side of a street a bus line drives along counts this many metres further (the cars pulling in
# and out there held up the buses and the ambulances: tools/check_bus_loss.py); 0 for none
BUS_SIDE_M = 0.0
PARK_SIDE = "route"          # (13 Nov) on, the user's decision: it failed its test with accidents (docs/WORKLOG.md)


def _bump(h: float, at: float, width: float) -> float:
    d = min(abs(h - at), 24 - abs(h - at))
    return math.exp(-0.5 * (d / width) ** 2)


# purpose: (weight at town hour h, dwell in town hours (mean, spread), from, to)
PURPOSES = {
    "work": (lambda h: 1.0 * _bump(h, 7.8, 0.9) + 0.12 * _bump(h, 13.5, 1.0), (8.0, 1.0), "home", "jobs"),
    "shopping": (lambda h: 0.30 * _bump(h, 10.5, 1.8) + 0.40 * _bump(h, 17.5, 1.6), (1.0, 0.4), "home", "shops"),
    "eating": (lambda h: 0.12 * _bump(h, 8.0, 0.8) + 0.35 * _bump(h, 12.6, 0.9) + 0.40 * _bump(h, 20.0, 1.2),
               (1.2, 0.4), "home", "food"),
    "park": (lambda h: 0.10 * _bump(h, 15.0, 3.0), (1.2, 0.5), "home", "park"),
    "care": (lambda h: 0.05 * _bump(h, 13.0, 4.0), (1.5, 0.6), "home", "care"),
    "errand": (lambda h: 0.04 * _bump(h, 12.5, 3.0), (0.5, 0.2), "home", "errand"),
    "visitor": (lambda h: 0.18 * _bump(h, 14.0, 4.5), (2.0, 1.0), "out", "any"),
}
DELIVERY = (lambda h: 1.0 * _bump(h, 8.5, 2.0), (0.3, 0.1), "out", "deliver")

KIND_PURPOSE = {"office": "work", "market": "shopping", "mall": "shopping", "cafe": "eating", "restaurant": "eating",
                "park": "park", "hospital": "care", "police": "errand", "garage": "errand"}
DEST_KINDS = {
    "jobs": None,                                            # any place with jobs, by its jobs
    "shops": ("market", "mall"),
    "food": ("cafe", "restaurant"),
    "park": ("park",),
    "care": ("hospital",),
    "errand": ("police", "garage"),
    "deliver": ("market", "mall", "cafe", "restaurant"),
    "any": ("market", "mall", "cafe", "restaurant", "park", "office"),
}


@dataclass
class Plan:
    vid: str
    purpose: str
    origin: str | None           # place id (None: from out of town)
    dest: str
    home: str | None             # where it goes back to (None: out of town)
    areas: list = field(default_factory=list)   # the parking areas it stops at, in order
    t0: float = 0.0


class Places:
    """places.json, with what is open, how full each car park is, and the routes between them."""

    def __init__(self, router=None, path: Path = PLACES):
        card = json.loads(Path(path).read_text(encoding="utf-8"))
        self.path = Path(path)
        self.places = {p["id"]: p for p in card["places"]}
        self.areas = {a["id"]: a for a in card["parking_areas"]}
        if router is None:
            sys.path.insert(0, str(ROOT / "agents" / "route_recommendation"))
            from router import Router

            router = Router()
        self.router = router
        self.cap = {a["id"]: (len(a["spaces"]) if "spaces" in a else int(a.get("roadside", 0))) for a in
                    self.areas.values()}
        # where each place's cars park: its own area(s), or those of the place it parks at
        self.park_of: dict[str, list[str]] = {}
        for pid in self.places:
            self.park_of[pid] = self._areas_of(pid)
        self.xy = {aid: self._area_xy(a) for aid, a in self.areas.items()}
        # who parks in each area: the homes, the other places (weighted: homes; jobs and visits)
        self.homes_at: dict[str, list] = {}
        self.dests_at: dict[str, list] = {}
        for pid, p in self.places.items():
            for aid in self.park_of[pid]:
                if p.get("homes"):
                    self.homes_at.setdefault(aid, []).append((pid, p["homes"]))
                elif p["kind"] in KIND_PURPOSE:
                    self.dests_at.setdefault(aid, []).append((pid, p.get("jobs", 0) + p.get("visits", 0) or 10))

    def _areas_of(self, pid: str) -> list[str]:
        p = self.places[pid]
        if "parking" in p:
            return [p["parking"]]
        if "kerbs" in p:
            return [k["id"] for k in p["kerbs"]]
        if "parks_at" in p:
            return self._areas_of(p["parks_at"])
        return []

    def _area_xy(self, a):
        road = self.router.roads.get(a["edge"])
        return road.shape[len(road.shape) // 2] if road is not None and road.shape else (0.0, 0.0)

    def is_open(self, pid: str, hour: float) -> bool:
        hrs = self.places[pid].get("hours")
        if not hrs:
            return True
        a, b = hrs
        return a <= hour % 24 < b

    def choose(self, rng, what: str, hour: float, exclude: str | None = None) -> str | None:
        """A destination place for `what` (DEST_KINDS), open at `hour`, weighted by its jobs or its visits."""
        kinds = DEST_KINDS[what]
        cands = []
        for pid, p in self.places.items():
            if pid == exclude or p["kind"] == "kerb" or not self.park_of.get(pid):
                continue
            if what == "jobs":
                w = p.get("jobs", 0)
            elif p["kind"] in kinds:
                w = p.get("visits", p.get("jobs", 10)) or 10
            else:
                continue
            if w > 0 and (what == "jobs" or self.is_open(pid, hour)):
                cands.append((pid, w))
        if not cands:
            return None
        return rng.choices([c[0] for c in cands], [c[1] for c in cands])[0]

    def home(self, rng) -> str:
        homes = [(pid, p["homes"]) for pid, p in self.places.items() if p.get("homes")]
        return rng.choices([h[0] for h in homes], [h[1] for h in homes])[0]


class PurposeDemand:
    """The trips, the cars parked and on their way, the car parks' counts. One per SUMO run."""

    def __init__(self, conn, clock_speed: float, seed: int = 1, places: Places | None = None,
                 through_share: float = THROUGH_SHARE):
        self.c = conn
        self.places = places or Places()
        self.rng = random.Random(seed + 11)
        self.clock_speed = clock_speed
        self.through_share = through_share
        self.plans: dict[str, Plan] = {}
        self.parked: dict[str, str] = {}          # vehicle -> area it is parked in now
        self.idle: dict[str, list[str]] = {}      # area -> cars parked at home there, free for a trip
        self.inbound: dict[str, int] = {}         # area -> cars on their way to park there
        self.where: dict[str, str] = {}           # vehicle parked now -> the place it is at (its destination, home)
        self.home_pending: dict[str, tuple] = {}  # put in at home (prefill), not in its space yet -> (home, area)
        self.idle_t: dict[str, float] = {}         # parked at home since (SUMO seconds)
        # the viewer drives a car out of its space (cosim/parking_motion.py): exit_delay(vid, area) -> seconds before
        # SUMO may put it on the road; resume_at: those cars, and when
        self.exit_delay = None
        self.resume_at: dict[str, float] = {}
        self.n = 0
        self.park_side = PARK_SIDE
        self.bus_side_m = BUS_SIDE_M
        self._bus_edges = None
        self.stats = {"trips": {}, "reused": 0, "inserted": 0, "through": 0, "redirected_full": 0, "failed": 0,
                      "arrived_parked": 0, "by_hour": {}, "sides_changed": 0}

    # ---------------------------------------------------------------- the car parks
    def free(self, aid: str) -> int:
        return self.places.cap[aid] - sum(1 for a in self.parked.values() if a == aid) - self.inbound.get(aid, 0)

    def occupancy(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.parked.values():
            out[a] = out.get(a, 0) + 1
        return out

    def here(self) -> dict[str, int]:
        """place -> its cars parked now for it (people at work there, guests, cars at home)."""
        out: dict[str, int] = {}
        for v in self.parked:
            pid = self.where.get(v)
            if pid is not None:
                out[pid] = out.get(pid, 0) + 1
        return out

    def live(self, hour: float) -> dict:
        """What the viewer shows of the places now (cosim/place_hud.py, route_minimap.py): new dicts, safe to read
        from another thread."""
        return {"occ": self.occupancy(), "here": self.here(), "parked": frozenset(self.parked), "cap": self.places.cap, "park_of": self.places.park_of,
                "hour": hour}

    def area_for(self, pid: str) -> str | None:
        """A parking area of place pid with room, else the nearest other one with room (the car park is full)."""
        own = [a for a in self.places.park_of.get(pid, []) if self.free(a) > 0]
        if own:
            return self.rng.choice(own)
        mine = self.places.park_of.get(pid) or []
        if not mine:
            return None
        x0, y0 = self.places.xy[mine[0]]
        others = sorted((a for a in self.places.areas if a not in mine and self.free(a) > 0),
                        key=lambda a: math.hypot(self.places.xy[a][0] - x0, self.places.xy[a][1] - y0))
        if others:
            self.stats["redirected_full"] += 1
            return others[0]
        return None

    def bus_edges(self) -> set:
        """(13 Nov) The roads (one direction each) the bus lines drive along (scenarios/town05/town05.pt.json)."""
        if self._bus_edges is None:
            try:
                lines = json.loads((ROOT / "scenarios" / "town05" / "town05.pt.json").read_text(encoding="utf-8"))
                self._bus_edges = {e for v in lines.values() for e in (v if isinstance(v, list) else v.get("edges", []))}
            except (OSError, ValueError):
                self._bus_edges = set()
        return self._bus_edges

    def _len(self, edges) -> float:
        roads = self.places.router.roads
        return sum(roads[e].length for e in edges if e in roads) if edges else float("inf")

    def _sides(self, dest, a_dest, home, a_home, reuse, start, back, start_ok=None):
        """(12 Nov, PARK_SIDE "route") The destination's area and the home's (for the way back; a new car also
        starts there) that make the trip there and back shortest, among the place's areas with room. Only between
        a place's own areas (a car sent to another car park, its own full, keeps it); ties keep the draw."""
        pl, r = self.places, self.places.router
        own_d = [a for a in pl.park_of.get(dest, []) if self.free(a) > 0]
        D = own_d if a_dest in own_d else [a_dest]
        if home is None:
            H = [None]
        elif reuse:
            H = sorted({a for a in pl.park_of.get(home, []) if self.free(a) > 0} | {reuse[0]})
        else:
            H = [a for a in pl.park_of.get(home, []) if self.free(a) > 0
                 and (start_ok is None or start_ok(pl.areas[a]["edge"]))] or [a_home]
        if len(D) == 1 and len(H) == 1:
            return a_dest, a_home, start, back
        best, pick = None, (a_dest, a_home, start, back)
        for h in H:
            hs = pl.areas[h]["edge"] if h is not None else None
            s = start if h is None or reuse else hs
            b = back if h is None else hs
            for d in D:
                e = pl.areas[d]["edge"]
                if s == e and h is not None and not reuse:
                    continue                                   # (a trip from home to its own street's side)
                cost = self._len(r.best(s, e)) + (self._len(r.best(e, b)) if b != e else 0.0)
                cost += self.bus_side_m * ((e in self.bus_edges()) + (h is not None and hs in self.bus_edges()))
                key = (cost, d != a_dest, h != a_home)
                if best is None or key < best:
                    best, pick = key, (d, h, s, b)
        if pick[:2] != (a_dest, a_home):
            self.stats["sides_changed"] += 1
        return pick

    def dwell_s(self, mean_h: float, spread_h: float) -> float:
        h = max(0.15, self.rng.gauss(mean_h, spread_h))
        return h * 3600.0 / self.clock_speed

    # ---------------------------------------------------------------- the trips
    def purpose_at(self, hour: float, vclass: str) -> str:
        if vclass == "delivery":
            return "deliver"
        names = list(PURPOSES)
        return self.rng.choices(names, [PURPOSES[n][0](hour) + 1e-6 for n in names])[0]

    def launch(self, vid: str, type_id, vclass: str, hour: float, sinks, sources, start_ok=None) -> bool:
        """A new trip for car vid (a new car, or one parked at home). False: none possible now (the caller keeps the
        old way for it). type_id: the new car's type, or a function giving it (called only when a new car is put in:
        the incidents' type_for marks the car it is asked about)."""
        r = self.places.router
        purpose = self.purpose_at(hour, vclass)
        if purpose == "deliver":
            _, dwell, frm, to = DELIVERY
        else:
            _, dwell, frm, to = PURPOSES[purpose]
        home = self.places.home(self.rng) if frm == "home" else None
        dest = self.places.choose(self.rng, to, hour, exclude=home)
        if dest is None:
            return False
        a_dest = self.area_for(dest)
        if a_dest is None:
            return False
        dest_edge = self.places.areas[a_dest]["edge"]
        reuse = None
        if home is not None:
            now = c_now = self.c.simulation.getTime()
            for a in self.places.park_of.get(home, []):
                ready = [v for v in self.idle.get(a, []) if now - self.idle_t.get(v, -1e9) >= MIN_IDLE_S]
                if ready:
                    self.idle[a].remove(ready[-1])
                    reuse = (a, ready[-1])
                    break
            a_home = reuse[0] if reuse else self.area_for(home)
            if a_home is None:
                return False
            if not reuse and start_ok is not None and not start_ok(self.places.areas[a_home]["edge"]):
                return False                      # (the viewer: a new car does not pull out where you look)
            start = self.places.areas[a_home]["edge"]
            back = self.places.areas[a_home]["edge"]
        else:
            a_home = None
            start = self.rng.choice(sources)
            back = self.rng.choice(sinks)
        if self.park_side == "route":
            a_dest, a_home, start, back = self._sides(dest, a_dest, home, a_home, reuse, start, back,
                                                      start_ok if not reuse else None)
            dest_edge = self.places.areas[a_dest]["edge"]
        out = r.best(start, dest_edge)
        ret = r.best(dest_edge, back) if back != dest_edge else [dest_edge]
        if out is None or ret is None or (start == dest_edge and a_home is not None):
            if reuse:
                self.idle[reuse[0]].append(reuse[1])
            return False
        edges = out + ret[1:]
        c = self.c
        try:
            if reuse:
                vid = reuse[1]
                c.vehicle.setRoute(vid, edges)
                self.stats["reused"] += 1
            else:
                rid = f"pr{self.n}"
                self.n += 1
                c.route.add(rid, edges)
                if callable(type_id):
                    type_id = type_id()
                if a_home is not None:            # it pulls out at the place's door
                    ar = self.places.areas[a_home]
                    lane = int(ar["lane"].rsplit("_", 1)[1])
                    c.vehicle.add(vid, rid, typeID=type_id, depart="now", departPos=str(ar["start"]),
                                  departLane=str(lane), departSpeed="0")
                else:
                    c.vehicle.add(vid, rid, typeID=type_id, depart="now", departLane="best", departSpeed="max")
                self.stats["inserted"] += 1
            c.vehicle.setParkingAreaStop(vid, a_dest, duration=self.dwell_s(*dwell))
            # (its home stop is set once it is parked at the destination: its route starts on the same road, and a stop
            # set now would be made on that first passage, before the destination)
            if reuse:
                delay = self.exit_delay(vid, reuse[0]) if self.exit_delay is not None else 0.0
                if delay > 0:                     # (the viewer drives it out of its space first)
                    self.resume_at[vid] = c_now + delay
                else:
                    c.vehicle.resume(vid)
        except Exception as e:  # noqa: BLE001 - a car that cannot have it keeps the old way
            log.debug("trip for %s failed: %s", vid, e)
            self.stats["failed"] += 1
            if reuse:
                self.idle[reuse[0]].append(reuse[1])
            return False
        plan = Plan(vid, purpose, home, dest, home, [a_dest] + ([a_home] if a_home else []))
        self.plans[vid] = plan
        self.inbound[a_dest] = self.inbound.get(a_dest, 0) + 1
        # (a car taken out of its space stays `parked` until SUMO puts it on the road: step(), parking ending)
        self.stats["trips"][purpose] = self.stats["trips"].get(purpose, 0) + 1
        hb = self.stats["by_hour"].setdefault(str(int(hour) % 24), {})
        hb[purpose] = hb.get(purpose, 0) + 1
        return True

    def prefill_one(self, vid: str, type_id, aid: str, hour: float) -> bool:
        """Car vid put straight into a space of area aid, as the town would have it at `hour` (the viewer's start, a
        run's start): at home there (idle until a trip takes it out), or at the place it came to, going home after
        what is left of its time there."""
        if self.free(aid) <= 0:
            return False
        rng, pl = self.rng, self.places
        homes, dests = pl.homes_at.get(aid), pl.dests_at.get(aid)
        at_home = bool(homes) and (not dests or not 7.0 <= hour % 24 < 19.0)
        edge = pl.areas[aid]["edge"]
        if at_home:
            home = rng.choices([h[0] for h in homes], [h[1] for h in homes])[0]
            route, plan = [edge], None
        elif dests:
            dest = rng.choices([d[0] for d in dests], [d[1] for d in dests])[0]
            purpose = KIND_PURPOSE[pl.places[dest]["kind"]]
            home = pl.home(rng)
            a_home = self.area_for(home)
            if a_home and self.park_side == "route":   # (12 Nov) home on the side its way back reaches first
                own = [a for a in pl.park_of.get(home, []) if self.free(a) > 0]
                if a_home in own and len(own) > 1:
                    a_home = min(own, key=lambda a: (self._len(pl.router.best(edge, pl.areas[a]["edge"]))
                                                     + self.bus_side_m * (pl.areas[a]["edge"] in self.bus_edges()),
                                                     a != a_home))
            ret = pl.router.best(edge, pl.areas[a_home]["edge"]) if a_home else None
            if ret is None:
                return False
            mean_h, spread_h = PURPOSES[purpose][1]
            dwell = max(0.1, rng.uniform(0.15, 1.0) * rng.gauss(mean_h, spread_h)) * 3600.0 / self.clock_speed
            route, plan = ret, Plan(vid, purpose, home, dest, home, [aid, a_home])
        else:
            return False
        c = self.c
        try:
            rid = f"pf{self.n}"
            self.n += 1
            c.route.add(rid, route)
            c.vehicle.add(vid, rid, typeID=type_id() if callable(type_id) else type_id, depart="now",
                          departPos="stop")
            c.vehicle.setParkingAreaStop(vid, aid, duration=OPEN_STOP_S if plan is None else dwell)
        except Exception as e:  # noqa: BLE001
            log.debug("prefill %s at %s: %s", vid, aid, e)
            return False
        if plan is None:            # idle at home once SUMO has it in its space (step: a trip may take it out then)
            self.home_pending[vid] = (home, aid)
            self.inbound[aid] = self.inbound.get(aid, 0) + 1
        else:
            self.plans[vid] = plan
            self.inbound[aid] = self.inbound.get(aid, 0) + 1
            self.stats["arrived_parked"] -= 1             # (its parking now is not an arrival)
        self.stats["prefilled"] = self.stats.get("prefilled", 0) + 1
        return True

    def holds(self, vid: str) -> bool:
        """Its route is its trip's, or it is parked (the viewer's keep_driving leaves it alone)."""
        return vid in self.plans or vid in self.parked or vid in self.home_pending

    def next_stop(self, vid: str) -> str | None:
        p = self.plans.get(vid)
        return p.areas[0] if p and p.areas else None

    # ---------------------------------------------------------------- every step
    def step(self, arrived=()) -> None:
        c = self.c
        for vid in c.simulation.getParkingStartingVehiclesIDList():
            p = self.plans.get(vid)
            try:
                aid = c.vehicle.getStops(vid, 1)[0].stoppingPlaceID
            except Exception:  # noqa: BLE001
                aid = None
            if not aid:
                continue
            self.parked[vid] = aid
            hp = self.home_pending.pop(vid, None)
            if hp is not None:                        # (prefill) at home: idle until a trip takes it out
                self.inbound[hp[1]] = max(0, self.inbound.get(hp[1], 0) - 1)
                self.idle.setdefault(aid, []).append(vid)
                self.idle_t[vid] = c.simulation.getTime() - MIN_IDLE_S     # (put there at the start: free at once)
                self.where[vid] = hp[0]
                continue
            if p is None or not p.areas or p.areas[0] != aid:    # (its first stop: pulling out of its home space)
                continue
            p.areas.pop(0)
            self.inbound[aid] = max(0, self.inbound.get(aid, 0) - 1)
            self.stats["arrived_parked"] += 1
            self.where[vid] = p.dest if p.areas else (p.home or p.dest)
            if p.areas:                               # back home next: that space is now taken for it
                self.inbound[p.areas[0]] = self.inbound.get(p.areas[0], 0) + 1
                try:
                    c.vehicle.setParkingAreaStop(vid, p.areas[0], duration=OPEN_STOP_S)
                except Exception as e:  # noqa: BLE001 - it will end its trip at home's door instead
                    log.debug("home stop for %s: %s", vid, e)
                    self.inbound[p.areas[0]] -= 1
                    p.areas.clear()
            else:                                     # parked at home (or out of town): idle until a trip takes it
                self.plans.pop(vid, None)
                self.idle.setdefault(aid, []).append(vid)
                self.idle_t[vid] = c.simulation.getTime()
        if self.resume_at:                            # driven out of their space by the viewer: SUMO's turn
            now = c.simulation.getTime()
            for vid in [v for v, t in self.resume_at.items() if now >= t]:
                del self.resume_at[vid]
                try:
                    c.vehicle.resume(vid)
                except Exception as e:  # noqa: BLE001
                    log.debug("resume %s: %s", vid, e)
        for vid in c.simulation.getParkingEndingVehiclesIDList():
            self.parked.pop(vid, None)
            self.where.pop(vid, None)
        self._k = getattr(self, "_k", 0) + 1
        if self._k % 20 == 0:                         # cars taken off the roads by others (not "arrived")
            alive = set(c.vehicle.getIDList())
            pending = set(c.simulation.getPendingVehicles())
            arrived = list(arrived) + [v for v in set(self.parked) | set(self.plans) | set(self.home_pending)
                                       if v not in alive and v not in pending]
        for vid in arrived:
            hp = self.home_pending.pop(vid, None)
            if hp is not None:
                self.inbound[hp[1]] = max(0, self.inbound.get(hp[1], 0) - 1)
            p = self.plans.pop(vid, None)
            if p is not None:
                for a in p.areas[:1]:
                    self.inbound[a] = max(0, self.inbound.get(a, 0) - 1)
            self.where.pop(vid, None)
            a = self.parked.pop(vid, None)
            if a is not None and vid in self.idle.get(a, []):
                self.idle[a].remove(vid)


def purpose_population(base_cls):
    """A Population class (cosim/live.Population or the viewer's ViewAwarePopulation) whose new trips are mostly
    trips with a purpose. Set `demand` (a PurposeDemand) and `sources` / `sinks` (road ids) before its first step."""

    class PurposePopulation(base_cls):
        demand: PurposeDemand | None = None
        hour = 8.0
        sources: tuple = ()
        sinks: tuple = ()

        def add(self, k: int, anywhere: bool = False) -> None:
            d = self.demand
            if d is None or anywhere:
                return super().add(k, anywhere)
            for _ in range(k):
                t = self.rng.choices(self.types, self.weights)[0]
                vclass = (self.pool.vclass.get(t) if hasattr(self.pool, "vclass") else None) or _vclass(self.pool, t)
                if vclass not in PURPOSE_CLASSES or self.rng.random() < d.through_share:
                    d.stats["through"] += 1
                    super().add(1, anywhere=False)
                    continue
                vid = f"live{self.n}"
                if d.launch(vid, lambda vid=vid, t=t: self.typed(vid, t), vclass, self.hour, self.sinks, self.sources,
                            start_ok=getattr(self, "start_ok", None)):
                    if vid in d.plans:            # a new car (not one taken out of its home car park)
                        self.n += 1
                        self.added += 1
                else:
                    super().add(1, anywhere=False)

        def prefill(self, hour: float, table_path: Path = TABLE) -> int:
            """The town's car parks and homes as they are at `hour` on a measured day (TABLE: the cars parked in
            each area, hour by hour, on a training seed's day): those cars put in their spaces now."""
            d = self.demand
            if d is None or not Path(table_path).is_file():
                return 0
            table = json.loads(Path(table_path).read_text(encoding="utf-8"))["hours"]
            counts = table.get(str(int(hour) % 24), {})
            cars = [(t, w) for t, w in zip(self.types, self.weights)
                    if ((self.pool.vclass.get(t) if hasattr(self.pool, "vclass") else None) or _vclass(self.pool, t))
                    == "passenger"]
            n = 0
            for aid, k in sorted(counts.items()):
                if aid not in d.places.areas:
                    continue
                for _ in range(round(k)):
                    t = self.rng.choices([c[0] for c in cars], [c[1] for c in cars])[0]
                    vid = f"live{self.n}"
                    if d.prefill_one(vid, lambda vid=vid, t=t: self.typed(vid, t), aid, hour):
                        self.n += 1
                        n += 1
            return n

        def step(self, running: dict, seconds_of_day: float, may_remove) -> float:
            d = self.demand
            self.hour = seconds_of_day / 3600.0
            if d is not None:
                moving = {v: x for v, x in running.items() if v not in d.parked}
                return super().step(moving, seconds_of_day,
                                    lambda vid, v: may_remove(vid, v) and vid not in d.plans)
            return super().step(running, seconds_of_day, may_remove)

    return PurposePopulation


def _vclass(pool, t: str) -> str:
    import re

    vt = getattr(pool, "vtypes", {}).get(t, "")
    m = re.search(r'vClass="([^"]+)"', vt)
    return m.group(1) if m else "passenger"

"""A trip's time left, updated on the way as a navigation app does it (10 Nov), beside the estimate made at the start
(cosim/trip_estimates.py, never changed).

What a navigation service has, and nothing else: the road times of its users' past trips (the start estimate's table)
and, live, its users' phones: a share of the town's cars run the app (`ProbeTraffic`, chosen by a hash of the car's
id) and report the roads they drive. From them, each road's live time: the median time of the app cars that left it in
the last 2 minutes (Google Maps' real-time input: speeds of the last 2 minutes, Derrow-Pinion et al., "ETA Prediction
with Graph Neural Networks in Google Maps", CIKM 2021), shrunk toward its usual time when few cars give it.
SUMO's truth only grades it.

The time left (`time_left`): from where the car is, each road ahead takes its live time where the app has one and its
usual time (the table's median) otherwise, blended by how far ahead it is: weight exp(-tau / blend_s) for the live
time, tau the time until the car gets there (live traffic says much about the next minute, less about the next ten:
the paper's real-time sum beats the historical one at horizon 0 and loses to it at an hour). The same with no live
time is "usual traffic", the delay against it colours the time left (green, orange, red) and each road ahead is
coloured by its live time against its usual one (green, orange, red, dark red), as Google Maps colours a route.
"""
from __future__ import annotations

import math
import statistics as st
import zlib
from collections import deque

WINDOW_S = 120.0          # a road's live time: app cars that left it in the last 2 minutes
SHARE = 0.25              # cars running the app (tools/eval_trip_estimates.py measures 0.10, 0.25, 0.50)
BLEND_S = 120.0           # live weight exp(-tau / BLEND_S), tau the time until the car reaches the road (measured)
SHRINK = 10.0             # a road's live time from n app cars: (n * live + SHRINK * usual) / (n + SHRINK). Dev seeds
                          # 11-20 (30 runs, 58 022 checkpoints of the time left): with usual times only, mean error
                          # 44.7 s; app 25 %, blend 120 s, shrink 10: 44.3 s, better in 25 of 30 runs; shrink 3: 44.3,
                          # 21 of 30; no shrink 47.0 (worse); app 10 % 44.5, 50 % 44.1 (docs/WORKLOG.md 10 Nov)
APP_PREFIXES = ("live", "hw")   # the town's cars and the ring road's; not buses, trucks, emergency or incident stand-ins
_ROAD = 0x50              # VAR_ROAD_ID

# colours (our thresholds: Google does not publish its own): a road by live / usual time, the trip by its delay
ROAD_LEVELS = (1.25, 2.0, 3.5)       # below: green; orange; red; above the last: dark red
DELAY_ORANGE = (0.15, 20.0)          # delay over 15 % of the usual time left and over 20 s: orange
DELAY_RED = (0.40, 45.0)             # over 40 % and over 45 s: red


def uses_app(vid: str, share: float) -> bool:
    """The same cars at any share: those at 10 % also run it at 25 % and 50 %."""
    return vid.startswith(APP_PREFIXES) and zlib.crc32(vid.encode()) % 1000 < share * 1000


class ProbeTraffic:
    """The app's live view of the roads, from its users' cars only."""

    def __init__(self, share: float = SHARE, window_s: float = WINDOW_S):
        self.share, self.window_s = share, window_s
        self.on: dict[str, list] = {}        # app car -> [road, entered at (None: started on it), halted on it]
        self.recent: dict[str, deque] = {}   # road -> (left at, seconds on it) of app cars, last window_s
        self.users: set[str] = set()
        self.others: set[str] = set()        # cars without the app (asked once)

    def step(self, c, now: float, arrived=(), subs: dict | None = None) -> None:
        """After each SUMO step. subs: the vehicles' subscription results (their road), read again if not given."""
        if subs is None:
            subs = c.vehicle.getAllSubscriptionResults()
        try:
            halted = set(c.simulation.getParkingStartingVehiclesIDList()) | \
                set(c.simulation.getStopStartingVehiclesIDList())
        except Exception:  # noqa: BLE001
            halted = set()
        for vid, r in subs.items():
            if vid not in self.users:
                if vid in self.others:
                    continue
                if not uses_app(vid, self.share):
                    self.others.add(vid)
                    continue
                self.users.add(vid)
            road = r.get(_ROAD, "")
            s = self.on.get(vid)
            if s is None:
                self.on[vid] = [road if road and not road.startswith(":") else "", None, False]
                continue
            if vid in halted:
                s[2] = True
            if not road:                       # teleporting
                s[:] = ["", None, False]
            elif road.startswith(":"):
                if s[0] and len(s) == 3:
                    s.append(now)              # left its road: the time it left
            elif road != s[0]:
                t_out = s[3] if len(s) > 3 else now
                if s[0] and s[1] is not None and not s[2]:
                    self.recent.setdefault(s[0], deque()).append((t_out, t_out - s[1]))
                s[:] = [road, t_out if s[0] else None, False]
            elif len(s) > 3:
                del s[3:]
        for vid in arrived:
            self.on.pop(vid, None)
            self.users.discard(vid)
            self.others.discard(vid)
        lim = now - self.window_s
        for q in self.recent.values():
            while q and q[0][0] < lim:
                q.popleft()

    def live_time(self, road: str, now: float, usual: float | None = None, shrink: float = 0.0) -> float | None:
        """The road's live time (s): the median of the app cars that left it in the window (shrunk toward the usual
        time by how few they are: (n * median + shrink * usual) / (n + shrink); one car caught by a red light is not
        the road's traffic); None when the app knows nothing. (Tried, not kept: raising it to how long app cars
        standing on it now have been there when that was over twice the usual time and a minute more: on a dev run
        the time left was worse, mean error 63.7 s against 54.3 s with usual times: those queues clear, or the cars
        behind are sent around them.)"""
        q = self.recent.get(road)
        t = st.median(d for _, d in q) if q else None
        if t is not None and usual is not None and shrink > 0:
            t = (len(q) * t + shrink * usual) / (len(q) + shrink)
        return t


def level(live: float | None, usual: float) -> int | None:
    """0 green, 1 orange, 2 red, 3 dark red; None: no live data."""
    if live is None or usual <= 0:
        return None
    r = live / usual
    return sum(r >= x for x in ROAD_LEVELS)


def delay_level(left: float, usual: float) -> int:
    d = left - usual
    if d > max(DELAY_RED[0] * usual, DELAY_RED[1]):
        return 2
    if d > max(DELAY_ORANGE[0] * usual, DELAY_ORANGE[1]):
        return 1
    return 0


def time_left(te, probes: ProbeTraffic | None, roads: list[str], pos: float, end_pos: float, now: float,
              blend_s: float = BLEND_S, shrink: float = SHRINK) -> dict:
    """Time left (s) driving `roads` (the one it is on first, up to its trip's end on the last) from pos on the first,
    with the app's live times blended in (probes None or blend_s 0: usual times only). te: TripEstimates (its table,
    lengths and limits). Returns {"left_s", "usual_s", "roads": [(road, level or None)], "known": share of the time
    left with a live time}."""
    if not roads:
        return {"left_s": 0.0, "usual_s": 0.0, "roads": [], "known": 0.0}
    if len(roads) == 1:
        s = max(0.0, end_pos - pos) / max(te.speed.get(roads[0], 13.89), 0.1)
        return {"left_s": s, "usual_s": s, "roads": [(roads[0], None)], "known": 0.0}
    left = usual_left = tau = known = 0.0
    colours = []
    for i in range(len(roads) - 1):
        a, b = roads[i], roads[i + 1]
        fa = te.free(a)
        usual = te.typical.get((a, b), fa)
        live = probes.live_time(a, now, usual, shrink) if probes is not None else None
        colours.append((a, level(live, usual)))
        if i == 0:                             # from where it is: its drive scaled, the wait at its signal whole
            frac = max(0.0, 1.0 - pos / max(te.len.get(a, 1.0), 1.0))
            usual = fa * frac + max(0.0, usual - fa)
            if live is not None:
                live = fa * frac + max(0.0, live - fa)
        w = math.exp(-tau / blend_s) if (live is not None and blend_s > 0) else 0.0
        t = w * live + (1.0 - w) * usual if live is not None else usual
        left += t
        usual_left += usual
        known += w * t
        tau += t
    last = roads[-1]
    tail = min(end_pos, te.len.get(last, end_pos)) / max(te.speed.get(last, 13.89), 0.1)
    colours.append((last, None))
    return {"left_s": left + tail, "usual_s": usual_left + tail, "roads": colours,
            "known": known / (left + tail) if left + tail > 0 else 0.0}


def roads_ahead(te, c, vid: str, leg) -> list[str] | None:
    """The roads from the one car vid is on to its trip's end, on its route in SUMO now. A ring car's route goes round
    200 times (cosim/highway.py): its trip's roads left come from its plan, not from where its last road next shows up
    in the route (that took the nearest lap: the time left 83 s short, 11 Nov). None: its end is off its way."""
    route, idx = list(c.vehicle.getRoute(vid)), max(c.vehicle.getRouteIndex(vid), 0)
    end = te._end(vid) if getattr(leg, "end_v", None) is not None else None
    if end is not None:
        roads = route[idx:idx + max(1, end[2])]
        return roads if roads and roads[-1] == leg.dest_edge else None
    try:
        j = route.index(leg.dest_edge, idx)
    except ValueError:
        return None
    return route[idx:j + 1]


def for_car(te, probes: ProbeTraffic | None, c, vid: str, now: float, subs: dict | None = None,
            blend_s: float = BLEND_S, shrink: float = SHRINK) -> dict | None:
    """The time left of car vid's trip (TripEstimates te, its active leg) from where it is now, the app's live times
    in: {"left_s", "usual_s", "delay_level", "roads", "known", "left_m", "at"}; None when it is on a junction, has
    no trip, or its trip's end is no longer on its way."""
    leg = te.active.get(vid)
    if leg is None:
        return None
    r = (subs or {}).get(vid) or {}
    road, pos = r.get(_ROAD), r.get(0x56)
    try:
        if road is None or pos is None:
            road, pos = c.vehicle.getRoadID(vid), c.vehicle.getLanePosition(vid)
        if not road or road.startswith(":"):
            return None
        roads = roads_ahead(te, c, vid, leg)
    except Exception:  # noqa: BLE001 - gone
        return None
    if not roads or roads[0] != road:
        return None
    out = time_left(te, probes, roads, pos, leg.dest_pos, now, blend_s, shrink)
    if len(roads) == 1:
        m = max(0.0, leg.dest_pos - pos)
    else:
        m = max(0.0, te.len.get(roads[0], 0.0) - pos) + sum(te.len.get(e, 0.0) for e in roads[1:-1]) + \
            min(leg.dest_pos, te.len.get(roads[-1], leg.dest_pos))
    out.update(left_m=round(m), at=now, delay_level=delay_level(out["left_s"], out["usual_s"]))
    return out

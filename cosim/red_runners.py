"""Drivers who run the end of the amber (31 Oct): the commonest red-light offence in real life.

The careless drivers of cosim/incidents.py look away and run a red now and then, but that is rare (a few a half hour
in the whole town, most of them crashes). What a red-light camera mostly catches is something else: a driver who
sees the light turn amber when stopping would mean braking hard (the "dilemma zone") and keeps going, reaching the
stop line late in the amber or in the first second or two of red.

Here a share of the private cars (GAMBLER_SHARE, chosen once per car by its id, so the same car always drives the
same way) are such drivers. When their next signal turns amber and the stop line is GO_T_S seconds or less ahead at
their speed, they decide once: with GO_P they keep going (SUMO's checks for red lights and right of way off, speed
kept), else they stop as everybody does. The same drivers, coming up to a signal that has just turned red (under
LATE_RED_S ago) with the line LATE_T_S or less ahead, go on with LATE_P: they saw the change too late. Whether they reach the line on amber or on red depends only on where they
were when it changed: nothing is placed or timed. Emergency vehicles, buses, the accidents' vehicles and the careless
drivers already looking away are left alone. Once they are through the junction, their normal driving is back.

`RedLightTruth` is the simulation's own record of every vehicle that enters a signalised junction from a watched road
(SUMO alone: the time, the signal it had, how long that had been red): the ground truth a red-light camera's
decisions are graded against (cosim/enforcement.py). It is never shown to the camera's agent.
"""

from __future__ import annotations

import logging
import random
import zlib
from dataclasses import dataclass, field

log = logging.getLogger("red_runners")

GAMBLER_SHARE = 0.10      # of private cars
GO_T_S = 5.0              # the stop line at most this many seconds ahead when the amber shows
MIN_T_S = 1.2             # nearer than this, everybody goes on (stopping is not possible): not a decision
GO_P = 0.8                # a gambler's chance to go on
MIN_SPEED_MS = 6.0
LATE_RED_S = 2.0          # a red this young ...
LATE_T_S = 2.5            # ... with the line this close (s at their speed) ...
LATE_P = 0.5              # ... is run with this chance
NEVER = ("emergency", "authority", "bus", "coach")


def is_gambler(vid: str, share: float = GAMBLER_SHARE) -> bool:
    return zlib.crc32(vid.encode()) % 10000 < share * 10000


class RedRunners:
    """Runs in the SUMO thread after each step (see the module doc). busy(vid) -> True: leave this one alone (an
    accident's vehicle, a careless driver looking away)."""

    def __init__(self, conn, share: float = GAMBLER_SHARE, seed: int = 5, busy=None):
        self.conn = conn
        self.share = share
        self.rng = random.Random(seed)
        self.busy = busy or (lambda vid: False)
        self.decided: dict[str, tuple] = {}   # vid -> (tls, link, went?)
        self.going: dict[str, float] = {}     # vid -> when it went on (its red-light checks off)
        self.stats = {"decisions": 0, "went": 0, "stopped": 0, "late": 0}
        self._red_since: dict = {}            # (tls, link) -> when it turned red
        self._prev: dict = {}

    def step(self, now: float, vehicles: dict, tls_states: dict | None = None) -> None:
        c = self.conn
        for t, st in (tls_states or {}).items():
            pv = self._prev.get(t, "")
            for i, ch in enumerate(st):
                if ch in "rR" and (i >= len(pv) or pv[i] not in "rR"):
                    self._red_since[(t, i)] = now
            self._prev[t] = st
        for vid in list(self.decided):
            if vid not in vehicles:
                self.decided.pop(vid, None)
                self.going.pop(vid, None)
        for vid, v in vehicles.items():
            if v.vclass in NEVER or vid.startswith("pt_") or not is_gambler(vid, self.share):
                continue
            if vid in self.going:
                try:
                    lane = c.vehicle.getLaneID(vid)
                except Exception:  # noqa: BLE001
                    continue
                d = self.decided.get(vid)
                if not lane.startswith(":") and now - self.going[vid] > 1.0:
                    nt = self._next(vid)
                    if not nt or (d and (nt[0], nt[1]) != d[:2]) or now - self.going[vid] > 15.0:
                        self._normal(vid)  # through the junction (or it never got there): drives as before
                continue
            if v.speed < MIN_SPEED_MS or self.busy(vid):
                continue
            nt = self._next(vid)
            if not nt:
                continue
            tls, link, dist, state = nt
            d = self.decided.get(vid)
            if d is not None and d[:2] == (tls, link):
                continue  # decided for this signal already
            t = dist / max(v.speed, 0.1)
            if state in "rR":  # just turned red: seen too late?
                age = now - self._red_since.get((tls, link), -1e9)
                if age > LATE_RED_S or not 0.3 < t <= LATE_T_S:
                    continue
                go = self.rng.random() < LATE_P
                if go:
                    self.stats["late"] += 1
            elif state in "yY":
                if not MIN_T_S < t <= GO_T_S:
                    continue
                go = self.rng.random() < GO_P
            else:
                continue
            self.decided[vid] = (tls, link, go)
            self.stats["decisions"] += 1
            if not go:
                self.stats["stopped"] += 1
                continue
            try:
                c.vehicle.setSpeedMode(vid, 0b00111)  # safe gap to the car ahead kept; red and right of way not
            except Exception:  # noqa: BLE001
                continue
            self.going[vid] = now
            self.stats["went"] += 1
            log.debug("%s goes on through the amber at %s (%.0f m, %.1f s)", vid, tls, dist, t)

    def _next(self, vid: str):
        try:
            nt = self.conn.vehicle.getNextTLS(vid)
        except Exception:  # noqa: BLE001
            return None
        return nt[0] if nt else None

    def _normal(self, vid: str) -> None:
        self.going.pop(vid, None)
        try:
            self.conn.vehicle.setSpeedMode(vid, 31)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class Entry:
    """One vehicle entering a watched junction: SUMO's own record."""
    t: float            # the step it was first seen inside the junction (SUMO seconds)
    vid: str
    tls: str
    lane: str           # the road's lane it came from
    link: int
    state: str          # its signal at that step: G g y r ...
    red_s: float        # how long that signal had been red (0 when not red)
    speed: float
    vclass: str
    emergency: bool     # on a run (siren and beacons on)


@dataclass
class RedLightTruth:
    """Every vehicle entering the watched junctions from the watched roads (SUMO alone; module doc). watch:
    {tls: {incoming lane ids}}; on_run(vid) -> True for an emergency vehicle on a run."""
    conn: object
    watch: dict
    on_run: object = None
    entries: list = field(default_factory=list)
    _last: dict = field(default_factory=dict)       # vid -> (tls, link, lane) while on a watched lane
    _red_since: dict = field(default_factory=dict)  # (tls, link) -> when it turned red
    _prev: dict = field(default_factory=dict)       # tls -> its state the step before
    step_s: float = 0.5

    def step(self, now: float, vehicles: dict, tls_states: dict) -> list[Entry]:
        """After each SUMO step. SUMO switches its signals at the start of a step and then moves the vehicles, so the
        state read after the step to t is the one they moved under from t - step to t. A vehicle seen on its road at
        t0 (d metres from the line at speed v) and inside the junction at t crossed the line at about t0 + d / v
        (at most t), under the state read at t."""
        for t in self.watch:
            st = tls_states.get(t, "")
            pv = self._prev.get(t, "")
            for i, ch in enumerate(st):
                if ch in "rR" and (i >= len(pv) or pv[i] not in "rR"):
                    self._red_since[(t, i)] = now
            self._prev[t] = st
        lanes = {ln: t for t, ls in self.watch.items() for ln in ls}
        c = self.conn
        new = []
        for vid, v in vehicles.items():
            try:
                ln = c.vehicle.getLaneID(vid)
            except Exception:  # noqa: BLE001
                continue
            if ln in lanes:
                try:
                    nt = c.vehicle.getNextTLS(vid)
                except Exception:  # noqa: BLE001
                    nt = []
                if nt and nt[0][0] == lanes[ln]:
                    t, link, dist, _ch = nt[0]
                    self._last[vid] = (t, link, ln, now, dist, v.speed)
                continue
            prev = self._last.pop(vid, None)
            if prev is None or not ln.startswith(":"):
                continue
            t, link, from_lane, t0, dist, speed = prev
            if dist > max(speed, v.speed) * (now - t0) + 3.0:
                continue  # (far from the line a step ago: another junction's lane, or moved by the scene)
            t_cross = min(now, t0 + dist / max(speed, 0.1)) if speed > 0.1 else now
            st = tls_states.get(t, "")
            ch = st[link] if link < len(st) else "?"
            # red in effect from the step before it was first read (see above)
            red_s = max(0.0, t_cross - (self._red_since.get((t, link), now) - self.step_s)) if ch in "rR" else 0.0
            e = Entry(round(t_cross, 2), vid, t, from_lane, link, ch, round(red_s, 2), round(v.speed, 2), v.vclass,
                      bool(self.on_run(vid)) if self.on_run else False)
            self.entries.append(e)
            new.append(e)
        for vid in [x for x in self._last if x not in vehicles]:
            self._last.pop(vid, None)
        return new

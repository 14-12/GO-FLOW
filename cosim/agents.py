"""Step 3, version 1 (kept for the comparison): one max-pressure agent per signalised junction.

The agents used now are version 2, cosim/agents_actuated.py. Version 1 replaced the actuated program
with max-pressure and lost to it (x0.5: 125 s waiting against 77 s; x0.8: gridlock).

Each agent is walk-aware max-pressure with pre-emption (cosim/controllers.py), plus what step 1
showed it lacked:

1. Switching cost. Plain max-pressure changed phase about every 16 s per junction and lost 6 s of
   yellow and all-red each time. Here the current green counts `switch_gain` vehicles more than it
   is, and the minimum green is 10 s, so a change has to be worth its lost time.
2. Spillback (the neighbour's message). A movement whose exit stretch is nearly full
   (`full_share` of its storage, 7.5 m a vehicle) gets no pressure: a green there moves nobody and
   blocks the junction box. In a city the downstream agent says "my entry is full"; in SUMO both
   agents read the same lanes, so the message is the lane count itself.
3. Emergency corridor. A vehicle on a run asks its next signal within `detect_m` (150 m), as
   before, and that signal passes the request on to the following one when the vehicle is within
   `corridor_m` (300 m) of it, so the second green is ready when it arrives.
4. Fallback. Where cameras see too little (incoming coverage under 70 %: 2014 and 2328), the agent
   hands its junction back to the actuated program.
"""

from __future__ import annotations

import json
from pathlib import Path

from cosim.controllers import PriorityMaxPressure, WalkAware

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "out" / "town05_cameras" / "coverage.json"
VEH_M = 7.5


class AgentMixin:
    switch_gain = 4.0
    full_share = 0.85
    corridor_m = 300.0
    min_coverage = 0.70

    def __init__(self, fallback: set[str] | None = None, **kw):
        kw.setdefault("min_green", 10.0)
        super().__init__(**kw)
        if fallback is None:
            fallback = set()
            if COVERAGE.is_file():
                cov = json.loads(COVERAGE.read_text(encoding="utf-8"))["junctions"]
                fallback = {t for t, v in cov.items() if (v["in_share"] or 0) < self.min_coverage}
        self.fallback = fallback
        self.handed_back: list[str] = []
        self.corridor_requests = 0

    def setup(self, conn) -> None:
        super().setup(conn)
        tc = conn.tc
        for tls in sorted(self.fallback & set(self.junctions)):
            if any(lg.programID == "actuated" for lg in tc.trafficlight.getAllProgramLogics(tls)):
                tc.trafficlight.setProgram(tls, "actuated")
                del self.junctions[tls]
                self.handed_back.append(tls)

    def _capacity(self, lanes) -> float:
        return sum(self._len.get(x, 0.0) for x in lanes) / VEH_M

    def pressure(self, tc, j, k, counts) -> float:
        total = 0.0
        for a, b in j.pairs[k]:
            out = self._reach(b, upstream=False)
            n_out = sum(self._n(tc, x, counts) for x in out)
            cap = self._capacity(out)
            if cap > 0 and n_out >= self.full_share * cap:
                continue  # exit full: a green here moves nobody
            total += self.queue_in(tc, a, counts) - n_out
        total += self.ped_weight * self.peds(j, k)  # WalkAware's part (its pressure() is bypassed here)
        if j.mode == "green" and k == j.current:
            total += self.switch_gain
        return total

    def step(self, conn) -> None:
        tc = conn.tc
        super().step(conn)  # requests for the next signal, then the decisions
        # corridor: the signal after the next prepares while the vehicle is within corridor_m;
        # applied from the next step on (requests are rebuilt every step)
        extra: dict[str, set[int]] = {}
        wanted = set(self.emergency())
        for vid in (wanted & set(tc.vehicle.getIDList())) if wanted else ():
            upcoming = tc.vehicle.getNextTLS(vid)
            if len(upcoming) < 2:
                continue
            tls, link, dist, _ = upcoming[1]
            if tls in self.junctions and dist <= self.corridor_m:
                extra.setdefault(tls, set()).add(int(link))
        self._corridor = extra

    def _decide(self, tc, j, now, counts) -> None:
        extra = getattr(self, "_corridor", {}).get(j.tls)
        if extra:
            self.requests.setdefault(j.tls, set()).update(extra)
            self.corridor_requests += 1
        super()._decide(tc, j, now, counts)


class JunctionAgents(AgentMixin, WalkAware, PriorityMaxPressure):
    name = "agents_v1"


class ObservedJunctionAgents(AgentMixin, WalkAware, PriorityMaxPressure):
    """The agents on the cameras' counts (cosim/camera_reports.py) instead of SUMO's."""

    name = "agents_v1_observed"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.observed: dict[str, int] = {}

    def _n(self, tc, lane, counts) -> int:
        return self.observed.get(lane, 0)

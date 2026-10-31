"""Watches the signals and the road users of a running SUMO simulation, step by step (cosim/signal_audit.py checks
the program files; this checks what the simulation actually showed and did, controller and agents included).

Per signal, from the states shown each step and SUMO's own junction logic (who conflicts with whom, who gives way):

* conflicting_greens   two conflicting links green at the same step, neither giving way to the other
* short_intergreen     a link turned green less than MIN_YELLOW_S + MIN_ALLRED_S after a conflicting vehicle link's
                       green ended (a conflicting link already green with it, a permissive turn, is not a release)
* green_to_red         a vehicle link went from green to red without its yellow
* ped_short_clearance  a crossing's conflicting vehicles were released less than its pedestrian clearance time
                       (length / 3.5 ft/s) after its walk ended while someone was still on it
* ped_on_crossing_at_release  people on a crossing at the step a vehicle link conflicting with it turned green
* ped_started_on_red   someone stepped onto a crossing whose signal was not walk
* carla_pole_disagree  a CARLA pole (linkSignalID landmark) showing links in two colours at once (it has one
                       state; cosim/convert.tl_state would show the most restrictive)

and the service: for every vehicle link with vehicles waiting, the longest a vehicle stood at its red; for every
crossing, the longest anyone waited at it (`max_wait`), which shows a movement or crossing left waiting for good.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cosim.signal_audit import MIN_ALLRED_S, MIN_YELLOW_S, PED_CLEAR_SPEED, load

GREEN = "Gg"


class SignalMonitor:
    def __init__(self, net_file: Path):
        self.j = load(Path(net_file))
        self.counts: Counter = Counter()
        self.examples: dict[str, list] = {}
        self._prev: dict[str, str] = {}
        self._green_end: dict[tuple[str, int], float] = {}
        self._yellow_seen: dict[tuple[str, int], bool] = {}
        self._cross_of: dict[str, tuple[str, int]] = {}
        for tid, jn in self.j.items():
            for lk in jn.links:
                if lk.kind == "crossing":
                    self._cross_of[lk.crossing] = (tid, lk.index)
        self._person_road: dict[str, str] = {}
        self.max_wait: dict[str, float] = {}       # "tls:link" or crossing id -> longest wait seen (s)
        self.steps = 0

    def _flag(self, kind: str, *info) -> None:
        self.counts[kind] += 1
        ex = self.examples.setdefault(kind, [])
        if len(ex) < 5:
            ex.append(info)

    def step(self, c, now: float, tls: dict[str, str]) -> None:
        """c: the TraCI connection; tls: the states shown this step (tls id -> state)."""
        self.steps += 1
        on_crossing: Counter = Counter()
        roads = {}
        for pid in c.person.getIDList():
            road = c.person.getRoadID(pid)
            roads[pid] = road
            if road in self._cross_of:
                on_crossing[road] += 1
                if self._person_road.get(pid) != road:  # stepped onto it this step
                    t, i = self._cross_of[road]
                    st = tls.get(t, "")
                    if i < len(st) and st[i] not in GREEN:
                        self._flag("ped_started_on_red", now, pid, road, st[i])
            elif road.startswith(":") and "_w" in road:
                try:
                    nxt = c.person.getNextEdge(pid)
                except Exception:  # noqa: BLE001
                    nxt = ""
                if nxt in self._cross_of:
                    w = c.person.getWaitingTime(pid)
                    if w > self.max_wait.get(nxt, 0.0):
                        self.max_wait[nxt] = w
        self._person_road = roads
        for tid, state in tls.items():
            jn = self.j.get(tid)
            if jn is None or len(state) != len(jn.links):
                continue
            prev = self._prev.get(tid)
            self._prev[tid] = state
            act = [i for i, lk in enumerate(jn.links) if lk.kind in ("vehicle", "crossing")]
            green = [i for i in act if state[i] in GREEN]
            for a in green:
                for b in green:
                    if a < b and jn.conflict(a, b) and not (jn.yields(a, b) or jn.yields(b, a)):
                        self._flag("conflicting_greens", now, tid, a, b, state)
            # poles
            for lm, idx in self._poles(tid).items():
                cols = {"green" if state[i] in GREEN else ("yellow" if state[i] == "y" else "red") for i in idx}
                if len(cols) > 1:
                    self._flag("carla_pole_disagree", now, tid, lm, state)
            if prev is None or prev == state:
                continue
            for i in act:
                key = (tid, i)
                if prev[i] in GREEN and state[i] not in GREEN:
                    self._green_end[key] = now
                    self._yellow_seen[key] = state[i] == "y"
                    if jn.links[i].kind == "vehicle" and state[i] == "r":
                        self._flag("green_to_red", now, tid, i, prev, state)
                elif state[i] == "y":
                    self._yellow_seen[key] = True
            for j in act:
                if not (state[j] in GREEN and prev[j] not in GREEN):
                    continue
                for i in act:
                    if i == j or not jn.conflict(i, j) or state[i] in GREEN:
                        continue
                    end = self._green_end.get((tid, i))
                    if end is None:
                        continue
                    gap = now - end
                    lk = jn.links[i]
                    if lk.kind == "vehicle":
                        if gap < MIN_YELLOW_S + MIN_ALLRED_S - 1e-6:
                            self._flag("short_intergreen", now, tid, i, j, round(gap, 2))
                    else:
                        n_on = on_crossing.get(lk.crossing, 0)
                        if n_on:
                            self._flag("ped_on_crossing_at_release", now, tid, lk.crossing, j, n_on, round(gap, 2))
                            if gap < lk.length / PED_CLEAR_SPEED - 1e-6:
                                self._flag("ped_short_clearance", now, tid, lk.crossing, j, round(gap, 2))

    def vehicle_waits(self, c, tls: dict[str, str]) -> None:
        """The longest a vehicle has stood at a red, per signal link (call now and then: it asks every vehicle)."""
        for vid in c.vehicle.getIDList():
            try:
                nxt = c.vehicle.getNextTLS(vid)
            except Exception:  # noqa: BLE001
                continue
            if not nxt:
                continue
            tid, idx, dist, st = nxt[0]
            if dist > 60.0 or st in GREEN:
                continue
            w = c.vehicle.getWaitingTime(vid)
            key = f"{tid}:{idx}"
            if w > self.max_wait.get(key, 0.0):
                self.max_wait[key] = w

    def _poles(self, tid: str) -> dict[str, list[int]]:
        cache = self.__dict__.setdefault("_pole_cache", {})
        if tid not in cache:
            jn = self.j[tid]
            out: dict[str, list[int]] = {}
            for lk in jn.links:
                if lk.landmark and lk.kind == "vehicle":
                    out.setdefault(lk.landmark, []).append(lk.index)
            cache[tid] = out
        return cache[tid]

    def report(self) -> dict:
        veh = {k: v for k, v in self.max_wait.items() if ":" in k and not k.startswith(":")}
        ped = {k: v for k, v in self.max_wait.items() if k.startswith(":")}
        return {"steps": self.steps, "violations": dict(self.counts),
                "examples": {k: [list(map(str, e)) for e in v] for k, v in self.examples.items()},
                "max_vehicle_wait_s": round(max(veh.values(), default=0.0), 1),
                "max_vehicle_wait_at": max(veh, key=veh.get) if veh else None,
                "max_ped_wait_s": round(max(ped.values(), default=0.0), 1),
                "max_ped_wait_at": max(ped, key=ped.get) if ped else None}

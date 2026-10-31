"""Pedestrian signals drawn in CARLA for the crosswalks of the free-fly viewer.

Town05's light poles carry vehicle heads only, and CARLA has no pedestrian-signal prop, so each end of
each signalised crossing (tools/build_town05_peds.py) gets a small pedestrian head drawn with CARLA's
debug drawing: a lamp about 2.5 m up on the nearest traffic-light pole of the corner (or on a thin post),
green while SUMO lets pedestrians walk on that crossing ("G"), red (a little higher) otherwise: flashing during
the crossing's walk clearance (flashing don't walk: nobody starts, the parallel cars still green), steady after
it (since 26 Oct, `ped_aspect`). Only the
lamp is drawn: a drawn housing on a drawn post looked like the traffic-camera markers (a box on a pole).
In camera-sensor pictures (tools/probe_crosswalk.py) debug shapes come out as white squares whatever their
colour; in the CARLA window they keep it.
Drawn only within DRAW_M of you, each lamp once to last until its signal's next switch (at most LAMP_MAX_S),
again at once when its colour changes (see `plan`). Each lamp is a call to the server that waits for
its next picture, so they are drawn by a thread of their own (`update` only hands it the view and the
signals): drawn from the viewer's main loop, 30 lamps four times a second held it to 6 passes a second.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

DRAW_M = 120.0
POST_M = 2.4
OUT_M = 1.0  # past the end of the crossing, onto the pavement
POLE_M = 7.0
EVERY_S = 0.25      # the lamps' thread looks at least this often
LIFE_S = 0.35       # a lamp's life when its signal's next switch is not known (or is now)
LAMP_MAX_S = 2.0    # at most: a switch earlier than SUMO said shows both colours no longer than this
RENEW_S = 0.1       # a lamp still the same colour is drawn again this long before its drawing runs out
POST_LIFE_S = 20.0


FLASH_S = 0.5       # flashing don't walk: half a second on, half off (MUTCD 4E.06: 50-60 flashes a minute)


def clearance_states(net_file: Path) -> dict[str, dict[int, set[str]]]:
    """For each crossing link (tls id -> link index), the signal states of the phases that clear it: the "walk
    clearance" phases (tools/build_town05_signals.py) that follow its walk, where the crossing is red to newcomers
    while the parallel cars keep their green. Its lamp flashes then (flashing don't walk); in the yellow and all red
    after it, and the cross street's green, it is steady (steady don't walk)."""
    import xml.etree.ElementTree as ET

    cross: dict[str, set[int]] = {}
    pat = re.compile(r'<connection from="[^"]*" to="(:[^"]+_c\d+)"[^>]*tl="([^"]+)" linkIndex="(\d+)"')
    with open(net_file, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                cross.setdefault(m.group(2), set()).add(int(m.group(3)))
    out: dict[str, dict[int, set[str]]] = {}
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag != "tlLogic":
            continue
        tid = el.get("id")
        phases = el.findall("phase")
        for k, ph in enumerate(phases):
            if ph.get("name") != "walk clearance":
                continue
            was, st = phases[k - 1].get("state"), ph.get("state")
            for i in cross.get(tid, ()):
                if was[i] in "Gg" and st[i] not in "Gg":
                    out.setdefault(tid, {}).setdefault(i, set()).add(st)
        el.clear()
    return out


def ped_aspect(state: str, idx: int, clear: set[str] | None) -> str:
    """"walk" (SUMO lets people start across), "flashing" (walk clearance: nobody starts, those on it get across,
    the parallel cars still green) or "dont_walk" (steady: the buffer before the conflicting cars go, and while
    they go)."""
    if idx < len(state) and state[idx] in "Gg":
        return "walk"
    if idx < len(state) and state[idx] == "u":  # 27 Oct: the clearance marks its crossings (cosim/signal_plan.py)
        return "flashing"
    if clear and state in clear:
        return "flashing"
    return "dont_walk"


class PedLights:
    def __init__(self, world, net_file: Path, frame, cmap):
        import sumolib

        from cosim.run import ground_z  # the road's level where roads stack (Town05's ring road)

        self.world = world
        link = {}  # crossing edge id -> (tls id, link index)
        pat = re.compile(r'<connection from="[^"]*" to="(:[^"]+_c\d+)"[^>]*tl="([^"]+)" linkIndex="(\d+)"')
        with open(net_file, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    link[m.group(1)] = (m.group(2), int(m.group(3)))
        net = sumolib.net.readNet(str(net_file), withInternal=True, withPedestrianConnections=True)
        self.heads = []  # (x, y, z of the ground, tls, link index)
        for e in net.getEdges():
            if e.getFunction() != "crossing" or e.getID() not in link:
                continue
            sh = e.getLanes()[0].getShape()
            (x0, y0), (x1, y1) = sh[0][:2], sh[-1][:2]
            d = math.hypot(x1 - x0, y1 - y0) or 1.0
            ux, uy = (x1 - x0) / d, (y1 - y0) / d
            for (px, py), s in (((x0, y0), -1.0), ((x1, y1), 1.0)):
                cx, cy = frame.to_carla_xy(px + s * ux * OUT_M, py + s * uy * OUT_M)
                self.heads.append((cx, cy, ground_z(cmap, cx, cy), *link[e.getID()]))
        # mount each head on the nearest CARLA traffic-light pole when one stands within POLE_M of the end of
        # the crossing (most corners have one), else on a post of its own
        poles = []
        for tl in world.get_actors().filter("traffic.traffic_light"):
            loc = tl.get_transform().location
            poles.append((loc.x, loc.y))
        self.mounted = []
        for hx, hy, hz, tid, idx in self.heads:
            best = min(poles, key=lambda p: (p[0] - hx) ** 2 + (p[1] - hy) ** 2) if poles else None
            on_pole = best is not None and math.hypot(best[0] - hx, best[1] - hy) <= POLE_M
            if on_pole:  # on the pole's side towards the crossing
                d = math.hypot(hx - best[0], hy - best[1]) or 1.0
                mx, my = best[0] + (hx - best[0]) / d * 0.3, best[1] + (hy - best[1]) / d * 0.3
            else:
                mx, my = hx, hy
            self.mounted.append((mx, my, hz, tid, idx, on_pole))
        self.fdw = clearance_states(net_file)  # tls -> link -> walk clearance states (flashing don't walk)
        self._t = 0.0
        self._drawn: dict[int, tuple[bool, float]] = {}  # lamp -> (walk, until)
        self._posts: dict[int, float] = {}               # lamp's post -> until
        self._next_due = None
        self._want = None  # (x, y, signals, visible) from update(), for the thread
        self._thread = None

    def update(self, x: float, y: float, tls: dict[str, str], visible: bool = True,
               remaining: dict[str, float] | None = None) -> None:
        """Where you are and the signals shown; the lamps' thread draws them (never waits). remaining: seconds
        until each signal's next switch (SUMO), so a lamp is drawn once to last until then."""
        import threading

        self._want = (x, y, tls, visible, remaining)
        if self._thread is None:
            self._wake = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True, name="ped-lamps")
            self._thread.start()
        if tls != getattr(self, "_tls", None):
            self._wake.set()

    def _run(self) -> None:
        while True:
            wait = EVERY_S
            due = getattr(self, "_next_due", None)
            if due is not None:
                wait = max(0.02, min(EVERY_S, due - time.time()))
            self._wake.wait(wait)
            self._wake.clear()
            want = self._want
            if want is None or not want[3]:
                continue
            try:
                self.draw(want[0], want[1], want[2], remaining=want[4])
            except Exception:  # noqa: BLE001 - CARLA gone: the viewer is closing
                time.sleep(1.0)

    def plan(self, x: float, y: float, tls: dict[str, str], remaining: dict[str, float] | None,
             now: float) -> list[tuple[int, bool, float, bool]]:
        """The lamps to draw now: (lamp, walk, life, with its post). A lamp is drawn when it comes within DRAW_M,
        when its colour changes, or just before its last drawing runs out; it lives until its signal's next
        switch (+0.1 s, LIFE_S to LAMP_MAX_S), when its colour may change. Each drawing is a call to the server,
        which handles calls for only a moment each picture: redrawn four times a second, 30-60 lamps made
        150-235 calls a second and put the vehicles' poses behind (cosim/pose_proc.py)."""
        out = []
        drawn, posts = self._drawn, self._posts
        fdw = getattr(self, "fdw", {})
        flashing = self.flashing = set()
        for i, (hx, hy, hz, tid, idx, on_pole) in enumerate(self.mounted):
            if (hx - x) ** 2 + (hy - y) ** 2 > DRAW_M * DRAW_M:
                continue
            state = tls.get(tid, "")
            aspect = ped_aspect(state, idx, fdw.get(tid, {}).get(idx))
            walk = aspect == "walk"
            key = walk
            if aspect == "flashing":  # red, half a second on, half off
                on = int(now / FLASH_S) % 2 == 0
                key = "flash_on" if on else "flash_off"
            d = drawn.get(i)
            if d is not None and d[0] == key and now < d[1] - RENEW_S:
                continue
            rem = remaining.get(tid) if remaining else None
            life = LIFE_S if rem is None else max(LIFE_S, min(rem + 0.1, LAMP_MAX_S))
            if aspect == "flashing":
                life = FLASH_S - (now % FLASH_S) + RENEW_S  # until the next flash
                drawn[i] = (key, now + life)
                if key == "flash_off":
                    continue  # dark: nothing to draw, the last drawing runs out
                flashing.add(i)
                post = not on_pole and now >= posts.get(i, 0.0) - RENEW_S
                if post:
                    posts[i] = now + POST_LIFE_S
                out.append((i, False, life, post))
                continue
            drawn[i] = (key, now + life)
            post = not on_pole and now >= posts.get(i, 0.0) - RENEW_S
            if post:
                posts[i] = now + POST_LIFE_S
            out.append((i, walk, life, post))
        ends = [e - RENEW_S for _, e in drawn.values() if e > now - 1.0]
        self._next_due = min(ends) if ends else None
        return out

    def draw(self, x: float, y: float, tls: dict[str, str], remaining: dict[str, float] | None = None) -> None:
        import carla

        if not tls:
            return
        self._tls = tls
        todo = self.plan(x, y, tls, remaining, time.time())
        if not todo:
            return
        dbg = self.world.debug
        post = carla.Color(30, 30, 30)
        red, green = carla.Color(255, 30, 20), carla.Color(20, 255, 80)
        for i, walk, life, with_post in todo:
            hx, hy, hz, _, _, _ = self.mounted[i]
            if with_post:
                dbg.draw_line(carla.Location(hx, hy, hz), carla.Location(hx, hy, hz + POST_M - 0.15), thickness=0.05,
                              color=post, life_time=POST_LIFE_S)
            # one lamp: red at the top while it says wait, green below it while it says walk
            dbg.draw_point(carla.Location(hx, hy, hz + POST_M + (0.0 if walk else 0.3)), size=0.18,
                           color=green if walk else red, life_time=life)
            self.calls = getattr(self, "calls", 0) + (2 if with_post else 1)

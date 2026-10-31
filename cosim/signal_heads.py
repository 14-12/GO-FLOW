"""Each head of each traffic-light pole shows its own lane and directions, drawn in CARLA (27 Oct).

A CARLA traffic light (one per pole on Town05, `Signal_3Light_Post01`) has one state for all its heads: three over
the lanes and one on the post (TrafficLight.get_light_boxes; tools/freefly_poles_demo.bat). The signals give each
lane its own green (cosim/signal_plan.py: the left-turn lane red while the lane beside it goes straight on), which
one colour per pole cannot show. So the poles' own lamps are switched off (CarlaSide.lights_off) and each head is
drawn lit with CARLA's debug drawing, as a real head with arrow lenses:

* the three heads over the lanes, from the left as the drivers see them, are the approach's lanes from the left:
  the left head the left lane, with an arrow for each way that lane may go (left only, or left and straight on),
  the middle head the next lane's straight on, the right head its right turn (two lanes, three heads); with three
  lanes one head each. Lit at the red (top), amber (middle) or green (bottom) lens, from SUMO's state of that
  lane's links (all the directions of a lane go together: cosim/signal_plan.py);
* the head on the post repeats the rightmost lane for the drivers at the line (it is a vehicle head; nothing is drawn
  as points: CARLA shows those as squares).

Each arrow is small (0.15 m, inside a 0.2 m lens) and drawn on the lens' glass, found with a ray from in front of the
head (`_lens_face`), so it sits inside the visor like a lit arrow lens instead of floating in front of the head.

Drawn within DRAW_M of the view, by a thread of its own, each lamp once to last until its signal's next switch (as
cosim/ped_lights.py does: each drawing is a call the server answers with its next picture).
"""

from __future__ import annotations

import math
import threading
import time

DRAW_M = 140.0
LIFE_S = 0.35
LAMP_MAX_S = 2.0
RENEW_S = 0.1
FLASH_S = 0.5
LENS_DZ = 0.27          # the lenses' height above / below the head's middle (a 0.88 m head, 0.2 m lenses)
FACE_M = 0.06           # the lens' face in front of the head's middle when the ray to it finds nothing
ON_LENS_M = 0.012       # the arrow just on the lens' glass
ARROW_M = 0.15          # inside a 0.2 m lens
ARROW_T = 0.03
ARROW_HEAD = 0.06
COLOURS = {"G": (40, 235, 90), "y": (255, 165, 20), "r": (255, 35, 25)}


def direction_state(state: str, idx: list[int]) -> str:
    chars = [state[i] for i in idx if i < len(state)]
    if any(c in "Gg" for c in chars):
        return "G"
    if any(c in "yYu" for c in chars):
        return "y"
    return "r"


def assign_heads(boxes: list[tuple[float, float, float]], travel_yaw_deg: float,
                 lanes: list[tuple[str, list[int]]]) -> list[tuple[int, str, list[int]]]:
    """boxes: the heads' (x, y, z); lanes: the approach's lanes from the left as the drivers see them, each
    (directions "l"/"s"/"r" as a string, SUMO link indices). Returns, per box, (box index, the arrows it shows, the
    links it shows): the highest three over the lanes, ordered from the left as the approaching drivers see them;
    the lowest (on the post) repeats the rightmost lane."""
    if not boxes or not lanes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][2])
    mast, post = order[:3], order[3:]
    if len(boxes) <= 2:
        mast, post = order, []
    a = math.radians(travel_yaw_deg)
    rx, ry = -math.sin(a), math.cos(a)  # CARLA (left-handed, y to the right): the drivers' right
    cx = sum(boxes[i][0] for i in mast) / len(mast)
    cy = sum(boxes[i][1] for i in mast) / len(mast)
    mast = sorted(mast, key=lambda i: (boxes[i][0] - cx) * rx + (boxes[i][1] - cy) * ry)
    # what each head shows: lanes one to a head; more heads than lanes: the rightmost lane's directions spread over
    # the heads left (its straight on, then its right turn); more lanes than heads: the rightmost head takes the rest
    shows: list[tuple[str, list[int]]] = []
    n = len(mast)
    if len(lanes) >= n:
        shows = [lanes[i] for i in range(n - 1)]
        rest = lanes[n - 1:]
        shows.append(("".join(sorted({d for ds, _ in rest for d in ds}, key="lsr".index)),
                      [i for _, idx in rest for i in idx]))
    else:
        shows = list(lanes[:-1])
        ds, idx = lanes[-1]
        spare = n - len(shows)
        parts = [d for d in "lsr" if d in ds]
        if spare >= 2 and len(parts) >= 2:
            # e.g. one lane "sr" over two heads: straight on, right turn (both the lane's colour)
            split = [parts[:-1]] + [[parts[-1]]] if spare == 2 else [[d] for d in parts]
            while len(split) < spare:
                split.append(split[-1])
            shows += [("".join(p), idx) for p in split[:spare]]
        else:
            shows += [(ds, idx)] * spare
    out = [(i, ds, idx) for i, (ds, idx) in zip(mast, shows)]
    # the low head on the post repeats the nearest (rightmost) lane for the drivers at the line (27 Oct: it is a
    # vehicle head; the dot drawn on it as a pedestrian signal looked like the white squares of 22 Oct)
    out += [(i, lanes[-1][0], lanes[-1][1]) for i in post]
    return out


class SignalHeads:
    def __init__(self, world, tl_links: dict, net_file, ped_lights_heads: list, clear_states: dict):
        """tl_links: cosim/carla_side.CarlaSide._tl_links (actor id -> (actor, [(tls, link)])); net_file: the SUMO
        network (each link's direction); ped_lights_heads: cosim/ped_lights.PedLights.heads (crossing ends: x, y,
        z, tls, link) and clear_states (cosim/ped_lights.clearance_states): kept for the callers, unused since 27 Oct."""
        import re

        self.world = world
        self.lamps = []  # (x, y, z, yaw of travel, tls, links, what) per head
        self.pole_of: list[int] = []  # the CARLA traffic-light actor of each lamp
        self.clear = clear_states
        lane_of: dict[tuple[str, int], tuple[str, int, str]] = {}  # (tls, link) -> (from edge, lane, dir)
        pat = re.compile(r'<connection from="([^"]+)" to="[^"]+" fromLane="(\d+)"[^>]*tl="([^"]+)" '
                         r'linkIndex="(\d+)"[^>]*dir="([a-zA-Z])"')
        with open(net_file, encoding="utf-8") as f:
            for line in f:
                m = pat.search(line)
                if m and not m.group(1).startswith(":"):
                    d = {"s": "s", "r": "r", "R": "r", "l": "l", "L": "l", "t": "l"}.get(m.group(5), "s")
                    lane_of[(m.group(3), int(m.group(4)))] = (m.group(1), int(m.group(2)), d)
        self.link_dir = {k: v[2] for k, v in lane_of.items()}  # (tls, link) -> l / s / r
        self.unlinked = 0
        self.faces_found = 0
        for tl_id, (tl, links) in tl_links.items():
            try:
                boxes = [(b.location.x, b.location.y, b.location.z) for b in tl.get_light_boxes()]
                wps = tl.get_affected_lane_waypoints()
                yaw = wps[0].transform.rotation.yaw if wps else tl.get_transform().rotation.yaw
            except Exception:  # noqa: BLE001, S112 - an actor without boxes: left as CARLA draws it
                continue
            tid = links[0][0] if links else None
            by_lane: dict[tuple[str, int], tuple[set, list]] = {}
            for t, i in links:
                ln = lane_of.get((t, i))
                if ln is not None:
                    ds, idx = by_lane.setdefault(ln[:2], (set(), []))
                    ds.add(ln[2])
                    idx.append(i)
            if not by_lane or tid is None:
                continue
            # SUMO numbers lanes from the right: the highest index is the drivers' left
            lanes = [("".join(sorted(ds, key="lsr".index)), idx)
                     for (_, _), (ds, idx) in sorted(by_lane.items(), key=lambda kv: -kv[0][1])]
            for bi, what, idx in assign_heads(boxes, yaw, lanes):
                x, y, z = self._lens_face(boxes[bi], yaw)
                self.lamps.append((x, y, z, yaw, tid, idx, what))
                self.pole_of.append(tl_id)
        self._drawn: dict[int, tuple[str, float]] = {}
        self._want = None
        self._thread = None
        self._wake = threading.Event()
        self.calls = 0

    def _lens_face(self, box: tuple[float, float, float], travel_yaw: float) -> tuple[float, float, float]:
        """The middle of the head's face (the glass of its lenses, inside the visors): a ray from 1 m in front of
        the head towards its middle, at the middle lens; FACE_M in front of the middle when it hits nothing."""
        x, y, z = box
        a = math.radians(travel_yaw)
        fx, fy = -math.cos(a), -math.sin(a)  # towards the drivers
        d = FACE_M
        try:
            import carla

            start = carla.Location(x + fx * 1.0, y + fy * 1.0, z)
            hits = self.world.cast_ray(start, carla.Location(x - fx * 0.3, y - fy * 0.3, z))
            ds = [1.0 - math.hypot(h.location.x - start.x, h.location.y - start.y) for h in hits]
            ds = [v for v in ds if -0.3 <= v <= 0.3]
            if ds:
                d = max(ds)  # the first surface the ray meets: the lens (the visor is open in the middle)
                self.faces_found += 1
        except Exception:  # noqa: BLE001, S110 - no cast_ray (older server): the default depth
            pass
        return x + fx * d, y + fy * d, z

    # ------------------------------------------------------------------ what to draw
    def arrows(self, k: int, tls: dict[str, str]) -> list[tuple[str, str]]:
        """(direction, G / y / r) for each arrow of head k: each direction of the lane its own colour (the left
        arrow of a straight-and-left lane red while its straight-on arrow is green, SIGNAL_UNIT "direction")."""
        *_, tid, idx, what = self.lamps[k]
        state = tls.get(tid, "")
        out = []
        for d in what:
            links = [i for i in idx if self.link_dir.get((tid, i)) == d] or idx
            out.append((d, direction_state(state, links)))
        return out

    def aspect(self, k: int, tls: dict[str, str]) -> str:
        """G / y / r: the head's colour now (its greenest arrow)."""
        cols = [c for _, c in self.arrows(k, tls)]
        return "G" if "G" in cols else ("y" if "y" in cols else "r")

    def plan(self, x: float, y: float, tls: dict[str, str], remaining: dict[str, float] | None, now: float):
        out = []
        for k, (hx, hy, hz, yaw, tid, idx, what) in enumerate(self.lamps):
            if (hx - x) ** 2 + (hy - y) ** 2 > DRAW_M * DRAW_M:
                continue
            asp = tuple(self.arrows(k, tls))
            key = asp
            d = self._drawn.get(k)
            if d is not None and d[0] == key and now < d[1] - RENEW_S:
                continue
            rem = remaining.get(tid) if remaining else None
            life = LIFE_S if rem is None else max(LIFE_S, min(rem + 0.1, LAMP_MAX_S))
            self._drawn[k] = (key, now + life)
            out.append((k, asp, life))
        return out

    # ------------------------------------------------------------------ drawing
    def update(self, x: float, y: float, tls: dict[str, str], visible: bool = True,
               remaining: dict[str, float] | None = None) -> None:
        self._want = (x, y, tls, visible, remaining)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="signal-heads")
            self._thread.start()
        if tls is not getattr(self, "_tls", None):
            self._tls = tls
            self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(0.25)
            self._wake.clear()
            w = self._want
            if w is None or not w[3]:
                continue
            try:
                self.draw(w[0], w[1], w[2], w[4])
            except Exception:  # noqa: BLE001 - CARLA gone: the viewer is closing
                time.sleep(1.0)

    def draw(self, x: float, y: float, tls: dict[str, str], remaining=None) -> None:
        import carla

        dbg = self.world.debug
        for k, asp, life in self.plan(x, y, tls, remaining, time.time()):
            hx, hy, hz, yaw, *_ = self.lamps[k]
            a = math.radians(yaw)
            fx, fy = -math.cos(a), -math.sin(a)          # towards the drivers
            rx, ry = -math.sin(a), math.cos(a)           # the drivers' right
            cx, cy = hx + fx * ON_LENS_M, hy + fy * ON_LENS_M  # (hx, hy: the face of the lenses, _lens_face)
            # each arrow on the lens of its colour (red top, amber middle, green bottom), as the drivers see it:
            # up for straight on, to their left / right for the turns; arrows of one colour share a lens (a lane
            # going two ways on one green: both arrows on its green lens)
            by_col: dict[str, str] = {}
            for d, c in asp:
                by_col[c] = by_col.get(c, "") + d
            for c, dirs in by_col.items():
                col = carla.Color(*COLOURS.get(c, (255, 35, 25)))
                z = hz + {"r": LENS_DZ, "y": 0.0, "G": -LENS_DZ}.get(c, LENS_DZ)
                base = carla.Location(cx, cy, z - (ARROW_M / 2 if "s" in dirs else 0.0))
                for d in dirs:
                    if d == "s":
                        b, e = base, carla.Location(cx, cy, base.z + ARROW_M)
                    else:
                        sgn = -1.0 if d == "l" else 1.0
                        if "s" in dirs:  # from the stem of the straight-on arrow
                            b = carla.Location(cx, cy, base.z + ARROW_M * 0.45)
                        else:
                            b = carla.Location(cx - sgn * rx * ARROW_M / 2, cy - sgn * ry * ARROW_M / 2, z)
                        e = carla.Location(b.x + sgn * rx * ARROW_M * (0.6 if "s" in dirs else 1.0),
                                           b.y + sgn * ry * ARROW_M * (0.6 if "s" in dirs else 1.0), b.z)
                    dbg.draw_arrow(b, e, thickness=ARROW_T, arrow_size=ARROW_HEAD, color=col, life_time=life)
            self.calls += 1

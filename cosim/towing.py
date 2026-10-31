"""The recovery truck: a rollback ("plateau") tow truck, and how it loads a wreck (24 Oct).

CARLA 0.9.16 has no tow truck. Ours is the European HGV's tractor with a flatbed of our own on its chassis
(SM_TnDeck, tools/plates/make_deck_mesh.py; a prop posed with the truck, like the number plates). It loads a wreck
the way a rollback does, in the closed lane, ahead of the wreck:

1. reverse: it has passed the wreck on the open lane and stopped ahead of it; it backs into the closed lane until
   its deck's end, once tilted, will touch the road just in front of the wreck;
2. the deck slides back along the chassis, then tilts about the chassis' rear end until its end is on the road;
3. the crew hooks the winch; the wreck, pulled by its front, comes straight behind the deck;
4. the winch draws it up: front wheels up the deck, rear wheels still on the road, the body pitching as it climbs,
   until all four are on the deck;
5. the deck tilts back level and slides forward with the wreck on it; the crew straps it down;
6. the truck drives off with it (SUMO drives the truck again; the wreck is carried on the deck).

Everything here is geometry and timing, with no CARLA: positions in the truck's own frame (x forward, z up, the origin
at the middle of the body on the road, as CARLA's vehicles have it). cosim/pose_proc.py turns them into poses;
cosim/incidents.py uses the same timing (SUMO holds the truck while it loads, then takes the wreck off the road).
"""

from __future__ import annotations

import math

TRUCK_MODEL = "vehicle.carlamotors.european_hgv"
TRUCK_HALF = 3.968                  # half the HGV's length (config/vehicle_models.json)
DECK_MESH = "/Game/Carla/Static/Other/SM_TnDeck.SM_TnDeck"
# the mesh is built standing (tools/plates/make_deck_mesh.py), its -X face the deck's top (in CARLA only a mesh's -X face
# shows its texture coordinates with this material: tools/probe_deck_uv.py, the plate's too): laid flat, -X up
MESH_ROT = (0.0, 0.0, 0.0, -90.0, 0.0, 0.0)
DECK_L, DECK_W, DECK_T = 3.20, 1.18, 0.07   # the deck's half length, half width, half thickness (m)
# where the deck sits on the HGV (tools/probe_tow.py measured the chassis): its front end behind the cab, its bottom
# on the chassis' top; it tilts about the chassis' rear end
DECK_FRONT_X = 1.45                 # the cab's back is at +1.55
CHASSIS_Z = 1.34                    # above the fifth wheel and the rear wings (1.33); the frame rails are at 1.18
PIVOT_X = -3.62                     # the chassis ends at -3.65
TILT_DEG = 13.0
CAR_X = 0.15                        # a loaded car's middle, on the deck (deck frame)
GAP_M = 0.35                        # between the wreck's front and the tilted deck's end
WINCH_MS = 0.45                     # the winch's pull
REV_S, SLIDE_S, TILT_S, ALIGN_S, STRAP_S = 9.0, 3.0, 3.5, 4.0, 3.5
REJOIN_S = 6.0                      # back to where SUMO has the truck, when loading was not where SUMO holds it


def _ease(k: float) -> float:
    k = min(max(k, 0.0), 1.0)
    return k * k * (3.0 - 2.0 * k)


# ------------------------------------------------------------------ the deck
def deck_level() -> tuple:
    """The deck at rest on the truck: (x, y, z, pitch, yaw) in the truck's frame (its middle)."""
    return (DECK_FRONT_X - DECK_L, 0.0, CHASSIS_Z + DECK_T, 0.0, 0.0)


def _pivot_local() -> float:
    """Where the pivot lies along the tilted deck (deck frame x): its end then rests on the road."""
    return -DECK_L + CHASSIS_Z / math.sin(math.radians(TILT_DEG))


def deck_pose(slide: float, tilt: float) -> tuple:
    """The deck slid back `slide` (0..1) along the chassis, then tilted `tilt` (0..1) about its rear end."""
    x0, _, z0, _, _ = deck_level()
    xs = PIVOT_X - _pivot_local()  # slid back: the pivot under the deck where the tilted deck will turn about it
    if tilt <= 0.0:
        return (x0 + (xs - x0) * _ease(slide), 0.0, z0, 0.0, 0.0)
    phi = math.radians(TILT_DEG) * _ease(tilt)
    a = _pivot_local()
    cx = PIVOT_X - a * math.cos(phi) - DECK_T * math.sin(phi)
    cz = CHASSIS_Z - a * math.sin(phi) + DECK_T * math.cos(phi)
    return (cx, 0.0, cz, math.degrees(phi), 0.0)


def _on_deck(deck: tuple, lx: float, lz: float) -> tuple[float, float]:
    """A point of the deck's frame (x along it, z up from its middle) in the truck's frame (x, z)."""
    cx, _, cz, p, _ = deck
    r = math.radians(p)
    return cx + lx * math.cos(r) - lz * math.sin(r), cz + lx * math.sin(r) + lz * math.cos(r)


def deck_end(deck: tuple) -> tuple[float, float]:
    """The deck's rear end, on its top (truck frame x, z)."""
    return _on_deck(deck, -DECK_L, DECK_T)


def truck_ahead() -> float:
    """How far ahead of the wreck's front the loading truck's front is (m): its tilted deck's end GAP_M before it."""
    ex, _ = deck_end(deck_pose(1.0, 1.0))
    return GAP_M - ex + TRUCK_HALF


# ------------------------------------------------------------------ the wreck's way up
def wheelbase(car_len: float) -> float:
    return 0.6 * car_len


class Path:
    """The way the wreck's wheels go, in the truck's frame (x, z): along the road up to the tilted deck's end, a short
    ramp onto it, up the deck. Arc length 0 is where the ramp meets the road."""

    def __init__(self):
        deck = deck_pose(1.0, 1.0)
        self.e = deck_end(deck)
        self.b = (self.e[0] - 0.20, 0.0)  # where the wheels leave the road
        r = math.radians(deck[3])
        self.dir = (math.cos(r), math.sin(r))
        self.ramp = math.hypot(self.e[0] - self.b[0], self.e[1] - self.b[1])

    def at(self, s: float) -> tuple[float, float]:
        if s <= 0.0:
            return self.b[0] + s, 0.0
        if s <= self.ramp:
            k = s / self.ramp
            return self.b[0] + (self.e[0] - self.b[0]) * k, self.e[1] * k
        d = s - self.ramp
        return self.e[0] + self.dir[0] * d, self.e[1] + self.dir[1] * d

    def s_start(self, car_len: float) -> float:
        """Arc of the front wheels with the car's front GAP_M behind the deck's end (on the road)."""
        return (self.e[0] - self.b[0]) - GAP_M - (car_len - wheelbase(car_len)) / 2.0

    def s_end(self, car_len: float) -> float:
        """Arc of the front wheels with the car at CAR_X on the deck."""
        return self.ramp + DECK_L + CAR_X + wheelbase(car_len) / 2.0

    def car(self, s: float, car_len: float) -> tuple[float, float, float]:
        """(x, z, pitch) of the car's origin (between its axles, on the ground under it) with its front wheels at s."""
        w = wheelbase(car_len)
        fx, fz = self.at(s)
        rx, rz = self.at(s - w)
        return (fx + rx) / 2.0, (fz + rz) / 2.0, math.degrees(math.atan2(fz - rz, fx - rx))


def car_on_deck() -> tuple:
    """A loaded car on the level deck, in the truck's frame: (x, y, z, pitch, yaw)."""
    x0, _, z0, _, _ = deck_level()
    return (x0 + CAR_X, 0.0, z0 + DECK_T, 0.0, 0.0)


# ------------------------------------------------------------------ timing
def timeline(car_len: float, rejoin: bool = False) -> dict:
    """When each part of the loading ends (s after it starts), for a wreck `car_len` long."""
    p = Path()
    winch = (p.s_end(car_len) - p.s_start(car_len)) / WINCH_MS
    t, out = 0.0, {}
    for name, d in (("reverse", REV_S), ("slide", SLIDE_S), ("tilt", TILT_S), ("align", ALIGN_S), ("winch", winch),
                    ("tilt_back", TILT_S), ("slide_back", SLIDE_S), ("strap", STRAP_S),
                    ("rejoin", REJOIN_S if rejoin else 0.0)):
        t += d
        out[name] = t
    out["total"] = t
    return out


def phase_at(tl: dict, t: float) -> tuple[str, float]:
    """(part, 0..1 through it) at t s after the start."""
    t0 = 0.0
    for name in ("reverse", "slide", "tilt", "align", "winch", "tilt_back", "slide_back", "strap", "rejoin"):
        t1 = tl[name]
        if t < t1 and t1 > t0:
            return name, (t - t0) / (t1 - t0)
        t0 = t1
    return "done", 1.0


def deck_at(tl: dict, t: float) -> tuple:
    name, k = phase_at(tl, t)
    if name == "reverse":
        return deck_pose(0.0, 0.0)
    if name == "slide":
        return deck_pose(k, 0.0)
    if name == "tilt":
        return deck_pose(1.0, k)
    if name in ("align", "winch"):
        return deck_pose(1.0, 1.0)
    if name == "tilt_back":
        return deck_pose(1.0, 1.0 - k)
    if name == "slide_back":
        return deck_pose(1.0 - k, 0.0)
    return deck_pose(0.0, 0.0)


# ------------------------------------------------------------------ the truck backing in
def manoeuvre(p0: tuple, p1: tuple, k: float) -> tuple:
    """The truck from pose p0 to p1 ((x, y, z, yaw) in CARLA's world, yaw in degrees), k 0..1: a smooth S along a
    cubic whose ends follow each pose's heading; backing when p1 is behind p0. Returns (x, y, z, yaw)."""
    x0, y0, z0, a0 = p0
    x1, y1, z1, a1 = p1
    h0 = (math.cos(math.radians(a0)), math.sin(math.radians(a0)))
    h1 = (math.cos(math.radians(a1)), math.sin(math.radians(a1)))
    dx, dy = x1 - x0, y1 - y0
    d0, d1 = dx * h0[0] + dy * h0[1], dx * h1[0] + dy * h1[1]
    sgn = 1.0 if d0 >= 0.0 else -1.0  # forwards or backwards
    m0 = (h0[0] * d0, h0[1] * d0)
    m1 = (h1[0] * d1, h1[1] * d1)
    e = _ease(k)  # starts and stops gently
    e2, e3 = e * e, e * e * e
    hh = (2 * e3 - 3 * e2 + 1, e3 - 2 * e2 + e, -2 * e3 + 3 * e2, e3 - e2)
    x = hh[0] * x0 + hh[1] * m0[0] + hh[2] * x1 + hh[3] * m1[0]
    y = hh[0] * y0 + hh[1] * m0[1] + hh[2] * y1 + hh[3] * m1[1]
    dh = (6 * e2 - 6 * e, 3 * e2 - 4 * e + 1, -6 * e2 + 6 * e, 3 * e2 - 2 * e)
    vx = dh[0] * x0 + dh[1] * m0[0] + dh[2] * x1 + dh[3] * m1[0]
    vy = dh[0] * y0 + dh[1] * m0[1] + dh[2] * y1 + dh[3] * m1[1]
    if math.hypot(vx, vy) < 1e-6:
        yaw = a0 + ((a1 - a0 + 180.0) % 360.0 - 180.0) * e
    else:
        yaw = math.degrees(math.atan2(sgn * vy, sgn * vx))
    return x, y, z0 + (z1 - z0) * e, yaw


# ------------------------------------------------------------------ its look
def deck_texture(w: int = 384, h: int = 96):
    """The deck's texture (Pillow RGB): u along the deck (rear at u=0), v across; rows above DECK_V0 the side band."""
    import random

    from PIL import Image, ImageDraw

    v0 = int(round(0.08 * h))
    im = Image.new("RGB", (w, h), (128, 132, 138))
    d = ImageDraw.Draw(im)
    rng = random.Random(5)
    # the side band: dark steel with red and white reflective blocks (as on European recovery trucks)
    d.rectangle([0, 0, w - 1, v0 - 1], fill=(38, 40, 44))
    bw = max(w // 24, 4)
    for i, x in enumerate(range(0, w, bw)):
        d.rectangle([x, 1, x + bw - 2, v0 - 2], fill=(196, 32, 36) if i % 2 == 0 else (232, 232, 228))
    # the deck: aluminium planks along it, seams, wear
    rows = h - v0
    planks = 9
    for i in range(planks):
        a = v0 + int(i * rows / planks)
        b = v0 + int((i + 1) * rows / planks) - 1
        g = 126 + rng.randint(-10, 10)
        d.rectangle([0, a, w - 1, b], fill=(g, g + 3, g + 8))
        d.line([0, b, w - 1, b], fill=(78, 80, 84))
    for _ in range(w * rows // 40):  # scuffs
        x, y = rng.randrange(w), rng.randrange(v0, h)
        c = rng.randint(90, 165)
        d.point((x, y), fill=(c, c + 2, c + 5))
    # the side rails (the deck's edges, darker), and the wheel stops near the front
    d.rectangle([0, v0, w - 1, v0 + 2], fill=(52, 54, 58))
    d.rectangle([0, h - 3, w - 1, h - 1], fill=(52, 54, 58))
    # the rear edge: yellow and black chevrons
    cw = max(w // 32, 6)
    for i, y in enumerate(range(v0, h, cw)):
        d.polygon([(0, y), (cw, y + cw // 2), (cw, y + cw + cw // 2), (0, y + cw)],
                  fill=(236, 186, 22) if i % 2 == 0 else (24, 24, 24))
    return im

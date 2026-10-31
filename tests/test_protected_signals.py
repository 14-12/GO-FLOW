"""The 27 Oct signals: protected greens (cosim/signal_plan.py), each head of a pole showing its own direction
(cosim/signal_heads.py), and walkers kept at the kerb while their crossing shows DON'T WALK
(cosim/sumo_side.PedRedHold)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from cosim.signal_audit import Link
from cosim.signal_heads import assign_heads, direction_state
from cosim.signal_plan import plan


def _cross_junction():
    """A four-way junction, two lanes an approach: right lane straight + right, left lane left only, one crossing
    per leg. Conflicts from the geometry (right-hand traffic)."""
    links = []
    dest = {"N": {"s": "S", "r": "W", "l": "E"}, "S": {"s": "N", "r": "E", "l": "W"},
            "E": {"s": "W", "r": "N", "l": "S"}, "W": {"s": "E", "r": "S", "l": "N"}}
    for a in "NESW":
        for d, lane in (("r", f"{a}_3"), ("s", f"{a}_3"), ("l", f"{a}_4")):
            links.append(Link(len(links), "vehicle", a, from_lane=lane, dir=d, speed=13.89))
    for a in "NESW":
        links.append(Link(len(links), "crossing", crossing=f":J_c{a}", length=14.0))
    veh = [lk for lk in links if lk.kind == "vehicle"]
    opp = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def vv(x, y):
        if x.approach == y.approach:
            return False
        if dest[x.approach][x.dir] == dest[y.approach][y.dir]:
            return True  # merge
        if opp[x.approach] == y.approach:  # opposite: only a left against the other's straight / right
            return (x.dir == "l") != (y.dir == "l") and "s" in (x.dir, y.dir) or (x.dir == "l" and y.dir == "r") \
                or (y.dir == "l" and x.dir == "r")
        # crossing streets: right turns stay clear of the cross street unless merging
        return x.dir != "r" and y.dir != "r"

    def vc(x, c):
        leg = c.crossing[-1]
        return x.approach == leg or dest[x.approach][x.dir] == leg

    def conflict(i, j):
        a, b = links[i], links[j]
        if a.kind == "vehicle" and b.kind == "vehicle":
            return vv(a, b)
        if a.kind == "crossing" and b.kind == "crossing":
            return False
        v, c = (a, b) if a.kind == "vehicle" else (b, a)
        return vc(v, c)

    assert veh
    return links, conflict


def test_plan_gives_only_free_greens():
    links, conflict = _cross_junction()
    timing = {"yellow": lambda i: 3.5, "allred": lambda i: 2.0, "walk_speed": 1.2, "clear_speed": 1.07}
    phases = plan(links, conflict, timing)
    greens = [p for p in phases if p[4] == "green"]
    assert len(greens) == 4
    for s, *_ in phases:
        on = [i for i, ch in enumerate(s) if ch == "G"]
        assert not any(conflict(a, b) for a in on for b in on if a < b), s
    for lk in links:  # every movement and crossing served
        assert any(p[0][lk.index] == "G" for p in phases), lk
    names = [p[4] for p in phases]
    assert names[-3:] == ["walk", "walk clearance", "walk buffer"]
    walk = phases[names.index("walk")][0]
    assert all(ch == ("G" if links[i].kind == "crossing" else "r") for i, ch in enumerate(walk))
    # after each green: yellow for what ends, then all red before anything crossing it starts
    for k, p in enumerate(phases):
        if p[4] == "green":
            assert phases[k + 1][4] == "yellow" and "y" in phases[k + 1][0]


def test_direction_state():
    assert direction_state("GrrG", [0, 3]) == "G"
    assert direction_state("ryrr", [1]) == "y"
    assert direction_state("rrrr", [0, 1]) == "r"


def test_heads_left_to_right_as_the_drivers_see_them():
    """Travelling along +x in CARLA (y to the drivers' right): the mast heads from the left are the lanes from the
    left; with two lanes under three heads the right lane's straight on and right turn get a head each (same
    colour, the lane's); the low head on the post repeats the right lane."""
    boxes = [(10.0, 2.0, 6.0), (10.0, -2.0, 6.0), (10.0, 0.0, 6.0), (10.0, 4.0, 3.0)]
    # dedicated left lane, right lane straight on + right
    out = {b: (w, idx) for b, w, idx in assign_heads(boxes, 0.0, [("l", [3]), ("sr", [1, 2])])}
    assert out == {1: ("l", [3]), 2: ("s", [1, 2]), 0: ("r", [1, 2]), 3: ("sr", [1, 2])}  # the post: the right lane
    # shared lanes: left lane straight on + left, right lane straight on + right
    out = {b: w for b, w, _ in assign_heads(boxes, 0.0, [("ls", [2, 3]), ("sr", [0, 1])])}
    assert out == {1: "ls", 2: "s", 0: "r", 3: "sr"}
    # three lanes: one head each
    out = {b: w for b, w, _ in assign_heads(boxes, 0.0, [("s", [5]), ("s", [4]), ("sr", [3])])}
    assert out[1] == "s" and out[2] == "s" and out[0] == "sr"
    # travelling the other way (yaw 180): the drivers' left is +y
    out = {b: w for b, w, _ in assign_heads(boxes, 180.0, [("l", [3]), ("sr", [1, 2])])}
    assert out[0] == "l" and out[1] == "r"


def test_walkers_wait_at_the_kerb_on_dont_walk(tmp_path):
    """PedRedHold: someone on a junction's corner whose next step is a crossing at DON'T WALK is held (max speed
    near 0) and let go at WALK with their own speed; someone already on the crossing, or going along the pavement,
    is left alone."""
    from cosim.sumo_side import PedRedHold

    net = tmp_path / "n.net.xml"
    net.write_text('<net>\n<connection from=":J_w0" to=":J_c0" fromLane="0" toLane="0" tl="J" linkIndex="4" '
                   'dir="s" state="M"/>\n</net>\n')
    speed = {"a": 1.3, "b": 1.2, "c": 1.1}
    nxt = {"a": ":J_c0", "b": "e1", "c": "e2"}
    c = NS(person=NS(getNextEdge=lambda p: nxt[p], getMaxSpeed=lambda p: speed[p],
                     setSpeed=lambda p, v: speed.__setitem__(p, v)))
    h = PedRedHold(net)
    roads = {"a": ":J_w0", "b": ":J_w0", "c": ":J_c0"}
    h.step(c, {"J": "GGGGr"}, roads)
    assert speed["a"] == PedRedHold.HOLD_SPEED and speed["b"] == 1.2 and speed["c"] == 1.1
    h.step(c, {"J": "GGGGr"}, roads)
    assert h.holds == 1
    h.step(c, {"J": "rrrrG"}, roads)
    assert speed["a"] == 1.3 and not h.held


def test_every_signal_runs_one_coordinated_cycle():
    """tools/build_town05_signals.py -> cosim/coordination.py: the coordinated programs share one cycle (90-120 s),
    every car green lasts at least 15 s (8 s for a green of turns only: no more greens of 4-5 s), the walk 7 s,
    and the offsets differ from signal to signal (the green waves)."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import pytest

    f = Path(__file__).resolve().parents[1] / "scenarios" / "town05" / "town05.coordinated.add.xml"
    if not f.is_file():
        pytest.skip("no coordinated programs in this signal set")
    cycles, offs = set(), set()
    for tl in ET.parse(f).getroot().iter("tlLogic"):
        assert tl.get("programID") == "coordinated" and tl.get("type") == "static"
        ph = tl.findall("phase")
        cycles.add(round(sum(float(p.get("duration")) for p in ph), 3))
        offs.add(float(tl.get("offset")))
        for p in ph:
            if p.get("name") == "green":
                assert float(p.get("duration")) >= 8.0, (tl.get("id"), p.attrib)
            if p.get("name") == "walk":
                assert float(p.get("duration")) >= 7.0
    assert len(cycles) == 1 and 90.0 <= next(iter(cycles)) <= 125.0, cycles
    assert len(offs) > 5


def test_the_cycle_keeper_brings_a_signal_back_in_step():
    """sumo_side.CycleKeeper: a coordinated signal whose first green ends late (an emergency vehicle held a green)
    has that green shortened at its next start, one that starts early (a skipped walk, greens that ended early) is
    held green until its planned end."""
    from cosim.sumo_side import CycleKeeper

    class TL:
        def __init__(self):
            self.phase, self.switch, self.set_to = 0, 0.0, None

        def getProgram(self, t):
            return "coordinated"

        def getAllProgramLogics(self, t):
            return [NS(programID="coordinated", type=0, phases=[NS(duration=30.0, state="G", name="green"),
                                                                 NS(duration=70.0, state="r", name="all red")])]

        def getControlledLinks(self, t):
            return [[("in_0", "out_0", ":J_0_0")]]

        def getPhase(self, t):
            return self.phase

        def getNextSwitch(self, t):
            return self.switch

        def setPhaseDuration(self, t, d):
            self.set_to = d

    tl = TL()
    c = NS(trafficlight=tl)
    k = CycleKeeper()
    # cycle 100 s; first start at t=0, phase 0 ends at 30: the reference
    tl.phase, tl.switch = 0, 30.0
    k.step(c, 0.0, {"J": "G"}, {}, gap_check=False)
    assert k.ref["J"] == 30.0
    tl.phase = 1
    k.step(c, 30.0, {"J": "r"}, {"J": "G"}, gap_check=False)
    # next cycle starts 6 s late (t=106): it would end at 136, due at 130 -> shortened to 24 s
    tl.phase, tl.switch = 0, 136.0
    k.step(c, 106.0, {"J": "G"}, {"J": "r"}, gap_check=False)
    assert tl.set_to == 24.0
    tl.phase = 1
    k.step(c, 130.0, {"J": "r"}, {"J": "G"}, gap_check=False)
    # starts 40 s early (t=160): ends at 190, due at 230 -> held green to 230 (70 s)
    tl.phase, tl.switch, tl.set_to = 0, 190.0, None
    k.step(c, 160.0, {"J": "G"}, {"J": "r"}, gap_check=False)
    assert tl.set_to == 70.0


def test_every_carla_pole_shows_one_colour():
    """29 Oct: CARLA's own lamps light the signals (a CARLA traffic light has one state for all its heads), so in every
    phase of every Town05 program all the links one pole stands for (tlmap: its landmark) show the same colour. With
    SIGNAL_UNIT approach every approach goes as one; under the old lane unit 8 poles at the T junctions mixed."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    import pytest

    from cosim.tlmap import links_from_net

    town = Path(__file__).resolve().parents[1] / "scenarios" / "town05"
    if not (town / "town05.approach.net.xml").is_file():
        pytest.skip("no approach signal set")
    poles = links_from_net(town / "town05.approach.net.xml")
    assert len(poles) == 54
    for prog in ("actuated", "webster", "coordinated"):
        states = {tl.get("id"): [p.get("state") for p in tl.iter("phase")]
                  for tl in ET.parse(town / f"town05.approach.{prog}.add.xml").getroot().iter("tlLogic")}
        for lm, links in poles.items():
            assert len({t for t, _ in links}) == 1
            for s in states[links[0][0]]:
                assert len({s[i].upper() for _, i in links}) == 1, (prog, lm, s)

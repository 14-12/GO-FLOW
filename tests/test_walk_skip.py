"""sumo_side.WalkSkip skips a walk nobody waits for and goes on to the green after it (9 Nov). Before, it always
jumped to phase 0: right when the walk ends the cycle (the actuated programs), but the coordinated programs have it
in the middle of the cycle (cosim/coordination.py puts the busiest green first), so every car green after a skipped
walk was skipped too."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cosim.sumo_side import WalkSkip

ROOT = Path(__file__).resolve().parents[1]
TOWN05 = ROOT / "scenarios" / "town05"

# 2 car links (0, 1) and a crossing (2): green A, its amber and all red, the walk, its clearance and buffer, green B
MID = ["Grr", "yrr", "rrr", "rrG", "rru", "rrr", "rGr", "ryr", "rrr"]
END = ["Grr", "yrr", "rrr", "rGr", "ryr", "rrr", "rrG", "rru", "rrr"]


class _TL:
    def __init__(self, phases, program, k, durs):
        self.phases, self.program, self.k, self.set_to = phases, program, k, None
        self.durs = durs

    def getProgram(self, tid):
        return self.program

    def getAllProgramLogics(self, tid):
        return [NS(programID=self.program, type=0, phases=[NS(state=s, duration=d, name="")
                                                           for s, d in zip(self.phases, self.durs, strict=True)])]

    def getPhase(self, tid):
        return self.k

    def setPhase(self, tid, k):
        self.set_to = k
        self.k = k

    def getNextSwitch(self, tid):
        return 107.0

    def getRedYellowGreenState(self, tid):
        return self.phases[self.k]

    def setPhaseDuration(self, tid, d):
        pass

    def getPhaseName(self, tid):
        return ""


def _skip(phases, program, k):
    ws = WalkSkip.__new__(WalkSkip)
    ws.cross = {"J": [2]}
    ws.node, ws.skipped, ws.served, ws.last_served = {}, 0, 0, None
    ws._last, ws._prev, ws._clr, ws._coord, ws.bank, ws.banked, ws._phases = {}, {}, {}, {}, {}, 0.0, {}
    durs = [20.0, 3.5, 2.5, 7.0, 12.0, 2.0, 15.0, 3.5, 2.5]
    tl = _TL(phases, program, k, durs)
    c = NS(trafficlight=tl, simulation=NS(getTime=lambda: 100.0), person=NS(getNextEdge=lambda p: ""))
    tls = {"J": phases[k]}
    ws.step(c, tls, {})
    return ws, tl


def test_a_skipped_walk_in_the_middle_goes_on_to_the_next_green():
    ws, tl = _skip(MID, "coordinated", 3)
    assert tl.set_to == 6 and ws.skipped == 1           # green B, not back to green A
    assert ws.bank["J"] == pytest.approx(7.0 + 12.0 + 2.0)  # what is left of the walk, its clearance and buffer only


def test_a_skipped_walk_at_the_end_of_the_cycle_goes_to_the_first_green():
    ws, tl = _skip(END, "actuated", 6)
    assert tl.set_to == 0 and ws.skipped == 1


@pytest.mark.slow
def test_every_car_green_of_the_coordinated_programs_is_served():
    """SUMO, the coordinated programs, nobody walking: every car green of every signal comes round each cycle."""
    if not (TOWN05 / "town05.coordinated.add.xml").is_file():
        pytest.skip("no coordinated programs in this signal set")
    from cosim.sumo_side import SumoSide

    s = SumoSide(TOWN05, "coordinated", seed=5, port=8917, label="walkskip-test", extras=False,
                 extra_args=["--scale", "0.5"])
    c = s.conn
    greens, seen = {}, {}
    for t in c.trafficlight.getIDList():
        lg = next(x for x in c.trafficlight.getAllProgramLogics(t) if x.programID == c.trafficlight.getProgram(t))
        greens[t] = {k for k, ph in enumerate(lg.phases) if getattr(ph, "name", "") == "green"}
        seen[t] = set()
    try:
        while s.time < 400:
            s.step()
            if s.time >= 120:  # after a first full cycle
                for t, ks in seen.items():
                    ks.add(c.trafficlight.getPhase(t))
    finally:
        s.close()
    missed = {t: sorted(greens[t] - seen[t]) for t in greens if greens[t] - seen[t]}
    assert not missed, f"car greens never shown: {missed}"

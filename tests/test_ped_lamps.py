"""The walk-signal lamps' drawing plan (cosim/ped_lights.PedLights.plan): each lamp drawn once to last until its
signal's next switch, again only when its colour changes or its drawing runs out."""

from __future__ import annotations

from cosim.ped_lights import DRAW_M, LAMP_MAX_S, LIFE_S, RENEW_S, PedLights


def _lamps():
    pl = PedLights.__new__(PedLights)
    pl.mounted = [(0.0, 0.0, 0.0, "A", 0, True), (5.0, 0.0, 0.0, "A", 1, False),
                  (DRAW_M + 50.0, 0.0, 0.0, "B", 0, True)]  # the third is out of range
    pl._drawn, pl._posts, pl._next_due = {}, {}, None
    return pl


def test_a_lamp_is_drawn_once_until_its_signal_switches():
    pl = _lamps()
    tls = {"A": "rG", "B": "G"}
    first = pl.plan(0.0, 0.0, tls, {"A": 1.5}, now=100.0)
    assert [(i, walk) for i, walk, _, _ in first] == [(0, False), (1, True)]
    assert all(abs(life - 1.6) < 1e-9 for _, _, life, _ in first)
    assert [post for *_, post in first] == [False, True]        # the post of the lamp not on a pole
    assert pl.plan(0.0, 0.0, tls, {"A": 1.0}, now=100.5) == []   # nothing changed: no calls
    assert abs(pl._next_due - (101.6 - RENEW_S)) < 1e-9
    again = pl.plan(0.0, 0.0, tls, {"A": 3.5}, now=101.55)        # about to run out, same colours
    assert [i for i, *_ in again] == [0, 1] and all(life == LAMP_MAX_S for _, _, life, _ in again)
    assert [post for *_, post in again] == [False, False]         # the post lives longer


def test_a_lamp_whose_colour_changes_is_drawn_at_once():
    pl = _lamps()
    pl.plan(0.0, 0.0, {"A": "rr"}, {"A": 10.0}, now=0.0)
    changed = pl.plan(0.0, 0.0, {"A": "Gr"}, {"A": 9.0}, now=0.3)
    assert [(i, walk) for i, walk, _, _ in changed] == [(0, True)]


def test_without_the_next_switch_a_lamp_lives_briefly():
    pl = _lamps()
    out = pl.plan(0.0, 0.0, {"A": "rr"}, None, now=0.0)
    assert all(life == LIFE_S for _, _, life, _ in out)


def test_the_lamp_flashes_during_the_walk_clearance():
    """cosim/ped_lights.ped_aspect / clearance_states: walk while the crossing is green; flashing don't walk in its
    walk clearance (since 27 Oct the walk has a phase of its own and its clearance marks the crossings "u": red to
    the walkers, every car red); steady don't walk in the all reds and the cars' greens, though they look alike
    otherwise. Each of Town05's crossings has its clearance state: 55, 51 since the four over the ring road went
    (27 Oct; the permissive set still has them)."""
    from pathlib import Path

    from cosim.ped_lights import FLASH_S, clearance_states, ped_aspect

    net = Path(__file__).resolve().parents[1] / "scenarios" / "town05" / "town05.net.xml"
    fdw = clearance_states(net)
    assert sum(len(v) for v in fdw.values()) in (51, 55)
    if "u" not in next(iter(fdw["1126"][21])):  # the permissive set in use (tools/use_signals.py): its own states
        return
    clear = fdw["1126"][21]
    assert ped_aspect("rrrrrrrrrrrrrrrrrrrrGGGG", 21, clear) == "walk"
    assert ped_aspect("rrrrrrrrrrrrrrrrrrrruuuu", 21, clear) == "flashing"
    assert ped_aspect("rrrrrrrrrrrrrrrrrrrrrrrr", 21, clear) == "dont_walk"   # the buffer and the all reds
    assert ped_aspect("rrrrrrrrrrrrrrrrGGGGrrrr", 21, clear) == "dont_walk"
    # the 26 Oct programs (the clearance with the parallel cars' green, no "u"): by its state
    assert ped_aspect("GgGGgrrrrrGgGGgrrrrrrrrr", 21, {"GgGGgrrrrrGgGGgrrrrrrrrr"}) == "flashing"
    pl = _lamps()
    pl.fdw = {"A": {0: {"rG"}}}
    on = pl.plan(0.0, 0.0, {"A": "rG", "B": "G"}, {"A": 5.0}, now=100.0)       # lamp 0 flashing, on
    assert (0, False) in [(i, w) for i, w, _, _ in on] and pl.flashing == {0}
    assert pl.plan(0.0, 0.0, {"A": "rG", "B": "G"}, {"A": 4.5}, now=100.0 + FLASH_S) == []  # off: nothing drawn
    again = pl.plan(0.0, 0.0, {"A": "rG", "B": "G"}, {"A": 4.0}, now=100.0 + 2 * FLASH_S)
    assert [i for i, *_ in again] == [0]

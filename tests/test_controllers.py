"""Per-step controllers (cosim/controllers.py): the transitions and program timing."""

from cosim.controllers import (
    all_red_state,
    greens,
    program_moves,
    program_timing,
    transition_state,
)


def test_transition_keeps_shared_greens_and_yellows_the_rest():
    assert transition_state("GGgrrGGG", "GGGrrrrr") == "GGgrryyy"
    assert transition_state("GGGrrrrr", "rrrGGGrr") == "yyyrrrrr"
    assert all_red_state("GGgrrGGG", "GGGrrrrr") == "GGgrrrrr"


def test_program_timing_reads_yellow_and_all_red():
    assert program_timing([("GGrr", 42), ("yyrr", 3), ("rrrr", 2), ("rrGG", 30), ("rryy", 4)]) == (4, 2)
    assert program_timing([("GGrr", 42), ("yyrr", 3), ("rrGG", 30), ("rryy", 3)]) == (3, 0.0)


def test_program_moves_follow_the_cycle():
    raw = [("GGrr", 30), ("yyrr", 3), ("ggGG", 10), ("yyyy", 3), ("rrGG", 20), ("rryy", 3)]
    phases = ["GGrr", "ggGG", "rrGG"]
    assert program_moves(raw, phases) == {(0, 1), (1, 2), (2, 0)}

"""Tests du Safety Controller : actions invalides injectées volontairement."""
import numpy as np
import pytest

from environment.intersection import Intersection
from safety.validator import ModeSafety, SafetyConfig, SafetyLog, SignalSafety

INTER = Intersection("C", {"N": "N2C", "E": "E2C", "S": "S2C", "W": "W2C"}, {}, {}, {}, [0, 2], {0: 1, 2: 3},
                     {"N": 0, "S": 0, "E": 2, "W": 2},
                     {"GGGgrrrrGGGgrrrr", "yyyyrrrryyyyrrrr", "rrrrGGGgrrrrGGGg", "rrrryyyyrrrryyyy"})


@pytest.fixture
def sig():
    return SignalSafety(SafetyConfig(), SafetyLog())


def test_mask_min_green_forces_keep(sig):
    assert sig.action_mask(INTER, 0, 4.0, False).tolist() == [True, False]


def test_mask_max_green_forces_switch(sig):
    assert sig.action_mask(INTER, 0, 61.0, False).tolist() == [False, True]


def test_mask_during_yellow(sig):
    assert sig.action_mask(INTER, 2, 0.0, True).tolist() == [False, True]


@pytest.mark.parametrize("target,tig,trans,age,expected", [
    (1, 20, False, 0, "absente"),          # phase jaune demandée
    (9, 20, False, 0, "absente"),          # phase inexistante
    (2, 5, False, 0, "vert minimal"),      # vert trop court
    (2, 20, False, 5, "trop ancienne"),    # observation périmée
    (0, 70, False, 0, "vert maximal"),     # maintien au-delà du max
    (2, 0, True, 0, "transition"),         # changement pendant le jaune
])
def test_invalid_actions_rejected(sig, target, tig, trans, age, expected):
    ok, reason = sig.validate(INTER, target, 0, tig, trans, age, 100.0)
    assert not ok and expected in reason
    assert sig.log.rejected_actions == 1


def test_valid_switch_accepted(sig):
    assert sig.validate(INTER, 2, 0, 15, False, 0.0, 10.0) == (True, "ok")


def test_arbitrary_state_detected(sig):
    assert not sig.verify_state(INTER, "GGGGGGGGGGGGGGGG", 1.0)  # tous verts = conflit
    assert sig.log.state_violations == 1
    assert sig.verify_state(INTER, "rrrrGGGgrrrrGGGg", 2.0)


AMB = {"vehicle_id": "amb", "priority_class": "EMERGENCY_MEDICAL", "siren_on": True, "confirmed": True}


def ms():
    return ModeSafety(SafetyConfig(), SafetyLog())


def test_mode_unknown_and_bad_tls():
    m = ms()
    assert not m.validate(0, "C", "TURBO", {"C"}, "N", {"N"}, AMB)[0]
    assert not m.validate(0, "X", "NORMAL", {"C"}, None, set(), None)[0]


def test_mode_requires_real_emergency():
    m = ms()
    assert not m.validate(0, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, None)[0]
    assert not m.validate(0, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**AMB, "siren_on": False})[0]
    assert not m.validate(0, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**AMB, "confirmed": False})[0]
    assert not m.validate(0, "C", "EMERGENCY_PREEMPTION", {"C"}, "Z", {"N"}, AMB)[0]
    assert m.validate(0, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, AMB)[0]


def test_transit_mode_only_for_buses():
    m = ms()
    assert not m.validate(0, "C", "TRANSIT_PRIORITY", {"C"}, "N", {"N"}, AMB)[0]
    bus = {"vehicle_id": "b", "priority_class": "TRANSIT", "siren_on": False, "confirmed": True}
    assert m.validate(0, "C", "TRANSIT_PRIORITY", {"C"}, "N", {"N"}, bus)[0]


POL = {**AMB, "priority_class": "POLICE", "priority_level": 2}


def test_preemption_rate_limit():
    m = ms()
    for i in range(3):
        assert m.validate(10 * i, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": f"a{i}"})[0]
    ok, reason = m.validate(40, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": "a9"})
    assert not ok and "trop de préemptions" in reason
    # la même urgence peut être reconfirmée, et la fenêtre glisse
    assert m.validate(41, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": "a1"})[0]
    assert m.validate(400, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": "a9"})[0]


def test_rate_limit_never_blocks_level3():
    """Ambulance et pompiers (niveau 3) ne sont jamais bloqués par la limite de fréquence."""
    m = ms()
    for i in range(3):
        assert m.validate(10 * i, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": f"p{i}"})[0]
    assert not m.validate(40, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, {**POL, "vehicle_id": "p9"})[0]
    for cls in ("EMERGENCY_MEDICAL", "FIRE"):
        v = {**AMB, "priority_class": cls, "priority_level": 3, "vehicle_id": cls}
        assert m.validate(45, "C", "EMERGENCY_PREEMPTION", {"C"}, "N", {"N"}, v)[0]

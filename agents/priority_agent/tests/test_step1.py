"""Tests de l'étape 1 : lecture, commande contrôlée, bluelight, V2I."""
from pathlib import Path

import pytest

from tools.sumo_interface import PRIORITY_LEVEL, SumoSession

CFG = Path(__file__).parents[1] / "simulation/scenarios/single_base.sumocfg"
TLS = "C"


@pytest.fixture
def sess():
    s = SumoSession(CFG, label="pytest", seed=42).start()
    yield s
    s.close()


def run_until(s, t):
    while s.time < t:
        s.step()
        s.apply_pending_targets()


def test_two_green_phases(sess):
    assert sess.get_green_phases(TLS) == [0, 2]


def test_state_is_readable(sess):
    run_until(sess, 60)
    st = sess.get_intersection_state(TLS)
    assert st["phase_index"] in range(4)
    assert len(st["lanes"]) == 8
    assert st["total_queue"] >= 0


def test_yellow_and_unknown_phases_rejected(sess):
    run_until(sess, 5)
    assert sess.request_green_phase(TLS, 1)["status"] == "rejected"
    assert sess.request_green_phase(TLS, 3)["status"] == "rejected"
    assert sess.request_green_phase(TLS, 9)["status"] == "rejected"


def test_switch_goes_through_yellow(sess):
    run_until(sess, 5)
    current = sess.conn.trafficlight.getPhase(TLS)
    target = 2 if current == 0 else 0
    assert sess.request_green_phase(TLS, target)["status"] == "transition"
    sess.step()
    assert "y" in sess.conn.trafficlight.getRedYellowGreenState(TLS)
    run_until(sess, sess.time + 4)
    assert sess.verify_signal_state(TLS, target)


def test_ambulance_has_no_bluelight(sess):
    run_until(sess, 132)
    assert "amb_0" in sess.conn.vehicle.getIDList()
    assert sess.conn.vehicle.getVehicleClass("amb_0") == "emergency"
    assert not sess.has_bluelight_device("amb_0")


def test_v2i_request_emitted_within_range(sess):
    seen = None
    while sess.time < 160 and seen is None:
        sess.step()
        for r in sess.get_priority_requests(TLS):
            if r.vehicle_id == "amb_0":
                seen = r
    assert seen is not None
    assert seen.priority_class == "EMERGENCY_MEDICAL"
    assert seen.approach_edge == "N2C"
    assert seen.distance_m <= sess.v2i_range_m
    assert sess.get_approach_green_phase(TLS, "N2C") == 0


# --------------------------------------------------------------------------- urgences et sirène
def test_all_emergency_types_without_bluelight(sess):
    seen = {}
    while sess.time < 700:
        sess.step()
        for vid in sess.conn.simulation.getDepartedIDList():
            cls = sess.get_priority_class(vid)
            if cls in ("EMERGENCY_MEDICAL", "FIRE", "POLICE"):
                seen[vid] = cls
                assert not sess.has_bluelight_device(vid), vid
    assert seen == {"amb_0": "EMERGENCY_MEDICAL", "police_patrol": "POLICE",
                    "fire_0": "FIRE", "police_0": "POLICE"}


def test_patrol_without_siren_sends_no_request(sess):
    while sess.time < 300:
        sess.step()
        assert all(r.vehicle_id != "police_patrol" for r in sess.get_priority_requests(TLS))
    assert sess.priority_tracks["police_patrol"].siren_ever_on is False


def test_siren_switched_on_mid_route_creates_request(sess):
    run_until(sess, 252)
    assert "police_patrol" in sess.conn.vehicle.getIDList()
    sess.set_siren("police_patrol", True)          # la patrouille part en intervention
    found = None
    while sess.time < 280 and found is None:
        sess.step()
        found = next((r for r in sess.get_priority_requests(TLS) if r.vehicle_id == "police_patrol"), None)
    assert found is not None and found.siren_on and found.priority_class == "POLICE"


def test_siren_switched_off_cancels_request(sess):
    req = None
    while sess.time < 160 and req is None:
        sess.step()
        req = next((r for r in sess.get_priority_requests(TLS) if r.vehicle_id == "amb_0"), None)
    assert req is not None
    sess.set_siren("amb_0", False)                 # fin d'intervention : sirène coupée
    assert all(r.vehicle_id != "amb_0" for r in sess.get_priority_requests(TLS))


def test_requests_sorted_by_priority_level(sess):
    run_until(sess, 140)
    c = sess.conn
    # Ajout simultané d'une police (niveau 2) et d'un camion de pompiers (niveau 3).
    c.vehicle.add("test_police", "r_EW", typeID="police", departSpeed="max")
    c.vehicle.add("test_fire", "r_WE", typeID="fire_truck", departSpeed="max")
    reqs = []
    while sess.time < 175:
        sess.step()
        reqs = [r for r in sess.get_priority_requests(TLS)
                if r.vehicle_id in ("test_police", "test_fire", "amb_0")]
        if len(reqs) >= 2:
            break
    levels = [r.priority_level for r in reqs]
    assert levels == sorted(levels, reverse=True)
    assert PRIORITY_LEVEL["FIRE"] == PRIORITY_LEVEL["EMERGENCY_MEDICAL"] > PRIORITY_LEVEL["POLICE"] \
        > PRIORITY_LEVEL["TRANSIT"]

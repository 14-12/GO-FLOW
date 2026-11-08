"""The route search on Town05 (router.py): drivable routes, alerts on the right road, different options."""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapter_traffic import accident_to_alert, forecast_to_alerts, load_approach_map
from router import Router


@pytest.fixture(scope="module")
def r():
    return Router()


def test_every_passenger_road_is_in_the_graph(r):
    assert len(r.roads) == 80          # Town05's 80 normal edges, the two parallel ring-road ones included
    assert {"35", "37"} <= set(r.roads)


def test_best_routes_are_drivable_for_every_pair_of_roads(r):
    ids = sorted(r.roads)
    n = 0
    for a, b in itertools.permutations(ids, 2):
        p = r.best(a, b)
        if p is None:
            continue
        n += 1
        assert p[0] == a and p[-1] == b
        assert r.drivable(p), (a, b, p)
    assert n > 0.9 * len(ids) * (len(ids) - 1)


def test_every_option_is_drivable(r):
    ids = sorted(r.roads)
    for a, b in itertools.islice(itertools.permutations(ids, 2), 0, None, 7):
        for p in r.k_routes(a, b):
            assert r.drivable(p), (a, b, p)


def test_options_differ_and_come_cheapest_first(r):
    opts = r.k_routes("11", "-48")
    assert 1 <= len(opts) <= 3
    costs = [r.route_cost(p, {}) for p in opts]
    assert costs == sorted(costs)
    for p, q in itertools.combinations(opts, 2):
        assert len(r._inner(q) & r._inner(p)) / max(len(r._inner(q)), 1) < 0.7


def test_an_alert_on_a_parallel_ring_road_stays_on_it(r):
    """35 and 37 run between the same junctions as another road: the penalty must hit 35 itself."""
    pen = r.penalties([{"route": "35", "penalty": 50.0}])
    assert pen == {"35": 50.0}
    twin = [x.id for x in r.roads.values() if (x.frm, x.to) == (r.roads["35"].frm, r.roads["35"].to) and x.id != "35"]
    assert twin and r.cost(twin[0], pen) == pytest.approx(r.roads[twin[0]].free_s)


def test_a_blocked_road_is_avoided_when_another_way_exists(r):
    ref = r.best_between("24.11", "359")
    mid = ref[len(ref) // 2]
    al = accident_to_alert({"edge": mid, "closed_lanes": 2, "total_lanes": 2})
    new = r.best_between("24.11", "359", [al])
    assert mid not in new and r.drivable(new)
    assert r.avoided(ref, new, [al]) == [al]


def test_dso4_approaches_are_roads_of_the_graph(r):
    amap = load_approach_map()
    assert len(amap) == 55 and set(amap.values()) <= set(r.roads)


def test_forecast_to_alerts():
    amap = load_approach_map()
    msg = {"type": "forecast", "nodes": {"1427-1": {"fill": [0.6, 0.8, 0.9]}, "1427-2": {"fill": [0.1, 0.2, 0.2]}}}
    al = forecast_to_alerts(msg, amap)
    assert al == [{"agent": 2, "route": amap["1427-1"], "type": "congestion", "node": "1427-1", "penalty": 2.6,
                   "fill": 0.8}]


def test_typical_times_change_costs_not_the_free_flow_time(r):
    import json

    from router import Router

    card = json.loads((Path(__file__).resolve().parents[3] / "models" / "route_guidance" / "movement_times.json")
                      .read_text(encoding="utf-8"))
    t = Router()
    t.set_typical(card, routing=True)
    p = r.best("11", "-48")
    assert t.free_s(p) == pytest.approx(r.free_s(p))                 # free flow stays free flow
    assert t.typical_s(p, skip_first=False) > 2 * r.free_s(p, skip_first=False)
    assert t.route_cost(p) == pytest.approx(t.typical_s(p, skip_first=False))
    q = t.best("11", "-48")
    assert q is not None and t.drivable(q)

"""Town scenario builder (cosim/town.py): Webster timing and flows."""

from pathlib import Path

from cosim.town import link_flows, webster

PHASES = [("GGrr", 42.0), ("yyrr", 3.0), ("rrrr", 3.0), ("rrGG", 42.0), ("rryy", 3.0), ("rrrr", 3.0)]


def test_webster_textbook_case():
    # y1 = 600/1800, y2 = 300/1800 -> Y = 0.5, L = 12 s: C = (1.5*12 + 5) / 0.5 = 46 s,
    # effective green 34 s split 2:1.
    out = webster(PHASES, {0: 600.0, 1: 200.0, 2: 300.0, 3: 100.0})
    assert [d for _, d in out] == [23.0, 3.0, 3.0, 11.0, 3.0, 3.0]
    assert [s for s, _ in out] == [s for s, _ in PHASES]


def test_webster_clamps_cycle_and_min_green():
    no_traffic = webster(PHASES, {})
    assert sum(d for _, d in no_traffic) == 30.0  # (1.5*12 + 5) = 23 s -> raised to 30
    assert no_traffic[0][1] == no_traffic[3][1] == 9.0
    saturated = webster(PHASES, {0: 1700.0, 2: 10.0})
    assert sum(d for _, d in saturated) <= 120.0 + 5.0
    assert saturated[3][1] == 5.0  # min green


def test_link_flows_share_an_edge_pair_over_its_lanes(tmp_path: Path):
    net = tmp_path / "n.net.xml"
    net.write_text("""<net>
      <connection from="a" to="b" fromLane="0" toLane="0" tl="J" linkIndex="0"/>
      <connection from="a" to="b" fromLane="1" toLane="1" tl="J" linkIndex="1"/>
      <connection from="a" to="c" fromLane="1" toLane="0" tl="J" linkIndex="2"/>
      <connection from="b" to="d" fromLane="0" toLane="0"/>
    </net>""")
    rou = tmp_path / "r.rou.xml"
    rou.write_text("""<routes>
      <vehicle id="1" type="pas_passenger" depart="0"><route edges="a b d"/></vehicle>
      <vehicle id="2" type="bus_bus" depart="10"><route edges="a c"/></vehicle>
      <vehicle id="3" type="pas_passenger" depart="4000"><route edges="a c"/></vehicle>
    </routes>""")
    assert link_flows(net, [rou]) == {"J": {0: 0.5, 1: 0.5, 2: 2.0}}


def test_webster_adds_links_that_share_a_lane():
    # Phase 1 serves lane a_0 by links 0 (straight, 400) and 1 (right, 200): the lane carries
    # 600 veh/h, not 400. Phase 2: one lane, 300.
    lanes = {0: "a_0", 1: "a_0", 2: "b_0", 3: "b_1"}
    flows = {0: 400.0, 1: 200.0, 2: 300.0, 3: 100.0}
    out = webster(PHASES, flows, lanes=lanes)
    # Y = 600/1800 + 300/1800 = 0.5 -> C = 46 s, greens 2:1 of 34 s.
    assert [d for _, d in out] == [23.0, 3.0, 3.0, 11.0, 3.0, 3.0]
    assert webster(PHASES, flows)[0][1] < 23.0  # per link only: 400 critical, smaller share

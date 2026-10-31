"""Town05's signal programs as a real junction runs them, every green a free go (27 Oct) (tools/build_town05_signals.py), checked against SUMO's
own junction logic (cosim/signal_audit.py) and, for the agents, against what a simulation shows step by step
(cosim/signal_monitor.py)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from cosim.signal_audit import (
    PED_CLEAR_SPEED,
    Junction,
    Link,
    audit,
    check_program,
    load,
    side,
)

ROOT = Path(__file__).resolve().parents[1]
TOWN05 = ROOT / "scenarios" / "town05"
NET = TOWN05 / "town05.net.xml"
ADDS = [TOWN05 / f"town05.{p}.add.xml" for p in ("actuated", "webster", "coordinated")
        if (TOWN05 / f"town05.{p}.add.xml").is_file()]
FOUR_WAY = ["1126", "1292", "1427", "1574", "1722", "207", "2086", "53", "838", "979"]


def _signals_in_use() -> str | None:
    """Which signal set tools/use_signals.py finds in scenarios/town05: "approach" (the default since 29 Oct), "split" (27 Oct),
    "leadlag", "permissive" (the 26 Oct programs) or None."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("use_signals", ROOT / "tools" / "use_signals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.in_use()


SIGNALS = _signals_in_use()
protected = pytest.mark.skipif(SIGNALS == "permissive", reason="the permissive set (26 Oct) lets turns give way")


@pytest.fixture(scope="module")
def rep():
    return audit(NET, ADDS)


@pytest.fixture(scope="module")
def junctions():
    return load(NET, ADDS)


@protected
def test_every_program_passes_the_audit(rep):
    """Static, actuated and Webster programs of the 15 signals: every link has a state in every phase and a green in
    the cycle; yellow >= 3 s and all red >= 1 s before a conflicting green; walk end -> conflicting green >= crossing
    / 3.5 ft/s with a buffer >= 2 s; and (27 Oct) no green has to give way: no "g", no two links that cross each
    other (cars or a crossing) green at once."""
    assert len(rep) == 15
    for tid, v in rep.items():
        assert set(v["programs"]) >= {"0", "actuated", "webster"}, tid
        for pid, p in v["programs"].items():
            assert p["errors"] == [], (tid, pid, p["errors"][:3])
            assert p["protected_errors"] == [], (tid, pid, p["protected_errors"][:3])


@protected
def test_heads_map_to_the_movements_of_one_approach(rep, junctions):
    """Each CARLA pole (linkSignalID landmark) shows the movements of one approach and every vehicle movement of an
    approach is on its pole. Four-way junctions: the right lane goes straight on and turns right, the left lane
    straight on and left (as CARLA has them); the parking lanes (53, 979, 1722) have no movement at a signal."""
    for tid, jn in junctions.items():
        by_pole: dict[str, set[str]] = {}
        for lk in jn.links:
            if lk.kind == "vehicle" and lk.landmark:
                by_pole.setdefault(lk.landmark, set()).add(lk.approach)
        assert by_pole, tid
        assert all(len(a) == 1 for a in by_pole.values()), (tid, by_pole)
        assert not any(lk.kind == "vehicle" and lk.lane_type == "parking" for lk in jn.links), tid
    for tid in FOUR_WAY:
        aps = rep[tid]["approaches"]
        assert set(aps) == set("NESW"), tid
        for side_, ap in aps.items():
            lanes = sorted(tuple(sorted(d)) for d in ap["lanes"].values())
            assert lanes == [("l", "s"), ("r", "s")], (tid, side_, ap["lanes"])
            assert len(ap["landmarks"]) == 1


@protected
def test_no_car_waits_inside_the_junction_for_a_green(junctions):
    """27 Oct: a green is a free go. In every phase of every program no link is "g" (green but give way) and no two
    green links cross each other, cars or crossings, by SUMO's own junction geometry; so a car that has its green
    never stops in the junction for an oncoming car or for someone walking. Four-way junctions (two lanes, each with
    a turn): four car greens a cycle, each approach on its own with all its directions, and one walk for all the
    crossings with every car red. (With the lead-lag set, tools/use_signals.py: six, each approach on its own and
    between them the two right lanes of the road together, its left lanes red.)"""
    for tid, jn in junctions.items():
        for pid, phases in jn.programs.items():
            for s, *_ in phases:
                assert "g" not in s, (tid, pid, s)
                green = [i for i, ch in enumerate(s) if ch == "G" and jn.links[i].kind in ("vehicle", "crossing")]
                for a in green:
                    for b in green:
                        assert a >= b or not jn.conflict(a, b), (tid, pid, s, a, b)
    for tid in FOUR_WAY:
        jn = junctions[tid]
        greens = [s for s, _, _, nm in jn.programs["0"] if nm == "green"]
        assert len(greens) == (6 if SIGNALS == "leadlag" else 4), (tid, SIGNALS)
        served = []
        for s in greens:
            on = {jn.links[i].approach for i, ch in enumerate(s) if ch == "G" and jn.links[i].kind == "vehicle"}
            if len(on) == 2:  # lead-lag: the two right lanes of a road together, its left lanes red
                assert SIGNALS == "leadlag" and on in ({"N", "S"}, {"E", "W"}), (tid, s, on)
                assert all(lk.dir != "l" for lk in jn.links if lk.kind == "vehicle" and s[lk.index] == "G"), (tid, s)
                continue
            assert len(on) == 1, (tid, s, on)
            a = on.pop()
            assert all(s[lk.index] == "G" for lk in jn.links if lk.kind == "vehicle" and lk.approach == a), (tid, s)
            served.append(a)
        assert sorted(served) == sorted("NESW"), (tid, served)
        walks = [s for s, _, _, nm in jn.programs["0"] if nm == "walk"]
        assert len(walks) == 1, tid
        w = walks[0]
        assert all(ch == ("G" if jn.links[i].kind == "crossing" else "r") for i, ch in enumerate(w)
                   if jn.links[i].kind in ("vehicle", "crossing")), (tid, w)


@protected
def test_each_lane_shows_one_colour(junctions):
    """Each head over the road shows a lane (cosim/signal_heads.py): all the directions of a lane are green, yellow
    or red together in every phase. Under the split set the heads of a pole differ where its lanes do (the ring
    road's left-turn lane red while its straight-on lanes go, at 2014 and 2328); under the approach set (29 Oct) no
    pole's heads differ, so CARLA's lamps can show them."""
    col = {"G": "G", "g": "G", "y": "y", "r": "r"}
    differ = set()
    for tid, jn in junctions.items():
        lanes: dict[str, list[int]] = {}
        pole_of: dict[str, str] = {}
        for lk in jn.links:
            if lk.kind == "vehicle":
                lanes.setdefault(lk.from_lane, []).append(lk.index)
                pole_of[lk.from_lane] = lk.landmark or ""
        for pid, phases in jn.programs.items():
            for s, *_ in phases:
                poles: dict[str, set[str]] = {}
                for ln, idx in lanes.items():
                    assert len({col[s[i]] for i in idx}) == 1, (tid, pid, ln, s)
                    poles.setdefault(pole_of[ln], set()).add(col[s[idx[0]]])
                differ |= {(tid, p) for p, c in poles.items() if p and len(c) > 1}
    if SIGNALS == "approach":
        assert not differ, differ
    else:
        assert {t for t, _ in differ} >= {"2014", "2328"}, differ


@protected
def test_walk_clearance_and_buffer_cover_the_crossing(junctions):
    """The flashing don't walk (walk clearance) plus the phases after it with no green (the steady don't walk
    buffer) last at least crossing length / 3.5 ft/s, the buffer at least 2 s (MUTCD 4I.06); the 24.5 m crossings
    over the ring road at 2014 and 2328 get 21 s of flashing don't walk."""
    for tid, jn in junctions.items():
        for pid, phases in jn.programs.items():
            for k, (s, d, _, nm) in enumerate(phases):
                if nm != "walk clearance":
                    continue
                was = phases[k - 1][0]
                ended = [i for i, lk in enumerate(jn.links) if lk.kind == "crossing" and was[i] == "G" and s[i] in "ru"]
                assert ended, (tid, pid, k)
                buffer = 0.0
                for q in range(k + 1, k + len(phases)):
                    st, dq, _, _ = phases[q % len(phases)]
                    if any(c in "Gg" for c in st):
                        break
                    buffer += dq
                need = max(jn.links[i].length for i in ended) / PED_CLEAR_SPEED
                assert d + buffer >= need - 1e-6, (tid, pid, k, d, buffer, need)
                assert buffer >= 2.0, (tid, pid, k, buffer)  # MUTCD 4I.06: steady don't walk >= 2 s
    for tid in ("2014", "2328"):  # (since 27 Oct no crossing over the ring road: the side road's 14 m crossing)
        assert max(d for _, d, _, nm in load(NET)[tid].programs["0"] if nm == "walk clearance") == 12.0


def test_webster_keeps_the_walks():
    """tools/build_town05_signals.webster_walks: Webster's split leaves the walk clearances as the network has them,
    the walk at least 7 s (it gave it 5 s) and each car green at least 5 s."""
    net = {tl.get("id"): tl for tl in ET.parse(NET).getroot().iter("tlLogic")}
    for tl in ET.parse(TOWN05 / "town05.webster.add.xml").getroot().iter("tlLogic"):
        ref = net[tl.get("id")].findall("phase")
        for p, r in zip(tl.findall("phase"), ref, strict=True):
            assert p.get("state") == r.get("state")
            if r.get("name") == "walk clearance":
                assert p.get("duration") == r.get("duration") and p.get("name") == "walk clearance"
            if r.get("name") == "walk":
                assert float(p.get("duration")) >= 7.0
            if r.get("name") == "green":
                assert float(p.get("duration")) >= 5.0


def _toy() -> Junction:
    """Two crossing through streams 0 and 1, a permissive left 2 that gives way to 0, a crossing 3 over 1's road."""
    links = [Link(0, "vehicle", "N", dir="s"), Link(1, "vehicle", "E", dir="s"), Link(2, "vehicle", "S", dir="l"),
             Link(3, "crossing", crossing=":J_c0", length=14.0)]
    foes = [{1, 2}, {0, 2, 3}, {0, 1}, {1}]
    resp = [set(), {3}, {0}, set()]
    return Junction("J", links, foes, resp)


def test_the_audit_catches_unsafe_programs():
    jn = _toy()
    ok = [("GrgG", 20, 20, "green"), ("Grgr", 13, 13, "walk clearance"), ("yryr", 3.5, 3.5, ""),
          ("rrrr", 2, 2, ""), ("rGrr", 20, 20, "green"), ("ryrr", 3.5, 3.5, ""), ("rrrr", 2, 2, "")]
    assert check_program(jn, ok) == []
    both = [("GGrr", 20, 20, ""), ("yyrr", 3, 3, ""), ("rrrr", 1, 1, "")]
    assert any("neither gives way" in e for e in check_program(jn, both))
    left_as_priority = [("GrGr", 20, 20, ""), ("yryr", 3, 3, ""), ("rrrr", 1, 1, ""), ("rGrr", 20, 20, ""),
                        ("ryrr", 3, 3, ""), ("rrrr", 1, 1, "")]
    assert any("'G' (priority) but gives way" in e for e in check_program(jn, left_as_priority))
    no_yellow = [("Grgr", 20, 20, ""), ("rrrr", 2, 2, ""), ("rGrr", 20, 20, ""), ("ryrr", 3, 3, ""),
                 ("rrrr", 2, 2, "")]
    assert any("yellow 0 s" in e for e in check_program(jn, no_yellow))
    no_allred = [("Grgr", 20, 20, ""), ("yryr", 3, 3, ""), ("rGrr", 20, 20, ""), ("ryrr", 3, 3, ""),
                 ("rrrr", 2, 2, "")]
    assert any("all red 0 s" in e for e in check_program(jn, no_allred))
    short_walk_clear = [("GrgG", 20, 20, "green"), ("Grgr", 5, 5, "walk clearance"), ("yryr", 3, 3, ""),
                        ("rrrr", 1, 1, ""), ("rGrr", 20, 20, ""), ("ryrr", 3, 3, ""), ("rrrr", 1, 1, "")]
    assert any("crossing link 3" in e for e in check_program(jn, short_walk_clear))
    stale = [("Grg", 20, 20, "")]
    assert any("states for 4 links" in e for e in check_program(jn, stale))


def test_side_of_the_junction():
    assert side(0.0) == "S" and side(90.0) == "W" and side(180.0) == "N" and side(270.0) == "E"


# ------------------------------------------------------------------ pedestrian demand (cosim/agents_actuated.py)


def _ped_agents(people: dict[str, tuple[str, str]]):
    """ActuatedAgents with a two-green junction (crossing link 4 walks with green 0, 5 with green 3) and people
    {id: (road, next edge)}."""
    from cosim.agents_actuated import ActuatedAgents, _Agent

    tc = NS(person=NS(getAllSubscriptionResults=lambda: {p: {} for p in people},
                      getIDList=lambda: list(people), getRoadID=lambda p: people[p][0],
                      getNextEdge=lambda p: people[p][1], getWaitingTime=lambda p: 10.0),
            inductionloop=NS(getTimeSinceDetection=lambda d: 1e9),
            lane=NS(getLastStepVehicleNumber=lambda ln: 0, getLastStepHaltingNumber=lambda ln: 0))
    ag = ActuatedAgents(fallback=set())
    states = ["GGrrGr", "yyrrrr", "rrrrrr", "rrGGrG", "rryyrr", "rrrrrr"]
    a = _Agent("J", states, ["car", "yellow", "red", "car", "yellow", "red"], [7.0] * 6, [45.0] * 6, [0, 3],
               {0: 5, 3: 2}, {0: [("n_0", "s_0")], 3: [("e_0", "w_0")]}, {4: ":J_c0", 5: ":J_c1"}, {0: [], 3: []})
    ag._cache = {}
    ag._approach = lambda tc, a, k: (0, 0)  # no vehicles anywhere
    ag._pedestrians(tc)
    return ag, a, tc


def test_people_waiting_call_the_green_their_walk_runs_with():
    """Since 25 Oct the walks run with the car greens, so someone waiting at a crossing is demand for the car green
    whose walk serves that crossing (a pedestrian call), not for the other one; a green with no cars and nobody
    waiting has none."""
    ag, a, tc = _ped_agents({"p1": (":J_w1", ":J_c1")})
    assert ag._waiting == {":J_c1": 1}
    assert ag._demand(tc, a, 3, now=100.0) and not ag._demand(tc, a, 0, now=100.0)
    ag, a, tc = _ped_agents({})
    assert not ag._demand(tc, a, 3, now=100.0) and not ag._demand(tc, a, 0, now=100.0)


def test_only_people_waiting_at_a_crossing_count():
    """Counted: on a walking area, next edge a crossing (waiting, or just arriving, at it). Not counted: already on
    the crossing, walking along a pavement, or on a walking area heading to a pavement. Each person counts once, for
    the one crossing they go to, so nobody is counted for two greens."""
    ag, a, tc = _ped_agents({"on": (":J_c1", "e_1"), "pave": ("e_1", ":J_w1"), "turn": (":J_w0", "n_1"),
                             "w0": (":J_w0", ":J_c0"), "w1": (":J_w2", ":J_c0"), "w2": (":J_w1", ":J_c1")})
    assert ag._waiting == {":J_c0": 2, ":J_c1": 1}
    assert ag._peds(a, 0) == 2 and ag._peds(a, 3) == 1
    assert ag._peds(a, 1) == 0 and ag._peds(a, 2) == 0  # yellow and all red serve no crossing


def test_skip_serves_the_waiting_pedestrians():
    """In the all red the agent jumps over a green nobody needs, but not over one with someone waiting to walk:
    with no cars anywhere and one person at the crossing of green 3, the all red before 3 does not jump back to 0."""
    ag, a, tc = _ped_agents({"p1": (":J_w1", ":J_c1")})
    assert ag._choose(tc, a, 3, now=100.0) == 3
    ag, a, tc = _ped_agents({"p1": (":J_w0", ":J_c0")})
    assert ag._choose(tc, a, 3, now=100.0) == 0      # green 3 empty, green 0's walk wanted: straight back to 0
    ag, a, tc = _ped_agents({})
    a.passed_over[3] = ag.skip_limit - 1
    assert ag._choose(tc, a, 3, now=100.0) == 3      # passed over too often: served even when empty


# ------------------------------------------------------------------ SUMO -> CARLA


def test_carla_tl_state_most_restrictive():
    """cosim/convert.tl_state: one colour for a CARLA pole's links, the most restrictive when they differ (shown only
    when the per-head drawing is off: cosim/signal_heads.py switches the poles' own lamps off and lights each head)."""
    carla = pytest.importorskip("carla")
    from cosim.convert import tl_state

    assert tl_state(["G", "g", "G"]) == (carla.TrafficLightState.Green, False)
    assert tl_state(["y", "y"]) == (carla.TrafficLightState.Yellow, False)
    assert tl_state(["G", "r"]) == (carla.TrafficLightState.Red, True)


# ------------------------------------------------------------------ in a simulation


@pytest.mark.slow
def test_the_agents_never_show_an_unsafe_signal():
    """10 minutes of Town05 at x0.8 with pedestrians and an emergency vehicle every 60 s, the junction agents on
    exact counts: at every step and every signal no conflicting greens without one giving way, no green straight to
    red, no conflicting green sooner than 4 s after a vehicle's green ended, nobody released onto a crossing whose
    clearance had not run, and nobody stepping onto a crossing at DON'T WALK (cosim/sumo_side.PedRedHold)."""
    from cosim.agents_actuated import ActuatedAgents
    from cosim.signal_monitor import SignalMonitor
    from cosim.sumo_side import SumoSide

    ctl = ActuatedAgents()
    s = SumoSide(TOWN05, "agents", seed=3, port=8931, label="signal-monitor-test", policy=ctl, extras=True,
                 emergency_every=60.0, extra_args=["--scale", "0.8"])
    s.extras.config.pedestrian_trips = True
    ex = s.extras
    ctl.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    mon = SignalMonitor(NET)
    try:
        while s.time < 600:
            r = s.step()
            mon.step(s.conn, r.time, r.tls)
    finally:
        s.close()
    v = mon.report()["violations"]
    for kind in ("conflicting_greens", "green_to_red", "short_intergreen", "ped_short_clearance",
                 "ped_started_on_red"):
        assert v.get(kind, 0) == 0, (kind, mon.examples.get(kind))
    assert sum(ctl.stats[k] for k in ("skipped", "ev_jumps", "ev_cuts")) > 0  # the agents did act


@protected
def test_three_lane_approaches(junctions, rep):
    """Only the ring road (E and W at 2014 and 2328) has three driving lanes; the N and S approaches at 53, 979 and
    1722 have two driving lanes and a parking lane (OpenDRIVE type parking: no movement at the signal, its
    connections are removed). Every lane gets a green each cycle. Nobody walks across the ring road (27 Oct): its
    crosswalks at 2014 and 2328 are gone, the side road's stay."""
    wide = []
    for tid, jn in junctions.items():
        lanes: dict[str, set[str]] = {}
        for lk in jn.links:
            if lk.kind == "vehicle":
                lanes.setdefault(lk.approach, set()).add(lk.from_lane)
        phases = jn.programs["0"]
        for app, ls in lanes.items():
            for ln in ls:
                assert any(s[lk.index] == "G" for s, *_ in phases for lk in jn.links if lk.from_lane == ln), (tid, ln)
            if len(ls) >= 3:
                wide.append((tid, app))
                crossing = [lk for lk in jn.links if lk.kind == "crossing"
                            and {ln.rsplit("_", 1)[0] for ln in ls} & set(lk.crosses)]
                assert not crossing, (tid, app, [c.crossing for c in crossing])
    assert sorted(wide) == [("2014", "E"), ("2014", "W"), ("2328", "E"), ("2328", "W")]
    for tid in ("2014", "2328"):
        assert sum(1 for lk in junctions[tid].links if lk.kind == "crossing") == 1
    for tid in ("53", "979", "1722"):
        for side_ in "NS":
            assert len(rep[tid]["approaches"][side_]["lanes"]) == 2, (tid, side_)  # the parking lane: no movement


def test_the_pedestrian_call_can_be_switched_off_for_the_ablation():
    """"agents-pedcall" is the agents as they were before 25 Oct: someone waiting to walk is no demand, so green 3
    (no cars) is skipped with its walk; "agents" keeps it."""
    from cosim.agents_actuated import make

    cars_on_0 = lambda tc, a, k: (1, 0) if k == 0 else (0, 0)
    ag, a, tc = _ped_agents({"p1": (":J_w1", ":J_c1")})
    ag._approach = cars_on_0
    assert "pedcall" in ag.on and ag._choose(tc, a, 3, now=100.0) == 3   # the walk with green 3 is served
    off = make("agents-pedcall")
    off._approach = cars_on_0
    off._cache = {}
    off._pedestrians(tc)
    assert "pedcall" not in off.on and off._waiting == {":J_c1": 1}
    assert not off._demand(tc, a, 3, now=100.0) and off._choose(tc, a, 3, now=100.0) == 0  # skipped with its walk

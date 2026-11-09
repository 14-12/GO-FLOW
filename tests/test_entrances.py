"""(9 Nov) The ways into the car parks inside buildings (tools/town05_entrances_build.py), on a made-up probe: a
road, and beside it a building whose walls stand on three sides and whose fourth side, away from the road, is open
under its roof. The way goes round to the opening and ends deep under the roof."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import town05_entrances_build as eb  # noqa: E402


def made_up_probe(open_side="north", door=False):
    step, x0, y0, nx, ny = 0.6, 0.0, 0.0, 100, 100            # 60 m x 60 m
    cols = []
    for j in range(ny):
        row = []
        for i in range(nx):
            x, y = x0 + i * step, y0 + j * step
            hits = []
            inside = 15.0 <= x <= 45.0 and 20.0 <= y <= 50.0
            wall = inside and (abs(x - 15.0) < 0.5 or abs(x - 45.0) < 0.5 or abs(y - 20.0) < 0.5
                               or (abs(y - 50.0) < 0.5 and open_side != "north"))
            if door and abs(y - 20.0) < 0.5 and 27.0 <= x <= 33.0:
                wall = False                       # (the door: the probe sees it shut, see doors below)
                hits.append([1.4, 3])
            if inside:
                hits.append([9.0, 3])              # the floor above
            if wall:
                hits.append([1.5, 4])
            hits.append([0.0, 1 if y < 12.0 else 2])
            row.append(hits)
        cols.append(row)
    lane = [[0.0, 6.0], [60.0, 6.0]]                     # the road, along the south, eastwards
    probe = {"areas": {"pa_test": {"place": "test", "x0": x0, "y0": y0, "nx": nx, "ny": ny, "step": step,
                                   "cols": cols, "lane": "t_0", "lane_shape": lane, "outline": []}}}
    if door:
        probe["doors"] = {"Door": {"id": 1, "sumo": [30.0, 20.0], "ext": [0.3, 3.0, 1.4], "yaw": 90.0}}
        probe["door_rays"] = {"Door_a": [[[25.0, 3]], [[25.0, 3]], [[25.0, 3]]], "Door_b": [[], [], []]}
    return probe


def test_the_way_goes_round_to_the_open_side_and_ends_deep_under_the_roof():
    res = eb.build(made_up_probe())[0]["pa_test"]
    assert res["found"], res
    end = res["way"][-1]
    assert 15.0 < end[0] < 45.0 and 20.0 < end[1] < 50.0              # inside
    assert min(end[1] - 20.0, 50.0 - end[1], end[0] - 15.0, 45.0 - end[0]) >= 1.0
    assert res["way"][0][1] > 45.0                                       # in by the open north side
    fx, fy = res["front"]
    assert not (15.0 <= fx <= 45.0 and 20.0 <= fy <= 50.0)               # from outside it
    assert abs(res["depart_pos"] - res["stop_end"] - 2 * eb.ROAD_ON_M) < 1e-6   # stops before its way, back after


def test_a_shut_door_with_room_behind_it_is_a_way_in():
    res = eb.build(made_up_probe(open_side="none", door=True))[0]["pa_test"]
    assert res["found"] and res["door"] == "Door", res
    assert abs(res["way"][0][1] - 20.0) < 1.5 and 27.0 <= res["way"][0][0] <= 33.0


def test_no_way_in_when_the_walls_are_shut_all_round():
    res = eb.build(made_up_probe(open_side="none"))[0]["pa_test"]
    assert not res["found"]

"""The live agent's rows (cosim/incident_agent_proc.rows_of, our venv) are his result_rows (his venv)."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import stream  # noqa: E402,F401  (puts his package on the path)
from agents.incident_detection.src.video_processing import result_rows  # noqa: E402

if not (REPO / "cosim" / "incident_agent_proc.py").is_file():
    pytest.skip("not inside the Urban Traffic CARLA project", allow_module_level=True)
from cosim.incident_agent_proc import rows_of  # noqa: E402


class V:
    def __init__(self, *values):
        self.v = np.array([values], dtype=float)

    def __getitem__(self, i):
        return self

    def item(self):
        return float(self.v[0, 0])

    def cpu(self):
        return self

    def numpy(self):
        return self.v[0]


class Box:
    def __init__(self, tid, cls, xyxy, conf):
        self.id = None if tid is None else V(tid)
        self.cls, self.conf, self.xyxy = V(cls), V(conf), V(*xyxy)


@pytest.mark.parametrize("seed", range(20))
def test_rows_of_equals_result_rows(seed):
    rng = random.Random(seed)
    cats = {0: "car", 1: "car", 2: "bus", 3: "truck", 5: "truck", 7: "car"}
    boxes = []
    for _ in range(rng.randint(0, 15)):
        x1, y1 = rng.uniform(0, 900), rng.uniform(0, 500)
        boxes.append(Box(rng.choice([None, rng.randint(1, 99)]), rng.choice(list(cats)),
                         (x1, y1, x1 + rng.uniform(5, 200), y1 + rng.uniform(5, 150)), rng.uniform(0.25, 1)))
    result = type("R", (), {"boxes": boxes})()
    t = rng.uniform(0, 1000)
    frame_his, dets_his = result_rows(result, t, 960 * 540, cats)
    frame, dets = rows_of(result, 960 * 540, cats)
    assert dict(frame, source_time_sec=t) == pytest.approx(frame_his)
    assert len(dets) == len(dets_his)
    for d, h in zip(dets, dets_his):
        assert dict(zip(stream.DET_KEYS, d), source_time_sec=t) == pytest.approx(h)

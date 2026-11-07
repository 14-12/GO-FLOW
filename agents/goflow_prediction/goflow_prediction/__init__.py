"""DSO4 congestion forecaster, Town05 version (integration plan, docs/INTEGRATION_PLAN_TOWN05.md in the DSO4 repo).

What changed from the METR-LA / PEMS08 version (the original stays in the DSO4 repository, unchanged):
* time is in SUMO seconds: a step of `dt_s` (30 s), 12 steps in, horizons 60 / 120 / 300 s (config.TimeBase);
* the target is the queue of each approach, `fill` = halting vehicles / storage (lane length / 7.5 m), as the
  junction agents read it (cosim/agents_actuated.py `_fill`), not the speed;
* the day is described by the town's scheduled demand (cosim/daily.demand_factor) now and at the target time,
  not by a 5-minute slot and a weekday;
* it talks to the town in JSON lines (serve.py, schema.py).
"""

__version__ = "0.1.0"

# Where the scenarios come from

## Town05 (generated)

`town05/` is written by `python -m cosim.town` (`tools/build_town05.bat`) from the OpenDRIVE map of CARLA's
Town05 (`town05.xodr`, from the CARLA 0.9.16 package), then given its crosswalks
(`tools/build_town05_peds.py`), bus lines (`tools/build_town05_pt.py`) and signal programs
(`tools/build_town05_signals.py`). `scenario.json` in it records the CARLA map, the input hashes, the vehicle
counts, the seed and the tool versions. The signal sets (`town05.<set>.*`) are chosen with `tools/use_signals.py`.

## Shared files

`twin_extras.add.xml` is the main project's emergency-vehicle and pedestrian type file
(ADR 0009 there; `sim/scenarios/twin_extras.add.xml`, sha256 15dd6f0ad85f0c0e, byte-identical).
It is loaded by `cosim/extras.py` (a port of the main project's `utm/twin/extras.py`).

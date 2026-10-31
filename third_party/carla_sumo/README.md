# CARLA's SUMO network converter (vendored)

`netconvert_carla.py` and `data/opendrive_netconvert.typ.xml` are copied unchanged from
CARLA 0.9.16, `Co-Simulation/Sumo/util/`
(https://github.com/carla-simulator/carla/tree/0.9.16/Co-Simulation/Sumo/util).

Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona,
MIT licence (header of the script; https://opensource.org/licenses/MIT).

Used by `cosim/town.py` to build the SUMO network of a CARLA town. It needs `carla`,
`sumolib`, `lxml` and SUMO's `netconvert` on PATH (`cosim/town.py` sets PATH and SUMO_HOME,
and PYTHONHASHSEED=0 so the output is the same on every run).

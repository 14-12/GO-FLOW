"""Pure conversions from SUMO state to CARLA values (no server needed; tests/test_convert.py)."""

from __future__ import annotations

from collections import Counter

import carla

from cosim.mapping import Frame, front_to_centre, sumo_angle_to_carla_yaw

# SUMO VAR_SIGNALS bits (sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html)
BLINKER_RIGHT, BLINKER_LEFT, BLINKER_EMERGENCY, BRAKELIGHT, FRONTLIGHT, FOGLIGHT, HIGHBEAM, \
    BACKDRIVE = (1 << i for i in range(8))

_TL = {
    "G": carla.TrafficLightState.Green, "g": carla.TrafficLightState.Green,
    "y": carla.TrafficLightState.Yellow, "Y": carla.TrafficLightState.Yellow,
    "r": carla.TrafficLightState.Red, "R": carla.TrafficLightState.Red,
    "s": carla.TrafficLightState.Red,   # stop sign behaviour: shown red
    "u": carla.TrafficLightState.Red,   # red-yellow (prepare to go): still red for traffic
    "o": carla.TrafficLightState.Off, "O": carla.TrafficLightState.Off,
}
# When one CARLA traffic light shows several SUMO links that disagree, show the most
# restrictive state: never draw a green that the simulation does not give to every movement.
_RESTRICTIVE = [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow,
                carla.TrafficLightState.Green, carla.TrafficLightState.Off]


def tl_state(chars: list[str]) -> tuple[carla.TrafficLightState, bool]:
    """CARLA state for one traffic-light actor showing these SUMO link states, and whether
    the links disagreed."""
    states = [_TL.get(ch, carla.TrafficLightState.Unknown) for ch in chars]
    counts = Counter(states)
    if len(counts) == 1:
        return states[0], False
    for s in _RESTRICTIVE:
        if s in counts:
            return s, True
    return carla.TrafficLightState.Unknown, True


def light_state(signals: int, night: bool, emergency_run: bool = False) -> carla.VehicleLightState:
    """CARLA vehicle lights from SUMO's signal bits (+ headlamps at night, + beacons)."""
    v = carla.VehicleLightState
    out = int(v.NONE)
    if signals & (BLINKER_RIGHT | BLINKER_EMERGENCY):
        out |= int(v.RightBlinker)
    if signals & (BLINKER_LEFT | BLINKER_EMERGENCY):
        out |= int(v.LeftBlinker)
    if signals & BRAKELIGHT:
        out |= int(v.Brake)
    if signals & FOGLIGHT:
        out |= int(v.Fog)
    if signals & HIGHBEAM:
        out |= int(v.HighBeam)
    if signals & BACKDRIVE:
        out |= int(v.Reverse)
    if night or signals & FRONTLIGHT:
        out |= int(v.Position) | int(v.LowBeam)
    if emergency_run:
        out |= int(v.Special1)  # siren lights on CARLA's emergency models
    return v(out)


def vehicle_transform(frame: Frame, x: float, y: float, angle: float, length: float,
                      z: float = 0.0, pitch: float = 0.0) -> carla.Transform:
    """CARLA transform for a SUMO vehicle (front bumper at x, y; heading `angle`)."""
    cx, cy = front_to_centre(x, y, angle, length / 2)
    ux, uy = frame.to_carla_xy(cx, cy)
    return carla.Transform(carla.Location(ux, uy, z), carla.Rotation(pitch, sumo_angle_to_carla_yaw(angle), 0.0))

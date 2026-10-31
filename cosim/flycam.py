"""Arrow-key turning for the free-fly camera, done in CARLA's tick callback.

CARLA's spectator moves with its own keys (Z Q S D, A / E: CarlaUE4/Config/DefaultInput.ini) and turns only
with the mouse, so the arrows turn it from Python, with set_transform, which also sends a place. Sent from
the viewer's main loop with get_transform's place, that place was a picture old: the camera was pulled
back every time and shook while Z Q S D moved it. Sent from CARLA's tick callback with the place of the
picture just made, nothing is lost (tools/probe_keys.py, Z held and 90 deg/s of turn for 2.5 s: 20.0 m/s
every frame, frame-to-frame change 0, the same as Z alone; turn 225 deg as asked).
"""

from __future__ import annotations

TURN_DEG_S, PITCH_DEG_S = 90.0, 60.0


class ArrowTurn:
    def __init__(self, spectator):
        self.sp = spectator
        self.view = None  # (yaw, pitch) while an arrow is held

    def tick(self, snap, arrows: set[str]) -> None:
        import carla

        if not arrows:
            self.view = None  # the mouse is free to turn it
            return
        a = snap.find(self.sp.id)
        if a is None:
            return
        tf = a.get_transform()
        if self.view is None:
            self.view = (tf.rotation.yaw, tf.rotation.pitch)
        dt = min(snap.timestamp.delta_seconds, 0.1)
        yaw = self.view[0] + (("RIGHT" in arrows) - ("LEFT" in arrows)) * TURN_DEG_S * dt
        pitch = max(-89.0, min(89.0, self.view[1] + (("UP" in arrows) - ("DOWN" in arrows)) * PITCH_DEG_S * dt))
        self.view = (yaw, pitch)
        self.sp.set_transform(carla.Transform(tf.location, carla.Rotation(pitch, yaw, 0.0)))

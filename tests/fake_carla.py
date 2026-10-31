"""A stand-in CARLA server for the runner's smoke test. Test-only.

Only the server is faked. The map is a real `carla.Map` built in process from the same
OpenDRIVE text (the client library can do that without a server), so landmarks, waypoints
and heights are CARLA's own. What the fake cannot tell: how the server renders, whether it
creates a traffic-light actor for every signal, and how fast it is.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import carla


class FakeBlueprint:
    def __init__(self, bp_id, attrs):
        self.id = bp_id
        self._a = dict(attrs)

    def has_attribute(self, name):
        return name in self._a

    def get_attribute(self, name):
        return FakeAttribute(name, self._a[name])

    def set_attribute(self, name, value):
        self._a[name] = value


class FakeAttribute:
    """Typed like carla.ActorAttribute: as_str() on a non-string raises, as the server does
    (0.9.16: "generation: bad attribute cast: cannot convert to String")."""

    def __init__(self, name, value):
        self.id, self._v = name, value
        if isinstance(value, bool):
            self.type = carla.ActorAttributeType.Bool
        elif isinstance(value, int):
            self.type = carla.ActorAttributeType.Int
        elif isinstance(value, float):
            self.type = carla.ActorAttributeType.Float
        elif isinstance(value, tuple):
            self.type = carla.ActorAttributeType.RGBColor
        else:
            self.type = carla.ActorAttributeType.String
        self.recommended_values = list(value) if isinstance(value, tuple) else []

    def as_str(self):
        if self.type != carla.ActorAttributeType.String:
            raise RuntimeError(f"{self.id}: bad attribute cast: cannot convert to String")
        return self._v

    def as_int(self):
        return int(self._v)

    def as_bool(self):
        return bool(self._v)

    def as_float(self):
        return float(self._v)


class FakeLibrary:
    def __init__(self):
        self.bps = [
            FakeBlueprint("vehicle.test.car", {"base_type": "car", "generation": 2, "color": ("255,0,0", "0,0,255")}),
            FakeBlueprint("vehicle.test.van", {"base_type": "van", "generation": 2}),
            FakeBlueprint("vehicle.test.truck", {"base_type": "truck", "generation": 2}),
            FakeBlueprint("vehicle.test.bus", {"base_type": "bus", "generation": 2}),
            FakeBlueprint("vehicle.test.moto", {"base_type": "motorcycle", "generation": 2}),
            FakeBlueprint("vehicle.test.ambulance", {"base_type": "van", "special_type": "emergency", "generation": 2}),
            FakeBlueprint("vehicle.test.police", {"base_type": "car", "special_type": "emergency", "generation": 2}),
            FakeBlueprint("walker.pedestrian.0001", {"is_invincible": "true"}),
            FakeBlueprint("sensor.camera.rgb", {"image_size_x": "", "image_size_y": "", "fov": ""}),
            FakeBlueprint("sensor.camera.depth", {"image_size_x": "", "image_size_y": "", "fov": ""}),
            FakeBlueprint("sensor.camera.semantic_segmentation", {"image_size_x": "", "image_size_y": "", "fov": ""}),
        ]

    def filter(self, pattern):
        prefix = pattern.rstrip("*")
        return [b for b in self.bps if b.id.startswith(prefix)]

    def find(self, bp_id):
        return next(b for b in self.bps if b.id == bp_id)


class FakeTrafficLight:
    def __init__(self, tl_id):
        self.id = tl_id
        self.state = carla.TrafficLightState.Unknown
        self.changes = 0

    def set_state(self, s):
        self.state = s
        self.changes += 1

    def get_state(self):
        return self.state


class FakeSensor:
    def __init__(self, world):
        self.world = world
        self.cb = None

    def listen(self, cb):
        self.cb = cb

    def set_transform(self, tf):
        pass

    def stop(self):
        self.cb = None

    def destroy(self):
        self.world.sensors.remove(self)


class FakeSpectator:
    """Keeps its transform, as CARLA's does: sensors are spawned where it is (camera.py)."""

    def __init__(self):
        self.tf = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation())

    def get_transform(self):
        return self.tf

    def set_transform(self, tf):
        self.tf = tf


class FakeWorld:
    def __init__(self, xodr_text, name="Carla/Maps/Fake"):
        self.map = carla.Map(name, xodr_text)
        self.settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None, no_rendering_mode=False)
        self.library = FakeLibrary()
        self.frame = 0
        self.sensors = []
        self.actors = {}  # id -> transform
        self.tls = {}
        self.weather = None
        self.ticks = 0
        self.spectator = FakeSpectator()

    def get_map(self):
        return self.map

    def get_settings(self):
        return self.settings

    def apply_settings(self, s):
        self.settings = s

    def get_blueprint_library(self):
        return self.library

    def get_traffic_light(self, landmark):
        return self.tls.setdefault(landmark.id, FakeTrafficLight(1000 + len(self.tls)))

    def freeze_all_traffic_lights(self, frozen):
        pass

    def set_weather(self, w):
        self.weather = w

    def cast_ray(self, a, b):
        return []  # an empty map: nothing in the way

    def get_spectator(self):
        return self.spectator

    def spawn_actor(self, bp, tf):
        s = FakeSensor(self)
        self.sensors.append(s)
        return s

    def tick(self):
        assert self.settings.synchronous_mode, "the runner must tick in synchronous mode"
        self.frame += 1
        self.ticks += 1
        for s in self.sensors:
            if s.cb:
                s.cb(SimpleNamespace(frame=self.frame, width=4, height=2, raw_data=bytes(32),
                                     save_to_disk=_save_to_disk))
        return self.frame


def _save_to_disk(path):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"fake")


class FakeSpawn:
    def __init__(self, bp, tf):
        self.bp, self.tf = bp, tf

    def then(self, cmd):
        return self


class FakeClient:
    """Executes the command batches the runner sends against FakeWorld."""

    def __init__(self, world):
        self.world = world
        self._ids = itertools.count(1)
        self.moves = 0

    def set_timeout(self, t):
        pass

    def get_server_version(self):
        return "fake"

    def get_client_version(self):
        return "0.9.16"

    def get_world(self):
        return self.world

    def load_world(self, name):
        return self.world

    def generate_opendrive_world(self, xodr, params):
        return self.world

    def apply_batch_sync(self, cmds, due_tick_cue=False):
        out = []
        for c in cmds:
            assert isinstance(c, FakeSpawn)
            aid = next(self._ids)
            self.world.actors[aid] = c.tf
            out.append(SimpleNamespace(error="", actor_id=aid))
        return out

    def apply_batch(self, cmds):
        for c in cmds:
            name = type(c).__name__
            if name == "ApplyTransform":
                assert c.actor_id in self.world.actors, "moved an actor that does not exist"
                self.world.actors[c.actor_id] = c.transform
                self.moves += 1
            elif name == "ApplyWalkerState":
                assert c.actor_id in self.world.actors, "moved a walker that does not exist"
                self.world.actors[c.actor_id] = c.transform
                self.walker_moves = getattr(self, "walker_moves", 0) + 1
            elif name == "DestroyActor":
                del self.world.actors[c.actor_id]

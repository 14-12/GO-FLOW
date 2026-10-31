"""Which traffic-light landmarks of the loaded map have no traffic-light actor (diagnostic)."""

import collections
import json
import sys

import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(60.0)
world = client.get_world()
m = world.get_map()
missing, ok = [], []
for lm in m.get_all_landmarks_of_type("1000001"):
    (ok if world.get_traffic_light(lm) is not None else missing).append(lm)
actors = world.get_actors().filter("traffic.traffic_light*")
by_tls = collections.Counter(lm.id.rsplit("_", 1)[0] for lm in missing)
out = {"map": m.name, "landmarks": len(ok) + len(missing), "with_actor": len(ok), "traffic_light_actors": len(actors),
       "missing_by_tls": dict(by_tls), "missing": [[lm.id, lm.road_id, round(lm.s, 2)] for lm in missing],
       "actors_without_sign": []}
for a in actors:
    try:
        ids = [lm.id for lm in a.get_affected_lane_waypoints()][:0]
    except Exception:  # noqa: BLE001
        ids = []
    out["actors_without_sign"].append([a.id, round(a.get_location().x, 1), round(a.get_location().y, 1),
                                       a.get_opendrive_id() if hasattr(a, "get_opendrive_id") else None])
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "actors_without_sign"}, indent=1))

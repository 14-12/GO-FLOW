"""The depots (cosim/depots.py) in CARLA before they go in the viewer (25 Oct): at each one its sign in four turnings
of the deck mesh (which way its face and text come out), and its vehicles parked in their slots, photographed from
the road and from above (out/_probe/depots). CARLA must be running (tools/probe_depots.bat starts it)."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import carla  # noqa: E402
from probe_plates2 import Cams, look, rgb_of  # noqa: E402

from cosim import depots, plates, towing  # noqa: E402
from cosim import scenario as scenario_mod  # noqa: E402
from cosim.vmodels import FIXED  # noqa: E402

OUT = ROOT / "out" / "_probe" / "depots"
MODEL = {"hospital": FIXED["emergency"], "police": FIXED["authority"], "tow": towing.TRUCK_MODEL}


def main() -> int:
    import sumolib

    OUT.mkdir(parents=True, exist_ok=True)
    scen = scenario_mod.load("town05", ROOT)
    net = sumolib.net.readNet(str(scen.net_file))
    frame = scen.frame
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(300.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05"):
        world = client.load_world("Town05")
    cmap = world.get_map()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    lib = world.get_blueprint_library()
    cams = Cams(world, lib)
    rep: dict = {"errors": [], "depots": {}}
    actors = []

    def ground(x, y):
        wp = cmap.get_waypoint(carla.Location(x, y, 0.0), project_to_road=True, lane_type=carla.LaneType.Any)
        return wp.transform.location.z if wp is not None else 0.0

    def carla_pose(x, y, hdg):
        cx, cy = frame.to_carla_xy(x, y)
        x2, y2 = frame.to_carla_xy(x + math.cos(math.radians(hdg)), y + math.sin(math.radians(hdg)))
        return cx, cy, math.degrees(math.atan2(y2 - cy, x2 - cx))

    try:
        for plan in depots.TOWN05:
            info: dict = {}
            e = net.getEdge(plan.edge)
            park = e.getLanes()[plan.lane]
            sx, sy, facing = depots.sign_spot(net, plan)
            cx, cy, cyaw = carla_pose(sx, sy, facing)
            z = ground(cx, cy)
            info["sign"] = (round(cx, 1), round(cy, 1), round(z, 2), round(cyaw, 1))
            tex = plates.to_texture(depots.sign_texture(plan))
            # four turnings, 4.5 m apart along the pavement: (yaw added to the facing, roll)
            along = math.radians(cyaw + 90.0)
            variants = [("A", 180.0, 90.0), ("B", 180.0, -90.0), ("C", 0.0, 90.0), ("D", 0.0, -90.0)]
            h = 2.36 * depots.SIGN_SCALE
            for k, (tag, dyaw, roll) in enumerate(variants):
                off = (k - 1.5) * 4.5
                loc = carla.Location(cx + off * math.cos(along), cy + off * math.sin(along), z + depots.SIGN_UP_M + h / 2)
                bp = lib.find("static.prop.mesh")
                bp.set_attribute("mesh_path", towing.DECK_MESH)
                bp.set_attribute("scale", f"{depots.SIGN_SCALE:.3f}")
                before = set(world.get_names_of_all_objects())
                a = world.spawn_actor(bp, carla.Transform(loc, carla.Rotation(0.0, cyaw + dyaw, roll)))
                actors.append(a)
                world.tick()
                new = sorted(set(world.get_names_of_all_objects()) - before)
                world.apply_color_texture_to_objects(new, carla.MaterialParameter.Diffuse, tex)
                info[f"variant_{tag}"] = {"at_m": off, "yaw": round(cyaw + dyaw, 1), "roll": roll, "names": new}
            # the vehicles in their slots
            bp = lib.find(MODEL[plan.name])
            vlen = plan.vehicle_len
            for pos in depots.slots(park.getLength(), vlen, plan.fleet, gap=plan.gap, keep_clear=plan.keep_clear):
                px, py, ph = depots.lane_point(park.getShape(), park.getLength(), pos - vlen / 2.0)
                qx, qy, qyaw = carla_pose(px, py, ph)
                v = world.try_spawn_actor(bp, carla.Transform(carla.Location(qx, qy, ground(qx, qy) + 0.3),
                                                              carla.Rotation(0.0, qyaw, 0.0)))
                if v is not None:
                    v.set_simulate_physics(False)
                    actors.append(v)
                info.setdefault("parked", []).append((pos, v is not None))
            world.tick()
            # views: from the road (the far lane, 18 m before the sign), across the road, from above
            rx, ry, rh = depots.lane_point(e.getLanes()[-1].getShape(), e.getLanes()[-1].getLength(),
                                           park.getLength() * plan.sign_at - 18.0, -4.0)
            ex, ey, _ = carla_pose(rx, ry, rh)
            target = carla.Location(cx, cy, z + 1.2)
            ax, ay, _ = carla_pose(*depots.lane_point(e.getLanes()[-1].getShape(), e.getLanes()[-1].getLength(),
                                                      park.getLength() * plan.sign_at, -9.0)[:2], 0.0)
            views = {"road": carla.Location(ex, ey, ground(ex, ey) + 1.7),
                     "across": carla.Location(ax, ay, ground(ax, ay) + 2.0),
                     "above": carla.Location(ax + (ax - cx) * 1.2, ay + (ay - cy) * 1.2, z + 22.0)}
            for tag, eye in views.items():
                rgb, _ = cams.shot(look(eye, target))
                if rgb is not None:
                    Image.fromarray(rgb_of(rgb)).save(OUT / f"{plan.name}_{tag}.png")
            rep["depots"][plan.name] = info
            print(plan.name, info, flush=True)
    except Exception:  # noqa: BLE001
        rep["errors"].append(traceback.format_exc())
        traceback.print_exc()
    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:  # noqa: BLE001
                pass
        cams.destroy()
        s.synchronous_mode, s.fixed_delta_seconds = False, None
        world.apply_settings(s)
        (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

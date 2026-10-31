"""Tunisian number plates on the free-fly viewer's cars, made in a process of its own (23 Oct).

Each car of the viewer's pool (cosim/freefly.pooled_carla_side) gets two plate actors of its own, rear and front,
made from SM_TnPlate (tools/plates/make_plate_mesh.py), with a texture drawn for it (cosim/plates.py: its own
number, never repeated) and given through CARLA's runtime texture API (World.apply_color_texture_to_objects, the
Diffuse parameter). Why separate actors: a car's own plate is a material slot of its skeletal mesh, which that API
does not reach (tools/probe_plates_api.py: no change on any of 18 models), and CARLA can't attach a prop to a car
(tools/probe_incident_assets.py: the prop stays where it was spawned), so the pose process moves the plates with
their car, picture by picture, when the car is near the camera (cosim/pose_proc.Driver, PLATE_M).

Where each model's plates go (centre, tilt, width): measured from CARLA itself, every model with its plate made
magenta and photographed with a depth camera (tools/probe_plates2.py; config/vehicle_models.json "plates": x, y,
z, pitch, yaw, scale). The width is the real one where the car's plate recess allows it (rear 52 cm, front 45 cm),
smaller on cars built for a small plate; motorcycles have a rear plate only.

Working the textures out (Python sets CARLA's texture pixel by pixel) and the server calls are done here, in
groups of CHUNK cars, while the viewer loads (its stage "Fitting the number plates"); cars added later get theirs
here too. Messages in: ("cars", [(actor id, model)]), ("decks", n); out: ("plates", {car actor: [(plate
actor, rel)]}, {car actor: "247 TU 3816"}), ("decks", [deck actor]) (the recovery trucks' flatbeds, 24 Oct);
("signs", [(name, image, pose, scale, mesh)]) -> ("signs", [(name, actor, objects textured)]) (the depots' boards, 25 Oct).
"""

from __future__ import annotations

import queue
import random
import time

MESH = "/Game/Carla/Static/Other/SM_TnPlate.SM_TnPlate"
CHUNK = 24  # cars whose plates are made together (one batch, one look at the scene's names)
PARK = (0.0, 0.0, -1000.0)


def run(host: str, port: int, q_in, q_out, plates_cfg: dict, seed: int = 21) -> None:
    try:
        _run(host, port, q_in, q_out, plates_cfg, seed)
    except Exception:  # noqa: BLE001 - reported to the viewer's log
        import traceback

        q_out.put(("fatal", traceback.format_exc()))


def _run(host: str, port: int, q_in, q_out, plates_cfg: dict, seed: int) -> None:
    import multiprocessing as mp

    import carla

    from cosim import plates as pl

    client = carla.Client(host, port)
    client.set_timeout(60.0)
    world = client.get_world()
    lib = world.get_blueprint_library()
    try:
        lib.find("static.prop.mesh")
    except Exception:  # noqa: BLE001
        q_out.put(("fatal", "no static.prop.mesh in this CARLA"))
        return
    rng = random.Random(seed)
    used: set = set()
    parent = mp.parent_process()
    park = carla.Transform(carla.Location(*PARK))
    made = 0
    t_all = 0.0
    all_ids: list[int] = []
    retry: list = []
    while parent is None or parent.is_alive():
        try:
            msg = q_in.get(timeout=0.5)
        except queue.Empty:
            continue
        if msg[0] == "stop":
            if all_ids:  # the plates go with the viewer
                try:
                    client.apply_batch_sync([carla.command.DestroyActor(a) for a in all_ids], False)
                except Exception:  # noqa: BLE001
                    pass
            break
        if msg[0] == "decks":  # (n,): the recovery trucks' decks (cosim/towing.py), all the same texture
            try:
                from cosim import towing

                bp = lib.find("static.prop.mesh")
                bp.set_attribute("mesh_path", towing.DECK_MESH)
                bp.set_attribute("mass", "1.0")
                before = set(world.get_names_of_all_objects())
                res = client.apply_batch_sync([carla.command.SpawnActor(bp, park).then(
                    carla.command.SetSimulatePhysics(carla.command.FutureActor, False)) for _ in range(msg[1])], False)
                ids = [r.actor_id for r in res if not r.error]
                new = [n for n in set(world.get_names_of_all_objects()) - before if n.startswith("StaticMeshActor_")]
                if new:
                    world.apply_color_texture_to_objects(new, carla.MaterialParameter.Diffuse,
                                                         pl.to_texture(towing.deck_texture()))
                all_ids += ids
                q_out.put(("decks", ids))
            except Exception as e:  # noqa: BLE001
                q_out.put(("fatal", f"decks: {e!r}"))
            continue
        if msg[0] == "signs":  # ([(name, Pillow image, (x, y, z, pitch, yaw, roll), scale, mesh)],): the depots'
            out = []         # boards (cosim/depots.py), each its own texture: spawned and named one at a time
            for name, image, pose, scale, mesh in msg[1]:
                try:
                    bp = lib.find("static.prop.mesh")
                    bp.set_attribute("mesh_path", mesh)
                    bp.set_attribute("scale", f"{scale:.3f}")
                    tf = carla.Transform(carla.Location(*pose[:3]), carla.Rotation(*pose[3:]))
                    before = set(world.get_names_of_all_objects())
                    res = client.apply_batch_sync([carla.command.SpawnActor(bp, tf)], False)
                    if res[0].error:
                        q_out.put(("fatal", f"sign {name}: {res[0].error}"))
                        continue
                    new = [n for n in set(world.get_names_of_all_objects()) - before
                           if n.startswith("StaticMeshActor_")]
                    if new:
                        world.apply_color_texture_to_objects(new, carla.MaterialParameter.Diffuse, pl.to_texture(image))
                    all_ids.append(res[0].actor_id)
                    out.append((name, res[0].actor_id, len(new)))
                except Exception as e:  # noqa: BLE001
                    q_out.put(("fatal", f"sign {name}: {e!r}"))
            q_out.put(("signs", out))
            continue
        if msg[0] != "cars":
            continue
        cars = [(aid, model) for aid, model in msg[1] if plates_cfg.get(model)]
        for k in range(0, len(cars), CHUNK):
            chunk = cars[k:k + CHUNK]
            t0 = time.perf_counter()
            # the textures first (here, no server call), then every plate of the chunk in one batch and one look at
            # the scene's names before and after: each name of the scene list costs the server's game thread, and
            # asked twice a car (the first version) it made the traffic judder for the minute the plates took
            plates, batch, owner = [], [], []
            for aid, model in chunk:
                plate = pl.plate_for(model, rng, used)  # police cars and ambulances: the state plate
                plates.append((aid, plate, pl.to_texture(pl.render(plate))))
                for end, rel in plates_cfg[model].items():
                    bp = lib.find("static.prop.mesh")
                    bp.set_attribute("mesh_path", MESH)
                    bp.set_attribute("mass", "1.0")  # simulated at spawn, so CARLA makes it movable; physics off at once
                    bp.set_attribute("scale", f"{rel[5]:.4f}")
                    batch.append(carla.command.SpawnActor(bp, park).then(
                        carla.command.SetSimulatePhysics(carla.command.FutureActor, False)))
                    owner.append((aid, tuple(rel[:5])))
            t_tex = time.perf_counter() - t0
            before = set(world.get_names_of_all_objects())
            res = client.apply_batch_sync(batch, False)
            after = world.get_names_of_all_objects()
            ids = [r.actor_id for r in res]
            ok = [not r.error for r in res]
            new = [n for n in set(after) - before if n.startswith("StaticMeshActor_")]
            # CARLA numbers the names of actors spawned in one batch downwards, in spawn order (measured on the
            # vehicles, tools/probe_plates_api.py: BP_Lincoln2020_C_2147460896, ..._174, ..._081 for three in a row)
            new.sort(key=lambda n: -int(n.rsplit("_", 1)[1]) if n.rsplit("_", 1)[1].isdigit() else 0)
            if not all(ok) or len(new) != len(batch):
                # another prop appeared meanwhile (a scene's cones) or one failed: which name is which is unknown
                client.apply_batch([carla.command.DestroyActor(a) for a, good in zip(ids, ok) if good])
                for _, plate, _ in plates:
                    used.discard((("state", plate.series), plate.number) if plate.kind == "state"
                                 else (plate.series, plate.number))
                retry += chunk
                continue
            names_of: dict[int, list[str]] = {}
            props: dict[int, list] = {}
            for (aid, rel), pid, name in zip(owner, ids, new, strict=True):
                names_of.setdefault(aid, []).append(name)
                props.setdefault(aid, []).append((pid, rel))
            for aid, plate, tex in plates:
                world.apply_color_texture_to_objects(names_of[aid], carla.MaterialParameter.Diffuse, tex)
            all_ids += ids
            q_out.put(("plates", props, {aid: plate.text for aid, plate, _ in plates}))
            made += len(chunk)
            t_all += time.perf_counter() - t0
            q_out.put(("timing", len(chunk), round(1000 * t_tex), round(1000 * (time.perf_counter() - t0 - t_tex))))
        if retry and (msg[2] if len(msg) > 2 else 0) < 3:
            time.sleep(0.5)
            q_in.put(("cars", retry, (msg[2] if len(msg) > 2 else 0) + 1))
            retry = []
        q_out.put(("made", made, round(1000 * t_all / max(made, 1))))

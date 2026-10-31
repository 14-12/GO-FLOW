"""Detector labels from CARLA's instance-segmentation camera (pure functions, tested offline).

CARLA 0.9.16's `sensor.camera.instance_segmentation` writes, per pixel, the semantic tag in
the red channel and an instance id in green + 256 * blue. Measured on the server
(tools/probe_instance.py): for spawned vehicles that id *is* the actor id, so every visible
pixel can be traced to its SUMO vehicle and labelled with SUMO's vehicle class, not guessed
from appearance. Town10's parked cars are static meshes with ids of their own and no actor.

Classes are the main project's detector classes (twin-yolo26n.json), in its order, so the
data drops into its training script.

Rules (the main project's: at least 24 visible pixels, box at least 3 px; plus):
* an actor vehicle is labelled from its SUMO class;
* a static parked vehicle is labelled from its tag when the tag is unambiguous (Car -> car,
  Bus -> bus); CARLA's Truck tag covers vans and lorries alike, so a static one cannot be
  labelled and the frame is not used;
* a visible vehicle of a kind the classes do not have (motorbike, bicycle) also makes the
  frame unusable, because an unlabelled vehicle in a training image teaches "not a vehicle".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CLASSES = ["car", "van", "bus", "truck", "ambulance", "ambulance_lights", "police", "police_lights", "pedestrian"]
MIN_PIXELS = 24
MIN_BOX_PX = 3
# Below this mean brightness (0-255) a frame is treated as unlit and not labelled: the maps
# generated from SUMO networks have no street lights, so at night their RGB frames are black
# while the instance camera still sees every vehicle (measured: Ingolstadt7 at 22:30 has a
# mean of 0.9; Town10, which has street lights, 20-32 at the same hour).
UNLIT_LUMA = 8.0

# CARLA 0.9.16 semantic tags (CityObjectLabel values)
TAG_PEDESTRIAN, TAG_RIDER, TAG_CAR, TAG_TRUCK = 12, 13, 14, 15
TAG_BUS, TAG_TRAIN, TAG_MOTORCYCLE, TAG_BICYCLE = 16, 17, 18, 19
VEHICLE_TAGS = {TAG_CAR, TAG_TRUCK, TAG_BUS, TAG_MOTORCYCLE, TAG_BICYCLE}
ROAD_USER_TAGS = VEHICLE_TAGS | {TAG_PEDESTRIAN, TAG_RIDER}

# SUMO vClass -> detector class (None: no detector class for it)
VCLASS = {"passenger": "car", "taxi": "car", "private": "car", "evehicle": "car", "delivery": "van",
          "bus": "bus", "coach": "bus", "truck": "truck", "trailer": "truck",
          "emergency": "ambulance", "authority": "police", "motorcycle": None, "moped": None, "bicycle": None,
          "pedestrian": "pedestrian"}
STATIC_TAG = {TAG_CAR: "car", TAG_BUS: "bus", TAG_PEDESTRIAN: "pedestrian"}


@dataclass(frozen=True)
class Instance:
    tag: int
    id: int
    pixels: int
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 inclusive


def mean_luma(bgra: np.ndarray) -> float:
    """Mean Rec. 601 luma of a BGRA (or BGR) image, on every 4th pixel."""
    x = bgra[::4, ::4, :3].astype(np.float32)
    return float((0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]).mean())


def instances(bgra: np.ndarray) -> list[Instance]:
    """Road-user instances in an instance-segmentation image (H x W x 4, BGRA as CARLA sends)."""
    r = bgra[:, :, 2].astype(np.int64)
    ids = bgra[:, :, 1].astype(np.int64) + 256 * bgra[:, :, 0].astype(np.int64)
    key = r * 65536 + ids
    mask = np.isin(r, list(ROAD_USER_TAGS))
    out = []
    if not mask.any():
        return out
    keys, inverse, counts = np.unique(key[mask], return_inverse=True, return_counts=True)
    ys, xs = np.nonzero(mask)
    x1 = np.full(len(keys), 1 << 30)
    y1 = np.full(len(keys), 1 << 30)
    x2 = np.full(len(keys), -1)
    y2 = np.full(len(keys), -1)
    np.minimum.at(x1, inverse, xs)
    np.minimum.at(y1, inverse, ys)
    np.maximum.at(x2, inverse, xs)
    np.maximum.at(y2, inverse, ys)
    for k, n, a, b, c, d in zip(keys, counts, x1, y1, x2, y2, strict=True):
        out.append(Instance(int(k // 65536), int(k % 65536), int(n), (int(a), int(b), int(c), int(d))))
    return out


def label_frame(insts: list[Instance], actor_class: dict[int, str | None], width: int, height: int
                ) -> tuple[list[str] | None, dict]:
    """YOLO lines for one frame, or None when the frame must not be used; plus counts.

    actor_class: CARLA actor id -> detector class (None: a vehicle with no detector class).
    """
    lines, counts, reasons = [], {}, []
    for it in insts:
        x1, y1, x2, y2 = it.box
        big = it.pixels >= MIN_PIXELS and (x2 - x1 + 1) >= MIN_BOX_PX and (y2 - y1 + 1) >= MIN_BOX_PX
        if it.id in actor_class:
            cls = actor_class[it.id]
        elif it.tag in STATIC_TAG:
            cls = STATIC_TAG[it.tag]
        else:
            cls = None
        if cls is None:
            if big:
                reasons.append(f"unlabellable tag {it.tag} id {it.id} ({it.pixels} px)")
            continue
        if not big:
            continue
        c = CLASSES.index(cls)
        cx, cy = (x1 + x2 + 1) / 2 / width, (y1 + y2 + 1) / 2 / height
        w, h = (x2 - x1 + 1) / width, (y2 - y1 + 1) / height
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[cls] = counts.get(cls, 0) + 1
    if reasons:
        return None, {"rejected": reasons}
    return lines, counts

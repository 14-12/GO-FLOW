"""Labels from instance-segmentation images (cosim/labels.py)."""

import numpy as np

from cosim.labels import CLASSES, TAG_BUS, TAG_CAR, TAG_MOTORCYCLE, TAG_TRUCK, instances, label_frame


def img(h=20, w=40):
    return np.zeros((h, w, 4), dtype=np.uint8)


def paint(a, tag, iid, y1, y2, x1, x2):
    a[y1:y2, x1:x2, 2] = tag
    a[y1:y2, x1:x2, 1] = iid % 256
    a[y1:y2, x1:x2, 0] = iid // 256


def test_instances_split_by_tag_and_id():
    a = img()
    paint(a, TAG_CAR, 2680, 2, 8, 3, 10)      # 6 x 7
    paint(a, TAG_CAR, 2681, 10, 14, 20, 30)   # 4 x 10
    paint(a, 11, 5, 0, 20, 35, 40)            # a building: ignored
    got = {(i.tag, i.id): i for i in instances(a)}
    assert set(got) == {(TAG_CAR, 2680), (TAG_CAR, 2681)}
    assert got[(TAG_CAR, 2680)].box == (3, 2, 9, 7) and got[(TAG_CAR, 2680)].pixels == 42


def test_actor_class_wins_over_tag_and_yolo_geometry():
    a = img()
    paint(a, TAG_TRUCK, 300, 0, 10, 0, 20)  # CARLA says "Truck"; SUMO says delivery van
    lines, counts = label_frame(instances(a), {300: "van"}, 40, 20)
    assert counts == {"van": 1}
    c, cx, cy, w, h = lines[0].split()
    assert CLASSES[int(c)] == "van" and (float(cx), float(cy), float(w), float(h)) == (0.25, 0.25, 0.5, 0.5)


def test_static_parked_car_is_labelled_but_static_truck_rejects_the_frame():
    a = img()
    paint(a, TAG_CAR, 9000, 0, 10, 0, 10)
    lines, counts = label_frame(instances(a), {}, 40, 20)
    assert counts == {"car": 1}
    paint(a, TAG_TRUCK, 9001, 10, 20, 20, 30)
    lines, info = label_frame(instances(a), {}, 40, 20)
    assert lines is None and "tag 15" in info["rejected"][0]


def test_visible_motorbike_rejects_but_a_tiny_one_does_not():
    a = img()
    paint(a, TAG_BUS, 1, 0, 10, 0, 10)
    paint(a, TAG_MOTORCYCLE, 2, 12, 14, 30, 32)  # 4 px: below the 24 px rule
    lines, counts = label_frame(instances(a), {1: "bus", 2: None}, 40, 20)
    assert counts == {"bus": 1}
    paint(a, TAG_MOTORCYCLE, 2, 10, 20, 30, 40)  # now 100 px
    lines, _ = label_frame(instances(a), {1: "bus", 2: None}, 40, 20)
    assert lines is None


def test_too_small_objects_are_skipped():
    a = img()
    paint(a, TAG_CAR, 7, 0, 2, 0, 30)  # 60 px but only 2 px tall
    lines, counts = label_frame(instances(a), {7: "car"}, 40, 20)
    assert lines == [] and counts == {}


def test_tag_numbers_are_carlas():
    import carla

    from cosim import labels

    assert labels.TAG_CAR == int(carla.CityObjectLabel.Car)
    assert labels.TAG_TRUCK == int(carla.CityObjectLabel.Truck)
    assert labels.TAG_BUS == int(carla.CityObjectLabel.Bus)
    assert labels.TAG_PEDESTRIAN == int(carla.CityObjectLabel.Pedestrians)
    assert labels.TAG_MOTORCYCLE == int(carla.CityObjectLabel.Motorcycle)


def test_mean_luma_flags_unlit_frames():
    from cosim.labels import UNLIT_LUMA, mean_luma

    black = np.zeros((40, 40, 4), np.uint8)
    lit = black.copy()
    lit[..., :3] = 30  # Town10 at night measures 20-32
    assert mean_luma(black) < UNLIT_LUMA < mean_luma(lit)
    white = np.full((8, 8, 4), 255, np.uint8)
    assert abs(mean_luma(white) - 255) < 1e-3

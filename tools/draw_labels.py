"""Draw a dataset's YOLO labels onto its images, for checking by eye (needs Pillow).

    python tools/draw_labels.py out/dataset_carla train 12 out/dataset_carla/review
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["car", "van", "bus", "truck", "ambulance", "ambulance_lights", "police", "police_lights", "pedestrian"]
COLOURS = ["#00e5ff", "#ffd600", "#ff6d00", "#d500f9", "#ffffff", "#ff1744", "#2979ff", "#00e676", "#f50057"]


def main(root, split, n, out, seed=1):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    for p in imgs[: int(n)]:
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        W, H = im.size
        lab = root / "labels" / split / (p.stem + ".txt")
        for line in lab.read_text().split("\n") if lab.exists() else []:
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            c = int(c)
            x1, y1 = (float(cx) - float(w) / 2) * W, (float(cy) - float(h) / 2) * H
            x2, y2 = x1 + float(w) * W, y1 + float(h) * H
            d.rectangle([x1, y1, x2, y2], outline=COLOURS[c], width=2)
            d.text((x1 + 2, max(0, y1 - 11)), CLASSES[c], fill=COLOURS[c])
        im.save(out / p.name, quality=85)


if __name__ == "__main__":
    main(*sys.argv[1:5])

"""The traffic-camera detector as ONNX (31 Oct): out/detector_onnx/carla-yolo26n-t05.onnx, for tools that run it
without PyTorch (onnxruntime). Needs tools/setup_ml.bat's packages.

    python tools/export_detector_onnx.py [--imgsz 960]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/carla-yolo26n-t05.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    a = ap.parse_args(argv)
    from ultralytics import YOLO

    out = ROOT / "out" / "detector_onnx"
    out.mkdir(parents=True, exist_ok=True)
    src = out / Path(a.model).name
    shutil.copyfile(ROOT / a.model, src)
    path = YOLO(str(src)).export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False)
    print("exported", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

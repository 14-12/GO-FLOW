"""Fine-tune YOLO26n on the CARLA dataset (out/dataset_carla_full) and export it.

    python tools/train_detector.py                      # GPU 0, 960 px, 80 epochs
    python tools/train_detector.py --device cpu --epochs 2 --imgsz 320   # quick check

Same recipe as the main project's scripts/perception_train.py (Ultralytics YOLO26n from
the COCO weights, fixed seed, best epoch chosen on the val split), on a GPU and at 960 px
instead of 640: the frames here are 1280x720 and many vehicles are far away.

Writes models/<name>.pt, models/<name>.onnx and models/<name>.json (settings, data counts,
versions, time). Accuracy is measured separately (tools/eval_detector.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolved_yaml(data: Path, out_dir: Path) -> Path:
    """A copy of a dataset.yaml whose `path` is absolute (Ultralytics would otherwise read a
    relative path against its own datasets folder)."""
    lines = [ln for ln in data.read_text(encoding="utf-8").splitlines() if not ln.startswith("path:")]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{data.parent.name}.yaml"
    out.write_text(f"path: {data.parent.resolve().as_posix()}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/dataset_carla_full/dataset.yaml")
    ap.add_argument("--base", default="yolo26n.pt", help="starting weights (Ultralytics COCO)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4, help="4 fits a 4 GB GPU at 960 px (8 needed 5.4 GB)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="carla-yolo26n")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from out/detector/<name>/weights/last.pt")
    a = ap.parse_args(argv)

    import torch
    import ultralytics
    from ultralytics import YOLO

    runs = ROOT / "out" / "detector"
    data = resolved_yaml((ROOT / a.data).resolve(), runs)
    weights_dir = ROOT / "models"
    weights_dir.mkdir(exist_ok=True)
    base = weights_dir / a.base
    model = YOLO(str(base) if base.exists() else a.base)  # downloads the COCO weights once
    if not base.exists() and Path(a.base).exists():
        shutil.copy2(a.base, base)
    t0 = time.time()
    run = runs / a.name
    last = run / "weights" / "last.pt"
    resumed_after = None
    if a.resume and last.exists():
        # Ultralytics restores the run's own settings (data, epochs, seed, ...) from last.pt
        resumed_after = sum(1 for _ in (run / "results.csv").open()) - 1
        YOLO(str(last)).train(resume=True)
    else:
        model.train(data=str(data), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
                    device=a.device, workers=a.workers, seed=a.seed, deterministic=True, project=str(runs),
                    name=a.name, exist_ok=True, plots=True, verbose=False)
    seconds = round(time.time() - t0)
    best = run / "weights" / "best.pt"
    shutil.copy2(best, weights_dir / f"{a.name}.pt")
    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=a.imgsz, simplify=True))
    shutil.copy2(onnx, weights_dir / f"{a.name}.onnx")
    data_root = (ROOT / a.data).resolve().parent
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    results = (run / "results.csv").read_text(encoding="utf-8").splitlines()
    card = {
        "name": a.name,
        "classes": manifest["classes"],
        "base": f"{a.base} (Ultralytics COCO weights)",
        "data": {"root": data_root.relative_to(ROOT).as_posix() if data_root.is_relative_to(ROOT) else str(data_root),
                 "totals": manifest["totals"], "split": manifest.get("split"),
                 "conditions": manifest.get("conditions")},
        "training": {"epochs_max": a.epochs, "epochs_run": len(results) - 1, "patience": a.patience,
                     "imgsz": a.imgsz, "batch": a.batch, "device": a.device, "seed": a.seed,
                     "seconds": seconds, "resumed_after_epoch": resumed_after,
                     "selection": "best fitness on the val split"},
        "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                     "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     else None},
        "files": {"pt": f"models/{a.name}.pt", "onnx": f"models/{a.name}.onnx", "run": f"out/detector/{a.name}"},
        "licence": "AGPL-3.0 (Ultralytics YOLO26 weights and code)",
    }
    (weights_dir / f"{a.name}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card["training"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

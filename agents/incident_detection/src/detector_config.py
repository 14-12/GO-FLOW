"""Detector settings (weights, class map, confidence, image size) loaded from a JSON file."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

CATEGORIES = ("car", "motorcycle", "bus", "truck")
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_CONFIG = CONFIG_DIR / "detector_v1.json"


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    weights: str
    class_map: dict = field(default_factory=dict)  # detector class name -> feature category
    confidence: float = 0.25
    imgsz: int | None = None
    tracker: str = "bytetrack.yaml"

    def __post_init__(self):
        unknown = sorted(set(self.class_map.values()) - set(CATEGORIES))
        if not self.class_map or unknown:
            raise ValueError(f"class_map must map detector classes to {CATEGORIES}; got {unknown or 'nothing'}")
        if not 0 < self.confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")

    def class_ids(self, names):
        """Resolve class names against the model's own names ({id: name}) to {id: category}."""
        lookup = {str(name).lower(): int(index) for index, name in dict(names).items()}
        missing = sorted(name for name in self.class_map if name.lower() not in lookup)
        if missing:
            raise ValueError(f"Detector '{self.name}' has no classes {missing}; it knows {sorted(lookup)}")
        return {lookup[name.lower()]: category for name, category in self.class_map.items()}


def load_detector_config(path=None):
    """Read a config file; GOFLOW_DETECTOR_CONFIG picks the file, GOFLOW_YOLO_WEIGHTS overrides weights."""
    path = Path(path or os.environ.get("GOFLOW_DETECTOR_CONFIG") or DEFAULT_CONFIG)
    raw = json.loads(path.read_text(encoding="utf-8"))
    weights = os.environ.get("GOFLOW_YOLO_WEIGHTS") or raw["weights"]
    # Weights next to the config file win; otherwise Ultralytics resolves the name itself.
    if not Path(weights).is_absolute() and (path.parent / weights).is_file():
        weights = str(path.parent / weights)
    return DetectorConfig(name=raw["name"], weights=weights, class_map=raw["class_map"],
                          confidence=float(raw.get("confidence", 0.25)), imgsz=raw.get("imgsz"),
                          tracker=raw.get("tracker", "bytetrack.yaml"))

"""Causal features copied from the training notebook's inference contract."""
import pandas as pd

CURRENT_COLUMNS = (
    "unique_vehicle_count", "mean_vehicle_count_per_frame", "mean_normalized_speed",
    "median_normalized_speed", "stopped_vehicle_count", "stopped_ratio",
    "bbox_coverage_proxy", "max_bbox_coverage_proxy", "mean_detection_confidence",
    "car_unique", "motorcycle_unique", "bus_unique", "truck_unique",
)
TEMPORAL_COLUMNS = CURRENT_COLUMNS[:7]
FEATURE_COLUMNS = CURRENT_COLUMNS + tuple(f"delta_{c}" for c in TEMPORAL_COLUMNS) + tuple(
    f"roll3_{c}" for c in TEMPORAL_COLUMNS
)


def add_temporal_features(windows: pd.DataFrame) -> pd.DataFrame:
    missing = set(CURRENT_COLUMNS) - set(windows.columns)
    if missing:
        raise ValueError(f"Missing incident features: {sorted(missing)}")
    result = windows.sort_values("window_start_sec").copy()
    if result["window_start_sec"].duplicated().any():
        raise ValueError("Duplicate incident window timestamps")
    for column in TEMPORAL_COLUMNS:
        result[f"delta_{column}"] = result[column].diff().fillna(0)
        result[f"roll3_{column}"] = result[column].rolling(3, min_periods=1).mean()
    return result

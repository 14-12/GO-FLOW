"""Time base and thresholds. Every number here is a setting named in the integration plan, kept in one place."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

VEH_M = 7.5          # metres of lane per queued vehicle (cosim/agents_actuated.VEH_M)
FULL = 0.85          # fill at which an approach counts as full (agents_actuated full_share / balance_high)
LOW = 0.60           # an approach below this now is still "flowing" (alert only those)
CYCLE_S = 115.0      # the coordinated cycle (scenario.json "coordinated.cycle_s")


@dataclass(frozen=True)
class TimeBase:
    dt_s: float = 30.0                                  # one step = mean over this many SUMO seconds
    n_in: int = 12                                      # steps of history (6 min)
    horizons_s: tuple[int, ...] = (60, 120, 300)        # forecast horizons, SUMO seconds

    def __post_init__(self):
        for h in self.horizons_s:
            k = h / self.dt_s
            if abs(k - round(k)) > 1e-9 or k < 1:
                raise ValueError(f"horizon {h} s is not a whole number of {self.dt_s} s steps")

    @property
    def steps(self) -> dict[str, int]:
        """'60' -> 2 steps, ... (keys are strings: they are JSON keys too)."""
        return {str(h): int(round(h / self.dt_s)) for h in self.horizons_s}

    @property
    def max_steps(self) -> int:
        return max(self.steps.values())

    @property
    def cycle_steps(self) -> int:
        return max(1, int(round(CYCLE_S / self.dt_s)))


@dataclass(frozen=True)
class AlertRules:
    full: float = FULL             # fixed thresholds: used only when a model has no per-approach thresholds
    low: float = LOW
    cooldown_s: float = 120.0      # no new alert on the same approach for this long
    eval_window: int = 12          # matured forecasts used by the self-evaluation (per horizon)


# Per-approach thresholds (decision A, 6 Nov 2026). At the viewer's demand a queue almost never fills 85 % of a whole
# approach (D2_x1_coord_v1: 3 approaches of 55, no onset in the dev seeds), so "full" is relative to each approach:
# its fill exceeded only 1 % of the time on the TRAIN seeds (GO-FLOW's own method: quantiles of the train data).
FULL_Q = 0.99                      # "unusually long queue" for this approach
LOW_Q = 0.90                       # "still flowing": under what this approach exceeds 10 % of the time
MIN_QUEUE_VEH = 3.0                # never "unusually long" under 3 queued vehicles (nearly empty approaches)


def approach_thresholds(fill, storage, rows):
    """-> (full [N], low [N]) from the train rows only. full >= MIN_QUEUE_VEH / storage; low < full."""
    import numpy as np

    f = fill[rows]
    with np.errstate(all="ignore"):
        q_full = np.nanquantile(f, FULL_Q, axis=0)
        q_low = np.nanquantile(f, LOW_Q, axis=0)
    floor = MIN_QUEUE_VEH / np.maximum(storage, 1.0)
    full = np.fmax(np.nan_to_num(q_full, nan=1.0), floor)
    low = np.minimum(np.nan_to_num(q_low, nan=0.0), 0.7 * full)
    return full, low


@dataclass
class ModelCard:
    """Written next to every trained model: what it is and where it came from (no number without a run)."""
    version: str
    timebase: dict = field(default_factory=lambda: asdict(TimeBase()))
    source: str = "exact"           # exact | cameras
    nodes: list = field(default_factory=list)
    train_seeds: list = field(default_factory=list)
    dev_seeds: list = field(default_factory=list)
    features: list = field(default_factory=list)
    use_node_id: bool = True
    use_goflow: bool = False
    dataset: str = ""
    git_commit: str = ""
    metrics_dev: dict = field(default_factory=dict)

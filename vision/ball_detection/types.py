from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HsvRange:
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    valid: bool
    x_px: int
    y_px: int
    radius_px: float
    area_px: float

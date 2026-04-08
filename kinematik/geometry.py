
import math
from dataclasses import dataclass

import numpy as np


# alle Längen in mm, Winkel in Grad
@dataclass(frozen=True)
class PlatformGeometry:
    plate_radius: float = 125.0
    standard_height: float = 165.0
    upper_arm_length: float = 179.2
    lower_arm_length: float = 70.0
    platform_link_length: float = 216.5064
    delta_angle_deg: float = 120.0

    @property
    def arm_pos_front(self) -> np.ndarray:
        return np.array([0.0, self.plate_radius, 0.0], dtype=np.float64)

    @property
    def arm_pos_left(self) -> np.ndarray:
        delta = math.radians(self.delta_angle_deg - 90.0)
        return np.array(
            [-math.cos(delta) * self.plate_radius, -math.sin(delta) * self.plate_radius, 0.0],
            dtype=np.float64,
        )

    @property
    def arm_pos_right(self) -> np.ndarray:
        left = self.arm_pos_left
        return np.array([-left[0], left[1], 0.0], dtype=np.float64)

    @property
    def ref_point(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.standard_height], dtype=np.float64)

from __future__ import annotations

import numpy as np

from hardware.base import ServoDriver
from kinematik.engine import KinematicsEngine
from kinematik.geometry import PlatformGeometry


class Platform:
    # Verbindet Kinematik und Hardware, einziger Kontaktpunkt für den Regler

    def __init__(
        self,
        driver: ServoDriver,
        geometry: PlatformGeometry | None = None,
        solver_method: str = "scipy",  # "scipy" robuster, "newton" schneller
    ) -> None:
        self._driver = driver
        self._engine = KinematicsEngine(
            geometry=geometry or PlatformGeometry(),
            solver_method=solver_method,
        )

    @property
    def geometry(self):
        return self._engine.geometry

    def set_tilt(self, winkel_x: float, winkel_y: float, ref_point: np.ndarray | None = None) -> None:
        result = self._engine.solve(winkel_x, winkel_y, ref_point)
        self._driver.set_angles(result.servo_v, result.servo_l, result.servo_r)

    def neutral(self) -> None:
        self.set_tilt(0.0, 0.0)

    def close(self) -> None:
        self.neutral()
        self._driver.close()

    def __enter__(self) -> Platform:
        return self

    def __exit__(self, *_) -> None:
        self.close()

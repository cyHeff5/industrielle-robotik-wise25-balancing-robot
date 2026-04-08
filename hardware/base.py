from __future__ import annotations

from abc import ABC, abstractmethod


# Basisklasse für alle Servo-Treiber, alle Winkel in Grad [0, 180]
class ServoDriver(ABC):

    @abstractmethod
    def set_angles(self, angle_v: float, angle_l: float, angle_r: float) -> None: ...

    @abstractmethod
    def neutral(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> ServoDriver:
        return self

    def __exit__(self, *_) -> None:
        self.close()

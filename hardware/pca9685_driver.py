from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .base import ServoDriver


# offset_v/l/r: mechanischer Montage-Offset pro Servo; modifier: -1 zum Invertieren
@dataclass(frozen=True)
class ServoConfig:
    offset_v: float = 18.0
    offset_l: float = 35.0
    offset_r: float = 17.0
    modifier: float = 1.0
    pulse_min_us: int = 500
    pulse_max_us: int = 2500

    @staticmethod
    def from_config(path: str | Path) -> ServoConfig:
        data = json.loads(Path(path).read_text())
        return ServoConfig(
            offset_v=float(data.get("offset_v", 18.0)),
            offset_l=float(data.get("offset_l", 35.0)),
            offset_r=float(data.get("offset_r", 17.0)),
            modifier=float(data.get("modifier", 1.0)),
        )


def _apply_mapping(angle: float, offset: float, modifier: float) -> float:
    return max(0.0, min(180.0, angle * modifier + offset))


class PCA9685Driver(ServoDriver):

    PCA_ADDRESS = 0x40
    CHANNEL_V = 0
    CHANNEL_L = 1
    CHANNEL_R = 2

    def __init__(self, config: ServoConfig) -> None:
        from adafruit_servokit import ServoKit  # only imported on Pi

        self.config = config
        self._kit = ServoKit(channels=16, address=self.PCA_ADDRESS)

        for ch in (self.CHANNEL_V, self.CHANNEL_L, self.CHANNEL_R):
            self._kit.servo[ch].set_pulse_width_range(config.pulse_min_us, config.pulse_max_us)

    @staticmethod
    def from_config(path: str | Path) -> PCA9685Driver:
        return PCA9685Driver(ServoConfig.from_config(path))

    def set_angles(self, angle_v: float, angle_l: float, angle_r: float) -> None:
        cfg = self.config
        self._kit.servo[self.CHANNEL_V].angle = _apply_mapping(angle_v, cfg.offset_v, cfg.modifier)
        self._kit.servo[self.CHANNEL_L].angle = _apply_mapping(angle_l, cfg.offset_l, cfg.modifier)
        self._kit.servo[self.CHANNEL_R].angle = _apply_mapping(angle_r, cfg.offset_r, cfg.modifier)

    def neutral(self) -> None:
        self.set_angles(90.0, 90.0, 90.0)

    def close(self) -> None:
        pass

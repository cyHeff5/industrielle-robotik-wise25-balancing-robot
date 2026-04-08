
from .base import ServoDriver
from .pca9685_driver import ServoConfig, _apply_mapping


class GPIODriver(ServoDriver):

    def __init__(self, config: ServoConfig, pin_v: int = 27, pin_l: int = 17, pin_r: int = 22) -> None:
        from gpiozero import AngularServo  # only imported on Pi

        pulse_min = config.pulse_min_us / 1_000_000
        pulse_max = config.pulse_max_us / 1_000_000

        self.config = config
        self._servo_v = AngularServo(pin_v, min_angle=0, max_angle=180, min_pulse_width=pulse_min, max_pulse_width=pulse_max)
        self._servo_l = AngularServo(pin_l, min_angle=0, max_angle=180, min_pulse_width=pulse_min, max_pulse_width=pulse_max)
        self._servo_r = AngularServo(pin_r, min_angle=0, max_angle=180, min_pulse_width=pulse_min, max_pulse_width=pulse_max)

    def set_angles(self, angle_v: float, angle_l: float, angle_r: float) -> None:
        cfg = self.config
        self._servo_v.angle = _apply_mapping(angle_v, cfg.offset_v, cfg.modifier)
        self._servo_l.angle = _apply_mapping(angle_l, cfg.offset_l, cfg.modifier)
        self._servo_r.angle = _apply_mapping(angle_r, cfg.offset_r, cfg.modifier)

    def neutral(self) -> None:
        self.set_angles(90.0, 90.0, 90.0)

    def close(self) -> None:
        self._servo_v.close()
        self._servo_l.close()
        self._servo_r.close()

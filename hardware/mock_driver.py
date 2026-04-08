
from .base import ServoDriver


# kein Hardware-Import, zum Testen auf dem Laptop
class MockServoDriver(ServoDriver):

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.last_v: float = 90.0
        self.last_l: float = 90.0
        self.last_r: float = 90.0

    def set_angles(self, angle_v: float, angle_l: float, angle_r: float) -> None:
        self.last_v = angle_v
        self.last_l = angle_l
        self.last_r = angle_r
        if self.verbose:
            print(f"MockServo: V={angle_v:.1f}°  L={angle_l:.1f}°  R={angle_r:.1f}°")

    def neutral(self) -> None:
        self.set_angles(90.0, 90.0, 90.0)

    def close(self) -> None:
        pass

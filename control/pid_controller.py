from __future__ import annotations


class PIDController:
    """PID-Regler mit Anti-Windup und Ausgangs-Sattigung.

    Reglerparameter aus Polplatzierung:
        Kp = omega_n^2 / K_sys
        Kd = 2 * zeta * omega_n / K_sys
        mit K_sys = (5/7) * g fuer einen rollenden Ball
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        max_out: float,
        max_int: float,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_out = float(max_out)
        self.max_int = float(max_int)
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        error = setpoint - measured

        self._integral += error * dt
        self._integral = max(-self.max_int, min(self.max_int, self._integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.max_out, min(self.max_out, output))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class LowPassFilter:
    """Exponentieller Tiefpass fuer theta und phi (Plattformwinkel)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self._theta: float | None = None
        self._phi: float | None = None

    def update(self, raw_theta: float, raw_phi: float) -> tuple[float, float]:
        if self._theta is None:
            self._theta = raw_theta
            self._phi = raw_phi
        else:
            self._theta = self.alpha * raw_theta + (1.0 - self.alpha) * self._theta
            self._phi = self.alpha * raw_phi + (1.0 - self.alpha) * self._phi
        return self._theta, self._phi

    def reset(self) -> None:
        self._theta = None
        self._phi = None

from __future__ import annotations


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class PDController:
    """Minimal PD controller with derivative clamp and output saturation."""

    def __init__(
        self,
        kp: float,
        kd: float,
        u_min: float,
        u_max: float,
        d_max_abs: float,
    ) -> None:
        self.kp = float(kp)
        self.kd = float(kd)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.d_max_abs = float(d_max_abs)
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        derivative = 0.0
        if self._prev_error is not None and dt > 1e-9:
            derivative = (error - self._prev_error) / dt
            derivative = _clamp(derivative, -self.d_max_abs, self.d_max_abs)

        self._prev_error = float(error)
        u = self.kp * float(error) + self.kd * derivative
        return _clamp(u, self.u_min, self.u_max)


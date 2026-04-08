from __future__ import annotations

from dataclasses import dataclass

from .runtime import BallDetectorRuntime, BallMeasurement


@dataclass(frozen=True)
class BallState:
    valid: bool
    is_new_frame: bool  # True wenn dies ein frischer Kamera-Frame ist
    timestamp_s: float  # Zeitstempel der Bildaufnahme
    x: float            # Position X relativ zur Bildmitte [px]
    y: float            # Position Y relativ zur Bildmitte [px]
    vx: float           # Geschwindigkeit X [px/s]
    vy: float           # Geschwindigkeit Y [px/s]


class BallTracker:
    # is_new_frame=False: Frame schon gesehen, Regler soll praeditieren

    def __init__(self, runtime: BallDetectorRuntime) -> None:
        self._runtime = runtime
        self._prev: BallMeasurement | None = None
        self._last_state: BallState | None = None

    def get(self) -> BallState:
        m = self._runtime.get_last_measurement()

        if m is None or not m.valid:
            self._prev = None
            self._last_state = BallState(
                valid=False, is_new_frame=False, timestamp_s=0.0,
                x=0.0, y=0.0, vx=0.0, vy=0.0,
            )
            return self._last_state

        # Bereits verarbeiteten Frame erkennen (Kamera langsamer als Regler)
        if self._prev is not None and m.timestamp_s == self._prev.timestamp_s:
            return BallState(
                valid=True,
                is_new_frame=False,
                timestamp_s=self._last_state.timestamp_s,
                x=self._last_state.x,
                y=self._last_state.y,
                vx=self._last_state.vx,
                vy=self._last_state.vy,
            )

        # Echter neuer Frame: Geschwindigkeit berechnen
        vx_raw, vy_raw = 0.0, 0.0
        if self._prev is not None and self._prev.valid:
            dt = m.timestamp_s - self._prev.timestamp_s
            if dt > 0:
                vx_raw = (m.x_rel_px - self._prev.x_rel_px) / dt
                vy_raw = (m.y_rel_px - self._prev.y_rel_px) / dt

        self._prev = m
        self._last_state = BallState(
            valid=True,
            is_new_frame=True,
            timestamp_s=m.timestamp_s,
            x=float(m.x_rel_px),
            y=float(m.y_rel_px),
            vx=vx_raw,
            vy=vy_raw,
        )
        return self._last_state

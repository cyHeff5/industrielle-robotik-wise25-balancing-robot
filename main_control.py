from __future__ import annotations

import time

import numpy as np

from balancing_platform import Platform
from control.pid_controller import PIDController
from hardware import MockServoDriver, PCA9685Driver
from vision.ball_detection import BallDetectorRuntime, BallTracker, HsvRange

# --- Konfiguration ---

FRAME_SIZE   = (640, 480)
CONTROL_HZ   = 120.0
PX_TO_MM     = 0.5        # Pixel → Millimeter
BALL_LOSS_TIMEOUT_S = 0.5 # Sekunden bis Neutral nach Ball-Verlust

HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

# PD-Regler Parameter
KP       = 0.015
KD       = 0.0
MAX_TILT = 20.0   # maximaler Neigungswinkel [°]
D_MAX    = 600.0  # maximale Ableitung [°/s]

# --- Setup ---

driver   = PCA9685Driver.from_config("config.json")  # swap für MockServoDriver(verbose=False) zum Testen
platform = Platform(driver)

detector = BallDetectorRuntime(
    hsv_range=HSV_RANGE,
    frame_size=FRAME_SIZE,
    use_roi_tracking=True,
    downsample_factor=2,
)
tracker = BallTracker(detector, ema_alpha=0.3)

pid_x = PIDController(kp=KP, ki=0.0, kd=KD, max_out=MAX_TILT, max_int=10.0)
pid_y = PIDController(kp=KP, ki=0.0, kd=KD, max_out=MAX_TILT, max_int=10.0)

period_s = 1.0 / CONTROL_HZ
detector.start()

while detector.get_latest_frame() is None:
    time.sleep(0.05)

# Zustand für Prediction-Mode und Ball-Loss
pos_x_mm: float = 0.0
pos_y_mm: float = 0.0
last_seen_s: float | None = None

try:
    platform.neutral()
    next_tick = time.perf_counter()

    while True:
        now = time.perf_counter()
        state = tracker.get()

        if state.valid:
            last_seen_s = now

            if state.is_new_frame:
                # Neuer Kamera-Frame: Position direkt übernehmen
                pos_x_mm = state.x * PX_TO_MM
                pos_y_mm = state.y * PX_TO_MM
            else:
                # Prediction-Mode: Position mit gefilterter Geschwindigkeit extrapolieren
                pos_x_mm += state.vx * PX_TO_MM * period_s
                pos_y_mm += state.vy * PX_TO_MM * period_s

            # Regler: Ziel = 0,0 (Bildmitte)
            winkel_x = pid_x.update(0.0, pos_x_mm, period_s)
            winkel_y = pid_y.update(0.0, pos_y_mm, period_s)

            platform.set_tilt(winkel_x, winkel_y)

        else:
            # Ball nicht gesehen
            if last_seen_s is not None and (now - last_seen_s) >= BALL_LOSS_TIMEOUT_S:
                pid_x.reset()
                pid_y.reset()
                pos_x_mm = 0.0
                pos_y_mm = 0.0
                last_seen_s = None
                platform.neutral()

        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

except KeyboardInterrupt:
    pass
finally:
    platform.close()
    detector.stop()

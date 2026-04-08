"""Hauptregelloop fuer die Balancing-Plattform."""

from __future__ import annotations

import math
import time

import numpy as np

from balancing_platform import Platform
from control.pid_controller import LowPassFilter, PIDController
from hardware import PCA9685Driver
from hardware.pca9685_driver import ServoConfig
from vision.ball_detection import BallDetectorRuntime, BallTracker, HsvRange

FRAME_SIZE = (640, 480)
CONTROL_HZ = 120.0
PREDICTION_TIMEOUT = 0.5   # Sekunden bis Ball als "verloren" gilt

HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

# Kamera ist verdreht montiert -- Korrekturwinkel aus zwei Punkten auf der Plattform
_CAM_P1 = (-193, -142)
_CAM_P2 = (277, 116)

# Regler-Parameter (Polplatzierung: omega_n=3, zeta=1.0, K_sys=5/7*g)
KP = 0.05
KI = 0.008
KD = 0.1 / 3
MAX_TILT = 45.0
MAX_INT = MAX_TILT / KI

# Umrechnung Pixel -> mm und Zielposition auf der Platte
PX_TO_MM = 0.5
TARGET_X_MM = 15.0
TARGET_Y_MM = -15.0


def _camera_rotation_angle(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.pi / 2 - math.atan2(dy, dx)


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    return (
        x * math.cos(angle) - y * math.sin(angle),
        x * math.sin(angle) + y * math.cos(angle),
    )


def main() -> None:
    cam_angle = _camera_rotation_angle(_CAM_P1, _CAM_P2)

    config = ServoConfig(offset_v=20, offset_l=35, offset_r=20, modifier=1.0)
    driver = PCA9685Driver(config)   # Swap: MockServoDriver(verbose=False) fuer Laptop

    with Platform(driver=driver) as platform:
        detector = BallDetectorRuntime(
            hsv_range=HSV_RANGE,
            frame_size=FRAME_SIZE,
            use_roi_tracking=True,
        )
        tracker = BallTracker(detector)

        pid_x = PIDController(KP, KI, KD, MAX_TILT, MAX_INT)
        pid_y = PIDController(KP, KI, KD, MAX_TILT, MAX_INT)
        angle_filter = LowPassFilter(alpha=0.06)

        period_s = 1.0 / CONTROL_HZ
        detector.start()

        print("Warte auf ersten Frame...")
        while detector.get_latest_frame() is None:
            time.sleep(0.05)

        print("Bereit. Ctrl+C zum Beenden.")
        platform.neutral()
        next_tick = time.perf_counter()

        # Zustand fuer Praediktion
        last_cam_time: float | None = None
        rot_x_px, rot_y_px = 0.0, 0.0
        rot_vx_px, rot_vy_px = 0.0, 0.0
        prev_time = time.perf_counter()
        last_print_time = 0.0

        try:
            while True:
                current_time = time.perf_counter()
                dt = current_time - prev_time
                prev_time = current_time

                if current_time - last_print_time > 1.0:
                    print(f"Regelfrequenz: {1.0 / dt:.1f} Hz")
                    last_print_time = current_time

                state = tracker.get()

                if state.valid:
                    if state.is_new_frame:
                        # Frisches Kamerabild: Messwerte direkt uebernehmen
                        last_cam_time = state.timestamp_s
                        rot_x_px, rot_y_px = _rotate(state.x, state.y, cam_angle)
                        rot_vx_px, rot_vy_px = _rotate(state.vx, state.vy, cam_angle)
                    else:
                        # Kein neues Bild: Position praeditieren
                        rot_x_px += rot_vx_px * dt
                        rot_y_px += rot_vy_px * dt

                    current_x_mm = rot_x_px * PX_TO_MM
                    current_y_mm = rot_y_px * PX_TO_MM

                    acc_x = pid_x.update(TARGET_X_MM, current_x_mm, dt)
                    acc_y = pid_y.update(TARGET_Y_MM, current_y_mm, dt)

                    theta = math.degrees(math.asin(max(-1.0, min(1.0, (acc_x / MAX_TILT) * 0.7))))
                    phi = math.degrees(math.asin(max(-1.0, min(1.0, (acc_y / MAX_TILT) * 0.7))))

                    theta_f, phi_f = angle_filter.update(theta, phi)

                    z_default = platform.geometry.ref_point[2]
                    ref_point = np.array([current_x_mm, current_y_mm, z_default])
                    platform.set_tilt(-phi_f, -theta_f, ref_point)

                # Timeout: Ball verloren
                if last_cam_time is not None and (current_time - last_cam_time) > PREDICTION_TIMEOUT:
                    pid_x.reset()
                    pid_y.reset()
                    angle_filter.reset()
                    platform.neutral()
                    last_cam_time = None

                next_tick += period_s
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()

        except KeyboardInterrupt:
            print("Gestoppt.")
        finally:
            detector.stop()


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
import time

import numpy as np

from balancing_platform import Platform
from control.pid_controller import LowPassFilter, PIDController
from hardware import PCA9685Driver
from vision.ball_detection import BallDetectorRuntime, BallTracker, HsvRange


# --- Hilfsfunktionen ---

def calculate_rotation_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    current_angle = math.atan2(dy, dx)
    target_angle = math.pi / 2
    return target_angle - current_angle

def rotate_point(point, angle):
    x, y = point
    x_new = x * math.cos(angle) - y * math.sin(angle)
    y_new = x * math.sin(angle) + y * math.cos(angle)
    return (x_new, y_new)


# --- Konfiguration ---

p1_kamera = (-193, -142)
p2_kamera = (277, 116)
angle_rad = calculate_rotation_angle(p1_kamera, p2_kamera)

FRAME_SIZE   = (640, 480)
CONTROL_HZ   = 120.0
PREDICTION_TIMEOUT = 0.5  # Sekunden bis Neutral nach Ball-Verlust

HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

# --- Setup ---

driver   = PCA9685Driver.from_config("config.json")
platform = Platform(driver)

detector = BallDetectorRuntime(
    hsv_range=HSV_RANGE,
    frame_size=FRAME_SIZE,
    use_roi_tracking=True,
)
tracker = BallTracker(detector)

period_s = 1.0 / CONTROL_HZ
detector.start()

# Regler-Parameter (aus Polplatzierung: Kp = wn^2/Ksys, Kd = 2*zeta*wn/Ksys)
KP      = 0.05
KI      = 0.008
KD      = 0.1 / 3
MAX_TILT = 45.0
MAX_INT  = MAX_TILT / KI

pid_x = PIDController(kp=KP, ki=KI, kd=KD, max_out=MAX_TILT, max_int=MAX_INT)
pid_y = PIDController(kp=KP, ki=KI, kd=KD, max_out=MAX_TILT, max_int=MAX_INT)

PX_TO_MM    = 0.5
TARGET_X_MM = 15.0
TARGET_Y_MM = -15.0

print("Waiting for first frame...")
while detector.get_latest_frame() is None:
    time.sleep(0.05)

print("Ready. Ctrl+C to stop.")
try:
    # Alpha-Werte:
    # 0.044 → sehr sanfte Regelung | 0.06 → empfohlen | 0.3 → mittlere Reaktion
    angle_filter = LowPassFilter(alpha=0.03)
    platform.neutral()
    next_tick = time.perf_counter()

    last_cam_time = None
    rot_x_px, rot_y_px = 0.0, 0.0
    rot_vx_px, rot_vy_px = 0.0, 0.0
    prev_time = time.perf_counter()
    last_print_time = 0.0

    while True:
        current_time = time.perf_counter()
        dt = current_time - prev_time
        prev_time = current_time

        if current_time - last_print_time > 1.0:
            if dt > 0:
                print(f"Aktuelle Frequenz: {1.0 / dt:.2f} Hz")
            last_print_time = current_time

        state = tracker.get()

        if state.valid:
            if state.is_new_frame:
                # Frisches Kamerabild: Messwerte direkt übernehmen
                last_cam_time = state.timestamp_s
                rot_x_px, rot_y_px = rotate_point((state.x, state.y), angle_rad)
                rot_vx_px, rot_vy_px = rotate_point((state.vx, state.vy), angle_rad)
            else:
                # Kein neues Bild: Position mit Geschwindigkeit prädizieren
                rot_x_px += rot_vx_px * dt
                rot_y_px += rot_vy_px * dt

            current_x_mm = rot_x_px * PX_TO_MM
            current_y_mm = rot_y_px * PX_TO_MM

            acc_x = pid_x.update(TARGET_X_MM, current_x_mm, dt)
            acc_y = pid_y.update(TARGET_Y_MM, current_y_mm, dt)

            theta = math.degrees(math.asin((acc_x / MAX_TILT) * 0.7))
            phi   = math.degrees(math.asin((acc_y / MAX_TILT) * 0.7))

            theta_f, phi_f = angle_filter.update(theta, phi)

            z_default = platform._engine.geometry.ref_point[2]
            rot_point_ball = (current_x_mm, current_y_mm, z_default)

            platform.set_tilt(-phi_f, -theta_f, rot_point_ball)

        # Ball-Verlust Timeout
        if last_cam_time is not None and (current_time - last_cam_time) > PREDICTION_TIMEOUT:
            pid_x.reset()
            pid_y.reset()
            angle_filter.reset()
            last_cam_time = None
            platform.neutral()

        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

except KeyboardInterrupt:
    print("Stopping.")
finally:
    platform.close()
    detector.stop()

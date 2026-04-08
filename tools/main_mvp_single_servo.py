from __future__ import annotations

import time

import numpy as np
from gpiozero import AngularServo

from kinematik.kinematik_main import func_servo_drehen
from vision.ball_detection.runtime import BallDetectorRuntime
from vision.ball_detection.types import HsvRange


# Camera / detection settings
FRAME_SIZE = (2592, 2592)
CAMERA_INPUT_COLOR_ORDER = "BGR"  # set to "RGB" or "BGR" depending on your camera output
USE_ROI_TRACKING = True
ROI_SIZE = 700
ROI_MISS_LIMIT = 5
DOWNSAMPLE_FACTOR = 2
HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

# Single-servo setup (same mapping style as existing project)
SERVO_PIN = 27
SERVO_MODIFIER = -1
SERVO_OFFSET = 176
NEUTRAL_ONLY_MODE = True

# Servo logic angles (degrees before modifier/offset mapping)
ANGLE_UP = 100.0
ANGLE_DOWN = 80.0
ANGLE_NEUTRAL = 90.0

# Decision logic
DEADBAND_PX = 25          # around image center
SWITCH_COOLDOWN_S = 0.15  # avoid fast toggling
LOST_TO_NEUTRAL_S = 0.30  # ball missing -> neutral
LOG_INTERVAL_S = 0.5


def main() -> None:
    servo = AngularServo(
        SERVO_PIN,
        min_angle=0,
        max_angle=180,
        min_pulse_width=0.0005,
        max_pulse_width=0.0025,
    )
    detector = BallDetectorRuntime(
        hsv_range=HSV_RANGE,
        frame_size=FRAME_SIZE,
        input_color_order=CAMERA_INPUT_COLOR_ORDER,
        use_roi_tracking=USE_ROI_TRACKING,
        roi_size=ROI_SIZE,
        roi_miss_limit=ROI_MISS_LIMIT,
        downsample_factor=DOWNSAMPLE_FACTOR,
    )

    state = "neutral"
    last_switch_s = 0.0
    last_seen_s: float | None = None
    last_log_s = time.perf_counter()

    if NEUTRAL_ONLY_MODE:
        try:
            func_servo_drehen(servo, ANGLE_NEUTRAL, SERVO_OFFSET, SERVO_MODIFIER)
            print("NEUTRAL_ONLY_MODE active: servo set to neutral and held.")
            print("Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping single-servo neutral mode.")
        finally:
            servo.angle = None
            time.sleep(0.2)
        return

    detector.start()
    try:
        func_servo_drehen(servo, ANGLE_NEUTRAL, SERVO_OFFSET, SERVO_MODIFIER)
        time.sleep(1.0)

        while True:
            m = detector.read()
            now = time.perf_counter()

            if m.valid:
                last_seen_s = now
            elif last_seen_s is not None and (now - last_seen_s) >= LOST_TO_NEUTRAL_S and state != "neutral":
                func_servo_drehen(servo, ANGLE_NEUTRAL, SERVO_OFFSET, SERVO_MODIFIER)
                state = "neutral"
                last_switch_s = now

            if m.valid and (now - last_switch_s) >= SWITCH_COOLDOWN_S:
                x = m.x_rel_px
                if x > DEADBAND_PX and state != "down":
                    func_servo_drehen(servo, ANGLE_DOWN, SERVO_OFFSET, SERVO_MODIFIER)
                    state = "down"
                    last_switch_s = now
                elif x < -DEADBAND_PX and state != "up":
                    func_servo_drehen(servo, ANGLE_UP, SERVO_OFFSET, SERVO_MODIFIER)
                    state = "up"
                    last_switch_s = now
                elif -DEADBAND_PX <= x <= DEADBAND_PX and state != "neutral":
                    func_servo_drehen(servo, ANGLE_NEUTRAL, SERVO_OFFSET, SERVO_MODIFIER)
                    state = "neutral"
                    last_switch_s = now

            if now - last_log_s >= LOG_INTERVAL_S:
                print(f"valid={m.valid} x_rel_px={m.x_rel_px:+4d} state={state}")
                last_log_s = now
    except KeyboardInterrupt:
        print("Stopping single-servo MVP.")
    finally:
        func_servo_drehen(servo, ANGLE_NEUTRAL, SERVO_OFFSET, SERVO_MODIFIER)
        detector.stop()


if __name__ == "__main__":
    main()

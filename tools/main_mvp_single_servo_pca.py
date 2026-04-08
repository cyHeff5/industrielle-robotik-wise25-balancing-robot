from __future__ import annotations

import time
from threading import Lock, Thread

import cv2
import numpy as np
from adafruit_servokit import ServoKit
from flask import Flask, Response

from kinematik.kinematik_main import func_servo_drehen
from vision.ball_detection.runtime import BallDetectorRuntime
from vision.ball_detection.types import HsvRange


# PCA9685 setup
PCA_ADDRESS = 0x40
PCA_CHANNEL_V = 0
PCA_CHANNEL_L = 1
PCA_CHANNEL_R = 2
PULSE_MIN_US = 500
PULSE_MAX_US = 2500

# Camera / detection
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

# Servo mapping (logic angle -> hardware angle)
modifier_v, offset_v = -1, 176
modifier_l, offset_l = -1, 176
modifier_r, offset_r = -1, 176

# Active servo for 1D MVP: "v", "l", or "r"
ACTIVE_SERVO = "v"

# Logic angles for active servo
ANGLE_UP = 100.0
ANGLE_DOWN = 80.0
ANGLE_NEUTRAL = 90.0

# Decision parameters
DEADBAND_PX = 25
SWITCH_COOLDOWN_S = 0.15
LOST_TO_NEUTRAL_S = 0.30
ENABLE_STREAM = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5000
STREAM_JPEG_QUALITY = 80

app = Flask(__name__)
_latest_jpg: bytes | None = None
_jpg_lock = Lock()


def _set_all_neutral(servos: dict[str, object], mappings: dict[str, tuple[float, float]]) -> None:
    for name, servo in servos.items():
        modifier, offset = mappings[name]
        func_servo_drehen(servo, ANGLE_NEUTRAL, offset, modifier)


def _set_active_angle(
    servos: dict[str, object],
    mappings: dict[str, tuple[float, float]],
    active_name: str,
    logic_angle: float,
) -> None:
    # Keep non-active servos neutral to hold a neutral plate baseline.
    _set_all_neutral(servos, mappings)
    modifier, offset = mappings[active_name]
    func_servo_drehen(servos[active_name], logic_angle, offset, modifier)


def _stream_generator():
    while True:
        with _jpg_lock:
            jpg = _latest_jpg
        if jpg is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")


@app.route("/")
def index():
    return "Single-servo PCA stream: /video\n"


@app.route("/video")
def video():
    return Response(
        _stream_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def main() -> None:
    kit = ServoKit(channels=16, address=PCA_ADDRESS)
    servo_v = kit.servo[PCA_CHANNEL_V]
    servo_l = kit.servo[PCA_CHANNEL_L]
    servo_r = kit.servo[PCA_CHANNEL_R]
    servo_v.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
    servo_l.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)
    servo_r.set_pulse_width_range(PULSE_MIN_US, PULSE_MAX_US)

    servos = {"v": servo_v, "l": servo_l, "r": servo_r}
    mappings = {
        "v": (modifier_v, offset_v),
        "l": (modifier_l, offset_l),
        "r": (modifier_r, offset_r),
    }
    if ACTIVE_SERVO not in servos:
        raise ValueError("ACTIVE_SERVO must be one of: 'v', 'l', 'r'")

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

    detector.start()
    try:
        if ENABLE_STREAM:
            Thread(
                target=lambda: app.run(host=STREAM_HOST, port=STREAM_PORT, threaded=True),
                daemon=True,
            ).start()

        _set_all_neutral(servos, mappings)
        time.sleep(1.0)

        while True:
            m, frame_bgr = detector.read_with_frame()
            now = time.perf_counter()

            if m.valid:
                last_seen_s = now
            elif last_seen_s is not None and (now - last_seen_s) >= LOST_TO_NEUTRAL_S and state != "neutral":
                _set_all_neutral(servos, mappings)
                state = "neutral"
                last_switch_s = now

            if m.valid and (now - last_switch_s) >= SWITCH_COOLDOWN_S:
                x = m.x_rel_px
                if x > DEADBAND_PX and state != "down":
                    _set_active_angle(servos, mappings, ACTIVE_SERVO, ANGLE_DOWN)
                    state = "down"
                    last_switch_s = now
                elif x < -DEADBAND_PX and state != "up":
                    _set_active_angle(servos, mappings, ACTIVE_SERVO, ANGLE_UP)
                    state = "up"
                    last_switch_s = now
                elif -DEADBAND_PX <= x <= DEADBAND_PX and state != "neutral":
                    _set_all_neutral(servos, mappings)
                    state = "neutral"
                    last_switch_s = now

            if ENABLE_STREAM:
                h, w = frame_bgr.shape[:2]
                cx = w // 2
                cv2.line(frame_bgr, (cx, 0), (cx, h - 1), (180, 180, 180), 1)
                if m.valid:
                    x_px = cx + m.x_rel_px
                    y_px = (h // 2) - m.y_rel_px
                    cv2.circle(frame_bgr, (x_px, y_px), int(max(3.0, m.radius_px)), (255, 0, 0), 2)
                    cv2.circle(frame_bgr, (x_px, y_px), 3, (255, 0, 0), -1)
                cv2.putText(
                    frame_bgr,
                    f"state={state} x_rel={m.x_rel_px:+d}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
                )
                if ok:
                    with _jpg_lock:
                        global _latest_jpg
                        _latest_jpg = jpg.tobytes()
    except KeyboardInterrupt:
        print("Stopping PCA single-servo MVP.")
    finally:
        _set_all_neutral(servos, mappings)
        time.sleep(0.2)
        # Release PWM signal
        servo_v.angle = None
        servo_l.angle = None
        servo_r.angle = None
        detector.stop()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from threading import Lock, Thread

import cv2
import numpy as np
from flask import Flask, Response

from vision.ball_detection.runtime import BallDetectorRuntime
from vision.ball_detection.types import HsvRange


# Camera / detection debug settings
FRAME_SIZE = (2592, 2592)
CAMERA_INPUT_COLOR_ORDER = "BGR"  # "RGB" or "BGR"
USE_ROI_TRACKING = True
ROI_SIZE = 700
ROI_MISS_LIMIT = 5
DOWNSAMPLE_FACTOR = 2

HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5001
JPEG_QUALITY = 80

app = Flask(__name__)
_latest_jpg: bytes | None = None
_jpg_lock = Lock()
_running = True


def _draw_overlay(frame_bgr: np.ndarray, measurement, dbg: dict) -> np.ndarray:
    out = frame_bgr
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # Global reference axes
    cv2.line(out, (cx, 0), (cx, h - 1), (180, 180, 180), 1)
    cv2.line(out, (0, cy), (w - 1, cy), (180, 180, 180), 1)

    # ROI rectangle (TOI) in global coordinates
    roi = dbg.get("roi")
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(out, "ROI/TOI", (x0 + 8, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Ball position from global coordinates
    if measurement.valid:
        x_px = cx + measurement.x_rel_px
        y_px = cy - measurement.y_rel_px
        cv2.circle(out, (x_px, y_px), int(max(3.0, measurement.radius_px)), (255, 0, 0), 2)
        cv2.circle(out, (x_px, y_px), 3, (255, 0, 0), -1)

    # Downsample preview inset
    ds = int(dbg.get("downsample_factor", 1))
    if ds > 1:
        small_w = max(1, w // ds)
        small_h = max(1, h // ds)
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_AREA)

        inset_w = min(360, w // 4)
        inset_h = int(inset_w * (small_h / max(1, small_w)))
        inset = cv2.resize(small, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)

        x1 = w - 20
        y1 = 20 + inset_h
        x0 = x1 - inset_w
        y0 = 20
        out[y0:y1, x0:x1] = inset
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"Downsample x{ds}",
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        out,
        f"ROI={dbg.get('use_roi_tracking')} miss={dbg.get('roi_misses')} valid={measurement.valid}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return out


def _producer_loop(detector: BallDetectorRuntime) -> None:
    global _latest_jpg
    while _running:
        measurement, frame_bgr = detector.read_with_frame()
        dbg = detector.debug_state()
        vis = _draw_overlay(frame_bgr, measurement, dbg)
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        with _jpg_lock:
            _latest_jpg = jpg.tobytes()


def _mjpeg_generator():
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
    return "ROI/Downsample debug stream: /video\n"


@app.route("/video")
def video():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    global _running
    parser = argparse.ArgumentParser(description="ROI/downsample debug stream.")
    parser.add_argument("--width", type=int, default=FRAME_SIZE[0], help="Frame width.")
    parser.add_argument("--height", type=int, default=FRAME_SIZE[1], help="Frame height.")
    parser.add_argument("--roi-size", type=int, default=ROI_SIZE, help="ROI side length in pixels.")
    parser.add_argument("--roi-miss-limit", type=int, default=ROI_MISS_LIMIT, help="ROI miss limit before reset.")
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE_FACTOR, help="Downsample factor (>=1).")
    parser.add_argument(
        "--input-color-order",
        choices=("RGB", "BGR"),
        default=CAMERA_INPUT_COLOR_ORDER,
        help="Color order returned by camera before OpenCV processing.",
    )
    parser.add_argument("--port", type=int, default=STREAM_PORT, help="HTTP stream port.")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.roi_size <= 0:
        raise ValueError("--roi-size must be > 0")
    if args.roi_miss_limit < 1:
        raise ValueError("--roi-miss-limit must be >= 1")
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")

    detector = BallDetectorRuntime(
        hsv_range=HSV_RANGE,
        frame_size=(args.width, args.height),
        input_color_order=args.input_color_order,
        use_roi_tracking=USE_ROI_TRACKING,
        roi_size=args.roi_size,
        roi_miss_limit=args.roi_miss_limit,
        downsample_factor=args.downsample,
    )

    detector.start()
    producer = Thread(target=_producer_loop, args=(detector,), daemon=True)
    producer.start()
    print(
        "ROI/Downsample debug stream running on "
        f"http://{STREAM_HOST}:{args.port}/video "
        f"(roi_size={args.roi_size}, downsample={args.downsample})"
    )

    try:
        app.run(host=STREAM_HOST, port=args.port, threaded=True)
    finally:
        _running = False
        detector.stop()


if __name__ == "__main__":
    main()

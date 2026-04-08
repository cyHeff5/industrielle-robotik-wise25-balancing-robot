#Debug-only MJPEG stream for visualizing ball detection.

#Not intended for closed-loop control runtime due to additional CPU load
#(overlay drawing + JPEG encoding + Flask serving).

#Verwendet den Frame aus BallDetectorRuntime, keine eigene Kamera.


import threading
from typing import Callable

import cv2
import numpy as np
from flask import Flask, Response

from .utils import draw_coordinate_system, pixel_to_center_coords
from .detector import detect_ball
from .types import HsvRange
from .runtime import BallMeasurement

JPEG_QUALITY = 80


def _annotate_full(frame_bgr: np.ndarray, hsv_range: HsvRange) -> np.ndarray:
    vis = frame_bgr.copy()
    draw_coordinate_system(vis, grid_step=100)
    h, w = vis.shape[:2]

    result = detect_ball(vis, hsv_range)
    if result.valid:
        cx, cy = pixel_to_center_coords(result.x_px, result.y_px, w, h)
        cv2.circle(vis, (result.x_px, result.y_px), int(result.radius_px), (0, 255, 0), 2)
        cv2.circle(vis, (result.x_px, result.y_px), 3, (0, 255, 0), -1)
        cv2.putText(vis, f"BALL ({cx:+d}, {cy:+d})", (result.x_px + 10, result.y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "Kein Ball", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def _annotate_debug(
    frame_bgr: np.ndarray,
    get_measurement: Callable[[], BallMeasurement | None],
) -> np.ndarray:
    vis = frame_bgr.copy()
    draw_coordinate_system(vis, grid_step=100)

    measurement = get_measurement()
    if measurement is not None and measurement.valid:
        # Kreis lokal im Debug-Frame zeichnen (Ball ist immer in der Mitte des ROI)
        h, w = vis.shape[:2]
        cx_local, cy_local = w // 2, h // 2
        cv2.circle(vis, (cx_local, cy_local), 8, (0, 255, 0), -1)
        # Koordinaten aus dem Vollbild anzeigen
        cv2.putText(vis,
                    f"BALL (Vollbild: {measurement.x_rel_px:+d}, {measurement.y_rel_px:+d})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "Kein Ball", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def _mjpeg_generator_full(get_frame: Callable[[], np.ndarray | None], hsv_range: HsvRange):
    while True:
        frame = get_frame()
        if frame is None:
            continue
        vis = _annotate_full(frame, hsv_range)
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


def _mjpeg_generator_debug(
    get_debug_frame: Callable[[], np.ndarray | None],
    get_measurement: Callable[[], BallMeasurement | None],
):
    while True:
        frame = get_debug_frame()
        if frame is None:
            continue
        vis = _annotate_debug(frame, get_measurement)
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


def start_stream(
    get_frame: Callable[[], np.ndarray | None],
    hsv_range: HsvRange,
    host: str = "0.0.0.0",
    port: int = 5000,
) -> None:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return "Ball tracker stream: <a href='/video'>/video</a>\n"

    @app.route("/video")
    def video():
        return Response(
            _mjpeg_generator_full(get_frame, hsv_range),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True),
        daemon=True,
    ).start()


def start_debug_stream(
    get_frame: Callable[[], np.ndarray | None],
    get_debug_frame: Callable[[], np.ndarray | None],
    get_measurement: Callable[[], BallMeasurement | None],
    hsv_range: HsvRange,
    host: str = "0.0.0.0",
    port: int = 5000,
) -> None:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return (
            "Ball tracker stream:<br>"
            "<a href='/video'>/video</a> – volles Bild<br>"
            "<a href='/debug'>/debug</a> – was detect_ball sieht (ROI + downsampled)<br>"
        )

    @app.route("/video")
    def video():
        return Response(
            _mjpeg_generator_full(get_frame, hsv_range),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/debug")
    def debug():
        return Response(
            _mjpeg_generator_debug(get_debug_frame, get_measurement),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True),
        daemon=True,
    ).start()

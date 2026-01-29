from flask import Flask, Response
import cv2
import numpy as np
from picamera2 import Picamera2

from .detector import detect_ball
from .types import HsvRange
from .utils import draw_coordinate_system, pixel_to_center_coords


app = Flask(__name__)

# Kamera initialisieren
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# HSV-Startwerte (anpassen)
HSV_RANGE = HsvRange(
    lower=np.array([90, 80, 60]),
    upper=np.array([130, 255, 255])
)

GRID_STEP = 100


def mjpeg_generator():
    while True:
        frame_rgb = picam2.capture_array()

        # RGB -> BGR fuer OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # ggf. Rot/Blau tauschen, falls noetig
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        draw_coordinate_system(frame_bgr, grid_step=GRID_STEP)

        result = detect_ball(frame_bgr, HSV_RANGE)
        if result.valid:
            center = (result.x_px, result.y_px)
            cv2.circle(frame_bgr, center, int(result.radius_px), (255, 0, 0), 2)
            cv2.circle(frame_bgr, center, 3, (255, 0, 0), -1)

            h, w = frame_bgr.shape[:2]
            cx, cy = pixel_to_center_coords(center[0], center[1], w, h)
            label = f"BALL ({cx}, {cy})"
            cv2.putText(frame_bgr, label, (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


@app.route("/")
def index():
    return "Ball tracker stream: /video\n"


@app.route("/video")
def video():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

from __future__ import annotations

import argparse
import time
from threading import Lock, Thread

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from balancing_platform import Platform
from hardware.mock_driver import MockServoDriver
from vision.ball_detection.runtime import BallDetectorRuntime
from vision.ball_detection.types import HsvRange

# Konfiguration 

FRAME_SIZE   = (2500, 2500)
HSV_RANGE    = HsvRange(
    lower=np.array([90,  80,  60], dtype=np.uint8),
    upper=np.array([130, 255, 255], dtype=np.uint8),
)

MAX_TILT_DEG = 20.0   # maximaler Kippwinkel in jede Richtung
STEP_DEG     = 2.0    # Schrittweite pro Tastendruck
STREAM_HOST  = "0.0.0.0"
STREAM_PORT  = 5000
JPEG_QUALITY = 75

# Gemeinsamer Zustand 

_winkel_x: float = 0.0
_winkel_y: float = 0.0
_tilt_lock = Lock()

_latest_jpg: bytes | None = None
_jpg_lock = Lock()

_last_measurement = {"valid": False, "x": 0, "y": 0}
_meas_lock = Lock()

_running = True

# Overlay 

def _draw_overlay(frame_bgr: np.ndarray, measurement, dbg: dict) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    cv2.line(out, (cx, 0), (cx, h - 1), (180, 180, 180), 1)
    cv2.line(out, (0, cy), (w - 1, cy), (180, 180, 180), 1)

    roi = dbg.get("roi")
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 0), 2)
        cv2.putText(out, "ROI", (x0 + 6, max(20, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    if measurement.valid:
        x_px = cx + measurement.x_rel_px
        y_px = cy - measurement.y_rel_px
        r = int(max(5, measurement.radius_px))
        cv2.circle(out, (x_px, y_px), r, (255, 80, 0), 2)
        cv2.circle(out, (x_px, y_px), 4, (255, 80, 0), -1)

    with _tilt_lock:
        wx, wy = _winkel_x, _winkel_y
    status = f"X={wx:+.1f}  Y={wy:+.1f}  ball={'OK' if measurement.valid else '--'}"
    cv2.putText(out, status, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out

# Producer-Thread 

def _producer_loop(detector: BallDetectorRuntime) -> None:
    global _latest_jpg
    last_ts: float = -1.0
    while _running:
        measurement, frame_bgr = detector.read_with_frame()

        # Keinen Frame doppelt encoden -- warten bis Background-Thread neues Bild liefert
        if measurement.timestamp_s == last_ts:
            time.sleep(0.005)
            continue
        last_ts = measurement.timestamp_s

        dbg = detector.debug_state()

        with _meas_lock:
            _last_measurement["valid"] = measurement.valid
            _last_measurement["x"]     = measurement.x_rel_px
            _last_measurement["y"]     = measurement.y_rel_px

        vis = _draw_overlay(frame_bgr, measurement, dbg)
        ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
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

# Flask 

app = Flask(__name__)
_platform: Platform | None = None

@app.route("/video")
def video():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/state")
def state():
    with _tilt_lock:
        wx, wy = _winkel_x, _winkel_y
    with _meas_lock:
        meas = dict(_last_measurement)
    return jsonify(winkel_x=wx, winkel_y=wy, **meas)

@app.route("/tilt", methods=["POST"])
def tilt():
    global _winkel_x, _winkel_y
    data = request.get_json(force=True)
    action = data.get("action", "")

    with _tilt_lock:
        if action == "neutral":
            _winkel_x = 0.0
            _winkel_y = 0.0
        elif action == "up":
            _winkel_y = max(-MAX_TILT_DEG, _winkel_y - STEP_DEG)
        elif action == "down":
            _winkel_y = min( MAX_TILT_DEG, _winkel_y + STEP_DEG)
        elif action == "left":
            _winkel_x = max(-MAX_TILT_DEG, _winkel_x - STEP_DEG)
        elif action == "right":
            _winkel_x = min( MAX_TILT_DEG, _winkel_x + STEP_DEG)
        wx, wy = _winkel_x, _winkel_y

    if _platform is not None:
        _platform.set_tilt(wx, wy)

    return jsonify(winkel_x=wx, winkel_y=wy)

@app.route("/")
def index():
    return _HTML

# HTML 

_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Balancing Robot – Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #111;
    color: #eee;
    font-family: sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
    gap: 20px;
  }
  h1 { font-size: 1.4rem; color: #0af; letter-spacing: 0.05em; }
  #stream { border: 2px solid #0af; border-radius: 6px; max-width: 100%; }
  #info {
    display: flex;
    gap: 40px;
    font-size: 1.1rem;
  }
  .label { color: #888; font-size: 0.8rem; text-transform: uppercase; }
  .value { font-size: 1.6rem; font-weight: bold; color: #0af; }
  #ball-state { color: #f80; }
  #ball-state.valid { color: #0f8; }
  #tilt-vis {
    width: 160px;
    height: 160px;
    position: relative;
  }
  #tilt-vis svg { width: 100%; height: 100%; }
  #keys {
    display: grid;
    grid-template-columns: repeat(3, 52px);
    grid-template-rows: repeat(2, 52px);
    gap: 6px;
    place-items: center;
  }
  .key {
    width: 50px; height: 50px;
    border: 2px solid #444;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: #aaa;
    transition: background 0.1s, border-color 0.1s;
  }
  .key.active { background: #0af; border-color: #0af; color: #111; }
  #space-key {
    width: 160px; height: 40px;
    border: 2px solid #444;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; color: #aaa;
    transition: background 0.1s, border-color 0.1s;
  }
  #space-key.active { background: #f84; border-color: #f84; color: #111; }
  p.hint { font-size: 0.8rem; color: #555; }
</style>
</head>
<body>
<h1>Balancing Robot &mdash; Live Demo</h1>

<img id="stream" src="/video" alt="Kamerastream">

<div id="info">
  <div>
    <div class="label">Winkel X</div>
    <div class="value" id="wx">0.0°</div>
  </div>
  <div>
    <div class="label">Winkel Y</div>
    <div class="value" id="wy">0.0°</div>
  </div>
  <div>
    <div class="label">Ball</div>
    <div class="value" id="ball-state">--</div>
  </div>
</div>

<div id="tilt-vis">
  <svg viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="46" fill="none" stroke="#333" stroke-width="2"/>
    <line x1="50" y1="4" x2="50" y2="96" stroke="#333" stroke-width="1"/>
    <line x1="4" y1="50" x2="96" y2="50" stroke="#333" stroke-width="1"/>
    <circle id="dot" cx="50" cy="50" r="8" fill="#0af"/>
  </svg>
</div>

<div id="keys">
  <div></div>
  <div class="key" id="key-up">↑</div>
  <div></div>
  <div class="key" id="key-left">←</div>
  <div class="key" id="key-down">↓</div>
  <div class="key" id="key-right">→</div>
</div>
<div id="space-key">Leertaste &mdash; Neutral</div>

<p class="hint">Pfeiltasten oder WASD zum Steuern &bull; Leertaste = Neutral</p>

<script>
const MAX = 20.0;
const keyMap = {
  ArrowUp:    "up",    w: "up",
  ArrowDown:  "down",  s: "down",
  ArrowLeft:  "left",  a: "left",
  ArrowRight: "right", d: "right",
  " ":        "neutral",
};
const keyEl = {
  up: document.getElementById("key-up"),
  down: document.getElementById("key-down"),
  left: document.getElementById("key-left"),
  right: document.getElementById("key-right"),
};

function flash(action) {
  const el = action === "neutral"
    ? document.getElementById("space-key")
    : keyEl[action];
  if (!el) return;
  el.classList.add("active");
  setTimeout(() => el.classList.remove("active"), 150);
}

document.addEventListener("keydown", e => {
  const action = keyMap[e.key];
  if (!action) return;
  e.preventDefault();
  flash(action);
  fetch("/tilt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
});

function updateState() {
  fetch("/state").then(r => r.json()).then(d => {
    document.getElementById("wx").textContent = (d.winkel_x >= 0 ? "+" : "") + d.winkel_x.toFixed(1) + "°";
    document.getElementById("wy").textContent = (d.winkel_y >= 0 ? "+" : "") + d.winkel_y.toFixed(1) + "°";
    const ballEl = document.getElementById("ball-state");
    ballEl.textContent = d.valid ? "Gefunden" : "--";
    ballEl.className = d.valid ? "value valid" : "value";

    const dot = document.getElementById("dot");
    const px = 50 + (d.winkel_x / MAX) * 38;
    const py = 50 + (d.winkel_y / MAX) * 38;
    dot.setAttribute("cx", px.toFixed(1));
    dot.setAttribute("cy", py.toFixed(1));
  }).catch(() => {});
}
setInterval(updateState, 120);
</script>
</body>
</html>
"""

# Main

def main() -> None:
    global _running, _platform

    parser = argparse.ArgumentParser(description="Balancing Robot Live-Demo")
    parser.add_argument("--mock", action="store_true",
                        help="MockServoDriver verwenden (kein Pi nötig)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Ohne Kamera starten (nur Kinematik-Demo)")
    parser.add_argument("--port", type=int, default=STREAM_PORT)
    args = parser.parse_args()

    if args.mock:
        from hardware.mock_driver import MockServoDriver
        driver = MockServoDriver(verbose=True)
    else:
        from hardware.pca9685_driver import PCA9685Driver
        driver = PCA9685Driver.from_config("config.json")

    _platform = Platform(driver=driver)
    _platform.neutral()

    detector = None
    if not args.no_camera:
        detector = BallDetectorRuntime(
            hsv_range=HSV_RANGE,
            frame_size=FRAME_SIZE,
            input_color_order="BGR",
            use_roi_tracking=True,
        )
        detector.start()
        while detector.get_latest_frame() is None:
            time.sleep(0.05)
        Thread(target=_producer_loop, args=(detector,), daemon=True).start()

    print(f"Demo läuft auf http://{STREAM_HOST}:{args.port}")
    print("  --mock        : Servos simulieren")
    print("  --no-camera   : Ohne Kamera (nur Kinematik)")

    try:
        app.run(host=STREAM_HOST, port=args.port, threaded=True)
    finally:
        _running = False
        if detector:
            detector.stop()
        _platform.close()

if __name__ == "__main__":
    main()

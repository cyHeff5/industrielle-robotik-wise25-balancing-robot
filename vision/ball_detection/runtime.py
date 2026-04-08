from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .detector import detect_ball
from .types import DetectionResult, HsvRange
from .utils import pixel_to_center_coords


@dataclass(frozen=True)
class BallMeasurement:
    timestamp_s: float
    valid: bool
    x_rel_px: int
    y_rel_px: int
    radius_px: float
    area_px: float


class BallDetectorRuntime:
    # Hintergrund-Thread holt Frames und erkennt Ball direkt -- get_last_measurement() nie blockierend

    def __init__(
        self,
        hsv_range: HsvRange,
        frame_size: tuple[int, int] = (640, 480),
        input_color_order: str = "BGR",
        use_roi_tracking: bool = False,
        roi_size: int = 600,
        roi_miss_limit: int = 5,
        downsample_factor: int = 1,
    ) -> None:
        self.hsv_range = hsv_range
        self.frame_size = frame_size
        self.input_color_order = input_color_order.upper()
        if self.input_color_order not in {"RGB", "BGR"}:
            raise ValueError("input_color_order must be 'RGB' or 'BGR'")
        self.use_roi_tracking = bool(use_roi_tracking)
        self.roi_size = int(roi_size)
        self.roi_miss_limit = int(roi_miss_limit)
        self.downsample_factor = int(downsample_factor)
        if self.roi_size <= 0:
            raise ValueError("roi_size must be > 0")
        if self.roi_miss_limit < 1:
            raise ValueError("roi_miss_limit must be >= 1")
        if self.downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1")

        self._picam2 = None
        self._width = int(frame_size[0])
        self._height = int(frame_size[1])
        self._roi: tuple[int, int, int, int] | None = None
        self._roi_misses = 0

        self._frame_bgr: np.ndarray | None = None
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._last_measurement: BallMeasurement | None = None
        self._measurement_lock = threading.Lock()

    def start(self) -> None:
        from picamera2 import Picamera2

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": self.frame_size, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        self._picam2 = picam2

        self._stop_event.clear()
        threading.Thread(target=self._grab_and_detect_loop, daemon=True).start()

    def _grab_and_detect_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self._picam2.capture_array()
            ts = perf_counter()

            if self.input_color_order == "RGB":
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            with self._frame_lock:
                self._frame_bgr = frame_bgr

            result = self._detect_with_optional_roi(frame_bgr)

            if not result.valid:
                measurement = BallMeasurement(
                    timestamp_s=ts, valid=False,
                    x_rel_px=0, y_rel_px=0, radius_px=0.0, area_px=0.0,
                )
            else:
                x_rel, y_rel = pixel_to_center_coords(
                    result.x_px, result.y_px, self._width, self._height
                )
                measurement = BallMeasurement(
                    timestamp_s=ts, valid=True,
                    x_rel_px=int(x_rel), y_rel_px=int(y_rel),
                    radius_px=float(result.radius_px), area_px=float(result.area_px),
                )

            with self._measurement_lock:
                self._last_measurement = measurement

    def get_latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._frame_bgr.copy() if self._frame_bgr is not None else None

    def get_last_measurement(self) -> BallMeasurement | None:
        with self._measurement_lock:
            return self._last_measurement

    def read_with_frame(self) -> tuple[BallMeasurement, np.ndarray]:
        """Gibt letztes Messergebnis + Frame zurueck. Fuer demo.py und measure_latency."""
        frame_bgr = self.get_latest_frame()
        if frame_bgr is None:
            raise RuntimeError("Noch kein Frame verfuegbar. start() aufgerufen?")
        measurement = self.get_last_measurement()
        if measurement is None:
            measurement = BallMeasurement(
                timestamp_s=perf_counter(), valid=False,
                x_rel_px=0, y_rel_px=0, radius_px=0.0, area_px=0.0,
            )
        return measurement, frame_bgr

    def stop(self) -> None:
        self._stop_event.set()
        if self._picam2 is None:
            return
        self._picam2.stop()
        self._picam2 = None
        self._roi = None
        self._roi_misses = 0

    def debug_state(self) -> dict:
        return {
            "use_roi_tracking": self.use_roi_tracking,
            "roi": self._roi,
            "roi_misses": self._roi_misses,
            "downsample_factor": self.downsample_factor,
        }

    def _make_roi(self, x_px: int, y_px: int) -> tuple[int, int, int, int]:
        half = self.roi_size // 2
        return (
            max(0, int(x_px) - half),
            max(0, int(y_px) - half),
            min(self._width, int(x_px) + half),
            min(self._height, int(y_px) + half),
        )

    def _detect_with_optional_roi(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self.use_roi_tracking and self._roi is not None:
            x0, y0, x1, y1 = self._roi
            roi_result = self._detect_frame(frame_bgr[y0:y1, x0:x1], x_offset=x0, y_offset=y0)
            if roi_result.valid:
                self._roi = self._make_roi(roi_result.x_px, roi_result.y_px)
                self._roi_misses = 0
                return roi_result
            self._roi_misses += 1
            if self._roi_misses >= self.roi_miss_limit:
                self._roi = None
                self._roi_misses = 0

        full_result = self._detect_frame(frame_bgr, x_offset=0, y_offset=0)
        if self.use_roi_tracking and full_result.valid:
            self._roi = self._make_roi(full_result.x_px, full_result.y_px)
            self._roi_misses = 0
        return full_result

    def _detect_frame(self, frame_bgr: np.ndarray, x_offset: int, y_offset: int) -> DetectionResult:
        if self.downsample_factor == 1:
            local = detect_ball(frame_bgr, self.hsv_range)
            if not local.valid:
                return local
            return DetectionResult(
                valid=True,
                x_px=int(x_offset + local.x_px),
                y_px=int(y_offset + local.y_px),
                radius_px=float(local.radius_px),
                area_px=float(local.area_px),
            )

        f = self.downsample_factor
        h, w = frame_bgr.shape[:2]
        w_small, h_small = max(1, w // f), max(1, h // f)
        small = cv2.resize(frame_bgr, (w_small, h_small), interpolation=cv2.INTER_AREA)
        local_small = detect_ball(small, self.hsv_range)
        if not local_small.valid:
            return local_small

        sx, sy = w / float(w_small), h / float(h_small)
        x_local = min(max(int(round(local_small.x_px * sx)), 0), w - 1)
        y_local = min(max(int(round(local_small.y_px * sy)), 0), h - 1)
        return DetectionResult(
            valid=True,
            x_px=int(x_offset + x_local),
            y_px=int(y_offset + y_local),
            radius_px=float(local_small.radius_px * (sx + sy) * 0.5),
            area_px=float(local_small.area_px * sx * sy),
        )

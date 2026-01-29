import cv2
import numpy as np

from .types import DetectionResult, HsvRange
from .utils import mask_from_hsv, largest_blob


MIN_AREA = 500
MIN_RADIUS = 10


def detect_ball(frame_bgr: np.ndarray, hsv_range: HsvRange) -> DetectionResult:
    mask = mask_from_hsv(frame_bgr, hsv_range.lower, hsv_range.upper)
    cnt, area = largest_blob(mask)

    if cnt is None or area < MIN_AREA:
        return DetectionResult(False, 0, 0, 0.0, 0.0)

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    if radius < MIN_RADIUS:
        return DetectionResult(False, 0, 0, 0.0, area)

    return DetectionResult(True, int(x), int(y), float(radius), float(area))

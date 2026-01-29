import numpy as np
import cv2

from .types import HsvRange


def calibrate_hsv_from_center_roi(
    frame_bgr: np.ndarray,
    roi_size: int = 120,
    s_min: int = 40,
    v_min: int = 40,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    pad_h: int = 8,
    pad_s: int = 20,
    pad_v: int = 20,
) -> HsvRange:
    h, w = frame_bgr.shape[:2]
    half = roi_size // 2
    cx, cy = w // 2, h // 2

    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, w)
    y1 = min(cy + half, h)

    roi = frame_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Filter out low-sat/low-val pixels (background)
    mask = (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)
    if not np.any(mask):
        # Fallback: use full ROI
        sample = hsv.reshape(-1, 3)
    else:
        sample = hsv[mask]

    h_vals = sample[:, 0]
    s_vals = sample[:, 1]
    v_vals = sample[:, 2]

    h_lo = np.percentile(h_vals, low_pct)
    h_hi = np.percentile(h_vals, high_pct)
    s_lo = np.percentile(s_vals, low_pct)
    s_hi = np.percentile(s_vals, high_pct)
    v_lo = np.percentile(v_vals, low_pct)
    v_hi = np.percentile(v_vals, high_pct)

    # Range etwas erweitern, damit leichte Lichtaenderungen nicht sofort ausfiltern
    h_lo = max(h_lo - pad_h, 0)
    h_hi = min(h_hi + pad_h, 179)
    s_lo = max(s_lo - pad_s, 0)
    s_hi = min(s_hi + pad_s, 255)
    v_lo = max(v_lo - pad_v, 0)
    v_hi = min(v_hi + pad_v, 255)

    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

    return HsvRange(lower=lower, upper=upper)

import cv2
import numpy as np


def mask_from_hsv(frame_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Rauschen reduzieren
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def largest_blob(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_cnt = cnt
    return best_cnt, best_area


def draw_coordinate_system(frame: np.ndarray, grid_step: int = 100) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    axis_color = (200, 200, 200)
    grid_color = (80, 80, 80)

    cv2.line(frame, (0, cy), (w - 1, cy), axis_color, 1)
    cv2.line(frame, (cx, 0), (cx, h - 1), axis_color, 1)

    if grid_step and grid_step > 0:
        for x in range(cx % grid_step, w, grid_step):
            cv2.line(frame, (x, 0), (x, h - 1), grid_color, 1)
        for y in range(cy % grid_step, h, grid_step):
            cv2.line(frame, (0, y), (w - 1, y), grid_color, 1)

    cv2.putText(frame, "+x", (w - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
    cv2.putText(frame, "+y", (cx + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)
    cv2.putText(frame, "(0,0)", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1)


def pixel_to_center_coords(px: int, py: int, w: int, h: int) -> tuple[int, int]:
    cx, cy = w // 2, h // 2
    x = px - cx
    y = cy - py
    return x, y

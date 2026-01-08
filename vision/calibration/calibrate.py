import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class ChessboardConfig:
    inner_corners: tuple[int, int]  # (cols, rows) innere Ecken
    square_size_mm: float

@dataclass(frozen=True)
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    mean_reprojection_error: float

def _prepare_object_points(cfg: ChessboardConfig) -> np.ndarray:
    cols, rows = cfg.inner_corners
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= cfg.square_size_mm
    return objp

def calibrate_from_images(
    images_dir: Path,
    pattern: ChessboardConfig,
    out_debug_dir: Path | None = None
) -> CalibrationResult:
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    objp = _prepare_object_points(pattern)
    objpoints, imgpoints = [], []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    image_size = None
    used = 0

    if out_debug_dir:
        (out_debug_dir / "corners").mkdir(parents=True, exist_ok=True)

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # robuster: wenn verfügbar, SB nutzen
        found = False
        corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            sb_flags = (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            found, corners = cv2.findChessboardCornersSB(gray, pattern.inner_corners, sb_flags)
        else:
            found, corners = cv2.findChessboardCorners(gray, pattern.inner_corners, flags)

        if not found or corners is None:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners2)
        used += 1

        if out_debug_dir:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern.inner_corners, corners2, True)
            cv2.imwrite(str(out_debug_dir / "corners" / p.name), vis)

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Reprojection error
    total_err = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        total_err += err**2
        total_points += len(proj)
    mean_err = float(np.sqrt(total_err / total_points))

    return CalibrationResult(camera_matrix=mtx, dist_coeffs=dist, rms=float(rms), mean_reprojection_error=mean_err)

def save_undistortion_examples(
    images_dir: Path,
    out_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    max_examples: int = 5
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:max_examples]

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undist = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_mtx)
        both = cv2.hconcat([img, undist])
        cv2.imwrite(str(out_dir / f"undist_{p.name}"), both)

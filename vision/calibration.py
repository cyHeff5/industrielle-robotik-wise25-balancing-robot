import cv2
import numpy as np 
import glob
import os

CHECKERBOARD = (9,6)
SQUARE_SIZE = 25.0
INPUT_DIR = "Calibration Input"
OUTPUT_DIR = "Calibration Output"

os.makedirs(f"{OUTPUT_DIR}/01_corners", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/02_subpixel", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/03_reprojection", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/04_undistortion", exist_ok=True)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(f"{INPUT_DIR}/calib_*.jpg")

print(f"{len(images)} Kalibrierungsbilder gefunden.")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if not ret:
        print(f"Ecken nicht gefunden in {fname}")
        continue

    # Für Kalibrierung speichern
    objpoints.append(objp)

    # Subpixel-Refinement
    corners_sub = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    imgpoints.append(corners_sub)

    # Bild 1: gefundene Ecken
    img_corners = img.copy()
    cv2.drawChessboardCorners(img_corners, CHECKERBOARD, corners, ret)

    out1 = f"{OUTPUT_DIR}/01_corners/{os.path.basename(fname)}"
    cv2.imwrite(out1, img_corners)

    # Bild 2: Subpixel-Vergleich
    img_sub = img.copy()
    for i in range(len(corners)):
        x1, y1 = corners[i][0]
        x2, y2 = corners_sub[i][0]
        cv2.circle(img_sub, (int(x1), int(y1)), 3, (0,0,255), -1)  # rot
        cv2.circle(img_sub, (int(x2), int(y2)), 3, (0,255,0), -1)  # grün

    out2 = f"{OUTPUT_DIR}/02_subpixel/{os.path.basename(fname)}"
    cv2.imwrite(out2, img_sub)
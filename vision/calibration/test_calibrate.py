import argparse
from pathlib import Path

from calibrate import ChessboardConfig, calibrate_from_images, save_undistortion_examples

# Optional (nur falls vorhanden)
try:
    from io_utils import save_calibration_npz, save_calibration_yaml
    HAVE_IO_UTILS = True
except ImportError:
    HAVE_IO_UTILS = False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test für calibrate.py: Intrinsics aus Bildern berechnen + Debug/Undistortion speichern."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="calib_session/images",
        help="Ordner mit Kalibrierungsbildern (.jpg/.png). Default: calib_session/images",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="calib_session/test_output",
        help="Output-Ordner für Debug, Undistortion und Ergebnisdateien.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=7,
        help="Anzahl innerer Ecken (Spalten). Beispiel 7 bei 7x7.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=7,
        help="Anzahl innerer Ecken (Zeilen). Beispiel 7 bei 7x7.",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="Kantenlänge eines Schachbrett-Quadrats in mm (nur für Skalierung).",
    )
    parser.add_argument(
        "--max-undist",
        type=int,
        default=5,
        help="Wie viele Undistortion-Beispielbilder gespeichert werden sollen.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    debug_dir = out_dir / "debug"
    result_dir = out_dir / "result"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

    chess = ChessboardConfig(inner_corners=(args.cols, args.rows), square_size_mm=args.square_mm)

    print("[TEST] Images dir:", images_dir.resolve())
    print("[TEST] Output dir:", out_dir.resolve())
    print("[TEST] Pattern (inner corners):", chess.inner_corners)
    print("[TEST] Square size (mm):", chess.square_size_mm)

    # 1) Kalibrieren + Debug-Corner-Overlays speichern
    print("\n[TEST] Running calibrate_from_images(...)")
    result = calibrate_from_images(images_dir, chess, out_debug_dir=debug_dir)

    print("\n=== Calibration Result ===")
    print("RMS:", result.rms)
    print("Mean reprojection error (px):", result.mean_reprojection_error)
    print("Camera matrix:\n", result.camera_matrix)
    print("Dist coeffs:\n", result.dist_coeffs)

    # 2) Undistortion-Beispiele speichern (Original | Undistorted)
    print("\n[TEST] Saving undistortion examples...")
    save_undistortion_examples(
        images_dir=images_dir,
        out_dir=result_dir / "undist_examples",
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        max_examples=args.max_undist,
    )

    # 3) Intrinsics speichern (optional)
    result_dir.mkdir(parents=True, exist_ok=True)
    if HAVE_IO_UTILS:
        print("[TEST] Saving calibration files (npz + yaml)...")
        save_calibration_npz(result_dir / "camera_intrinsics.npz", result.camera_matrix, result.dist_coeffs)
        save_calibration_yaml(result_dir / "camera_intrinsics.yaml", result.camera_matrix, result.dist_coeffs)
    else:
        print("[TEST] io_utils.py nicht gefunden → überspringe .npz/.yaml Export.")

    print("\n[OK] Fertig.")
    print("Debug corner overlays:", (debug_dir / "corners").resolve())
    print("Undist examples:", (result_dir / "undist_examples").resolve())


if __name__ == "__main__":
    main()

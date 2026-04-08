import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

import numpy as np

from calibrate import (
    ChessboardConfig,
    calibrate_from_images,
    save_undistortion_examples,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    # Settings (an Schachbrett anpassen)
    chess = ChessboardConfig(inner_corners=(7, 7), square_size_mm=25.0)

    output_dir.mkdir(parents=True, exist_ok=True)

    result = calibrate_from_images(
        images_dir=input_dir,
        pattern=chess,
        out_debug_dir=output_dir / "debug",
    )

    # Intrinsics speichern
    np.savez(
        output_dir / "intrinsics.npz",
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
    )

    meta = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "chessboard": asdict(chess),
        "rms": result.rms,
        "mean_reprojection_error": result.mean_reprojection_error,
        "input_dir": str(input_dir),
    }
    (output_dir / "intrinsics_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Optional: Undistortion-Beispiele
    save_undistortion_examples(
        images_dir=input_dir,
        out_dir=output_dir / "undistortion",
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        max_examples=5,
    )



if __name__ == "__main__":
    main()

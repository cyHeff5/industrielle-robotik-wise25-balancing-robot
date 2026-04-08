import json
import time
from dataclasses import dataclass
from pathlib import Path

from capture import CameraCapture, CaptureConfig, prepare_input_dir

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from balancing_platform import Platform
from hardware import PCA9685Driver


@dataclass(frozen=True)
class Position:
    name: str
    roll_deg: float
    pitch_deg: float
    settle_s: float = 1.0


def _load_positions(path: Path) -> list[Position]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "positions" not in data or not isinstance(data["positions"], list):
        raise ValueError("JSON must contain a list under 'positions'.")
    positions = [
        Position(
            name=str(item.get("name", "pos")),
            roll_deg=float(item["roll_deg"]),
            pitch_deg=float(item["pitch_deg"]),
            settle_s=float(item.get("settle_s", 1.0)),
        )
        for item in data["positions"]
    ]
    if not positions:
        raise ValueError("Positions list is empty.")
    return positions


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    image_prefix = "calib"
    resolution = (1280, 720)

    prepare_input_dir(input_dir, purge=True)
    positions = _load_positions(base_dir / "positions.json")

    cam = CameraCapture(CaptureConfig(size=resolution, warmup_s=2.0))
    cam.start()

    config_path = Path(__file__).resolve().parents[2] / "config.json"
    with Platform(PCA9685Driver.from_config(config_path)) as platform:
        platform.neutral()
        time.sleep(1.0)

        try:
            for idx, pos in enumerate(positions, start=1):
                platform.set_tilt(winkel_x=pos.pitch_deg, winkel_y=pos.roll_deg)
                time.sleep(pos.settle_s)

                filename = input_dir / f"{image_prefix}_{idx:02d}_{pos.name}.jpg"
                cam.capture_to_file(filename)
        finally:
            cam.stop()


if __name__ == "__main__":
    main()

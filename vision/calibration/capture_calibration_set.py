import json
import time
from dataclasses import dataclass
from pathlib import Path

from capture import CameraCapture, CaptureConfig, prepare_input_dir


@dataclass(frozen=True)
class Position:
    name: str
    roll_deg: float
    pitch_deg: float
    settle_s: float = 1.0


def _load_positions(path: Path) -> list[Position]:
    # Positionsliste fuer die Kalibrierungsfahrt laden
    if not path.exists():
        raise FileNotFoundError(
            f"Positions JSON not found: {path}\n"
            "Expected format:\n"
            "{\n"
            "  \"positions\": [\n"
            "    {\"name\": \"center\", \"roll_deg\": 0.0, \"pitch_deg\": 0.0, \"settle_s\": 1.0}\n"
            "  ]\n"
            "}\n"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if "positions" not in data or not isinstance(data["positions"], list):
        raise ValueError("JSON must contain a list under 'positions'.")

    positions: list[Position] = []
    for item in data["positions"]:
        positions.append(
            Position(
                name=str(item.get("name", "pos")),
                roll_deg=float(item["roll_deg"]),
                pitch_deg=float(item["pitch_deg"]),
                settle_s=float(item.get("settle_s", 1.0)),
            )
        )
    if not positions:
        raise ValueError("Positions list is empty.")
    return positions


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "input"
    positions_json = base_dir / "positions.json"

    # Settings
    # Aufnahmeparameter einstellen
    image_prefix = "calib"
    resolution = (1280, 720)

    # Prepare folder (Alte Bilder löschen)
    prepare_input_dir(input_dir, purge=True)

    positions = _load_positions(positions_json)

    cam = CameraCapture(CaptureConfig(size=resolution, warmup_s=2.0))
    cam.start()

    try:
        for idx, pos in enumerate(positions, start=1):
            # TODO: Platte auf die gewuenschte Lage fahren (Servo-Logik fehlt noch)
            print(f"[TODO] Move plate to roll={pos.roll_deg} pitch={pos.pitch_deg}")

            time.sleep(pos.settle_s)

            filename = input_dir / f"{image_prefix}_{idx:02d}_{pos.name}.jpg"
            cam.capture_to_file(filename)
            print(f"Saved: {filename}")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()

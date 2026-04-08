import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from picamera2 import Picamera2

@dataclass(frozen=True)
class CaptureConfig:
    # Kamera-Settings
    size: tuple[int, int] = (720, 720)
    warmup_s: float = 2.0

class CameraCapture:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": self.cfg.size, "format": "RGB888"}
        )
        self.picam2.configure(config)

    def start(self):
        self.picam2.start()
        time.sleep(self.cfg.warmup_s)

    def stop(self):
        self.picam2.stop()

    def capture_to_file(self, out_path: Path) -> None:
        frame = self.picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)


def prepare_input_dir(out_dir: Path, purge: bool = True) -> None:
    # Vor jeder Kalibrierung den Input-Ordner aufraeumen
    out_dir.mkdir(parents=True, exist_ok=True)
    if not purge:
        return
    for p in out_dir.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)

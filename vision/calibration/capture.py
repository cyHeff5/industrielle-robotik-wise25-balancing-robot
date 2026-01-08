from picamera2 import Picamera2
import cv2
import time
from pathlib import Path 
from dataclasses import dataclass

@dataclass(frozen=True)
class CaptureConfig:
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
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
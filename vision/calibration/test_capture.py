from pathlib import Path
from capture import CameraCapture, CaptureConfig

def main():
    print("[TEST] Starte Kamera-Capture-Test...")

    # Ausgabeordner
    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)

    # Kamera-Konfiguration
    cfg = CaptureConfig(
        size=(1280, 720),
        warmup_s=2.0
    )

    # Kamera-Wrapper initialisieren
    cam = CameraCapture(cfg)

    print("[TEST] Kamera starten...")
    cam.start()

    try:
        out_file = out_dir / "capture_test.jpg"
        print(f"[TEST] Nehme Bild auf → {out_file}")
        cam.capture_to_file(out_file)
        print("[TEST] Bild gespeichert.")
    finally:
        print("[TEST] Kamera stoppen...")
        cam.stop()

    print("[TEST] Fertig. Prüfe die Datei:")
    print(f"       {out_file.resolve()}")

if __name__ == "__main__":
    main()

from picamera2 import Picamera2
import cv2
import time
import os

# -----------------------------
# EINSTELLUNGEN
# -----------------------------
OUTPUT_DIR = "Calibration_Input"
IMAGE_PREFIX = "calib"
NUM_IMAGES = 20          # wie viele Bilder aufnehmen?
DELAY_BETWEEN = 1.0      # Sekunden zwischen Aufnahmen
RESOLUTION = (1280, 720) # höhere Auflösung = bessere Kalibrierung

# -----------------------------
# Ordner anlegen
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Kamera initialisieren
# -----------------------------
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": RESOLUTION, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # Kamera „warm laufen“ lassen

print("\n=== Kamera-Kalibrierungsaufnahme ===")
print(f"Es werden {NUM_IMAGES} Bilder aufgenommen.")
print("Halte das Schachbrett in verschiedenen Positionen:")
print("- nah / fern")
print("- links / rechts")
print("- oben / unten")
print("- schräg kippen\n")

print("Starte in 3 Sekunden...")
time.sleep(3)

# -----------------------------
# Bilder aufnehmen
# -----------------------------
for i in range(1, NUM_IMAGES + 1):
    frame = picam2.capture_array()

    filename = f"{OUTPUT_DIR}/{IMAGE_PREFIX}_{i:02d}.jpg"
    cv2.imwrite(filename, frame)

    print(f"[{i:02d}/{NUM_IMAGES}] gespeichert: {filename}")

    time.sleep(DELAY_BETWEEN)

picam2.stop()

print("\nFertig!")
print(f"Alle Bilder liegen im Ordner: ./{OUTPUT_DIR}/")

import time

from balancing_platform import Platform
from hardware.pca9685_driver import PCA9685Driver, ServoConfig

# Servo-Kalibrierung (Offsets und Modifier pro Servo)
config = ServoConfig(
    offset_v=20,
    offset_l=35,
    offset_r=20,
    modifier=1.0,
)

driver = PCA9685Driver(config)

with Platform(driver=driver) as platform:

    platform.neutral()  # Servos gerade stellen

    # Testpositionen: (winkel_x, winkel_y, hoehe)
    poses = [
        (  0,   0),
        (-15,   0),
        (+10,   0),
        (  0,   0),
        (  0, -10),
        (  0, +10),
        ( 14, -10),
        (  0,   0),
        ( 45,   0),
    ]

    for winkel_x, winkel_y in poses:
        platform.set_tilt(winkel_x, winkel_y)
        time.sleep(2)

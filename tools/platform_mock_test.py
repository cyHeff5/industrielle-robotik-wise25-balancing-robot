"""
Laptop-Test fuer die Platform-API mit MockServoDriver.
Kein Raspberry Pi, keine Hardware noetig.

Ausfuehren aus dem Projektroot:
    python -m tools.platform_mock_test
"""

from balancing_platform import Platform
from hardware import MockServoDriver
from kinematik.geometry import PlatformGeometry


def main() -> None:
    driver = MockServoDriver(verbose=True)

    with Platform(driver, PlatformGeometry()) as platform:
        print("\n--- Neutral ---")
        platform.neutral()

        print("\n--- Kippe X+5 Grad ---")
        platform.set_tilt(winkel_x=5.0, winkel_y=0.0)

        print("\n--- Kippe X-5 Grad ---")
        platform.set_tilt(winkel_x=-5.0, winkel_y=0.0)

        print("\n--- Kippe Y+5 Grad ---")
        platform.set_tilt(winkel_x=0.0, winkel_y=5.0)

        print("\n--- Kippe X+10 Y+10 Grad ---")
        platform.set_tilt(winkel_x=10.0, winkel_y=10.0)

        print("\n--- Zurueck Neutral ---")
        platform.neutral()

    print("\nFertig.")


if __name__ == "__main__":
    main()

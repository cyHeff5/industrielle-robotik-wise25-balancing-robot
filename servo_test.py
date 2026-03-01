import time

from gpiozero import AngularServo


SERVO_PINS = {
    "v": 27,
    "l": 17,
    "r": 22,
}

# Slow synchronized movement settings
LOW_DEG = 70.0
HIGH_DEG = 110.0
STEP_DEG = 1.0
STEP_DELAY_S = 0.05
PAUSE_AT_END_S = 0.5


def set_all(servos: dict[str, AngularServo], angle: float) -> None:
    for servo in servos.values():
        servo.angle = angle


def main() -> None:
    servos = {
        name: AngularServo(
            pin,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
        )
        for name, pin in SERVO_PINS.items()
    }

    print("Testing all servos together (synchronized up/down).")
    print("Press Ctrl+C to stop.")

    try:
        angle = LOW_DEG
        direction = +1.0
        set_all(servos, angle)

        while True:
            angle += direction * STEP_DEG

            if angle >= HIGH_DEG:
                angle = HIGH_DEG
                direction = -1.0
                set_all(servos, angle)
                time.sleep(PAUSE_AT_END_S)
                continue

            if angle <= LOW_DEG:
                angle = LOW_DEG
                direction = +1.0
                set_all(servos, angle)
                time.sleep(PAUSE_AT_END_S)
                continue

            set_all(servos, angle)
            time.sleep(STEP_DELAY_S)
    except KeyboardInterrupt:
        print("Stopping servo test.")
    finally:
        set_all(servos, (LOW_DEG + HIGH_DEG) / 2.0)
        time.sleep(0.2)


if __name__ == "__main__":
    main()


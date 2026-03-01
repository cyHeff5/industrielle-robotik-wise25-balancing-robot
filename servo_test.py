import time

from gpiozero import AngularServo


# Select exactly one servo to test: "v", "l", or "r"
SERVO_NAME = "v"

SERVO_PINS = {
    "v": 27,
    "l": 17,
    "r": 22,
}

# Slow movement settings
LOW_DEG = 70.0
HIGH_DEG = 110.0
STEP_DEG = 1.0
STEP_DELAY_S = 0.05
PAUSE_AT_END_S = 0.5


def main() -> None:
    if SERVO_NAME not in SERVO_PINS:
        raise ValueError("SERVO_NAME must be one of: 'v', 'l', 'r'")

    pin = SERVO_PINS[SERVO_NAME]
    servo = AngularServo(
        pin,
        min_angle=0,
        max_angle=180,
        min_pulse_width=0.0005,
        max_pulse_width=0.0025,
    )

    print(f"Testing servo '{SERVO_NAME.upper()}' on GPIO {pin}")
    print("Press Ctrl+C to stop.")

    try:
        angle = LOW_DEG
        direction = +1.0
        servo.angle = angle

        while True:
            angle += direction * STEP_DEG

            if angle >= HIGH_DEG:
                angle = HIGH_DEG
                direction = -1.0
                servo.angle = angle
                time.sleep(PAUSE_AT_END_S)
                continue

            if angle <= LOW_DEG:
                angle = LOW_DEG
                direction = +1.0
                servo.angle = angle
                time.sleep(PAUSE_AT_END_S)
                continue

            servo.angle = angle
            time.sleep(STEP_DELAY_S)
    except KeyboardInterrupt:
        print("Stopping servo test.")
    finally:
        servo.angle = (LOW_DEG + HIGH_DEG) / 2.0
        time.sleep(0.2)


if __name__ == "__main__":
    main()


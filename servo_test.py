import time

from gpiozero import AngularServo


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
SERVO_SWITCH_DELAY_S = 1.0


def release_servo(servo: AngularServo) -> None:
    # Disable PWM output so servo no longer holds position.
    servo.angle = None


def release_all(servos: dict[str, AngularServo]) -> None:
    for servo in servos.values():
        release_servo(servo)


def run_single_servo_cycle(servos: dict[str, AngularServo], active_name: str) -> None:
    active = servos[active_name]

    # Ensure only one servo is powered at a time.
    for name, servo in servos.items():
        if name != active_name:
            release_servo(servo)

    angle = LOW_DEG
    direction = +1.0
    active.angle = angle

    while True:
        angle += direction * STEP_DEG

        if angle >= HIGH_DEG:
            angle = HIGH_DEG
            direction = -1.0
            active.angle = angle
            time.sleep(PAUSE_AT_END_S)
            continue

        if angle <= LOW_DEG and direction < 0:
            angle = LOW_DEG
            active.angle = angle
            time.sleep(PAUSE_AT_END_S)
            break

        active.angle = angle
        time.sleep(STEP_DELAY_S)

    # Release active servo after its motion cycle as well.
    release_servo(active)


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

    print("Sequential servo test (only active servo powered).")
    print("Inactive servos are released (no hold torque). Press Ctrl+C to stop.")

    try:
        release_all(servos)
        while True:
            for name in ("v", "l", "r"):
                print(f"Testing servo '{name.upper()}'")
                run_single_servo_cycle(servos, name)
                time.sleep(SERVO_SWITCH_DELAY_S)
    except KeyboardInterrupt:
        print("Stopping servo test.")
    finally:
        release_all(servos)
        time.sleep(0.2)


if __name__ == "__main__":
    main()


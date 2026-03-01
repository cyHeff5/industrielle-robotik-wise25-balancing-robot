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
SERVO_SWITCH_DELAY_S = 2.0


def set_all(servos: dict[str, AngularServo], angle: float) -> None:
    for servo in servos.values():
        servo.angle = angle


def run_single_servo_cycle(servos: dict[str, AngularServo], active_name: str) -> None:
    # Keep all other servos at neutral while one servo performs a full up/down cycle.
    neutral = (LOW_DEG + HIGH_DEG) / 2.0
    set_all(servos, neutral)
    servo = servos[active_name]

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

        if angle <= LOW_DEG and direction < 0:
            angle = LOW_DEG
            servo.angle = angle
            time.sleep(PAUSE_AT_END_S)
            break

        servo.angle = angle
        time.sleep(STEP_DELAY_S)


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

    print("Testing servos sequentially (V -> L -> R).")
    print(f"Delay between servos: {SERVO_SWITCH_DELAY_S:.1f}s")
    print("Press Ctrl+C to stop.")

    try:
        set_all(servos, (LOW_DEG + HIGH_DEG) / 2.0)

        while True:
            for name in ("v", "l", "r"):
                print(f"Testing servo '{name.upper()}'")
                run_single_servo_cycle(servos, name)
                time.sleep(SERVO_SWITCH_DELAY_S)
    except KeyboardInterrupt:
        print("Stopping servo test.")
    finally:
        set_all(servos, (LOW_DEG + HIGH_DEG) / 2.0)
        time.sleep(0.2)


if __name__ == "__main__":
    main()

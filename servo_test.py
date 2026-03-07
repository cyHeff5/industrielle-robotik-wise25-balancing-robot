import time
import math

from gpiozero import AngularServo


SERVO_PINS = {
    "v": 27,
    "l": 17,
    "r": 22,
}

# Motion settings (smooth sinusoidal motion)
LOW_DEG = 70.0
HIGH_DEG = 110.0
MOTION_PERIOD_S = 3.0
MOTION_DT_S = 0.02
PHASE_CYCLES = 2
PHASE_DELAY_S = 1.0

# Test sequence: first 1 servo, then 2, then 3
TEST_PHASES = [
    ("one", ("v",)),
    ("two", ("v", "l")),
    ("three", ("v", "l", "r")),
]


def release_servo(servo: AngularServo) -> None:
    # Disable PWM output so servo does not hold position.
    servo.angle = None


def release_all(servos: dict[str, AngularServo]) -> None:
    for servo in servos.values():
        release_servo(servo)


def set_active_angle(servos: dict[str, AngularServo], active_names: tuple[str, ...], angle: float) -> None:
    active = set(active_names)
    for name, servo in servos.items():
        if name in active:
            servo.angle = angle
        else:
            release_servo(servo)


def run_phase(servos: dict[str, AngularServo], phase_name: str, active_names: tuple[str, ...]) -> None:
    print(f"Phase '{phase_name}': moving {len(active_names)} servo(s): {', '.join(n.upper() for n in active_names)}")

    center = (LOW_DEG + HIGH_DEG) / 2.0
    amplitude = (HIGH_DEG - LOW_DEG) / 2.0
    omega = (2.0 * math.pi) / MOTION_PERIOD_S
    start = time.perf_counter()
    duration_s = PHASE_CYCLES * MOTION_PERIOD_S

    while True:
        now = time.perf_counter()
        t = now - start
        if t >= duration_s:
            break
        angle = center + amplitude * math.sin(omega * t)
        set_active_angle(servos, active_names, angle)
        time.sleep(MOTION_DT_S)

    # Release everything at phase end.
    release_all(servos)


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

    print("Servo load test: 1 servo -> 2 servos -> 3 servos.")
    print("Inactive servos are released (no hold current). Press Ctrl+C to stop.")

    try:
        release_all(servos)
        while True:
            for phase_name, active_names in TEST_PHASES:
                run_phase(servos, phase_name, active_names)
                time.sleep(PHASE_DELAY_S)
    except KeyboardInterrupt:
        print("Stopping servo load test.")
    finally:
        release_all(servos)
        time.sleep(0.2)


if __name__ == "__main__":
    main()

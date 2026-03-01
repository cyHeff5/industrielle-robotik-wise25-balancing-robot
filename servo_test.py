import time

from gpiozero import AngularServo


# GPIO pins
PIN_V = 27
PIN_L = 17
PIN_R = 22

# Servo mapping from your current setup
MODIFIER_V, OFFSET_V = -1, 176
MODIFIER_L, OFFSET_L = -1, 176
MODIFIER_R, OFFSET_R = -1, 176

# Test behavior
NEUTRAL_LOGIC_DEG = 90
TEST_POINTS_LOGIC_DEG = [80, 90, 100]
HOLD_S = 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def set_servo_logic_angle(servo: AngularServo, logic_deg: float, offset: float, modifier: float) -> None:
    hw_deg = logic_deg * modifier + offset
    servo.angle = _clamp(hw_deg, 0.0, 180.0)


def move_all(v: AngularServo, l: AngularServo, r: AngularServo, logic_deg: float) -> None:
    set_servo_logic_angle(v, logic_deg, OFFSET_V, MODIFIER_V)
    set_servo_logic_angle(l, logic_deg, OFFSET_L, MODIFIER_L)
    set_servo_logic_angle(r, logic_deg, OFFSET_R, MODIFIER_R)


def main() -> None:
    servo_v = AngularServo(PIN_V, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
    servo_l = AngularServo(PIN_L, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)
    servo_r = AngularServo(PIN_R, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)

    print("Starting servo test (no kinematics). Press Ctrl+C to stop.")
    try:
        move_all(servo_v, servo_l, servo_r, NEUTRAL_LOGIC_DEG)
        time.sleep(1.0)

        while True:
            for logic_deg in TEST_POINTS_LOGIC_DEG:
                print(f"Move all servos to logic angle: {logic_deg} deg")
                move_all(servo_v, servo_l, servo_r, logic_deg)
                time.sleep(HOLD_S)
    except KeyboardInterrupt:
        print("Stopping test, moving to neutral.")
    finally:
        move_all(servo_v, servo_l, servo_r, NEUTRAL_LOGIC_DEG)
        time.sleep(0.5)


if __name__ == "__main__":
    main()


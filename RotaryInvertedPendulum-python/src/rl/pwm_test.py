import time
import Jetson.GPIO as GPIO
from JetsonPWM import JetsonPWM

# --- GPIO Pin Mapping (Physical Board Pin Numbers) ---
DIR_PIN = 31      # Board Pin 31 (GPIO 12)
ENABLE_PIN = 29   # Board Pin 29 (GPIO 5)

# --- Motor & Velocity Config ---
STEPS_PER_RAD = 6400
MAX_VEL_RAD = 5.0                               # rad/s
MAX_FREQ_HZ = int(MAX_VEL_RAD * STEPS_PER_RAD)  # 32,000 Hz
MIN_FREQ_HZ = 60                                # Standstill limit (~0.0094 rad/s)

# --- Ramp Config ---
RAMP_TIME_S = 2.0                               # Seconds to reach max speed
LOOP_INTERVAL_S = 0.040                         # 40 ms cadence

total_steps = int(RAMP_TIME_S / LOOP_INTERVAL_S)
freq_step = (MAX_FREQ_HZ - MIN_FREQ_HZ) / total_steps

ramp_up = [int(MIN_FREQ_HZ + i * freq_step) for i in range(total_steps + 1)]
ramp_up[-1] = MAX_FREQ_HZ
ramp_down = list(reversed(ramp_up))

def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT)
    GPIO.setup(ENABLE_PIN, GPIO.OUT)

    # Pull DIR HIGH and ENABLE LOW
    GPIO.output(DIR_PIN, GPIO.HIGH)
    GPIO.output(ENABLE_PIN, GPIO.LOW)
    print(f"GPIO initialized: DIR (Pin {DIR_PIN}) -> HIGH | ENABLE (Pin {ENABLE_PIN}) -> LOW")

def cleanup_gpio():
    # Optional: disable driver output on exit (pull ENABLE HIGH)
    try:
        GPIO.output(ENABLE_PIN, GPIO.HIGH)
    except Exception:
        pass
    GPIO.cleanup()

def main():
    setup_gpio()
    
    # Initialize Pin 32 (PWM0 -> chip 0, channel 0)
    pwm = JetsonPWM(chip=0, channel=0)

    print(f"Starting sweep: {MIN_FREQ_HZ} Hz -> {MAX_FREQ_HZ} Hz (5 rad/s)")
    print(f"Ramp duration: {RAMP_TIME_S}s | Cadence: 40ms ({total_steps} steps)")

    try:
        pwm.start(initial_freq_hz=MIN_FREQ_HZ)
        time.sleep(0.5)

        while True:
            # 1. Accelerate to 32,000 Hz (5 rad/s)
            print("Accelerating...")
            for f in ramp_up:
                t0 = time.perf_counter()
                pwm.change_frequency(f)
                
                dt = time.perf_counter() - t0
                if dt < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - dt)

            # Hold peak velocity
            print("Holding 5 rad/s (32 kHz)...")
            time.sleep(1.0)

            # 2. Decelerate to standstill
            print("Decelerating...")
            for f in ramp_down:
                t0 = time.perf_counter()
                pwm.change_frequency(f)
                
                dt = time.perf_counter() - t0
                if dt < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - dt)

            # Rest at standstill
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping motion...")
    finally:
        pwm.stop()
        cleanup_gpio()
        print("PWM and GPIO shut down cleanly.")

if __name__ == "__main__":
    main()
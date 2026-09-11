import time
from pwm_controller import JetsonPWM

# --- Motor & Velocity Config ---
STEPS_PER_RAD = 6400
MAX_VEL_RAD = 5.0                        # rad/s
MAX_FREQ_HZ = int(MAX_VEL_RAD * STEPS_PER_RAD)  # 32,000 Hz

# Jetson Tegra PWM hardware divisor overflow floor (~48.6 Hz)
MIN_FREQ_HZ = 60                         # ~0.0094 rad/s (almost standstill)

# --- Ramp Config ---
RAMP_TIME_S = 2.0                        # Seconds to reach max speed
LOOP_INTERVAL_S = 0.040                  # Update cadence (40 ms)

# Compute frequency increment per 40 ms tick for linear acceleration
total_steps = int(RAMP_TIME_S / LOOP_INTERVAL_S)
freq_step = (MAX_FREQ_HZ - MIN_FREQ_HZ) / total_steps

# Generate the frequency profile array
ramp_up = [int(MIN_FREQ_HZ + i * freq_step) for i in range(total_steps + 1)]
# Ensure the top target hits exactly 32,000 Hz
ramp_up[-1] = MAX_FREQ_HZ
ramp_down = list(reversed(ramp_up))

def main():
    # Initialize Pin 32 (chip=0, channel=0)
    pwm = JetsonPWM(chip=0, channel=0)
    
    print(f"Starting sweep: {MIN_FREQ_HZ} Hz -> {MAX_FREQ_HZ} Hz (5 rad/s)")
    print(f"Acceleration time: {RAMP_TIME_S}s | Updates every 40ms ({total_steps} slices)")

    try:
        # Start hardware output at low speed
        pwm.start(initial_freq_hz=MIN_FREQ_HZ)
        time.sleep(0.5)  # Brief pause at low speed

        while True:
            # 1. Ramp Up to 32,000 Hz (0 to 5 rad/s)
            print("Accelerating...")
            for f in ramp_up:
                t0 = time.perf_counter()
                pwm.change_frequency(f)
                
                dt = time.perf_counter() - t0
                if dt < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - dt)

            # Hold top speed for 1 second
            print("Holding max speed (5 rad/s)...")
            time.sleep(1.0)

            # 2. Ramp Down to Standstill (5 to 0 rad/s)
            print("Decelerating...")
            for f in ramp_down:
                t0 = time.perf_counter()
                pwm.change_frequency(f)
                
                dt = time.perf_counter() - t0
                if dt < LOOP_INTERVAL_S:
                    time.sleep(LOOP_INTERVAL_S - dt)

            # Hold at low speed before repeating
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping sweep...")
    finally:
        pwm.stop()
        print("PWM output cleanly disabled.")

if __name__ == "__main__":
    main()
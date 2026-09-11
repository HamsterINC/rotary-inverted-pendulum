import time
from pwm_controller import JetsonPWM

# Pin 32 is typically chip=0, channel=0
pwm = JetsonPWM(chip=0, channel=0)

# Start at 200 Hz
pwm.start(initial_freq_hz=200)

target_frequencies = list(range(200, 3000, 50))  # Smooth ramp up

try:
    print("Running speed profile (updates every 40 ms)...")
    while True:
        # Ramp up
        for freq in target_frequencies:
            t0 = time.perf_counter()
            
            pwm.change_frequency(freq)
            
            # Keep loop cadence locked at 40 ms
            dt = time.perf_counter() - t0
            if dt < 0.040:
                time.sleep(0.040 - dt)

        # Ramp down
        for freq in reversed(target_frequencies):
            t0 = time.perf_counter()
            
            pwm.change_frequency(freq)
            
            dt = time.perf_counter() - t0
            if dt < 0.040:
                time.sleep(0.040 - dt)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    pwm.stop()
    print("PWM stopped cleanly.")
import csv
import time
import numpy as np

# Configuration
OUTPUT_FILE = "clamped_pendulum_decay.csv"
SAMPLE_RATE_HZ = 100.0          # 100 Hz = 10 ms interval
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE_HZ
DURATION_SECONDS = 8.0          # Stop recording after 8 seconds

# ---------------------------------------------------------
# HARDWARE HOOK: Replace this with your actual sensor call
# ---------------------------------------------------------
def read_raw_pendulum_radians():
    """
    Query your encoder/sensor and return angle in radians.
    Example for PySerial:
        line = ser.readline().decode().strip()
        return float(line)
    """
    # Placeholder: replace with actual hardware read
    raise NotImplementedError("Connect your encoder/ADC read call here.")


def main():
    print("=" * 60)
    print("CLAMPED PENDULUM LOGGER")
    print("1. Ensure motor arm is rigidly clamped/locked.")
    print("2. Deflect pendulum to ~30-45 deg.")
    print("=" * 60)
    input("Press ENTER to start recording and immediately release the pendulum...")

    data_log = []
    
    # State tracking for unwrap and velocity differentiation
    prev_raw = None
    accumulated_offset = 0.0
    prev_unwrapped = None
    prev_time = None

    t_start = time.perf_counter()
    next_sample_time = t_start

    print(f"Logging for {DURATION_SECONDS} seconds at {SAMPLE_RATE_HZ} Hz...")

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - t_start

            if elapsed >= DURATION_SECONDS:
                break

            # Rate-limiting / timer loop
            if now >= next_sample_time:
                # 1. Read raw angle
                raw_rad = read_raw_pendulum_radians()

                # 2. Phase unwrapping (handles rollover if sensor wraps at +/- pi)
                if prev_raw is not None:
                    diff = raw_rad - prev_raw
                    if diff > np.pi:
                        accumulated_offset -= 2.0 * np.pi
                    elif diff < -np.pi:
                        accumulated_offset += 2.0 * np.pi
                
                unwrapped_rad = raw_rad + accumulated_offset
                prev_raw = raw_rad

                # 3. Compute angular velocity (finite difference)
                if prev_time is not None:
                    dt = now - prev_time
                    vel_rad_s = (unwrapped_rad - prev_unwrapped) / dt if dt > 0 else 0.0
                else:
                    vel_rad_s = 0.0

                prev_unwrapped = unwrapped_rad
                prev_time = now

                # Store sample
                data_log.append((round(elapsed, 5), round(unwrapped_rad, 6), round(vel_rad_s, 6)))

                next_sample_time += SAMPLE_INTERVAL

            # Prevent high CPU spinning while waiting for next interval
            sleep_time = next_sample_time - time.perf_counter()
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording aborted early by user.")

    # Write to CSV
    print(f"\nWriting {len(data_log)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "pendulum_angle_unwrapped", "pendulum_vel_rad_s"])
        writer.writerows(data_log)

    print("Done! File ready for parameter fitting.")

if __name__ == "__main__":
    main()
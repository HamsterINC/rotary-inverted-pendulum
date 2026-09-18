import csv
import math
import time
import numpy as np
import spidev

# ==========================================
# 1. Configuration
# ==========================================
OUTPUT_FILE = "clamped_pendulum_decay.csv"
SAMPLE_RATE_HZ = 100.0  # 100 Hz = 10 ms interval
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE_HZ
DURATION_SECONDS = 8.0  # Stop recording after 8 seconds

PEND_ENCODER_RESOLUTION = 16384  # 14-bit (0x3FFF)
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
HALF_RESOLUTION = PEND_ENCODER_RESOLUTION // 2
PEND_ZERO_OFFSET_TICKS = 0

# ==========================================
# 2. SPI Hardware Setup
# ==========================================
spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 0)
spi_pendulum.max_speed_hz = 100_000
spi_pendulum.mode = 1
spi_pendulum.bits_per_word = 8


def read_raw_ticks(spi_device) -> int:
    """Reads 14-bit raw position from SPI encoder."""
    spi_device.xfer2([0xFF, 0xFF])
    data = spi_device.xfer2([0xC0, 0x00])
    raw = (data[0] << 8) | data[1]
    return raw & 0x3FFF


def get_delta_ticks(current_ticks: int, prev_ticks: int) -> int:
    """Computes tick difference handling circular encoder rollover (0 <-> 16383)."""
    delta = current_ticks - prev_ticks
    if delta > HALF_RESOLUTION:
        delta -= PEND_ENCODER_RESOLUTION
    elif delta < -HALF_RESOLUTION:
        delta += PEND_ENCODER_RESOLUTION
    return delta

def process_encoder(current_ticks: int, prev_ticks: int):
    norm_angle = (current_ticks + 5000) / 8192
    norm_angle = (norm_angle + 1.0) % 2.0 - 1.0
    angle_rad = norm_angle * math.pi

    delta = current_ticks - prev_ticks


    # Physical pendulum velocity in rad/s
    pen_vel_rad_s = (delta * PEND_LSB_RAD) / SAMPLE_INTERVAL

    return angle_rad, pen_vel_rad_s


# ==========================================
# 3. Main Logging Loop
# ==========================================
def main():
    print("=" * 60)
    print("CLAMPED PENDULUM LOGGER")
    print("1. Ensure the motor arm is rigidly clamped/locked.")
    print("2. Deflect the pendulum to ~30-45 degrees.")
    print("=" * 60)
    input("Press ENTER to start recording and immediately release the pendulum...")

    data_log = []

    # Read initial baseline
    prev_ticks = read_raw_ticks(spi_pendulum)
    accumulated_offset = 0.0
    prev_raw_rad = None
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

            if now >= next_sample_time:
                # 1. Read hardware encoder
                current_ticks = read_raw_ticks(spi_pendulum)

                # # 2. Convert ticks to raw continuous radians [-pi, pi]
                # # Center using the calibrated zero-offset tick position
                # centered_ticks = (current_ticks - PEND_ZERO_OFFSET_TICKS) % PEND_ENCODER_RESOLUTION
                # raw_rad = (centered_ticks * PEND_LSB_RAD)
                # if raw_rad > math.pi:
                #     raw_rad -= 2.0 * math.pi

                # # 3. Phase unwrap (prevents 2*pi jumps when crossing -pi / +pi)
                # if prev_raw_rad is not None:
                #     diff = raw_rad - prev_raw_rad
                #     if diff > math.pi:
                #         accumulated_offset -= 2.0 * math.pi
                #     elif diff < -math.pi:
                #         accumulated_offset += 2.0 * math.pi

                # unwrapped_rad = raw_rad + accumulated_offset
                # prev_raw_rad = raw_rad

                # # 4. Direct velocity differentiation using modular tick step
                # if prev_time is not None:
                #     dt = now - prev_time
                #     delta_ticks = get_delta_ticks(current_ticks, prev_ticks)
                #     vel_rad_s = (delta_ticks * PEND_LSB_RAD) / dt if dt > 0 else 0.0
                # else:
                #     vel_rad_s = 0.0

                # prev_ticks = current_ticks
                # prev_unwrapped = unwrapped_rad
                # prev_time = now

                angle_rad, vel_rad_s = process_encoder(current_ticks, prev_ticks)
                prev_ticks = current_ticks

                # Store row: [time, unwrapped_angle, angular_velocity]
                data_log.append((round(elapsed, 5), round(angle_rad, 6), round(vel_rad_s, 6)))

                next_sample_time += SAMPLE_INTERVAL

            # Yield thread to keep CPU usage low
            sleep_time = next_sample_time - time.perf_counter()
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nRecording aborted early by user.")
    finally:
        spi_pendulum.close()

    # Save to CSV
    print(f"\nWriting {len(data_log)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "pendulum_angle_unwrapped", "pendulum_vel_rad_s"])
        writer.writerows(data_log)

    print("Done! CSV saved successfully.")


if __name__ == "__main__":
    main()

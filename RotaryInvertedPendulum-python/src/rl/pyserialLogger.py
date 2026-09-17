import csv
import datetime
import sys
import time
import serial
import serial.tools.list_ports

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
SERIAL_PORT = "COM6"
BAUD_RATE = 115200
TIMEOUT_SEC = 2.0

TIMESTAMP_STR = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"./Output/telemetry_run_{TIMESTAMP_STR}.csv"
CSV_HEADER = [
    "t_ms",
    "s1_pos",
    "s1_vel",
    "s2_cos",
    "s2_sin",
    "s2_vel",
    "inference",
]

def q8_8_to_float(raw_val: int) -> float:
    """Converts a signed 16-bit Q8.8 fixed-point integer to a Python float."""
    val_16 = raw_val & 0xFFFF
    if val_16 & 0x8000:
        signed_int = val_16 - 0x10000
    else:
        signed_int = val_16
    return signed_int / 256.0

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    candidates = [
        p.device
        for p in ports
        if "USB" in p.description or "FTDI" in p.description or "Serial" in p.description
    ]
    if candidates:
        return candidates[-1]
    return None

def main():
    port = SERIAL_PORT or auto_detect_port()
    if not port:
        print("Error: No serial port specified and none detected automatically.")
        sys.exit(1)

    print(f"Connecting to {port} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_SEC)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        print("Tip: Make sure PuTTY or any other terminal is closed before running this script.")
        sys.exit(1)

    print(f"Waiting for non-zero data to begin logging to '{CSV_FILENAME}'...")
    print("Press Ctrl+C at any time to safely stop logging.\n")

    samples_logged = 0
    logging_started = False  # Latches True on the first non-zero frame

    with open(CSV_FILENAME, mode="w", newline="", buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        csv_file.flush()

        try:
            while True:
                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue

                line = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")

                if len(parts) == 7:
                    try:
                        t_ms = int(parts[0])

                        # Convert non-timestep fields
                        s1_pos = q8_8_to_float(int(parts[1]))
                        s1_vel = q8_8_to_float(int(parts[2]))
                        s2_cos = q8_8_to_float(int(parts[3]))
                        s2_sin = q8_8_to_float(int(parts[4]))
                        s2_vel = q8_8_to_float(int(parts[5]))
                        inf_act = q8_8_to_float(int(parts[6]))

                        signals = [s1_pos, s1_vel, s2_cos, s2_sin, s2_vel, inf_act]

                        # Check if any signal has moved past 0.00
                        has_activity = any(abs(val) >= 0.005 for val in signals)

                        # Trigger logging on the first non-zero frame
                        if not logging_started:
                            if has_activity:
                                logging_started = True
                                print("\n[TRIGGER] Non-zero telemetry detected. Logging started.\n")
                            else:
                                sys.stdout.write(f"\r[WAITING] t: {t_ms:>7} ms | Signals all 0.00...")
                                sys.stdout.flush()
                                continue

                        # Write row once logging has started
                        float_row = [
                            t_ms,
                            f"{s1_pos:.4f}",
                            f"{s1_vel:.4f}",
                            f"{s2_cos:.4f}",
                            f"{s2_sin:.4f}",
                            f"{s2_vel:.4f}",
                            f"{inf_act:.4f}",
                        ]

                        writer.writerow(float_row)
                        csv_file.flush()
                        samples_logged += 1

                        if samples_logged % 10 == 0:
                            sys.stdout.write(
                                f"\r[LOG] t: {t_ms:>7} ms | "
                                f"Pos: {s1_pos:>6.2f} | Vel: {s1_vel:>6.2f} | "
                                f"Cos: {s2_cos:>5.2f} Sin: {s2_sin:>5.2f} | "
                                f"Act: {inf_act:>6.2f} | Count: {samples_logged}"
                            )
                            sys.stdout.flush()

                    except ValueError:
                        pass

        except KeyboardInterrupt:
            print("\n\nLogging stopped by user.")
        finally:
            ser.close()
            print(f"Saved {samples_logged} rows to {CSV_FILENAME}")


if __name__ == "__main__":
    main()
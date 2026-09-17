from __future__ import annotations

import csv
from datetime import datetime
import io
import math
import os
import threading
import time
import zipfile

import Jetson.GPIO as GPIO
import spidev
import torch
import torch.nn as nn
from JetsonPWM import JetsonPWM

SB3_ZIP_PATH = "./Saved_runs/1509.zip"

# ==========================================
# Timing Constants (40 Hz Canonical Cycle)
# ==========================================
TARGET_HZ = 40.0
PERIOD = 1.0 / TARGET_HZ  # 0.025 s

# ==========================================
# Stepper / Arm Step-Tracking Constants
# ==========================================
STEPS_PER_REV = 12800
ARM_RAD_PER_STEP = (2.0 * math.pi) / STEPS_PER_REV

# Safety hard limits (±125 deg = 2.18166 rad)
ARM_SAFE_LIMIT_RAD = math.radians(125.0)
ARM_MAX_SAFE_STEPS = int(ARM_SAFE_LIMIT_RAD / ARM_RAD_PER_STEP)

# Kinematic Envelopes (Directly from sim env)
MAX_ACCEL_RAD_S2 = 150.0
MAX_VELOCITY_RAD_S = 5.0
ARM_MAX_DELTA_STEPS = (MAX_VELOCITY_RAD_S / ARM_RAD_PER_STEP) * PERIOD

PWM_MIN_FREQ_HZ = 60.0

# ==========================================
# Pendulum Calibration Constants (SPI)
# ==========================================
PEND_ENCODER_RESOLUTION = 16384
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
MAX_PENDULUM_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (MAX_PENDULUM_VEL_RAD_S / PEND_LSB_RAD) * PERIOD
PEND_ZERO_OFFSET_TICKS = 303

# Hardware Pins
DIR_PIN = 16
EN_PIN = 22

# Telemetry Buffer: appended in real-time, written on exit
log_buffer: list[dict[str, float]] = []

# ==========================================
# 1. SB3-Compatible MLP Architecture
# ==========================================
class SB3PolicyMLP(nn.Module):
    def __init__(self, in_features=6, out_features=1):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(128, out_features)

    def forward(self, x):
        features = self.policy_net(x)
        return self.action_net(features)

# ==========================================
# 2. Checkpoint Loader
# ==========================================
def load_weights_from_sb3_zip(model_target: nn.Module, zip_path: str, device: str = "cpu"):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Checkpoint archive '{zip_path}' not found.")

    print(f"[Loader] Opening SB3 checkpoint: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        policy_bytes = archive.read("policy.pth")
        buffer = io.BytesIO(policy_bytes)
        sb3_state = torch.load(buffer, map_location=device)

    target_state = {}
    for key, tensor in sb3_state.items():
        if "mlp_extractor.policy_net" in key:
            target_state[key.replace("mlp_extractor.", "")] = tensor
        elif "action_net" in key:
            target_state[key] = tensor

    model_target.load_state_dict(target_state, strict=False)
    print("[Loader] Successfully loaded actor weights.")

# ==========================================
# 3. SPI Hardware Setup
# ==========================================
spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 0)
spi_pendulum.max_speed_hz = 100_000
spi_pendulum.mode = 1
spi_pendulum.bits_per_word = 8

def read_raw_ticks(spi_device) -> int:
    spi_device.xfer2([0xFF, 0xFF])
    data = spi_device.xfer2([0xC0, 0x00])
    raw = (data[0] << 8) | data[1]
    return raw & 0x3FFF

def process_encoder(current_ticks: int, prev_ticks: int):
    norm_angle = (current_ticks + 5000) / 8192
    norm_angle = (norm_angle + 1.0) % 2.0 - 1.0
    angle_rad = norm_angle * math.pi

    delta = current_ticks - prev_ticks
    norm_vel = delta / float(PEND_MAX_DELTA_TICKS)
    norm_vel = max(-1.0, min(1.0, norm_vel))

    # Physical pendulum velocity in rad/s
    pen_vel_rad_s = (delta * PEND_LSB_RAD) / PERIOD

    return angle_rad, norm_vel, norm_angle

# ==========================================
# 4. Control Loop (Matches Sim Step Logic)
# ==========================================
stop_event = threading.Event()

def cpu_control_loop(model: nn.Module):
    pwm_step = JetsonPWM(chip=0, channel=0)
    pwm_step.start(initial_freq_hz=PWM_MIN_FREQ_HZ)

    arm_current_steps = 0
    prev_arm_steps = 0

    # Sim kinematic states
    motor_vel_rad_s = 0.0
    motor_target_rad = 0.0

    prev_pend_ticks = read_raw_ticks(spi_pendulum)
    action = 0.0

    print(f"[CPU Thread] Starting 40 Hz control loop.")

    with torch.no_grad():
        t_start_session = time.perf_counter()
        last_loop_time = t_start_session
        next_tick = last_loop_time

        while not stop_event.is_set():
            loop_start = time.perf_counter()
            actual_dt_s = loop_start - last_loop_time
            last_loop_time = loop_start

            if actual_dt_s <= 0.0:
                actual_dt_s = PERIOD

            # 1. Arm Observations
            arm_delta_steps = arm_current_steps - prev_arm_steps
            prev_arm_steps = arm_current_steps

            norm_arm_pos = arm_current_steps / float(STEPS_PER_REV / 2)
            norm_arm_pos = max(-1.0, min(1.0, norm_arm_pos))

            norm_arm_vel = arm_delta_steps / float(ARM_MAX_DELTA_STEPS)
            norm_arm_vel = max(-1.0, min(1.0, norm_arm_vel))

            # Actual physical arm velocity (rad/s) computed from step delta
            physical_arm_vel_rad_s = (arm_delta_steps * ARM_RAD_PER_STEP) / actual_dt_s
            arm_pos_rad = arm_current_steps * ARM_RAD_PER_STEP

            # 2. Pendulum Observations
            pend_ticks = read_raw_ticks(spi_pendulum)
            pend_rad, norm_pend_vel, norm_angle_pend = process_encoder(pend_ticks, prev_pend_ticks)
            prev_pend_ticks = pend_ticks

            # 3. Observation Tensor
            features = torch.tensor(
                [[
                    norm_arm_pos,
                    math.cos(pend_rad),
                    math.sin(pend_rad),
                    norm_arm_vel,
                    norm_pend_vel,
                    action,
                ]],
                dtype=torch.float32,
                device="cpu",
            )

            # 4. Inference
            action = float(model(features).item())
            action = max(-1.0, min(1.0, action))

            # 5. Exact Sim Kinematics Match:
            accel_cmd = action * MAX_ACCEL_RAD_S2
            accel_cmd = max(-MAX_ACCEL_RAD_S2, min(MAX_ACCEL_RAD_S2, accel_cmd))

            # Update commanded velocity
            motor_vel_rad_s = max(
                -MAX_VELOCITY_RAD_S,
                min(MAX_VELOCITY_RAD_S, motor_vel_rad_s + accel_cmd * actual_dt_s)
            )

            # Outward boundary clamp
            if motor_target_rad >= ARM_SAFE_LIMIT_RAD and motor_vel_rad_s > 0.0:
                motor_vel_rad_s = 0.0
            elif motor_target_rad <= -ARM_SAFE_LIMIT_RAD and motor_vel_rad_s < 0.0:
                motor_vel_rad_s = 0.0

            # Direct forward-Euler position integration (identical to sim)
            motor_target_rad = max(
                -ARM_SAFE_LIMIT_RAD,
                min(ARM_SAFE_LIMIT_RAD, motor_target_rad + motor_vel_rad_s * actual_dt_s)
            )

            # 6. Actuation to Pulse Hardware
            target_pos_steps = int(round(motor_target_rad / ARM_RAD_PER_STEP))
            target_pos_steps = max(-ARM_MAX_SAFE_STEPS, min(ARM_MAX_SAFE_STEPS, target_pos_steps))

            actual_steps = target_pos_steps - arm_current_steps

            if actual_steps != 0:
                is_forward = actual_steps > 0
                GPIO.output(DIR_PIN, GPIO.HIGH if is_forward else GPIO.LOW)

                freq = abs(actual_steps) / PERIOD
                freq = max(PWM_MIN_FREQ_HZ, freq)

                pwm_step.change_frequency(freq)
                arm_current_steps += actual_steps
            else:
                pwm_step.pause()

            # 7. Record Telemetry (In-Memory Buffer)
            log_buffer.append({
                "timestamp_s": round(loop_start - t_start_session, 5),
                "arm_pos_rad": round(norm_arm_pos, 4),
                "arm_actual_vel_rad_s": round(physical_arm_vel_rad_s, 4),
                "pendulum_pos_rad": round(norm_angle_pend, 4),
                "pendulum_vel_rad_s": round(norm_pend_vel, 4),
                "action_cmd": round(action, 4),
            })

            # 8. Governor
            next_tick += PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    pwm_step.stop()

# ==========================================
# 5. Flush Telemetry Buffer to CSV
# ==========================================
def save_log_to_file():
    if not log_buffer:
        print("[Logger] No telemetry frames recorded.")
        return

    os.makedirs("./logs", exist_ok=True)
    filename = datetime.now().strftime("./logs/run_%Y%m%d_%H%M%S.csv")
    print(f"[Logger] Flushing {len(log_buffer)} frames to {filename}...")

    fieldnames = list(log_buffer[0].keys())
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_buffer)

    print(f"[Logger] File saved successfully.")

# ==========================================
# 6. Entry Point
# ==========================================
if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)

    model_cpu = SB3PolicyMLP(in_features=6, out_features=1).to("cpu")
    model_cpu.eval()
    load_weights_from_sb3_zip(model_cpu, SB3_ZIP_PATH, device="cpu")

    control_thread = threading.Thread(target=cpu_control_loop, args=(model_cpu,), daemon=True)
    control_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        control_thread.join()
        GPIO.cleanup()
        spi_pendulum.close()
        save_log_to_file()
        print("Shutdown clean.")

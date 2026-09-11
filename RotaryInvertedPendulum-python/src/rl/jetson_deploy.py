from __future__ import annotations

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
from pwm_controller import HardwarePWMStep

SB3_ZIP_PATH = "./Saved_runs/best_model0809.zip"

TARGET_HZ = 40
PERIOD = 1.0 / TARGET_HZ  # 0.025 s

# ==========================================
# Stepper / Arm Step-Tracking Constants
# ==========================================
# Microsteps per full revolution (e.g., 200 steps/rev * 64 microstepping = 12,800)
# Match this to your stepper motor driver microstep configuration!
STEPS_PER_REV = 12800  
ARM_RAD_PER_STEP = (2.0 * math.pi) / STEPS_PER_REV

# Safety hard limits (±125 deg = 2.18166 rad)
ARM_SAFE_LIMIT_RAD = math.radians(125.0)
ARM_MAX_SAFE_STEPS = int(ARM_SAFE_LIMIT_RAD / ARM_RAD_PER_STEP)

# Normalization constants (matching environment bounds)
ARM_MAX_VEL_RAD_S = 5.0  # max motor velocity rad/s
ARM_MAX_DELTA_STEPS = (ARM_MAX_VEL_RAD_S / ARM_RAD_PER_STEP) * PERIOD

# Steps scaling for output action ∈ [-1, 1] per control step
ACTION_SCALE_STEPS = 102

# ==========================================
# Pendulum Calibration Constants (SPI)
# ==========================================
PEND_ENCODER_RESOLUTION = 16384 # Counts per full revolution
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
PEND_MAX_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (PEND_MAX_VEL_RAD_S / PEND_LSB_RAD) * PERIOD  # ~47.7 ticks
PEND_ZERO_OFFSET_TICKS = 303


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
# 2. Extract Weights Directly From SB3 ZIP
# ==========================================
def load_weights_from_sb3_zip(model_target, zip_path, device="cpu"):
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
# 3. Hardware Interfaces (Single SPI for Pendulum)
# ==========================================
DIR_PIN = 16
STEP_PIN = 18
EN_PIN = 22

# Only one SPI bus for the pendulum encoder
spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 0)  # CE0
spi_pendulum.max_speed_hz = 100_000
spi_pendulum.mode = 1
spi_pendulum.bits_per_word = 8


def read_raw_ticks(spi_device, resolution: int) -> int:
    """Clocks 2 bytes over SPI and restricts to valid resolution space."""
    spi_device.xfer2([0xFF, 0xFF])
    data = spi_device.xfer2([0xC0, 0x00])
    raw = (data[0] << 8) | data[1]
    return (raw & 0x3FFF)


# Initialize Hardware PWM (Pin 32 corresponds to chip 0, channel 0 on most Jetsons)
pwm_step = HardwarePWMStep(chip=0, channel=0)


def process_encoder(
    current_ticks: int,
    prev_ticks: int,
    zero_offset_ticks: int,
    resolution: int,
    max_delta_ticks: float,
):


    norm_angle = (current_ticks + 5000) / 8192
    norm_angle = (norm_angle + 1.0) % 2.0 -1.0
    angle_rad = norm_angle * math.pi

    # 2. Delta with circular wrap-around correction
    delta = current_ticks - prev_ticks

    # 3. Normalized angular velocity
    norm_vel = delta / max_delta_ticks
    norm_vel = max(-1.0, min(1.0, norm_vel))

    return norm_angle, angle_rad, norm_vel


# ==========================================
# 4. Setup Model
# ==========================================
stop_event = threading.Event()
model_cpu = SB3PolicyMLP(in_features=6, out_features=1).to("cpu")
model_cpu.eval()
load_weights_from_sb3_zip(model_cpu, SB3_ZIP_PATH, device="cpu")


# ==========================================
# 5. 100 Hz Real-Time Loop
# ==========================================
def cpu_control_loop():
    # Motor/Arm step-tracking state (starts at 0 / centered)
    arm_current_steps = 0
    prev_arm_steps = 0

    # Pendulum encoder initialization
    prev_pend_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
    action = 0.0

    print("[CPU Thread] Motor zeroed. Starting 40 Hz control loop.")

    with torch.no_grad():
        next_tick = time.perf_counter()

        while not stop_event.is_set():
            # 1. Compute Arm State from Accumulated Steps
            arm_delta_steps = arm_current_steps - prev_arm_steps
            prev_arm_steps = arm_current_steps

            # Normalize arm position to [-1.0, 1.0] relative to safe travel limits
            norm_arm_pos = arm_current_steps / float(STEPS_PER_REV/2)
            norm_arm_pos = max(-1.0, min(1.0, norm_arm_pos))

            # Normalize arm velocity to [-1.0, 1.0]
            norm_arm_vel = arm_delta_steps / float(ARM_MAX_DELTA_STEPS)
            norm_arm_vel = max(-1.0, min(1.0, norm_arm_vel))

            # 2. Read Pendulum SPI Encoder
            pendulum_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
            _, pendulum_rad, norm_pendulum_velocity = process_encoder(
                current_ticks=pendulum_ticks,
                prev_ticks=prev_pend_ticks,
                zero_offset_ticks=PEND_ZERO_OFFSET_TICKS,
                resolution=PEND_ENCODER_RESOLUTION,
                max_delta_ticks=PEND_MAX_DELTA_TICKS,
            )
            prev_pend_ticks = pendulum_ticks
            #print("Pendulum Count:", pendulum_ticks, "Angle (rad):", pendulum_rad, "Norm Vel:", norm_pendulum_velocity)
            # 3. Assemble observation vector:
            # [motor_pos_norm, cos(theta), sin(theta), motor_vel_norm, pen_vel_norm, prev_action]
            features = torch.tensor(
                [[
                    norm_arm_pos,
                    math.cos(pendulum_rad),
                    math.sin(pendulum_rad),
                    norm_arm_vel,
                    norm_pendulum_velocity,
                    action,
                ]],
                dtype=torch.float32,
                device="cpu",
            )

            # 4. CPU Inference
            action = float(model_cpu(features).item())
            action = max(-1.0, min(1.0, action))

            # 5. Calculate Desired Steps & Software Rail Clamping
            desired_steps = int(round(action * ACTION_SCALE_STEPS))
            target_pos_steps = arm_current_steps + desired_steps

            # Clamp commanded steps so arm never exceeds physical safety limit
            clamped_pos_steps = max(-ARM_MAX_SAFE_STEPS, min(ARM_MAX_SAFE_STEPS, target_pos_steps))
            actual_steps = clamped_pos_steps - arm_current_steps
            #print("Actual Steps:", actual_steps)
            # 5. Output via Hardware PWM
            if actual_steps != 0:
                is_forward = actual_steps > 0
                GPIO.output(DIR_PIN, GPIO.HIGH if is_forward else GPIO.LOW)
                
                # Frequency to deliver 'actual_steps' over PERIOD (0.025s)
                freq = abs(actual_steps) / PERIOD  # e.g., 50 steps / 0.025s = 2000 Hz
                pwm_step.set_frequency(freq)
                
                arm_current_steps += actual_steps
            else:
                pwm_step.stop()

            # 7. Maintain strict 100 Hz cycle
            next_tick += PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    pwm_step.stop()
    pwm_step.close()


if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)  # Enable driver

    t = threading.Thread(target=cpu_control_loop, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        t.join()
        GPIO.cleanup()
        spi_pendulum.close()
        print("Shutdown clean.")
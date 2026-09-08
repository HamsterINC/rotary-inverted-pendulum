import io
import math
import os
import threading
import time
import zipfile
import spidev
import torch
import torch.nn as nn
import Jetson.GPIO as GPIO

SB3_ZIP_PATH = "ppo_cartpole_stepper.zip"

TARGET_HZ = 100
PERIOD = 1.0 / TARGET_HZ  # 0.010 s

# ==========================================
# Arm Calibration Constants
# ==========================================
ARM_ENCODER_RESOLUTION = 1000  # Counts per full revolution
ARM_LSB_RAD = (2.0 * math.pi) / ARM_ENCODER_RESOLUTION
ARM_MAX_VEL_RAD_S = 10.0
ARM_MAX_DELTA_TICKS = (ARM_MAX_VEL_RAD_S / ARM_LSB_RAD) * PERIOD  # ~15.9 ticks
ARM_ZERO_OFFSET_TICKS = 902

# ==========================================
# Pendulum Calibration Constants
# ==========================================
PEND_ENCODER_RESOLUTION = 1000  # Counts per full revolution
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
PEND_MAX_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (PEND_MAX_VEL_RAD_S / PEND_LSB_RAD) * PERIOD  # ~47.7 ticks
PEND_ZERO_OFFSET_TICKS = 10


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
# 3. Hardware Interfaces
# ==========================================
DIR_PIN = 16
STEP_PIN = 18
EN_PIN = 22

spi_arm = spidev.SpiDev()
spi_arm.open(0, 0)
spi_arm.max_speed_hz = 500_000
spi_arm.mode = 0

spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 1)
spi_pendulum.max_speed_hz = 500_000
spi_pendulum.mode = 0


def read_raw_ticks(spi_device, resolution):
    """Clocks 2 bytes over SPI and restricts to valid resolution space."""
    data = spi_device.xfer2([0x00, 0x00])
    raw = (data[0] << 8) | data[1]
    return (raw & 0x0FFF) % resolution


def pulse_stepper(steps: int, forward: bool):
    """Pulses stepper without yielding thread to kernel scheduler."""
    GPIO.output(DIR_PIN, GPIO.HIGH if forward else GPIO.LOW)
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        # Python function execution overhead provides sufficient >1 µs setup time
        GPIO.output(STEP_PIN, GPIO.LOW)


def process_encoder(
    current_ticks: int,
    prev_ticks: int,
    zero_offset_ticks: int,
    resolution: int,
    max_delta_ticks: float,
):
    half_res = resolution / 2.0

    # 1. Zero-centered angle relative to calibrated offset
    centered = (current_ticks - zero_offset_ticks) % resolution
    if centered >= half_res:
        centered -= resolution

    norm_angle = centered / half_res
    norm_angle = max(-1.0, min(1.0, norm_angle))
    angle_rad = norm_angle * math.pi

    # 2. Delta with circular boundary wrap-around correction
    delta = current_ticks - prev_ticks
    if delta > half_res:
        delta -= resolution
    elif delta < -half_res:
        delta += resolution

    # 3. Normalized angular velocity
    norm_vel = delta / max_delta_ticks
    norm_vel = max(-1.0, min(1.0, norm_vel))

    return norm_angle, angle_rad, norm_vel


# ==========================================
# 4. Setup Model
# ==========================================
stop_event = threading.Event()
weights_lock = threading.Lock()

model_cpu = SB3PolicyMLP(in_features=6, out_features=1).to("cpu")
model_cpu.eval()
load_weights_from_sb3_zip(model_cpu, SB3_ZIP_PATH, device="cpu")

pending_cpu_weights = None
new_weights_available = False


# ==========================================
# 5. 100 Hz Real-Time Loop
# ==========================================
def cpu_control_loop():
    global new_weights_available, pending_cpu_weights

    # Initialize previous tick and action variables locally
    prev_arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
    prev_pend_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
    action = 0.0

    print("[CPU Thread] Pre-sampled encoders. Starting 100 Hz loop.")

    with torch.no_grad():
        next_tick = time.perf_counter()

        while not stop_event.is_set():
            # # Atomic weight reload from GPU thread (if enabled)
            # if new_weights_available:
            #     with weights_lock:
            #         if pending_cpu_weights is not None:
            #             model_cpu.load_state_dict(pending_cpu_weights)
            #             pending_cpu_weights = None
            #             new_weights_available = False

            # 1. Read SPI
            arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
            norm_arm_angle, arm_rad, norm_arm_vel = process_encoder(
                current_ticks=arm_ticks,
                prev_ticks=prev_arm_ticks,
                zero_offset_ticks=ARM_ZERO_OFFSET_TICKS,
                resolution=ARM_ENCODER_RESOLUTION,
                max_delta_ticks=ARM_MAX_DELTA_TICKS,
            )
            prev_arm_ticks = arm_ticks

            pendulum_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
            norm_pendulum_angle, pendulum_rad, norm_pendulum_velocity = process_encoder(
                current_ticks=pendulum_ticks,
                prev_ticks=prev_pend_ticks,
                zero_offset_ticks=PEND_ZERO_OFFSET_TICKS,
                resolution=PEND_ENCODER_RESOLUTION,
                max_delta_ticks=PEND_MAX_DELTA_TICKS,
            )
            prev_pend_ticks = pendulum_ticks

            # 2. Assemble observation vector
            features = torch.tensor(
                [[
                    norm_arm_angle,
                    norm_arm_vel,
                    math.sin(pendulum_rad),
                    math.cos(pendulum_rad),
                    norm_pendulum_velocity,
                    action,
                ]],
                dtype=torch.float32,
                device="cpu",
            )

            # 3. CPU Inference
            action = model_cpu(features).item()

            # 4. Pulse Stepper Motor
            steps = int(abs(action) * 20)
            if steps > 0:
                pulse_stepper(steps=steps, forward=(action >= 0.0))

            # 5. Maintain strict 100 Hz cycle
            next_tick += PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()


if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)

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
        spi_arm.close()
        print("Shutdown clean.")
import collections
import io
import math
import os
import threading
import time
import zipfile
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import spidev
import torch
import torch.nn as nn
import Jetson.GPIO as GPIO

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# Paths & Hyperparameters
# ==========================================
SB3_ZIP_PATH = "ppo_cartpole_stepper.zip"
SAVED_CHECKPOINT_PATH = "adapted_policy.zip"

TARGET_HZ = 100
PERIOD = 1.0 / TARGET_HZ  # 10 ms

# Arm Calibration
ARM_ENCODER_RESOLUTION = 1000
ARM_LSB_RAD = (2.0 * math.pi) / ARM_ENCODER_RESOLUTION
ARM_MAX_VEL_RAD_S = 10.0
ARM_MAX_DELTA_TICKS = (ARM_MAX_VEL_RAD_S / ARM_LSB_RAD) * PERIOD  # ~15.9 ticks
ARM_ZERO_OFFSET_TICKS = 902

# Pendulum Calibration
PEND_ENCODER_RESOLUTION = 1000
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
PEND_MAX_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (PEND_MAX_VEL_RAD_S / PEND_LSB_RAD) * PERIOD  # ~47.7 ticks
PEND_ZERO_OFFSET_TICKS = 10

# Hardware Pinout
DIR_PIN = 16
STEP_PIN = 18
EN_PIN = 22

# ==========================================
# 1. Lean CPU Inference Architecture
# ==========================================
class SB3PolicyMLP(nn.Module):
    """Matches SB3 MlpPolicy actor head for microsecond CPU inference."""
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
        return self.action_net(self.policy_net(x))


def extract_actor_weights(sb3_model, target_cpu_model):
    """Extracts actor weights from an active SB3 PPO policy into our lean CPU model."""
    sb3_state = sb3_model.policy.state_dict()
    target_state = {}
    for key, tensor in sb3_state.items():
        if "mlp_extractor.policy_net" in key:
            target_state[key.replace("mlp_extractor.", "")] = tensor.cpu()
        elif "action_net" in key:
            target_state[key] = tensor.cpu()
    target_cpu_model.load_state_dict(target_state, strict=False)


# ==========================================
# 2. Hardware Interfaces (SPI & GPIO)
# ==========================================
spi_arm = spidev.SpiDev()
spi_arm.open(0, 0)
spi_arm.max_speed_hz = 500_000
spi_arm.mode = 0

spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 1)
spi_pendulum.max_speed_hz = 500_000
spi_pendulum.mode = 0

def read_raw_ticks(spi_device, resolution):
    data = spi_device.xfer2([0x00, 0x00])
    raw = (data[0] << 8) | data[1]
    return (raw & 0x0FFF) % resolution

def pulse_stepper(steps: int, forward: bool):
    GPIO.output(DIR_PIN, GPIO.HIGH if forward else GPIO.LOW)
    for _ in range(steps):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        GPIO.output(STEP_PIN, GPIO.LOW)

def process_encoder(current_ticks: int, prev_ticks: int, zero_offset_ticks: int, resolution: int, max_delta_ticks: float):
    half_res = resolution / 2.0
    centered = (current_ticks - zero_offset_ticks) % resolution
    if centered >= half_res:
        centered -= resolution

    norm_angle = centered / half_res
    norm_angle = max(-1.0, min(1.0, norm_angle))
    angle_rad = norm_angle * math.pi

    delta = current_ticks - prev_ticks
    if delta > half_res:
        delta -= resolution
    elif delta < -half_res:
        delta += resolution

    norm_vel = delta / max_delta_ticks
    norm_vel = max(-1.0, min(1.0, norm_vel))
    return norm_angle, angle_rad, norm_vel

def compute_reward(norm_arm_angle, norm_arm_vel, cos_pend, norm_pend_vel, action):
    r_upright = (cos_pend + 1.0) / 2.0
    r_arm = -0.2 * (norm_arm_angle ** 2)
    r_vel = -0.05 * (norm_pend_vel ** 2 + norm_arm_vel ** 2)
    r_ctrl = -0.01 * (action ** 2)
    return float(r_upright + r_arm + r_vel + r_ctrl)


# ==========================================
# 3. Custom Thread-Bridging Gym Environment
# ==========================================
transition_event = threading.Event()
latest_transition = {}

class JetsonHardwareEnv(gym.Env):
    """
    Gym environment that feeds real transitions collected by the CPU loop
    into Stable-Baselines3 PPO running on the Jetson GPU.
    """
    def __init__(self):
        super().__init__()
        # Observation space matches our 6 inputs:
        # [arm_pos, arm_vel, sin(pend), cos(pend), pend_vel, prev_action]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Continuous action space for stepper [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.current_obs = np.zeros(6, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        transition_event.wait()
        transition_event.clear()
        self.current_obs = latest_transition["obs"]
        return self.current_obs, {}

    def step(self, action):
        # Wait until the 100 Hz CPU loop records the next hardware cycle
        transition_event.wait()
        transition_event.clear()

        obs = latest_transition["obs"]
        reward = latest_transition["reward"]
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


# ==========================================
# 4. Multi-Thread State & Hot-Swap Callback
# ==========================================
stop_event = threading.Event()
weights_lock = threading.Lock()

model_cpu = SB3PolicyMLP(in_features=6, out_features=1).to("cpu")
model_cpu.eval()

pending_cpu_weights = None
new_weights_available = False

class OnPolicyWeightSyncCallback(BaseCallback):
    """
    Invoked by SB3 on the GPU after completing a batch of rollouts and optimization steps.
    Stages new policy parameters into RAM for the CPU inference loop.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        global new_weights_available, pending_cpu_weights
        
        # Extract actor parameters from GPU policy
        sb3_state = self.model.policy.state_dict()
        cpu_dict = {}
        for key, tensor in sb3_state.items():
            if "mlp_extractor.policy_net" in key:
                cpu_dict[key.replace("mlp_extractor.", "")] = tensor.detach().to("cpu", copy=True)
            elif "action_net" in key:
                cpu_dict[key] = tensor.detach().to("cpu", copy=True)

        with weights_lock:
            pending_cpu_weights = cpu_dict
            new_weights_available = True

        print("[GPU Learner] Finished PPO iteration. Staged updated policy to CPU.")


# ==========================================
# 5. Thread A: 100 Hz Real-Time Control Loop
# ==========================================
def cpu_control_loop():
    global new_weights_available, pending_cpu_weights, latest_transition

    prev_arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
    prev_pend_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
    action = 0.0

    print("[CPU Thread] Starting 100 Hz real-time control loop.")

    with torch.no_grad():
        next_tick = time.perf_counter()

        while not stop_event.is_set():
            # 1. Hot-swap updated weights from SB3 GPU learner
            if new_weights_available:
                with weights_lock:
                    if pending_cpu_weights is not None:
                        model_cpu.load_state_dict(pending_cpu_weights)
                        pending_cpu_weights = None
                        new_weights_available = False

            # 2. Read Sensors over SPI
            arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
            norm_arm_angle, _, norm_arm_vel = process_encoder(
                arm_ticks, prev_arm_ticks, ARM_ZERO_OFFSET_TICKS, ARM_ENCODER_RESOLUTION, ARM_MAX_DELTA_TICKS
            )
            prev_arm_ticks = arm_ticks

            pend_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
            norm_pend_angle, pend_rad, norm_pend_vel = process_encoder(
                pend_ticks, prev_pend_ticks, PEND_ZERO_OFFSET_TICKS, PEND_ENCODER_RESOLUTION, PEND_MAX_DELTA_TICKS
            )
            prev_pend_ticks = pend_ticks

            sin_pend = math.sin(pend_rad)
            cos_pend = math.cos(pend_rad)

            # 3. Build Observation Vector
            current_obs = np.array(
                [norm_arm_angle, norm_arm_vel, sin_pend, cos_pend, norm_pend_vel, action],
                dtype=np.float32
            )

            # 4. Notify Gym Env of live transition
            reward = compute_reward(norm_arm_angle, norm_arm_vel, cos_pend, norm_pend_vel, action)
            latest_transition = {
                "obs": current_obs,
                "reward": reward,
            }
            transition_event.set()

            # 5. Low-latency deterministic CPU inference (< 40 us)
            features = torch.tensor(current_obs, dtype=torch.float32, device="cpu").unsqueeze(0)
            action = model_cpu(features).item()

            # 6. Actuate Stepper Motor
            steps = int(abs(action) * 20)
            if steps > 0:
                pulse_stepper(steps=steps, forward=(action >= 0.0))

            # 7. Regulate 100 Hz Interval
            next_tick += PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    print("[CPU Thread] Exiting.")


# ==========================================
# 6. Thread B: Background SB3 PPO GPU Training
# ==========================================
def gpu_sb3_training_loop():
    print("[GPU Thread] Initializing SB3 PPO environment and model...")

    env = JetsonHardwareEnv()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(SB3_ZIP_PATH):
        print(f"[GPU Thread] Loading existing model from {SB3_ZIP_PATH} onto {device}...")
        sb3_model = PPO.load(SB3_ZIP_PATH, env=env, device=device)
    else:
        print(f"[GPU Thread] Creating new PPO policy on {device}...")
        sb3_model = PPO(
            "MlpPolicy",
            env,
            device=device,
            n_steps=512,           # 512 steps at 100 Hz = 5.12 seconds per rollout
            batch_size=64,         # Mini-batch size for GPU optimization
            n_epochs=10,           # PPO update epochs per rollout
            learning_rate=3e-4,
            gamma=0.99,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=torch.nn.ReLU
            ),
            verbose=1
        )

    # Initial copy of loaded weights to the CPU model before starting loop
    extract_actor_weights(sb3_model, model_cpu)

    callback = OnPolicyWeightSyncCallback()

    print("[GPU Thread] Starting SB3 learning loop...")
    while not stop_event.is_set():
        # Train for 2048 steps at a time in the background
        sb3_model.learn(total_timesteps=2048, reset_num_timesteps=False, callback=callback)
        sb3_model.save(SAVED_CHECKPOINT_PATH)
        print(f"[GPU Thread] Checkpoint saved to {SAVED_CHECKPOINT_PATH}")

    print("[GPU Thread] Exiting.")


# ==========================================
# 7. Main Execution & Safe Exit
# ==========================================
if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)

    cpu_thread = threading.Thread(target=cpu_control_loop, daemon=True)
    gpu_thread = threading.Thread(target=gpu_sb3_training_loop, daemon=True)

    # Start CPU control loop first so initial readings are streaming
    cpu_thread.start()
    time.sleep(0.1)
    gpu_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] Shutdown signal received...")
        stop_event.set()
        transition_event.set()  # Unblock any waiting steps in gym env

        cpu_thread.join()
        gpu_thread.join()

        GPIO.cleanup()
        spi_pendulum.close()
        spi_arm.close()
        print("[Main] Hardware released. Safe shutdown complete.")
import collections
import math
import os
import threading
import time
from typing import Tuple

import numpy as np
import spidev
import torch
import torch.nn as nn
import Jetson.GPIO as GPIO

# JAX & Scientific Imports
import jax
import jax.numpy as jnp
import optax
from flax import struct
from purejaxrl.purejaxrl.ppo_continuous_action import make_train

# Enforce GPU backend for JAX operations
jax.config.update("jax_platform_name", "gpu")

# ==========================================
# 1. Hardware Calibration & Pinout
# ==========================================
TARGET_HZ = 100
PERIOD = 1.0 / TARGET_HZ  # 10 ms

ARM_ENCODER_RESOLUTION = 1000
ARM_LSB_RAD = (2.0 * math.pi) / ARM_ENCODER_RESOLUTION
ARM_MAX_VEL_RAD_S = 10.0
ARM_MAX_DELTA_TICKS = (ARM_MAX_VEL_RAD_S / ARM_LSB_RAD) * PERIOD
ARM_ZERO_OFFSET_TICKS = 902

PEND_ENCODER_RESOLUTION = 1000
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
PEND_MAX_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (PEND_MAX_VEL_RAD_S / PEND_LSB_RAD) * PERIOD
PEND_ZERO_OFFSET_TICKS = 10

DIR_PIN = 16
STEP_PIN = 18
EN_PIN = 22

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


# ==========================================
# 3. Lean CPU Inference Architecture
# ==========================================
class SB3PolicyMLP(nn.Module):
    """Low-latency inference model executed on CPU."""
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


def flax_to_torch_state(flax_params):
    """Zero-copy/numpy conversion of Flax weights to PyTorch tensors."""
    d0_kernel = np.array(flax_params["Dense_0"]["kernel"])
    d0_bias = np.array(flax_params["Dense_0"]["bias"])
    d1_kernel = np.array(flax_params["Dense_1"]["kernel"])
    d1_bias = np.array(flax_params["Dense_1"]["bias"])
    d2_kernel = np.array(flax_params["Dense_2"]["kernel"])
    d2_bias = np.array(flax_params["Dense_2"]["bias"])

    return {
        "policy_net.0.weight": torch.from_numpy(d0_kernel.T).float(),
        "policy_net.0.bias": torch.from_numpy(d0_bias).float(),
        "policy_net.2.weight": torch.from_numpy(d1_kernel.T).float(),
        "policy_net.2.bias": torch.from_numpy(d1_bias).float(),
        "action_net.weight": torch.from_numpy(d2_kernel.T).float(),
        "action_net.bias": torch.from_numpy(d2_bias).float(),
    }


# ==========================================
# 4. Pure JAX Simulation Environment
# ==========================================
@struct.dataclass
class EnvState:
    theta: float
    theta_dot: float
    alpha: float
    alpha_dot: float
    prev_action: float
    time_step: int

@struct.dataclass
class EnvParams:
    dt: float = 0.01
    g: float = 9.81
    m_p: float = 0.024
    L_p: float = 0.129
    l_p: float = 0.0645
    J_p: float = (0.024 * (0.129 ** 2)) / 3.0
    b_p: float = 5e-5
    b_arm: float = 1e-3
    arm_accel_scale: float = 25.0
    arm_max_vel: float = 10.0
    pend_max_vel: float = 30.0
    max_steps: int = 500

class RotaryPendulumJAX:
    def __init__(self):
        self.obs_dim = 6
        self.act_dim = 1

    def get_obs(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        norm_arm_pos = jnp.clip(state.theta / jnp.pi, -1.0, 1.0)
        norm_arm_vel = jnp.clip(state.theta_dot / params.arm_max_vel, -1.0, 1.0)
        sin_pend = jnp.sin(state.alpha)
        cos_pend = jnp.cos(state.alpha)
        norm_pend_vel = jnp.clip(state.alpha_dot / params.pend_max_vel, -1.0, 1.0)

        return jnp.array([
            norm_arm_pos, norm_arm_vel, sin_pend, cos_pend, norm_pend_vel, state.prev_action
        ], dtype=jnp.float32)

    def reset(self, rng: jax.Array, params: EnvParams = None) -> Tuple[jnp.ndarray, EnvState]:
        if params is None:
            params = EnvParams()
        k1, k2 = jax.random.split(rng)
        init_arm = jax.random.uniform(k1, shape=(), minval=-0.05, maxval=0.05)
        init_pend = jax.random.uniform(k2, shape=(), minval=-0.08, maxval=0.08)

        state = EnvState(
            theta=init_arm,
            theta_dot=0.0,
            alpha=init_pend,
            alpha_dot=0.0,
            prev_action=0.0,
            time_step=0
        )
        return self.get_obs(state, params), state

    def step(self, rng: jax.Array, state: EnvState, action: jnp.ndarray, params: EnvParams = None):
        if params is None:
            params = EnvParams()

        a = jnp.clip(action[0], -1.0, 1.0)
        arm_accel = a * params.arm_accel_scale - params.b_arm * state.theta_dot

        alpha_ddot = (
            (params.m_p * params.g * params.l_p * jnp.sin(state.alpha))
            - (params.b_p * state.alpha_dot)
            - (params.m_p * params.l_p * arm_accel * jnp.cos(state.alpha))
        ) / params.J_p

        theta_dot = state.theta_dot + arm_accel * params.dt
        theta = state.theta + theta_dot * params.dt
        alpha_dot = state.alpha_dot + alpha_ddot * params.dt
        alpha = state.alpha + alpha_dot * params.dt
        alpha = (alpha + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        new_state = EnvState(
            theta=theta, theta_dot=theta_dot, alpha=alpha, alpha_dot=alpha_dot,
            prev_action=a, time_step=state.time_step + 1
        )

        obs = self.get_obs(new_state, params)
        r_upright = (obs[3] + 1.0) / 2.0
        r_arm = -0.2 * (obs[0] ** 2)
        r_vel = -0.05 * (obs[4] ** 2 + obs[1] ** 2)
        r_ctrl = -0.01 * (a ** 2)
        reward = r_upright + r_arm + r_vel + r_ctrl

        terminated = jnp.abs(alpha) > (jnp.pi / 3.0)
        truncated = new_state.time_step >= params.max_steps
        done = jnp.logical_or(terminated, truncated)

        reset_obs, reset_state = self.reset(rng, params)
        final_obs = jax.tree.map(lambda x, y: jnp.where(done, x, y), reset_obs, obs)
        final_state = jax.tree.map(lambda x, y: jnp.where(done, x, y), reset_state, new_state)

        return final_obs, final_state, reward, done, {"discount": 1.0 - terminated.astype(jnp.float32)}


# ==========================================
# 5. Differentiable System Identification (SysID)
# ==========================================
@struct.dataclass
class LearnableParams:
    log_m_p: float
    log_l_p: float
    log_b_p: float
    log_b_arm: float
    log_accel_scale: float

def init_learnable_params(env_params: EnvParams) -> LearnableParams:
    return LearnableParams(
        log_m_p=jnp.log(env_params.m_p),
        log_l_p=jnp.log(env_params.l_p),
        log_b_p=jnp.log(env_params.b_p),
        log_b_arm=jnp.log(env_params.b_arm),
        log_accel_scale=jnp.log(env_params.arm_accel_scale)
    )

def forward_step_sysid(state, action, params: LearnableParams, dt: float = 0.01):
    theta, theta_dot, alpha, alpha_dot = state
    a = jnp.clip(action, -1.0, 1.0)

    m_p = jnp.exp(params.log_m_p)
    l_p = jnp.exp(params.log_l_p)
    b_p = jnp.exp(params.log_b_p)
    b_arm = jnp.exp(params.log_b_arm)
    accel_scale = jnp.exp(params.log_accel_scale)

    g = 9.81
    J_p = (m_p * ((2.0 * l_p) ** 2)) / 3.0

    arm_accel = a * accel_scale - b_arm * theta_dot
    alpha_ddot = (
        (m_p * g * l_p * jnp.sin(alpha))
        - (b_p * alpha_dot)
        - (m_p * l_p * arm_accel * jnp.cos(alpha))
    ) / J_p

    theta_dot_next = theta_dot + arm_accel * dt
    theta_next = theta + theta_dot_next * dt
    alpha_dot_next = alpha_dot + alpha_ddot * dt
    alpha_next = alpha + alpha_dot_next * dt
    alpha_next = (alpha_next + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    next_s = jnp.array([theta_next, theta_dot_next, alpha_next, alpha_dot_next])
    return next_s, next_s

def sysid_loss_fn(learn_params: LearnableParams, real_states, real_actions):
    chunk_size = 50
    num_chunks = real_states.shape[0] // chunk_size

    def chunk_loss(i):
        idx = i * chunk_size
        init_state = real_states[idx]
        sub_actions = real_actions[idx:idx + chunk_size]
        sub_real_states = real_states[idx + 1:idx + chunk_size + 1]

        def scan_fn(carrier, act):
            nxt, out = forward_step_sysid(carrier, act[0], learn_params)
            return nxt, out

        _, pred_states = jax.lax.scan(scan_fn, init_state, sub_actions)

        pos_arm_err = (pred_states[:, 0] - sub_real_states[:, 0]) ** 2
        vel_arm_err = 0.1 * (pred_states[:, 1] - sub_real_states[:, 1]) ** 2
        alpha_err = (jnp.sin(pred_states[:, 2]) - jnp.sin(sub_real_states[:, 2])) ** 2 + \
                    (jnp.cos(pred_states[:, 2]) - jnp.cos(sub_real_states[:, 2])) ** 2
        vel_pend_err = 0.1 * (pred_states[:, 3] - sub_real_states[:, 3]) ** 2

        return jnp.mean(pos_arm_err + vel_arm_err + 5.0 * alpha_err + vel_pend_err)

    losses = jax.vmap(chunk_loss)(jnp.arange(num_chunks - 1))
    return jnp.mean(losses)


# ==========================================
# 6. Synchronization & Shared State
# ==========================================
stop_event = threading.Event()
weights_lock = threading.Lock()
buffer_lock = threading.Lock()

model_cpu = SB3PolicyMLP(in_features=6, out_features=1).to("cpu")
model_cpu.eval()

pending_torch_weights = None
new_weights_available = False

# Trajectory memory bridging 100 Hz loop -> JAX SysID
shared_states_buffer = collections.deque(maxlen=1000)   # 10 seconds of physical data
shared_actions_buffer = collections.deque(maxlen=1000)


# ==========================================
# 7. Thread A: 100 Hz Real-Time Control Loop (CPU)
# ==========================================
def cpu_control_loop():
    global new_weights_available, pending_torch_weights

    prev_arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
    prev_pend_ticks = read_raw_ticks(spi_pendulum, PEND_ENCODER_RESOLUTION)
    action = 0.0

    print("[CPU Thread] Real-time 100 Hz hardware loop active.")

    with torch.no_grad():
        next_tick = time.perf_counter()

        while not stop_event.is_set():
            # 1. Non-blocking hot-swap of trained policy weights
            if new_weights_available:
                with weights_lock:
                    if pending_torch_weights is not None:
                        model_cpu.load_state_dict(pending_torch_weights)
                        pending_torch_weights = None
                        new_weights_available = False

            # 2. Sample hardware encoders over SPI
            arm_ticks = read_raw_ticks(spi_arm, ARM_ENCODER_RESOLUTION)
            norm_arm_angle, arm_rad, norm_arm_vel = process_encoder(
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

            # 3. Log un-normalized physical state for SysID: [theta, theta_dot, alpha, alpha_dot]
            actual_arm_vel = norm_arm_vel * ARM_MAX_VEL_RAD_S
            actual_pend_vel = norm_pend_vel * PEND_MAX_VEL_RAD_S
            physical_state = np.array([arm_rad, actual_arm_vel, pend_rad, actual_pend_vel], dtype=np.float32)

            with buffer_lock:
                shared_states_buffer.append(physical_state)
                shared_actions_buffer.append(np.array([action], dtype=np.float32))

            # 4. Low-latency policy inference on CPU (< 40 us)
            obs = np.array(
                [norm_arm_angle, norm_arm_vel, sin_pend, cos_pend, norm_pend_vel, action],
                dtype=np.float32
            )
            features = torch.tensor(obs, dtype=torch.float32, device="cpu").unsqueeze(0)
            action = float(model_cpu(features).item())

            # 5. Actuate stepper motor
            steps = int(abs(action) * 20)
            if steps > 0:
                pulse_stepper(steps=steps, forward=(action >= 0.0))

            # 6. Regulate 100 Hz strict interval
            next_tick += PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    print("[CPU Thread] Terminated.")


# ==========================================
# 8. Thread B: Continual SysID + 100-Env JAX PPO (GPU)
# ==========================================
def jax_continual_learning_loop():
    global new_weights_available, pending_torch_weights

    print("[JAX Thread] Initializing continual training engine on GPU...")

    current_env_params = EnvParams()
    env = RotaryPendulumJAX()

    ppo_config = {
        "ENV_NAME": "RotaryPendulumJAX",
        "NUM_ENVS": 100,             # 100 parallel vectorized environments
        "NUM_STEPS": 256,            # 100 * 256 = 25,600 transitions per update
        "TOTAL_TIMESTEPS": 256_000,  # 10 PPO iterations per training cycle
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ANNEAL_LR": False,
        "DEBUG": False,
    }

    # Setup SysID optimizer
    sysid_optimizer = optax.adam(learning_rate=0.01)

    @jax.jit
    def sysid_step(p, opt_st, states, actions):
        loss, grads = jax.value_and_grad(sysid_loss_fn)(p, states, actions)
        updates, new_opt_st = sysid_optimizer.update(grads, opt_st)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_opt_st, loss

    rng = jax.random.PRNGKey(42)
    cycle = 1

    while not stop_event.is_set():
        print(f"\n========== [Cycle {cycle}] Starting Iteration ==========")

        # ----------------------------------------------------
        # STAGE 1: Check Trajectory Buffer
        # ----------------------------------------------------
        with buffer_lock:
            buffer_len = len(shared_states_buffer)

        if buffer_len < 300:
            print(f"[JAX Thread] Accumulating real transitions ({buffer_len}/300 frames)...")
            time.sleep(2.0)
            continue

        with buffer_lock:
            real_states = jnp.array(np.array(shared_states_buffer))
            real_actions = jnp.array(np.array(shared_actions_buffer))

        # ----------------------------------------------------
        # STAGE 2: Differentiable System Identification
        # ----------------------------------------------------
        print(f"[JAX Thread] Running SysID on {real_states.shape[0]} real-world transitions...")
        learn_p = init_learnable_params(current_env_params)
        opt_st = sysid_optimizer.init(learn_p)

        # 100 gradient steps are sufficient to adapt parameters online
        for epoch in range(100):
            learn_p, opt_st, loss_val = sysid_step(learn_p, opt_st, real_states, real_actions)

        # Extract optimized parameters
        fitted_mp = float(jnp.exp(learn_p.log_m_p))
        fitted_lp = float(jnp.exp(learn_p.log_l_p))
        fitted_bp = float(jnp.exp(learn_p.log_b_p))
        fitted_barm = float(jnp.exp(learn_p.log_b_arm))
        fitted_scale = float(jnp.exp(learn_p.log_accel_scale))

        current_env_params = EnvParams(
            m_p=fitted_mp,
            l_p=fitted_lp,
            L_p=2.0 * fitted_lp,
            J_p=(fitted_mp * ((2.0 * fitted_lp) ** 2)) / 3.0,
            b_p=fitted_bp,
            b_arm=fitted_barm,
            arm_accel_scale=fitted_scale
        )

        print(f"[SysID] Updated Sim Model -> m_p: {fitted_mp:.4f} kg | b_p: {fitted_bp:.6f} | scale: {fitted_scale:.2f}")

        # ----------------------------------------------------
        # STAGE 3: Train PPO on 100 Vectorized Envs with New Sim
        # ----------------------------------------------------
        print("[JAX Thread] Training PPO on 100 vectorized GPU simulations...")
        train_fn = make_train(ppo_config, env=env, env_params=current_env_params)

        rng, subkey = jax.random.split(rng)
        out = train_fn(subkey)

        # ----------------------------------------------------
        # STAGE 4: Hot-Swap Updated Weights to CPU Inference
        # ----------------------------------------------------
        train_state = out["runner_state"][0]
        flax_params = train_state.params["params"]
        updated_torch_state = flax_to_torch_state(flax_params)

        with weights_lock:
            pending_torch_weights = updated_torch_state
            new_weights_available = True

        print(f"[JAX Thread] Staged newly adapted policy to CPU. Cycle {cycle} complete.")
        cycle += 1

    print("[JAX Thread] Terminated.")


# ==========================================
# 9. Main Execution Entry & Safe Shutdown
# ==========================================
if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)

    cpu_thread = threading.Thread(target=cpu_control_loop, daemon=True)
    jax_thread = threading.Thread(target=jax_continual_learning_loop, daemon=True)

    # Start CPU control loop first to begin accumulating transitions
    cpu_thread.start()
    jax_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Main] Shutdown signal intercepted. Halting threads...")
        stop_event.set()

        cpu_thread.join()
        jax_thread.join()

        GPIO.cleanup()
        spi_pendulum.close()
        spi_arm.close()
        print("[Main] Hardware unpinned and closed. Safe exit complete.")
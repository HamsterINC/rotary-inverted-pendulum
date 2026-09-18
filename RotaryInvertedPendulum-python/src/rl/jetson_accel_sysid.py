#!/usr/bin/env python3
"""Jetson native acceleration-mode deployment & sysid testbench.

Controls the physical stepper via direct 100 Hz PWM/GPIO integration,
logs hardware telemetry, and replays the identical waveform in MuJoCo.

Usage:
    python jetson_deploy.py --waveform all
    python jetson_deploy.py --waveform step --skip-real   # Dry-run without Jetson GPIO
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import Jetson.GPIO as GPIO
from JetsonPWM import JetsonPWM
from jetson_logger import JetsonTelemetryLogger
from pendulum_env import (
    MAX_ACCEL_RAD_S2,
    MAX_VELOCITY_RAD_S,
    RotaryInvertedPendulumEnv,
)
import spidev

# ---------------------------------------------------------------------------
# Hardware & Kinematics Configuration
# ---------------------------------------------------------------------------
# Hardware Pins
DIR_PIN = 16
EN_PIN = 22
STEP_UPDATE_PERIOD = 0.01        # 100 Hz stepper integration (10 ms)
PWM_MIN_FREQ_HZ = 10.0           # Lowest nonzero PWM frequency
ARM_SAFE_LIMIT_RAD = 1.25        # Arm travel limit (~71 deg)
ARM_RAD_PER_STEP = 0.0019634954  # 2*pi / (200 steps * 16 microsteps) ~ 3200 steps/rev
ARM_MAX_SAFE_STEPS = int(ARM_SAFE_LIMIT_RAD / ARM_RAD_PER_STEP)

STEP_DURATION_S = 3.5
CHIRP_DURATION_S = 9.0


# ---------------------------------------------------------------------------
# Global Shared Thread State
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
stop_event = threading.Event()

shared_accel_cmd = 0.0
arm_current_steps = 0
motor_vel_rad_s = 0.0
motor_target_rad = 0.0
raw_pendulum_angle_rad = 0.0

PEND_ENCODER_RESOLUTION = 16384
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
MAX_PENDULUM_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (MAX_PENDULUM_VEL_RAD_S / PEND_LSB_RAD) * STEP_UPDATE_PERIOD
PEND_ZERO_OFFSET_TICKS = 303

# ---------------------------------------------------------------------------
# SPI Encoder Configuration & State
# ---------------------------------------------------------------------------

# Thread-safe encoder tracking state
_encoder_lock = threading.Lock()
_prev_ticks = 0
_initialized_ticks = False

# Safe SPI initialization (bypassed during --skip-real)
spi_pendulum = None
def init_spi_sensor():
    global spi_pendulum, _prev_ticks, _initialized_ticks
    try:
        
        spi_pendulum = spidev.SpiDev()
        spi_pendulum.open(0, 0)
        spi_pendulum.max_speed_hz = 100_000
        spi_pendulum.mode = 1
        spi_pendulum.bits_per_word = 8
        
        # Read initial tick position
        _prev_ticks = read_raw_ticks(spi_pendulum)
        _initialized_ticks = True
        print(f"[SPI Sensor] Initialized successfully. Start ticks: {_prev_ticks}")
    except Exception as e:
        print(f"[SPI Sensor Warning] Could not open SPI bus: {e}. Running in mock/skip-real mode.")
        spi_pendulum = None


def read_raw_ticks(spi_device) -> int:
    if spi_device is None:
        return 0
    try:
        spi_device.xfer2([0xFF, 0xFF])
        data = spi_device.xfer2([0xC0, 0x00])
        raw = (data[0] << 8) | data[1]
        return raw & 0x3FFF
    except Exception:
        return 0


def process_encoder(current_ticks: int, prev_ticks: int):
    # Normalized angle calculation
    norm_angle = (current_ticks + 5000) / 8192.0
    norm_angle = (norm_angle + 1.0) % 2.0 - 1.0
    angle_rad = norm_angle * math.pi

    delta = current_ticks - prev_ticks
    # Handle 14-bit encoder rollover wrap-around if delta is huge
    if delta > 8192:
        delta -= 16384
    elif delta < -8192:
        delta += 16384

    norm_vel = delta / float(PEND_MAX_DELTA_TICKS)
    norm_vel = max(-1.0, min(1.0, norm_vel))

    # Physical pendulum velocity in rad/s
    pen_vel_rad_s = (delta * PEND_LSB_RAD) / STEP_UPDATE_PERIOD

    return angle_rad, norm_vel, pen_vel_rad_s


# ---------------------------------------------------------------------------
# Sensor Hook
# ---------------------------------------------------------------------------
def read_pendulum_sensor() -> tuple[float, float]:
    """Reads SPI encoder and returns (angle_rad, velocity_rad_s)."""
    global _prev_ticks, _initialized_ticks, spi_pendulum
    
    with _encoder_lock:
        if spi_pendulum is None:
            return 0.0, 0.0
        
        current_ticks = read_raw_ticks(spi_pendulum)
        if not _initialized_ticks:
            _prev_ticks = current_ticks
            _initialized_ticks = True

        angle_rad, _, pen_vel_rad_s = process_encoder(current_ticks, _prev_ticks)
        _prev_ticks = current_ticks
        
        return angle_rad, pen_vel_rad_s


# ---------------------------------------------------------------------------
# 100 Hz Native Stepper Integration Thread
# ---------------------------------------------------------------------------
def step_update_loop() -> None:
    """100 Hz discrete stepper kinematics and PWM generator thread."""
    global arm_current_steps, motor_vel_rad_s, motor_target_rad

    pwm_step = JetsonPWM(chip=0, channel=0)
    pwm_step.start(initial_freq_hz=PWM_MIN_FREQ_HZ)

    print("[Step Thread] 100 Hz PWM/Kinematics engine started.")
    next_tick = time.perf_counter()

    try:
        while not stop_event.is_set():
            with state_lock:
                accel_cmd = shared_accel_cmd
                current_steps = arm_current_steps

            dt = STEP_UPDATE_PERIOD

            # 1. Acceleration -> Velocity
            motor_vel_rad_s += accel_cmd * dt
            motor_vel_rad_s = max(
                -MAX_VELOCITY_RAD_S,
                min(MAX_VELOCITY_RAD_S, motor_vel_rad_s),
            )

            # 2. Boundary handling
            if motor_target_rad >= ARM_SAFE_LIMIT_RAD and motor_vel_rad_s > 0.0:
                motor_vel_rad_s = 0.0
                motor_target_rad = ARM_SAFE_LIMIT_RAD
            elif motor_target_rad <= -ARM_SAFE_LIMIT_RAD and motor_vel_rad_s < 0.0:
                motor_vel_rad_s = 0.0
                motor_target_rad = -ARM_SAFE_LIMIT_RAD

            # 3. Velocity -> Position
            motor_target_rad += motor_vel_rad_s * dt
            motor_target_rad = max(
                -ARM_SAFE_LIMIT_RAD,
                min(ARM_SAFE_LIMIT_RAD, motor_target_rad),
            )

            # 4. Position -> Target Steps
            target_pos_steps = int(round(motor_target_rad / ARM_RAD_PER_STEP))
            target_pos_steps = max(
                -ARM_MAX_SAFE_STEPS,
                min(ARM_MAX_SAFE_STEPS, target_pos_steps),
            )

            step_error = target_pos_steps - current_steps

            # 5. Target Steps -> PWM execution
            if step_error != 0:
                is_forward = step_error > 0
                GPIO.output(DIR_PIN, GPIO.HIGH if is_forward else GPIO.LOW)

                max_steps_this_update = max(
                    1,
                    int(math.floor(MAX_VELOCITY_RAD_S * dt / ARM_RAD_PER_STEP)),
                )
                actual_steps = max(
                    -max_steps_this_update,
                    min(max_steps_this_update, step_error),
                )

                new_current_steps = current_steps + actual_steps
                new_current_steps = max(
                    -ARM_MAX_SAFE_STEPS,
                    min(ARM_MAX_SAFE_STEPS, new_current_steps),
                )

                freq = abs(actual_steps) / dt
                freq = max(PWM_MIN_FREQ_HZ, freq)
                pwm_step.change_frequency(freq)

                with state_lock:
                    arm_current_steps = new_current_steps
            else:
                pwm_step.pause()
                if abs(motor_target_rad - current_steps * ARM_RAD_PER_STEP) < ARM_RAD_PER_STEP * 0.5:
                    motor_target_rad = current_steps * ARM_RAD_PER_STEP

            # 6. Timing synchronization
            next_tick += STEP_UPDATE_PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    finally:
        pwm_step.stop()
        GPIO.cleanup()
        print("[Step Thread] Stopped cleanly.")


# ---------------------------------------------------------------------------
# Test Waveforms
# ---------------------------------------------------------------------------
def waveform_step(t: float) -> float:
    """Bipolar acceleration pulses designed to bound arm position."""
    if t < 0.3: return 0.0
    if t < 0.4: return +50.0
    if t < 0.5: return -50.0
    if t < 1.0: return 0.0
    if t < 1.05: return +100.0
    if t < 1.10: return -100.0
    if t < 1.6:  return 0.0
    if t < 1.633: return +150.0
    if t < 1.667: return -150.0
    if t < 2.2:   return 0.0
    if t < 2.25:  return -100.0
    if t < 2.30:  return +100.0
    if t < 2.8:   return 0.0
    # Dynamic zero-crossing reversal
    if t < 2.85:  return +100.0
    if t < 2.95:  return -100.0
    if t < 3.00:  return +100.0
    return 0.0


def waveform_chirp(t: float) -> float:
    """Sinusoidal accel chirp sweep 0.5 Hz -> 3.0 Hz."""
    if t < 0.3: return 0.0
    s = t - 0.3
    if s > 8.0: return 0.0
    f0, f1 = 0.5, 3.0
    freq = f0 + (f1 - f0) * s / 8.0
    return 100.0 * math.sin(2.0 * math.pi * freq * s)


# ---------------------------------------------------------------------------
# Hardware Waveform Coordinator Loop
# ---------------------------------------------------------------------------
def run_hardware_profile(
    waveform_fn,
    duration: float,
    sample_rate: float,
    logger: JetsonTelemetryLogger,
) -> tuple[float, float]:
    """Feeds waveform acceleration into shared state and logs telemetry."""
    global shared_accel_cmd, arm_current_steps

    period = 1.0 / sample_rate
    n_samples = int(duration * sample_rate)

    # Reset position registers
    with state_lock:
        shared_accel_cmd = 0.0
        arm_current_steps = 0

    try:
        input("  Center motor arm, steady pendulum hanging downward, press [ENTER]...")
    except EOFError:
        time.sleep(1.0)

    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)
    # Start the 100 Hz stepper engine in background
    stop_event.clear()
    thread = threading.Thread(target=step_update_loop, daemon=True)
    thread.start()
    time.sleep(0.2)

    init_motor = 0.0
    init_pen = read_pendulum_sensor()

    t0 = time.perf_counter()
    next_tick = t0

    try:
        for i in range(n_samples):
            t_now = time.perf_counter()
            t_rel = t_now - t0

            # Calculate and inject waveform acceleration command
            accel = waveform_fn(t_rel)
            with state_lock:
                shared_accel_cmd = accel
                current_steps = arm_current_steps
                current_vel = motor_vel_rad_s

            arm_pos = current_steps * ARM_RAD_PER_STEP
            pen_pos = read_pendulum_sensor()

            logger.log(
                t_s=t_rel,
                arm_pos=arm_pos,
                arm_vel=current_vel,
                pen_pos=pen_pos,
                accel_cmd=accel,
            )

            next_tick += period
            sleep_duration = next_tick - time.perf_counter()
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    finally:
        # Zero command, allow motor to stop, signal thread termination
        with state_lock:
            shared_accel_cmd = 0.0
        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=1.0)

    return init_motor, init_pen


# ---------------------------------------------------------------------------
# MuJoCo Simulation Replay (Discrete Matching Model)
# ---------------------------------------------------------------------------
def run_sim_replay(
    waveform_fn,
    duration: float,
    sample_rate: float,
    initial_motor: float,
    initial_pen: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replays the exact command waveform through the MuJoCo simulation environment."""
    period = 1.0 / sample_rate
    n_samples = int(duration * sample_rate)

    env = RotaryInvertedPendulumEnv(
        control_freq_hz=sample_rate,
        domain_randomization=False,
        action_delay_steps=0,
        max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        motor_max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        episode_length_s=duration + 2.0,
    )
    env.reset(seed=0)

    # Initial state placement
    env.data.qpos[env._motor_qpos_addr] = float(initial_motor)
    env.data.qpos[env._pen_qpos_addr] = float(initial_pen)
    env.data.qvel[:] = 0.0
    env._motor_target = float(initial_motor)
    env._motor_vel = 0.0
    env._prev_action = 0.0
    mujoco.mj_forward(env.model, env.data)

    sim_motor = np.zeros(n_samples)
    sim_pen = np.zeros(n_samples)
    sim_t = np.arange(n_samples) * period

    sim_motor[0] = initial_motor
    sim_pen[0] = initial_pen

    for i in range(1, n_samples):
        t_sim = i * period
        accel = waveform_fn(t_sim)
        normalized_action = float(np.clip(accel / MAX_ACCEL_RAD_S2, -1.0, 1.0))

        _, _, _, _, info = env.step(np.array([normalized_action], dtype=np.float32))
        sim_motor[i] = float(info["motor_pos"])
        sim_pen[i] = float(info["phi"])

    return sim_t, sim_motor, sim_pen


# ---------------------------------------------------------------------------
# Plot & Error Statistics
# ---------------------------------------------------------------------------
def plot_and_evaluate(
    telemetry: dict[str, np.ndarray],
    sim_t: np.ndarray,
    sim_m: np.ndarray,
    sim_p: np.ndarray,
    tag: str,
    out_dir: Path,
) -> None:
    t_real = telemetry["time_s"]
    accel_cmd = telemetry["control_action"]
    m_real = telemetry["arm_pos_rad"]
    p_real = telemetry["pendulum_angle_unwrapped"]

    common_len = min(len(t_real), len(sim_t))
    motor_rmse = float(np.sqrt(np.mean((sim_m[:common_len] - m_real[:common_len]) ** 2)))
    
    pen_diff = (sim_p[:common_len] - p_real[:common_len] + np.pi) % (2.0 * np.pi) - np.pi
    pen_rmse = float(np.sqrt(np.mean(pen_diff**2)))

    print(f"\n  === Alignment Statistics ({tag}) ===")
    print(f"  Motor Tracking RMSE:    {motor_rmse:.4f} rad ({math.degrees(motor_rmse):.2f}°)")
    print(f"  Pendulum Wrapped RMSE:  {pen_rmse:.4f} rad ({math.degrees(pen_rmse):.2f}°)")
    print(f"  Max Real Motor Speed:   {np.max(np.abs(telemetry['arm_vel_rad_s'])):.3f} rad/s")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_real, accel_cmd, "k-", label="Commanded Acceleration")
    axes[0].set_ylabel("Accel (rad/s²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t_real, m_real, "C0-", label="Jetson Stepper (100 Hz)", linewidth=1.2)
    axes[1].plot(sim_t, sim_m, "C3--", label="MuJoCo Simulation Replay", linewidth=1.2)
    axes[1].set_ylabel("Motor Pos (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t_real, p_real, "C0-", label="Real Pendulum (Unwrapped)", linewidth=1.2)
    axes[2].plot(sim_t, sim_p, "C3--", label="MuJoCo Pendulum Angle", linewidth=1.2)
    axes[2].set_ylabel("Pendulum Angle (rad)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.suptitle(f"Jetson Hardware vs MuJoCo Sim-to-Real: {tag}")
    plt.tight_layout()

    out_png = out_dir / f"{tag}_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot written to: {out_png}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson native acceleration-mode deployment & sysid.")
    parser.add_argument("--waveform", choices=["step", "chirp", "all"], default="all")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sampling frequency (default: 100 Hz)")
    parser.add_argument("--out-dir", default="/tmp", help="Output directory for telemetry & plots")
    parser.add_argument("--skip-real", action="store_true", help="Dry run without physical PWM/GPIO")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles: list[tuple[str, Any, float]] = []
    if args.waveform in ("step", "all"):
        profiles.append(("step", waveform_step, STEP_DURATION_S))
    if args.waveform in ("chirp", "all"):
        profiles.append(("chirp", waveform_chirp, CHIRP_DURATION_S))

    def handle_sigint(signum, frame):
        print("\n[Interrupt] Terminating execution...")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    for tag, wave_fn, duration in profiles:
        print(f"\n================ Running Profile: {tag} ({duration} s) ================")
        logger = JetsonTelemetryLogger(output_dir=out_dir, tag=f"jetson_accel_{tag}")

        if args.skip_real:
            print("  [DRY RUN] Synthesizing zero hardware motion...")
            init_m, init_p = 0.0, 0.0
            n_samples = int(duration * args.sample_rate)
            for i in range(n_samples):
                t = i / args.sample_rate
                logger.log(t, 0.0, 0.0, 0.0, wave_fn(t))
        else:
            init_m, init_p = run_hardware_profile(wave_fn, duration, args.sample_rate, logger)

        print("  Replaying trajectory in MuJoCo physics engine...")
        sim_t, sim_m, sim_p = run_sim_replay(wave_fn, duration, args.sample_rate, init_m, init_p)

        csv_path = logger.write_csv()
        npz_path = logger.write_npz(sim_motor=sim_m, sim_pen=sim_p)
        print(f"  Logs saved:\n    CSV: {csv_path}\n    NPZ: {npz_path}")

        plot_and_evaluate(logger.to_dataframe_dict(), sim_t, sim_m, sim_p, tag, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
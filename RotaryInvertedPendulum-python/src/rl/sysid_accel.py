"""Characterise the motor + pendulum response to accel-mode commands.

Drives `LowLevelServer.set_acceleration()` on the real rig with a programmed
accel waveform, logs motor and pendulum state at high rate, then replays the
SAME waveform through `pendulum_env` and produces a side-by-side comparison.

This is the accel-mode analogue of the old `sysid_record.py chirp` test
which used position commands. The point is to validate the sim's accel-mode
integration against the real rig's `moveByAcceleration` behaviour — sim_vs_real
on the latest deployment log showed sim pendulum spinning while real pendulum
balances under identical actions, so something in the sim model disagrees.

Two waveforms:
    `step`  short bipolar accel pulses at amplitudes 50, 100, 150 rad/s² plus
            a zero-crossing reversal. Each pulse pair returns velocity to ~0
            so the motor stays bounded near its start position.
    `chirp` sinusoidal accel sweep 0.5 → 3 Hz at amplitude 100 rad/s². Tests
            the frequency-resolved coupling — analogous to the position-mode
            chirp from `sysid_record.py`.

Usage:
    python sysid_accel.py --port /dev/cu.usbserial-1130 --waveform all
    python sysid_accel.py --waveform step --skip-real   # sim-only dry run

Outputs:
    /tmp/sysid_accel_<waveform>.npz  raw recording + sim replay
    /tmp/sysid_accel_<waveform>.png  comparison plot

The pendulum is free to swing during the test. If you want to isolate pure
motor dynamics from pendulum coupling, hold the pendulum still by hand.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

from lowlevel_client import LowLevelClient
from pendulum_env import RotaryInvertedPendulumEnv, MAX_ACCEL_RAD_S2, MAX_VELOCITY_RAD_S


SAMPLE_RATE_HZ = 200.0


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------

def waveform_step(t: float) -> float:
    """Bipolar accel pulses; each pair brings velocity back to ~0.

    With MAX_VELOCITY=5 rad/s and given amplitudes, each "+A for T" reaches
    the velocity cap exactly at the end of T; the matched "-A for T" brings
    velocity back to 0. Motor position drift bounded to ~0.5 rad per pulse.
    """
    # Settle
    if t < 0.3: return 0.0
    # Pulse 1: amplitude 50, T=0.1s (50*0.1 = 5 rad/s reached at cap)
    if t < 0.4: return +50.0
    if t < 0.55: return -50.0
    if t < 1.0: return 0.0
    # Pulse 2: amplitude 100, T=0.05s
    if t < 1.10: return +100.0
    if t < 1.15: return -100.0
    if t < 1.6:  return 0.0
    # Pulse 3: amplitude 150, T=0.033s
    if t < 1.633: return +150.0
    if t < 1.667: return -150.0
    if t < 2.2:   return 0.0
    # Pulse 4: negative-side first
    if t < 2.25:  return -100.0
    if t < 2.30:  return +100.0
    if t < 2.8:   return 0.0
    # Zero-crossing reversal: accelerate to +cap, reverse through zero
    # to -cap, then decel to 0. This is the test that exposes whether
    # moveByAcceleration(..., allow_reverse=true) actually does what we
    # think and whether sim's integration matches.
    if t < 2.85:  return +100.0   # 0 → +5
    if t < 2.95:  return -100.0   # +5 → -5 (through zero)
    if t < 3.00:  return +100.0   # -5 → 0
    return 0.0


STEP_DURATION_S = 3.5


def waveform_chirp(t: float) -> float:
    """Sinusoidal accel sweep 0.5 → 3 Hz over 8 s at A=100 rad/s²."""
    if t < 0.3: return 0.0
    s = t - 0.3
    if s > 8.0: return 0.0
    f0, f1 = 0.5, 3.0
    f = f0 + (f1 - f0) * s / 8.0
    return 100.0 * math.sin(2.0 * math.pi * f * s)


CHIRP_DURATION_S = 9.0


# ---------------------------------------------------------------------------
# Recording on real rig
# ---------------------------------------------------------------------------

def run_real(port: str, baud: int, waveform_fn, duration: float,
             sample_rate: float):
    n = int(duration * sample_rate)
    period = 1.0 / sample_rate

    log_t = np.zeros(n)
    log_accel = np.zeros(n)
    log_motor = np.zeros(n)
    log_pen = np.zeros(n)

    with LowLevelClient(port, baud=baud) as client:
        if not client.wait_until_ready():
            raise RuntimeError("LowLevelServer did not respond.")
        # Motor is disengaged (LowLevelServer boots that way). Prompt the
        # user to centre the motor and steady the pendulum before each
        # waveform; press Enter to engage and begin.
        try:
            input("  Centre the motor by hand, steady the pendulum, then press Enter to engage and start...")
        except EOFError:
            print("  (no stdin; engaging in 2 s)")
            time.sleep(2.0)
        client.set_acceleration(0.0)
        client.engage_motor()
        client.tare_pendulum()
        time.sleep(0.5)

        s0 = client.get_state()
        t0 = s0.time_us * 1e-6
        initial_motor = -s0.motor_pos_rad   # sim-frame
        initial_pen = -s0.pendulum_pos_rad  # sim-frame
        print(f"  initial: motor={initial_motor:+.3f} rad, pendulum={initial_pen:+.3f} rad")

        next_tick = time.monotonic()
        overruns = 0
        for i in range(n):
            t_wf = i * period
            accel = waveform_fn(t_wf)
            client.set_acceleration(accel)
            s = client.get_state()
            log_t[i] = s.time_us * 1e-6 - t0
            log_accel[i] = accel
            log_motor[i] = -s.motor_pos_rad
            log_pen[i] = -s.pendulum_pos_rad

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                overruns += 1

        client.set_acceleration(0.0)
        time.sleep(0.3)
        client.disengage_motor()

    if overruns:
        print(f"  WARNING: {overruns}/{n} loop ticks overran the {1000*period:.1f} ms budget")
    return log_t, log_accel, log_motor, log_pen, initial_motor, initial_pen


# ---------------------------------------------------------------------------
# Replay in sim
# ---------------------------------------------------------------------------

def run_sim(waveform_fn, duration: float, sample_rate: float,
            initial_motor: float, initial_pen: float):
    n = int(duration * sample_rate)
    period = 1.0 / sample_rate

    env = RotaryInvertedPendulumEnv(
        control_freq_hz=sample_rate,
        domain_randomization=False,
        action_delay_steps=0,
        max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        # Use the nominal motor envelope (no per-episode DR sampling here).
        motor_max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        episode_length_s=duration + 1.0,
    )
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = float(initial_motor)
    env.data.qpos[env._pen_qpos_addr] = float(initial_pen) 
    env.data.qvel[:] = 0.0
    env._motor_target = float(initial_motor)
    env._motor_vel = 0.0
    env._prev_action = 0.0
    mujoco.mj_forward(env.model, env.data)

    log_t = np.arange(n) * period
    log_accel = np.zeros(n)
    log_motor = np.zeros(n)
    log_pen = np.zeros(n)
    log_motor[0] = initial_motor
    log_pen[0] = initial_pen

    for i in range(1, n):
        t_wf = i * period
        accel = waveform_fn(t_wf)
        action = float(np.clip(accel / MAX_ACCEL_RAD_S2, -1.0, 1.0))
        _, _, _, _, info = env.step(np.array([action], dtype=np.float32))
        log_accel[i] = accel
        log_motor[i] = float(info["motor_pos"])
        log_pen[i] = float(info["phi"])

    return log_t, log_accel, log_motor, log_pen


# ---------------------------------------------------------------------------
# Plot + stats
# ---------------------------------------------------------------------------

def plot_comparison(real, sim, name: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_real, accel_real, m_real, p_real = real
    t_sim, accel_sim, m_sim, p_sim = sim

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t_real, accel_real, "k", label="commanded accel (rad/s²)")
    axes[0].set_ylabel("accel cmd (rad/s²)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_real, m_real, "C0", label="real motor_pos", linewidth=1.0)
    axes[1].plot(t_sim, m_sim, "C3", label="sim motor_pos", linewidth=1.0)
    axes[1].set_ylabel("motor_pos (rad)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_real, p_real, "C0", label="real pendulum_pos", linewidth=1.0)
    axes[2].plot(t_sim, p_sim, "C3", label="sim pendulum_pos", linewidth=1.0)
    axes[2].set_ylabel("pendulum_pos (rad)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Accel-mode sysid: {name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"  plot saved to {out_path}")


def print_stats(real, sim, name: str):
    _, _, m_real, p_real = real
    _, _, m_sim, p_sim = sim
    common = min(len(m_real), len(m_sim))

    motor_err = m_sim[:common] - m_real[:common]
    motor_rmse = float(np.sqrt(np.mean(motor_err**2)))

    pen_err = p_sim[:common] - p_real[:common]
    pen_wrap = (pen_err + np.pi) % (2 * np.pi) - np.pi
    pen_wrap_rmse = float(np.sqrt(np.mean(pen_wrap**2)))

    motor_real_range = float(m_real.max() - m_real.min())
    motor_sim_range = float(m_sim.max() - m_sim.min())

    # Motor velocity (numerical diff): peak achieved velocity
    dt_real = float(np.median(np.diff(real[0]))) if len(real[0]) > 1 else 1.0
    dt_sim = float(np.median(np.diff(sim[0]))) if len(sim[0]) > 1 else 1.0
    v_real = np.diff(m_real) / dt_real if dt_real > 0 else np.zeros_like(m_real)
    v_sim = np.diff(m_sim) / dt_sim if dt_sim > 0 else np.zeros_like(m_sim)

    print(f"  ── {name} stats ──")
    print(f"  motor_pos RMSE              : {motor_rmse:.4f} rad ({math.degrees(motor_rmse):.2f}°)")
    print(f"  pendulum_pos wrapped RMSE   : {pen_wrap_rmse:.4f} rad ({math.degrees(pen_wrap_rmse):.2f}°)")
    print(f"  motor range  real / sim     : {motor_real_range:.3f} / {motor_sim_range:.3f} rad")
    print(f"  motor peak |velocity| real  : {float(np.max(np.abs(v_real))):.3f} rad/s")
    print(f"  motor peak |velocity| sim   : {float(np.max(np.abs(v_sim))):.3f} rad/s")
    print(f"  velocity-cap target          : {MAX_VELOCITY_RAD_S:.1f} rad/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Accel-mode sysid: real-rig + sim replay")
    p.add_argument("--port", default="/dev/cu.usbserial-1130",
                   help="serial port for the LowLevelServer-flashed Arduino")
    p.add_argument("--baud", type=int, default=2_000_000)
    p.add_argument("--waveform", choices=["step", "chirp", "all"], default="all")
    p.add_argument("--sample-rate", type=float, default=SAMPLE_RATE_HZ,
                   help="state-sampling rate in Hz (max ~200 over 2 Mbaud)")
    p.add_argument("--skip-real", action="store_true",
                   help="skip real-rig recording; sim only (for testing the script)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    todo = []
    if args.waveform in ("step", "all"):
        todo.append(("step",  waveform_step,  STEP_DURATION_S))
    if args.waveform in ("chirp", "all"):
        todo.append(("chirp", waveform_chirp, CHIRP_DURATION_S))

    for name, fn, dur in todo:
        print(f"\n=== {name} ({dur:.1f}s, {args.sample_rate:.0f} Hz sampling) ===")

        if args.skip_real:
            print("  --skip-real: synthesising real_data = zero state")
            n = int(dur * args.sample_rate)
            t_arr = np.arange(n) / args.sample_rate
            accel = np.array([fn(ti) for ti in t_arr])
            real = (t_arr, accel, np.zeros(n), np.zeros(n))
            initial_motor = 0.0
            initial_pen = 0.0
        else:
            real_t, real_accel, real_m, real_p, initial_motor, initial_pen = run_real(
                args.port, args.baud, fn, dur, args.sample_rate,
            )
            real = (real_t, real_accel, real_m, real_p)
            print(f"  collected {len(real_t)} samples; "
                  f"motor range [{real_m.min():+.3f}, {real_m.max():+.3f}], "
                  f"pen range [{real_p.min():+.3f}, {real_p.max():+.3f}]")

        print(f"  replaying in sim (initial motor={initial_motor:+.3f}, pen={initial_pen:+.3f})...")
        sim = run_sim(fn, dur, args.sample_rate, initial_motor, initial_pen)

        out_npz = f"/tmp/sysid_accel_{name}.npz"
        np.savez(out_npz,
                 t=real[0], accel_cmd=real[1],
                 real_motor=real[2], real_pen=real[3],
                 sim_motor=sim[2], sim_pen=sim[3],
                 waveform=name)
        print(f"  saved {out_npz}")

        out_png = f"/tmp/sysid_accel_{name}.png"
        plot_comparison(real, sim, name, out_png)
        print_stats(real, sim, name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay a real-hardware action sequence in the sim and compare trajectories.

Loads a `run_policy.py --log` .npz, then steps the same actions through the
sim env starting from the real initial state. Plots motor_pos and pendulum
trajectories side-by-side. Differences pinpoint sim-to-real model gaps that
domain randomisation should be widened to cover.

Usage:
    python sim_vs_real.py /tmp/policy_run.npz [--out cmp.png]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from Simulation.pendulum_env import RotaryInvertedPendulumEnv


def replay_in_sim(env: RotaryInvertedPendulumEnv, *,
                  initial_motor: float, initial_phi: float,
                  actions: np.ndarray) -> dict:
    """Step `actions` through `env`, starting from the given joint state."""
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = initial_motor
    env.data.qpos[env._pen_qpos_addr] = initial_phi
    env.data.qvel[:] = 0.0
    env._motor_target = initial_motor
    env._motor_applied = initial_motor
    mujoco.mj_forward(env.model, env.data)

    n = len(actions)
    motor_pos = np.zeros(n)
    pen_pos = np.zeros(n)
    motor_target = np.zeros(n)

    for i, a in enumerate(actions):
        obs, r, term, trunc, info = env.step(np.array([float(a)], dtype=np.float32))
        motor_pos[i] = info["motor_pos"]
        pen_pos[i] = info["phi"]
        motor_target[i] = info["motor_target"]
        if term:
            break

    return {
        "motor_pos": motor_pos[:i + 1],
        "pen_pos": pen_pos[:i + 1],
        "motor_target": motor_target[:i + 1],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replay logged actions in sim and compare")
    p.add_argument("log", help=".npz from run_policy.py --log")
    p.add_argument("--out", default=None, help="optional matplotlib figure path")
    p.add_argument("--no-plot", action="store_true",
                   help="skip plotting; just print summary statistics")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    d = dict(np.load(args.log, allow_pickle=True))
    n = len(d["time_us"])
    control_freq = float(d["control_freq_hz"])
    dt_s = 1.0 / control_freq
    actions = d["action"].astype(np.float32)
    real_motor = d["motor_pos_rad"].astype(np.float64)
    real_pen = d["pendulum_pos_rad"].astype(np.float64)
    # Position-mode logs have motor_target_rad (commanded position);
    # accel-mode logs have accel_cmd_rad_s2 (commanded acceleration).
    if "motor_target_rad" in d:
        real_cmd = d["motor_target_rad"].astype(np.float64)
        cmd_label = "commanded target (rad)"
    elif "accel_cmd_rad_s2" in d:
        real_cmd = d["accel_cmd_rad_s2"].astype(np.float64)
        cmd_label = "commanded accel (rad/s²)"
    else:
        real_cmd = None
        cmd_label = ""

    print(f"Replaying {n} actions ({n * dt_s:.2f}s) in sim ...")

    env = RotaryInvertedPendulumEnv(
        control_freq_hz=control_freq,
        episode_length_s=n * dt_s + 1.0,  # plenty
    )
    sim = replay_in_sim(
        env,
        initial_motor=float(real_motor[0]),
        initial_phi=float(real_pen[0]),
        actions=actions,
    )
    sim_motor = sim["motor_pos"]
    sim_pen = sim["pen_pos"]

    common_n = min(len(sim_motor), len(real_motor))
    motor_err = sim_motor[:common_n] - real_motor[:common_n]
    pen_err = sim_pen[:common_n] - real_pen[:common_n]

    print(f"  motor_pos rmse:    {float(np.sqrt(np.mean(motor_err**2))):.4f} rad")
    print(f"  motor_pos max abs err: {float(np.max(np.abs(motor_err))):.4f} rad")
    print(f"  pendulum_pos rmse: {float(np.sqrt(np.mean(pen_err**2))):.4f} rad")
    print(f"  pendulum_pos max abs err: {float(np.max(np.abs(pen_err))):.4f} rad")

    # Lag estimate: how many steps to align sim_motor with real_motor
    # via cross-correlation
    from scipy.signal import correlate
    if np.std(sim_motor) > 1e-6 and np.std(real_motor[:common_n]) > 1e-6:
        x = sim_motor - sim_motor.mean()
        y = real_motor[:common_n] - real_motor[:common_n].mean()
        c = correlate(y, x, mode="full")
        centre = len(x) - 1
        win = slice(max(0, centre - 10), min(len(c), centre + 10))
        lag = int(np.argmax(c[win]) + win.start - centre)
        print(f"  motor cross-corr lag (sim -> real): {lag} steps ({lag * dt_s * 1000:+.1f} ms)")

    if args.no_plot:
        return 0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return 0

    t = np.arange(common_n) * dt_s
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, real_motor[:common_n], "C0", label="real motor_pos")
    axes[0].plot(t, sim_motor[:common_n], "C3", label="sim motor_pos")
    if real_cmd is not None and cmd_label.startswith("commanded target"):
        # Position-mode: command and motor_pos share units (rad), can overlay.
        axes[0].plot(t, real_cmd[:common_n], "k--", label=cmd_label, alpha=0.5)
    axes[0].set_ylabel("motor_pos (rad)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, real_pen[:common_n], "C0", label="real pendulum_pos")
    axes[1].plot(t, sim_pen[:common_n], "C3", label="sim pendulum_pos")
    axes[1].set_ylabel("pendulum_pos (rad)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, actions[:common_n], "C2", label="action")
    axes[2].set_ylabel("action")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    out = args.out or str(Path(args.log).with_suffix(".png"))
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"plot saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

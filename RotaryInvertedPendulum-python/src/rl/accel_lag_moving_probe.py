"""Measure FastAccelStepper accel-mode lag from a *moving* baseline.

Sister script to `accel_step_probe.py`. That one fired step commands
from rest, but it conflated three different lag sources:

  - First-step latency (physics: from v=0, the motor can't step until
    ½·A·t² ≥ 1 step). At our resolution this is 12-28 ms depending on A.
  - Velocity-estimator window saturation (firmware's 10-sample
    regression window holds 18 ms of zeros until they all roll out).
  - The actual firmware scheduling lag we care about.

This script does a slope-change test instead:
  Phase 1: command A=A_base for T_base seconds. Motor reaches a steady
           velocity ramp at slope A_base (no first-step issue once
           motion has begun).
  Phase 2: switch to A=A_step. Sample tight for probe_duration seconds.

The pre- and post-switch velocity traces are each linear with slopes
≈ A_base and ≈ A_step. We fit a line to each, find their intersection,
and the lag τ is (intersection_time - command_send_time). No
assumptions about pure-delay vs first-order shape — works for both.

Usage:
  python accel_lag_moving_probe.py --port /dev/cu.usbserial-1130
  python accel_lag_moving_probe.py --port ... --steps 40 80 -20 -40
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np


FPGA = True
if FPGA: 
    from FPGA_lowlevel_client import FpgaLowLevelClient as LowLevelClient
else:
    from lowlevel_client import LowLevelClient


SAFETY_POS_RAD = 1.5  # ~86° — abort if motor crosses this


def probe_step_change(
    client: LowLevelClient,
    baseline_accel: float,
    baseline_duration_s: float,
    step_accel: float,
    probe_duration_s: float,
) -> dict:
    """Apply baseline_accel for baseline_duration_s, then switch to step_accel.

    Returns a dict with `t_rel_s` (Arduino-clock, zero at baseline start),
    `t_switch_s` (Arduino-clock moment the step_accel command was sent),
    motor_vel/pos arrays.
    """
    # Ensure starting from rest. Brake any residual velocity.
    for _ in range(50):
        s = client.get_state()
        if abs(s.motor_vel_rad_s) < 0.1:
            break
        client.set_acceleration(-np.sign(-s.motor_vel_rad_s) * 20.0)
        time.sleep(0.01)
    client.set_acceleration(0.0)
    time.sleep(0.3)

    # Baseline arduino time for relative timestamps.
    s0 = client.get_state()
    t_us0 = s0.time_us
    pos0 = -s0.motor_pos_rad

    n_max = 3000
    t_us = np.zeros(n_max, dtype=np.int64)
    pos = np.zeros(n_max, dtype=np.float64)
    vel = np.zeros(n_max, dtype=np.float64)
    n = 0
    aborted = False

    # --- Phase 1: baseline accel ---
    client.set_acceleration(baseline_accel)

    while n < n_max:
        s = client.get_state()
        t_us[n] = s.time_us
        pos[n] = -s.motor_pos_rad
        vel[n] = -s.motor_vel_rad_s
        n += 1
        t_rel = (s.time_us - t_us0) * 1e-6
        if t_rel >= baseline_duration_s:
            break
        if abs(pos[n - 1] - pos0) > SAFETY_POS_RAD:
            aborted = True
            break

    if aborted:
        client.set_acceleration(-np.sign(baseline_accel) * abs(baseline_accel))
        return dict(aborted=True, accel_base=baseline_accel,
                    accel_step=step_accel, t_rel_s=(t_us[:n] - t_us0) * 1e-6,
                    motor_pos_rad=pos[:n], motor_vel_rad_s=vel[:n],
                    t_switch_s=float("nan"))

    # --- Phase 2: step change ---
    # Capture the Arduino time of the most recent sample as our reference for
    # "when the step command was about to be sent". This is the cleanest
    # firmware-clock anchor.
    t_switch_us = t_us[n - 1]
    client.set_acceleration(step_accel)

    probe_end_us = t_switch_us + int(probe_duration_s * 1e6)
    while n < n_max:
        s = client.get_state()
        t_us[n] = s.time_us
        pos[n] = -s.motor_pos_rad
        vel[n] = -s.motor_vel_rad_s
        n += 1
        if s.time_us > probe_end_us:
            break
        if abs(pos[n - 1] - pos0) > SAFETY_POS_RAD:
            aborted = True
            break

    # --- Brake: command opposite of current velocity until ≈ 0. Brake
    # magnitude matches the largest in-trial accel so we always have
    # enough authority to stop within the safety window.
    brake_mag = max(abs(baseline_accel), abs(step_accel))
    for _ in range(200):
        s = client.get_state()
        v = -s.motor_vel_rad_s
        if abs(v) < 0.1:
            break
        client.set_acceleration(-np.sign(v) * brake_mag)
        time.sleep(0.005)
    client.set_acceleration(0.0)

    # --- Re-center back near pos0 (gentle PD) ---
    for _ in range(200):
        s = client.get_state()
        err = pos0 - (-s.motor_pos_rad)
        v = -s.motor_vel_rad_s
        if abs(err) < 0.05 and abs(v) < 0.2:
            break
        a = 20.0 * err - 8.0 * v
        a = max(-30.0, min(30.0, a))
        client.set_acceleration(a)
        time.sleep(0.02)
    client.set_acceleration(0.0)
    time.sleep(0.2)

    t_us = t_us[:n]
    pos = pos[:n]
    vel = vel[:n]
    t_rel = (t_us - t_us0) * 1e-6
    t_switch_s = (t_switch_us - t_us0) * 1e-6

    return dict(
        aborted=aborted,
        accel_base=baseline_accel,
        accel_step=step_accel,
        t_rel_s=t_rel,
        t_switch_s=t_switch_s,
        motor_pos_rad=pos,
        motor_vel_rad_s=vel,
        n_samples=n,
    )


def fit_knee(trial: dict, *, fit_skip_pre_ms: float = 30.0,
             fit_skip_post_ms: float = 25.0) -> dict:
    """Fit two lines (pre-switch, post-switch) to motor_vel(t), return knee.

    `fit_skip_pre_ms` excludes the very early baseline region (motor still
    in first-step physics). `fit_skip_post_ms` excludes the transition
    region right after the switch (where slope is changing).
    Lag τ = (knee_time - t_switch_s).
    """
    t = trial["t_rel_s"]
    v = trial["motor_vel_rad_s"]
    t_sw = trial["t_switch_s"]

    pre_mask = (t < t_sw) & (t > fit_skip_pre_ms * 1e-3)
    post_mask = (t > t_sw + fit_skip_post_ms * 1e-3)

    if pre_mask.sum() < 4 or post_mask.sum() < 4:
        return dict(slope_pre=float("nan"), slope_post=float("nan"),
                    knee_t_s=float("nan"), lag_s=float("nan"),
                    n_pre=int(pre_mask.sum()), n_post=int(post_mask.sum()))

    m_pre, c_pre = np.polyfit(t[pre_mask], v[pre_mask], 1)
    m_post, c_post = np.polyfit(t[post_mask], v[post_mask], 1)

    # Lines intersect where m_pre·t + c_pre = m_post·t + c_post.
    if abs(m_pre - m_post) < 1e-6:
        knee_t = float("nan")
    else:
        knee_t = (c_post - c_pre) / (m_pre - m_post)

    return dict(
        slope_pre=float(m_pre),
        slope_post=float(m_post),
        knee_t_s=float(knee_t),
        lag_s=float(knee_t - t_sw) if np.isfinite(knee_t) else float("nan"),
        n_pre=int(pre_mask.sum()),
        n_post=int(post_mask.sum()),
    )


def plot_trials(trials: list[dict], fits: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(trials)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, trial, fit in zip(axes, trials, fits):
        if trial.get("aborted"):
            ax.set_title(f"A_base={trial['accel_base']} → A_step={trial['accel_step']}  ABORTED")
            ax.plot(trial["t_rel_s"] * 1000, trial["motor_vel_rad_s"], ".-", ms=2)
            continue

        t_ms = trial["t_rel_s"] * 1000
        v = trial["motor_vel_rad_s"]
        t_sw_ms = trial["t_switch_s"] * 1000

        ax.plot(t_ms, v, "o", ms=2, alpha=0.6, label="measured motor_vel")
        ax.axvline(t_sw_ms, color="k", ls="--", alpha=0.6, label="cmd sent")

        # Fitted lines
        m_pre = fit["slope_pre"]; m_post = fit["slope_post"]
        if np.isfinite(m_pre) and np.isfinite(m_post):
            t_dense = np.linspace(t_ms.min(), t_ms.max(), 200) * 1e-3
            v_pre_line = m_pre * t_dense + np.polyfit(
                trial["t_rel_s"][(trial["t_rel_s"] < trial["t_switch_s"])],
                v[(trial["t_rel_s"] < trial["t_switch_s"])], 1)[1]
            v_post_line = m_post * t_dense + np.polyfit(
                trial["t_rel_s"][(trial["t_rel_s"] > trial["t_switch_s"] + 0.025)],
                v[(trial["t_rel_s"] > trial["t_switch_s"] + 0.025)], 1)[1]
            ax.plot(t_dense * 1000, v_pre_line, "C2-", lw=1, alpha=0.7,
                    label=f"pre fit (m={m_pre:.1f})")
            ax.plot(t_dense * 1000, v_post_line, "C3-", lw=1, alpha=0.7,
                    label=f"post fit (m={m_post:.1f})")
            knee_t_ms = fit["knee_t_s"] * 1000
            ax.axvline(knee_t_ms, color="C3", ls=":", alpha=0.7,
                       label=f"knee (τ={fit['lag_s']*1000:.1f} ms)")

        ax.set_ylabel("motor_vel (rad/s)")
        ax.set_xlabel("t (ms, Arduino-clock)")
        ax.set_title(f"A_base={trial['accel_base']:g} → A_step={trial['accel_step']:g}  "
                     f"({trial['n_samples']} samples)", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Accel-mode lag — slope-change probe", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=False, help="Arduino serial port")
    p.add_argument("--baseline-accel", type=float, default=20.0,
                   help="Pre-accel value (rad/s²). Default 20.")
    p.add_argument("--baseline-duration", type=float, default=0.15,
                   help="Duration of pre-accel before switch (s). Default 0.15.")
    p.add_argument("--probe-duration", type=float, default=0.08,
                   help="Duration to sample after switch (s). Default 0.08. "
                        "(Kept short so the post-switch motion + brake stays "
                        "inside the ±86° safety window.)")
    p.add_argument("--steps", type=float, nargs="+",
                   default=[-10.0, -30.0, -50.0, -80.0],
                   help="Step-change accel targets (rad/s²). Defaults are "
                        "deceleration/reversal trials only — positive "
                        "step-ups quickly saturate the motor at "
                        "MOTOR_MIN_STEP_US's max-speed cap (~5 rad/s) and "
                        "the post-switch slope fit becomes meaningless. "
                        "Negative steps from a positive baseline give "
                        "clean, saturation-free measurements.")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args(argv)

    if args.out_dir:
        out = Path(args.out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path(__file__).resolve().parent / "accel_lag_moving_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    trials = []
    fits = []
    with LowLevelClient(port=args.port) as client:
        if not client.wait_until_ready():
            print("Arduino not responding", file=sys.stderr)
            return 1
        client.engage_motor()
        time.sleep(0.3)

        for i, A_step in enumerate(args.steps):
            print(f"\n[{i + 1}/{len(args.steps)}] "
                  f"A_base={args.baseline_accel:g} → A_step={A_step:g}")
            trial = probe_step_change(
                client,
                baseline_accel=args.baseline_accel,
                baseline_duration_s=args.baseline_duration,
                step_accel=A_step,
                probe_duration_s=args.probe_duration,
            )
            fit = fit_knee(trial)
            trials.append(trial)
            fits.append(fit)

            if trial.get("aborted"):
                print("  [safety abort]")
                continue

            span = trial["t_rel_s"][-1] - trial["t_rel_s"][0]
            hz = (trial["n_samples"] - 1) / span if span > 0 else 0
            print(f"  samples: {trial['n_samples']}  ({hz:.0f} Hz effective)")
            print(f"  slope pre: {fit['slope_pre']:.2f}  "
                  f"(expected {args.baseline_accel:g})")
            print(f"  slope post: {fit['slope_post']:.2f}  "
                  f"(expected {A_step:g})")
            print(f"  lag τ: {fit['lag_s'] * 1000:.2f} ms  "
                  f"(n_pre={fit['n_pre']}, n_post={fit['n_post']})")

            np.savez(out / f"trial_step{A_step:g}.npz", **trial, **{
                f"fit_{k}": v for k, v in fit.items()
            })

        client.set_acceleration(0.0)
        client.disengage_motor()

    plot_path = out / "step_change_response.png"
    plot_trials(trials, fits, plot_path)
    print(f"\nPlot: {plot_path}")

    print("\n" + "─" * 70)
    print(f"{'A_base':>8}  {'A_step':>8}  {'m_pre':>8}  {'m_post':>8}  {'τ (ms)':>10}")
    print("─" * 70)
    for trial, fit in zip(trials, fits):
        if trial.get("aborted"):
            print(f"{trial['accel_base']:>8.1f}  {trial['accel_step']:>8.1f}  "
                  f"     [aborted]")
            continue
        print(f"{trial['accel_base']:>8.1f}  {trial['accel_step']:>8.1f}  "
              f"{fit['slope_pre']:>8.2f}  {fit['slope_post']:>8.2f}  "
              f"{fit['lag_s'] * 1000:>10.2f}")
    print("─" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

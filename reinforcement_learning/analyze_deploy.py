"""Analyse a deploy log from `run_policy.py --log` for policy quality.

Reports balance metrics and motor/action chatter, then renders a 4-panel
plot of a zoomed steady-state window. Useful for:

  - Comparing two policies on the same rig (chatter regression test).
  - Deciding whether the policy is in the calm vs active attractor.
  - Spotting θ-bias (motor mean offset during balance).

Differs from `analyze_run.py`, which fits motor first-order lag from a
position-mode log. This script characterises the *policy's* behaviour
in steady-state balance — it doesn't assume a position-mode `motor_target`.

Usage:
    python analyze_deploy.py /tmp/2026-05-20_16-31_75hz.npz
    python analyze_deploy.py /tmp/foo.npz --window 5 9     # zoom window in seconds
    python analyze_deploy.py /tmp/a.npz /tmp/b.npz         # side-by-side compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_log(path: str | Path) -> dict:
    """Load deploy npz. Normalises into a dict of arrays + scalars."""
    d = dict(np.load(path, allow_pickle=True))
    # time_us → relative seconds
    t = (d["time_us"] - d["time_us"][0]) / 1e6
    return dict(
        path=str(path),
        t=t,
        motor_pos=d["motor_pos_rad"].astype(np.float64),
        motor_vel=d["motor_vel_rad_s"].astype(np.float64),
        pendulum_pos=d["pendulum_pos_rad"].astype(np.float64),
        pendulum_vel=d["pendulum_vel_rad_s"].astype(np.float64),
        accel_cmd=d["accel_cmd_rad_s2"].astype(np.float64),
        action=d["action"].astype(np.float64) if "action" in d else None,
        control_freq_hz=float(d["control_freq_hz"]),
        max_accel=float(d["max_accel_rad_s2"]),
    )


def _theta_to_upright(pendulum_pos: np.ndarray) -> np.ndarray:
    """Map raw firmware angle (0 = hang down, ±π = upright) to error vs upright.

    Returns angles in [-π, π] where 0 means upright. Uses the convention that
    raw position near ±π is upright and raw 0 is hanging.
    """
    # Wrap (pendulum_pos - π) into [-π, π].
    e = (pendulum_pos - np.pi + np.pi) % (2 * np.pi) - np.pi
    return e


def find_balance_phase(log: dict, threshold_rad: float = 0.3) -> int:
    """Index of the first sample where the pendulum first crosses into the
    near-upright band. Returns 0 if it never does.
    """
    theta_err = _theta_to_upright(log["pendulum_pos"])
    near = np.abs(theta_err) < threshold_rad
    if not near.any():
        return 0
    return int(np.argmax(near))


def steady_state_stats(log: dict, settle_s: float = 1.0,
                       saturation_mag: float | None = None) -> dict:
    """Stats over the post-settle balance phase."""
    i0 = find_balance_phase(log)
    t_start = log["t"][i0] + settle_s if i0 < len(log["t"]) - 1 else log["t"][-1]
    i_start = int(np.searchsorted(log["t"], t_start))
    n = len(log["t"]) - i_start
    if n < 50:
        return dict(n_samples=n, error="insufficient balance time")

    sl = slice(i_start, None)
    t = log["t"][sl]
    theta_err = _theta_to_upright(log["pendulum_pos"][sl])
    mp = log["motor_pos"][sl]
    mv = log["motor_vel"][sl]
    ac = log["accel_cmd"][sl]
    if saturation_mag is None:
        saturation_mag = log["max_accel"] * 0.99

    # Time-near-upright proxy.
    upright = (np.cos(theta_err) + 1.0) * 0.5  # 1 at theta=0, 0 at theta=π
    avg_upright = float(upright.mean())

    # Action chatter: sign-flip rate.
    sign_flips = int(np.sum(np.sign(ac[1:]) != np.sign(ac[:-1])))
    sign_flip_rate_hz = sign_flips / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
    mean_flip_interval_ms = 1000.0 / sign_flip_rate_hz if sign_flip_rate_hz > 0 else float("inf")

    # Saturation fraction.
    sat_pos = int((ac >= saturation_mag).sum())
    sat_neg = int((ac <= -saturation_mag).sum())
    sat_frac = (sat_pos + sat_neg) / n

    # Motor velocity zero-crossings (oscillation freq).
    vel_zc = int(((mv[1:] * mv[:-1]) < 0).sum())
    osc_hz = (vel_zc / 2) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0

    # Motor-vel saturation (against the firmware's MAX_VELOCITY_RAD_S = 5).
    vel_sat_mag = 4.95
    mv_sat = int((np.abs(mv) >= vel_sat_mag).sum())
    mv_sat_frac = mv_sat / n

    return dict(
        n_samples=n,
        balance_start_s=float(t[0]),
        balance_duration_s=float(t[-1] - t[0]),
        avg_upright=avg_upright,
        theta_err_std_rad=float(theta_err.std()),
        theta_err_std_deg=float(np.degrees(theta_err.std())),
        motor_pos_mean=float(mp.mean()),
        motor_pos_std=float(mp.std()),
        motor_pos_std_deg=float(np.degrees(mp.std())),
        motor_vel_rms=float(np.sqrt((mv * mv).mean())),
        motor_vel_sat_frac=mv_sat_frac,
        accel_cmd_mean=float(ac.mean()),
        accel_cmd_std=float(ac.std()),
        accel_sat_pos_frac=sat_pos / n,
        accel_sat_neg_frac=sat_neg / n,
        accel_sat_frac=sat_frac,
        accel_sign_flip_rate_hz=sign_flip_rate_hz,
        accel_sign_flip_interval_ms=mean_flip_interval_ms,
        motor_osc_hz=osc_hz,
    )


def print_stats(log: dict, stats: dict) -> None:
    print(f"\n  {Path(log['path']).name}  ({log['control_freq_hz']:.0f} Hz, "
          f"MAX_ACCEL={log['max_accel']:.0f} rad/s²)")
    if "error" in stats:
        print(f"    {stats['error']} (only {stats['n_samples']} samples)")
        return
    print(f"    balance window:           t={stats['balance_start_s']:.1f}s, "
          f"duration={stats['balance_duration_s']:.1f}s")
    print(f"    avg upright proxy:        {stats['avg_upright']:.3f}")
    print(f"    theta error std:          {stats['theta_err_std_rad']:.3f} rad "
          f"({stats['theta_err_std_deg']:.1f}°)")
    print(f"    motor_pos mean / std:     {stats['motor_pos_mean']:+.3f} rad / "
          f"{stats['motor_pos_std']:.3f} rad ({stats['motor_pos_std_deg']:.1f}°)")
    print(f"    motor_vel RMS:            {stats['motor_vel_rms']:.2f} rad/s "
          f"({stats['motor_vel_sat_frac']*100:.1f}% at |v|≥4.95)")
    print(f"    accel_cmd mean / std:     {stats['accel_cmd_mean']:+.1f} / "
          f"{stats['accel_cmd_std']:.1f} rad/s²")
    print(f"    accel saturation:         "
          f"{stats['accel_sat_pos_frac']*100:.1f}% at +max, "
          f"{stats['accel_sat_neg_frac']*100:.1f}% at -max "
          f"({stats['accel_sat_frac']*100:.1f}% total)")
    print(f"    accel sign flips:         {stats['accel_sign_flip_rate_hz']:.1f} Hz "
          f"(every {stats['accel_sign_flip_interval_ms']:.0f} ms)")
    print(f"    motor osc rate:           {stats['motor_osc_hz']:.1f} Hz")


def plot_logs(logs: list[dict], window_s: tuple[float, float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    colors = ["C0", "C3", "C2", "C4"]

    for idx, log in enumerate(logs):
        c = colors[idx % len(colors)]
        label = Path(log["path"]).name
        t = log["t"]
        sel = (t >= window_s[0]) & (t <= window_s[1])
        if not sel.any():
            print(f"  warning: window {window_s} not in log {label}")
            continue
        theta_err = _theta_to_upright(log["pendulum_pos"][sel])
        axes[0].plot(t[sel], theta_err, c, lw=1, label=label)
        axes[1].plot(t[sel], log["motor_pos"][sel], c, lw=1, label=label)
        axes[2].plot(t[sel], log["motor_vel"][sel], c, lw=1, label=label)
        axes[3].plot(t[sel], log["accel_cmd"][sel], c, lw=0.7, label=label)

    # Reference lines.
    axes[0].axhline(0, color="k", ls=":", alpha=0.3)
    axes[1].axhline(0, color="k", ls=":", alpha=0.3)
    axes[2].axhline(5, color="k", ls="--", alpha=0.3)
    axes[2].axhline(-5, color="k", ls="--", alpha=0.3)
    max_accel = max(log["max_accel"] for log in logs)
    axes[3].axhline(max_accel, color="k", ls="--", alpha=0.3)
    axes[3].axhline(-max_accel, color="k", ls="--", alpha=0.3)

    axes[0].set_ylabel("θ − π (rad)\n(0 = upright)")
    axes[1].set_ylabel("motor_pos (rad)")
    axes[2].set_ylabel("motor_vel (rad/s)\n(±5 = sat)")
    axes[3].set_ylabel(f"accel_cmd (rad/s²)\n(±{max_accel:.0f} = sat)")
    axes[3].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    if len(logs) > 1:
        axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Deploy quality — window [{window_s[0]:g}, {window_s[1]:g}] s", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="One or more deploy npz logs")
    p.add_argument("--window", type=float, nargs=2, default=[5.0, 9.0],
                   metavar=("T_START", "T_END"),
                   help="Time window (seconds) to plot. Default [5, 9].")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path. Default: alongside the first input log.")
    p.add_argument("--settle-s", type=float, default=1.0,
                   help="Seconds to skip after first balance crossing "
                        "before computing steady-state stats. Default 1.0.")
    args = p.parse_args(argv)

    logs = [load_log(path) for path in args.paths]

    print("Steady-state balance statistics:")
    for log in logs:
        stats = steady_state_stats(log, settle_s=args.settle_s)
        print_stats(log, stats)

    if args.out:
        out_path = Path(args.out)
    else:
        first = Path(args.paths[0])
        out_path = first.with_suffix(".png")
    plot_logs(logs, tuple(args.window), out_path)
    print(f"\nPlot: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

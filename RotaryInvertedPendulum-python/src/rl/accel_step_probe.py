"""Measure FastAccelStepper's accel-mode step response lag.

Question we're answering: when we send `set_acceleration(A)` from rest,
how long until the motor's actual angular velocity starts tracking the
commanded ramp `v(t) = A·t`?

We've been quoting "~75 ms" as the lag, but that number was inferred
from trapezoidal-profile undershoot — not from a clean step-response
test. This script does the clean test.

Method:
  For each test acceleration A in a small sweep:
    1. Engage motor, confirm zero velocity.
    2. Grab a baseline Arduino `time_us` so all timestamps align to
       firmware time (not Python wall time — eliminates RTT noise).
    3. Send `set_acceleration(A)` and immediately tight-poll
       `get_state()` for ~200 ms (no rate limiter — use raw serial RTT).
    4. Stop with `set_acceleration(0)` then `disengage_motor()`.
    5. Fit a line to the linear portion of `motor_vel(t)` (the slope
       should ≈ A); its t-intercept is the effective lag τ.

Output:
  - One npz per accel value with the raw samples.
  - A single PNG with one subplot per accel value showing measured
    velocity, ideal A·t, and lag-corrected A·(t − τ).

Usage:
  python accel_step_probe.py --port /dev/cu.usbserial-1130
  python accel_step_probe.py --port ... --accels 10 20 30 50
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
PROBE_DURATION_S = 0.15  # how long to apply each accel step


def probe_one(client: LowLevelClient, accel: float) -> dict:
    """Single step-response trial at the given accel. Returns sample arrays."""
    # Ensure starting from rest with no commanded accel.
    client.set_acceleration(0.0)
    time.sleep(0.4)

    # Baseline: grab Arduino time + initial position so subsequent samples
    # are referenced to "command-sent" firmware time.
    s0 = client.get_state()
    t_us0 = s0.time_us
    pos0 = -s0.motor_pos_rad

    # We don't know the buffer size; reserve generously.
    n_max = 2000
    t_us = np.zeros(n_max, dtype=np.int64)
    pos = np.zeros(n_max, dtype=np.float64)
    vel = np.zeros(n_max, dtype=np.float64)

    # Send command — record send time as the Python-side "command-sent" moment
    # (used only as a sanity check; we mostly rely on Arduino time_us).
    t_py_send = time.monotonic()
    client.set_acceleration(accel)

    n = 0
    aborted = False
    while n < n_max:
        s = client.get_state()
        t_us[n] = s.time_us
        pos[n] = -s.motor_pos_rad
        vel[n] = -s.motor_vel_rad_s
        n += 1
        # Stop conditions: duration elapsed (in firmware time) or safety.
        t_rel = (s.time_us - t_us0) * 1e-6
        if t_rel > PROBE_DURATION_S:
            break
        if abs(pos[n - 1] - pos0) > SAFETY_POS_RAD:
            print(f"  [safety] motor crossed ±{SAFETY_POS_RAD} rad, aborting trial")
            aborted = True
            break

    # Brake: command opposite-sign accel until velocity crosses zero, then
    # zero it. (set_acceleration(0) only holds velocity, doesn't decelerate.)
    brake_accel = -accel
    for _ in range(200):
        s = client.get_state()
        v = -s.motor_vel_rad_s
        if (accel > 0 and v <= 0.05) or (accel < 0 and v >= -0.05):
            break
        client.set_acceleration(brake_accel)
        time.sleep(0.005)
    client.set_acceleration(0.0)
    # Center it back near origin between trials so we don't drift into the rail.
    # Simple closed-loop nudge: damp velocity, push toward pos0.
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

    return dict(
        accel_cmd=accel,
        t_rel_s=t_rel,
        motor_pos_rad=pos,
        motor_vel_rad_s=vel,
        t_us=t_us,
        t_us0=t_us0,
        t_py_send=t_py_send,
        n_samples=n,
        aborted=aborted,
    )


def fit_lag(trial: dict) -> dict:
    """Fit the linear-acceleration region of v(t) and extract effective lag.

    Returns {fitted_accel, fitted_lag_s, fit_t_window_s, n_fit_points}.
    Strategy: pick samples where |v| > 0.3 * A_expected (i.e. motor is
    clearly accelerating) but |v| < 0.9 * A_expected*duration (i.e. still
    in the linear regime, no saturation). Fit v = m·t + c → lag = -c/m.
    """
    t = trial["t_rel_s"]
    v = trial["motor_vel_rad_s"]
    A = trial["accel_cmd"]

    v_target = abs(A) * PROBE_DURATION_S
    if A >= 0:
        mask = (v > 0.3 * abs(A) * PROBE_DURATION_S * 0.5) & (v < 0.85 * v_target)
    else:
        mask = (v < -0.3 * abs(A) * PROBE_DURATION_S * 0.5) & (v > -0.85 * v_target)

    if mask.sum() < 5:
        return dict(fitted_accel=float("nan"), fitted_lag_s=float("nan"),
                    fit_t_window_s=(float("nan"), float("nan")), n_fit_points=int(mask.sum()))

    t_fit = t[mask]
    v_fit = v[mask]
    m, c = np.polyfit(t_fit, v_fit, 1)
    lag = -c / m if abs(m) > 1e-9 else float("nan")
    return dict(
        fitted_accel=float(m),
        fitted_lag_s=float(lag),
        fit_t_window_s=(float(t_fit.min()), float(t_fit.max())),
        n_fit_points=int(mask.sum()),
    )


def plot_trials(trials: list[dict], fits: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(trials)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, trial, fit in zip(axes, trials, fits):
        A = trial["accel_cmd"]
        t = trial["t_rel_s"] * 1000.0  # → ms
        v = trial["motor_vel_rad_s"]

        ax.plot(t, v, "o-", ms=2.5, lw=1, label="measured motor_vel")
        # Ideal (zero-lag) ramp.
        t_ideal = np.linspace(0, PROBE_DURATION_S * 1000.0, 100)
        ax.plot(t_ideal, A * t_ideal / 1000.0, "k--", lw=1,
                label=f"ideal A·t (A={A:g} rad/s²)")
        # Lag-corrected line.
        lag_s = fit["fitted_lag_s"]
        if np.isfinite(lag_s):
            a_fit = fit["fitted_accel"]
            v_corr = a_fit * (t_ideal / 1000.0 - lag_s)
            ax.plot(t_ideal, v_corr, "C3-", lw=1.2,
                    label=f"fit A={a_fit:.1f}, τ={lag_s * 1000:.1f} ms")
            ax.axvline(lag_s * 1000.0, color="C3", alpha=0.3, ls=":")

        ax.set_ylabel("motor_vel (rad/s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"A = {A} rad/s²  ({trial['n_samples']} samples)", fontsize=10)

    axes[-1].set_xlabel("t since accel command sent (ms, Arduino-clock)")
    fig.suptitle("Accel-mode step response — lag measurement", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=False, help="Arduino serial port")
    p.add_argument("--accels", type=float, nargs="+", default=[10.0, 20.0, 30.0, 50.0],
                   help="Accel values to test (rad/s²)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: ./accel_step_runs/<timestamp>/)")
    args = p.parse_args(argv)

    if args.out_dir:
        out = Path(args.out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path(__file__).resolve().parent / "accel_step_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}")

    trials: list[dict] = []
    fits: list[dict] = []

    with LowLevelClient(port=args.port) as client:
        if not client.wait_until_ready():
            print("Arduino not responding", file=sys.stderr)
            return 1
        client.engage_motor()
        time.sleep(0.3)

        for i, A in enumerate(args.accels):
            print(f"\n[{i + 1}/{len(args.accels)}] A = {A} rad/s²  ...")
            trial = probe_one(client, A)
            fit = fit_lag(trial)
            trials.append(trial)
            fits.append(fit)

            # Effective serial sample rate.
            if trial["n_samples"] > 1:
                span = trial["t_rel_s"][-1] - trial["t_rel_s"][0]
                hz = (trial["n_samples"] - 1) / span if span > 0 else 0
            else:
                hz = 0
            print(f"  samples: {trial['n_samples']}  ({hz:.0f} Hz effective)")
            print(f"  fitted accel: {fit['fitted_accel']:.2f} rad/s²"
                  f"   (commanded {A:g})")
            print(f"  fitted lag τ: {fit['fitted_lag_s'] * 1000:.2f} ms"
                  f"   (n_fit_pts={fit['n_fit_points']})")

            np.savez(out / f"trial_A{A:g}.npz", **trial, **{
                f"fit_{k}": v for k, v in fit.items() if not isinstance(v, tuple)
            })

        client.set_acceleration(0.0)
        client.disengage_motor()

    plot_path = out / "step_response.png"
    plot_trials(trials, fits, plot_path)
    print(f"\nPlot: {plot_path}")

    # Summary table
    print("\n" + "─" * 60)
    print(f"{'A_cmd':>10}  {'A_fit':>10}  {'τ (ms)':>10}  {'rate (Hz)':>10}")
    print("─" * 60)
    for trial, fit in zip(trials, fits):
        span = trial["t_rel_s"][-1] - trial["t_rel_s"][0]
        hz = (trial["n_samples"] - 1) / span if span > 0 else 0
        print(f"{trial['accel_cmd']:>10.1f}  {fit['fitted_accel']:>10.2f}"
              f"  {fit['fitted_lag_s'] * 1000:>10.2f}  {hz:>10.0f}")
    print("─" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

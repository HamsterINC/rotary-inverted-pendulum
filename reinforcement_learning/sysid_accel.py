"""Characterise the motor + pendulum response to accel-mode commands.

Drives `LowLevelServer.set_acceleration()` on the real rig with a programmed
accel waveform, logs motor and pendulum state, then replays the accel the rig
ACTUALLY received through `pendulum_env` and produces a side-by-side comparison.

Key differences from the first version
--------------------------------------
* The waveform is evaluated against WALL-CLOCK time, not the loop index. If the
  serial link can't keep up with the requested rate, the waveform is still
  played at the right speed; you just get fewer samples (the firmware holds the
  last accel between commands).
* Every command is logged with the host time it was sent; every state sample
  carries the device timestamp. The sim replay uses a zero-order hold of the
  logged commands on its own fixed time grid, and sim output is interpolated
  onto the real timestamps before computing stats.
* Motor is always zeroed and disengaged, even on Ctrl-C or an exception.
* Default sample rate is 200 Hz (what the link can actually do). Check the
  "achieved rate" line printed after each recording.

Waveforms:
    `step`  short bipolar accel pulses at 50, 100, 150 rad/s^2 plus a
            zero-crossing reversal. Each pulse pair returns velocity to ~0.
    `chirp` sinusoidal accel sweep 0.5 -> 3 Hz at amplitude 100 rad/s^2.
    `pairs` N identical +A/-A pulse pairs from rest, alternating sign: N that
            stay under the velocity cap, then N that hit it. Run-to-run spread
            of the residual velocity separates random command-timing errors
            (spread) from repeatable ones (cap / ramp / gain): --repeat N.

Usage:
    python sysid_accel.py --port COM3 --waveform step
    python sysid_accel.py --port COM3 --waveform all --sample-rate 200
    # re-analyse an earlier recording without touching the rig:
    python sysid_accel.py --waveform step --fit-from sysid_runs/2026-10-30_103221

Outputs (in sysid_runs/<timestamp>/ unless --out-dir given):
    sysid_accel_<waveform>.npz  raw recording + sim replay
    sysid_accel_<waveform>.png  comparison plot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

FPGA = True
if FPGA:
    from Deploy.FPGA.FPGA_lowlevel_client import FpgaLowLevelClient as LowLevelClient
else:
    from Deploy.ESP32.lowlevel_client import LowLevelClient

from Simulation.pendulum_env import (
    RotaryInvertedPendulumEnv,
    MAX_ACCEL_RAD_S2,
    MAX_VELOCITY_RAD_S,
)


SAMPLE_RATE_HZ = 100.0   # real-rig sampling/command rate (serial round trip bound)
SIM_RATE_HZ = 100.0      # sim control rate (any rate works; commands are ZOH)


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------

def waveform_step(t: float) -> float:
    """Bipolar accel pulses; each pair brings velocity back to ~0.

    With MAX_VELOCITY=5 rad/s and given amplitudes, each "+A for T" reaches
    the velocity cap exactly at the end of T; the matched "-A for T" brings
    velocity back to 0. Motor position drift bounded to ~0.5 rad per pulse.
    """
    if t < 0.3: return 0.0
    # Pulse 1: amplitude 50, T=0.1s
    if t < 0.4: return +50.0
    if t < 0.50: return -50.0
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
    # Zero-crossing reversal: +cap, through zero to -cap, then decel to 0.
    if t < 2.85:  return +100.0   # 0 -> +5
    if t < 2.95:  return -100.0   # +5 -> -5 (through zero)
    if t < 3.00:  return +100.0   # -5 -> 0
    return 0.0


STEP_DURATION_S = 3.5


def waveform_chirp(t: float) -> float:
    """Sinusoidal accel sweep 0.5 -> 3 Hz over 8 s at A=100 rad/s^2."""
    if t < 0.3: return 0.0
    s = t - 0.3
    if s > 8.0: return 0.0
    f0, f1 = 0.5, 3.0
    f = f0 + (f1 - f0) * s / 8.0
    return 100.0 * math.sin(2.0 * math.pi * f * s)


CHIRP_DURATION_S = 9.0


PAIR_A = 100.0
UNDER_PULSE_S = 0.025        # +A 25 ms then -A 25 ms: peak 2.5 rad/s, stays under the 5 rad/s cap
CAP_UP_S, CAP_DOWN_S = 0.100, 0.050   # +A 100 ms (reaches the cap after 50 ms) then -A 50 ms
UNDER_PERIOD_S = 0.40
CAP_PERIOD_S = 0.50
PAIRS_SETTLE_S = 0.3


def make_waveform_pairs(n: int, sample_rate: float):
    """n under-cap pairs then n cap-hitting pairs, alternating direction.

    Pulse lengths are whole numbers of host ticks and the edges sit half a tick
    off the tick grid, so every repetition sends exactly the same number of
    commands (wall-clock jitter cannot move an edge across a tick). Returns
    (waveform_fn, duration_s).
    """
    tick = 1.0 / sample_rate
    def ticks(sec):
        return max(1, int(round(sec * sample_rate))) * tick
    sched = []
    t = PAIRS_SETTLE_S + 0.5 * tick
    for k in range(n):
        sched.append((t, ticks(UNDER_PULSE_S), ticks(UNDER_PULSE_S), 1 if k % 2 == 0 else -1))
        t += UNDER_PERIOD_S
    for k in range(n):
        sched.append((t, ticks(CAP_UP_S), ticks(CAP_DOWN_S), 1 if k % 2 == 0 else -1))
        t += CAP_PERIOD_S
    duration = t + PAIRS_SETTLE_S

    def fn(tt: float) -> float:
        for t0, up, down, sgn in sched:
            if t0 <= tt < t0 + up:
                return sgn * PAIR_A
            if t0 + up <= tt < t0 + up + down:
                return -sgn * PAIR_A
        return 0.0
    return fn, duration


# ---------------------------------------------------------------------------
# Recording on real rig
# ---------------------------------------------------------------------------

def run_real(port, baud: int, waveform_fn, duration: float, sample_rate: float) -> dict:
    """Play `waveform_fn` against wall-clock time and log what happened.

    Returns a dict with:
        t_cmd    host time (s since waveform start) each command was sent
        accel    the command value sent at t_cmd
        t_state  device time (s since waveform start) of each state sample
        motor    motor position, sim frame
        pen      pendulum position, sim frame
    """
    period = 1.0 / sample_rate
    n_max = int(duration * sample_rate) + 2

    t_cmd = np.zeros(n_max)
    accel_log = np.zeros(n_max)
    t_state = np.zeros(n_max)
    motor = np.zeros(n_max)
    pen = np.zeros(n_max)
    n = 0
    late = 0
    initial_motor = initial_pen = 0.0

    with LowLevelClient(port=port, baud=baud) as client:
        if not client.wait_until_ready():
            raise RuntimeError("LowLevelServer did not respond.")
        try:
            input("  Centre the motor by hand, steady the pendulum, then press Enter to engage and start...")
        except EOFError:
            print("  (no stdin; engaging in 2 s)")
            time.sleep(2.0)

        try:
            client.set_acceleration(0.0)
            client.engage_motor()
            client.tare_pendulum()
            time.sleep(0.5)

            s0 = client.get_state()
            dev_t0 = s0.time_us * 1e-6
            initial_motor = -s0.motor_pos_rad   # sim frame
            initial_pen = -s0.pendulum_pos_rad  # sim frame
            print(f"  initial: motor={initial_motor:+.3f} rad, pendulum={initial_pen:+.3f} rad")

            start = time.perf_counter()
            next_tick = start
            while n < n_max:
                # Busy-wait to the tick (time.sleep is too coarse on Windows).
                while time.perf_counter() < next_tick:
                    pass
                now = time.perf_counter()
                t_wf = now - start
                if t_wf >= duration:
                    break

                a = waveform_fn(t_wf)
                client.set_acceleration(a)
                s = client.get_state()

                t_cmd[n] = t_wf
                accel_log[n] = a
                t_state[n] = s.time_us * 1e-6 - dev_t0
                motor[n] = -s.motor_pos_rad
                pen[n] = -s.pendulum_pos_rad
                n += 1

                next_tick += period
                after = time.perf_counter()
                if after > next_tick:
                    # Round trip blew the budget. Resync instead of bursting
                    # to catch up, which would distort the timing further.
                    late += 1
                    next_tick = after
        finally:
            # Always leave the motor safe, even on Ctrl-C / exceptions.
            try:
                client.set_acceleration(0.0)
                time.sleep(0.3)
                client.disengage_motor()
            except Exception as e:  # pragma: no cover
                print(f"  WARNING: failed to safely stop motor: {e}")

    sl = slice(0, n)
    rec = dict(
        t_cmd=t_cmd[sl], accel=accel_log[sl],
        t_state=t_state[sl], motor=motor[sl], pen=pen[sl],
        initial_motor=initial_motor, initial_pen=initial_pen,
    )
    report_timing(rec, sample_rate, late)
    return rec


def report_timing(rec: dict, target_rate: float, late: int) -> None:
    t = rec["t_cmd"]
    n = len(t)
    if n < 3:
        print("  WARNING: fewer than 3 samples recorded")
        return
    dt = np.diff(t)
    achieved = 1.0 / float(np.median(dt))
    mean_rate = (n - 1) / float(t[-1] - t[0])
    print(f"  timing: target {target_rate:.0f} Hz | median-dt rate {achieved:.0f} Hz | "
          f"mean rate {mean_rate:.0f} Hz | dt p95 {1e3*np.percentile(dt, 95):.1f} ms | "
          f"dt max {1e3*dt.max():.1f} ms")
    print(f"  {late}/{n} ticks exceeded the {1e3/target_rate:.1f} ms budget")
    if mean_rate < 0.9 * target_rate:
        print(f"  WARNING: achieved rate is well below target. Lower --sample-rate "
              f"(try {int(mean_rate * 0.8)}). The waveform was still played in real time, "
              f"but pulses are only sampled every ~{1e3/mean_rate:.1f} ms.")


# ---------------------------------------------------------------------------
# Replay in sim
# ---------------------------------------------------------------------------

def zoh(t_query: np.ndarray, t_cmd: np.ndarray, accel: np.ndarray) -> np.ndarray:
    """Zero-order hold of logged commands onto `t_query` (0 before first cmd)."""
    idx = np.searchsorted(t_cmd, t_query, side="right") - 1
    out = accel[np.clip(idx, 0, len(accel) - 1)]
    return np.where(idx >= 0, out, 0.0)


def run_sim(t_cmd: np.ndarray, accel_cmd: np.ndarray, duration: float,
            sim_rate: float, domain_randomization: bool = False,
            transport_delay_s: float = 0.0):
    """Replay the logged accel commands (ZOH) in the sim on a fixed grid.

    The sim starts at motor=0, pendulum=0 with zero velocity, so compare
    against the real data with its initial values subtracted.

    Only plant behaviour is enabled: the policy-side firmware boxcar is OFF
    (action_smooth_window=1; the rig receives raw set_acceleration() here),
    and `transport_delay_s` is the residual command-to-motion delay AFTER the
    host sent the command (serial + firmware parse + FastAccelStepper's
    forward-planning queue + any host/device clock offset).
    """
    period = 1.0 / sim_rate
    n = int(duration * sim_rate) + 1

    # The env's FractionalDelay holds max_delay_steps=8 control ticks.
    max_delay_s = 8.0 / sim_rate
    if transport_delay_s > max_delay_s + 1e-9:
        print(f"  WARNING: sim delay {1e3*transport_delay_s:.1f} ms exceeds the env's "
              f"8-tick buffer at {sim_rate:.0f} Hz ({1e3*max_delay_s:.1f} ms); it will saturate. "
              f"Lower --sim-rate to extend the range.")

    env = RotaryInvertedPendulumEnv(
        control_freq_hz=sim_rate,
        domain_randomization=domain_randomization,   # DR resamples the delay: keep off for fitting
        max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        motor_max_accel_rad_s2=MAX_ACCEL_RAD_S2,
        max_velocity_rad_s=MAX_VELOCITY_RAD_S,
        action_smooth_window=1,
        transport_delay_s=transport_delay_s,
        dr_theta_bias_max_rad=0.0,
        terminate_on_hard_stop=False,
        firmware_obs_model=False,
        episode_length_s=duration + 1.0,
    )
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = 0.0
    env.data.qpos[env._pen_qpos_addr] = 0.0
    env.data.qvel[:] = 0.0
    env._motor_target = 0.0
    env._motor_vel = 0.0
    env._prev_cmd_pos = 0.0
    env._prev_action = 0.0
    mujoco.mj_forward(env.model, env.data)

    t_sim = np.arange(n) * period
    # Step i advances the sim from t_{i-1} to t_i. Use the command in force at
    # the MIDDLE of that step: a command edge then moves exactly one whole step
    # and host-timing jitter of a few tens of microseconds cannot flip it
    # (sampling at the step start would round every edge unpredictably).
    accel_steps = zoh(t_sim[:-1] + 0.5 * period, t_cmd, accel_cmd)
    accel_grid = np.concatenate(([accel_steps[0]], accel_steps))

    motor = np.zeros(n)
    pen = np.zeros(n)
    vel_cmd = np.zeros(n)
    for i in range(1, n):
        action = float(np.clip(accel_steps[i - 1] / MAX_ACCEL_RAD_S2, -1.0, 1.0))
        _, _, _, _, info = env.step(np.array([action], dtype=np.float32))
        motor[i] = float(info["motor_pos"])
        pen[i] = float(info["phi"])
        vel_cmd[i] = float(info["motor_vel_cmd"])   # the sim's exact integrated velocity

    return dict(t=t_sim, accel=accel_grid, motor=motor, pen=pen,
                vel_cmd=vel_cmd, delay_s=float(transport_delay_s))


# ---------------------------------------------------------------------------
# Load a previous recording
# ---------------------------------------------------------------------------

def read_data(in_dir, waveform: str) -> dict:
    npz_path = Path(in_dir) / f"sysid_accel_{waveform}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find {npz_path}")
    d = np.load(npz_path)
    t_state = d["t_state"] if "t_state" in d else d["t"]
    # Old recordings have a single time axis and no separate command times.
    t_cmd = d["t_cmd"] if "t_cmd" in d else d["t"]
    accel = d["accel"] if "accel" in d else d["accel_cmd"]
    motor = d["real_motor"]
    pen = d["real_pen"]
    return dict(t_cmd=t_cmd, accel=accel, t_state=t_state, motor=motor, pen=pen,
                initial_motor=float(motor[0]), initial_pen=float(pen[0]))


# ---------------------------------------------------------------------------
# Plot + stats
# ---------------------------------------------------------------------------

VEL_WINDOW_S = 0.02   # velocity-estimate window (s); same filter for real and sim


def smoothed_velocity(t, x, window_s: float = VEL_WINDOW_S) -> np.ndarray:
    """Causal (backward) finite-difference velocity over ~window_s.

    v(t) only uses positions at or before t, so a velocity change can never
    appear before the command that caused it. (A centred difference reads
    position from the future and makes velocity look like it rises up to
    window/2 early.) The price is a lag of ~window/2 relative to true velocity.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) < 3:
        return np.zeros_like(x)
    dt = float(t[-1] - t[0]) / (len(t) - 1)   # mean spacing (median flips 4<->6 ms on 2 ms-quantised stamps)
    k = max(1, int(round(window_s / dt)))
    idx = np.arange(len(t))
    lo = np.clip(idx - k, 0, len(t) - 1)
    denom = t[idx] - t[lo]
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, (x[idx] - x[lo]) / safe, 0.0)


def ideal_velocity(t_cmd, accel, v_cap: float) -> np.ndarray:
    """Velocity if the commanded accel were delivered perfectly (ZOH), clipped at the cap."""
    v = np.zeros(len(t_cmd))
    for i in range(1, len(t_cmd)):
        v[i] = np.clip(v[i - 1] + accel[i - 1] * (t_cmd[i] - t_cmd[i - 1]),
                       -v_cap, v_cap)
    return v


FW_SAMPLE_PERIOD_MS = 2.0     # firmware samples at 500 Hz (SAMPLE_PERIOD_US)
LATE_MARGIN_MS = 1.5          # flag a tick when its reply is this far above the normal band


def latency_series(rec: dict):
    """d = device sample time - host send time, minus its median, in ms.

    A normal tick reads the newest firmware sample, which is 0-2 ms old, so d
    wanders inside a ~2 ms band (a sawtooth, because the host period is not a
    multiple of 2 ms). A delayed round trip (USB-serial latency, a busy ESP32
    loop) shifts that tick's d upward. This measures the GET_STATE round trip;
    it assumes the SET_ACCEL sent just before it is delayed alike. Returns None
    for old recordings that have a single time axis.
    """
    tc, ts = np.asarray(rec["t_cmd"]), np.asarray(rec["t_state"])
    if len(tc) != len(ts) or len(tc) < 10 or np.array_equal(tc, ts):
        return None
    d = 1e3 * (ts - tc)
    return d - float(np.median(d))


def print_latency_stats(rec: dict) -> None:
    d = latency_series(rec)
    if d is None:
        print("  -- latency check skipped (recording has a single time axis) --")
        return
    p1, p99 = np.percentile(d, [1, 99])
    spread = float(p99 - p1)
    half = 0.5 * FW_SAMPLE_PERIOD_MS
    late = d > half + LATE_MARGIN_MS
    print("  -- host/device latency (device sample time - host send time) --")
    print(f"  band p1..p99 = {p1:+.2f}..{p99:+.2f} ms (spread {spread:.2f} ms; "
          f"expected ~{FW_SAMPLE_PERIOD_MS:.1f} ms from 500 Hz sampling alone)")
    print(f"  max excursion above median  : {float(d.max()):+.2f} ms")
    print(f"  late ticks (> +{half + LATE_MARGIN_MS:.1f} ms) : {int(late.sum())}/{len(d)}"
          + (f"  at t = {', '.join(f'{x:.2f}' for x in np.asarray(rec['t_cmd'])[late][:8])} s" if late.any() else ""))
    if spread > FW_SAMPLE_PERIOD_MS + LATE_MARGIN_MS or late.any():
        print("  -> round-trip latency varies by more than the sampling quantisation: "
              "a command can land a tick late. Command-timing errors are plausible.")
    else:
        print("  -> round-trip latency is steady to within the sampling quantisation. "
              "The GET_STATE path shows no timing slips (SET_ACCEL is not observed directly).")


def _relative(rec: dict, sim: dict):
    """Return real/sim traces with real's initial values removed (sim starts at 0)."""
    m_real = rec["motor"] - rec["motor"][0]
    p_real = rec["pen"] - rec["pen"][0]
    # Sim on the real timestamps so errors are compared at identical times.
    m_sim_at_real = np.interp(rec["t_state"], sim["t"], sim["motor"])
    p_sim_at_real = np.interp(rec["t_state"], sim["t"], sim["pen"])
    return m_real, p_real, m_sim_at_real, p_sim_at_real


def plot_comparison(rec: dict, sim: dict, name: str, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_real, p_real, _, _ = _relative(rec, sim)

    fig, axes = plt.subplots(6, 1, figsize=(11, 14), sharex=True)
    axes[0].step(rec["t_cmd"], rec["accel"], "k", where="post", label="commanded accel (rad/s²)")
    axes[0].set_ylabel("accel cmd (rad/s²)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec["t_state"], m_real, "C0", label="real motor_pos (rel. to start)", linewidth=1.0)
    axes[1].plot(sim["t"], sim["motor"], "C3", label="sim motor_pos", linewidth=1.0)
    axes[1].set_ylabel("motor_pos (rad)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    v_real = smoothed_velocity(rec["t_state"], m_real)
    v_sim = smoothed_velocity(sim["t"], sim["motor"])
    v_ideal = ideal_velocity(rec["t_cmd"], rec["accel"], MAX_VELOCITY_RAD_S)
    axes[2].plot(rec["t_cmd"], v_ideal, "0.5", linestyle="--", linewidth=1.0,
                 label="integrated cmd accel (clipped at cap)")
    axes[2].plot(rec["t_state"], v_real, "C0", label="real motor velocity", linewidth=1.0)
    axes[2].plot(sim["t"], v_sim, "C3", label="sim motor velocity (same filter)", linewidth=1.0)
    axes[2].plot(sim["t"], sim["vel_cmd"], "C3", linestyle=":", linewidth=1.0,
                 label="sim exact velocity (no filter)")
    axes[2].axhline(0.0, color="k", linewidth=0.5)
    axes[2].set_ylabel("motor vel (rad/s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.3)

    axes[3].plot(rec["t_state"], p_real, "C0", label="real pendulum_pos (rel. to start)", linewidth=1.0)
    axes[3].plot(sim["t"], sim["pen"], "C3", label="sim pendulum_pos", linewidth=1.0)
    axes[3].set_ylabel("pendulum_pos (rad)")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    dt_ms = 1e3 * np.diff(rec["t_cmd"])
    axes[4].plot(rec["t_cmd"][1:], dt_ms, "C2", linewidth=0.8, label="real loop interval")
    axes[4].set_ylabel("dt (ms)")
    axes[4].legend(loc="upper right")
    axes[4].grid(alpha=0.3)

    d = latency_series(rec)
    if d is not None:
        half = 0.5 * FW_SAMPLE_PERIOD_MS
        axes[5].plot(rec["t_cmd"], d, "C4", linewidth=0.8, label="device sample time - host send time")
        axes[5].axhspan(-half, half, color="0.85", label="expected 500 Hz sampling band")
        axes[5].axhline(half + LATE_MARGIN_MS, color="r", linestyle="--", linewidth=0.8, label="late-reply threshold")
    else:
        axes[5].text(0.5, 0.5, "no separate host/device time axes in this recording",
                     transform=axes[5].transAxes, ha="center", va="center")
    axes[5].set_ylabel("latency (ms)")
    axes[5].set_xlabel("time (s)")
    if d is not None:
        axes[5].legend(loc="upper right")
    axes[5].grid(alpha=0.3)

    fig.suptitle(f"Accel-mode sysid: {name}   (sim transport delay {1e3*sim['delay_s']:.1f} ms)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  plot saved to {out_path}")


def print_stats(rec: dict, sim: dict, name: str):
    m_real, p_real, m_sim, p_sim = _relative(rec, sim)

    motor_rmse = float(np.sqrt(np.mean((m_sim - m_real) ** 2)))
    pen_err = p_sim - p_real
    pen_wrap = (pen_err + np.pi) % (2 * np.pi) - np.pi
    pen_rmse = float(np.sqrt(np.mean(pen_wrap ** 2)))

    t = rec["t_state"]
    v_real = smoothed_velocity(t, m_real)
    v_sim = smoothed_velocity(t, m_sim)   # sim interpolated onto real timestamps
    vel_rmse = float(np.sqrt(np.mean((v_sim - v_real) ** 2)))
    tail = t > (t[-1] - 0.3)              # waveform ends at rest, so v should be ~0
    v_end_real = float(np.mean(v_real[tail]))
    v_end_sim = float(np.mean(v_sim[tail]))

    print(f"  -- {name} stats --")
    print(f"  motor_pos RMSE              : {motor_rmse:.4f} rad ({math.degrees(motor_rmse):.2f}°)")
    print(f"  pendulum_pos wrapped RMSE   : {pen_rmse:.4f} rad ({math.degrees(pen_rmse):.2f}°)")
    print(f"  motor range  real / sim     : {np.ptp(m_real):.3f} / {np.ptp(sim['motor']):.3f} rad")
    print(f"  pendulum range real / sim   : {np.ptp(p_real):.3f} / {np.ptp(sim['pen']):.3f} rad")
    print(f"  motor velocity RMSE         : {vel_rmse:.3f} rad/s")
    print(f"  motor peak |velocity| real  : {float(np.max(np.abs(v_real))):.3f} rad/s  (20 ms causal)")
    print(f"  motor peak |velocity| sim   : {float(np.max(np.abs(v_sim))):.3f} rad/s")
    print(f"  residual velocity, last 0.3s: real {v_end_real:+.3f} / sim {v_end_sim:+.3f} rad/s")
    print(f"  velocity-cap target         : {MAX_VELOCITY_RAD_S:.1f} rad/s")


def scan_delay(rec: dict, sim_dur: float, sim_rate: float,
               t_max_s: float = 1.0, step_ms: float = 2.0) -> float:
    """Replay with a range of residual transport delays; return the best (ms).

    Metric is motor-velocity RMSE (same causal filter on real and sim) over
    t <= t_max_s. Default 1 s covers the first clean pulse and ends before
    any velocity offset has built up, so the argmin reflects TIMING rather
    than gain/cap effects. Use --scan-t-max 0 for the whole record.
    The fitted delay also absorbs the host/device clock offset.
    """
    max_ms = 1e3 * 8.0 / sim_rate
    delays_ms = np.arange(0.0, max_ms + 1e-9, step_ms)
    t = rec["t_state"]
    win = (t <= t_max_s) if t_max_s > 0 else np.ones(len(t), dtype=bool)
    label = f"t <= {t_max_s:.1f} s" if t_max_s > 0 else "whole record"
    print(f"  -- transport-delay scan (window RMSE over {label}) --")
    print("   delay_ms | motor RMSE (rad) | vel RMSE all (rad/s) | vel RMSE window (rad/s)")
    best_d, best_v = 0.0, np.inf
    for d in delays_ms:
        sim = run_sim(rec["t_cmd"], rec["accel"], sim_dur, sim_rate,
                      domain_randomization=False, transport_delay_s=d * 1e-3)
        m_real, _, m_sim, _ = _relative(rec, sim)
        v_real = smoothed_velocity(t, m_real)
        v_sim = smoothed_velocity(t, m_sim)
        m_rmse = float(np.sqrt(np.mean((m_sim - m_real) ** 2)))
        v_all = float(np.sqrt(np.mean((v_sim - v_real) ** 2)))
        v_win = float(np.sqrt(np.mean((v_sim[win] - v_real[win]) ** 2)))
        print(f"   {d:8.1f} | {m_rmse:16.4f} | {v_all:20.3f} | {v_win:23.3f}")
        if v_win < best_v:
            best_d, best_v = float(d), v_win
    print(f"  best delay ~ {best_d:.0f} ms (window vel RMSE {best_v:.3f} rad/s)")
    if best_d in (0.0, float(delays_ms[-1])):
        print("  NOTE: best value is at the edge of the scan range; the true delay "
              "may be outside it (or the mismatch isn't a delay).")
    return best_d


def find_events(tc, accel, merge_gap_s: float = 0.1):
    """(t_start, t_end) of each run of non-zero commands (gaps < merge_gap_s merged)."""
    idxs = np.flatnonzero(np.abs(accel) > 1e-6)
    if idxs.size == 0:
        return []
    events = []
    start = prev = idxs[0]
    for j in idxs[1:]:
        if tc[j] - tc[prev] > merge_gap_s:
            events.append((tc[start], tc[prev]))
            start = j
        prev = j
    events.append((tc[start], tc[prev]))
    return events


def _event_velocities(rec: dict, sim: dict):
    t, tc = rec["t_state"], rec["t_cmd"]
    m_real, _, m_sim, _ = _relative(rec, sim)
    v_real = smoothed_velocity(t, m_real)
    v_sim = smoothed_velocity(t, m_sim)
    v_cmd = np.interp(t, tc, ideal_velocity(tc, rec["accel"], MAX_VELOCITY_RAD_S))
    return t, v_real, v_sim, v_cmd


def _event_dv(t, v, events, k):
    """Velocity change across event k: settled window after minus window before."""
    a, b = events[k]
    nxt = events[k + 1][0] if k + 1 < len(events) else t[-1]
    b0, b1 = a - 0.06, a - 0.02
    a0, a1 = b + 0.10, min(b + 0.25, nxt - 0.03)
    if a1 <= a0:
        return None, None, None

    def avg(lo, hi):
        m = (t >= lo) & (t <= hi)
        return float(np.mean(v[m])) if m.any() else float("nan")
    return avg(a0, a1) - avg(b0, b1), avg(a0, a1), (a0, a1)


def print_event_table(rec: dict, sim: dict, merge_gap_s: float = 0.1) -> None:
    """Per-event velocity change: where does the real/sim velocity gap come from?

    An "event" is a run of non-zero commands (pulses closer than `merge_gap_s`
    are merged, so a +A/-A pair is one event). For each event we average the
    velocity just before it and a settled window after it, and report the net
    change for: the commanded accel (integrated from the LOGGED commands,
    clipped at the cap), the real rig and the sim.
    """
    events = find_events(rec["t_cmd"], rec["accel"], merge_gap_s)
    if not events:
        return
    t, v_real, v_sim, v_cmd = _event_velocities(rec, sim)

    print("  -- per-event velocity change (rad/s) --")
    print("   t_start  t_end |  dv_cmd  dv_real   dv_sim | real-sim | v_after real / sim")
    for k, (a, b) in enumerate(events):
        d_cmd, _, _ = _event_dv(t, v_cmd, events, k)
        d_real, af_real, _ = _event_dv(t, v_real, events, k)
        d_sim, af_sim, _ = _event_dv(t, v_sim, events, k)
        if d_cmd is None:
            continue
        print(f"   {a:6.2f}  {b:5.2f} | {d_cmd:+7.2f}  {d_real:+7.2f}  {d_sim:+7.2f} | "
              f"{d_real - d_sim:+7.2f}  | {af_real:+6.2f} / {af_sim:+6.2f}")


def print_pairs_summary(rec: dict, sim: dict) -> None:
    """Run-to-run spread of the residual velocity left by identical pulse pairs.

    err = (real dv - dv of the logged commands) * sign of the pair, so a late
    decel always reads positive. Comparing to the LOGGED commands removes any
    host tick-count differences. Two groups: pairs that stay under the cap, and
    pairs that hit it.
    """
    events = find_events(rec["t_cmd"], rec["accel"])
    n = len(events) // 2
    if n < 2 or len(events) != 2 * n:
        print(f"  -- pairs summary skipped: found {len(events)} events, expected 2n with n >= 2 --")
        return
    t, v_real, v_sim, v_cmd = _event_velocities(rec, sim)
    tc, acc = rec["t_cmd"], rec["accel"]

    rows = []
    for k, (a, _) in enumerate(events):
        d_cmd, _, _ = _event_dv(t, v_cmd, events, k)
        d_real, _, _ = _event_dv(t, v_real, events, k)
        d_sim, _, _ = _event_dv(t, v_sim, events, k)
        if d_cmd is None:
            continue
        sgn = float(np.sign(acc[int(np.argmin(np.abs(tc - a)))])) or 1.0
        rows.append((k, sgn, d_cmd, d_real, d_sim))

    print("  -- repeated pulse pairs (rad/s) --")
    print("   pair  group      sign |  dv_cmd  dv_real   dv_sim | real err  sim err")
    for k, sgn, dc, dr, ds in rows:
        grp = "under cap" if k < n else "hits cap "
        print(f"   {k:4d}  {grp}  {sgn:+.0f} | {dc:+7.2f}  {dr:+7.2f}  {ds:+7.2f} | "
              f"{sgn * (dr - dc):+8.2f}  {sgn * (ds - dc):+7.2f}")

    print("   group      n | real err mean  std   min   max | sim err mean | dv_cmd spread")
    for label, lo, hi in (("under cap", 0, n), ("hits cap ", n, 2 * n)):
        g = [r for r in rows if lo <= r[0] < hi]
        if len(g) < 2:
            continue
        er = np.array([r[1] * (r[3] - r[2]) for r in g])
        es = np.array([r[1] * (r[4] - r[2]) for r in g])
        dcs = np.array([r[1] * r[2] for r in g])
        print(f"   {label} {len(g):2d} | {er.mean():+13.3f} {er.std():5.3f} {er.min():+5.2f} {er.max():+5.2f} "
              f"| {es.mean():+12.3f} | {np.ptp(dcs):.3f}")
    print("   reading: |mean| >> std  -> repeatable error (cap / ramp / gain).\n"
          "            std ~ |mean| or bigger, mixed signs -> run-to-run error (command timing).\n"
          "            ~0 for 'under cap' but not for 'hits cap' -> the cap is involved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_output_dir(arg_out_dir) -> Path:
    if arg_out_dir:
        out = Path(arg_out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path(__file__).resolve().parent / "sysid_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Accel-mode sysid: real-rig + sim replay")
    p.add_argument("--port", default=None,
                   help="serial port for the LowLevelServer-flashed board")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--waveform", choices=["step", "chirp", "pairs", "all"], default="all",
                   help="'all' = step + chirp; 'pairs' is run on its own")
    p.add_argument("--repeat", type=int, default=10,
                   help="--waveform pairs: pairs per group (N under-cap, then N cap-hitting); N >= 2")
    p.add_argument("--sample-rate", type=float, default=SAMPLE_RATE_HZ,
                   help="real-rig command/sample rate in Hz (limited by serial round trip)")
    p.add_argument("--sim-rate", type=float, default=SIM_RATE_HZ,
                   help="sim control rate in Hz (real commands are zero-order-held)")
    p.add_argument("--sim-dr", action="store_true",
                   help="enable domain randomization in the sim replay (off by default)")
    p.add_argument("--sim-delay-ms", type=float, default=0.0,
                   help="residual command-to-motion delay in the sim (ms), after the host sent "
                        "the command. Ignored with --sim-dr (DR resamples it).")
    p.add_argument("--scan-delay", action="store_true",
                   help="scan the sim delay, print RMSE vs delay, and use the best value")
    p.add_argument("--scan-t-max", type=float, default=1.0,
                   help="scan metric window in s (0 = whole record)")
    p.add_argument("--fit-from", default=None, metavar="DIR",
                   help="skip the rig; re-run the sim replay on recordings in DIR")
    p.add_argument("--out-dir", default=None, help="output directory (default: sysid_runs/<timestamp>)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    todo = []
    if args.waveform in ("step", "all"):
        todo.append(("step", waveform_step, STEP_DURATION_S))
    if args.waveform in ("chirp", "all"):
        todo.append(("chirp", waveform_chirp, CHIRP_DURATION_S))
    if args.waveform == "pairs":
        if args.repeat < 2:
            p.error("--repeat must be >= 2 for --waveform pairs")
        pairs_fn, pairs_dur = make_waveform_pairs(args.repeat, args.sample_rate)
        todo.append(("pairs", pairs_fn, pairs_dur))

    out_dir = make_output_dir(args.out_dir)

    for name, fn, dur in todo:
        print(f"\n=== {name} ({dur:.1f}s, target {args.sample_rate:.0f} Hz) ===")

        if args.fit_from:
            print(f"  loading recording from {args.fit_from}")
            rec = read_data(args.fit_from, name)
        else:
            rec = run_real(args.port, args.baud, fn, dur, args.sample_rate)
            print(f"  collected {len(rec['t_state'])} samples; "
                  f"motor range [{rec['motor'].min():+.3f}, {rec['motor'].max():+.3f}], "
                  f"pen range [{rec['pen'].min():+.3f}, {rec['pen'].max():+.3f}]")

        if len(rec["t_state"]) < 3:
            print("  not enough data, skipping replay")
            continue

        sim_dur = float(rec["t_state"][-1])
        # Whole number of sim steps per host tick, so command edges land on the
        # sim grid (a 500 Hz sim against 5 ms host ticks rounds edges by up to 2 ms).
        host_rate = (len(rec["t_cmd"]) - 1) / float(rec["t_cmd"][-1] - rec["t_cmd"][0])
        sim_rate = max(1, math.ceil(args.sim_rate / host_rate - 1e-6)) * host_rate   # round UP: more delay-scan range
        note = "" if abs(sim_rate - args.sim_rate) < 0.5 else f" (snapped from {args.sim_rate:.0f} Hz to {sim_rate / host_rate:.0f} x the {host_rate:.0f} Hz host rate)"
        print(f"  replaying logged accel in sim ({sim_rate:.0f} Hz{note}, {sim_dur:.2f} s)...")
        delay_s = args.sim_delay_ms * 1e-3
        if args.scan_delay:
            delay_s = scan_delay(rec, sim_dur, sim_rate, args.scan_t_max) * 1e-3
        sim = run_sim(rec["t_cmd"], rec["accel"], sim_dur, sim_rate,
                      domain_randomization=args.sim_dr, transport_delay_s=delay_s)

        out_npz = out_dir / f"sysid_accel_{name}.npz"
        np.savez(out_npz,
                 t_cmd=rec["t_cmd"], accel=rec["accel"],
                 t_state=rec["t_state"],
                 real_motor=rec["motor"], real_pen=rec["pen"],
                 sim_t=sim["t"], sim_accel=sim["accel"],
                 sim_motor=sim["motor"], sim_pen=sim["pen"],
                 sim_vel_cmd=sim["vel_cmd"], sim_delay_s=sim["delay_s"],
                 initial_motor=rec["initial_motor"], initial_pen=rec["initial_pen"],
                 waveform=name)
        print(f"  saved {out_npz}")

        plot_comparison(rec, sim, name, out_dir / f"sysid_accel_{name}.png")
        print_stats(rec, sim, name)
        if name == "pairs":
            print_pairs_summary(rec, sim)
        else:
            print_event_table(rec, sim)
        print_latency_stats(rec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
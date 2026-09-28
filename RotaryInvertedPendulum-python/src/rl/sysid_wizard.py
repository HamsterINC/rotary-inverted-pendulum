"""Interactive system-identification wizard.

Drives the full sysid workflow against `LowLevelServer.ino` — the same
firmware the RL deployment uses, so the sysid parameters are guaranteed
to describe the rig under the conditions the policy will actually see.

The wizard splits cleanly into two phases:

    collect    — operator-driven data acquisition. Writes raw recordings
                 (NumPy .npz + a metadata.json) to a timestamped directory.
                 The only side effects are on disk.
    fit        — pure post-processing. Reads a recordings directory,
                 derives parameters, generates sim-vs-real validation
                 plots, and writes sysid_params.json. No device required.

Splitting the two means re-fitting after a code change (e.g. an
improved derivation formula) doesn't require re-recording on the rig —
useful because iterating on math is fast, but rig sessions are slow
and depend on the operator being present.

Usage:
    python sysid_wizard.py                       # full pipeline (collect+fit+validate)
    python sysid_wizard.py collect               # only record data
    python sysid_wizard.py fit --in-dir <path>   # only fit existing data
    python sysid_wizard.py validate-motor        # only the motor ±90° sanity sweep
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from lowlevel_client import LowLevelClient
from sysid_core import (
    DEFAULT_FREE_SWING_DURATION_S,
    DEFAULT_FREE_SWING_RUNS,
    aggregate_free_swing_fits,
    derive_pendulum_friction,
    fit_free_swing,
    suggest_control_rate,
    validate_free_swing,
    write_params_json,
)


# ---------------------------------------------------------------------------
# Terminal helpers (ANSI; degrade gracefully when stdout isn't a tty)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)
def _yellow(s: str) -> str: return _c("33", s)
def _red(s: str) -> str: return _c("31", s)
def _green(s: str) -> str: return _c("32", s)
def _cyan(s: str) -> str: return _c("36", s)


def _hr(title: str = "") -> None:
    width = 72
    if title:
        print("\n" + _bold(f"── {title} ".ljust(width, "─")))
    else:
        print("\n" + ("─" * width))


def _print_warnings(warnings) -> None:
    if not warnings:
        return
    for w in warnings:
        if w.severity == "error":
            print(_red(f"  [error] {w.message}"))
        elif w.severity == "warn":
            print(_yellow(f"  [warn]  {w.message}"))
        else:
            print(_dim(f"  [info]  {w.message}"))


def _prompt_yes_no(label: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        s = input(f"{label} {suffix}: ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False


def _prompt_enter(label: str) -> None:
    input(f"{label} (press Enter)")


def _countdown(seconds: int = 3) -> None:
    for k in range(seconds, 0, -1):
        print(f"    {k}...")
        time.sleep(1)
    print(_bold("    GO."))


# ---------------------------------------------------------------------------
# Port discovery (lifted from the old sysid_client; same heuristic)
# ---------------------------------------------------------------------------
import serial.tools.list_ports
# def auto_detect_port() -> list[str]:
#     """Return candidate USB-serial ports that look Arduino-ish."""
#     import serial.tools.list_ports
#     needles = ("usbserial", "usbmodem", "wchusbserial", "ttyUSB", "ttyACM")
#     return [
#         p.device for p in serial.tools.list_ports.comports()
#         if any(n in p.device for n in needles)
#     ]

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    candidates = [
        p.device
        for p in ports
        if "USB" in p.description or "FTDI" in p.description or "Serial" in p.description
    ]
    if candidates:
        return candidates[-1]
    return None

def pick_port(arg_port: str | None) -> str:
    if arg_port:
        return arg_port
    candidates = auto_detect_port()
    if not candidates:
        print(_red("No USB-serial ports detected. Pass --port explicitly."))
        sys.exit(1)
    if len(candidates) == 1:
        return candidates[0]
    print(_bold("Multiple candidate ports found:"))
    for i, p in enumerate(candidates):
        print(f"  [{i}] {p}")
    while True:
        s = input("Pick one [0]: ").strip() or "0"
        try:
            idx = int(s)
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            pass
        print(_red("  invalid choice, try again"))


def make_output_dir(arg_out_dir: str | None) -> Path:
    if arg_out_dir:
        out = Path(arg_out_dir)
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out = Path(__file__).resolve().parent / "sysid_runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Recording helpers — use LowLevelClient with motor disengaged (free-swing)
# or engaged with a simple P controller (motor sweep).
# ---------------------------------------------------------------------------

def _record_state_stream(
    client: LowLevelClient, duration_s: float, sample_hz: float
) -> dict:
    """Sample state at `sample_hz` for `duration_s`. Returns a dict of arrays.

    Signs are flipped to sim-frame convention (matches `pendulum_env.py`).
    """
    n = int(duration_s * sample_hz)
    dt = 1.0 / sample_hz
    t = np.zeros(n, dtype=np.float64)
    pen = np.zeros(n, dtype=np.float64)
    motor = np.zeros(n, dtype=np.float64)
    pvel = np.zeros(n, dtype=np.float64)
    mvel = np.zeros(n, dtype=np.float64)

    t_start = time.monotonic()
    next_tick = t_start
    for i in range(n):
        s = client.get_state()
        t[i] = time.monotonic() - t_start
        pen[i] = -s.pendulum_pos_rad
        motor[i] = -s.motor_pos_rad
        pvel[i] = -s.pendulum_vel_rad_s
        mvel[i] = -s.motor_vel_rad_s
        next_tick += dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    return dict(t=t, pen=pen, motor=motor, pvel=pvel, mvel=mvel)


def record_tare(client: LowLevelClient, out_dir: Path,
                duration_s: float = 3.0, sample_hz: float = 100.0) -> float:
    """Sample a few seconds of pendulum-at-rest to establish the hanging zero.

    The firmware's pendulum-angle accumulator has an arbitrary offset (the
    raw AS5600 reading at Arduino boot is its zero). Recording while
    everything is at rest lets us pin "physical hanging" to a known value
    in the saved coordinate frame, then express every subsequent recording
    relative to it. Removes the operator-error class where the user
    starts the wizard with the pendulum at an unknown angle.
    """
    _hr("Tare hanging position")
    print(_dim(
        "  Let the pendulum hang at rest (motor disengaged). Don't touch\n"
        "  it during the recording. We sample for a few seconds and use\n"
        "  the median as 'physical hanging = 0' for all downstream fits."
    ))
    client.disengage_motor()
    time.sleep(0.5)
    _prompt_enter("\n  Pendulum still and undisturbed?")

    print(_dim(f"  Recording {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)
    hanging_zero = float(np.median(rec["pen"]))
    motion = float(np.std(rec["pen"]))
    print(_dim(f"  Hanging zero: {hanging_zero:+.4f} rad   "
               f"(std during tare: {motion*1000:.2f} mrad)"))
    if motion > 0.01:
        print(_yellow(
            "  WARNING: pendulum moved during tare (>10 mrad std). "
            "The hanging-zero estimate may be off; consider re-doing."))
    np.savez(out_dir / "tare.npz", **rec, hanging_zero=hanging_zero)
    return hanging_zero


def record_free_swing(
    client: LowLevelClient, out_dir: Path, run_i: int, n_runs: int,
    duration_s: float, sample_hz: float = 100.0,
) -> Path:
    """Record one free-swing trial. Motor is disengaged; operator releases."""
    print(_bold(f"\n  Run {run_i + 1}/{n_runs}"))
    print(_dim(
        "    Hold the motor arm firmly. Slide a box under the pendulum so\n"
        "    its tip rests on the box (the lift angle doesn't have to be\n"
        "    exact — the sensor will record whatever angle you start from).\n"
        "    When ready, press Enter. After the countdown, slide the box\n"
        "    out perpendicular to the swing plane to release."
    ))
    _prompt_enter("    Ready to release")
    print("    Releasing in...")
    _countdown(3)
    print(_dim(f"    Recording for {duration_s:.0f}s..."))
    rec = _record_state_stream(client, duration_s, sample_hz)

    # Quick sanity: did the pendulum actually move?
    amp = float(np.max(np.abs(rec["pen"] - np.mean(rec["pen"]))))
    if amp < 0.05:
        print(_yellow(f"    WARNING: only {amp:.3f} rad of pendulum motion "
                      "— did the release work?"))
    else:
        print(_dim(f"    -> max amplitude during recording: {amp:.3f} rad "
                   f"({math.degrees(amp):.0f}°)"))

    path = out_dir / f"free_run_{run_i + 1}.npz"
    np.savez(path, **rec)
    return path


def move_motor_to(
    client: LowLevelClient, target_pos: float, *,
    max_accel: float = 40.0, kp: float = 20.0, kd: float = 10.0,
    tolerance_rad: float = 0.03, control_hz: float = 50.0,
    timeout_s: float = 4.0, log: list | None = None,
    log_t_offset: float = 0.0,
) -> float:
    """Drive the motor to `target_pos` with a closed-loop PD controller
    on acceleration.

    Both bang-bang (provably time-optimal in continuous time) and
    open-loop trapezoidal feedforward (provably correct given a perfect
    actuator) failed on this rig because the motor + FastAccelStepper
    pipeline has a ~75 ms response delay between calling
    `moveByAcceleration` and the motor actually changing velocity. That
    delay is comparable to the per-segment move time and trashes both
    approaches:
      - Bang-bang's "switch to brake exactly here" never matches reality
        because the brake doesn't take effect until 75 ms later.
      - Open-loop schedules the brake too early; the motor never reaches
        the velocity cap and ends up far short of the target.

    PD with moderate gains sidesteps the issue. Commanded accels stay
    well below `max_accel` for typical errors, so we never push the
    motor into the regime where the response delay matters. The motor
    just lags slightly and PD keeps pushing until things converge.

    Gains chosen to keep commanded accel roughly in [-max_accel/2,
    max_accel/2] across a ±π/2 move. Tighter gains drift back into
    the chatter regime; looser gains take longer to settle.
    """
    dt = 1.0 / control_hz
    next_tick = time.monotonic()
    t_start = next_tick
    settle_count = 0
    settle_needed = int(0.3 * control_hz)
    while time.monotonic() - t_start < timeout_s:
        s = client.get_state()
        motor_pos = -s.motor_pos_rad
        motor_vel = -s.motor_vel_rad_s

        error = target_pos - motor_pos
        accel = kp * error - kd * motor_vel
        accel = float(np.clip(accel, -max_accel, max_accel))

        client.set_acceleration(accel)

        if log is not None:
            log.append((time.monotonic() - t_start + log_t_offset,
                        motor_pos, motor_vel, accel, target_pos))

        if abs(error) < tolerance_rad and abs(motor_vel) < 0.3:
            settle_count += 1
            if settle_count >= settle_needed:
                break
        else:
            settle_count = 0

        next_tick += dt
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    client.set_acceleration(0.0)
    return time.monotonic() - t_start


def record_motor_sweep(client: LowLevelClient, out_dir: Path) -> Path:
    """Drive motor through 0 → +90° → 0 → −90° → 0 and record the trajectory.

    The operator watches the rig: if the motor visibly stops short of
    the ±90° marks, step-skipping is happening and the firmware
    position-counter is lying to us. Caught by the recorded trace too.
    """
    _hr("Motor ±90° sanity sweep")
    print(_dim(
        "  We drive the motor through 0 → +90° → 0 → −90° → 0 in accel\n"
        "  mode (same as the policy uses). Watch the rig: the motor arm\n"
        "  should reach roughly perpendicular to its starting position\n"
        "  at the peaks. If it visibly under-shoots, the stepper has\n"
        "  skipped steps and the firmware position no longer matches\n"
        "  the actual mechanical position — that's a deployment-blocker.\n"
    ))
    print(_yellow("  Clear space around the rig; the arm will sweep ~180°."))
    _prompt_enter("  Ready")

    client.set_acceleration(0.0)
    client.engage_motor()
    time.sleep(0.5)

    # Reset accumulator: zero out the motor position so 0 = current pose.
    # (We assume the operator placed the arm where they want "0" to be.)
    log: list = []
    cumulative_t = 0.0
    targets = [
        (0.0, "centre"),
        (+math.pi / 2, "+90°"),
        (0.0, "centre"),
        (-math.pi / 2, "−90°"),
        (0.0, "centre"),
    ]
    for tgt, label in targets:
        print(_dim(f"  → {label} ({math.degrees(tgt):+.0f}°)..."))
        elapsed = move_motor_to(client, tgt, log=log, log_t_offset=cumulative_t)
        cumulative_t += elapsed
        # Hold at the peak briefly so the operator can visually verify.
        time.sleep(0.4)
        cumulative_t += 0.4

    client.set_acceleration(0.0)
    time.sleep(0.2)
    client.disengage_motor()

    log_arr = np.asarray(log, dtype=np.float64)
    path = out_dir / "motor_sweep.npz"
    np.savez(path,
             t=log_arr[:, 0],
             motor=log_arr[:, 1],
             mvel=log_arr[:, 2],
             accel_cmd=log_arr[:, 3],
             target=log_arr[:, 4])
    print(_dim(f"  Saved {len(log_arr)} samples to {path.name}"))
    return path


# ---------------------------------------------------------------------------
# Metadata save/load
# ---------------------------------------------------------------------------

def save_metadata(out_dir: Path, **kwargs) -> None:
    meta_path = out_dir / "metadata.json"
    existing: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            existing = json.load(f)
    existing.update(kwargs)
    existing.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_metadata(in_dir: Path) -> dict:
    with open(in_dir / "metadata.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Collection phase — device-bound
# ---------------------------------------------------------------------------

def collect_data(args: argparse.Namespace) -> Path:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    print(_bold(f"\nRecording session: {out_dir}"))
    print(_dim(f"Port: {port}"))
    print(_dim(
        "  Pendulum geometry (mass, COM, I_com) is loaded from urdf/model.urdf"
        " — no operator measurement needed."
    ))

    with LowLevelClient(port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print(_red("Arduino did not respond. Is LowLevelServer.ino flashed?"))
            sys.exit(2)
        print(_green("  Arduino ready.\n"))

        # Tare
        hanging_zero = record_tare(client, out_dir)

        save_metadata(out_dir,
                      port=port,
                      baud=args.baud,
                      hanging_zero_rad=hanging_zero,
                      n_free_runs=args.n_free_runs,
                      free_swing_duration_s=args.duration_s,
                      sample_hz=args.sample_hz,
                      firmware="LowLevelServer.ino")

        # Free-swing × N
        _hr(f"Free-swing × {args.n_free_runs}")
        for i in range(args.n_free_runs):
            while True:
                try:
                    record_free_swing(client, out_dir, i, args.n_free_runs,
                                       args.duration_s, args.sample_hz)
                    break
                except Exception as e:
                    print(_red(f"    recording failed: {e}"))
                    if not _prompt_yes_no("    Retry?", default=True):
                        sys.exit(3)

        # Motor sanity sweep
        if not args.skip_motor:
            record_motor_sweep(client, out_dir)

    print(_green(f"\nCollection complete: {out_dir}\n"))
    return out_dir


# ---------------------------------------------------------------------------
# Fit phase — pure post-processing
# ---------------------------------------------------------------------------

def fit_data(in_dir: Path, out_json: Path) -> dict:
    _hr("Fit + analysis")
    print(_dim(f"  In:  {in_dir}"))
    print(_dim(f"  Out: {out_json}\n"))

    meta = load_metadata(in_dir)
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        print(_red("  No free_run_*.npz files in directory."))
        sys.exit(4)

    print(_dim(f"  {len(free_files)} free-swing recording(s)"))
    fits: list[dict] = []
    for path in free_files:
        rec = dict(np.load(path))
        pen_centered = rec["pen"] - hanging_zero
        try:
            fit = fit_free_swing(rec["t"], pen_centered)
        except Exception as e:
            print(_red(f"    {path.name}: fit failed ({e})"))
            continue
        fits.append(fit)
        print(_dim(
            f"    {path.name}: T={fit['period_s']*1000:.0f} ms  "
            f"T₀={fit['small_amp_period_s']*1000:.0f} ms  "
            f"α={fit['decay_constant_s_inv']:.4f} 1/s  "
            f"peaks={fit['n_extrema_used']}"
        ))
    if not fits:
        print(_red("  All fits failed."))
        sys.exit(5)

    fit = aggregate_free_swing_fits(fits)
    derived = derive_pendulum_friction(fit)

    rel_err = abs(
        derived["inertia_measured_kg_m2"] - derived["inertia_predicted_kg_m2"]
    ) / derived["inertia_predicted_kg_m2"]
    print(_bold("\n  Derived parameters:"))
    print(f"    viscous friction:        {derived['viscous_friction_N_m_s']:.4e} N·m·s")
    print(f"    Coulomb friction:        {derived['coulomb_friction_N_m']:.4e} N·m")
    print(_dim(
        f"    I (CAD-predicted):       {derived['inertia_predicted_kg_m2']:.4e} kg·m²"
    ))
    print(_dim(
        f"    I (measured from T₀):    {derived['inertia_measured_kg_m2']:.4e} kg·m²  "
        f"({rel_err*100:+.1f}% vs CAD)"
    ))

    # Validate
    warnings = validate_free_swing(fit, derived)
    if warnings:
        print()
        _print_warnings(warnings)

    # Suggested control-rate window. (Without a motor-bandwidth measurement
    # we just print the pendulum half — operator can decide.)
    print(_bold("\n  Control rate suggestion (based on pendulum dynamics alone):"))
    pendulum_period = fit["small_amp_period_s"]
    f_n = 1.0 / pendulum_period
    print(f"    pendulum natural freq f_n ≈ {f_n:.2f} Hz")
    print(f"    minimum useful control rate (5·f_n) ≈ {5*f_n:.1f} Hz")
    print(_dim("    (current rig deploys at 35 Hz; this run won't change that)"))

    # Write the JSON. Schema: only friction is per-rig and lives here;
    # mass/COM/I_com come from urdf/model.urdf via pendulum_geometry. The
    # measured/predicted inertia pair is included for traceability.
    pendulum_payload = {
        "derived": {
            "viscous_friction_N_m_s": derived["viscous_friction_N_m_s"],
            "coulomb_friction_N_m": derived["coulomb_friction_N_m"],
            "inertia_predicted_kg_m2": derived["inertia_predicted_kg_m2"],
            "inertia_measured_kg_m2": derived["inertia_measured_kg_m2"],
        },
        "fit": fit,
    }
    run_meta = {
        "wizard_version": "2026-05-20",
        "collected_at": meta.get("timestamp"),
        "fitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "in_dir": str(in_dir.resolve()),
        "n_free_swing_runs": len(fits),
        "hanging_zero_rad": hanging_zero,
        "firmware": meta.get("firmware", "unknown"),
    }
    write_params_json(str(out_json),
                      pendulum_payload=pendulum_payload, run_meta=run_meta)
    print(_green(f"\n  Wrote {out_json}"))

    # Generate validation plots
    plot_free_swing_compare(in_dir, derived, out_dir=in_dir)
    if (in_dir / "motor_sweep.npz").exists():
        plot_motor_sweep(in_dir / "motor_sweep.npz", out_dir=in_dir)

    return derived


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_free_swing_compare(in_dir: Path, derived: dict, *, out_dir: Path) -> None:
    """Replay the longest free-swing in sim and compare envelopes.

    Time-aligned to the actual release moment, not the recording start —
    the operator typically holds the pendulum on the box for some
    fraction of a second after pressing GO. We detect release as the
    first sample whose pen-velocity exceeds a small threshold, then
    initialise sim from that sample's (pos, vel) state and plot both
    traces with that moment as t=0.
    """
    import matplotlib.pyplot as plt
    import mujoco
    from pendulum_env import RotaryInvertedPendulumEnv

    free_files = sorted(in_dir.glob("free_run_*.npz"))
    if not free_files:
        return
    rec = dict(np.load(free_files[0]))
    meta = load_metadata(in_dir)
    hanging_zero = float(meta.get("hanging_zero_rad", 0.0))

    real_t = rec["t"]
    real_pen = rec["pen"] - hanging_zero
    real_pvel = rec["pvel"]
    sample_dt = float(np.median(np.diff(real_t)))

    # Find the release index: first sample where |pen_vel| > 0.5 rad/s
    # (well above sensor noise, well below typical release speeds).
    moving = np.where(np.abs(real_pvel) > 0.5)[0]
    rel_idx = int(moving[0]) if len(moving) else 0
    # Step back a sample so the trace shows the very start of motion.
    rel_idx = max(0, rel_idx - 1)

    # Trim both traces to start at release.
    real_t_trim = real_t[rel_idx:] - real_t[rel_idx]
    real_pen_trim = real_pen[rel_idx:]
    release_pos = float(real_pen_trim[0])
    release_vel = float(real_pvel[rel_idx])

    # Sim with the derived params loaded by the env. Note: the env reads
    # sysid_params.json at construction time, so we rely on `fit_data` having
    # just written that file.
    env = RotaryInvertedPendulumEnv(
        control_freq_hz=200.0, domain_randomization=False,
        episode_length_s=real_t_trim[-1] + 0.5,
    )
    env.reset(seed=0)
    env.data.qpos[env._motor_qpos_addr] = 0.0
    env.data.qpos[env._pen_qpos_addr]   = release_pos
    env.data.qvel[env._motor_qvel_addr] = 0.0
    env.data.qvel[env._pen_qvel_addr]   = release_vel
    env._motor_target = 0.0; env._motor_vel = 0.0
    mujoco.mj_forward(env.model, env.data)

    n = len(real_t_trim)
    sim_pen = np.zeros(n)
    sim_dt = 1.0 / 200.0
    t = 0.0; idx = 0; next_t = 0.0
    while idx < n:
        env.step(np.array([0.0], dtype=np.float32))
        t += sim_dt
        while idx < n and t >= next_t:
            sim_pen[idx] = float(env.data.qpos[env._pen_qpos_addr])
            idx += 1; next_t += sample_dt

    # Use trimmed arrays for the rest of the function.
    real_t = real_t_trim
    real_pen = real_pen_trim

    # Fit envelopes
    def _fit(t, y):
        crossings = np.where(np.diff(np.sign(y)) != 0)[0]
        pt, pa = [], []
        for k in range(len(crossings) - 1):
            a, b = crossings[k] + 1, crossings[k + 1] + 1
            j = a + int(np.argmax(np.abs(y[a:b])))
            pt.append(t[j]); pa.append(abs(y[j]))
        pt, pa = np.array(pt), np.array(pa)
        if len(pt) < 3:
            return None
        period = 2 * np.median(np.diff(pt))
        decay = -float(np.polyfit(pt, np.log(pa + 1e-9), 1)[0])
        return period, decay, pt, pa

    ix_start = int(0.3 / sample_dt)
    fr = _fit(real_t[ix_start:], real_pen[ix_start:])
    fs = _fit(real_t[ix_start:], sim_pen[ix_start:])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(real_t, real_pen, label="real", lw=1.0)
    axes[0].plot(real_t, sim_pen, label="sim (from derived params)",
                  lw=1.0, alpha=0.85)
    axes[0].axhline(0, color="k", lw=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("angle from hanging (rad)")
    axes[0].grid(alpha=0.3)
    if fr and fs:
        axes[0].set_title(
            f"real T={fr[0]*1000:.0f} ms  τ={fr[1]:.3f}/s  vs  "
            f"sim T={fs[0]*1000:.0f} ms  τ={fs[1]:.3f}/s  "
            f"(Δ {(fs[0]/fr[0]-1)*100:+.1f}%, {(fs[1]/fr[1]-1)*100:+.1f}%)"
        )
        axes[1].semilogy(fr[2], fr[3], "o-", label=f"real peaks (τ={fr[1]:.3f}/s)")
        axes[1].semilogy(fs[2], fs[3], "s--", label=f"sim peaks (τ={fs[1]:.3f}/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("|peak amplitude| (rad, log)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = out_dir / "freeswing_compare.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))


def plot_motor_sweep(npz_path: Path, *, out_dir: Path) -> None:
    """Plot motor target vs actual during the ±90° sanity sweep.

    Older recordings collected before 2026-05-20 wrote per-segment time
    (each `move_motor_to` reset t to 0), which makes the trace appear as
    overlapping segments. Detect that case and rebuild a cumulative
    time axis from the per-segment resets.
    """
    import matplotlib.pyplot as plt
    d = dict(np.load(npz_path))
    t = d["t"].copy()

    # Cumulative-time backfill for legacy per-segment recordings.
    # If t drops backwards between samples, every drop marks the start
    # of a new move_motor_to call. Use the previous sample's cumulative
    # value as the new offset.
    dt = np.diff(t)
    if (dt < 0).any():
        offset = 0.0
        prev = 0.0
        for i in range(len(t)):
            if i > 0 and d["t"][i] < d["t"][i - 1]:
                offset = prev
            t[i] = d["t"][i] + offset
            prev = t[i]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, np.degrees(d["motor"]),  label="motor (firmware-reported)")
    axes[0].plot(t, np.degrees(d["target"]), "--", label="target", alpha=0.7)
    axes[0].axhline(+90, color="k", lw=0.3); axes[0].axhline(-90, color="k", lw=0.3)
    axes[0].axhline(0, color="k", lw=0.2)
    axes[0].set_ylabel("angle (°)"); axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor ±90° sanity sweep — firmware tracks commanded angle")
    axes[1].plot(t, d["accel_cmd"], label="commanded accel (rad/s²)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("accel (rad/s²)")
    axes[1].grid(alpha=0.3); axes[1].legend(loc="upper right")
    plt.tight_layout()
    fig_path = out_dir / "motor_sweep.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(_dim(f"  -> {fig_path.name}"))


# ---------------------------------------------------------------------------
# Standalone motor-validation entry point (uses last collected dir, or
# records its own one-off motor sweep)
# ---------------------------------------------------------------------------

def validate_motor(args: argparse.Namespace) -> None:
    port = pick_port(args.port)
    out_dir = make_output_dir(args.out_dir)
    with LowLevelClient(port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print(_red("Arduino did not respond.")); sys.exit(2)
        record_motor_sweep(client, out_dir)
    plot_motor_sweep(out_dir / "motor_sweep.npz", out_dir=out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common_device(sp):
        sp.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
        sp.add_argument("--baud", type=int, default=2_000_000)
        sp.add_argument("--out-dir", default=None,
                        help="output directory (auto-timestamped if omitted)")

    pc = sub.add_parser("collect", help="acquire data only")
    common_device(pc)
    pc.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    pc.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    pc.add_argument("--sample-hz", type=float, default=100.0)
    pc.add_argument("--skip-motor", action="store_true",
                    help="skip the motor ±90° sanity sweep")

    pf = sub.add_parser("fit", help="fit parameters from a collected directory")
    pf.add_argument("--in-dir", required=True, type=Path)
    pf.add_argument("--out-json", default=Path(__file__).resolve().parent / "sysid_params.json",
                    type=Path)

    pv = sub.add_parser("validate-motor", help="motor ±90° sanity sweep only")
    common_device(pv)

    # Default (no subcommand): full pipeline.
    common_device(p)
    p.add_argument("--n-free-runs", type=int, default=DEFAULT_FREE_SWING_RUNS)
    p.add_argument("--duration-s", type=float, default=DEFAULT_FREE_SWING_DURATION_S)
    p.add_argument("--sample-hz", type=float, default=100.0)
    p.add_argument("--skip-motor", action="store_true")
    p.add_argument("--out-json", default=Path(__file__).resolve().parent / "sysid_params.json",
                    type=Path)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd == "collect":
        collect_data(args)
    elif args.cmd == "fit":
        fit_data(args.in_dir, args.out_json)
    elif args.cmd == "validate-motor":
        validate_motor(args)
    else:
        # Full pipeline
        out_dir = collect_data(args)
        fit_data(out_dir, args.out_json)
        print(_bold("\nDone. Use the policy training scripts next:"))
        print(_dim(f"  ./curriculum_train.sh <run-prefix>"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a trained policy on the real rotary inverted pendulum.

Accepts either:
  - a Stable-Baselines3 SAC checkpoint (`.zip`) — the standard teacher path, or
  - a distilled student MLP (`.pt`) produced by `distill.py` — used to
    validate the student on the rig before flashing
    `RotaryInvertedPendulum-arduino/RLControl/RLControl.ino`.

Drives the device through the LowLevelServer binary protocol at a fixed
control rate. The observation matches what `pendulum_env.py` produces:

    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel]

with theta=0 at upright. Velocities are computed by finite difference and
low-pass filtered to attenuate quantisation noise.

Safety:
- The commanded motor target is clamped to ±125° (inside the ±135°
  mechanical hard stops).
- A staleness check kills the loop if a get_state call returns garbage
  or stalls.
- Ctrl-C disengages the motor cleanly via LowLevelClient.__exit__.

Usage (teacher):
    python run_policy.py --policy runs/desktop_run_2026-05-01/best_model.zip \\
        --port /dev/cu.usbserial-1130

Usage (distilled student, before flashing the Nano):
    python run_policy.py \\
        --policy runs/async_35hz_v2_extend/distill_h32_aug/student.pt \\
        --port /dev/cu.usbserial-1130 \\
        --duration-s 30
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from lowlevel_client import LowLevelClient


MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)


def _wrap_pi(x: float) -> float:
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def make_obs(motor_pos: float, phi: float, motor_vel: float, pen_vel: float,
             prev_action: float) -> np.ndarray:
    """Build the 6-dim observation matching pendulum_env.py.

    phi is the pendulum joint angle (0 = hanging down, +/- pi = upright).
    theta = phi - pi   (so theta=0 at upright, theta=+/-pi at hanging down).
    prev_action is the action issued last tick, in [-1, 1] — restores Markov
    property under action delay by giving the policy a read on its own queue.
    """
    theta = _wrap_pi(phi - math.pi)
    return np.array([
        motor_pos,
        math.sin(theta),
        math.cos(theta),
        motor_vel,
        pen_vel,
        prev_action,
    ], dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a trained SAC policy on the device")
    p.add_argument("--policy", required=True, help="path to a .zip checkpoint")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbserial-1130")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--control-freq", type=float, default=40.0,
                   help="control loop frequency in Hz. MUST match the rate "
                        "the policy was trained at — see "
                        "docs/control_rate_selection.md. Default 35 Hz "
                        "matches this rig's canonical design rate.")
    p.add_argument("--max-accel-rad-s2", type=float, default=150.0,
                   help="action ∈ [-1, 1] maps to commanded angular accel "
                        "[-max, +max] rad/s². Must match the training-time "
                        "max_accel_rad_s2 (default 150 per current env).")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="run the loop but never engage the motor (sanity-check the protocol)")
    p.add_argument("--log", default=None,
                   help="save the trajectory (state + action) to this .npz path "
                        "for refined sysid / sim-to-real analysis")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions from the policy's Gaussian (matches SAC "
                        "training-time behaviour). Default is deterministic = mean. "
                        "Useful while ent_coef is still high and the deterministic "
                        "mean lands in degenerate compromises.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"Loading policy from {args.policy}")
    if str(args.policy).endswith(".pt"):
        # Distilled student MLP. The float (StudentMLP) and QAT (QATStudent)
        # variants share the same predict() signature; the only difference is
        # the FakeQuant nodes in QAT's forward pass. Detect QAT by the
        # presence of the activation observer buffer in the state dict.
        from distill import StudentMLP, _student_predict_factory
        import torch
        from gymnasium import spaces
        ckpt = torch.load(args.policy, map_location=args.device, weights_only=True)
        is_qat = "obs_in.max_abs" in ckpt["state_dict"]
        if is_qat:
            from distill_quantised import QATStudent
            student = QATStudent(
                hidden=int(ckpt["hidden"]),
                obs_dim=int(ckpt["obs_dim"]),
                act_dim=int(ckpt["act_dim"]),
            )
        else:
            student = StudentMLP(
                hidden=int(ckpt["hidden"]),
                obs_dim=int(ckpt["obs_dim"]),
                act_dim=int(ckpt["act_dim"]),
            )
        student.load_state_dict(ckpt["state_dict"])
        predict_fn = _student_predict_factory(student, device=args.device)
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])

        class _StudentShim:
            # Real Box objects so the obs/action-space prints match what
            # `SAC.load(...).observation_space` produces. Bounds match the
            # sim env (`pendulum_env.py`): obs are loose, actions clamped
            # to [-1, 1] by SAC's tanh squash.
            observation_space = spaces.Box(
                low=np.array(
                    [-MOTOR_SAFE_LIMIT_RAD, -1.0, -1.0, -200.0, -200.0, -1.0],
                    dtype=np.float32),
                high=np.array(
                    [MOTOR_SAFE_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0],
                    dtype=np.float32),
                dtype=np.float32,
            )
            action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
            )

            def predict(self, obs, deterministic: bool = True):
                return predict_fn(obs, deterministic=deterministic)

        model = _StudentShim()
        kind = "QAT (int8)" if is_qat else "float"
        print(f"  loaded {kind} distilled student "
              f"({ckpt['obs_dim']}->{ckpt['hidden']}->{ckpt['hidden']}->{ckpt['act_dim']}, "
              f"val_mse={ckpt.get('val_mse', float('nan')):.6f})")
    else:
        model = SAC.load(args.policy, device=args.device)
    print(f"Policy obs space: {model.observation_space}")
    print(f"Policy action space: {model.action_space}")

    dt = 1.0 / args.control_freq
    print(f"Control dt = {dt*1000:.2f} ms")

    interrupted = False

    def _interrupt(*_):
        nonlocal interrupted
        interrupted = True

    # SIGTERM handler matters because `timeout` and many shell-level kills
    # send SIGTERM, which would otherwise skip Python's cleanup path and
    # leave the motor engaged.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)

    with LowLevelClient(args.port, baud=args.baud) as client:
        if not client.wait_until_ready():
            print("ERROR: LowLevelServer did not respond. Is it flashed?", file=sys.stderr)
            return 1
        print("Arduino ready.")

        # Prime the firmware's command state with zero accel before engaging
        # so the motor stays at rest until the policy issues its first action.
        client.set_acceleration(0.0)
        client.tare_pendulum()

        if not args.dry_run:
            client.engage_motor()
            print("Motor engaged.")
        else:
            print("DRY RUN: motor stays disengaged.")

        prev_action = 0.0

        loop_count = 0
        next_tick = time.monotonic()
        max_steps = int(args.duration_s * args.control_freq)
        ep_reward_proxy = 0.0

        # Trajectory log buffers (sim convention throughout for ease of analysis)
        log_t_us = np.zeros(max_steps, dtype=np.int64)
        log_motor_pos = np.zeros(max_steps, dtype=np.float32)
        log_pen_pos = np.zeros(max_steps, dtype=np.float32)
        log_motor_vel = np.zeros(max_steps, dtype=np.float32)
        log_pen_vel = np.zeros(max_steps, dtype=np.float32)
        log_accel_cmd = np.zeros(max_steps, dtype=np.float32)
        log_action = np.zeros(max_steps, dtype=np.float32)

        try:
            while loop_count < max_steps and not interrupted:
                # Pace to the requested control rate.
                next_tick += dt
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

                s = client.get_state()

                # LowLevelServer flips signs of motor/pendulum positions AND
                # velocities on output (but not on set_accel input). Un-flip
                # here so the observation matches the sim convention.
                motor_pos = -s.motor_pos_rad
                phi = -s.pendulum_pos_rad
                motor_vel = -s.motor_vel_rad_s
                pen_vel = -s.pendulum_vel_rad_s

                obs = make_obs(motor_pos, phi, motor_vel, pen_vel, prev_action)
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                a = float(np.clip(action.flatten()[0], -1.0, 1.0))

                # Accel-mode: action maps directly to commanded angular accel.
                # Safety clamp: if we're at the position limit and the policy
                # would push us further into it, zero the accel command.
                accel_cmd = a * args.max_accel_rad_s2
                if motor_pos >= MOTOR_SAFE_LIMIT_RAD and accel_cmd > 0.0:
                    accel_cmd = 0.0
                elif motor_pos <= -MOTOR_SAFE_LIMIT_RAD and accel_cmd < 0.0:
                    accel_cmd = 0.0

                if not args.dry_run:
                    try:
                        client.set_acceleration(accel_cmd)
                    except OSError:
                        # Serial syscall interrupted, almost always by SIGTERM
                        # /SIGINT. Treat as interruption and exit cleanly.
                        interrupted = True
                        break

                # Reward for live monitoring
                theta = _wrap_pi(s.pendulum_pos_rad - math.pi)
                ep_reward_proxy += 0.5 * (1.0 + math.cos(theta))

                # Log this step (sim convention; un-flip already applied above).
                # In accel-mode the "commanded" quantity is the angular accel,
                # not a position target — log it under the same array name for
                # downstream tooling compatibility.
                if args.log:
                    log_t_us[loop_count] = s.time_us
                    log_motor_pos[loop_count] = motor_pos
                    log_pen_pos[loop_count] = phi
                    log_motor_vel[loop_count] = motor_vel
                    log_pen_vel[loop_count] = pen_vel
                    log_accel_cmd[loop_count] = accel_cmd
                    log_action[loop_count] = a

                if loop_count % args.control_freq == 0:
                    print(
                        f"t={loop_count * dt:.1f}s  motor={motor_pos:+.3f} "
                        f"accel_cmd={accel_cmd:+6.1f}  theta={theta:+.3f}  "
                        f"upright={0.5 * (1.0 + math.cos(theta)):.2f}"
                    )

                prev_action = a
                loop_count += 1
        finally:
            # Belt-and-braces: stop further motion before disengaging coils.
            # If we got here via SIGTERM/SIGINT we want a deterministic stop
            # rather than relying solely on LowLevelClient.__exit__. In
            # accel-mode "stop" means command zero acceleration; the firmware's
            # safety logic will decelerate the stepper before we cut power.
            try:
                client.set_acceleration(0.0)
                client.disengage_motor()
            except Exception:
                pass
            print(f"Loop finished. Steps: {loop_count}, "
                  f"avg upright proxy: {ep_reward_proxy / max(1, loop_count):.3f}, "
                  f"motor disengaged.")

            if args.log and loop_count > 0:
                np.savez(
                    args.log,
                    time_us=log_t_us[:loop_count],
                    motor_pos_rad=log_motor_pos[:loop_count],
                    pendulum_pos_rad=log_pen_pos[:loop_count],
                    motor_vel_rad_s=log_motor_vel[:loop_count],
                    pendulum_vel_rad_s=log_pen_vel[:loop_count],
                    accel_cmd_rad_s2=log_accel_cmd[:loop_count],
                    action=log_action[:loop_count],
                    control_freq_hz=np.float32(args.control_freq),
                    max_accel_rad_s2=np.float32(args.max_accel_rad_s2),
                    policy_path=str(args.policy),
                )
                print(f"Saved trajectory log to {args.log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

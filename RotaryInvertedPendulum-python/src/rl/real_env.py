"""Gymnasium environment that uses the *real* rotary inverted pendulum
as its dynamics model.

This is the Phase 4 "sim-to-real fine-tuning" environment: SAC's
`learn()` collects (s, a, r, s') transitions by physically driving the
device, instead of stepping MuJoCo. The policy thus adapts to the
actual hardware dynamics, closing the residual sim-to-real gap that
pure sim training can't.

Observation, action, and reward exactly mirror `pendulum_env.py` so
that a sim-trained checkpoint loads without modification.

Reset behaviour:
    - Disengage the motor.
    - Wait `reset_settle_s` seconds for the pendulum to stop swinging
      under bearing friction.
    - Re-engage the motor at its current position. The motor stays where
      the previous episode ended (no homing) — this matches the typical
      Furuta sim-to-real fine-tuning recipe and avoids unnecessary motor
      wear.

Termination/Truncation:
    - Truncated when `episode_length_s` elapses.
    - Terminated (with `hard_stop_penalty`) if the un-flipped motor
      position |motor_pos| exceeds MOTOR_LIMIT_RAD (the firmware also
      enforces this in hardware, so the policy should never command
      past it; the check here is the policy's "feedback" if it does).

Safety:
    - SIGINT/SIGTERM handlers disengage the motor and close the serial
      port cleanly.
    - Failures during `step()` (lost serial, etc.) raise immediately;
      the SAC training loop will surface them.
"""

from __future__ import annotations

import math
import signal
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lowlevel_client import LowLevelClient
from reward import RewardWeights, compute_reward
from pendulum_env import (
    MOTOR_LIMIT_RAD,
    MOTOR_SAFE_LIMIT_RAD,
    MAX_PENDULUM_VEL_RAD_S,
    MAX_VELOCITY_RAD_S,
    PendulumParams,
    _wrap_pi,
)


HERE = Path(__file__).resolve().parent

# Rest-detection internals (used by _wait_for_pendulum_rest):
# - 0.5 rad/s is above the firmware vel-regression noise floor. AS5600
#   is 12-bit (2π/4096 ≈ 1.53 mrad/count), the regression window is 8 ms
#   (5 samples × 2 ms), so ±1 count of quantisation/I²C jitter alone
#   produces velocity spikes up to ~0.4 rad/s on a stationary pendulum.
#   Earlier threshold of 0.1 was below this noise floor — a "still"
#   pendulum could never accumulate enough consecutive sub-threshold
#   samples to register as rested.
# - 1.0 s of sustained sub-threshold velocity rejects long-tail noise
#   transients (a single I²C glitch or a tiny breath of air won't
#   trigger a false-positive tare).
REST_THRESHOLD_RAD_S = 0.5
REST_DURATION_S = 1.0


class RealRotaryInvertedPendulumEnv(gym.Env):
    """Drives the actual hardware rig as the RL environment.

    Operates over the LowLevelServer binary protocol. Sign conventions
    match `run_policy.py`: server flips motor and pendulum positions on
    output, so we un-flip on read. Action is angular acceleration in
    rad/s² (accel-mode); see `RL_PLAN.md`'s accel-mode decision-log
    entry for the rationale.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        port: str = "/dev/cu.usbserial-110",
        baud: int = 2_000_000,
        control_freq_hz: float = 40.0,  # canonical for this rig — see docs/control_rate_selection.md
        max_accel_rad_s2: float = 150.0,
        episode_length_s: float = 5.0,
        # Max seconds to wait for the pendulum to come to rest between
        # episodes before giving up. While waiting, polls pen_vel; once
        # |pen_vel| stays below REST_THRESHOLD_RAD_S for REST_DURATION_S
        # consecutive seconds, we tare and start the next episode. On
        # timeout (operator didn't stop the pendulum, bearing issue, etc.)
        # we SKIP the tare and proceed using the previous tare's zero —
        # safer than capturing a moving pendulum as the new zero. 15 s
        # gives the operator generous time to settle the pendulum.
        reset_settle_s: float = 15.0,
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # Reward weights (default = Quanser quadratic-cost form, same as
        # the sim env so a sim-trained checkpoint produces equivalent
        # advantage estimates on the first real rollout).
        reward_motor_pos_weight: float = 0.5,
        reward_motor_vel_weight: float = 0.005,
        reward_action_weight: float = 0.20,
        reward_pen_vel_weight: float = 0.001,
        # Smoothness penalty on (a_t - a_{t-1})². Default None → 0
        # (canonical disabled). Mirror of pendulum_env's flag.
        reward_action_rate_weight: float | None = None,
        # Stillness bonus near upright — mirrors pendulum_env.py
        # exactly so sim-trained policies have continuity in the
        # gradient signal when fine-tuned on rig. Default None → 0
        # (disabled). Required if the sim training used this bonus —
        # otherwise fine-tune will gradient-descend out of the bonus
        # basin back toward whatever the classic reward prefers.
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,
        params_path: str | Path | None = None,
    ):
        super().__init__()
        self.port = port
        self.baud = baud
        self.control_freq_hz = control_freq_hz
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.episode_length_s = episode_length_s
        self.reset_settle_s = reset_settle_s  # rest-detection timeout
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        # All reward terms live in a single RewardWeights dataclass shared
        # with pendulum_env.py via reward.py — single source of truth so
        # the sim training and rig fine-tune optimise the SAME objective.
        # Add new reward terms to reward.py, not here.
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_action_rate=(
                float(reward_action_rate_weight)
                if reward_action_rate_weight is not None else 0.0
            ),
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
        )
        # Motor-jerk term needs prev_motor_vel tracking. real_env doesn't
        # currently expose a CLI flag for it (sim-only feature), but the
        # compute_reward signature takes it — initialise to 0 and never update,
        # which makes k_motor_jerk=0 the only sensible weight for real env.
        self._prev_motor_vel = 0.0

        self._dt = 1.0 / control_freq_hz
        self._max_steps = int(episode_length_s * control_freq_hz)

        # Lazy-init the serial connection on the first reset() so that
        # SB3's vectorised env wrappers don't open the port at construction
        # time (which would race during process spawning).
        self._client: LowLevelClient | None = None
        self._motor_engaged = False

        # Per-episode state
        self._step_count = 0
        self._last_accel_cmd = 0.0  # most recent accel command sent (rad/s²)
        self._prev_action = 0.0     # last applied action (∈ [-1, 1]); fed into obs
        self._motor_pos_prev = 0.0
        self._phi_prev = 0.0
        self._motor_vel = 0.0
        self._pen_vel = 0.0
        self._next_tick = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Observation bounds match the sim env exactly. Last dim is
        # prev_action ∈ [-1, 1], present from the accel-mode rework.
        obs_high = np.array(
            [MOTOR_SAFE_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Cooperative shutdown flag. NOT auto-wired to signal handlers
        # anymore — `finetune_async.main()` installs handlers on the main
        # thread and owns shutdown. Direct callers who want signal-driven
        # shutdown can wire `signal.signal(SIGINT, self._on_signal)`
        # themselves.
        self._sigterm_received = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_signal(self, *_):
        self._sigterm_received = True
        try:
            if self._client is not None:
                self._client.disengage_motor()
        except Exception:
            pass

    def _ensure_open(self) -> LowLevelClient:
        if self._client is None:
            self._client = LowLevelClient(self.port, baud=self.baud)
            self._client.__enter__()
            if not self._client.wait_until_ready():
                raise RuntimeError(
                    "LowLevelServer did not respond. Is the sketch flashed and "
                    "the port correct?"
                )
        return self._client

    def _read_raw_state(self):
        s = self._client.get_state()  # type: ignore[union-attr]
        # Un-flip server's sign convention to put us in sim frame.
        return (s.time_us, -s.motor_pos_rad, -s.pendulum_pos_rad,
                -s.motor_vel_rad_s, -s.pendulum_vel_rad_s)

    def _wait_for_pendulum_rest(self, client: LowLevelClient) -> bool:
        """Block until the pendulum has been at rest for REST_DURATION_S.

        At rest = |pen_vel| < REST_THRESHOLD_RAD_S. Polls at 20 Hz; resets
        the streak any time a single sample exceeds the threshold so noise
        transients don't false-positive.

        Returns True if rest was achieved within `reset_settle_s`. Returns
        False on timeout — the caller (real_env.reset) skips the tare in
        that case rather than capturing a moving pendulum's reading as the
        new zero.
        """
        poll_dt_s = 0.05  # 20 Hz
        required_consecutive = int(REST_DURATION_S / poll_dt_s)
        t_start = time.monotonic()
        consecutive_rest = 0
        last_pen_vel = 0.0
        while time.monotonic() - t_start < self.reset_settle_s:
            _, _, _, _, pen_vel = self._read_raw_state()
            last_pen_vel = pen_vel
            if abs(pen_vel) < REST_THRESHOLD_RAD_S:
                consecutive_rest += 1
                if consecutive_rest >= required_consecutive:
                    return True
            else:
                consecutive_rest = 0
            time.sleep(poll_dt_s)
        print(
            f"  [warn] pendulum did not settle within "
            f"{self.reset_settle_s:.1f}s (last |pen_vel|={abs(last_pen_vel):.3f} "
            f"rad/s, threshold={REST_THRESHOLD_RAD_S:.3f}); "
            f"skipping tare for this episode"
        )
        return False

    def _build_obs(self, motor_pos: float, phi: float) -> np.ndarray:
        theta = _wrap_pi(phi - math.pi)
        return np.array(
            [
                motor_pos,
                math.cos(theta),
                math.sin(theta),
                self._motor_vel,
                self._pen_vel,
                self._prev_action,
            ],
            dtype=np.float32,
        )

    def _reward(self, action: float, motor_pos: float, phi: float) -> float:
        # Delegates to reward.compute_reward — same code path the sim
        # env uses, so SAC's gradient signal is identical in sim and on
        # the rig. Add new terms via reward.py.
        return compute_reward(
            theta=_wrap_pi(phi - math.pi),
            pen_vel=self._pen_vel,
            motor_pos=motor_pos,
            motor_vel=self._motor_vel,
            action=action,
            prev_action=self._prev_action,
            prev_motor_vel=self._prev_motor_vel,
            weights=self._reward_weights,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        client = self._ensure_open()

        # Disengage the motor and wait for the pendulum to come to rest.
        # Adaptive — polls pen_vel and only proceeds once the pendulum has
        # been at rest for `rest_duration_s` continuous seconds. Fixed-sleep
        # was unreliable: hard-spun pendulums can still oscillate after 3 s.
        # Taring a moving pendulum captures a non-rest sample as the new
        # encoder zero, polluting the bias signal we rely on.
        client.disengage_motor()
        self._motor_engaged = False
        rested = self._wait_for_pendulum_rest(client)

        # Re-tare the pendulum encoder — but ONLY if the pendulum is
        # actually at rest. If the rest check timed out (operator wasn't
        # paying attention, pendulum still oscillating from previous
        # episode), skipping the tare is safer than capturing a moving
        # sample as the new encoder zero. The previous tare's zero
        # stays in effect for this episode; the next reset gets another
        # chance.
        #
        # When it does fire, the tare records the rig's current
        # static-friction-bounded rest position (±~1.9° from true
        # vertical-down) as the new encoder zero. Across many successful
        # resets the tare value varies within that band — effectively
        # per-episode bias sampling from the rig's actual physical
        # distribution (the hardware analogue of sim's bias DR).
        if rested:
            ok = client.tare_pendulum()
            if not ok:
                raise RuntimeError(
                    "tare_pendulum did not ack — re-flash LowLevelServer.ino "
                    "to pick up the CMD_TARE_PENDULUM=0x06 handler."
                )

        # Read current state and prime the accel command to 0 (hold whatever
        # velocity the stepper is at — zero, since it's been disengaged).
        _, motor_pos, phi, _, _ = self._read_raw_state()
        client.set_acceleration(0.0)
        client.engage_motor()
        self._motor_engaged = True

        self._motor_pos_prev = motor_pos
        self._phi_prev = phi
        self._motor_vel = 0.0
        self._pen_vel = 0.0
        self._prev_action = 0.0
        self._step_count = 0
        self._next_tick = time.monotonic()

        return self._build_obs(motor_pos, phi), {}

    def apply_action(self, action) -> float:
        """Send `action` to the motor as an angular acceleration command.

        Returns the clipped action that was actually applied (to feed into
        the reward / replay-buffer transition).

        Accel-mode: action ∈ [-1, 1] → commanded angular accel ∈ [-max, +max]
        rad/s². The firmware (FastAccelStepper.moveByAcceleration) handles
        smooth velocity ramping and zero-crossing direction reversals; the
        host only needs to push the new accel each tick. Position-limit
        safety: zero the commanded accel if we're at the safety rail and
        pushing outward (the firmware also enforces this independently).
        """
        if self._sigterm_received:
            raise KeyboardInterrupt("SIGTERM received during step")
        client = self._client
        if client is None or not self._motor_engaged:
            raise RuntimeError("env.apply_action() called before reset() or motor disengaged")

        a = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))
        accel_cmd = a * self.max_accel_rad_s2
        if self._motor_pos_prev >= MOTOR_SAFE_LIMIT_RAD and accel_cmd > 0.0:
            accel_cmd = 0.0
        elif self._motor_pos_prev <= -MOTOR_SAFE_LIMIT_RAD and accel_cmd < 0.0:
            accel_cmd = 0.0
        self._last_accel_cmd = accel_cmd
        client.set_acceleration(accel_cmd)
        self._prev_action = a
        return a

    def observe_and_step(self, applied_action: float):
        """Read state from the rig, update filtered velocities, advance step
        counter, compute reward + termination + truncation, and return the
        gym-style 5-tuple `(obs, reward, terminated, truncated, info)`.

        Call this AFTER the control-loop sleep, so the motor has had time to
        react to whatever `apply_action` last sent.
        """
        try:
            t_us, motor_pos, phi, motor_vel, pen_vel = self._read_raw_state()
        except OSError:
            return self._build_obs(self._motor_pos_prev, self._phi_prev), 0.0, True, False, {}

        self._motor_vel = motor_vel
        self._pen_vel = pen_vel

        self._step_count += 1
        reward = self._reward(applied_action, motor_pos, phi)

        terminated = False
        if self.terminate_on_hard_stop and abs(motor_pos) >= MOTOR_LIMIT_RAD:
            terminated = True
            reward -= self.hard_stop_penalty

        truncated = self._step_count >= self._max_steps
        info = {
            "motor_pos": motor_pos,
            "phi": phi,
            "accel_cmd_rad_s2": getattr(self, "_last_accel_cmd", 0.0),
            "time_us": int(t_us),
        }

        self._motor_pos_prev = motor_pos
        self._phi_prev = phi
        return self._build_obs(motor_pos, phi), reward, terminated, truncated, info

    def step(self, action):
        """Synchronous step: apply action, sleep to maintain the configured
        control rate, then observe. Used by single-threaded callers / tests
        and by the legacy synchronous `finetune_real.py` flow.

        The async control loop in `async_control.py` does NOT call this; it
        composes `apply_action` + externally-paced sleep + `observe_and_step`
        directly so the control rate is held even when the learner is busy.
        """
        try:
            a = self.apply_action(action)
        except OSError:
            # Serial syscall interrupted during set_acceleration — treat as termination.
            return self._build_obs(self._motor_pos_prev, self._phi_prev), 0.0, True, False, {}

        # Pace to the requested control rate. Note: this is the bug-prone
        # part — if the SAC training loop runs gradient updates between
        # step() calls and they exceed `_dt`, we silently drop the rate.
        # The async architecture in `async_control.py` paces externally so
        # this `time.sleep` is bypassed.
        self._next_tick += self._dt
        sleep_for = self._next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            self._next_tick = time.monotonic()

        return self.observe_and_step(a)

    def disengage_safely(self) -> None:
        """Disengage the motor from any thread.

        Safe to call from a signal handler or from the orchestrator's main
        thread while a control thread is mid-`get_state` — the lock inside
        `LowLevelClient` serialises serial transactions cleanly.
        """
        try:
            if self._client is not None:
                self._client.disengage_motor()
                self._motor_engaged = False
        except Exception:
            pass

    def close(self):
        if self._client is not None:
            try:
                self._client.disengage_motor()
            except Exception:
                pass
            self._client.__exit__(None, None, None)
            self._client = None
            self._motor_engaged = False

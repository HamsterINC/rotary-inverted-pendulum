"""Gymnasium environment that uses the *real* rotary inverted pendulum
as its dynamics model.

This is the Phase 4 "sim-to-real fine-tuning" environment: SAC's
`learn()` collects (s, a, r, s') transitions by physically driving the
device, instead of stepping MuJoCo. The policy thus adapts to the
actual hardware dynamics, closing the residual sim-to-real gap that
pure sim training can't.

Observation, action, and reward exactly mirror `pendulum_env.py` so
that a sim-trained checkpoint loads without modification:
    - Observation: `obs_history_len` stacked frames (oldest -> newest) of
      [motor_pos, sin θ, cos θ, (motor_vel, pen_vel,) prev_action],
      normalised with the same `obs_norm` preset as the sim env.
      `obs_history_len`, `obs_include_velocities` and `obs_norm` MUST match
      the values the checkpoint was trained with (config.json).
    - Action: commanded angular acceleration in [-max_accel, +max_accel].
      The firmware applies the action boxcar (ACTION_SMOOTH_WINDOW in
      RLControl.ino) and the ±125° rail brake itself, so the host sends the
      raw action — the sim env models those stages explicitly.
    - Reward: Training/reward.py (single source of truth).

Reset behaviour:
    - Disengage the motor.
    - Wait (up to `reset_settle_s`) for the pendulum to come to rest, then
      re-tare the pendulum encoder.
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
    - `_on_signal` / `disengage_safely` disengage the motor; the
      orchestrator (`finetune_async.main()`) owns signal wiring.
    - Failures during `step()` (lost serial, etc.) raise immediately;
      the SAC training loop will surface them.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Deploy.FPGA.FPGA_lowlevel_client import FpgaLowLevelClient
from Training.reward import RewardWeights, compute_reward
from Simulation.pendulum_env import (
    CONTROL_FREQ_HZ,
    MAX_ACCEL_RAD_S2,
    MOTOR_LIMIT_RAD,
    MOTOR_SAFE_LIMIT_RAD,
    OBS_NORM_PRESETS,
    _apply_norm,
    _wrap_pi,
)


HERE = Path(__file__).resolve().parent

# Rest-detection internals (used by _wait_for_pendulum_rest):
# - The pendulum encoder is the 14-bit AS5047P (2π/16384 ≈ 0.38 mrad/count)
#   and the firmware velocity is a finite difference over an 8 ms window,
#   so ±1 LSB of quantisation is only ≈ 0.05 rad/s. 0.5 rad/s therefore
#   sits well above the noise floor (it was originally chosen for the
#   12-bit AS5600, where ±1 count gave spikes up to ~0.4 rad/s). It could
#   be tightened, but a looser threshold only affects how still the
#   pendulum must be before the tare, and it is bounded by the
#   static-friction rest band anyway.
# - 1.0 s of sustained sub-threshold velocity rejects long-tail noise
#   transients (a single glitch or a tiny breath of air won't trigger a
#   false-positive tare).
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
        # Canonical rig rate, shared with the sim env's default.
        control_freq_hz: float = CONTROL_FREQ_HZ,
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,
        episode_length_s: float = 5.0,
        # Max seconds to wait for the pendulum to come to rest between
        # episodes before giving up. While waiting, polls pen_vel; once
        # |pen_vel| stays below REST_THRESHOLD_RAD_S for REST_DURATION_S
        # consecutive seconds, we tare and start the next episode. On
        # timeout (operator didn't stop the pendulum, bearing issue, etc.)
        # we SKIP the tare and proceed using the previous tare's zero —
        # safer than capturing a moving pendulum as the new zero.
        reset_settle_s: float = 15.0,
        # --- Observation layout: MUST match the sim-trained checkpoint ---
        obs_history_len: int = 4,
        obs_include_velocities: bool = True,
        obs_norm: str = "raw",
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # Reward weights — same defaults and meaning as pendulum_env.py so a
        # sim-trained checkpoint produces equivalent advantage estimates on
        # the first real rollout. Must match the sim training run.
        reward_motor_pos_weight: float = 0.5,
        reward_motor_vel_weight: float = 0.005,
        reward_action_weight: float = 0.20,
        reward_pen_vel_weight: float = 0.001,
        # Stillness bonus near upright. Default None → 0 (disabled).
        # Required if the sim training used this bonus — otherwise fine-tune
        # will gradient-descend out of the bonus basin.
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,
        # Alive terms (audit F1). None → 0. See reward.py.
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        params_path: str | Path | None = None,  # accepted for CLI parity; unused
    ):
        super().__init__()
        if obs_norm not in OBS_NORM_PRESETS:
            raise ValueError(f"obs_norm must be one of {list(OBS_NORM_PRESETS)}")
        if int(obs_history_len) < 1:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
        if not obs_include_velocities and int(obs_history_len) < 2:
            raise ValueError(
                "obs_include_velocities=False needs obs_history_len >= 2 — "
                "velocities must be inferable from the position history"
            )

        self.port = port
        self.baud = baud
        self.control_freq_hz = control_freq_hz
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.episode_length_s = episode_length_s
        self.reset_settle_s = reset_settle_s
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)

        self.obs_history_len = int(obs_history_len)
        self.obs_include_velocities = bool(obs_include_velocities)
        self.obs_norm_name = obs_norm
        self._obs_norm = OBS_NORM_PRESETS[obs_norm]
        self._obs_history: deque = deque(maxlen=self.obs_history_len)

        # Same RewardWeights construction as pendulum_env.py.
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
            k_alive_offset=(
                float(reward_alive_offset)
                if reward_alive_offset is not None else 0.0
            ),
            k_upright_alive=(
                float(reward_upright_alive_weight)
                if reward_upright_alive_weight is not None else 0.0
            ),
        )

        self._dt = 1.0 / control_freq_hz
        self._max_steps = int(episode_length_s * control_freq_hz)

        # Lazy-init the serial connection on the first reset() so that
        # SB3's vectorised env wrappers don't open the port at construction
        # time (which would race during process spawning).
        self._client: FpgaLowLevelClient | None = None
        self._motor_engaged = False

        # Per-episode state
        self._step_count = 0
        self._last_accel_cmd = 0.0  # most recent accel command sent (rad/s²)
        self._prev_action = 0.0     # last applied action (∈ [-1, 1]); fed into obs
        self._motor_pos_prev = 0.0
        self._phi_prev = 0.0
        self._motor_vel = 0.0
        self._pen_vel = 0.0
        self._last_time_us = 0
        self._next_tick = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # Same (loose) bounds and shape as the sim env.
        n_frame = 6 if self.obs_include_velocities else 4
        obs_high = np.full(n_frame * self.obs_history_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )

        # Cooperative shutdown flag. NOT auto-wired to signal handlers —
        # `finetune_async.main()` installs handlers on the main thread and
        # owns shutdown. Direct callers who want signal-driven shutdown can
        # wire `signal.signal(SIGINT, self._on_signal)` themselves.
        self._sigterm_received = False

        w = self._reward_weights
        print(
            f"[real_env] reward: alive_offset={w.k_alive_offset} "
            f"upright_alive={w.k_upright_alive} stillness={w.k_stillness_bonus} "
            f"motor_pos={w.k_motor_pos} motor_vel={w.k_motor_vel} "
            f"action={w.k_action} pen_vel={w.k_pen_vel} | "
            f"obs: history={self.obs_history_len} "
            f"vel={self.obs_include_velocities} norm={self.obs_norm_name} | "
            f"{self.control_freq_hz:g} Hz, {self.episode_length_s:g} s episodes"
        )

    # ------------------------------------------------------------------
    # Construction from a training config
    # ------------------------------------------------------------------

    # config.json key -> __init__ kwarg. Everything that defines the
    # observation, the action scale, or the reward MUST come from the
    # checkpoint so fine-tuning optimises the objective the policy was
    # trained on.
    _CONFIG_KEYS = (
        "control_freq_hz",
        "max_accel_rad_s2",
        "obs_history_len",
        "obs_include_velocities",
        "obs_norm",
        "reward_motor_pos_weight",
        "reward_motor_vel_weight",
        "reward_action_weight",
        "reward_pen_vel_weight",
        "reward_stillness_bonus_weight",
        "reward_stillness_sigma_theta_rad",
        "reward_stillness_sigma_motor_vel_rad_s",
        "reward_alive_offset",
        "reward_upright_alive_weight",
        "episode_length_s",
        "hard_stop_penalty",
        "terminate_on_hard_stop",
    )

    # Settings that live in the firmware, not the host. They can't be
    # applied from here, so they're only reported for a manual check.
    _FIRMWARE_KEYS = {
        "action_smooth_window": "ACTION_SMOOTH_WINDOW in RLControl.ino",
        "motor_microsteps": "microstepping (200 × N steps/rev) in LowLevelServer.ino",
        "max_velocity_rad_s": "MOTOR_MIN_STEP_US velocity cap in LowLevelServer.ino",
    }

    @classmethod
    def from_config(
        cls, config: str | Path | dict, **overrides
    ) -> "RealRotaryInvertedPendulumEnv":
        """Build the env from a training run's config.json (path or dict).

        Keys the real env understands are taken from the config; any keyword
        in `overrides` (e.g. port=..., reset_settle_s=...) wins over the
        config. Keys absent from the config fall back to the same defaults as
        pendulum_env.py, except `episode_length_s`, which defaults to the sim
        env's 8.0 s so per-episode returns are comparable.
        """
        if not isinstance(config, dict):
            with open(config) as f:
                config = json.load(f)

        kwargs = {k: config[k] for k in cls._CONFIG_KEYS if k in config}
        kwargs.setdefault("episode_length_s", 8.0)
        kwargs.update(overrides)

        for key, where in cls._FIRMWARE_KEYS.items():
            if key in config:
                print(f"[real_env] checkpoint expects {key}={config[key]} — "
                      f"verify {where} matches")
        if config.get("firmware_obs_model") is False:
            print("[real_env] note: checkpoint trained WITHOUT firmware_obs_model; "
                  "the rig always delivers firmware-measured state")
        return cls(**kwargs)

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

    def _ensure_open(self) -> FpgaLowLevelClient:
        if self._client is None:
            self._client = FpgaLowLevelClient(port=self.port, baud=self.baud)
            self._client.__enter__()
            if not self._client.wait_until_ready():
                raise RuntimeError(
                    "LowLevelServer did not respond. Is the sketch flashed and "
                    "the port correct?"
                )
        return self._client

    def _read_raw_state(self):
        """Return (time_us, motor_pos, phi, motor_vel, pen_vel) in sim frame."""
        s = self._client.get_state()  # type: ignore[union-attr]
        # Un-flip server's sign convention to put us in sim frame.
        return (s.time_us, -s.motor_pos_rad, -s.pendulum_pos_rad,
                -s.motor_vel_rad_s, -s.pendulum_vel_rad_s)

    def _wait_for_pendulum_rest(self, client: FpgaLowLevelClient) -> bool:
        """Block until the pendulum has been at rest for REST_DURATION_S.

        At rest = |pen_vel| < REST_THRESHOLD_RAD_S. Polls at 20 Hz; resets
        the streak any time a single sample exceeds the threshold so noise
        transients don't false-positive.

        Returns True if rest was achieved within `reset_settle_s`. Returns
        False on timeout — the caller (reset) skips the tare in that case
        rather than capturing a moving pendulum's reading as the new zero.
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
        """Build one frame exactly like pendulum_env._obs and push it onto the
        history. Called exactly once per reset/step so the history stays in
        lockstep with control ticks."""
        theta = _wrap_pi(phi - math.pi)
        vals = {
            "motor_pos": motor_pos,
            "sin": math.sin(theta),
            "cos": math.cos(theta),
        }
        if self.obs_include_velocities:
            vals["motor_vel"] = self._motor_vel
            vals["pen_vel"] = self._pen_vel
        vals["prev_act"] = float(np.clip(self._prev_action, -1.0, 1.0))

        feats = self._obs_norm.get("feats", {})
        g = self._obs_norm.get("gain", 1.0)
        frame = np.array(
            [_apply_norm(feats.get(k), v) * g for k, v in vals.items()],
            dtype=np.float32,
        )

        if not self._obs_history:
            self._obs_history.extend([frame] * self.obs_history_len)
        else:
            self._obs_history.append(frame)
        if self.obs_history_len == 1:
            return frame
        return np.concatenate(self._obs_history)

    def _reward(self, action: float, motor_pos: float, phi: float) -> float:
        # Same call as pendulum_env._reward — identical objective in sim and
        # on the rig. Add new terms via reward.py.
        return compute_reward(
            theta=_wrap_pi(phi - math.pi),
            pen_vel=self._pen_vel,
            motor_pos=motor_pos,
            motor_vel=self._motor_vel,
            action=action,
            weights=self._reward_weights,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        client = self._ensure_open()

        # Disengage the motor and wait for the pendulum to come to rest.
        # Taring a moving pendulum captures a non-rest sample as the new
        # encoder zero, polluting the bias signal we rely on.
        client.disengage_motor()
        self._motor_engaged = False
        rested = self._wait_for_pendulum_rest(client)

        # Re-tare only if actually at rest. The tare records the rig's
        # static-friction-bounded rest position (±~1.9° from true
        # vertical-down) as the new zero — the hardware analogue of the
        # sim's per-episode theta-bias DR.
        if rested:
            ok = client.tare_pendulum()
            if not ok:
                raise RuntimeError(
                    "tare_pendulum did not ack — re-flash LowLevelServer.ino "
                    "to pick up the CMD_TARE_PENDULUM=0x06 handler."
                )

        # Read current state and prime the accel command to 0.
        time_us, motor_pos, phi, motor_vel, pen_vel = self._read_raw_state()
        client.set_acceleration(0.0)
        client.engage_motor()
        self._motor_engaged = True

        self._motor_pos_prev = motor_pos
        self._phi_prev = phi
        # Motor is stationary after disengage; the pendulum reading is the
        # firmware's measured velocity (≈0 when rested), like the sim's
        # measurement ring seeded from the true initial state.
        self._motor_vel = 0.0
        self._pen_vel = pen_vel
        self._last_time_us = time_us
        self._prev_action = 0.0
        self._last_accel_cmd = 0.0
        self._step_count = 0
        self._obs_history.clear()  # _build_obs reseeds with the initial frame
        self._next_tick = time.monotonic()

        return self._build_obs(motor_pos, phi), {}

    def apply_action(self, action) -> float:
        """Send `action` to the motor as an angular acceleration command.

        Returns the clipped action that was actually applied (to feed into
        the reward / replay-buffer transition).

        Accel-mode: action ∈ [-1, 1] → commanded angular accel ∈ [-max, +max]
        rad/s². Smoothing (boxcar) and the ±125° rail brake are applied in
        firmware. As a host-side belt-and-braces check we also zero the
        commanded accel if we're at the safety rail and pushing outward.
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
        return a

    def observe_and_step(self, applied_action: float):
        """Read state from the rig, advance the step counter, compute reward +
        termination + truncation, and return the gym-style 5-tuple
        `(obs, reward, terminated, truncated, info)`.

        Call this AFTER the control-loop sleep, so the motor has had time to
        react to whatever `apply_action` last sent.
        """
        try:
            time_us, motor_pos, phi, motor_vel, pen_vel = self._read_raw_state()
        except OSError:
            return self._build_obs(self._motor_pos_prev, self._phi_prev), 0.0, True, False, {}

        self._motor_vel = motor_vel
        self._pen_vel = pen_vel
        self._last_time_us = time_us

        self._step_count += 1
        # Order mirrors pendulum_env.step: reward on the raw action, THEN
        # prev_action is updated so the next obs frame carries it.
        reward = self._reward(applied_action, motor_pos, phi)
        self._prev_action = float(applied_action)

        terminated = False
        if self.terminate_on_hard_stop and abs(motor_pos) >= MOTOR_LIMIT_RAD:
            terminated = True
            reward -= self.hard_stop_penalty

        truncated = self._step_count >= self._max_steps
        info = {
            "motor_pos": motor_pos,
            "phi": phi,
            "accel_cmd_rad_s2": self._last_accel_cmd,
            "time_us": int(time_us),
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

        # Pace to the requested control rate. If gradient updates between
        # step() calls exceed `_dt`, the rate silently drops — the async
        # architecture in `async_control.py` paces externally instead.
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
        the client serialises serial transactions cleanly.
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
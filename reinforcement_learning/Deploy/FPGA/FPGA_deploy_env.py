"""Observation-only environment for the on-PL policy (deployment bitstream).

Unlike FPGA_env.RealRotaryInvertedPendulumEnv, the host never chooses an action:
the PL runs the MLP at 40 Hz and drives the stepper itself. Every `step()` just
waits for the next PL inference and returns what happened:

    obs, reward, terminated, truncated, info = env.step()
    info["action"]   action the PL took from the PREVIOUS obs (clipped to [-1, 1])

so (prev_obs, info["action"], reward, obs, terminated) is one transition, built
from the PL step log: the observation is exactly what the network saw and the
action exactly what it produced. `collect_episode()` runs a whole episode and
returns those transitions as arrays for a train-between-episodes loop
(upload weights -> episode -> SAC updates -> ...).

Observation / action / reward follow pendulum_env so a sim checkpoint matches:
    - Observation: obs_history_len stacked frames (oldest -> newest) of
      [motor_pos, sin θ, cos θ, motor_vel, pen_vel, prev_act] in raw units
      (obs_norm="raw", velocities on), seeded with copies of the first frame —
      exactly the network input the PL builds (OBS_HISTORY in board_top).
      The step log holds the newest frame; the env stacks them the same way.
    - Action: commanded acceleration / ACCEL_FULL_SCALE, in [-1, 1].
    - Reward: Training/reward.py, on the state AFTER the action (as in FPGA_env).

Reset behaviour (same sequence as FPGA_env):
    - engage with the policy OFF and PD-home the motor to angle 0 with PS
      acceleration commands (live telemetry, 50 Hz),
    - disengage, wait for the pendulum to be still (|pen_vel| below a
      threshold AND the angle not moving by more than a small range, both for
      `rest_duration_s`); starts as soon as that holds, gives up after
      `reset_settle_s`,
    - tare (motor angle and pendulum) if it rested,
    - optional `before_start` hook (reset options / collect_episode), e.g. to
      upload the newest weights right before the policy runs,
    - start the PL policy, wait for its first inference = obs of step 0.

Termination / truncation (the episode really stops on the rig):
    - truncated after `episode_length_s` of PL steps (8 s by default),
    - terminated with `hard_stop_penalty` if |motor_pos| >= MOTOR_LIMIT_RAD
      (the PL rail brake at 125 deg normally stops it before that),
    - either way step() disengages the motor, so the policy stops running.

Settings baked into the bitstream (board_top localparams) and checked against
config.json when the env opens the rig: control_freq_hz, obs_history_len,
action_smooth_window (PL boxcar on the actuator), max_velocity_rad_s.

Differences from the sim the policy should know about (reported, not fixed):
    - the PL applies clamp(mu, -1, 1); SAC's deterministic action is tanh(mu),
    - prev_act is the PL's raw (unclipped) last output.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Deploy.FPGA.deploy_client import (
    ACCEL_FULL_SCALE,
    PL_CONTROL_HZ,
    FpgaDeployClient,
    LogOverrun,
    StepRecord,
)
from Deploy.FPGA.rest_detect import (
    REST_ANGLE_RANGE_RAD,
    REST_DURATION_S,
    REST_THRESHOLD_RAD_S,
    wait_until_still,
)
from Training.reward import RewardWeights, compute_reward
from Simulation.pendulum_env import MOTOR_LIMIT_RAD

# Rest detection: see rest_detect.py

HOME_TARGET_RAD = 0.0      # motor angle to home to (0 = last tare)
HOME_KP = 40.0             # [1/s^2]  position gain -> accel command
HOME_KD = 12.6             # [1/s]    ~2*sqrt(KP) for critical damping
HOME_MAX_ACCEL = 20.0      # [rad/s^2] clip on homing accel
HOME_TOL_RAD = 0.02        # position tolerance to count as "at home"
HOME_VEL_TOL = 0.3         # [rad/s]  velocity tolerance to count as "at home"
HOME_HZ = 50.0             # homing control rate
HOME_TIMEOUT_S = 4.0       # give up after this long
HOME_SETTLE_S = 0.3        # must stay inside tolerance this long
HOME_ACCEL_SIGN = 1.0      # PS accel and motor angle share the PL frame

FRAME_DIM = 6
FIRST_STEP_TIMEOUT_S = 1.0     # engage -> first inference (SETTLE_TICKS + 1-2 ticks)


class FpgaDeployEnv(gym.Env):
    """The rig running the PL policy, observed step by step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        bitstream_path: str | Path | None = None,
        download: bool = True,
        episode_length_s: float = 8.0,
        reset_settle_s: float = 15.0,
        rest_vel_threshold_rad_s: float = REST_THRESHOLD_RAD_S,
        rest_angle_range_rad: float = REST_ANGLE_RANGE_RAD,
        rest_duration_s: float = REST_DURATION_S,
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # Reward weights: same defaults / meaning as pendulum_env.py and FPGA_env.
        reward_motor_pos_weight: float = 0.5,
        reward_motor_vel_weight: float = 0.005,
        reward_action_weight: float = 0.20,
        reward_pen_vel_weight: float = 0.001,
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        # Must equal the bitstream's OBS_HISTORY / CONTROL_HZ; checked when the
        # rig is opened (they size the observation space and the episode).
        obs_history_len: int = 1,
        control_freq_hz: float = PL_CONTROL_HZ,
        client: FpgaDeployClient | None = None,
        run_config: dict | None = None,  # config.json: checked against the PL settings
    ):
        super().__init__()
        self._bitstream_path = bitstream_path
        self._download = download
        self.episode_length_s = float(episode_length_s)
        self.reset_settle_s = float(reset_settle_s)
        self.rest_vel_threshold_rad_s = float(rest_vel_threshold_rad_s)
        self.rest_angle_range_rad = float(rest_angle_range_rad)
        self.rest_duration_s = float(rest_duration_s)
        self.last_rest_wait_s = float("nan")    # how long the last reset waited
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        self.control_freq_hz = float(control_freq_hz)
        self.max_accel_rad_s2 = ACCEL_FULL_SCALE
        self._max_steps = int(round(self.episode_length_s * self.control_freq_hz))
        self.obs_history_len = int(obs_history_len)
        self.obs_dim = FRAME_DIM * self.obs_history_len
        self._run_config = run_config
        self._frames: list[np.ndarray] = []

        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_stillness_bonus=float(reward_stillness_bonus_weight or 0.0),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
            k_alive_offset=float(reward_alive_offset or 0.0),
            k_upright_alive=float(reward_upright_alive_weight or 0.0),
        )

        # Same spaces as pendulum_env, so SB3 models load unchanged.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._client = None
        if client is not None:
            self._attach(client)
        self._next_index = 0            # next log index to consume
        self._episode_id = -1           # PL engage counter of the running episode
        self._prev: StepRecord | None = None
        self._step_count = 0
        self._pending: list[StepRecord] = []
        self._sigterm_received = False

        w = self._reward_weights
        print(
            f"[deploy_env] reward: alive_offset={w.k_alive_offset} "
            f"upright_alive={w.k_upright_alive} stillness={w.k_stillness_bonus} "
            f"motor_pos={w.k_motor_pos} motor_vel={w.k_motor_vel} "
            f"action={w.k_action} pen_vel={w.k_pen_vel} | "
            f"obs: history={self.obs_history_len} | "
            f"{self.control_freq_hz:g} Hz, {self.episode_length_s:g} s episodes"
        )

    # ------------------------------------------------------------------
    # Construction from a training config
    # ------------------------------------------------------------------

    _CONFIG_KEYS = (
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

    @classmethod
    def from_config(cls, config: str | Path | dict, **overrides) -> "FpgaDeployEnv":
        """Build from a training run's config.json. Fails if the checkpoint's
        observation layout is not the one the PL computes."""
        if not isinstance(config, dict):
            with open(config) as f:
                config = json.load(f)

        problems = []
        if not bool(config.get("obs_include_velocities", True)):
            problems.append("obs_include_velocities=False (PL: True)")
        if config.get("obs_norm", "raw") != "raw":
            problems.append(f"obs_norm={config['obs_norm']!r} (PL: 'raw')")
        # net-arch, history, smoothing, v_max and rate are checked against the
        # bitstream when the rig is opened (PLConfig.mismatches)
        if problems:
            raise ValueError("checkpoint does not match the PL network: " + ", ".join(problems))

        accel = float(config.get("max_accel_rad_s2", ACCEL_FULL_SCALE))
        if abs(accel - ACCEL_FULL_SCALE) > 1e-6:
            print(f"[deploy_env] WARNING: checkpoint max_accel_rad_s2={accel:g}, PL full "
                  f"scale is {ACCEL_FULL_SCALE:g} rad/s^2 (ACCEL_FULL_SCALE)")

        kwargs = {k: config[k] for k in cls._CONFIG_KEYS if k in config}
        kwargs["obs_history_len"] = int(config.get("obs_history_len", 4))
        kwargs["control_freq_hz"] = float(config.get("control_freq_hz", 50.0))
        kwargs["run_config"] = config
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> FpgaDeployClient:
        if self._client is None:
            kw = {"download": self._download}
            if self._bitstream_path is not None:
                kw["bitstream_path"] = self._bitstream_path
            self._attach(FpgaDeployClient(**kw))
        return self._client

    def _attach(self, client: FpgaDeployClient) -> None:
        """Check the bitstream's baked-in settings against this env / config.json."""
        pl = client.weights.config
        problems = []
        if pl.obs_history != self.obs_history_len:
            problems.append(f"obs_history_len: env {self.obs_history_len}, PL {pl.obs_history}")
        if abs(pl.control_freq_hz - self.control_freq_hz) > 1e-3:
            problems.append(f"control_freq_hz: env {self.control_freq_hz:g}, PL {pl.control_freq_hz:g}")
        if self._run_config is not None:
            problems += [m for m in pl.mismatches(self._run_config)
                         if not m.startswith(("obs_history_len", "control_freq_hz"))]
        if problems:
            client.disengage_safely()
            raise ValueError("bitstream settings do not match the policy: " + "; ".join(problems)
                             + " (board_top localparams vs config.json)")
        print(f"[deploy_env] PL: network {pl.shape}, action smoothing "
              f"{pl.action_smooth_window}, v_max {pl.max_velocity_rad_s:g} rad/s, "
              f"{pl.control_freq_hz:g} Hz")
        self._client = client

    def _stack(self, frame: np.ndarray, first: bool = False) -> np.ndarray:
        """Push one frame (newest) and return the stacked observation, the way
        the PL builds the network input."""
        if first:
            self._frames = [frame] * self.obs_history_len
        else:
            self._frames = self._frames[1:] + [frame]
        return np.concatenate(self._frames).astype(np.float32)

    def _on_signal(self, *_):
        self._sigterm_received = True
        self.disengage_safely()

    def _wait_for_pendulum_rest(self) -> bool:
        """Return as soon as the pendulum is still (see rest_detect). False on
        timeout (`reset_settle_s`)."""
        def read():
            st = self.client.get_state()
            return st.theta, st.pendulum_vel_rad_s
        rested, waited, detail = wait_until_still(
            read, max_wait_s=self.reset_settle_s,
            vel_threshold=self.rest_vel_threshold_rad_s,
            angle_range=self.rest_angle_range_rad,
            duration_s=self.rest_duration_s,
            should_stop=lambda: self._sigterm_received)
        self.last_rest_wait_s = waited
        if not rested:
            if self.reset_settle_s > self.rest_duration_s:
                print(f"  [warn] pendulum not still within {self.reset_settle_s:.1f}s "
                      f"({detail}); engaging anyway without tare")
            else:
                print(f"  pendulum moving ({detail}): starting without tare")
        return rested

    def _next_record(self, timeout: float) -> StepRecord:
        """Next log record of the running episode (buffered, never skips)."""
        while not self._pending:
            recs = self.client.wait_for_records(self._next_index, timeout=timeout)
            self._pending = [r for r in recs if r.episode == self._episode_id]
            self._next_index = recs[-1].index + 1
        return self._pending.pop(0)

    def _reward(self, rec: StepRecord, action: float) -> float:
        motor_pos, s, c, motor_vel, pen_vel, _ = (float(x) for x in rec.obs)
        return compute_reward(
            theta=math.atan2(s, c),
            pen_vel=pen_vel,
            motor_pos=motor_pos,
            motor_vel=motor_vel,
            action=action,
            weights=self._reward_weights,
        )

    # ------------------------------------------------------------------
    # Gymnasium-style API (observation only)
    # ------------------------------------------------------------------

    def _home_motor(self) -> bool:
        """PD-servo the motor to HOME_TARGET_RAD with PS acceleration commands
        (policy off). Returns True if it settled. Same loop as FPGA_env."""
        client = self.client
        dt = 1.0 / HOME_HZ
        settle_needed = max(1, int(round(HOME_SETTLE_S * HOME_HZ)))
        settle_count = 0
        homed = False
        client.engage_motor()                    # driver on, policy off, accel 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            while time.monotonic() - t_start < HOME_TIMEOUT_S:
                if self._sigterm_received:
                    break
                st = client.get_state()
                error = HOME_TARGET_RAD - st.motor_pos_rad
                accel = HOME_KP * error - HOME_KD * st.motor_vel_rad_s
                accel = float(np.clip(accel, -HOME_MAX_ACCEL, HOME_MAX_ACCEL))
                client.set_acceleration(HOME_ACCEL_SIGN * accel)

                if abs(error) < HOME_TOL_RAD and abs(st.motor_vel_rad_s) < HOME_VEL_TOL:
                    settle_count += 1
                    if settle_count >= settle_needed:
                        homed = True
                        break
                else:
                    settle_count = 0

                next_tick += dt
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
        finally:
            client.set_acceleration(0.0)
        if not homed:
            print(f"  [warn] homing did not settle within {HOME_TIMEOUT_S:.1f}s "
                  f"(motor at {client.get_state().motor_pos_rad:+.3f} rad)")
        return homed

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        client = self.client
        client.disengage_motor()                 # stops a still-running episode
        if not client.is_armed():
            raise RuntimeError("SW1 is off (not armed): the PL will not engage")

        homed = self._home_motor()
        client.disengage_motor()
        rested = self._wait_for_pendulum_rest()
        if rested:
            client.tare_pendulum()               # motor angle and pendulum -> 0

        before_start = (options or {}).get("before_start")
        if before_start is not None:
            before_start()                       # e.g. upload the newest weights
        if self._sigterm_received:               # stopped during the wait: never engage
            raise KeyboardInterrupt("stop requested before the policy started")

        start = client.log_count()
        client.start_policy()                    # driver on + PL policy, new episode
        # first record of the new episode = first record after `start` with episode_step 0
        t0 = time.monotonic()
        while True:
            recs = client.wait_for_records(start, timeout=FIRST_STEP_TIMEOUT_S)
            first = next((r for r in recs if r.episode_step == 0), None)
            if first is not None:
                break
            start = recs[-1].index + 1
            if time.monotonic() - t0 > FIRST_STEP_TIMEOUT_S:
                raise TimeoutError("no first step after engage")
        self._episode_id = first.episode
        self._next_index = first.index + 1
        self._pending = []
        self._prev = first
        self._step_count = 0
        return self._stack(first.obs.copy(), first=True), {"record": first, "homed": homed, "tared": rested}

    def step(self, action=None):
        """Wait for the next PL step. `action` is ignored (the PL decides)."""
        if self._sigterm_received:
            raise KeyboardInterrupt("SIGTERM received during step")
        if self._prev is None:
            raise RuntimeError("step() called before reset()")

        rec = self._next_record(timeout=4.0 / self.control_freq_hz)
        prev = self._prev
        gap = rec.episode_step != prev.episode_step + 1
        a = prev.action                          # action taken from prev.obs
        reward = self._reward(rec, a)

        self._step_count += 1
        motor_pos = float(rec.obs[0])
        terminated = self.terminate_on_hard_stop and abs(motor_pos) >= MOTOR_LIMIT_RAD
        if terminated:
            reward -= self.hard_stop_penalty
        truncated = self._step_count >= self._max_steps

        info = {
            "action": a,
            "action_raw": prev.action_raw,
            "brake": prev.brake,                 # rail brake overrode the action
            "clipped": prev.clipped,
            "gap": gap,                          # True if PL steps were missing
            "weight_bank": prev.weight_bank,
            "time_us": rec.time_us,
            "dt_s": (rec.time_us - prev.time_us) * 1e-6,
            "record": rec,
        }
        self._prev = rec
        if terminated or truncated:
            self.disengage_safely()              # episode over: stop the PL policy
        return self._stack(rec.obs.copy()), float(reward), bool(terminated), bool(truncated), info

    def collect_episode(self, *, disengage_at_end: bool = True, before_start=None,
                        on_step=None) -> dict:
        """reset() + step() until the episode ends; returns transition arrays:
        obs, actions, rewards, next_obs, terminated (bool), valid (False across
        gaps or rail-brake steps), plus episode statistics.
        `before_start()` runs after homing/rest/tare, right before the PL
        policy starts (see reset). `on_step()` runs after every step that does
        not end the episode, e.g. to upload newer weights mid-episode (the PL
        swaps weight banks between two inferences)."""
        obs, reset_info = self.reset(options={"before_start": before_start})
        O, A, R, O2, T, V, dts, gaps, brakes = [], [], [], [], [], [], [], [], []
        trunc = False
        try:
            while True:
                nxt, r, term, trunc, info = self.step()
                O.append(obs); A.append(info["action"]); R.append(r); O2.append(nxt)
                T.append(term); V.append(not (info["gap"] or info["brake"]))
                dts.append(info["dt_s"]); gaps.append(info["gap"]); brakes.append(info["brake"])
                obs = nxt
                if term or trunc:
                    break
                if on_step is not None:
                    on_step()
        except LogOverrun as err:
            print(f"  [warn] step log overrun ({err}); episode cut short")
        finally:
            if disengage_at_end:
                self.disengage_safely()

        n = len(R)
        return {
            "obs": np.asarray(O, dtype=np.float32).reshape(n, self.obs_dim),
            "actions": np.asarray(A, dtype=np.float32).reshape(n, 1),
            "rewards": np.asarray(R, dtype=np.float32),
            "next_obs": np.asarray(O2, dtype=np.float32).reshape(n, self.obs_dim),
            "terminated": np.asarray(T, dtype=bool),
            "valid": np.asarray(V, dtype=bool),
            "return": float(np.sum(R)),
            "length": n,
            "mean_dt_s": float(np.mean(dts)) if dts else float("nan"),
            "dt_s": np.asarray(dts, dtype=np.float64),
            "gap": np.asarray(gaps, dtype=bool),
            "brake": np.asarray(brakes, dtype=bool),
            "truncated": bool(trunc),
            "homed": bool(reset_info["homed"]),
            "tared": bool(reset_info["tared"]),
            "rest_wait_s": self.last_rest_wait_s,
        }

    def disengage_safely(self) -> None:
        try:
            if self._client is not None:
                self._client.disengage_motor()
        except Exception:
            pass

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

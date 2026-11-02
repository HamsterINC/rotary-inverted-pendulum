"""Asynchronous control architecture for real-device SAC fine-tuning.

Decouples the rig's 100 Hz control loop from SB3's gradient updates so the
configured control rate is the actual control rate, regardless of how slow
the learner is.

Two threads, three small primitives:

  TransitionQueue  — thread-safe FIFO that the control thread writes and
                     the learner thread drains in batches.
  PolicySnapshot   — inference-only deepcopy of the SAC actor that the
                     control thread reads from. The learner refreshes it
                     after every model.train() call. Required because
                     optimizer.step() does in-place .data writes that
                     would otherwise race with concurrent forward passes.
  AsyncControlLoop — the 100 Hz control thread itself. Reads state,
                     predicts via the snapshot, commands the motor, posts
                     transitions to the queue. Externally paced; never
                     blocks on the learner.

The orchestrator (finetune_async.py) owns the threads, the SAC model,
and the replay buffer. See docs/rl_transitions.md for the full
architectural rationale.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# TransitionQueue
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One (s, a, r, s', done) tuple plus info, posted by the control thread.

    `done` follows SB3 convention: True if either terminated (env's hard-stop
    reached) or truncated (episode time-limit reached). The `info` dict
    carries `TimeLimit.truncated` so the learner can distinguish via
    SB3's handle_timeout_termination machinery.
    """
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    info: dict


class TransitionQueue:
    """Thread-safe FIFO from the control thread to the learner thread.

    Bounded `deque` (default 4096) so a hung learner can't run us out of
    memory. The control thread `put`s; the learner `drain`s in batches.
    """

    def __init__(self, maxlen: int = 4096):
        self._dq: "deque[Transition]" = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, t: Transition) -> None:
        with self._lock:
            self._dq.append(t)

    def drain(self) -> list[Transition]:
        """Move all queued transitions into a fresh list and return it."""
        with self._lock:
            out = list(self._dq)
            self._dq.clear()
        return out

    def discard_recent(self, n: int) -> int:
        """Drop the most recent `n` transitions from the queue. Used by
        the control loop on a TimingViolation to avoid posting transitions
        whose dt was stretched past the timing threshold. Returns the
        actual number discarded (≤ n)."""
        with self._lock:
            actual = min(n, len(self._dq))
            for _ in range(actual):
                self._dq.pop()
        return actual

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)


# ---------------------------------------------------------------------------
# PolicySnapshot
# ---------------------------------------------------------------------------

class PolicySnapshot:
    """Inference-only deepcopy of the SAC actor.

    The control thread calls `predict` for each action. The learner thread
    calls `refresh_from(actor)` after every `model.train()` call. Both paths
    acquire `policy_lock` — required because torch's `optimizer.step()` does
    in-place `.data` writes that would otherwise race with forward passes
    and could yield a corrupted action driving the motor.

    Cost in steady state: one `state_dict` copy per train cycle (~5 KB) and
    one short lock acquire per inference (microseconds; uncontended).
    """

    def __init__(self, actor: nn.Module, device: str = "cpu"):
        self._device = torch.device(device)
        self._net = copy.deepcopy(actor).to(self._device).eval()
        for p in self._net.parameters():
            p.requires_grad_(False)
        self._lock = threading.Lock()

    def refresh_from(self, actor: nn.Module) -> None:
        """Copy `actor`'s weights into the snapshot atomically."""
        new_state = {k: v.detach().to(self._device).clone()
                     for k, v in actor.state_dict().items()}
        with self._lock:
            self._net.load_state_dict(new_state)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict an action. Returns numpy array shape `(action_dim,)`."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        with self._lock, torch.no_grad():
            action = self._net(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# AsyncControlLoop
# ---------------------------------------------------------------------------

class TimingViolation(RuntimeError):
    """Raised when the control thread falls behind the configured rate by
    more than `timing_violation_threshold_s` for `timing_violation_strikes`
    consecutive ticks. Indicates a misconfiguration (e.g. the learner is
    holding the GIL too long, or the threshold is set too tight)."""


@dataclass
class TickStats:
    n_ticks: int = 0
    sum_dt_ms: float = 0.0
    sum_dt2_ms: float = 0.0
    max_overrun_ms: float = 0.0
    n_violations: int = 0


@dataclass
class EpisodeStats:
    episode_idx: int
    n_steps: int
    cumulative_reward: float
    terminated: bool
    truncated: bool
    tick_stats: TickStats

    @property
    def mean_dt_ms(self) -> float:
        if self.tick_stats.n_ticks == 0:
            return 0.0
        return self.tick_stats.sum_dt_ms / self.tick_stats.n_ticks

    @property
    def std_dt_ms(self) -> float:
        n = self.tick_stats.n_ticks
        if n < 2:
            return 0.0
        mean = self.tick_stats.sum_dt_ms / n
        var = (self.tick_stats.sum_dt2_ms / n) - (mean * mean)
        return float(max(var, 0.0) ** 0.5)


class AsyncControlLoop:
    """Drives the rig at strict `control_freq_hz` from a background thread.

    Owns: a reference to the (already-reset) `RealRotaryInvertedPendulumEnv`,
    the `PolicySnapshot` (read-only), the `TransitionQueue` (writes only),
    and a `stop_flag` Event for clean shutdown.

    Does NOT own: the SAC model, the replay buffer, or any signals.

    The orchestrator calls `env.reset()` (which runs the 5 s settle on the
    main thread) BEFORE constructing/starting an instance. This loop
    assumes the env is engaged and ready.
    """

    def __init__(
        self,
        real_env,
        policy: PolicySnapshot,
        queue: TransitionQueue,
        stop_flag: threading.Event,
        *,
        control_freq_hz: float = 35.0,  # canonical for this rig — see docs/control_rate_selection.md
        timing_violation_threshold_s: float = 0.005,
        timing_violation_strikes: int = 3,
        deterministic_actions: bool = False,
        dt_jitter_frac: float = 0.0,
        dt_jitter_seed: int | None = None,
    ):
        self.env = real_env
        self.policy = policy
        self.queue = queue
        self.stop_flag = stop_flag
        self._dt = 1.0 / float(control_freq_hz)
        self._timing_threshold = float(timing_violation_threshold_s)
        self._max_strikes = int(timing_violation_strikes)
        self.deterministic = bool(deterministic_actions)
        # Control-rate jitter as DR (see docs/control_rate_selection.md).
        # Each tick interval is multiplied by uniform(1-frac, 1+frac).
        # Mimics the legacy variable-rate fine-tune which empirically
        # produced the calm "minimal action" attractor on this rig.
        self._dt_jitter_frac = float(dt_jitter_frac)
        self._jitter_rng = random.Random(dt_jitter_seed) if dt_jitter_seed is not None else random.Random()
        self.last_episode_stats: EpisodeStats | None = None
        self.last_error: BaseException | None = None

    def run_episode(self, episode_idx: int) -> EpisodeStats:
        """Run one full episode synchronously. Intended target of `Thread`.

        Returns episode statistics. On `stop_flag` set, terminates cleanly
        (motor disengaged) and returns whatever stats accumulated. On a
        timing violation 3-strike, raises `TimingViolation` after
        disengaging the motor.
        """
        env = self.env
        # The orchestrator already called env.reset(); we read the current
        # state to bootstrap the obs without consuming an episode step.
        try:
            obs = env._build_obs(env._motor_pos_prev, env._phi_prev)
        except Exception as e:
            self.last_error = e
            raise

        next_tick = time.perf_counter()
        last_tick = next_tick
        stats = TickStats()
        cumulative_reward = 0.0
        consecutive_violations = 0
        terminated = False
        truncated = False
        n_steps = 0

        try:
            while not (terminated or truncated):
                if self.stop_flag.is_set():
                    break

                # 1. Predict + apply action (immediate, sends target to motor)
                action = self.policy.predict(obs, deterministic=self.deterministic)
                try:
                    a_applied = env.apply_action(action)
                except (OSError, KeyboardInterrupt):
                    terminated = True
                    break

                # 2. Pace to the next tick. This is THE critical line:
                #    sleep ONLY for the remaining slice of the dt tick.
                #    The learner thread does its work in parallel and
                #    cannot push us past this deadline.
                #
                #    macOS `time.sleep` typically overshoots by ~1-2 ms.
                #    Two-stage strategy: coarse sleep up to 1 ms before the
                #    deadline, then busy-wait the last 1 ms. Costs <1% CPU
                #    at 100 Hz and pulls jitter from ~2 ms down to <0.1 ms.
                #
                #    Optional dt_jitter applies ±frac to this tick's
                #    interval, replicating the legacy variable-rate
                #    regularization that produced the calm attractor.
                if self._dt_jitter_frac > 0.0:
                    jitter = self._jitter_rng.uniform(
                        -self._dt_jitter_frac, self._dt_jitter_frac
                    )
                    next_tick += self._dt * (1.0 + jitter)
                else:
                    next_tick += self._dt
                sleep_for = next_tick - time.perf_counter()
                slept = sleep_for > 0
                if slept:
                    if sleep_for > 0.001:
                        time.sleep(sleep_for - 0.001)
                    while time.perf_counter() < next_tick:
                        pass

                # 3. Measure tick health (for telemetry + violation check)
                now = time.perf_counter()
                overrun = now - next_tick
                tick_dt_ms = (now - last_tick) * 1000.0
                last_tick = now
                stats.n_ticks += 1
                stats.sum_dt_ms += tick_dt_ms
                stats.sum_dt2_ms += tick_dt_ms * tick_dt_ms
                if overrun * 1000.0 > stats.max_overrun_ms:
                    stats.max_overrun_ms = overrun * 1000.0

                if overrun > self._timing_threshold:
                    consecutive_violations += 1
                    stats.n_violations += 1
                    if consecutive_violations >= self._max_strikes:
                        env.disengage_safely()
                        # Discard recent transitions whose dt was stretched
                        # past the threshold. The previous (max_strikes-1)
                        # ticks were already over threshold and queued
                        # transitions with non-canonical dt; drop them so
                        # the critic sees only clean 10 ms transitions.
                        n_dropped = self.queue.discard_recent(self._max_strikes - 1)
                        raise TimingViolation(
                            f"Control thread fell {overrun*1000:.1f} ms "
                            f"behind for {self._max_strikes} consecutive "
                            f"ticks (threshold {self._timing_threshold*1000:.1f} ms). "
                            f"Disengaged motor and dropped {n_dropped} "
                            "stretched-dt transitions from the queue. "
                            "Likely cause: learner thread holding GIL too "
                            "long, or threshold set too tight."
                        )
                else:
                    consecutive_violations = 0

                if not slept:
                    # We were already behind before sleeping — reset to
                    # current time so the deficit doesn't burn future ticks
                    # one by one. If we DID sleep (even with overshoot),
                    # leave next_tick alone so future shorter sleeps absorb
                    # the small overshoot — keeps the long-run mean rate
                    # locked to control_freq_hz.
                    next_tick = now

                # 4. Observe (read state + filter velocities)
                try:
                    next_obs, reward, terminated, truncated, info = env.observe_and_step(a_applied)
                except (OSError, KeyboardInterrupt):
                    terminated = True
                    break

                # 5. Post the transition for the learner thread
                done = bool(terminated or truncated)
                # SB3's handle_timeout_termination uses this key to bootstrap
                # next_obs even when done=True (correct for time-limit truncations).
                info_for_buffer = dict(info)
                info_for_buffer["TimeLimit.truncated"] = bool(truncated and not terminated)

                self.queue.put(Transition(
                    obs=np.asarray(obs, dtype=np.float32).copy(),
                    action=np.asarray(a_applied, dtype=np.float32).reshape(-1),
                    reward=float(reward),
                    next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=info_for_buffer,
                ))

                cumulative_reward += float(reward)
                obs = next_obs
                n_steps += 1
        finally:
            env.disengage_safely()

        ep = EpisodeStats(
            episode_idx=episode_idx,
            n_steps=n_steps,
            cumulative_reward=cumulative_reward,
            terminated=terminated,
            truncated=truncated,
            tick_stats=stats,
        )
        self.last_episode_stats = ep
        return ep

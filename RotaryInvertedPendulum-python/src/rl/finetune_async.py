"""Phase 4 (proper): asynchronous SAC fine-tuning against the real device.

Replaces `finetune_real.py`'s use of `model.learn()`, which silently dropped
the control rate when gradient updates ran longer than `1/control_freq_hz`.
This orchestrator owns two threads:

  Main thread (this function):
    - Loads the SAC checkpoint and (optionally) a saved replay buffer.
    - Per episode: env.reset() → spawn control thread → drain queue and
      train while episode runs → wait for episode end → save checkpoints.
    - Holds the SIGINT handler (Python only delivers signals to the main
      thread). On Ctrl-C: set stop_flag → disengage motor → save artifacts.

  Control thread (one per episode, see `async_control.AsyncControlLoop`):
    - Strict 100 Hz loop: read state, predict via PolicySnapshot, command
      motor, post transition to TransitionQueue. Never blocks on the
      learner.

Three locks: `LowLevelClient._lock` (per-serial-transaction),
`replay_buffer_lock` (held during buffer.add and model.train),
`PolicySnapshot._lock` (around predict and refresh_from). See
docs/rl_transitions.md for the full architectural rationale.

Usage:
    python finetune_async.py \\
        --policy runs/<base>/best_model.zip \\
        --port /dev/cu.usbserial-1130 \\
        --episodes 50 --gradient-steps 4

Resume from a previous async session preserving the real-data buffer:
    python finetune_async.py \\
        --policy runs/<prev>/last.zip \\
        --resume-buffer runs/<prev>/replay_buffer.pkl \\
        --port /dev/cu.usbserial-1130 \\
        --episodes 50 --gradient-steps 4 \\
        --run-name async_continued
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import configure_logger

from async_control import (
    AsyncControlLoop,
    EpisodeStats,
    PolicySnapshot,
    TimingViolation,
    TransitionQueue,
)
from real_env import RealRotaryInvertedPendulumEnv


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"


def _add_to_buffer(model: SAC, transitions, lock: threading.Lock) -> int:
    """Drain a list of `Transition`s into model.replay_buffer under lock.

    Returns the number of transitions added. Updates model.num_timesteps
    so SAC's `learning_starts` and TB step counters advance correctly.
    """
    if not transitions:
        return 0
    n = 0
    with lock:
        for t in transitions:
            model.replay_buffer.add(
                obs=np.expand_dims(t.obs, axis=0),
                next_obs=np.expand_dims(t.next_obs, axis=0),
                action=np.expand_dims(t.action, axis=0),
                reward=np.array([t.reward], dtype=np.float32),
                done=np.array([t.terminated or t.truncated], dtype=np.bool_),
                infos=[t.info],
            )
            n += 1
        model.num_timesteps += n
    return n


def _save_artifacts(model: SAC, run_dir: Path, *, suffix: str = "") -> None:
    """Save policy + replay buffer atomically alongside any existing run."""
    pol_path = run_dir / (f"last{suffix}.zip" if not suffix else f"checkpoint_{suffix}.zip")
    buf_path = run_dir / (f"replay_buffer{suffix}.pkl" if not suffix else f"replay_buffer_{suffix}.pkl")
    model.save(pol_path)
    model.save_replay_buffer(buf_path)
    print(f"  saved policy → {pol_path.name}, buffer ({model.replay_buffer.size()} transitions) → {buf_path.name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True,
                   help="path to a sim-trained or previously-finetuned SAC checkpoint")
    p.add_argument("--port", default="/dev/cu.usbserial-1130",
                   help="serial port for the LowLevelServer Arduino")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--episodes", type=int, default=50,
                   help="number of real-device episodes to collect")
    p.add_argument("--episode-length-s", type=float, default=8.0)
    p.add_argument("--reset-settle-s", type=float, default=15.0,
                   help="max seconds to wait for the pendulum to come to rest "
                        "between episodes (motor disengaged, waiting for "
                        "operator + bearing damping). Polls until "
                        "|pen_vel| stays below threshold for ~1 s — then "
                        "tares and proceeds. On timeout, skips the tare "
                        "and proceeds anyway. 15 s gives the operator time "
                        "to stop the pendulum manually between episodes.")
    p.add_argument("--control-freq", type=float, default=40.0,
                   help="strict control rate in Hz; this orchestrator holds "
                        "this rate even at high --gradient-steps. 35 Hz is "
                        "the canonical operating point for this rig — see "
                        "docs/control_rate_selection.md.")
    p.add_argument("--max-accel-rad-s2", type=float, default=150.0,
                   help="action ∈ [-1, 1] maps to angular accel ∈ [-max, +max]"
                        " rad/s². Must match the value used at sim training time"
                        " (default 150 per current env).")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="SAC gradient updates per train() call. Higher "
                        "extracts more from each real transition")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-starts", type=int, default=100,
                   help="number of buffer transitions before training begins")
    p.add_argument("--checkpoint-freq", type=int, default=10,
                   help="save policy + replay buffer every N episodes")
    p.add_argument("--run-name", default=None,
                   help="run dir name under runs/. Default: timestamp")
    p.add_argument("--resume-buffer", default=None,
                   help="path to a replay_buffer.pkl from a previous async run "
                        "(do NOT load buffers from synchronous fine-tunes — "
                        "they were collected at the wrong rate)")
    p.add_argument("--device", default="cpu",
                   help="torch device. CPU is fine for this small policy")
    p.add_argument("--timing-violation-threshold-ms", type=float, default=10.0,
                   help="control-loop overrun above this for 3 consecutive "
                        "ticks raises TimingViolation and disengages motor")
    p.add_argument("--deterministic", action="store_true",
                   help="use deterministic actions during data collection. "
                        "Default is stochastic, matching SAC training-time "
                        "behaviour and what the policy was optimised for.")
    p.add_argument("--reward-action-rate-weight", type=float, default=None,
                   help="Mirror of train_sac.py's flag — penalty on action "
                        "delta. Must match the sim training value, otherwise "
                        "fine-tune gradient pulls the policy off the basin "
                        "found in sim.")
    p.add_argument("--reward-stillness-bonus-weight", type=float, default=None,
                   help="Mirror of train_sac.py's flag — multiplicative "
                        "stillness bonus near upright. Must match the sim "
                        "training value, otherwise fine-tune gradient pulls "
                        "the policy out of the stillness basin back toward "
                        "Kapitza on the rig.")
    p.add_argument("--dt-jitter-frac", type=float, default=0.05,
                   help="control-rate jitter (DR on dt). Each tick interval "
                        "is multiplied by uniform(1-frac, 1+frac). Mimics "
                        "the legacy variable-rate fine-tune that produced "
                        "the calm 'minimal action' attractor. Default 0.05 "
                        "matches the sim DR_CONTROL_DT_JITTER_FRAC constant — "
                        "sim and fine-tune should agree on the dt distribution. "
                        "Set 0.0 to disable for strict reproducible timing.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    torch.set_num_threads(1)  

    # ---- Setup ----
    run_name = args.run_name or f"async_finetune_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    env_kwargs = dict(
        port=args.port,
        baud=args.baud,
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        episode_length_s=args.episode_length_s,
        reset_settle_s=args.reset_settle_s,
    )
    if args.reward_action_rate_weight is not None:
        env_kwargs["reward_action_rate_weight"] = args.reward_action_rate_weight
    if args.reward_stillness_bonus_weight is not None:
        env_kwargs["reward_stillness_bonus_weight"] = args.reward_stillness_bonus_weight
    env = RealRotaryInvertedPendulumEnv(**env_kwargs)
    print(f"Control: {args.control_freq} Hz")

    print(f"Loading policy from {args.policy}")
    model = SAC.load(args.policy, env=env, device=args.device)
    # Override the loaded checkpoint's tensorboard_log (often a foreign
    # absolute path from the box that trained it).
    model.tensorboard_log = str(run_dir / "tb")
    model.learning_rate = args.learning_rate
  
    if args.resume_buffer:
        rb_path = Path(args.resume_buffer)
        if not rb_path.exists():
            print(f"ERROR: --resume-buffer path does not exist: {rb_path}", file=sys.stderr)
            return 2
        print(f"Loading replay buffer from {rb_path}")
        model.load_replay_buffer(str(rb_path))
        # Sanity check spaces match
        if model.replay_buffer.observation_space.shape != env.observation_space.shape:
            print("ERROR: loaded buffer's observation space does not match env's. "
                  "Was the buffer collected with a different obs construction?",
                  file=sys.stderr)
            return 2
        print(f"  loaded {model.replay_buffer.size()} transitions")

    # Reset num_timesteps to reflect actual real-data buffer size — checkpoints
    # carry their sim-training step counter (e.g. 280k from the curriculum)
    # which would falsely satisfy learning_starts before any real data
    # exists. Buffer size IS the right counter for our gating.
    model.num_timesteps = model.replay_buffer.size()
    model._total_timesteps = 0  # SB3 sets this in _setup_learn; placeholder.
    model._num_timesteps_at_start = model.num_timesteps

    # SB3's `model.train()` calls `self.logger`, which lazy-resolves to
    # `self._logger`. That attribute is set by `_setup_learn()` — which we
    # bypass since we're not calling `learn()`. Build an equivalent logger
    # by hand.
    model._logger = configure_logger(
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        tb_log_name="async",
        reset_num_timesteps=True,
    )

    # ---- Async machinery ----
    snapshot = PolicySnapshot(model.actor, device=args.device)
    queue = TransitionQueue(maxlen=4096)
    stop_flag = threading.Event()
    replay_buffer_lock = threading.Lock()

    # Signal handlers (main thread only — Python guarantee).
    def _on_signal(*_):
        if not stop_flag.is_set():
            print("\nSIGINT/SIGTERM received — shutting down cleanly.")
            stop_flag.set()
            env.disengage_safely()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ---- Telemetry ----
    timing_csv = run_dir / "timing.csv"
    timing_f = open(timing_csv, "w", newline="")
    timing_writer = csv.writer(timing_f)
    timing_writer.writerow([
        "episode", "n_steps", "cumulative_reward", "terminated", "truncated",
        "mean_dt_ms", "std_dt_ms", "max_overrun_ms", "n_violations",
        "learner_train_calls", "learner_total_train_s",
    ])
    timing_f.flush()

    def _log_episode(ep: EpisodeStats, n_train: int, train_s: float) -> None:
        timing_writer.writerow([
            ep.episode_idx, ep.n_steps, f"{ep.cumulative_reward:.3f}",
            int(ep.terminated), int(ep.truncated),
            f"{ep.mean_dt_ms:.3f}", f"{ep.std_dt_ms:.3f}",
            f"{ep.tick_stats.max_overrun_ms:.3f}", ep.tick_stats.n_violations,
            n_train, f"{train_s:.3f}",
        ])
        timing_f.flush()

    print(f"\nStarting async fine-tune: {args.episodes} episodes × "
          f"{args.episode_length_s} s @ {args.control_freq} Hz, "
          f"K={args.gradient_steps}, lr={args.learning_rate}")
    print(f"Buffer starts with {model.replay_buffer.size()} transitions, "
          f"learning_starts={args.learning_starts}")

    # Rolling window over the last 100 episodes for SB3-style running-mean
    # scalars (rollout/ep_rew_mean, rollout/ep_len_mean). Same default
    # window SB3's `_dump_logs` uses.
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)

    # ---- Main loop ----
    try:
        for ep_idx in range(args.episodes):
            if stop_flag.is_set():
                break
            print(f"\n[ep {ep_idx + 1}/{args.episodes}] reset (settle "
                  f"{args.reset_settle_s} s) → engage → run")
            try:
                env.reset()
            except Exception as e:
                print(f"ERROR during env.reset(): {e}", file=sys.stderr)
                break

            loop = AsyncControlLoop(
                env, snapshot, queue, stop_flag,
                control_freq_hz=args.control_freq,
                timing_violation_threshold_s=args.timing_violation_threshold_ms / 1000.0,
                deterministic_actions=args.deterministic,
                dt_jitter_frac=args.dt_jitter_frac,
            )
            ctrl_thread = threading.Thread(
                target=loop.run_episode,
                args=(ep_idx,),
                name=f"control-{ep_idx}",
                daemon=True,
            )
            ctrl_thread.start()

            n_train = 0
            train_s = 0.0
            try:
                while ctrl_thread.is_alive():
                    drained = queue.drain()
                    added = _add_to_buffer(model, drained, replay_buffer_lock)

                    can_train = (
                        model.num_timesteps > args.learning_starts
                        and model.replay_buffer.size() >= model.batch_size
                    )
                    if can_train:
                        t0 = time.monotonic()
                        with replay_buffer_lock:
                            model.train(gradient_steps=args.gradient_steps,
                                        batch_size=model.batch_size)
                        train_s += time.monotonic() - t0
                        n_train += 1
                        # Refresh the snapshot so the control thread picks
                        # up the new weights on the next predict() call.
                        snapshot.refresh_from(model.actor)
                    else:
                        # Warmup: avoid busy-loop while we have nothing to train on.
                        time.sleep(0.001)


                # Drain any straggler transitions after episode end.
                drained = queue.drain()
                _add_to_buffer(model, drained, replay_buffer_lock)
            finally:
                ctrl_thread.join(timeout=2.0)

            ep_stats = loop.last_episode_stats
            if ep_stats is None:
                print(f"  WARNING: episode {ep_idx} produced no stats "
                      "(thread crashed before completion).", file=sys.stderr)
                continue

            print(f"  steps={ep_stats.n_steps}  reward={ep_stats.cumulative_reward:+.1f}  "
                  f"mean_dt={ep_stats.mean_dt_ms:.2f}ms  "
                  f"max_overrun={ep_stats.tick_stats.max_overrun_ms:.2f}ms  "
                  f"violations={ep_stats.tick_stats.n_violations}  "
                  f"train_calls={n_train}  train_s={train_s:.2f}")
            _log_episode(ep_stats, n_train, train_s)

            # Flush logger to TB once per episode. SB3's `train()` records
            # scalars internally but doesn't dump — that normally happens in
            # learn()'s outer loop. We also append our own episode-level
            # metrics so reward/timing curves show up alongside SAC's
            # internals. The `rollout/ep_rew_mean` and `rollout/ep_len_mean`
            # keys match SB3's standard names so TB layouts that look for
            # them populate correctly.
            recent_rewards.append(ep_stats.cumulative_reward)
            recent_lengths.append(ep_stats.n_steps)
            ep_rew_mean = sum(recent_rewards) / len(recent_rewards)
            ep_len_mean = sum(recent_lengths) / len(recent_lengths)

            model.logger.record("rollout/ep_rew_mean", ep_rew_mean)
            model.logger.record("rollout/ep_len_mean", ep_len_mean)
            model.logger.record("rollout/ep_reward", ep_stats.cumulative_reward)
            model.logger.record("rollout/ep_length", ep_stats.n_steps)
            model.logger.record("rollout/episode", ep_idx + 1)
            model.logger.record("time/mean_tick_dt_ms", ep_stats.mean_dt_ms)
            model.logger.record("time/std_tick_dt_ms", ep_stats.std_dt_ms)
            model.logger.record("time/max_overrun_ms", ep_stats.tick_stats.max_overrun_ms)
            model.logger.record("time/n_timing_violations", ep_stats.tick_stats.n_violations)
            model.logger.record("time/learner_train_calls", n_train)
            model.logger.record("time/learner_total_train_s", train_s)
            model.logger.dump(step=model.num_timesteps)

            if (ep_idx + 1) % args.checkpoint_freq == 0:
                _save_artifacts(model, run_dir, suffix=f"ep{ep_idx + 1:03d}")

    finally:
        stop_flag.set()
        env.disengage_safely()
        try:
            env.close()
        except Exception:
            pass
        print("\nFinal save:")
        _save_artifacts(model, run_dir)
        timing_f.close()
        print(f"Run artifacts in: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

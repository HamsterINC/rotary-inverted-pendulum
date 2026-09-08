"""SAC trainer for the rotary inverted pendulum.

Trains an SB3 SAC policy against `RotaryInvertedPendulumEnv` and saves
checkpoints + the best model under `runs/<run_name>/`.

Usage:
    python train_sac.py                          # default 500k steps
    python train_sac.py --total-steps 1_000_000  # 1M
    python train_sac.py --resume runs/sac_2026-05-01/last.zip

After training, render a 30 s evaluation rollout in the MuJoCo viewer:
    python train_sac.py --eval runs/sac_2026-05-01/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pendulum_env import RotaryInvertedPendulumEnv


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"

class TanhActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Wrap the default linear action_net with Tanh
        self.action_net = nn.Sequential(
            self.action_net,
            nn.Tanh()
        )


def make_env(
    monitor_dir: Path | None = None,
    *,
    domain_randomization: bool = False,
    dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
    dr_action_delay_steps_range: tuple[int, int] | None = None,
    dr_action_lag_tau_range_s: tuple[float, float] | None = None,
    dr_control_dt_jitter_frac: float | None = None,
    control_freq_hz: float = 100.0,
    max_accel_rad_s2: float = 150.0,
    max_velocity_rad_s: float | None = None,
    reward_action_rate_weight: float | None = None,
    reward_motor_vel_weight: float | None = None,
    reward_motor_jerk_weight: float | None = None,
    reward_stillness_bonus_weight: float | None = None,
    dr_theta_bias_max_rad: float | None = None,
):
    def _thunk():
        env_kwargs = dict(
            domain_randomization=domain_randomization,
            dr_motor_accel_range_rad_s2=dr_motor_accel_range_rad_s2,
            dr_action_delay_steps_range=dr_action_delay_steps_range,
            dr_action_lag_tau_range_s=dr_action_lag_tau_range_s,
            dr_control_dt_jitter_frac=dr_control_dt_jitter_frac,
            control_freq_hz=control_freq_hz,
            max_accel_rad_s2=max_accel_rad_s2,
            reward_action_rate_weight=reward_action_rate_weight,
            reward_motor_jerk_weight=reward_motor_jerk_weight,
            reward_stillness_bonus_weight=reward_stillness_bonus_weight,
            dr_theta_bias_max_rad=dr_theta_bias_max_rad,
        )
        # These two have non-None defaults in the env; only pass when the
        # caller explicitly set a value, preserving env canonical defaults.
        if reward_motor_vel_weight is not None:
            env_kwargs["reward_motor_vel_weight"] = reward_motor_vel_weight
        if max_velocity_rad_s is not None:
            env_kwargs["max_velocity_rad_s"] = max_velocity_rad_s
        env = RotaryInvertedPendulumEnv(**env_kwargs)
        # Always wrap in Monitor so SB3's evaluate_policy can read canonical
        # episode reward/length. monitor_dir=None means in-memory only
        # (no CSV written) — used by the eval env.
        monitor_filename = str(monitor_dir / "monitor") if monitor_dir is not None else None
        env = Monitor(env, filename=monitor_filename)
        return env
    return _thunk


def train(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"ppo_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    dr_accel = (args.dr_accel_min, args.dr_accel_max) if args.dr_accel_max is not None else None
    dr_delay = (args.dr_delay_min, args.dr_delay_max) if args.dr_delay_max is not None else None
    dr_action_lag = (
        (args.dr_action_lag_tau_min, args.dr_action_lag_tau_max)
        if args.dr_action_lag_tau_max is not None else None
    )
    train_env = DummyVecEnv([make_env(
        run_dir,
        domain_randomization=args.domain_randomization,
        dr_motor_accel_range_rad_s2=dr_accel,
        dr_action_delay_steps_range=dr_delay,
        dr_action_lag_tau_range_s=dr_action_lag,
        dr_control_dt_jitter_frac=args.dr_dt_jitter_frac,
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_action_rate_weight=args.reward_action_rate_weight,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_jerk_weight=args.reward_motor_jerk_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
        dr_theta_bias_max_rad=args.dr_theta_bias_max_rad,
    )])
    # Eval env is always deterministic — no DR (no action-lag, no obs
    # noise) AND no theta-bias (so best_model is selected on the
    # bias-free reference scenario, not on a particular bias sample).
    eval_env = DummyVecEnv([make_env(
        domain_randomization=False,
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_action_rate_weight=args.reward_action_rate_weight,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_jerk_weight=args.reward_motor_jerk_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
        dr_theta_bias_max_rad=0.0,  # force bias-free eval reference
    )])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=args.device)
    else:
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Clean size for FPGA deployment
            log_std_init=-1.0,                             # Lower initial exploration noise (std ~0.36)
        )
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.001,
            clip_range=0.2,
            clip_range_vf=0.2,
            n_epochs=5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
        )

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ])

    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
        reset_num_timesteps=not args.resume,
    )

    final_path = run_dir / "last.zip"
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    return run_dir


def evaluate(args: argparse.Namespace) -> None:
    print(f"Loading {args.eval}")
    # Eval env must match the training config — control rate especially,
    # since a 75 Hz-trained policy run at 35 Hz produces garbage. Reward
    # weights don't affect inference but we pass them for cleaner reward
    # reporting in the eval log. Bias DR explicitly disabled (0.0) so
    # the eval runs against the bias-free deterministic reference (same
    # as the train-time best_model eval).
    # Only pass kwargs whose CLI value is not None — env constructor has
    # non-None defaults for some of these (e.g. max_velocity_rad_s, which
    # crashes if None reaches it). Match the make_env() pattern.
    env_kwargs = dict(
        render_mode="human",
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_action_rate_weight=args.reward_action_rate_weight,
        reward_motor_jerk_weight=args.reward_motor_jerk_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        dr_theta_bias_max_rad=0.0,
    )
    if args.reward_motor_vel_weight is not None:
        env_kwargs["reward_motor_vel_weight"] = args.reward_motor_vel_weight
    if args.max_velocity_rad_s is not None:
        env_kwargs["max_velocity_rad_s"] = args.max_velocity_rad_s
    env = RotaryInvertedPendulumEnv(**env_kwargs)
    model = PPO.load(args.eval, device=args.device)

    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    n_steps = 0
    target_steps = int(args.eval_seconds * env.control_freq_hz)
    # Pace the loop to wall clock so 1 sim second = 1 real second; otherwise
    # mj_step + predict run sub-millisecond and the viewer flashes shut.
    dt = 1.0 / env.control_freq_hz
    next_tick = time.monotonic()

    try:
        while n_steps < target_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            n_steps += 1
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()
            next_tick += dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We fell behind real time (rare on CPU but possible if the
                # viewer is slow); resync without compounding the lag.
                next_tick = time.monotonic()
    finally:
        env.close()

    print(f"Eval: {n_steps} steps, total reward {total_reward:.2f}, "
          f"mean per step {total_reward / n_steps:.4f}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on the rotary inverted pendulum")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="how often (env steps) to run the eval callback")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto",
                   help="torch device (cpu, cuda, mps, auto)")
    p.add_argument("--run-name", default=None,
                   help="run dir name under runs/. Default: timestamp.")
    p.add_argument("--resume", default=None, help="path to a .zip to resume from")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--domain-randomization", action="store_true",
                   help="enable Phase 2 randomisation: motor lag, action delay, "
                        "physics randomisation, observation noise. Eval env stays "
                        "deterministic.")
    # Curriculum-learning DR overrides. If unset, env uses module constants.
    p.add_argument("--dr-accel-min", type=float, default=None,
                   help="lower bound on motor_max_accel_rad_s2 sampled per episode")
    p.add_argument("--dr-accel-max", type=float, default=None,
                   help="upper bound on motor_max_accel_rad_s2. Set this to "
                        "override env defaults.")
    p.add_argument("--dr-delay-min", type=int, default=0,
                   help="lower bound on action_delay_steps sampled per episode")
    p.add_argument("--dr-delay-max", type=int, default=None,
                   help="upper bound on action_delay_steps. Set this to override env defaults.")
    p.add_argument("--dr-action-lag-tau-min", type=float, default=0.0,
                   help="lower bound on first-order action-lag time constant "
                        "(seconds) sampled per episode.")
    p.add_argument("--dr-action-lag-tau-max", type=float, default=None,
                   help="upper bound on first-order action-lag time constant "
                        "(seconds). Continuous analogue of --dr-delay-max. "
                        "Set this to override env defaults. See "
                        "docs/transport_delay.md.")
    p.add_argument("--control-freq", type=float, default=100.0,
                   help="sim control rate (Hz). Must match the rate used in "
                        "fine-tuning and deployment. 35 Hz is the empirically-best "
                        "operating point for this rig — see "
                        "docs/control_rate_selection.md for the principled selection.")
    p.add_argument("--max-accel-rad-s2", type=float, default=150.0,
                   help="action ∈ [-1, 1] maps to angular accel ∈ [-max, +max]"
                        " rad/s². Default 150 ≈ 76% of the motor's physical "
                        "envelope (~196 rad/s² at 50 kSteps/s²). Bumped from "
                        "100 after observing the policy saturating accel_cmd "
                        "in the first accel-mode deployment.")
    p.add_argument("--max-velocity-rad-s", type=float, default=None,
                   help="motor angular-velocity saturation cap (rad/s). "
                        "Default None → env default (5.0). Lower values "
                        "force the policy below the Kapitza parametric "
                        "stabilisation regime, which requires a·ω above "
                        "a threshold proportional to sqrt(2gL). Capping "
                        "below the rig's natural Kapitza window directly "
                        "disrupts resonance-pumping policies.")
    p.add_argument("--reward-motor-vel-weight", type=float, default=None,
                   help="penalty on motor_vel² in the reward. Default None "
                        "→ env default (0.005). Bumping to e.g. 0.05 makes "
                        "the optimizer prefer policies that keep the motor "
                        "still, not just the pendulum upright. Targets the "
                        "'chattery but balanced' attractor directly.")
    p.add_argument("--dr-theta-bias-max-rad", type=float, default=None,
                   help="Per-episode pendulum encoder θ-bias DR range "
                        "(rad). Default None → env default "
                        "(DR_THETA_BIAS_MAX_RAD = 0.05, i.e. ±2.9°, "
                        "covering the rig's measured ±1.9° rest band "
                        "with headroom). Active in ALL stages "
                        "independent of --domain-randomization, because "
                        "encoder bias is always present on the rig and "
                        "the policy must be robust to it from stage 1. "
                        "Set 0.0 to disable explicitly (eval env auto-"
                        "uses 0.0 for a deterministic reference).")
    p.add_argument("--reward-stillness-bonus-weight", type=float, default=None,
                   help="Multiplicative stillness bonus weight. Default None "
                        "→ 0 (disabled, canonical Quanser reward). When set "
                        ">0, ADDS k · exp(-θ²/σ_θ²) · exp(-α̇²/σ_v²) to the "
                        "reward. The product means a high bonus requires "
                        "BOTH theta and motor_vel near zero simultaneously, "
                        "directly penalising Kapitza-style resonance "
                        "stabilisation (which has α̇ ≈ 3 rad/s during "
                        "balance). Suggested starting value: 5.0.")
    p.add_argument("--reward-motor-jerk-weight", type=float, default=None,
                   help="penalty on (motor_vel_t - motor_vel_{t-1})² in "
                        "the reward — physical motor jerk. NOT in the "
                        "Quanser paper; default None → env default (0.0, "
                        "disabled). Distinct from --reward-action-rate-weight "
                        "(command jerk). Try 0.01 as a gentle starting point.")
    p.add_argument("--reward-action-rate-weight", type=float, default=None,
                   help="penalty on (action_t - action_{t-1})² in the reward. "
                        "Default None → env default (0.0; disabled in accel "
                        "mode). Re-enabling with a small value (e.g. 0.02) "
                        "discourages chatter — risk is the 'entropy collapse "
                        "into low-reward basin' failure mode that motivated "
                        "the original disable in position mode; test on a "
                        "short run before committing to a full curriculum.")
    p.add_argument("--dr-dt-jitter-frac", type=float, default=None,
                   help="DR magnitude on control timestep. Each tick the "
                        "physics step count is multiplied by uniform "
                        "(1-frac, 1+frac). Empirically protects SAC from the "
                        "'active correction' attractor on this rig. "
                        "Default (None) uses DR_CONTROL_DT_JITTER_FRAC=0.05 "
                        "from pendulum_env. Set 0.0 to disable.")
    p.add_argument("--eval", default=None,
                   help="if set, skip training and render an eval rollout from this checkpoint")
    p.add_argument("--eval-seconds", type=float, default=30.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.eval:
        evaluate(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

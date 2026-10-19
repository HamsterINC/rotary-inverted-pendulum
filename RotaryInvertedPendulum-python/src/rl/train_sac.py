"""SAC trainer for the rotary inverted pendulum.

Trains an SB3 SAC policy against `RotaryInvertedPendulumEnv` and saves
checkpoints + the best model under `runs/<run_name>/`.

Usage:
    python train_sac.py                          # default 180k steps
    python train_sac.py --total-steps 1_000_000  # 1M
    python train_sac.py --obs-norm orig          # pick an observation preset
    python train_sac.py --resume runs/sac_2026-05-01/last.zip

Each run writes runs/<run_name>/config.json (obs_norm, control rate, ...).
--eval and --resume read it back, so the observation preset used for
training is applied automatically.

After training, render a 30 s evaluation rollout in the MuJoCo viewer:
    python train_sac.py --eval runs/sac_2026-05-01/best_model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pendulum_env import OBS_NORM_PRESETS, RotaryInvertedPendulumEnv


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"
CONFIG_NAME = "config.json"
DEFAULT_OBS_NORM = "raw"


# --- config.json helpers ----------------------------------------------------

def find_config(model_path: str | Path) -> Path | None:
    """Locate config.json for a saved model.

    Looks next to the model (best_model.zip, last.zip) and one level up
    (checkpoints/sac_*_steps.zip live in run_dir/checkpoints/).
    """
    p = Path(model_path).resolve()
    for d in (p.parent, p.parent.parent):
        cfg = d / CONFIG_NAME
        if cfg.exists():
            return cfg
    return None


def load_config(model_path: str | Path) -> dict | None:
    cfg = find_config(model_path)
    if cfg is None:
        return None
    with open(cfg) as f:
        return json.load(f)


def resolve_obs_norm(cli_value: str | None, model_path: str | Path | None, *, what: str) -> str:
    """Decide which obs_norm preset to use.

    - config.json found, CLI unset  -> use config value
    - config.json found, CLI set    -> must match, else error
    - no config.json, CLI set       -> use CLI value
    - no config.json, CLI unset     -> fall back to DEFAULT_OBS_NORM with a warning
    """
    cfg = load_config(model_path) if model_path is not None else None
    cfg_value = cfg.get("obs_norm") if cfg else None

    if cfg_value is not None:
        if cli_value is not None and cli_value != cfg_value:
            raise SystemExit(
                f"--obs-norm {cli_value!r} does not match the preset this model was "
                f"trained with ({cfg_value!r}, from {find_config(model_path)}). "
                f"Drop --obs-norm to use the saved one."
            )
        print(f"[{what}] obs_norm={cfg_value!r} (from {find_config(model_path)})")
        return cfg_value

    if cli_value is not None:
        print(f"[{what}] obs_norm={cli_value!r} (from CLI; no config.json found)")
        return cli_value

    print(
        f"[{what}] WARNING: no config.json next to {model_path} and no --obs-norm given; "
        f"assuming {DEFAULT_OBS_NORM!r}. If the model was trained with another preset, "
        f"pass --obs-norm explicitly."
    )
    return DEFAULT_OBS_NORM


def write_config(run_dir: Path, args: argparse.Namespace, obs_norm: str) -> None:
    cfg = {
        "obs_norm": obs_norm,
        "control_freq": args.control_freq,
        "max_accel_rad_s2": args.max_accel_rad_s2,
        "max_velocity_rad_s": args.max_velocity_rad_s,
        "lr": args.lr,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "domain_randomization": args.domain_randomization,
        "resumed_from": args.resume,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: v for k, v in vars(args).items()},
    }
    with open(run_dir / CONFIG_NAME, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"Wrote {run_dir / CONFIG_NAME}")


# --- env factory ------------------------------------------------------------

def make_env(
    monitor_dir: Path | None = None,
    *,
    domain_randomization: bool = False,
    dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
    dr_action_delay_steps_range: tuple[int, int] | None = None,
    dr_action_lag_tau_range_s: tuple[float, float] | None = None,
    dr_control_dt_jitter_frac: float | None = None,
    control_freq_hz: float = 35.0,
    max_accel_rad_s2: float = 150.0,
    max_velocity_rad_s: float | None = None,
    reward_action_rate_weight: float | None = None,
    reward_motor_vel_weight: float | None = None,
    reward_motor_jerk_weight: float | None = None,
    reward_stillness_bonus_weight: float | None = None,
    dr_theta_bias_max_rad: float | None = None,
    obs_norm: str = DEFAULT_OBS_NORM,
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
            obs_norm=obs_norm,
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


# --- training ---------------------------------------------------------------

def train(args: argparse.Namespace) -> Path:
    run_name = args.run_name or f"sac_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Resolve the observation preset. On --resume the preset must match the
    # one the checkpoint was trained with (config.json next to it).
    obs_norm = resolve_obs_norm(args.obs_norm, args.resume, what="train")
    if args.resume is None and args.obs_norm is None:
        obs_norm = DEFAULT_OBS_NORM  # fresh run, nothing to inherit
    write_config(run_dir, args, obs_norm)

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
        obs_norm=obs_norm,
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
        obs_norm=obs_norm,
    )])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(args.resume, env=train_env, device=args.device)
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128]),  # Clean size for FPGA deployment
        )
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            buffer_size=200_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            ent_coef="auto",
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
            name_prefix="sac",
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


# --- evaluation -------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    print(f"Loading {args.eval}")

    # Observation preset comes from the run's config.json; a conflicting
    # --obs-norm is an error rather than silently feeding the policy
    # differently scaled inputs.
    obs_norm = resolve_obs_norm(args.obs_norm, args.eval, what="eval")

    cfg = load_config(args.eval)
    if cfg and cfg.get("control_freq") is not None and cfg["control_freq"] != args.control_freq:
        print(
            f"WARNING: model was trained at control_freq={cfg['control_freq']} Hz "
            f"but --control-freq={args.control_freq}. A policy run at the wrong "
            f"rate produces garbage."
        )

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
        obs_norm=obs_norm,
    )
    if args.reward_motor_vel_weight is not None:
        env_kwargs["reward_motor_vel_weight"] = args.reward_motor_vel_weight
    if args.max_velocity_rad_s is not None:
        env_kwargs["max_velocity_rad_s"] = args.max_velocity_rad_s
    env = RotaryInvertedPendulumEnv(**env_kwargs)
    model = SAC.load(args.eval, device=args.device)

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


# --- CLI --------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on the rotary inverted pendulum")
    p.add_argument("--total-steps", type=int, default=180_000)
    p.add_argument("--eval-freq", type=int, default=20_000,
                   help="how often (env steps) to run the eval callback")
    p.add_argument("--checkpoint-freq", type=int, default=60_000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="SAC learning rate (actor, critic, entropy coef).")
    p.add_argument("--device", default="cuda:4",
                   help="torch device (cpu, cuda, mps, auto)")
    p.add_argument("--run-name", default=None,
                   help="run dir name under runs/. Default: timestamp.")
    p.add_argument("--resume", default=None, help="path to a .zip to resume from")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--obs-norm", default=None, choices=list(OBS_NORM_PRESETS),
                   help="Observation normalization preset (see OBS_NORM_PRESETS in "
                        "pendulum_env). Default: 'raw' for a fresh training run. "
                        "For --eval and --resume the preset is read from the run's "
                        "config.json; passing a different value here is an error. "
                        "The same preset must be used on the rig (real_env.py).")
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
    p.add_argument("--dr-action-lag-tau-max", type=float, default=0.0,
                   help="upper bound on first-order action-lag time constant "
                        "(seconds). Continuous analogue of --dr-delay-max. "
                        "Set this to override env defaults. See "
                        "docs/transport_delay.md.")
    p.add_argument("--control-freq", type=float, default=50.0,
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
    p.add_argument("--max-velocity-rad-s", type=float, default=5.0,
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
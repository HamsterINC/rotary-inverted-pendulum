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

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import Training.symmetry as symmetry
from Simulation.pendulum_env import (
    DR_TRANSPORT_DELAY_RANGE_S,
    OBS_NORM_PRESETS,
    MAX_VELOCITY_RAD_S,
    MOTOR_MICROSTEPS,
    RotaryInvertedPendulumEnv,
)
from run_config import check_config, save_run_config


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"


def _resolved_config(args: argparse.Namespace) -> dict:
    """The config knobs that must match across train/fine-tune/deploy,
    with None CLI values resolved to the env defaults they map to."""
    return {
        "control_freq_hz": float(args.control_freq),
        "max_accel_rad_s2": float(args.max_accel_rad_s2),
        "max_velocity_rad_s": float(
            args.max_velocity_rad_s if args.max_velocity_rad_s is not None
            else MAX_VELOCITY_RAD_S
        ),
        "reward_alive_offset": float(args.reward_alive_offset),
        "reward_upright_alive_weight": float(args.reward_upright_alive_weight),
        "reward_stillness_bonus_weight": float(args.reward_stillness_bonus_weight or 0.0),
        "reward_motor_vel_weight": (
            float(args.reward_motor_vel_weight)
            if args.reward_motor_vel_weight is not None else 0.005
        ),
        "reward_motor_pos_weight": (
            float(args.reward_motor_pos_weight)
            if args.reward_motor_pos_weight is not None else 0.5
        ),
        "obs_history_len": int(args.obs_history_len),
        "obs_include_velocities": not args.drop_velocity_obs,
        "firmware_obs_model": bool(args.firmware_obs_model),
        "action_smooth_window": int(args.action_smooth_window),
        "net-arch": tuple(args.net_arch),
        # The motor-step quantisation the firmware measurement model applied.
        # Only bites when firmware_obs_model is on, but recorded either way:
        # a policy trained against a coarser step LSB than it deploys at sees
        # less velocity noise than it trained on (benign), while the reverse
        # is what makes a rig-swapped policy misbehave.
        "motor_microsteps": int(MOTOR_MICROSTEPS),
        # Which rig's measured friction the sim was built from. Recorded so a
        # checkpoint says which physical rig it was trained for, and so
        # distill / DAgger / analyze_sim inherit the same rig without the
        # operator having to remember. Provenance only (see run_config).
        "params_path": (str(args.params_path)
                        if args.params_path is not None else None),
        # Provenance only (see run_config.PROVENANCE_ONLY_KEYS): mirror
        # augmentation changes what the buffer holds, not the obs/action
        # layout or the objective, so it is legal to switch between stages.
        "mirror_augment": bool(args.mirror_augment),
        "obs_norm": str(args.obs_norm),
    }


def _nominal_rig_kwargs(args: argparse.Namespace) -> dict:
    """Env kwargs that pin the deterministic eval/render env to the NOMINAL RIG.

    The reference scenario is the nominal rig, not an idealised zero-delay
    plant. Selecting best_model at zero delay when every training episode has
    some delay picks checkpoints out-of-distribution (observed on vel_v7:
    stage-3 rollout reward climbed while the zero-delay eval declined).

    - Transport delay: with DR on, the midpoint of the DR range actually used
      in training (explicit CLI range, else the env default); with DR off, the
      fixed `--transport-delay-s` the training env uses.
    - Observation staleness: if explicitly randomised, the DR midpoint;
      otherwise the env nominal.
    """
    if args.domain_randomization:
        lo, hi = (
            (args.dr_transport_delay_min, args.dr_transport_delay_max)
            if args.dr_transport_delay_max is not None
            else DR_TRANSPORT_DELAY_RANGE_S
        )
        delay_s = (lo + hi) / 2.0
    else:
        delay_s = float(args.transport_delay_s)
    kwargs = {"transport_delay_s": delay_s}
    if args.dr_obs_staleness_max is not None:
        kwargs["obs_staleness_s"] = (
            args.dr_obs_staleness_min + args.dr_obs_staleness_max
        ) / 2.0
    return kwargs


def make_env(
    monitor_dir: Path | None = None,
    *,
    domain_randomization: bool = False,
    dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
    dr_transport_delay_range_s: tuple[float, float] | None = None,
    dr_control_dt_jitter_frac: float | None = None,
    control_freq_hz: float = 50.0,
    max_accel_rad_s2: float = 150.0,
    max_velocity_rad_s: float | None = None,
    reward_motor_vel_weight: float | None = None,
    reward_motor_pos_weight: float | None = None,
    reward_stillness_bonus_weight: float | None = None,
    reward_alive_offset: float | None = None,
    reward_upright_alive_weight: float | None = None,
    dr_theta_bias_max_rad: float | None = None,
    upright_reset_frac: float = 0.0,
    obs_history_len: int = 4,
    obs_include_velocities: bool = True,
    firmware_obs_model: bool = False,
    transport_delay_s: float = 0.0,
    action_smooth_window: int = 4,
    dr_obs_staleness_range_s: tuple[float, float] | None = None,
    obs_staleness_s: float | None = None,
    params_path: str | Path | None = None,
    obs_norm: str = "raw",
):
    def _thunk():
        env_kwargs = dict(
            params_path=params_path,
            upright_reset_frac=upright_reset_frac,
            obs_history_len=obs_history_len,
            obs_include_velocities=obs_include_velocities,
            firmware_obs_model=firmware_obs_model,
            transport_delay_s=transport_delay_s,
            action_smooth_window=action_smooth_window,
            reward_alive_offset=reward_alive_offset,
            reward_upright_alive_weight=reward_upright_alive_weight,
            domain_randomization=domain_randomization,
            dr_motor_accel_range_rad_s2=dr_motor_accel_range_rad_s2,
            dr_transport_delay_range_s=dr_transport_delay_range_s,
            dr_control_dt_jitter_frac=dr_control_dt_jitter_frac,
            control_freq_hz=control_freq_hz,
            max_accel_rad_s2=max_accel_rad_s2,
            reward_stillness_bonus_weight=reward_stillness_bonus_weight,
            dr_theta_bias_max_rad=dr_theta_bias_max_rad,
            obs_norm=obs_norm,
        )
        if dr_obs_staleness_range_s is not None:
            env_kwargs["dr_obs_staleness_range_s"] = dr_obs_staleness_range_s
        if obs_staleness_s is not None:
            env_kwargs["obs_staleness_s"] = obs_staleness_s
        # These have non-None defaults in the env; only pass when the
        # caller explicitly set a value, preserving env canonical defaults.
        if reward_motor_vel_weight is not None:
            env_kwargs["reward_motor_vel_weight"] = reward_motor_vel_weight
        if reward_motor_pos_weight is not None:
            env_kwargs["reward_motor_pos_weight"] = reward_motor_pos_weight
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
    # Guard resume against silently switching the must-match knobs
    # (checked before creating the run dir so an abort leaves no debris).
    config = _resolved_config(args)
    if args.resume:
        check_config(args.resume, config, ignore=args.ignore_config_mismatch)

    run_name = args.run_name or f"sac_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Record the knobs so deploy/fine-tune can validate their flags.
    save_run_config(run_dir, config)
    obs_norm = args.obs_norm
    dr_accel = (args.dr_accel_min, args.dr_accel_max) if args.dr_accel_max is not None else None
    dr_delay = (
        (args.dr_transport_delay_min, args.dr_transport_delay_max)
        if args.dr_transport_delay_max is not None else None
    )
    # Opt-in only. The measured rig value is 15.6 ms, outside the env default
    # range of 2-10 ms, but widening it changes which checkpoint EvalCallback
    # picks, so the default is left alone until a from-scratch run validates
    # it. See website/src/content/docs/reference/transport-delay.md.
    dr_stale = (
        (args.dr_obs_staleness_min, args.dr_obs_staleness_max)
        if args.dr_obs_staleness_max is not None else None
    )
    train_env = DummyVecEnv([make_env(
        run_dir,
        params_path=args.params_path,
        domain_randomization=args.domain_randomization,
        dr_motor_accel_range_rad_s2=dr_accel,
        dr_transport_delay_range_s=dr_delay,
        dr_obs_staleness_range_s=dr_stale,
        dr_control_dt_jitter_frac=args.dr_dt_jitter_frac,
        transport_delay_s=args.transport_delay_s,  # used when DR is off
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_pos_weight=args.reward_motor_pos_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        reward_alive_offset=args.reward_alive_offset,
        reward_upright_alive_weight=args.reward_upright_alive_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
        dr_theta_bias_max_rad=args.dr_theta_bias_max_rad,
        upright_reset_frac=args.upright_reset_frac,
        obs_history_len=args.obs_history_len,
        obs_include_velocities=not args.drop_velocity_obs,
        firmware_obs_model=args.firmware_obs_model,
        action_smooth_window=args.action_smooth_window,
        obs_norm=obs_norm,
    )])
    # Eval env is always deterministic — no DR (no obs noise, no random
    # delay) AND no theta-bias (so best_model is selected on the bias-free
    # reference scenario, not on a particular bias sample). It also keeps
    # hanging-only resets (upright_reset_frac=0) so the eval score always
    # reflects the full swing-up + balance task.
    #
    # Transport delay and staleness are NOT zeroed: the reference scenario
    # is the NOMINAL RIG, so `_nominal_rig_kwargs` pins them to the midpoint
    # of the training DR ranges (see its docstring for why).
    eval_env = DummyVecEnv([make_env(
        params_path=args.params_path,
        domain_randomization=False,
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_pos_weight=args.reward_motor_pos_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        reward_alive_offset=args.reward_alive_offset,
        reward_upright_alive_weight=args.reward_upright_alive_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
        dr_theta_bias_max_rad=0.0,  # force bias-free eval reference
        obs_history_len=args.obs_history_len,
        obs_include_velocities=not args.drop_velocity_obs,
        # Part of the plant model, not DR: the deterministic eval reference
        # should score the policy against the same measurement pipeline it
        # was trained for (quantisation + nominal staleness; no noise here).
        firmware_obs_model=args.firmware_obs_model,
        action_smooth_window=args.action_smooth_window,
        obs_norm=obs_norm,
        **_nominal_rig_kwargs(args),
    )])

    # γ sets the effective horizon in *steps*, so a fixed 0.99 silently
    # shortens the time horizon when the control rate goes up (~2.9 s at
    # 35 Hz but ~2.0 s at 50 Hz). Unless --gamma is given, hold the
    # canonical 35 Hz horizon constant: gamma = 0.99^(35/f_ctrl).
    gamma = (
        float(args.gamma) if args.gamma is not None
        else 0.99 ** (35.0 / args.control_freq)
    )
    print(f"gamma = {gamma:.4f} "
          f"({'explicit' if args.gamma is not None else f'auto-derived for {args.control_freq:g} Hz'})")

    # Mirror data augmentation (the DUP method): every transition is stored
    # alongside its mirror image, so the policy sees both halves of the state
    # space instead of whichever half it happens to prefer. Exact, not
    # approximate — see symmetry.py. Each add() takes two slots, so the
    # buffer is doubled to hold the same span of experience as a baseline run.
    buffer_size = 400_000 if args.mirror_augment else 200_000
    replay_buffer_class = None
    replay_buffer_kwargs = None
    if args.mirror_augment:
        obs_sign = symmetry.obs_signs(
            obs_history_len=args.obs_history_len,
            obs_include_velocities=not args.drop_velocity_obs,
        )
        replay_buffer_class = symmetry.MirrorReplayBuffer
        replay_buffer_kwargs = {"obs_sign": obs_sign}
        print(f"mirror augmentation ON — buffer_size {buffer_size} "
              f"(2 slots per env step)")

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(args.resume, env=train_env, device=args.device)
        if abs(model.gamma - gamma) > 1e-6:
            print(f"  overriding checkpoint gamma {model.gamma:.4f} → {gamma:.4f} "
                  "(constant-horizon policy; pass --gamma to pin a value)")
            model.gamma = gamma
        # SAC.load rebuilds an EMPTY buffer of the checkpoint's buffer class
        # (the buffer itself is never saved), so swapping the class here loses
        # nothing and lets --mirror-augment be turned on or off per stage.
        want_mirror = bool(args.mirror_augment)
        has_mirror = isinstance(model.replay_buffer, symmetry.MirrorReplayBuffer)
        if want_mirror != has_mirror:
            model.replay_buffer_class = replay_buffer_class or ReplayBuffer
            model.replay_buffer_kwargs = dict(replay_buffer_kwargs or {})
            model.buffer_size = buffer_size
            model.replay_buffer = model.replay_buffer_class(
                buffer_size,
                model.observation_space,
                model.action_space,
                device=model.device,
                n_envs=model.n_envs,
                optimize_memory_usage=model.optimize_memory_usage,
                **model.replay_buffer_kwargs,
            )
            print(f"  replay buffer rebuilt: mirror augmentation "
                  f"{'ON' if want_mirror else 'OFF'}")
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.lr,
            buffer_size=buffer_size,
            batch_size=256,
            tau=0.005,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            use_sde=args.use_sde,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=(dict(net_arch=args.net_arch) if args.net_arch else None),
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


def evaluate(args: argparse.Namespace) -> None:
    check_config(args.eval, _resolved_config(args),
                 ignore=args.ignore_config_mismatch)
    print(f"Loading {args.eval}")
    # Eval env must match the training config — control rate and obs_norm
    # especially, since a policy run at the wrong rate or with the wrong
    # normalisation produces garbage. Reward weights don't affect inference
    # but we pass them for cleaner reward reporting in the eval log. Bias DR
    # explicitly disabled (0.0) so the eval runs against the bias-free
    # deterministic reference (same as the train-time best_model eval), and
    # delay/staleness are pinned to the nominal rig via _nominal_rig_kwargs.
    # Only pass kwargs whose CLI value is not None — env constructor has
    # non-None defaults for some of these (e.g. max_velocity_rad_s, which
    # crashes if None reaches it). Match the make_env() pattern.
    env_kwargs = dict(
        render_mode="human",
        params_path=args.params_path,
        control_freq_hz=args.control_freq,
        obs_history_len=args.obs_history_len,
        obs_include_velocities=not args.drop_velocity_obs,
        firmware_obs_model=args.firmware_obs_model,
        action_smooth_window=args.action_smooth_window,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        reward_alive_offset=args.reward_alive_offset,
        reward_upright_alive_weight=args.reward_upright_alive_weight,
        dr_theta_bias_max_rad=0.0,
        obs_norm=args.obs_norm,
        **_nominal_rig_kwargs(args),
    )
    if args.reward_motor_vel_weight is not None:
        env_kwargs["reward_motor_vel_weight"] = args.reward_motor_vel_weight
    if args.reward_motor_pos_weight is not None:
        env_kwargs["reward_motor_pos_weight"] = args.reward_motor_pos_weight
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on the rotary inverted pendulum")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="how often (env steps) to run the eval callback")
    p.add_argument("--checkpoint-freq", type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="SAC learning rate (actor, critic, entropy coef).")
    p.add_argument("--device", default="auto",
                   help="torch device (cpu, cuda, mps, auto)")
    p.add_argument("--run-name", default=None,
                   help="run dir name under runs/. Default: timestamp.")
    p.add_argument("--resume", default=None, help="path to a .zip to resume from")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--obs-norm", default="raw", choices=list(OBS_NORM_PRESETS),
                   help="Observation normalization preset (see OBS_NORM_PRESETS in "
                        "pendulum_env). Default: 'raw' for a fresh training run. "
                        "For --eval and --resume the preset is read from the run's "
                        "config.json; passing a different value here is an error. "
                        "The same preset must be used on the rig (real_env.py).")
    p.add_argument("--domain-randomization", action="store_true",
                   help="enable Phase 2 randomisation: motor accel envelope, "
                        "residual transport delay, physics randomisation, "
                        "observation noise. Eval env stays deterministic.")
    # Curriculum-learning DR overrides. If unset, env uses module constants.
    p.add_argument("--dr-accel-min", type=float, default=None,
                   help="lower bound on motor_max_accel_rad_s2 sampled per episode")
    p.add_argument("--dr-accel-max", type=float, default=None,
                   help="upper bound on motor_max_accel_rad_s2. Set this to "
                        "override env defaults.")
    p.add_argument("--dr-obs-staleness-min", type=float, default=0.002,
                   help="lower bound of the sampled observation staleness (s); "
                        "only used when --dr-obs-staleness-max is given")
    p.add_argument("--dr-obs-staleness-max", type=float, default=None,
                   help="upper bound of the sampled observation staleness (s). "
                        "Unset keeps the env default 2-10 ms. Measured on the "
                        "rig: 15.6 ms, so 0.020 covers reality — UNVALIDATED, "
                        "needs a from-scratch run before becoming the default")
    p.add_argument("--transport-delay-s", type=float, default=0.0,
                   help="fixed RESIDUAL transport delay (s) applied after the "
                        "firmware boxcar when DR is OFF (and the eval "
                        "reference when DR is off). Total sim delay = boxcar "
                        "(N-1)/2 ticks + this. Measure end-to-end with the "
                        "boxcar on, subtract (N-1)/2 ticks. See "
                        "website/src/content/docs/reference/transport-delay.md.")
    p.add_argument("--dr-transport-delay-min", type=float, default=0.0,
                   help="lower bound on the residual transport delay (s) "
                        "sampled per episode (DR on). Only used when "
                        "--dr-transport-delay-max is given.")
    p.add_argument("--dr-transport-delay-max", type=float, default=None,
                   help="upper bound on the residual transport delay (s). "
                        "Unset keeps the env default range "
                        "(DR_TRANSPORT_DELAY_RANGE_S, 0-20 ms). Fractional "
                        "ticks are supported (linear interpolation).")
    p.add_argument("--control-freq", type=float, default=50.0,
                   help="sim control rate (Hz). Must match the rate used in "
                        "fine-tuning and deployment. 50 Hz is the empirically-best "
                        "operating point for this rig — see "
                        "website/src/content/docs/reference/control-rate.md for the principled selection.")
    p.add_argument("--max-accel-rad-s2", type=float, default=150.0,
                   help="action ∈ [-1, 1] maps to angular accel ∈ "
                        "[-max, +max] rad/s². Default 150 ≈ 76%% of the motor's "
                        "physical envelope (~196 rad/s² at 50 kSteps/s²). Bumped "
                        "from 100 after observing the policy saturating accel_cmd "
                        "in the first accel-mode deployment.")
    p.add_argument("--max-velocity-rad-s", type=float, default=None,
                   help="motor angular-velocity saturation cap (rad/s). "
                        "Default None → env default (5.0, matching the "
                        "firmware's MOTOR_MIN_STEP_US cap). Lower values "
                        "force the policy below the Kapitza parametric "
                        "stabilisation regime, which requires a·ω above "
                        "a threshold proportional to sqrt(2gL). Capping "
                        "below the rig's natural Kapitza window directly "
                        "disrupts resonance-pumping policies.")
    p.add_argument("--reward-motor-pos-weight", type=float, default=None,
                   help="k_motor_pos: quadratic centering penalty on motor "
                        "angle (env default 0.5). Raise to keep the arm near "
                        "center — kills slow drift/wander. Recorded in config.json.")
    p.add_argument("--net-arch", type=lambda s: [int(x) for x in s.split(",")],
                   default="256,256",
                   help="actor/critic hidden sizes, e.g. '128,128' (SB3 default "
                        "256,256). Smaller nets train faster on this low-dim "
                        "problem; the teacher is distilled to h16 regardless. "
                        "Only used for fresh stage-1 runs; resumes inherit the "
                        "checkpoint's architecture.")
    p.add_argument("--reward-motor-vel-weight", type=float, default=None,
                   help="penalty on motor_vel² in the reward. Default None "
                        "→ env default (0.005). Bumping to e.g. 0.05 makes "
                        "the optimizer prefer policies that keep the motor "
                        "still, not just the pendulum upright. Targets the "
                        "'chattery but balanced' attractor directly.")
    p.add_argument("--gamma", type=float, default=None,
                   help="SAC discount factor. Default None → auto-derived "
                        "as 0.99^(35/control_freq) so the effective time "
                        "horizon (~2.9 s, the canonical 35 Hz setting) "
                        "stays constant across control rates instead of "
                        "silently shrinking at higher Hz.")
    p.add_argument("--mirror-augment", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="store the mirror image (Ms, -a, r, Ms') of every "
                        "transition alongside it — the plant, the reward and "
                        "the reset distribution are all mirror-symmetric, so "
                        "the mirrored transition is genuine, not synthetic. "
                        "Stops SAC from picking an arbitrary preferred "
                        "swing-up direction; doubles buffer_size to keep the "
                        "same experience span. Legal to switch on per "
                        "curriculum stage. See "
                        "website/src/content/docs/reference/symmetry.md.")
    p.add_argument("--use-sde", action="store_true",
                   help="use generalized State-Dependent Exploration "
                        "(gSDE, Raffin 2022) instead of per-step Gaussian "
                        "noise. Gives temporally-smooth exploration — the "
                        "principled alternative to action-rate reward "
                        "penalties (which collapsed entropy in accel "
                        "mode). Fresh runs only; ignored on --resume "
                        "(SAC.load keeps the checkpoint's setting).")
    p.add_argument("--obs-history-len", type=int, default=4,
                   help="number of past 6-dim frames stacked into the "
                        "observation (oldest → newest). Each frame carries "
                        "prev_action, so K>1 gives the policy obs AND action "
                        "history — lets it filter velocity noise, infer the "
                        "per-episode θ-bias, and account for in-flight "
                        "commands. Must match fine-tuning and deployment "
                        "(recorded in config.json). Try 4.")
    p.add_argument("--drop-velocity-obs", action="store_true",
                   help="exclude velocities from the observation frames "
                        "([motor_pos, sin θ, cos θ, prev_action] only) so "
                        "the policy derives its own velocity estimate from "
                        "the stacked position history — removes the "
                        "finite-difference window and its noise spikes from "
                        "the loop entirely. Requires --obs-history-len >= 2 "
                        "(use 4). Must match fine-tuning and deployment "
                        "(recorded in config.json).")
    p.add_argument("--params-path", type=Path, default=None,
                   help="sysid params file describing the rig this policy is "
                        "for. Default None → sysid_params.json next to "
                        "pendulum_env.py. Use a per-rig file (e.g. "
                        "sysid_params_rig1.json) when rigs differ "
                        "mechanically — bearings and grease change the "
                        "measured friction, and the value is outside the DR "
                        "range if the bearing itself is different. Recorded in "
                        "config.json and inherited by distill / dagger / "
                        "analyze_sim.")
    p.add_argument("--firmware-obs-model", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="model the firmware measurement pipeline in sim: "
                        "positions quantised to encoder/step resolution, "
                        "velocities finite-differenced over the firmware's "
                        "8 ms window, snapshot stale by 2–10 ms (DR) — fed "
                        "to the observation, mirroring the real host's "
                        "GET_STATE read. Sim-only realism knob (obs shape "
                        "unchanged); recorded in config.json so "
                        "resumes/evals stay consistent.")
    p.add_argument("--action-smooth-window", type=int, default=4,
                   help="firmware-side action smoothing: the actuator "
                        "receives the moving average of the last N policy "
                        "outputs (1 = off). N=4 nulls the learned PWM "
                        "dither at rate/2 and rate/4 exactly, at a fixed "
                        "1.5-tick delay. Must match ACTION_SMOOTH_WINDOW "
                        "in RLControl.ino; recorded in config.json.")
    p.add_argument("--upright-reset-frac", type=float, default=0.25,
                   help="fraction of TRAINING episodes that reset near "
                        "upright (phi = pi ± 0.3 rad, gentle spin) instead "
                        "of hanging. Trains the catch/balance skill "
                        "directly instead of only after successful "
                        "swing-ups — mitigates the stage-2/3 collapse-to-"
                        "spinning failure mode. The eval env always uses "
                        "hanging-only resets so scores stay comparable. "
                        "Set 0.0 for the legacy behaviour.")
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
    p.add_argument("--reward-alive-offset", type=float, default=15.0,
                   help="constant per-step reward offset (audit F1). Chosen "
                        "≥ the worst realistic per-step quadratic cost "
                        "(~14 on this rig) so per-step reward is non-negative "
                        "and terminating (hard-stop crash) always forfeits "
                        "value — removes the 'suicide by hard stop' local "
                        "optimum that all-negative rewards + termination "
                        "create. Set 0.0 for the legacy canonical reward.")
    p.add_argument("--reward-upright-alive-weight", type=float, default=5.0,
                   help="velocity-gated upright bonus (audit F1): +k per step "
                        "while |θ| ≤ 15° AND |θ̇| ≤ 2 rad/s (same gates as the "
                        "honest balance metrics). Swinging *through* upright "
                        "at speed earns nothing, so spin-through farming is "
                        "unprofitable. Set 0.0 for the legacy reward.")
    p.add_argument("--reward-stillness-bonus-weight", type=float, default=5.0,
                   help="Multiplicative stillness bonus weight. Default 5.0 "
                        "(the canonical operating point); pass 0 to disable "
                        "and recover the plain Quanser reward. When set "
                        ">0, ADDS k · exp(-θ²/σ_θ²) · exp(-α̇²/σ_v²) to the "
                        "reward. The product means a high bonus requires "
                        "BOTH theta and motor_vel near zero simultaneously, "
                        "directly penalising Kapitza-style resonance "
                        "stabilisation (which has α̇ ≈ 3 rad/s during "
                        "balance). Suggested starting value: 5.0.")
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
    p.add_argument("--ignore-config-mismatch", action="store_true",
                   help="downgrade the config.json validation abort (on "
                        "--resume / --eval) to a warning")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if (args.dr_transport_delay_max is not None
            and args.dr_transport_delay_min > args.dr_transport_delay_max):
        raise SystemExit(
            "--dr-transport-delay-min must be <= --dr-transport-delay-max"
        )
    if args.eval:
        evaluate(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
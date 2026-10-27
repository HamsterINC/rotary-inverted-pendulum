"""Quantization-aware SAC trainer for the rotary inverted pendulum.

A thin layer on top of train_sac.py: ALL existing flags (--total-steps,
--domain-randomization, --control-freq, reward weights, --gamma, --lr,
--mirror-augment, --obs-norm, ...) are parsed by train_sac.parse_args and
applied here exactly as train_sac.train() applies them. This script only adds
the quantization flags below.

The observation quantizer is built ONCE from the env's own configuration
(obs_norm preset, obs_include_velocities, obs_history_len) via a throwaway
probe env, then shared by the train and eval envs. The quantization settings
are recorded under the "quant" key of the run's config.json and are checked
against it on --resume / --qat-eval.

Examples
--------
  # float baseline (same as train_sac.py, plus a "quant" section in config.json)
  python train_sac_qat.py --run-name fp

  # 12-bit observations only (policy stays float)
  python train_sac_qat.py --obs-word-bits 12

  # full pipeline: 10-bit obs, 6-bit weights, 8-bit activations, QAT from step 0
  python train_sac_qat.py --obs-word-bits 10 --weight-bits 6 --act-bits 8

  # float warm-up for 60k steps, then switch fake-quant on
  python train_sac_qat.py --weight-bits 4 --act-bits 6 --qat-start-step 60000

  # fine-tune an existing float model under quantization
  python train_sac_qat.py --weight-bits 6 --act-bits 8 --init-from runs/fp/last.zip

  # headless eval of a trained checkpoint (uses the same quant flags)
  python train_sac_qat.py --qat-eval runs/q/last.zip --obs-word-bits 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import symmetry
import train_sac
from qat import (
    FixedPointQuantizer,
    QATConfig,
    QATSACPolicy,
    QATScheduleCallback,
    QuantizedObsWrapper,
    _expand_per_dim,
    default_obs_abs_max,
    env_frame_layout,
    export_actor,
)


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------

def parse_qat_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = p.add_argument_group("quantization")
    g.add_argument("--obs-word-bits", default=None,
                   help="Total bits per observation dim (sign+int+frac). One int, one value "
                        "per frame feature [motor_pos,sin,cos,(motor_vel,pen_vel,)prev_act] "
                        "(tiled over the history), or one per full stacked-obs dim. "
                        "Fractional bits = word - 1 - int_bits.")
    g.add_argument("--obs-frac-bits", default=None,
                   help="Alternative to --obs-word-bits: fixed fractional bits (same shapes).")
    g.add_argument("--obs-abs-max", default=None,
                   help="Override |max| per obs dim (per-frame vector or full stacked vector). "
                        "Default: derived from the env's obs_norm preset, "
                        "obs_include_velocities and obs_history_len.")
    g.add_argument("--weight-bits", type=int, default=None,
                   help="Actor weight bits (symmetric). Unset -> float weights.")
    g.add_argument("--act-bits", type=int, default=None,
                   help="Actor activation bits. Unset -> float activations.")
    g.add_argument("--bias-bits", type=int, default=16)
    g.add_argument("--no-pow2-scales", action="store_true",
                   help="Use arbitrary float scales instead of powers of two.")
    g.add_argument("--per-channel", action="store_true",
                   help="Per-output-channel weight scales (default per-tensor).")
    g.add_argument("--qat-start-step", type=int, default=0,
                   help="Train float until this step, then enable fake-quant.")
    g.add_argument("--init-from", default=None,
                   help="Float SAC .zip whose weights initialise the QAT model.")
    g.add_argument("--qat-eval", default=None,
                   help="Headless eval of this .zip (quantized env), then exit.")
    g.add_argument("--qat-eval-episodes", type=int, default=10)
    return p.parse_known_args(argv)


def parse_all(argv: list[str]):
    qargs, rest = parse_qat_args(argv)
    args = train_sac.parse_args(rest)
    return args, qargs


def _ints(s: str):
    v = [int(t) for t in str(s).split(",")]
    return v[0] if len(v) == 1 else v


def _floats(s: str):
    return [float(t) for t in str(s).split(",")]


def build_quantizer(qargs, env) -> FixedPointQuantizer | None:
    """Build the obs quantizer from the env's own layout (history, velocities, obs_norm)."""
    if qargs.obs_word_bits is None and qargs.obs_frac_bits is None:
        return None
    names, _ = env_frame_layout(env)
    if qargs.obs_abs_max:
        abs_max = np.asarray(_floats(qargs.obs_abs_max), dtype=np.float64)
        full = default_obs_abs_max(env).shape[0]
        if abs_max.shape[0] == len(names):
            abs_max = np.tile(abs_max, full // len(names))
        elif abs_max.shape[0] != full:
            raise ValueError(f"--obs-abs-max needs {len(names)} (per frame) or {full} "
                             f"(full) values, got {abs_max.shape[0]}")
    else:
        abs_max = default_obs_abs_max(env)
    n_total = abs_max.shape[0]
    if qargs.obs_word_bits is not None:
        return FixedPointQuantizer.from_word_bits(
            abs_max, _expand_per_dim(_ints(qargs.obs_word_bits), len(names), n_total))
    return FixedPointQuantizer(
        abs_max, _expand_per_dim(_ints(qargs.obs_frac_bits), len(names), n_total))


def quantizer_from_thunk(qargs, base_thunk):
    """Build the quantizer ONCE from a throwaway probe env.

    Returns (quantizer, (frame_feature_names, history_len)). The quantizer is None
    when no observation quantization was requested. The probe is closed immediately.
    """
    probe = base_thunk()
    try:
        names, hist = env_frame_layout(probe)
        return build_quantizer(qargs, probe), (names, hist)
    finally:
        probe.close()


def build_qcfg(qargs) -> QATConfig | None:
    if qargs.weight_bits is None and qargs.act_bits is None:
        return None
    return QATConfig(
        weight_bits=qargs.weight_bits,
        act_bits=qargs.act_bits,
        bias_bits=qargs.bias_bits,
        pow2_scales=not qargs.no_pow2_scales,
        per_channel=qargs.per_channel,
    )


# --------------------------------------------------------------------------
# Quantization-config consistency check
# --------------------------------------------------------------------------

def _find_run_config(ckpt) -> Path | None:
    """config.json next to the checkpoint, or one level up (runs/<name>/checkpoints/x.zip)."""
    ckpt = Path(ckpt)
    for d in (ckpt.parent, ckpt.parent.parent):
        p = d / "config.json"
        if p.exists():
            return p
    return None


def check_quant_config(ckpt, quantizer, qcfg, *, check_qat: bool = True,
                       ignore: bool = False) -> None:
    """Abort (or warn) if the quantization settings differ from the ones the
    checkpoint was trained with. Obs formats are always compared; the QAT
    weight/act settings only when check_qat is True (resume, not headless eval,
    where those flags needn't be repeated)."""
    path = _find_run_config(ckpt)
    saved = json.loads(path.read_text()).get("quant") if path else None
    if saved is None:
        print(f"[quant check] no quant section found for {ckpt}; "
              f"cannot verify quantization settings")
        return

    now = {"obs_formats": quantizer.describe() if quantizer else None}
    was = {"obs_formats": saved.get("obs_formats")}
    if check_qat:
        now["qat"] = asdict(qcfg) if qcfg else None
        was["qat"] = saved.get("qat")

    diffs = [f"  {k}: checkpoint={was[k]}  now={now[k]}" for k in now if now[k] != was[k]]
    if not diffs:
        return
    msg = "Quantization settings differ from the checkpoint's run:\n" + "\n".join(diffs)
    if check_qat and was.get("qat") is None and now.get("qat") is not None:
        msg += ("\nNote: a resumed checkpoint keeps its own policy class, so QAT flags "
                "would be ignored on a float checkpoint. Use --init-from to start a "
                "QAT run from float weights.")
    if ignore:
        print("[quant check] WARNING (ignored): " + msg)
    else:
        raise SystemExit(msg + "\nPass --ignore-config-mismatch to downgrade this to a warning.")


# --------------------------------------------------------------------------
# Environments (kwargs mirror train_sac.train() one-to-one)
# --------------------------------------------------------------------------

def _dr_ranges(args) -> dict:
    """The optional DR ranges, as train_sac.train() derives them from args."""
    return dict(
        accel=((args.dr_accel_min, args.dr_accel_max)
               if args.dr_accel_max is not None else None),
        lag=((args.dr_action_lag_tau_min, args.dr_action_lag_tau_max)
             if args.dr_action_lag_tau_max is not None else None),
        stale=((args.dr_obs_staleness_min, args.dr_obs_staleness_max)
               if args.dr_obs_staleness_max is not None else None),
    )


def _shared_env_kwargs(args) -> dict:
    """Settings identical for the train and eval env."""
    return dict(
        params_path=args.params_path,
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        max_action_delta_rad=args.max_action_delta_rad,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_pos_weight=args.reward_motor_pos_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        reward_alive_offset=args.reward_alive_offset,
        reward_upright_alive_weight=args.reward_upright_alive_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
        obs_history_len=args.obs_history_len,
        obs_include_velocities=not args.drop_velocity_obs,
        firmware_obs_model=args.firmware_obs_model,
        action_smooth_window=args.action_smooth_window,
        obs_norm=args.obs_norm,
    )


def _train_env_kwargs(args) -> dict:
    dr = _dr_ranges(args)
    return dict(
        **_shared_env_kwargs(args),
        domain_randomization=args.domain_randomization,
        dr_motor_accel_range_rad_s2=dr["accel"],
        dr_action_lag_tau_range_s=dr["lag"],
        dr_obs_staleness_range_s=dr["stale"],
        dr_control_dt_jitter_frac=args.dr_dt_jitter_frac,
        dr_theta_bias_max_rad=args.dr_theta_bias_max_rad,
        upright_reset_frac=args.upright_reset_frac,
    )


def _eval_env_kwargs(args) -> dict:
    """Deterministic reference scenario: no DR, no theta-bias, hanging-only resets.
    Transport lag / staleness are pinned to the midpoint of the training DR ranges
    (same rule as train_sac.train()), so best_model is not selected by a test the
    training episodes never saw."""
    dr = _dr_ranges(args)
    eval_lag_tau_s = (
        (args.dr_action_lag_tau_min + args.dr_action_lag_tau_max) / 2.0
        if args.dr_action_lag_tau_max is not None else 0.0
    )
    eval_stale_s = (sum(dr["stale"]) / 2.0) if dr["stale"] else None
    return dict(
        **_shared_env_kwargs(args),
        domain_randomization=False,
        action_lag_tau_s=eval_lag_tau_s,
        obs_staleness_s=eval_stale_s,
        dr_theta_bias_max_rad=0.0,
    )


def make_qenv(base_thunk, quantizer):
    """Wrap train_sac.make_env's thunk with the (shared) observation quantizer.
    The wrapper sits outside Monitor, which is fine: Monitor only touches
    rewards/infos."""
    def _thunk():
        env = base_thunk()
        return QuantizedObsWrapper(env, quantizer) if quantizer is not None else env
    return _thunk


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_qat(args, qargs) -> Path:
    qcfg = build_qcfg(qargs)
    if qcfg is not None and args.use_sde:
        raise SystemExit("--use-sde is not supported with quantized actors (QuantActor has no gSDE).")

    # Guard resume against silently switching the must-match knobs
    # (checked before creating the run dir so an abort leaves no debris).
    config = train_sac._resolved_config(args)
    if args.resume:
        train_sac.check_config(args.resume, config, ignore=args.ignore_config_mismatch)

    # Probe env (writes no files) -> shared quantizer. Built before the run dir
    # so a quant mismatch aborts cleanly.
    eval_thunk = train_sac.make_env(**_eval_env_kwargs(args))
    quantizer, (obs_names, obs_hist) = quantizer_from_thunk(qargs, eval_thunk)
    if args.resume:
        check_quant_config(args.resume, quantizer, qcfg,
                           ignore=args.ignore_config_mismatch)

    run_name = args.run_name or f"qsac_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = train_sac.RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    train_sac.save_run_config(run_dir, config)

    if qcfg is not None:
        print(f"QAT config: {qcfg}  start_step={qargs.qat_start_step}")
    print(f"Obs layout: {obs_hist} frames x {obs_names}")
    if quantizer is not None:
        print(f"Obs formats: {quantizer.describe()}  (word bits {quantizer.word_bits.tolist()})")

    train_env = DummyVecEnv([make_qenv(
        train_sac.make_env(run_dir, **_train_env_kwargs(args)), quantizer)])
    eval_env = DummyVecEnv([make_qenv(eval_thunk, quantizer)])
    # Seed the eval env too: SB3 only seeds the training env, so without this the
    # eval episodes (and therefore best_model.zip selection) differ run to run.
    eval_env.seed(args.seed + 10_000)

    # gamma: hold the canonical 35 Hz horizon constant unless --gamma is given.
    gamma = (
        float(args.gamma) if args.gamma is not None
        else 0.99 ** (35.0 / args.control_freq)
    )
    print(f"gamma = {gamma:.4f} "
          f"({'explicit' if args.gamma is not None else f'auto-derived for {args.control_freq:g} Hz'})")

    # Mirror data augmentation (see train_sac.py / symmetry.py).
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
        # SAC.load rebuilds an EMPTY buffer, so swapping the class loses nothing.
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
        # Same net_arch rule as train_sac: --net-arch if given, else SB3's default.
        policy_kwargs = dict(net_arch=args.net_arch) if args.net_arch else {}
        if qcfg is not None:
            policy_kwargs["qcfg"] = qcfg
        model = SAC(
            QATSACPolicy if qcfg is not None else "MlpPolicy",
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
            policy_kwargs=policy_kwargs or None,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
        )
        if qargs.init_from:
            src = SAC.load(qargs.init_from, device=args.device)
            res = model.policy.load_state_dict(src.policy.state_dict(), strict=False)
            print(f"Loaded float weights from {qargs.init_from}; "
                  f"missing (new quant buffers, expected): {len(res.missing_keys)}, "
                  f"unexpected: {res.unexpected_keys}")
            if hasattr(src, "log_ent_coef") and hasattr(model, "log_ent_coef"):
                with th.no_grad():
                    model.log_ent_coef.copy_(src.log_ent_coef)

    # One config file: append the quant section to the config.json that
    # train_sac.save_run_config already wrote.
    cfg_path = run_dir / "config.json"
    doc = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    doc["quant"] = dict(
        obs_frame_features=obs_names,
        obs_history_len=obs_hist,
        obs_formats=quantizer.describe() if quantizer else None,
        obs_word_bits=quantizer.word_bits.tolist() if quantizer else None,
        qat=asdict(qcfg) if qcfg else None,
        qat_start_step=qargs.qat_start_step,
    )
    cfg_path.write_text(json.dumps(doc, indent=2, default=str))

    cbs = []
    if qcfg is not None:
        cbs.append(QATScheduleCallback(qargs.qat_start_step))  # must come first
    cbs += [
        EvalCallback(eval_env, best_model_save_path=str(run_dir),
                     log_path=str(run_dir / "eval"), eval_freq=args.eval_freq,
                     n_eval_episodes=5, deterministic=True, render=False),
        CheckpointCallback(save_freq=args.checkpoint_freq,
                           save_path=str(run_dir / "checkpoints"), name_prefix="sac",
                           save_replay_buffer=False, save_vecnormalize=False),
    ]

    model.learn(total_timesteps=args.total_steps, callback=CallbackList(cbs),
                log_interval=args.log_interval, progress_bar=args.progress_bar,
                reset_num_timesteps=not args.resume)

    final_path = run_dir / "last.zip"
    model.save(final_path)
    print(f"Final model saved to {final_path}")
    if qcfg is not None and qcfg.weight_bits is not None:
        export_actor(model.policy.actor, str(run_dir / "actor_int.npz"), quantizer)
        print(f"Integer actor exported to {run_dir / 'actor_int.npz'}")
    return run_dir


# --------------------------------------------------------------------------
# Headless evaluation (quantized env + quantized policy)
# --------------------------------------------------------------------------

def evaluate_headless(model_path, args, qargs, n_episodes: int = 10, seed: int = 1000) -> dict:
    train_sac.check_config(model_path, train_sac._resolved_config(args),
                           ignore=args.ignore_config_mismatch)
    thunk = train_sac.make_env(**_eval_env_kwargs(args))
    quantizer, _ = quantizer_from_thunk(qargs, thunk)
    # Only obs formats are compared here; weight/act settings live in the checkpoint.
    check_quant_config(model_path, quantizer, None, check_qat=False,
                       ignore=args.ignore_config_mismatch)
    env = make_qenv(thunk, quantizer)()
    model = SAC.load(model_path, device=args.device)
    returns, upright = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, n, up = False, 0.0, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            ret += r
            n += 1
            up += abs(env.unwrapped._theta_upright()) < 0.2   # within ~11 deg of upright
            done = term or trunc
        returns.append(ret)
        upright.append(up / n)
    env.close()
    return dict(mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                upright_frac=float(np.mean(upright)))


def main(argv: list[str] | None = None) -> int:
    args, qargs = parse_all(argv if argv is not None else sys.argv[1:])
    if args.eval:
        raise SystemExit("--eval renders an unquantized env; use --qat-eval for quantized checkpoints.")
    if qargs.qat_eval:
        print(evaluate_headless(qargs.qat_eval, args, qargs, qargs.qat_eval_episodes))
    else:
        train_qat(args, qargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Quantization-aware SAC trainer for the rotary inverted pendulum.

A thin layer on top of train_sac.py: ALL existing flags (--total-steps,
--domain-randomization, --control-freq, reward weights, ...) still work and
are parsed by train_sac.parse_args. This script only adds the quantization
flags below, so train_sac.py needs no edits.

Examples
--------
  # float baseline (same as train_sac.py, plus a quant_config.json)
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
import inspect
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import train_sac
from qat import (
    FixedPointQuantizer,
    QATConfig,
    QATSACPolicy,
    QATScheduleCallback,
    QuantizedObsWrapper,
    default_obs_abs_max,
    export_actor,
)


# --------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------

def parse_qat_args(argv: list[str]):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    g = p.add_argument_group("quantization")
    g.add_argument("--obs-word-bits", default=None,
                   help="Total bits per observation dim (sign+int+frac). One int or "
                        "6 comma-separated ints. Fractional bits = word - 1 - int_bits, "
                        "where int_bits comes from --obs-abs-max.")
    g.add_argument("--obs-frac-bits", default=None,
                   help="Alternative to --obs-word-bits: fixed fractional bits per dim.")
    g.add_argument("--obs-abs-max", default=None,
                   help="6 comma-separated |max| per obs dim "
                        "[motor_pos,cos,sin,motor_vel,pen_vel,prev_act]. Default: "
                        "preset chosen from pendulum_env.NORMILIZED_OBS "
                        "(all 1.0 if normalized).")
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


def build_quantizer(qargs) -> FixedPointQuantizer | None:
    if qargs.obs_word_bits is None and qargs.obs_frac_bits is None:
        return None
    abs_max = _floats(qargs.obs_abs_max) if qargs.obs_abs_max else default_obs_abs_max()
    if qargs.obs_word_bits is not None:
        return FixedPointQuantizer.from_word_bits(abs_max, _ints(qargs.obs_word_bits))
    return FixedPointQuantizer(abs_max, _ints(qargs.obs_frac_bits))


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
# Environments
# --------------------------------------------------------------------------

# make_env parameters that callers below set explicitly (train vs eval differ), so
# they must never be auto-forwarded from args.
_EXPLICIT_ENV_PARAMS = {
    "monitor_dir", "domain_randomization", "dr_motor_accel_range_rad_s2",
    "dr_action_delay_steps_range", "dr_action_lag_tau_range_s",
    "dr_control_dt_jitter_frac", "dr_theta_bias_max_rad",
}


def _forwarded_extras(args) -> dict:
    """Any other make_env parameter that has a same-named parsed argument is forwarded
    automatically, so options you add to train_sac.py (e.g. --obs-norm -> obs_norm) reach
    the env without editing this file. Requires: make_env(..., obs_norm=...) and the
    argparse dest obs_norm."""
    known = {"control_freq_hz", "max_accel_rad_s2", "reward_action_rate_weight",
             "reward_motor_vel_weight", "reward_motor_jerk_weight",
             "reward_stillness_bonus_weight", "max_velocity_rad_s"}
    extras = {}
    for name in inspect.signature(train_sac.make_env).parameters:
        if name in known or name in _EXPLICIT_ENV_PARAMS:
            continue
        if hasattr(args, name):
            extras[name] = getattr(args, name)
    return extras


def _common_env_kwargs(args) -> dict:
    kw = dict(
        control_freq_hz=args.control_freq,
        max_accel_rad_s2=args.max_accel_rad_s2,
        reward_action_rate_weight=args.reward_action_rate_weight,
        reward_motor_vel_weight=args.reward_motor_vel_weight,
        reward_motor_jerk_weight=args.reward_motor_jerk_weight,
        reward_stillness_bonus_weight=args.reward_stillness_bonus_weight,
        max_velocity_rad_s=args.max_velocity_rad_s,
    )
    kw.update(_forwarded_extras(args))
    return kw


def make_qenv(base_thunk, quantizer):
    """Wrap train_sac.make_env's thunk with the observation quantizer.
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
    run_name = args.run_name or f"qsac_{time.strftime('%Y-%m-%d_%H%M')}"
    run_dir = train_sac.RUNS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    quantizer = build_quantizer(qargs)
    qcfg = build_qcfg(qargs)
    extras = _forwarded_extras(args)
    if extras:
        print(f"Extra env kwargs forwarded to make_env: {extras}")
    if quantizer is not None:
        print(f"Obs formats: {quantizer.describe()}  (word bits {quantizer.word_bits.tolist()})")
    if qcfg is not None:
        print(f"QAT config: {qcfg}  start_step={qargs.qat_start_step}")

    dr_accel = (args.dr_accel_min, args.dr_accel_max) if args.dr_accel_max is not None else None
    dr_delay = (args.dr_delay_min, args.dr_delay_max) if args.dr_delay_max is not None else None
    dr_lag = ((args.dr_action_lag_tau_min, args.dr_action_lag_tau_max)
              if args.dr_action_lag_tau_max is not None else None)

    train_env = DummyVecEnv([make_qenv(train_sac.make_env(
        run_dir,
        domain_randomization=args.domain_randomization,
        dr_motor_accel_range_rad_s2=dr_accel,
        dr_action_delay_steps_range=dr_delay,
        dr_action_lag_tau_range_s=dr_lag,
        dr_control_dt_jitter_frac=args.dr_dt_jitter_frac,
        dr_theta_bias_max_rad=args.dr_theta_bias_max_rad,
        **_common_env_kwargs(args),
    ), quantizer)])
    eval_env = DummyVecEnv([make_qenv(train_sac.make_env(
        domain_randomization=False,
        dr_theta_bias_max_rad=0.0,
        **_common_env_kwargs(args),
    ), quantizer)])
    # Seed the eval env too: SB3 only seeds the training env, so without this the
    # eval episodes (and therefore best_model.zip selection) differ run to run.
    eval_env.seed(args.seed + 10_000)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(args.resume, env=train_env, device=args.device)
    else:
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128]))
        if qcfg is not None:
            policy_kwargs["qcfg"] = qcfg
        model = SAC(
            QATSACPolicy if qcfg is not None else "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=400_000,
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
        if qargs.init_from:
            src = SAC.load(qargs.init_from, device=args.device)
            res = model.policy.load_state_dict(src.policy.state_dict(), strict=False)
            print(f"Loaded float weights from {qargs.init_from}; "
                  f"missing (new quant buffers, expected): {len(res.missing_keys)}, "
                  f"unexpected: {res.unexpected_keys}")
            if hasattr(src, "log_ent_coef") and hasattr(model, "log_ent_coef"):
                with th.no_grad():
                    model.log_ent_coef.copy_(src.log_ent_coef)

    with open(run_dir / "quant_config.json", "w") as f:
        json.dump(dict(
            obs_formats=quantizer.describe() if quantizer else None,
            obs_word_bits=quantizer.word_bits.tolist() if quantizer else None,
            qat=asdict(qcfg) if qcfg else None,
            qat_start_step=qargs.qat_start_step,
            args=vars(args),
        ), f, indent=2, default=str)

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
    quantizer = build_quantizer(qargs)
    env = make_qenv(train_sac.make_env(
        domain_randomization=False, dr_theta_bias_max_rad=0.0,
        **_common_env_kwargs(args)), quantizer)()
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
    if qargs.qat_eval:
        print(evaluate_headless(qargs.qat_eval, args, qargs, qargs.qat_eval_episodes))
    else:
        train_qat(args, qargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
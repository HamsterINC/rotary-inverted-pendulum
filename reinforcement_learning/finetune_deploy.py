"""Fine-tune a SAC actor on the rig while the PL runs the policy.

Loop, per episode:
    1. export the current actor to the PL weight format and upload it
       (double-buffered: becomes active between two inferences)
    2. run one episode on the PL (FpgaDeployEnv.collect_episode) — the PS only
       observes; the step log gives exact (obs, action) pairs
    3. push the transitions into the SAC replay buffer
    4. run SAC gradient steps on the PS CPU (the motor is off meanwhile)

Exploration: the PL policy is deterministic, so exploration comes from
parameter noise (`--param-noise`, Gaussian noise on the uploaded weights only;
the learner's actor stays clean). The transitions are off-policy data for SAC
either way.

Usage on the Zynq (from reinforcement_learning/):
    python -m Deploy.FPGA.finetune_deploy \\
        --checkpoint runs/<run>/last.zip --config runs/<run>/config.json \\
        --episodes 50 --param-noise 0.01 --save-dir runs/<run>/rig

The checkpoint must match the PL network: obs_history_len=1, velocities on,
obs_norm="raw", net-arch [128, 128] (FpgaDeployEnv.from_config checks this).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from Deploy.FPGA.FPGA_deploy_env import FpgaDeployEnv
from Deploy.FPGA.weight_loader import actor_to_pl_words


def add_episode_to_buffer(model, ep: dict, *, drop_invalid: bool = True) -> int:
    """Append one collected episode to an SB3 off-policy replay buffer.
    done = terminated only (time-limit truncation still bootstraps)."""
    keep = ep["valid"] if drop_invalid else np.ones(ep["length"], dtype=bool)
    buf = model.replay_buffer
    for o, a, r, o2, t in zip(ep["obs"][keep], ep["actions"][keep], ep["rewards"][keep],
                              ep["next_obs"][keep], ep["terminated"][keep]):
        buf.add(o[None], o2[None], a[None], np.array([r]), np.array([t]), [{}])
    return int(keep.sum())


def finetune_on_rig(
    model,
    env: FpgaDeployEnv,
    *,
    n_episodes: int,
    gradient_steps: int | None = None,     # None -> one per collected transition
    batch_size: int = 256,
    learning_starts: int = 0,
    param_noise_std: float = 0.0,
    save_dir: str | Path | None = None,
    save_every: int = 10,
    seed: int | None = None,
    log=print,
) -> list[dict]:
    """Episode loop: upload -> run on PL -> store -> train. Returns per-episode stats."""
    if getattr(model, "_logger", None) is None:      # SAC.train() logs; learn() normally sets this
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, ["stdout"]))
    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for ep_i in range(n_episodes):
        # 1. weights -> PL
        model.policy.set_training_mode(False)
        words = actor_to_pl_words(model.actor, param_noise_std=param_noise_std, rng=rng,
                                  warn=(ep_i == 0))
        t0 = time.monotonic()
        st = env.client.load_weights(words)
        t_upload = time.monotonic() - t0

        # 2. episode on the rig (PS only observes)
        ep = env.collect_episode()

        # 3. replay buffer
        n_added = add_episode_to_buffer(model, ep)
        model.num_timesteps += ep["length"]

        # 4. learn while the motor is off
        t0 = time.monotonic()
        n_grad = 0
        if model.replay_buffer.size() >= max(batch_size, learning_starts):
            n_grad = gradient_steps if gradient_steps is not None else n_added
            if n_grad > 0:
                model.train(gradient_steps=n_grad, batch_size=batch_size)
        t_train = time.monotonic() - t0

        stats = {
            "episode": ep_i,
            "return": ep["return"],
            "length": ep["length"],
            "added": n_added,
            "gradient_steps": n_grad,
            "mean_dt_ms": 1e3 * ep["mean_dt_s"],
            "upload_s": t_upload,
            "train_s": t_train,
            "weight_bank": st.active_bank,
            "buffer": model.replay_buffer.size(),
        }
        history.append(stats)
        log(f"[rig] ep {ep_i:4d}  return {ep['return']:8.2f}  len {ep['length']:4d}  "
            f"added {n_added:4d}  grad {n_grad:4d}  dt {stats['mean_dt_ms']:.1f} ms  "
            f"upload {t_upload * 1e3:.0f} ms  train {t_train:.1f} s")

        if save_dir is not None and ((ep_i + 1) % save_every == 0 or ep_i + 1 == n_episodes):
            model.save(save_dir / f"rig_ep{ep_i + 1:04d}.zip")
            with open(save_dir / "rig_history.json", "w") as f:
                json.dump(history, f, indent=1)
    return history


def main(argv: list[str] | None = None) -> int:
    from Deploy.FPGA.zynq_load import load_sac_for_finetune

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="SB3 SAC .zip")
    p.add_argument("--config", required=True, help="the run's config.json")
    p.add_argument("--buffer-dir", default=None, help="portable replay buffer to preload")
    p.add_argument("--bitstream", default=None, help="deployment .xsa/.bit (default: FpgaDeployClient's)")
    p.add_argument("--no-download", action="store_true", help="overlay already loaded")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episode-length", type=float, default=None, help="seconds (default: config)")
    p.add_argument("--gradient-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--param-noise", type=float, default=0.0, help="std of weight noise on the PL copy")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = json.load(f)
    overrides = {"download": not args.no_download}
    if args.bitstream:
        overrides["bitstream_path"] = args.bitstream
    if args.episode_length is not None:
        overrides["episode_length_s"] = args.episode_length
    env = FpgaDeployEnv.from_config(cfg, **overrides)

    signal.signal(signal.SIGINT, env._on_signal)
    signal.signal(signal.SIGTERM, env._on_signal)
    try:
        model = load_sac_for_finetune(args.checkpoint, env, cfg, buffer_dir=args.buffer_dir)
        finetune_on_rig(
            model, env,
            n_episodes=args.episodes,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            param_noise_std=args.param_noise,
            save_dir=args.save_dir,
            save_every=args.save_every,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("interrupted, motor disengaged")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fine-tune a SAC actor on the rig while the PL runs the policy — asynchronously.

Two threads, like finetune_async.py, so the rig never waits for the learner:

  Rig thread (all FPGA access happens here):
      per episode: PD-home the motor -> wait until the pendulum is STILL
      (starts the moment it is, no fixed wait) -> tare -> ask the learner for
      the newest weights and upload them (double-buffered on the PL) -> run
      the episode on the PL (the PS only observes; the step log gives exact
      (obs, action) pairs) -> hand the transitions to the learner -> repeat.

  Main thread (learner):
      trains SAC continuously in small chunks (--gradient-steps per train()
      call) while episodes run, homing happens and the pendulum settles.
      Between chunks it adds finished episodes to the replay buffer and, when
      the rig asks, exports the current actor to the PL weight format.

Each episode runs ONE fixed set of weights (the newest at the moment the
policy starts), so parameter noise stays constant within an episode. The data
is off-policy for SAC either way.

Exploration: the PL policy is deterministic, so exploration comes from
parameter noise (`--param-noise`, Gaussian noise on the uploaded weights only;
the learner's actor stays clean).

Rest detection (FpgaDeployEnv): |pen_vel| < --rest-vel-threshold AND the
pendulum angle within a --rest-angle-range band, both for 1 s. Gives up after
--reset-settle-s (then runs without a tare).

Usage on the Zynq (from reinforcement_learning/):
    python finetune_deploy.py \\
        --checkpoint runs/<run>/last.zip --config runs/<run>/config.json \\
        --episodes 50 --param-noise 0.01 --save-dir runs/<run>/rig

The checkpoint must match the PL network: obs_history_len=1, velocities on,
obs_norm="raw", net-arch [128, 128] (FpgaDeployEnv.from_config checks this).
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from Deploy.FPGA.FPGA_deploy_env import FpgaDeployEnv
from Deploy.FPGA.weight_loader import actor_to_pl_words

WEIGHT_REQUEST_TIMEOUT_S = 5.0   # rig waits this long for fresh weights, else keeps the old ones


def add_episode_to_buffer(model, ep: dict, *, drop_invalid: bool = True) -> int:
    """Append one collected episode to an SB3 off-policy replay buffer.
    done = terminated only (time-limit truncation still bootstraps)."""
    keep = ep["valid"] if drop_invalid else np.ones(ep["length"], dtype=bool)
    buf = model.replay_buffer
    for o, a, r, o2, t in zip(ep["obs"][keep], ep["actions"][keep], ep["rewards"][keep],
                              ep["next_obs"][keep], ep["terminated"][keep]):
        buf.add(o[None], o2[None], a[None], np.array([r]), np.array([t]), [{}])
    return int(keep.sum())


class WeightHandoff:
    """Rig thread asks for weights; the learner answers between train() calls,
    so the actor is never read while the optimizer is writing it."""

    def __init__(self):
        self._requested = threading.Event()
        self._ready = threading.Event()
        self._words = None
        self._meta: dict = {}

    def request(self, timeout: float):
        """Rig thread: block until the learner provides weights (or timeout -> None)."""
        self._ready.clear()
        self._requested.set()
        if not self._ready.wait(timeout):
            self._requested.clear()
            return None, {}
        return self._words, self._meta

    def pending(self) -> bool:
        return self._requested.is_set()

    def provide(self, words, meta: dict) -> None:
        """Learner thread."""
        self._words, self._meta = words, meta
        self._requested.clear()
        self._ready.set()


def _rig_worker(env: FpgaDeployEnv, n_episodes: int, handoff: WeightHandoff,
                out: queue.Queue, stop: threading.Event, log=print) -> None:
    """Run episodes back to back on the PL; post ("episode", i, ep, upload) per
    episode, then ("done",) or ("error", exc)."""
    try:
        for ep_i in range(n_episodes):
            if stop.is_set():
                break
            upload: dict = {}

            def before_start():
                # Runs after homing + rest + tare: fetch and upload the newest weights.
                words, meta = handoff.request(WEIGHT_REQUEST_TIMEOUT_S)
                if words is None:
                    log("  [warn] learner did not provide weights in time; "
                        "reusing the previous ones")
                    upload.update(uploaded=False)
                    return
                t0 = time.monotonic()
                st = env.client.load_weights(words)
                upload.update(uploaded=True, upload_s=time.monotonic() - t0,
                              weight_bank=st.active_bank, **meta)

            ep = env.collect_episode(before_start=before_start)
            out.put(("episode", ep_i, ep, upload))
    except KeyboardInterrupt:
        pass                                    # stop requested (signal handler)
    except Exception as err:                    # surface rig errors in the main thread
        out.put(("error", err))
    finally:
        env.disengage_safely()
        out.put(("done",))


def finetune_on_rig(
    model,
    env: FpgaDeployEnv,
    *,
    n_episodes: int,
    gradient_steps: int = 4,               # per train() call
    batch_size: int = 256,
    learning_starts: int = 0,
    max_grad_per_transition: float = 0.0,  # 0 = no cap on the update-to-data ratio
    param_noise_std: float = 0.0,
    save_dir: str | Path | None = None,
    save_every: int = 10,
    seed: int | None = None,
    stop: threading.Event | None = None,
    log=print,
) -> list[dict]:
    """Rig thread collects episodes while this thread trains. Returns per-episode stats."""
    if getattr(model, "_logger", None) is None:      # SAC.train() logs; learn() normally sets this
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, ["stdout"]))
    rng = np.random.default_rng(seed)
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    stop = stop or threading.Event()

    _ = env.client                           # open the overlay here, not in the rig thread
    handoff = WeightHandoff()
    episodes: queue.Queue = queue.Queue()
    rig = threading.Thread(target=_rig_worker, name="rig",
                           args=(env, n_episodes, handoff, episodes, stop, log), daemon=True)

    history: list[dict] = []
    n_grad_total = 0
    n_added_total = 0
    n_exports = 0
    grad_since_ep, train_s_since_ep = 0, 0.0
    t_start = time.monotonic()
    rig_error = None

    def save(tag: str) -> None:
        if save_dir is None:
            return
        model.save(save_dir / f"{tag}.zip")
        with open(save_dir / "rig_history.json", "w") as f:
            json.dump(history, f, indent=1)

    rig.start()
    try:
        while True:
            # 1. Rig wants weights for the next episode.
            if handoff.pending():
                words = actor_to_pl_words(model.actor, param_noise_std=param_noise_std,
                                          rng=rng, warn=(n_exports == 0))
                n_exports += 1
                handoff.provide(words, {"weights_grad_step": n_grad_total})

            # 2. Finished episodes -> replay buffer.
            done = False
            while True:
                try:
                    msg = episodes.get_nowait()
                except queue.Empty:
                    break
                if msg[0] == "done":
                    done = True
                    continue
                if msg[0] == "error":
                    rig_error = msg[1]
                    continue
                _, ep_i, ep, upload = msg
                n_added = add_episode_to_buffer(model, ep)
                n_added_total += n_added
                model.num_timesteps += ep["length"]
                stats = {
                    "episode": ep_i,
                    "return": ep["return"],
                    "length": ep["length"],
                    "added": n_added,
                    "gradient_steps": grad_since_ep,     # since the previous episode arrived
                    "train_s": train_s_since_ep,
                    "grad_total": n_grad_total,
                    "mean_dt_ms": 1e3 * ep["mean_dt_s"],
                    "rest_wait_s": ep["rest_wait_s"],
                    "tared": ep["tared"],
                    "homed": ep["homed"],
                    "uploaded": upload.get("uploaded", False),
                    "upload_s": upload.get("upload_s", float("nan")),
                    "weights_grad_step": upload.get("weights_grad_step", -1),
                    "weight_bank": upload.get("weight_bank", -1),
                    "buffer": model.replay_buffer.size(),
                    "wall_min": (time.monotonic() - t_start) / 60.0,
                }
                history.append(stats)
                grad_since_ep, train_s_since_ep = 0, 0.0
                log(f"[rig] ep {ep_i:4d}  return {ep['return']:8.2f}  len {ep['length']:4d}  "
                    f"added {n_added:4d}  grad {stats['gradient_steps']:5d} "
                    f"(total {n_grad_total})  weights@{stats['weights_grad_step']}  "
                    f"rest {stats['rest_wait_s']:.1f} s{'' if ep['tared'] else ' (NO TARE)'}  "
                    f"dt {stats['mean_dt_ms']:.1f} ms  train {stats['train_s']:.1f} s")
                if (ep_i + 1) % save_every == 0:
                    save(f"rig_ep{ep_i + 1:04d}")
            if done or stop.is_set():
                break

            # 3. Train a small chunk (keeps the weight hand-off responsive).
            can_train = model.replay_buffer.size() >= max(batch_size, learning_starts)
            if can_train and max_grad_per_transition > 0:
                can_train = n_grad_total < max_grad_per_transition * n_added_total
            if can_train:
                t0 = time.monotonic()
                model.train(gradient_steps=gradient_steps, batch_size=batch_size)
                dt = time.monotonic() - t0
                n_grad_total += gradient_steps
                grad_since_ep += gradient_steps
                train_s_since_ep += dt
            else:
                time.sleep(0.005)
    finally:
        stop.set()
        env._on_signal()                    # disengage + make the rig thread exit
        rig.join(timeout=10.0)
        if rig.is_alive():
            log("  [warn] rig thread did not exit within 10 s")
        if history:
            save("rig_last")
        log(f"[rig] {len(history)} episodes, {n_added_total} transitions, "
            f"{n_grad_total} gradient steps, {(time.monotonic() - t_start) / 60:.1f} min")
    if rig_error is not None:
        raise rig_error
    return history


def main(argv: list[str] | None = None) -> int:
    import torch
    from Deploy.FPGA.zynq_load import load_sac_for_finetune

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="SB3 SAC .zip")
    p.add_argument("--config", required=True, help="the run's config.json")
    p.add_argument("--buffer-dir", default=None, help="portable replay buffer to preload")
    p.add_argument("--bitstream", default=None, help="deployment .xsa/.bit (default: FpgaDeployClient's)")
    p.add_argument("--no-download", action="store_true", help="overlay already loaded")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episode-length", type=float, default=None, help="seconds (default: config)")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="SAC gradient steps per train() call (training runs continuously)")
    p.add_argument("--max-grad-per-transition", type=float, default=0.0,
                   help="cap on total gradient steps / real transitions (update-to-data "
                        "ratio), so long rest waits don't over-train on old data. 0 = no cap")
    p.add_argument("--learning-starts", type=int, default=0,
                   help="replay buffer size before training starts (at least --batch-size)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--param-noise", type=float, default=0.0, help="std of weight noise on the PL copy")
    p.add_argument("--reset-settle-s", type=float, default=60.0,
                   help="max seconds to wait for a still pendulum between episodes; the "
                        "episode starts as soon as it is still. On timeout: no tare, run anyway")
    p.add_argument("--rest-vel-threshold", type=float, default=None,
                   help="|pen_vel| below this counts as still [rad/s] (default 0.5)")
    p.add_argument("--rest-angle-range", type=float, default=None,
                   help="pendulum angle must stay within this band for 1 s [rad] (default 0.03)")
    p.add_argument("--torch-threads", type=int, default=1,
                   help="torch CPU threads for training; leave a core for the rig thread")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.set_num_threads(args.torch_threads)
    sys.setswitchinterval(0.001)            # rig thread gets the GIL back quickly between torch ops

    with open(args.config) as f:
        cfg = json.load(f)
    overrides = {"download": not args.no_download, "reset_settle_s": args.reset_settle_s}
    if args.bitstream:
        overrides["bitstream_path"] = args.bitstream
    if args.episode_length is not None:
        overrides["episode_length_s"] = args.episode_length
    if args.rest_vel_threshold is not None:
        overrides["rest_vel_threshold_rad_s"] = args.rest_vel_threshold
    if args.rest_angle_range is not None:
        overrides["rest_angle_range_rad"] = args.rest_angle_range
    env = FpgaDeployEnv.from_config(cfg, **overrides)

    stop = threading.Event()

    def on_signal(*_):                      # signals are delivered to the main thread only
        if not stop.is_set():
            print("\nstop requested: disengaging, saving")
        stop.set()
        env._on_signal()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        model = load_sac_for_finetune(args.checkpoint, env, cfg, buffer_dir=args.buffer_dir)
        finetune_on_rig(
            model, env,
            n_episodes=args.episodes,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            max_grad_per_transition=args.max_grad_per_transition,
            param_noise_std=args.param_noise,
            save_dir=args.save_dir,
            save_every=args.save_every,
            seed=args.seed,
            stop=stop,
        )
    except KeyboardInterrupt:
        print("interrupted, motor disengaged")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 4 (proper): asynchronous SAC fine-tuning against the real device.

Replaces `finetune_real.py`'s use of `model.learn()`, which silently dropped
the control rate when gradient updates ran longer than `1/control_freq_hz`.
This orchestrator owns two threads:

  Main thread (this function):
    - Loads the SAC checkpoint and (optionally) saved replay buffers.
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

Two replay buffers (sim + real, mixed per batch):
    Sim transitions and real transitions are kept in SEPARATE buffers. The
    sim buffer is loaded from the source run and frozen (nothing is added to
    it). Every real transition goes into the real buffer. Each SAC minibatch
    is assembled from both at a controlled ratio (RLPD-style "symmetric
    sampling"), so real data makes up a fixed share of every gradient step
    from the very first real episode, instead of being a tiny fraction of
    one large FIFO buffer.

    The real share ramps linearly with the number of real transitions:
        real_frac = start + (end - start) * min(1, n_real / ramp_steps)
    set with --real-frac-start / --real-frac-end / --real-frac-ramp-steps.
    Early on, if the real buffer holds fewer transitions than its share of
    the batch, only what exists is used and the rest comes from sim. If the
    sim buffer is empty, batches are 100% real.

    Combine with --gradient-steps (updates per train() call) to get more
    learning out of each real transition.

Usage — point it at ONE run folder (a sim training run or a previous async
fine-tune run); everything else is found inside it:
    python finetune_async.py runs/<run> --port /dev/ttyPS1 --episodes 50

Looked up in <run>/:
    config.json          training hyperparameters (net_arch, rate, gamma, ...)
    best_model.zip       checkpoint, preferred if present (sim runs) ...
    last.zip             ... otherwise this (fine-tune runs). --checkpoint
                         last|best|<file> overrides.
    replay_buffer/       portable SIM buffer directory, loaded if present
    replay_buffer.pkl    legacy SB3 pickle, used only if the dir is absent
    real_buffer/         portable REAL buffer directory (written by this
                         script), loaded if present
                         (--no-buffer skips both, --no-sim-buffer only sim)

    Note: a replay_buffer/ written by an older version of this script holds
    sim and real data mixed together; it is loaded as the sim buffer.

Output goes to a NEW run folder (runs/async_<source>_<timestamp>, or
--run-name), which contains its own config.json, last.zip, replay_buffer/
(the sim buffer, written once at start) and real_buffer/, so it can be
passed straight back in to continue:
    python finetune_async.py runs/async_<source>_<timestamp> --port ...

The model is rebuilt via `zynq_load.load_sac_for_finetune`, which never
unpickles SB3 metadata and restores the optimizers and the learned entropy
coefficient. Buffers are portable .npy directories (Training/portable_buffer.py),
readable by any NumPy version on either the Zynq or the desktop.

Fine-tuning metrics:
    Every --eval-every'th episode (the 1st, then every Nth after it) runs the
    policy DETERMINISTICALLY with no dt jitter — the way it will be deployed.
    Its transitions still go into the replay buffer, but the policy snapshot
    is frozen for that episode so it scores one fixed set of weights. The
    first eval runs before any real-data gradient step, giving the sim-only
    baseline.

    Per episode (train and eval), computed from the transitions:
        balanced            caught: |θ| < --balance-theta-rad held for
                            --balance-hold-s consecutive seconds
        time_to_balance_s   start of that hold (swing-up + catch time)
        frac_upright        fraction of the episode with |θ| below threshold
        frac_upright_after_catch, theta_rms_deg, motor_pos_std_deg,
        mean_abs_action_balanced (balance quality / dither, after the catch)
        dropped             fell past horizontal after the catch

    Written to <run>/metrics.csv (mode = train|eval) and TensorBoard
    (rollout/* for stochastic episodes, eval/* for deterministic ones), with
    real transitions collected and rig minutes as the x-axes for
    convergence. The best eval policy is saved as best_eval.zip, and
    eval_summary.json records baseline vs best vs final and the first eval
    that caught and held.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
# Only needed for a legacy replay_buffer.pkl in the run folder (pickles written by
# NumPy 2 on the desktop). The portable .npy path never unpickles anything.
if not hasattr(np, "_core"):
    import numpy.core as _core
    sys.modules["numpy._core"] = _core
    sys.modules["numpy._core.numeric"] = _core.numeric

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
from Deploy.FPGA.FPGA_env import RealRotaryInvertedPendulumEnv
from Deploy.FPGA.zynq_load import load_sac_for_finetune
from Training.portable_buffer import export_buffer, load_buffer


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"


class MixedReplayBuffer:
    """Two replay buffers (real + frozen sim) behind one SB3-style interface.

    Installed as `model.replay_buffer`, so SAC's own `train()` uses it
    unchanged: `add()` goes to the real buffer only, and `sample()` builds
    each minibatch from both buffers at the ratio given by `real_frac_fn`
    (a function of the number of real transitions collected so far).
    Any other attribute is forwarded to the real buffer.
    """

    def __init__(self, real, sim, real_frac_fn):
        self.real = real
        self.sim = sim
        self.real_frac_fn = real_frac_fn
        self.last_real_frac = 0.0   # real share of the most recent batch

    def __getattr__(self, name):
        # Only called for attributes not found normally; guard against
        # recursion before __init__ has run (copy/pickle).
        if name in ("real", "sim", "real_frac_fn", "last_real_frac"):
            raise AttributeError(name)
        return getattr(self.real, name)

    def add(self, *args, **kwargs):
        return self.real.add(*args, **kwargs)

    def size(self) -> int:
        return self.real.size() + self.sim.size()

    def target_real_frac(self) -> float:
        """Real share the schedule asks for right now."""
        if self.sim.size() == 0:
            return 1.0
        if self.real.size() == 0:
            return 0.0
        return float(min(1.0, max(0.0, self.real_frac_fn(self.real.size()))))

    def sample(self, batch_size: int, env=None):
        n_real_avail, n_sim_avail = self.real.size(), self.sim.size()
        if n_sim_avail == 0:
            n_real = batch_size
        else:
            n_real = int(round(batch_size * self.target_real_frac()))
            # Don't resample a handful of real transitions dozens of times
            # per batch at the very start; fill the rest from sim.
            n_real = min(n_real, n_real_avail)
        n_sim = batch_size - n_real
        self.last_real_frac = n_real / batch_size

        parts = []
        if n_real > 0:
            parts.append(self.real.sample(n_real, env=env))
        if n_sim > 0:
            parts.append(self.sim.sample(n_sim, env=env))
        if len(parts) == 1:
            return parts[0]
        a, b = parts
        # ReplayBufferSamples is a NamedTuple of tensors; concatenate field
        # by field (optional fields, e.g. `discounts`, may be None).
        return type(a)(*[
            None if x is None else torch.cat((x, y), dim=0)
            for x, y in zip(a, b)
        ])


def _add_to_buffer(model: SAC, transitions, lock: threading.Lock) -> int:
    """Drain a list of `Transition`s into model.replay_buffer under lock.

    model.replay_buffer is a MixedReplayBuffer, so they land in the real
    buffer. Returns the number of transitions added. Updates
    model.num_timesteps (= real transitions collected) so TB step counters
    advance correctly.
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


def _save_artifacts(model: SAC, real_buf, run_dir: Path, *, suffix: str = "") -> None:
    """Save policy (.zip) + REAL replay buffer (portable .npy directory).

    The sim buffer is frozen, so it is written once at startup
    (replay_buffer/) rather than at every checkpoint. Buffers are written
    with portable_buffer.export_buffer instead of SB3's pickle, so they load
    on any NumPy version (Zynq or desktop).
    """
    pol_path = run_dir / ("last.zip" if not suffix else f"checkpoint_{suffix}.zip")
    buf_dir = run_dir / ("real_buffer" if not suffix else f"real_buffer_{suffix}")
    model.save(pol_path)
    n = export_buffer(real_buf, buf_dir)
    print(f"  saved policy → {pol_path.name}, real buffer ({n} transitions) → {buf_dir.name}/")


def _wrap_pi(x: float) -> float:
    """Wrap angle into [-pi, pi] (same as pendulum_env._wrap_pi)."""
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


METRIC_KEYS = (
    "balanced", "time_to_balance_s", "frac_upright",
    "frac_upright_after_catch", "theta_rms_deg", "motor_pos_std_deg",
    "mean_abs_action_balanced", "dropped", "n_valid",
)


def _episode_metrics(transitions, control_freq: float,
                     theta_thresh: float, hold_s: float) -> dict:
    """Balance metrics for one episode from its transitions, in order.

    Uses info["phi"] (tared pendulum angle, 0 = hanging, pi = upright) and
    info["motor_pos"] as reported by RealRotaryInvertedPendulumEnv. Times are
    nominal (step index / control rate). Transitions without "phi" (e.g. a
    serial error) are skipped.
    """
    nan = float("nan")
    out = dict(balanced=0, time_to_balance_s=nan, frac_upright=nan,
               frac_upright_after_catch=nan, theta_rms_deg=nan,
               motor_pos_std_deg=nan, mean_abs_action_balanced=nan,
               dropped=0, n_valid=0)
    th, mp, act = [], [], []
    for t in transitions:
        info = t.info or {}
        if "phi" not in info:
            continue
        th.append(_wrap_pi(float(info["phi"]) - math.pi))
        mp.append(float(info.get("motor_pos", nan)))
        act.append(abs(float(np.asarray(t.action).ravel()[0])))
    n = len(th)
    out["n_valid"] = n
    if n == 0:
        return out
    th, mp, act = np.asarray(th), np.asarray(mp), np.asarray(act)
    up = np.abs(th) < theta_thresh
    out["frac_upright"] = float(up.mean())

    # Catch = first run of hold_n consecutive upright steps.
    hold_n = max(1, int(round(hold_s * control_freq)))
    run, i_catch = 0, None
    for i, u in enumerate(up):
        run = run + 1 if u else 0
        if run >= hold_n:
            i_catch = i - hold_n + 1
            break
    if i_catch is None:
        return out

    th_a, mp_a, act_a, up_a = th[i_catch:], mp[i_catch:], act[i_catch:], up[i_catch:]
    out.update(
        balanced=1,
        # Transition i observes the state after step i+1.
        time_to_balance_s=(i_catch + 1) / control_freq,
        frac_upright_after_catch=float(up_a.mean()),
        theta_rms_deg=float(np.degrees(np.sqrt(np.mean(th_a[up_a] ** 2)))),
        motor_pos_std_deg=float(np.degrees(np.nanstd(mp_a[up_a]))),
        mean_abs_action_balanced=float(act_a[up_a].mean()),
        dropped=int(np.any(np.abs(th_a) > math.pi / 2)),
    )
    return out


def _fmt(x, spec: str = ".2f") -> str:
    return "-" if (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def _write_eval_summary(eval_history: list, run_dir: Path) -> None:
    """Print and save baseline / best / final eval and first successful catch."""
    if not eval_history:
        return
    print("\nDeterministic eval episodes:")
    print("    ep  real_steps  rig_min    return  t_bal_s  upright  held")
    for e in eval_history:
        held = "yes" if e["balanced"] and not e["dropped"] else "no"
        print(f"  {e['episode']:4d}  {e['real_steps']:10d}  {e['rig_minutes']:7.1f}"
              f"  {e['return']:+8.1f}  {_fmt(e['time_to_balance_s']):>7}"
              f"  {_fmt(e['frac_upright']):>7}  {held:>4}")
    best = max(eval_history, key=lambda e: e["return"])
    first_held = next((e for e in eval_history
                       if e["balanced"] and not e["dropped"]), None)
    summary = {
        "baseline": eval_history[0],
        "best": best,
        "final": eval_history[-1],
        "first_held": first_held,
        "n_evals": len(eval_history),
    }
    if first_held is None:
        print("  No eval episode caught and held the pendulum yet.")
    else:
        print(f"  First eval that caught and held: episode {first_held['episode']} "
              f"after {first_held['real_steps']} real transitions "
              f"({first_held['rig_minutes']:.1f} rig minutes)")
    print(f"  Baseline return {eval_history[0]['return']:+.1f} → best "
          f"{best['return']:+.1f} (ep {best['episode']}) → final "
          f"{eval_history[-1]['return']:+.1f}")
    with open(run_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


def _resolve_run_folder(source: Path, checkpoint: str, no_buffer: bool,
                        no_sim_buffer: bool):
    """Locate config, checkpoint and replay buffers inside one run folder.

    Returns (cfg dict, cfg_path or None, policy_path,
             sim_buffer_path or None, real_buffer_path or None).
    Exits with a clear message if the folder or checkpoint is missing.
    """
    if not source.is_dir():
        raise SystemExit(f"ERROR: run folder not found: {source}")

    # Config
    cfg_path = source / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg_path = None

    # Checkpoint
    best, last = source / "best_model.zip", source / "last.zip"
    if checkpoint == "best":
        candidates = [best]
    elif checkpoint == "last":
        candidates = [last]
    elif checkpoint == "auto":
        candidates = [best, last]
    else:  # explicit file name, relative to the run folder
        candidates = [source / checkpoint]
    policy_path = next((c for c in candidates if c.exists()), None)
    if policy_path is None:
        raise SystemExit("ERROR: no checkpoint in %s (looked for: %s)"
                         % (source, ", ".join(c.name for c in candidates)))

    # Sim buffer: portable dir first, legacy pickle as fallback
    sim_buffer_path = None
    if not (no_buffer or no_sim_buffer):
        for c in (source / "replay_buffer", source / "replay_buffer.pkl"):
            if c.exists():
                sim_buffer_path = c
                break

    # Real buffer (only exists in runs written by this script)
    real_buffer_path = None
    if not no_buffer and (source / "real_buffer").is_dir():
        real_buffer_path = source / "real_buffer"

    return cfg, cfg_path, policy_path, sim_buffer_path, real_buffer_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path,
                   help="run folder to start from (sim training run or a "
                        "previous async fine-tune run). config.json, the "
                        "checkpoint and the replay buffers are found inside.")
    p.add_argument("--checkpoint", default="auto",
                   help="which checkpoint in the run folder: 'auto' "
                        "(best_model.zip if present, else last.zip), 'best', "
                        "'last', or a file name inside the folder, e.g. "
                        "checkpoint_ep020.zip")
    p.add_argument("--no-buffer", action="store_true",
                   help="don't load any replay buffer from the run folder; "
                        "start fine-tuning with nothing (real data only)")
    p.add_argument("--no-sim-buffer", action="store_true",
                   help="don't load the sim buffer (replay_buffer/), but do "
                        "load real_buffer/ if present. Batches are then 100%% "
                        "real.")
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
    p.add_argument("--control-freq", type=float, default=None,
                   help="strict control rate in Hz; this orchestrator holds "
                        "this rate even at high --gradient-steps. Default: "
                        "control_freq_hz from config.json (else 50). "
                        "Overriding it changes what the policy's dynamics "
                        "model sees — only do it on purpose.")
    p.add_argument("--max-accel-rad-s2", type=float, default=None,
                   help="action ∈ [-1, 1] maps to angular accel ∈ [-max, +max]"
                        " rad/s². Default: max_accel_rad_s2 from config.json "
                        "(else 150). Must match sim training.")
    p.add_argument("--buffer-size", type=int, default=200_000,
                   help="capacity of EACH replay buffer (sim and real) on the "
                        "device. If a loaded buffer is larger, its most recent "
                        "transitions are kept. ~200 B per transition for a "
                        "24-dim obs, so both together need ~2x that.")
    p.add_argument("--real-frac-start", type=float, default=0.5,
                   help="share of each minibatch drawn from the REAL buffer "
                        "at the start of fine-tuning (0..1)")
    p.add_argument("--real-frac-end", type=float, default=0.9,
                   help="share of each minibatch drawn from the REAL buffer "
                        "once --real-frac-ramp-steps real transitions exist")
    p.add_argument("--real-frac-ramp-steps", type=int, default=40_000,
                   help="number of real transitions over which the real "
                        "share ramps linearly from start to end. 0 = use "
                        "--real-frac-end from the beginning.")
    p.add_argument("--gamma", type=float, default=None,
                   help="override the discount. Default: 0.99^(35/f) from "
                        "the config's control rate, same as train_sac.py. "
                        "Pass it if the policy was trained with --gamma.")
    p.add_argument("--use-sde", action="store_true",
                   help="set if the policy was trained with --use-sde "
                        "(changes actor parameter shapes; not in config.json)")
    p.add_argument("--no-mirror", action="store_true",
                   help="use a plain ReplayBuffer even if the run trained "
                        "with mirror augmentation (e.g. Training.symmetry "
                        "isn't available on the device)")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="SAC gradient updates per train() call. Higher "
                        "extracts more from each real transition")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-starts", type=int, default=100,
                   help="number of REAL transitions before training begins")
    p.add_argument("--checkpoint-freq", type=int, default=10,
                   help="save policy + real replay buffer every N episodes")
    p.add_argument("--run-name", default=None,
                   help="OUTPUT run dir name under runs/. Default: "
                        "async_<source run>_<timestamp>. Must differ from "
                        "the source run so its files aren't overwritten.")
    p.add_argument("--device", default="cpu",
                   help="torch device. CPU is fine for this small policy")
    p.add_argument("--timing-violation-threshold-ms", type=float, default=10.0,
                   help="control-loop overrun above this for 3 consecutive "
                        "ticks raises TimingViolation and disengages motor")
    p.add_argument("--deterministic", action="store_true",
                   help="use deterministic actions during data collection. "
                        "Default is stochastic, matching SAC training-time "
                        "behaviour and what the policy was optimised for.")
    # Reward weights are read from config.json (via
    # RealRotaryInvertedPendulumEnv.from_config) so the rig optimises the
    # same objective as the sim run. These flags only exist to override the
    # config on purpose; a mismatch is warned about.
    p.add_argument("--reward-stillness-bonus-weight", type=float, default=None,
                   help="override config.json's reward_stillness_bonus_weight")
    p.add_argument("--reward-alive-offset", type=float, default=None,
                   help="override config.json's reward_alive_offset")
    p.add_argument("--reward-upright-alive-weight", type=float, default=None,
                   help="override config.json's reward_upright_alive_weight")
    p.add_argument("--dt-jitter-frac", type=float, default=0.05,
                   help="control-rate jitter (DR on dt). Each tick interval "
                        "is multiplied by uniform(1-frac, 1+frac). Mimics "
                        "the legacy variable-rate fine-tune that produced "
                        "the calm 'minimal action' attractor. Default 0.05 "
                        "matches the sim DR_CONTROL_DT_JITTER_FRAC constant — "
                        "sim and fine-tune should agree on the dt distribution. "
                        "Set 0.0 to disable for strict reproducible timing.")
    p.add_argument("--eval-every", type=int, default=5,
                   help="run a deterministic eval episode as the 1st episode "
                        "and then every N episodes (1, N+1, 2N+1, ...). Its "
                        "transitions still go into the buffer. 0 disables.")
    p.add_argument("--eval-dt-jitter-frac", type=float, default=0.0,
                   help="dt jitter during eval episodes. Default 0 (strict "
                        "timing, as deployed); set it equal to "
                        "--dt-jitter-frac to eval under training conditions.")
    p.add_argument("--balance-theta-rad", type=float, default=0.2,
                   help="|θ| from upright below which the pendulum counts as "
                        "upright for the balance metrics (0.2 rad ≈ 11°)")
    p.add_argument("--balance-hold-s", type=float, default=1.0,
                   help="seconds |θ| must stay below --balance-theta-rad to "
                        "count as a catch")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    for name in ("real_frac_start", "real_frac_end"):
        v = getattr(args, name)
        if not 0.0 <= v <= 1.0:
            print(f"ERROR: --{name.replace('_', '-')} must be in [0, 1], got {v}",
                  file=sys.stderr)
            return 2

    torch.set_num_threads(1)

    # ---- Resolve the source run folder (before creating any output) ----
    source = args.run.resolve()
    cfg, cfg_path, policy_path, sim_buffer_path, real_buffer_path = \
        _resolve_run_folder(source, args.checkpoint, args.no_buffer,
                            args.no_sim_buffer)
    print(f"Source run: {source}")
    print(f"  checkpoint:    {policy_path.name}")
    print(f"  config:        {cfg_path.name if cfg_path else 'MISSING'}")
    print(f"  sim buffer:    {sim_buffer_path.name if sim_buffer_path else 'none'}")
    print(f"  real buffer:   {real_buffer_path.name if real_buffer_path else 'none (starting empty)'}")
    if cfg_path is None:
        print("WARNING: no config.json in the run folder; using "
              "train_sac.py defaults (net_arch 256,256, 50 Hz, 150 rad/s², "
              "no mirror). Override with --control-freq / "
              "--max-accel-rad-s2 / --no-mirror if these are wrong.",
              file=sys.stderr)

    # ---- Output run folder ----
    run_name = (args.run_name or
                f"async_{source.name}_{time.strftime('%Y-%m-%d_%H%M')}")
    run_dir = (RUNS_ROOT / run_name).resolve()
    if run_dir == source:
        print("ERROR: output run folder is the source run folder; pick a "
              "different --run-name so the source isn't overwritten.",
              file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output run:  {run_dir}")

    def _resolve(cli_value, key, fallback):
        trained = cfg.get(key)
        if cli_value is None:
            return float(trained) if trained is not None else fallback
        if trained is not None and abs(float(trained) - cli_value) > 1e-9:
            print(f"WARNING: --{key} {cli_value:g} differs from training "
                  f"value {float(trained):g}", file=sys.stderr)
        return cli_value

    control_freq = _resolve(args.control_freq, "control_freq_hz", 50.0)
    max_accel = _resolve(args.max_accel_rad_s2, "max_accel_rad_s2", 150.0)

    # Record the config in the output run so passing this run folder back in
    # finds it automatically. Kept key-for-key compatible with train_sac.py's
    # config.json (run_config.check_config compares them), so provenance goes
    # in a separate file.
    cfg_out = dict(cfg)
    cfg_out["control_freq_hz"] = control_freq
    cfg_out["max_accel_rad_s2"] = max_accel
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(run_dir / "source.json", "w") as f:
        json.dump({"source_run": str(source),
                   "checkpoint": policy_path.name,
                   "sim_buffer": sim_buffer_path.name if sim_buffer_path else None,
                   "real_buffer": real_buffer_path.name if real_buffer_path else None,
                   "real_frac_start": args.real_frac_start,
                   "real_frac_end": args.real_frac_end,
                   "real_frac_ramp_steps": args.real_frac_ramp_steps,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    # Observation layout and reward weights come from the run's config.json,
    # exactly as the sim env was built; only rig-specific settings and
    # deliberate CLI overrides are passed on top.
    env_overrides = dict(
        port=args.port,
        baud=args.baud,
        control_freq_hz=control_freq,
        max_accel_rad_s2=max_accel,
        episode_length_s=args.episode_length_s,
        reset_settle_s=args.reset_settle_s,
    )
    for flag, key in (
        (args.reward_stillness_bonus_weight, "reward_stillness_bonus_weight"),
        (args.reward_alive_offset, "reward_alive_offset"),
        (args.reward_upright_alive_weight, "reward_upright_alive_weight"),
    ):
        if flag is not None:
            trained = cfg.get(key)
            if trained is not None and abs(float(trained) - flag) > 1e-9:
                print(f"WARNING: --{key.replace('_', '-')} {flag:g} differs from "
                      f"training value {float(trained):g}", file=sys.stderr)
            env_overrides[key] = flag
    if not any(k in cfg for k in ("reward_alive_offset",
                                  "reward_upright_alive_weight")):
        print("WARNING: config.json has no alive-reward keys; the rig will use "
              "the plain quadratic cost (all-negative reward). If the sim run "
              "used --reward-alive-offset / --reward-upright-alive-weight, pass "
              "them here.", file=sys.stderr)
    env = RealRotaryInvertedPendulumEnv.from_config(cfg, **env_overrides)
    print(f"Control: {control_freq:g} Hz, max accel {max_accel:g} rad/s²")

    # Persist any reward overrides so continuing from this output run keeps
    # optimising the same objective.
    reward_overrides = {k: v for k, v in env_overrides.items()
                        if k.startswith("reward_")}
    if reward_overrides:
        cfg_out.update(reward_overrides)
        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

    # ---- Model ----
    # Rebuilt from config.json with the training hyperparameters; restores
    # policy, target critics, optimizers and the learned entropy coefficient
    # straight from the zip's .pth members — no cloudpickle, so NumPy/CUDA
    # mismatches between the desktop and the Zynq don't matter.
    # Same constant-horizon rule as train_sac.py, tied to the rate actually
    # used on the rig.
    gamma = (args.gamma if args.gamma is not None
             else 0.99 ** (35.0 / control_freq))
    print(f"Loading policy from {policy_path}")
    model = load_sac_for_finetune(
        str(policy_path), env, run_config=cfg_out,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,  # sets lr_schedule, which train() reads
        learning_starts=args.learning_starts,
        gamma=gamma,
        use_sde=args.use_sde,
        mirror=False if args.no_mirror else None,
        device=args.device,
    )
    model.tensorboard_log = str(run_dir / "tb")

    # ---- Replay buffers ----
    # The buffer load_sac_for_finetune built is still empty and has the right
    # class (incl. mirror augmentation) and capacity; copy it as the real
    # buffer before the sim data is loaded into the original.
    real_buf = copy.deepcopy(model.replay_buffer)

    if sim_buffer_path is not None:
        rb_path = sim_buffer_path
        print(f"Loading sim replay buffer from {rb_path}")
        if rb_path.is_dir():
            # Portable format: shape checks (obs/action/n_envs) happen inside.
            try:
                load_buffer(model.replay_buffer, rb_path)
            except (ValueError, IOError) as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
        else:
            # Legacy SB3 pickle. Replaces the buffer object, including its
            # class and capacity; needs the numpy._core shim above.
            model.load_replay_buffer(str(rb_path))
            if model.replay_buffer.observation_space.shape != env.observation_space.shape:
                print("ERROR: loaded buffer's observation space does not match env's. "
                      "Was the buffer collected with a different obs construction?",
                      file=sys.stderr)
                return 2
    sim_buf = model.replay_buffer

    if real_buffer_path is not None:
        print(f"Loading real replay buffer from {real_buffer_path}")
        try:
            load_buffer(real_buf, real_buffer_path)
        except (ValueError, IOError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    def real_frac_schedule(n_real: int) -> float:
        if args.real_frac_ramp_steps <= 0:
            return args.real_frac_end
        t = min(1.0, n_real / args.real_frac_ramp_steps)
        return args.real_frac_start + t * (args.real_frac_end - args.real_frac_start)

    mixed = MixedReplayBuffer(real_buf, sim_buf, real_frac_schedule)
    model.replay_buffer = mixed
    print(f"  sim buffer:  {sim_buf.size()} transitions (frozen)")
    print(f"  real buffer: {real_buf.size()} transitions")
    if sim_buf.size() > 0:
        print(f"  real share of each batch: {args.real_frac_start:g} → "
              f"{args.real_frac_end:g} over {args.real_frac_ramp_steps} real "
              f"transitions (now {mixed.target_real_frac():.2f})")
    else:
        print("  no sim data: batches are 100% real")

    # The sim buffer never changes, so write it into the output run once;
    # this keeps the output folder self-contained for continuing.
    if sim_buf.size() > 0:
        n = export_buffer(sim_buf, run_dir / "replay_buffer")
        print(f"  wrote sim buffer ({n} transitions) → replay_buffer/")

    # num_timesteps counts REAL transitions only. Checkpoints carry their
    # sim-training step counter (e.g. 280k from the curriculum), and the sim
    # buffer is not real data either; neither should satisfy learning_starts
    # or show up as "real_steps" in the metrics.
    model.num_timesteps = real_buf.size()
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

    metrics_csv = run_dir / "metrics.csv"
    metrics_f = open(metrics_csv, "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow([
        "episode", "mode", "real_steps", "rig_minutes", "return", "n_steps",
        "terminated", "real_frac", *METRIC_KEYS,
    ])
    metrics_f.flush()
    eval_history: list[dict] = []
    best_eval_return = -math.inf

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
          f"{args.episode_length_s} s @ {control_freq:g} Hz, "
          f"K={args.gradient_steps}, lr={args.learning_rate}")
    print(f"Real transitions so far: {real_buf.size()}, "
          f"learning_starts={args.learning_starts}")

    # Rolling window over the last 100 episodes for SB3-style running-mean
    # scalars (rollout/ep_rew_mean, rollout/ep_len_mean). Same default
    # window SB3's `_dump_logs` uses.
    recent_rewards: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)

    if args.eval_every > 0:
        print(f"Deterministic eval every {args.eval_every} episodes "
              f"(first = sim-only baseline), dt jitter "
              f"{args.eval_dt_jitter_frac:g}; balance = |θ| < "
              f"{args.balance_theta_rad:g} rad for {args.balance_hold_s:g} s")

    # ---- Main loop ----
    rig_t0 = time.monotonic()
    try:
        for ep_idx in range(args.episodes):
            if stop_flag.is_set():
                break
            is_eval = args.eval_every > 0 and ep_idx % args.eval_every == 0
            mode = "eval" if is_eval else "train"
            print(f"\n[ep {ep_idx + 1}/{args.episodes}] ({mode}) reset (settle "
                  f"{args.reset_settle_s} s) → engage → run")
            if is_eval:
                # Save exactly the weights this eval will score (training keeps
                # running during the episode, so `model` moves on). Promoted
                # to best_eval.zip if this eval wins.
                model.save(run_dir / "eval_candidate.zip")
            try:
                env.reset()
            except Exception as e:
                print(f"ERROR during env.reset(): {e}", file=sys.stderr)
                break

            loop = AsyncControlLoop(
                env, snapshot, queue, stop_flag,
                control_freq_hz=control_freq,
                timing_violation_threshold_s=args.timing_violation_threshold_ms / 1000.0,
                deterministic_actions=args.deterministic or is_eval,
                dt_jitter_frac=(args.eval_dt_jitter_frac if is_eval
                                else args.dt_jitter_frac),
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
            ep_transitions: list = []
            try:
                while ctrl_thread.is_alive():
                    drained = queue.drain()
                    ep_transitions.extend(drained)
                    added = _add_to_buffer(model, drained, replay_buffer_lock)

                    # Gate on REAL transitions; the sim buffer alone never
                    # starts training.
                    can_train = (
                        real_buf.size() >= args.learning_starts
                        and mixed.size() >= model.batch_size
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
                        # Eval episodes keep training but freeze the
                        # snapshot, so they score one fixed policy.
                        if not is_eval:
                            snapshot.refresh_from(model.actor)
                    else:
                        # Warmup: avoid busy-loop while we have nothing to train on.
                        time.sleep(0.001)


                # Drain any straggler transitions after episode end.
                drained = queue.drain()
                ep_transitions.extend(drained)
                _add_to_buffer(model, drained, replay_buffer_lock)
            finally:
                ctrl_thread.join(timeout=2.0)
                if is_eval and n_train > 0:
                    # Hand the weights trained during the eval to the next episode.
                    snapshot.refresh_from(model.actor)

            ep_stats = loop.last_episode_stats
            if ep_stats is None:
                print(f"  WARNING: episode {ep_idx} produced no stats "
                      "(thread crashed before completion).", file=sys.stderr)
                continue

            real_frac_now = mixed.last_real_frac if n_train > 0 else float("nan")
            print(f"  steps={ep_stats.n_steps}  reward={ep_stats.cumulative_reward:+.1f}  "
                  f"mean_dt={ep_stats.mean_dt_ms:.2f}ms  "
                  f"max_overrun={ep_stats.tick_stats.max_overrun_ms:.2f}ms  "
                  f"violations={ep_stats.tick_stats.n_violations}  "
                  f"train_calls={n_train}  train_s={train_s:.2f}")
            print(f"  buffers: real={real_buf.size()}  sim={sim_buf.size()}  "
                  f"real share of batch={_fmt(real_frac_now)}")
            _log_episode(ep_stats, n_train, train_s)

            # ---- Balance metrics ----
            m = _episode_metrics(ep_transitions, control_freq,
                                 args.balance_theta_rad, args.balance_hold_s)
            rig_minutes = (time.monotonic() - rig_t0) / 60.0
            if m["n_valid"] == 0:
                print("  WARNING: no transition carried info['phi']; balance "
                      "metrics unavailable for this episode.", file=sys.stderr)
            print(f"  [{mode}] balanced={'yes' if m['balanced'] else 'no'}  "
                  f"t_bal={_fmt(m['time_to_balance_s'])}s  "
                  f"upright={_fmt(m['frac_upright'])}  "
                  f"θ_rms={_fmt(m['theta_rms_deg'])}°  "
                  f"|a|={_fmt(m['mean_abs_action_balanced'])}  "
                  f"dropped={'yes' if m['dropped'] else 'no'}")
            metrics_writer.writerow([
                ep_idx + 1, mode, model.num_timesteps, f"{rig_minutes:.2f}",
                f"{ep_stats.cumulative_reward:.3f}", ep_stats.n_steps,
                int(ep_stats.terminated), _fmt(real_frac_now, ".3f"),
                *[m[k] for k in METRIC_KEYS],
            ])
            metrics_f.flush()

            prefix = "eval" if is_eval else "rollout"
            model.logger.record(f"{prefix}/balanced", m["balanced"])
            model.logger.record(f"{prefix}/frac_upright",
                                0.0 if math.isnan(m["frac_upright"]) else m["frac_upright"])
            model.logger.record(f"{prefix}/dropped", m["dropped"])
            if m["balanced"]:
                # Only logged when there was a catch, so TB curves aren't
                # polluted by NaN / placeholder values.
                for k in ("time_to_balance_s", "frac_upright_after_catch",
                          "theta_rms_deg", "motor_pos_std_deg",
                          "mean_abs_action_balanced"):
                    model.logger.record(f"{prefix}/{k}", m[k])
            model.logger.record("time/rig_minutes", rig_minutes)
            model.logger.record("replay/real_size", real_buf.size())
            model.logger.record("replay/sim_size", sim_buf.size())
            if n_train > 0:
                model.logger.record("replay/real_frac", real_frac_now)

            if is_eval:
                entry = dict(episode=ep_idx + 1, real_steps=int(model.num_timesteps),
                             rig_minutes=rig_minutes,
                             **{"return": float(ep_stats.cumulative_reward)},
                             n_steps=int(ep_stats.n_steps), **m)
                eval_history.append(entry)
                model.logger.record("eval/ep_reward", ep_stats.cumulative_reward)
                model.logger.record("eval/ep_length", ep_stats.n_steps)
                model.logger.record("eval/n_evals", len(eval_history))
                if ep_stats.cumulative_reward > best_eval_return:
                    best_eval_return = ep_stats.cumulative_reward
                    (run_dir / "eval_candidate.zip").replace(run_dir / "best_eval.zip")
                    print(f"  new best deterministic eval ({best_eval_return:+.1f}) "
                          f"→ best_eval.zip")

            # Flush logger to TB once per episode. SB3's `train()` records
            # scalars internally but doesn't dump — that normally happens in
            # learn()'s outer loop. We also append our own episode-level
            # metrics so reward/timing curves show up alongside SAC's
            # internals. The `rollout/ep_rew_mean` and `rollout/ep_len_mean`
            # keys match SB3's standard names so TB layouts that look for
            # them populate correctly.
            # rollout/* covers stochastic (training) episodes only; the
            # deterministic evals are logged under eval/* above.
            if not is_eval:
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
                _save_artifacts(model, real_buf, run_dir, suffix=f"ep{ep_idx + 1:03d}")

    finally:
        stop_flag.set()
        env.disengage_safely()
        try:
            env.close()
        except Exception:
            pass
        print("\nFinal save:")
        _save_artifacts(model, real_buf, run_dir)
        timing_f.close()
        metrics_f.close()
        _write_eval_summary(eval_history, run_dir)
        (run_dir / "eval_candidate.zip").unlink(missing_ok=True)
        print(f"Run artifacts in: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
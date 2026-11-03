"""Asynchronous SAC fine-tuning on the real rig — policy inference on the CPU
or on the PL, selected with --inference. Everything else is shared, so the two
deployment methods can be compared one-to-one.

    python finetune_async.py runs/<run> --inference cpu --port /dev/ttyPS1 --episodes 50
    python finetune_async.py runs/<run> --inference pl  --param-noise 0.01 --episodes 50

Threads (same for both modes):

  Rig thread — owns all rig I/O. Per episode:
      PD-home the motor -> short stillness check (--rest-check-s): tare only
      if the pendulum is still, otherwise start anyway without a tare (it may
      still be swinging; --reset-settle-s > 0 waits up to that long for
      stillness instead) -> sync the policy with the learner -> run the
      episode -> report it.
        cpu: `async_control.AsyncControlLoop` runs the policy on the CPU at a
             strict control rate (PolicySnapshot), streaming transitions.
             Env: Deploy/FPGA/FPGA_env.RealRotaryInvertedPendulumEnv.
        pl : the PL runs the MLP itself; the CPU only observes the step log
             and gets exact (obs, action) pairs at the end of the episode.
             Env: Deploy/FPGA/FPGA_deploy_env.FpgaDeployEnv (deployment bitstream).

  Main thread (learner) — trains SAC continuously in chunks of
      --gradient-steps while episodes run, the motor homes and the pendulum
      settles. Between chunks it adds transitions to the replay buffer and
      answers the rig's policy-sync request. Holds the SIGINT handler (Python
      only delivers signals to the main thread): Ctrl-C disengages the motor,
      stops the rig thread and saves.

Policy sync (--weight-sync):
    continuous  (default) the newest weights at episode start, then updated
                during the episode as training goes on:
                  cpu: PolicySnapshot refreshed after every train() call;
                  pl : new weights uploaded to the inactive bank of the
                       double-buffered weight memory at most every
                       --pl-upload-interval-s; the PL swaps banks between two
                       inferences.
    episode     the policy is fixed for a whole episode (newest weights when
                it starts).
    Eval episodes always keep one fixed policy. With --param-noise the noise is
    seeded per episode (seed, episode), so every mid-episode upload carries the
    SAME noise draw and only the training updates change the PL weights.

Exploration:
    cpu: stochastic SAC actions (unless --deterministic) + dt jitter.
    pl : the PL policy is deterministic -> --param-noise (Gaussian noise on the
         uploaded weights, one draw per episode, also across mid-episode
         uploads; the learner's actor stays clean).

Update-to-data ratio: the learner trains whenever it can, also during homing
and rest waits, so gradient steps per real transition depend on how long those
take. Both modes report it (grad_steps per episode, grad_steps_total,
grad_per_transition); --max-grad-per-transition caps it.

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

Point it at ONE run folder (a sim training run or a previous fine-tune run);
everything else is found inside it:
    config.json          training hyperparameters (net_arch, rate, gamma, ...)
    best_model.zip       checkpoint, preferred if present (sim runs) ...
    last.zip             ... otherwise this (fine-tune runs). --checkpoint
                         last|best|<file> overrides.
    replay_buffer/       portable SIM buffer directory, loaded if present
    replay_buffer.pkl    legacy SB3 pickle, used only if the dir is absent
    real_buffer/         portable REAL buffer directory (written by this
                         script), loaded if present
                         (--no-buffer skips both, --no-sim-buffer only sim)

    The PL mode needs a checkpoint that matches the PL network (obs_history_len,
    net-arch, velocities on, obs_norm="raw"); FpgaDeployEnv checks this.

Output goes to a NEW run folder (runs/async_<inference>_<source>_<timestamp>,
or --run-name) with its own config.json, last.zip, replay_buffer/ (sim buffer,
written once) and real_buffer/, so it can be passed straight back in.
source.json records the inference mode and the fine-tune settings.

The model is rebuilt via `zynq_load.load_sac_for_finetune`, which never
unpickles SB3 metadata and restores the optimizers and the learned entropy
coefficient. Buffers are portable .npy directories (Training/portable_buffer.py).

Fine-tuning metrics (finetune_metrics.py):
    Every --eval-every'th episode (the 1st, then every Nth after it) runs the
    policy DETERMINISTICALLY with fixed weights — the way it will be deployed
    (cpu: deterministic actions, --eval-dt-jitter-frac; pl: no parameter
    noise). Its transitions still go into the replay buffer. The first eval
    runs before any real-data gradient step, giving the sim-only baseline.

    Per episode: balanced, time_to_balance_s, frac_upright,
    frac_upright_after_catch, theta_rms_deg, motor_pos_std_deg,
    mean_abs_action_balanced, dropped, plus gradient steps.

    Written to <run>/metrics.csv (mode = train|eval), <run>/timing.csv and
    TensorBoard (rollout/* for training episodes, eval/* for evals), with
    real transitions collected and rig minutes as the x-axes. The best eval
    policy is saved as best_eval.zip, and eval_summary.json records baseline
    vs best vs final and the first eval that caught and held.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import queue
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
    PolicySnapshot,
    TimingViolation,
    Transition,
    TransitionQueue,
)
from Deploy.FPGA.zynq_load import load_sac_for_finetune
from Training.portable_buffer import export_buffer, load_buffer
from finetune_metrics import (
    METRIC_KEYS,
    balance_metrics,
    fmt as _fmt,
    wrap_pi as _wrap_pi,
    write_eval_summary as _write_eval_summary,
)


HERE = Path(__file__).resolve().parent
RUNS_ROOT = HERE / "runs"
SYNC_TIMEOUT_S = 5.0   # rig waits this long for the learner's policy sync


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


def _add_to_buffer(model: SAC, transitions) -> int:
    """Add a list of `Transition`s to model.replay_buffer (main thread only).

    model.replay_buffer is a MixedReplayBuffer, so they land in the real
    buffer. Updates model.num_timesteps (= real transitions collected) so TB
    step counters advance correctly. Returns the number added.
    """
    for t in transitions:
        model.replay_buffer.add(
            obs=np.expand_dims(t.obs, axis=0),
            next_obs=np.expand_dims(t.next_obs, axis=0),
            action=np.expand_dims(t.action, axis=0),
            reward=np.array([t.reward], dtype=np.float32),
            done=np.array([t.terminated or t.truncated], dtype=np.bool_),
            infos=[t.info],
        )
    model.num_timesteps += len(transitions)
    return len(transitions)


def _pl_episode_transitions(ep: dict) -> list:
    """FpgaDeployEnv.collect_episode() arrays -> `Transition`s. Steps across a
    log gap or overridden by the rail brake are dropped (their action is not
    what was applied). Truncation is marked on the last step, as in the CPU loop."""
    n = ep["length"]
    out = []
    for i in range(n):
        if not ep["valid"][i]:
            continue
        term = bool(ep["terminated"][i])
        trunc = bool(ep["truncated"] and i == n - 1 and not term)
        out.append(Transition(
            obs=ep["obs"][i], action=ep["actions"][i].reshape(-1),
            reward=float(ep["rewards"][i]), next_obs=ep["next_obs"][i],
            terminated=term, truncated=trunc,
            info={"TimeLimit.truncated": trunc},
        ))
    return out


def _save_artifacts(model: SAC, real_buf, run_dir: Path, *, suffix: str = "") -> None:
    """Save policy (.zip) + REAL replay buffer (portable .npy directory).

    The sim buffer is frozen, so it is written once at startup
    (replay_buffer/) rather than at every checkpoint.
    """
    pol_path = run_dir / ("last.zip" if not suffix else f"checkpoint_{suffix}.zip")
    buf_dir = run_dir / ("real_buffer" if not suffix else f"real_buffer_{suffix}")
    model.save(pol_path)
    n = export_buffer(real_buf, buf_dir)
    print(f"  saved policy → {pol_path.name}, real buffer ({n} transitions) → {buf_dir.name}/")


def _cpu_episode_metrics(transitions, control_freq: float,
                         theta_thresh: float, hold_s: float) -> dict:
    """Balance metrics from CPU-loop transitions, in order.

    Uses info["phi"] (tared pendulum angle, 0 = hanging, pi = upright) and
    info["motor_pos"] as reported by RealRotaryInvertedPendulumEnv.
    Transitions without "phi" (e.g. a serial error) are skipped.
    """
    th, mp, act = [], [], []
    for t in transitions:
        info = t.info or {}
        if "phi" not in info:
            continue
        th.append(_wrap_pi(float(info["phi"]) - math.pi))
        mp.append(float(info.get("motor_pos", float("nan"))))
        act.append(abs(float(np.asarray(t.action).ravel()[0])))
    return balance_metrics(th, mp, act, control_freq, theta_thresh, hold_s)


def _pl_episode_metrics(ep: dict, control_freq: float,
                        theta_thresh: float, hold_s: float) -> dict:
    """Balance metrics from the PL episode arrays (newest frame of next_obs =
    [motor_pos, sin θ, cos θ, ...], θ = 0 upright)."""
    frame = ep["next_obs"][:, -6:]
    return balance_metrics(np.arctan2(frame[:, 1], frame[:, 2]), frame[:, 0],
                           np.abs(ep["actions"][:, 0]), control_freq,
                           theta_thresh, hold_s)


def _resolve_run_folder(source: Path, checkpoint: str, no_buffer: bool,
                        no_sim_buffer: bool):
    """Locate config, checkpoint and replay buffers inside one run folder.

    Returns (cfg dict, cfg_path or None, policy_path,
             sim_buffer_path or None, real_buffer_path or None).
    Exits with a clear message if the folder or checkpoint is missing.
    """
    if not source.is_dir():
        raise SystemExit(f"ERROR: run folder not found: {source}")

    cfg_path = source / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg_path = None

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

    sim_buffer_path = None
    if not (no_buffer or no_sim_buffer):
        for c in (source / "replay_buffer", source / "replay_buffer.pkl"):
            if c.exists():
                sim_buffer_path = c
                break

    real_buffer_path = None
    if not no_buffer and (source / "real_buffer").is_dir():
        real_buffer_path = source / "real_buffer"

    return cfg, cfg_path, policy_path, sim_buffer_path, real_buffer_path


# ---------------------------------------------------------------------------
# Rig thread
# ---------------------------------------------------------------------------

class PolicySync:
    """Rig thread asks for a policy sync right before an episode starts; the
    learner answers between train() calls, so the actor is never read while
    the optimizer is writing it."""

    def __init__(self):
        self._requested = threading.Event()
        self._ready = threading.Event()
        self._payload = None
        self._meta: dict = {}
        self.requested_episode = 0

    def request(self, timeout: float, episode: int):
        """Rig thread: block until the learner answers (or timeout -> None, {})."""
        self._ready.clear()
        self.requested_episode = episode
        self._requested.set()
        if not self._ready.wait(timeout):
            self._requested.clear()
            return None, {}
        return self._payload, self._meta

    def pending(self) -> bool:
        return self._requested.is_set()

    def provide(self, payload, meta: dict) -> None:
        """Learner thread."""
        self._payload, self._meta = payload, meta
        self._requested.clear()
        self._ready.set()


class LatestWeights:
    """PL continuous sync: newest exported weight image for the running
    episode. The learner overwrites it; the rig thread uploads it between
    steps when the version is newer than what the PL holds."""

    def __init__(self):
        self._lock = threading.Lock()
        self.episode, self.version, self.words, self.grad_step = -1, 0, None, -1

    def put(self, episode: int, words, grad_step: int) -> int:
        with self._lock:
            self.version += 1
            self.episode, self.words, self.grad_step = episode, words, grad_step
            return self.version

    def get(self):
        with self._lock:
            return self.episode, self.version, self.words, self.grad_step


def _cpu_rig_worker(env, snapshot, tq, sync: PolicySync, out: queue.Queue,
                    stop: threading.Event, *, n_episodes: int, control_freq: float,
                    timing_threshold_s: float, deterministic: bool,
                    dt_jitter: float, eval_dt_jitter: float) -> None:
    """CPU inference: reset (home, still, tare, sync) then the strict-rate
    AsyncControlLoop, which streams transitions into `tq`."""
    try:
        for ep_i in range(n_episodes):
            if stop.is_set():
                break
            info: dict = {}

            def before_start():
                payload, meta = sync.request(SYNC_TIMEOUT_S, ep_i)
                info.update(meta, synced=payload is not None)
                if payload is None:
                    print("  [warn] learner did not sync the policy in time; "
                          "running the current snapshot")

            _, reset_info = env.reset(options={"before_start": before_start})
            is_eval = bool(info.get("synced") and info.get("is_eval"))
            loop = AsyncControlLoop(
                env, snapshot, tq, stop,
                control_freq_hz=control_freq,
                timing_violation_threshold_s=timing_threshold_s,
                deterministic_actions=deterministic or is_eval,
                dt_jitter_frac=eval_dt_jitter if is_eval else dt_jitter,
            )
            try:
                stats = loop.run_episode(ep_i)
                err = None
            except TimingViolation as e:
                stats, err = None, e
            out.put(("episode", ep_i, {"stats": stats, "error": err, **reset_info}, info))
    except KeyboardInterrupt:
        pass                                    # stop requested
    except Exception as err:                    # surface rig errors in the main thread
        out.put(("error", err))
    finally:
        env.disengage_safely()
        out.put(("done",))


def _pl_rig_worker(env, sync: PolicySync, latest: LatestWeights | None,
                   out: queue.Queue, stop: threading.Event, *, n_episodes: int) -> None:
    """PL inference: reset (home, tare check, upload newest weights) then
    observe the PL episode; with `latest` (continuous sync) newer weights are
    uploaded between steps. Transitions arrive with the episode."""
    try:
        for ep_i in range(n_episodes):
            if stop.is_set():
                break
            info: dict = {"n_weight_updates": 0}
            uploaded = {"version": -1}

            def before_start():
                words, meta = sync.request(SYNC_TIMEOUT_S, ep_i)
                info.update(meta, synced=words is not None)
                if words is None:
                    print("  [warn] learner did not provide weights in time; "
                          "reusing the previous ones")
                    return
                t0 = time.monotonic()
                st = env.client.load_weights(words)
                info.update(upload_s=time.monotonic() - t0, weight_bank=st.active_bank,
                            weights_grad_step_end=meta.get("weights_grad_step", -1))
                uploaded["version"] = meta.get("version", -1)

            def on_step():
                ep, ver, words, gs = latest.get()
                if ep != ep_i or ver <= uploaded["version"] or words is None:
                    return
                t0 = time.monotonic()
                env.client.load_weights(words)
                dt = time.monotonic() - t0
                uploaded["version"] = ver
                info["n_weight_updates"] += 1
                info["weights_grad_step_end"] = gs
                info["max_upload_s"] = max(info.get("max_upload_s", 0.0), dt)

            ep = env.collect_episode(before_start=before_start,
                                     on_step=on_step if latest is not None else None)
            out.put(("episode", ep_i, ep, info))
    except KeyboardInterrupt:
        pass
    except Exception as err:
        out.put(("error", err))
    finally:
        env.disengage_safely()
        out.put(("done",))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path,
                   help="run folder to start from (sim training run or a "
                        "previous fine-tune run). config.json, the "
                        "checkpoint and the replay buffers are found inside.")
    p.add_argument("--inference", choices=("cpu", "pl"), default="cpu",
                   help="where the policy runs: 'cpu' = PS control loop over the "
                        "low-level bitstream (FPGA_env), 'pl' = MLP on the PL "
                        "(deployment bitstream, FPGA_deploy_env)")
    p.add_argument("--weight-sync", choices=("continuous", "episode"),
                   default="continuous",
                   help="'continuous' = weights also updated during the episode "
                        "(cpu: snapshot after every train() call; pl: upload to the "
                        "double-buffered weights every --pl-upload-interval-s). "
                        "'episode' = policy fixed per episode")
    p.add_argument("--pl-upload-interval-s", type=float, default=0.2,
                   help="[pl] min seconds between mid-episode weight uploads "
                        "(continuous sync). Each export costs learner time.")
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
    # --- cpu inference (low-level bitstream) ---
    p.add_argument("--port", default="/dev/cu.usbserial-1130",
                   help="[cpu] port for the low-level client")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--deterministic", action="store_true",
                   help="[cpu] deterministic actions during data collection. "
                        "Default is stochastic, matching SAC training-time "
                        "behaviour and what the policy was optimised for.")
    p.add_argument("--dt-jitter-frac", type=float, default=0.05,
                   help="[cpu] control-rate jitter (DR on dt). Each tick interval "
                        "is multiplied by uniform(1-frac, 1+frac). Default 0.05 "
                        "matches the sim DR_CONTROL_DT_JITTER_FRAC constant. "
                        "0.0 = strict reproducible timing.")
    p.add_argument("--eval-dt-jitter-frac", type=float, default=0.0,
                   help="[cpu] dt jitter during eval episodes. Default 0 (strict "
                        "timing, as deployed).")
    p.add_argument("--timing-violation-threshold-ms", type=float, default=10.0,
                   help="[cpu] control-loop overrun above this for 3 consecutive "
                        "ticks raises TimingViolation and disengages motor")
    # --- pl inference (deployment bitstream) ---
    p.add_argument("--param-noise", type=float, default=0.0,
                   help="[pl] std of Gaussian noise on the uploaded weights "
                        "(one draw per episode; exploration for the deterministic PL policy)")
    p.add_argument("--bitstream", default=None,
                   help="[pl] deployment .xsa/.bit (default: FpgaDeployClient's)")
    p.add_argument("--no-download", action="store_true",
                   help="[pl] overlay already loaded")
    # --- episodes / reset ---
    p.add_argument("--episodes", type=int, default=50,
                   help="number of real-device episodes to collect")
    p.add_argument("--episode-length-s", type=float, default=8.0)
    p.add_argument("--reset-settle-s", type=float, default=0.0,
                   help="max seconds to wait for a still pendulum between "
                        "episodes. 0 (default) = never wait: one --rest-check-s "
                        "check, tare only if still, start either way.")
    p.add_argument("--rest-check-s", type=float, default=0.5,
                   help="the pendulum must be still for this long (velocity and "
                        "angle thresholds) to be tared [s]. With ~half a swing period "
                        "the window always sees at least the swing amplitude, so a tare "
                        "is off by less than --rest-angle-range")
    p.add_argument("--rest-vel-threshold", type=float, default=None,
                   help="|pen_vel| below this counts as still [rad/s] (default 0.5)")
    p.add_argument("--rest-angle-range", type=float, default=None,
                   help="pendulum angle must stay within this band for 1 s to "
                        "count as still [rad] (default 0.03)")
    p.add_argument("--control-freq", type=float, default=None,
                   help="control rate in Hz. Default: control_freq_hz from "
                        "config.json (else 50). pl: must equal the bitstream's rate. "
                        "Overriding it changes what the policy's dynamics "
                        "model sees — only do it on purpose.")
    p.add_argument("--max-accel-rad-s2", type=float, default=None,
                   help="action ∈ [-1, 1] maps to angular accel ∈ [-max, +max]"
                        " rad/s². Default: max_accel_rad_s2 from config.json "
                        "(else 150). Must match sim training (pl: fixed in the bitstream).")
    # --- learner ---
    p.add_argument("--buffer-size", type=int, default=200_000,
                   help="capacity of EACH replay buffer (sim and real) on the "
                        "device. If a loaded buffer is larger, its most recent "
                        "transitions are kept.")
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
                        "the config's control rate, same as train_sac.py.")
    p.add_argument("--use-sde", action="store_true",
                   help="set if the policy was trained with --use-sde "
                        "(changes actor parameter shapes; not in config.json)")
    p.add_argument("--no-mirror", action="store_true",
                   help="use a plain ReplayBuffer even if the run trained "
                        "with mirror augmentation")
    p.add_argument("--gradient-steps", type=int, default=4,
                   help="SAC gradient updates per train() call (training runs continuously)")
    p.add_argument("--max-grad-per-transition", type=float, default=0.0,
                   help="cap on total gradient steps / real transitions collected "
                        "in this run (update-to-data ratio), so long rest waits "
                        "don't over-train on old data. 0 = no cap")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--learning-starts", type=int, default=100,
                   help="number of REAL transitions before training begins")
    p.add_argument("--checkpoint-freq", type=int, default=10,
                   help="save policy + real replay buffer every N episodes")
    p.add_argument("--run-name", default=None,
                   help="OUTPUT run dir name under runs/. Default: "
                        "async_<inference>_<source run>_<timestamp>.")
    p.add_argument("--device", default="cpu",
                   help="torch device. CPU is fine for this small policy")
    p.add_argument("--seed", type=int, default=0, help="seed for the parameter noise")
    # Reward weights are read from config.json so the rig optimises the same
    # objective as the sim run. These flags only exist to override on purpose.
    p.add_argument("--reward-stillness-bonus-weight", type=float, default=None,
                   help="override config.json's reward_stillness_bonus_weight")
    p.add_argument("--reward-alive-offset", type=float, default=None,
                   help="override config.json's reward_alive_offset")
    p.add_argument("--reward-upright-alive-weight", type=float, default=None,
                   help="override config.json's reward_upright_alive_weight")
    # --- metrics ---
    p.add_argument("--eval-every", type=int, default=5,
                   help="run a deterministic eval episode as the 1st episode "
                        "and then every N episodes (1, N+1, 2N+1, ...). Its "
                        "transitions still go into the buffer. 0 disables.")
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
    use_pl = args.inference == "pl"
    weight_sync = args.weight_sync
    if not use_pl and args.param_noise > 0:
        print("ERROR: --param-noise is for --inference pl; cpu explores with "
              "stochastic actions", file=sys.stderr)
        return 2
    if use_pl and (args.deterministic or args.dt_jitter_frac != 0.05):
        print("note: --deterministic / --dt-jitter-frac only apply to --inference cpu")

    torch.set_num_threads(1)
    sys.setswitchinterval(0.001)   # rig thread gets the GIL back quickly between torch ops

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
    print(f"Inference: {args.inference}, policy sync: {weight_sync}")
    if cfg_path is None:
        print("WARNING: no config.json in the run folder; using "
              "train_sac.py defaults (net_arch 256,256, 50 Hz, 150 rad/s², "
              "no mirror). Override with --control-freq / "
              "--max-accel-rad-s2 / --no-mirror if these are wrong.",
              file=sys.stderr)

    # ---- Output run folder ----
    run_name = (args.run_name or
                f"async_{args.inference}_{source.name}_{time.strftime('%Y-%m-%d_%H%M')}")
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

    # Kept key-for-key compatible with train_sac.py's config.json
    # (run_config.check_config compares them); provenance goes in source.json.
    cfg_out = dict(cfg)
    cfg_out["control_freq_hz"] = control_freq
    cfg_out["max_accel_rad_s2"] = max_accel

    # ---- Env (observation layout + reward weights from config.json) ----
    env_overrides = dict(
        control_freq_hz=control_freq,
        episode_length_s=args.episode_length_s,
        reset_settle_s=args.reset_settle_s,
    )
    if args.rest_vel_threshold is not None:
        env_overrides["rest_vel_threshold_rad_s"] = args.rest_vel_threshold
    if args.rest_angle_range is not None:
        env_overrides["rest_angle_range_rad"] = args.rest_angle_range
    env_overrides["rest_duration_s"] = args.rest_check_s
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

    if use_pl:
        from Deploy.FPGA.FPGA_deploy_env import FpgaDeployEnv
        env_overrides["download"] = not args.no_download
        if args.bitstream:
            env_overrides["bitstream_path"] = args.bitstream
        env = FpgaDeployEnv.from_config(cfg, **env_overrides)
        _ = env.client      # open the overlay here (checks the bitstream), not in the rig thread
    else:
        from Deploy.FPGA.FPGA_env import RealRotaryInvertedPendulumEnv
        env_overrides.update(port=args.port, baud=args.baud, max_accel_rad_s2=max_accel)
        env = RealRotaryInvertedPendulumEnv.from_config(cfg, **env_overrides)
    print(f"Control: {control_freq:g} Hz, max accel {max_accel:g} rad/s²")

    reward_overrides = {k: v for k, v in env_overrides.items() if k.startswith("reward_")}
    cfg_out.update(reward_overrides)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(run_dir / "source.json", "w") as f:
        json.dump({"source_run": str(source),
                   "checkpoint": policy_path.name,
                   "sim_buffer": sim_buffer_path.name if sim_buffer_path else None,
                   "real_buffer": real_buffer_path.name if real_buffer_path else None,
                   "inference": args.inference,
                   "weight_sync": weight_sync,
                   "pl_upload_interval_s": args.pl_upload_interval_s if use_pl else None,
                   "rest_check_s": args.rest_check_s,
                   "param_noise": args.param_noise if use_pl else None,
                   "deterministic": bool(args.deterministic) if not use_pl else True,
                   "dt_jitter_frac": args.dt_jitter_frac if not use_pl else None,
                   "gradient_steps": args.gradient_steps,
                   "max_grad_per_transition": args.max_grad_per_transition,
                   "learning_rate": args.learning_rate,
                   "real_frac_start": args.real_frac_start,
                   "real_frac_end": args.real_frac_end,
                   "real_frac_ramp_steps": args.real_frac_ramp_steps,
                   "episode_length_s": args.episode_length_s,
                   "reset_settle_s": args.reset_settle_s,
                   "argv": list(argv if argv is not None else sys.argv[1:]),
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    # ---- Model ----
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
            try:
                load_buffer(model.replay_buffer, rb_path)
            except (ValueError, IOError) as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 2
        else:
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

    if sim_buf.size() > 0:
        n = export_buffer(sim_buf, run_dir / "replay_buffer")
        print(f"  wrote sim buffer ({n} transitions) → replay_buffer/")

    # num_timesteps counts REAL transitions only (not the checkpoint's sim
    # step counter, not the sim buffer).
    model.num_timesteps = real_buf.size()
    model._total_timesteps = 0  # SB3 sets this in _setup_learn; placeholder.
    model._num_timesteps_at_start = model.num_timesteps

    # SB3's train() logs via self.logger, normally set up by learn().
    model._logger = configure_logger(
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        tb_log_name=f"async_{args.inference}",
        reset_num_timesteps=True,
    )

    # ---- Threads ----
    stop = threading.Event()
    sync = PolicySync()
    msgs: queue.Queue = queue.Queue()
    latest = None
    if use_pl:
        from Deploy.FPGA.weight_loader import actor_to_pl_words
        snapshot = tq = None
        latest = LatestWeights() if weight_sync == "continuous" else None
        rig = threading.Thread(target=_pl_rig_worker, name="rig", daemon=True,
                               args=(env, sync, latest, msgs, stop),
                               kwargs=dict(n_episodes=args.episodes))

        def export_words(ep: int, is_eval: bool):
            # Noise seeded per episode: every upload of one episode carries the
            # same draw, so mid-episode updates change only the trained part.
            noise = 0.0 if is_eval else args.param_noise
            rng = np.random.default_rng([args.seed, ep])
            return actor_to_pl_words(model.actor, param_noise_std=noise, rng=rng,
                                     warn=(ep == 0)), noise
    else:
        snapshot = PolicySnapshot(model.actor, device=args.device)
        tq = TransitionQueue(maxlen=4096)
        rig = threading.Thread(target=_cpu_rig_worker, name="rig", daemon=True,
                               args=(env, snapshot, tq, sync, msgs, stop),
                               kwargs=dict(
                                   n_episodes=args.episodes,
                                   control_freq=control_freq,
                                   timing_threshold_s=args.timing_violation_threshold_ms / 1000.0,
                                   deterministic=args.deterministic,
                                   dt_jitter=args.dt_jitter_frac,
                                   eval_dt_jitter=args.eval_dt_jitter_frac))

    def _on_signal(*_):
        if not stop.is_set():
            print("\nSIGINT/SIGTERM received — shutting down cleanly.")
        stop.set()
        env._on_signal()          # disengage + make the rig thread exit
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ---- Telemetry ----
    timing_f = open(run_dir / "timing.csv", "w", newline="")
    timing_writer = csv.writer(timing_f)
    timing_writer.writerow([
        "episode", "n_steps", "cumulative_reward", "terminated", "truncated",
        "mean_dt_ms", "std_dt_ms", "max_overrun_ms", "n_violations",
        "learner_train_calls", "learner_total_train_s",
        "grad_steps", "grad_steps_total", "max_dt_ms", "n_gaps", "n_brake",
        "rest_wait_s", "tared", "weights_grad_step", "upload_s",
        "n_weight_updates", "weights_grad_step_end",
    ])
    metrics_f = open(run_dir / "metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow([
        "episode", "mode", "real_steps", "rig_minutes", "return", "n_steps",
        "terminated", "real_frac", *METRIC_KEYS,
        "grad_steps", "grad_steps_total", "grad_per_transition",
    ])
    timing_f.flush()
    metrics_f.flush()

    eval_history: list[dict] = []
    best_eval_return = -math.inf
    recent_rewards: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)
    n_real_run = 0              # real transitions collected in THIS run
    n_grad_total = 0
    n_episodes_done = 0
    calls_ep, grad_ep, train_s_ep = 0, 0, 0.0
    cur_transitions: list = []  # cpu: transitions of the running episode
    frozen_for_eval = False     # continuous sync: no updates during an eval
    episode_running = False     # between the start-of-episode sync and its report
    running_ep = -1
    updates_ep = 0              # cpu: mid-episode snapshot refreshes
    last_update_grad = -1       # cpu: grad step of the last mid-episode refresh
    last_export_t = 0.0         # pl: time of the last mid-episode export
    rig_error = None

    def _report(ep_i: int, mode: str, *, ret: float, n_steps: int, terminated: bool,
                truncated: bool, m: dict, timing: dict, info: dict, reset: dict) -> None:
        """Print + CSV + TensorBoard for one finished episode (both modes)."""
        nonlocal best_eval_return, n_episodes_done
        n_episodes_done += 1
        is_eval = mode == "eval"
        rig_minutes = (time.monotonic() - rig_t0) / 60.0
        real_frac_now = mixed.last_real_frac if calls_ep > 0 else float("nan")
        utd = n_grad_total / n_real_run if n_real_run else float("nan")
        nan = float("nan")

        print(f"\n[ep {ep_i + 1}/{args.episodes}] ({mode}, {args.inference})  "
              f"steps={n_steps}  reward={ret:+.1f}  "
              f"mean_dt={_fmt(timing.get('mean_dt_ms', nan))}ms  "
              f"rest={_fmt(reset.get('rest_wait_s', nan), '.1f')}s"
              f"{'' if reset.get('tared', True) else ' (NO TARE)'}")
        print(f"  learner: grad_steps={grad_ep} (total {n_grad_total}, "
              f"{_fmt(utd)} per real transition)  train_calls={calls_ep}  "
              f"train_s={train_s_ep:.2f}  weights@{info.get('weights_grad_step', '-')}"
              f"→{info.get('weights_grad_step_end', '-')} "
              f"({info.get('n_weight_updates', 0)} mid-episode updates)")
        print(f"  buffers: real={real_buf.size()}  sim={sim_buf.size()}  "
              f"real share of batch={_fmt(real_frac_now)}")
        print(f"  [{mode}] balanced={'yes' if m['balanced'] else 'no'}  "
              f"t_bal={_fmt(m['time_to_balance_s'])}s  "
              f"upright={_fmt(m['frac_upright'])}  "
              f"θ_rms={_fmt(m['theta_rms_deg'])}°  "
              f"|a|={_fmt(m['mean_abs_action_balanced'])}  "
              f"dropped={'yes' if m['dropped'] else 'no'}")

        def cell(x, spec=".3f"):
            return "" if x is None else _fmt(float(x), spec)
        timing_writer.writerow([
            ep_i + 1, n_steps, f"{ret:.3f}", int(terminated), int(truncated),
            cell(timing.get("mean_dt_ms")), cell(timing.get("std_dt_ms")),
            cell(timing.get("max_overrun_ms")),
            "" if timing.get("n_violations") is None else timing["n_violations"],
            calls_ep, f"{train_s_ep:.3f}", grad_ep, n_grad_total,
            cell(timing.get("max_dt_ms")),
            "" if timing.get("n_gaps") is None else timing["n_gaps"],
            "" if timing.get("n_brake") is None else timing["n_brake"],
            cell(reset.get("rest_wait_s"), ".2f"),
            "" if reset.get("tared") is None else int(reset["tared"]),
            info.get("weights_grad_step", ""), cell(info.get("upload_s"), ".4f"),
            info.get("n_weight_updates", 0), info.get("weights_grad_step_end", ""),
        ])
        timing_f.flush()
        metrics_writer.writerow([
            ep_i + 1, mode, model.num_timesteps, f"{rig_minutes:.2f}",
            f"{ret:.3f}", n_steps, int(terminated), _fmt(real_frac_now, ".3f"),
            *[m[k] for k in METRIC_KEYS],
            grad_ep, n_grad_total, _fmt(utd, ".3f"),
        ])
        metrics_f.flush()

        lg = model.logger
        prefix = "eval" if is_eval else "rollout"
        lg.record(f"{prefix}/balanced", m["balanced"])
        lg.record(f"{prefix}/frac_upright",
                  0.0 if math.isnan(m["frac_upright"]) else m["frac_upright"])
        lg.record(f"{prefix}/dropped", m["dropped"])
        if m["balanced"]:
            for k in ("time_to_balance_s", "frac_upright_after_catch",
                      "theta_rms_deg", "motor_pos_std_deg",
                      "mean_abs_action_balanced"):
                lg.record(f"{prefix}/{k}", m[k])
        lg.record("time/rig_minutes", rig_minutes)
        lg.record("replay/real_size", real_buf.size())
        lg.record("replay/sim_size", sim_buf.size())
        if calls_ep > 0:
            lg.record("replay/real_frac", real_frac_now)
        lg.record("train/grad_steps_total", n_grad_total)
        lg.record("time/grad_steps_episode", grad_ep)
        if n_real_run:
            lg.record("train/grad_per_transition", utd)

        if is_eval:
            entry = dict(episode=ep_i + 1, real_steps=int(model.num_timesteps),
                         rig_minutes=rig_minutes, **{"return": float(ret)},
                         n_steps=int(n_steps), grad_steps_total=n_grad_total, **m)
            eval_history.append(entry)
            lg.record("eval/ep_reward", ret)
            lg.record("eval/ep_length", n_steps)
            lg.record("eval/n_evals", len(eval_history))
            cand = run_dir / "eval_candidate.zip"
            if ret > best_eval_return and cand.exists():
                best_eval_return = ret
                cand.replace(run_dir / "best_eval.zip")
                print(f"  new best deterministic eval ({best_eval_return:+.1f}) "
                      f"→ best_eval.zip")
        else:
            recent_rewards.append(ret)
            recent_lengths.append(n_steps)
            lg.record("rollout/ep_rew_mean", sum(recent_rewards) / len(recent_rewards))
            lg.record("rollout/ep_len_mean", sum(recent_lengths) / len(recent_lengths))
            lg.record("rollout/ep_reward", ret)
            lg.record("rollout/ep_length", n_steps)
        lg.record("rollout/episode", ep_i + 1)
        for key, tb in (("mean_dt_ms", "time/mean_tick_dt_ms"),
                        ("std_dt_ms", "time/std_tick_dt_ms"),
                        ("max_overrun_ms", "time/max_overrun_ms"),
                        ("n_violations", "time/n_timing_violations"),
                        ("max_dt_ms", "time/max_tick_dt_ms"),
                        ("n_gaps", "time/n_gaps"),
                        ("n_brake", "time/n_brake")):
            v = timing.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                lg.record(tb, v)
        if reset.get("rest_wait_s") is not None:
            lg.record("rig/rest_wait_s", reset["rest_wait_s"])
        if reset.get("tared") is not None:
            lg.record("rig/tared", int(reset["tared"]))
        lg.record("time/learner_train_calls", calls_ep)
        lg.record("time/learner_total_train_s", train_s_ep)
        lg.record("time/n_weight_updates", info.get("n_weight_updates", 0))
        lg.dump(step=model.num_timesteps)

        if (ep_i + 1) % args.checkpoint_freq == 0:
            _save_artifacts(model, real_buf, run_dir, suffix=f"ep{ep_i + 1:03d}")

    def _finish_episode(ep_i: int, payload, info: dict) -> None:
        """Buffer the episode (pl) or its stragglers (cpu), then report it."""
        nonlocal n_real_run, cur_transitions, frozen_for_eval, calls_ep, grad_ep, train_s_ep
        nonlocal episode_running
        episode_running = False
        frozen_for_eval = False
        if not use_pl:
            info = dict(info, n_weight_updates=updates_ep,
                        weights_grad_step_end=(last_update_grad if updates_ep
                                               else info.get("weights_grad_step", -1)))
        synced = bool(info.get("synced"))
        mode = "eval" if synced and info.get("is_eval") else "train"
        if use_pl:
            ep = payload
            n_real_run += _add_to_buffer(model, _pl_episode_transitions(ep))
            m = _pl_episode_metrics(ep, control_freq, args.balance_theta_rad,
                                    args.balance_hold_s)
            dts = 1e3 * ep["dt_s"]
            timing = dict(mean_dt_ms=float(dts.mean()) if dts.size else float("nan"),
                          std_dt_ms=float(dts.std()) if dts.size else float("nan"),
                          max_dt_ms=float(dts.max()) if dts.size else float("nan"),
                          n_gaps=int(ep["gap"].sum()), n_brake=int(ep["brake"].sum()))
            reset = dict(rest_wait_s=ep["rest_wait_s"], tared=ep["tared"])
            _report(ep_i, mode, ret=ep["return"], n_steps=ep["length"],
                    terminated=bool(ep["terminated"].any()), truncated=ep["truncated"],
                    m=m, timing=timing, info=info, reset=reset)
        else:
            stragglers = tq.drain()
            n_real_run += _add_to_buffer(model, stragglers)
            cur_transitions.extend(stragglers)
            transitions, cur_transitions = cur_transitions, []
            stats = payload["stats"]
            if weight_sync == "continuous":
                snapshot.refresh_from(model.actor)   # hand the eval-time training on
            reset = dict(rest_wait_s=payload.get("rest_wait_s"), tared=payload.get("tared"))
            if stats is None:
                print(f"  WARNING: episode {ep_i + 1} aborted ({payload.get('error')}); "
                      f"its {len(transitions)} transitions are kept, no metrics.",
                      file=sys.stderr)
            else:
                m = _cpu_episode_metrics(transitions, control_freq,
                                         args.balance_theta_rad, args.balance_hold_s)
                if m["n_valid"] == 0:
                    print("  WARNING: no transition carried info['phi']; balance "
                          "metrics unavailable for this episode.", file=sys.stderr)
                ts = stats.tick_stats
                timing = dict(mean_dt_ms=stats.mean_dt_ms, std_dt_ms=stats.std_dt_ms,
                              max_overrun_ms=ts.max_overrun_ms, n_violations=ts.n_violations)
                _report(ep_i, mode, ret=stats.cumulative_reward, n_steps=stats.n_steps,
                        terminated=stats.terminated, truncated=stats.truncated,
                        m=m, timing=timing, info=info, reset=reset)
        calls_ep, grad_ep, train_s_ep = 0, 0, 0.0

    print(f"\nStarting fine-tune ({args.inference} inference): {args.episodes} episodes × "
          f"{args.episode_length_s} s @ {control_freq:g} Hz, "
          f"K={args.gradient_steps}, lr={args.learning_rate}, sync={weight_sync}"
          + (f", param noise {args.param_noise:g}" if use_pl else ""))
    print(f"Real transitions so far: {real_buf.size()}, "
          f"learning_starts={args.learning_starts}")
    if args.eval_every > 0:
        print(f"Deterministic eval every {args.eval_every} episodes "
              f"(first = sim-only baseline); balance = |θ| < "
              f"{args.balance_theta_rad:g} rad for {args.balance_hold_s:g} s")

    # ---- Learner loop ----
    rig_t0 = time.monotonic()
    rig.start()
    try:
        while True:
            # 1. Policy sync for the episode about to start.
            if sync.pending():
                ep_next = sync.requested_episode
                is_eval = args.eval_every > 0 and ep_next % args.eval_every == 0
                if is_eval:
                    # Exactly the weights this eval scores (training moves on);
                    # promoted to best_eval.zip if it wins.
                    model.save(run_dir / "eval_candidate.zip")
                meta = {"weights_grad_step": n_grad_total, "is_eval": is_eval}
                if use_pl:
                    payload, noise = export_words(ep_next, is_eval)
                    meta["param_noise"] = noise
                    if latest is not None:
                        meta["version"] = latest.put(ep_next, payload, n_grad_total)
                    last_export_t = time.monotonic()
                else:
                    snapshot.refresh_from(model.actor)
                    payload = True
                frozen_for_eval = is_eval
                episode_running, running_ep = True, ep_next
                updates_ep, last_update_grad = 0, -1
                sync.provide(payload, meta)

            # 2. Streaming transitions (cpu) -> replay buffer.
            if tq is not None:
                drained = tq.drain()
                if drained:
                    n_real_run += _add_to_buffer(model, drained)
                    cur_transitions.extend(drained)

            # 3. Finished episodes.
            done = False
            while True:
                try:
                    msg = msgs.get_nowait()
                except queue.Empty:
                    break
                if msg[0] == "done":
                    done = True
                elif msg[0] == "error":
                    rig_error = msg[1]
                else:
                    _finish_episode(*msg[1:])
            if done or stop.is_set():
                break

            # 4. Train a chunk. Gate on REAL transitions; the sim buffer alone
            #    never starts training.
            can_train = (real_buf.size() >= args.learning_starts
                         and mixed.size() >= model.batch_size)
            if can_train and args.max_grad_per_transition > 0:
                can_train = n_grad_total < args.max_grad_per_transition * n_real_run
            if can_train:
                t0 = time.monotonic()
                model.train(gradient_steps=args.gradient_steps,
                            batch_size=model.batch_size)
                train_s_ep += time.monotonic() - t0
                n_grad_total += args.gradient_steps
                grad_ep += args.gradient_steps
                calls_ep += 1
                if weight_sync == "continuous" and not frozen_for_eval:
                    if snapshot is not None:
                        snapshot.refresh_from(model.actor)
                        if episode_running:
                            updates_ep += 1
                            last_update_grad = n_grad_total
                    elif (episode_running and time.monotonic() - last_export_t
                          >= args.pl_upload_interval_s):
                        words, _ = export_words(running_ep, False)
                        latest.put(running_ep, words, n_grad_total)
                        last_export_t = time.monotonic()
            else:
                time.sleep(0.001)
    finally:
        stop.set()
        env._on_signal()
        rig.join(timeout=10.0)
        if rig.is_alive():
            print("  [warn] rig thread did not exit within 10 s", file=sys.stderr)
        if tq is not None:                       # transitions of an interrupted episode
            n_real_run += _add_to_buffer(model, tq.drain())
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
        minutes = (time.monotonic() - rig_t0) / 60.0
        utd = n_grad_total / n_real_run if n_real_run else float("nan")
        totals = {"inference": args.inference, "weight_sync": weight_sync,
                  "episodes": n_episodes_done, "real_transitions": n_real_run,
                  "grad_steps_total": n_grad_total, "grad_per_transition": utd,
                  "rig_minutes": minutes}
        with open(run_dir / "run_summary.json", "w") as f:
            json.dump(totals, f, indent=2, default=float)
        print(f"\n[{args.inference}] {n_episodes_done} episodes, {n_real_run} real transitions, "
              f"{n_grad_total} gradient steps ({_fmt(utd)} per transition), "
              f"{minutes:.1f} rig minutes")
        print(f"Run artifacts in: {run_dir}")
    if rig_error is not None:
        raise rig_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

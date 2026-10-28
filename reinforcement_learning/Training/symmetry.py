"""Single source of truth for the rig's mirror symmetry.

The Furuta pendulum has exactly one non-trivial symmetry: reflect the whole
machine through the vertical plane that contains the arm at motor_pos = 0.
The group is Z2 = {identity, mirror}. Under the mirror,

    motor_pos -> -motor_pos      theta   -> -theta
    motor_vel -> -motor_vel      pen_vel -> -pen_vel
    action    -> -action

and since the observation frame stores the pendulum angle as (sin θ, cos θ),
`sin` flips sign while `cos` does NOT. So on one observation frame the map is
a fixed diagonal of ±1, tiled over the stacked frames:

    [motor_pos, sin θ, cos θ, motor_vel, pen_vel, prev_action]
    [   -1        -1     +1       -1        -1        -1     ]

This is a symmetry of the *plant* (mirror-image geometry, same physics), of
the *reward* (every term in `reward.py` is even in these quantities, and the
alive gates use |θ| / |θ̇|), and of the *reset distribution* (motor_pos and
the θ-bias are sampled symmetrically; the base-tilt azimuth is uniform). So
for any transition (s, a, r, s') the mirrored transition (Ms, -a, r, Ms') is
an equally valid transition of the SAME MDP — not an approximation.

An ideal policy is therefore *equivariant*, π(Ms) = -π(s), with an
*invariant* critic, Q(Ms, -a) = Q(s, a). Nothing in SAC enforces this, so a
deterministic policy generically breaks the symmetry and picks an arbitrary
preferred swing-up direction — differently on every training run. That is
measurable: see `asymmetry()` below and
`website/src/content/docs/reference/symmetry.md`.

CAUTION — the mirror fixed point. States with motor_pos = 0, θ = ±π (or 0),
zero velocities and zero prev_action are their OWN mirror image, so an
*exactly* equivariant policy must output 0 there. That is precisely the
state the rig starts from (both `RLControl.ino`'s `prime_initial_state()`
and `real_env.reset()` tare the arm and the encoder to zero), so an exactly
odd policy would sit there forever. Soft symmetrisation (mirror data
augmentation, symmetrised distillation targets) leaves a small residual
asymmetry that breaks the tie and does not have this problem; an exactly
equivariant architecture would need a deliberate tie-break (e.g. engage
from a non-centred arm). `engage_obs()` builds that state so any policy can
be checked against it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


# Per-frame sign flips. Index 2 is cos θ in both layouts, so the "velocities
# included" variant is the no-velocity one with two more flipped channels.
FRAME_SIGNS_NO_VEL = (-1.0, -1.0, +1.0, -1.0)          # motor_pos, sin, cos, prev_action
FRAME_SIGNS_WITH_VEL = (-1.0, -1.0, +1.0, -1.0, -1.0, -1.0)

FRAME_DIM_NO_VEL = len(FRAME_SIGNS_NO_VEL)
FRAME_DIM_WITH_VEL = len(FRAME_SIGNS_WITH_VEL)


def frame_signs(obs_include_velocities: bool = True) -> np.ndarray:
    """The ±1 sign vector for ONE observation frame."""
    return np.array(
        FRAME_SIGNS_WITH_VEL if obs_include_velocities else FRAME_SIGNS_NO_VEL,
        dtype=np.float32,
    )


def obs_signs(*, obs_history_len: int, obs_include_velocities: bool = True) -> np.ndarray:
    """The ±1 sign vector for a full stacked observation."""
    if int(obs_history_len) < 1:
        raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
    return np.tile(frame_signs(obs_include_velocities), int(obs_history_len))


def obs_signs_for_dim(obs_dim: int, *, obs_history_len: int) -> np.ndarray:
    """Sign vector for `obs_dim`, inferring the frame layout from K.

    Safer than trusting a config flag: the frame dimension is `obs_dim // K`,
    which pins down whether velocities are in the frame. Raises if the two
    are inconsistent, which is exactly the silent-garbage case (mirroring an
    observation with the wrong layout flips `cos θ` and leaves `sin θ` alone,
    producing a "mirrored" state that is not on the manifold at all).
    """
    obs_dim = int(obs_dim)
    k = int(obs_history_len)
    if k < 1 or obs_dim % k != 0:
        raise ValueError(
            f"obs_dim={obs_dim} is not divisible by obs_history_len={k}"
        )
    frame = obs_dim // k
    if frame == FRAME_DIM_WITH_VEL:
        return obs_signs(obs_history_len=k, obs_include_velocities=True)
    if frame == FRAME_DIM_NO_VEL:
        return obs_signs(obs_history_len=k, obs_include_velocities=False)
    raise ValueError(
        f"obs_dim={obs_dim} / K={k} gives frame_dim={frame}, expected "
        f"{FRAME_DIM_WITH_VEL} (with velocities) or {FRAME_DIM_NO_VEL} (without)"
    )


def obs_signs_from_config(obs_dim: int, config: dict | None) -> np.ndarray:
    """Sign vector from a run's `config.json` (see `run_config.py`)."""
    cfg = config or {}
    k = int(cfg.get("obs_history_len") or 1)
    return obs_signs_for_dim(obs_dim, obs_history_len=k)


def mirror_obs(obs, signs: np.ndarray) -> np.ndarray:
    """Ms — mirror one observation or a batch of them."""
    return np.asarray(obs, dtype=np.float32) * signs


def mirror_action(action) -> np.ndarray:
    """-a — the action is a scalar torque-like command, so it simply flips."""
    return -np.asarray(action, dtype=np.float32)


def engage_obs(*, obs_history_len: int, obs_include_velocities: bool = True,
               theta: float = np.pi) -> np.ndarray:
    """The rig's post-tare engage state: arm centred, pendulum hanging, at rest.

    This is a fixed point of the mirror map, so an exactly equivariant policy
    outputs 0 here (see the module docstring). Any nonzero action a policy
    returns for this observation IS its hard-coded swing-up direction.
    """
    frame = [0.0, float(np.sin(theta)), float(np.cos(theta))]
    if obs_include_velocities:
        frame += [0.0, 0.0]
    frame += [0.0]  # prev_action
    return np.tile(np.array(frame, dtype=np.float32), int(obs_history_len))


@dataclass(frozen=True)
class AsymmetryReport:
    """How far a policy is from π(Ms) = -π(s) over a state distribution."""

    n: int
    rms_error: float      # RMS of π(s) + π(Ms) — 0 iff exactly equivariant
    rms_action: float     # RMS of π(s), for scale
    max_error: float
    mean_action: float    # 0 for an equivariant policy on a symmetric state set
    engage_action: float | None = None  # π at the mirror fixed point

    @property
    def relative_error(self) -> float:
        """`rms_error` as a fraction of RMS action.

        0 = exactly equivariant. ~1.41 = π(Ms) statistically unrelated to
        -π(s) (two independent draws of the same magnitude), so anything
        near or above 1 means the policy carries essentially no mirror
        structure.
        """
        return float(self.rms_error / self.rms_action) if self.rms_action > 0 else float("nan")

    def report_lines(self) -> list[str]:
        lines = [
            f"  states scored:           {self.n}",
            f"  RMS |pi(s) + pi(Ms)|:    {self.rms_error:.4f}   (0 = equivariant)",
            f"  RMS |pi(s)|:             {self.rms_action:.4f}",
            f"  relative asymmetry:      {self.relative_error:.3f}   "
            f"(0 = equivariant, ~1.41 = unrelated)",
            f"  max |pi(s) + pi(Ms)|:    {self.max_error:.4f}",
            f"  mean pi(s):              {self.mean_action:+.4f}   (0 = unbiased)",
        ]
        if self.engage_action is not None:
            lines.append(
                f"  action at engage state:  {self.engage_action:+.4f}   "
                f"(0 = no baked-in swing direction)")
        return lines


def _as_flat_action(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim > 1 else arr.reshape(-1)


def asymmetry(predict, obs, signs: np.ndarray, *,
              engage: np.ndarray | None = None) -> AsymmetryReport:
    """Score a policy's mirror-equivariance over the observations `obs`.

    `predict` takes a batch of observations and returns actions — pass
    `lambda x: model.predict(x, deterministic=True)[0]` for an SB3 model.
    Scoring on states the policy actually visits (a rollout, a replay
    buffer, a distillation dataset) is the meaningful measure; a uniform
    random state box also works as a coarse global check.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"obs must be (n, obs_dim), got shape {obs.shape}")
    a = _as_flat_action(predict(obs))
    a_mirror = _as_flat_action(predict(mirror_obs(obs, signs)))
    err = a + a_mirror
    engage_action = None
    if engage is not None:
        engage_action = float(_as_flat_action(predict(engage.reshape(1, -1)))[0])
    return AsymmetryReport(
        n=int(obs.shape[0]),
        rms_error=float(np.sqrt((err ** 2).mean())),
        rms_action=float(np.sqrt((a ** 2).mean())),
        max_error=float(np.abs(err).max()),
        mean_action=float(a.mean()),
        engage_action=engage_action,
    )


class MirrorReplayBuffer(ReplayBuffer):
    """Replay buffer that stores every transition AND its mirror image.

    The DUP method of Abdolhosseini et al., "On Learning Symmetric
    Locomotion" (MIG 2019). For off-policy SAC this is exact rather than
    approximate: (Ms, -a, r, Ms') is a genuine transition of the same MDP,
    with the same reward, so nothing is being faked — the buffer simply
    covers both halves of the state space instead of whichever half the
    policy happens to prefer.

    Each `add()` therefore consumes two slots, so a given `buffer_size`
    holds half as much wall-clock experience; callers that care (SAC's
    `buffer_size`) should double it. `optimize_memory_usage` is rejected
    because it stores next_obs in the following slot, which the interleaved
    writes would corrupt.
    """

    def __init__(self, *args, obs_sign: np.ndarray | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError(
                "MirrorReplayBuffer does not support optimize_memory_usage "
                "(next_obs shares the following slot, which interleaved "
                "mirror writes would corrupt)"
            )
        if obs_sign is None:
            raise ValueError(
                "MirrorReplayBuffer needs obs_sign — build it with "
                "symmetry.obs_signs_for_dim(obs_dim, obs_history_len=K)"
            )
        sign = np.asarray(obs_sign, dtype=np.float32)
        expected = int(np.prod(self.obs_shape))
        if sign.shape != (expected,):
            raise ValueError(
                f"obs_sign has shape {sign.shape}, expected ({expected},) "
                f"to match the buffer's observation space"
            )
        self.obs_sign = sign

    def add(self, obs, next_obs, action, reward, done, infos):  # noqa: D102
        super().add(obs, next_obs, action, reward, done, infos)
        super().add(
            mirror_obs(obs, self.obs_sign),
            mirror_obs(next_obs, self.obs_sign),
            mirror_action(action),
            reward,
            done,
            infos,
        )


def add_transition_with_mirror(buffer, *, obs, next_obs, action, reward, done,
                               infos, obs_sign: np.ndarray | None) -> int:
    """`buffer.add(...)` plus its mirror when `obs_sign` is given.

    The explicit counterpart to `MirrorReplayBuffer` for call sites that
    already loop over transitions by hand (`finetune_async._add_to_buffer`),
    where swapping the buffer class would fight `--resume-buffer` (which
    unpickles a plain `ReplayBuffer` and replaces the attribute).
    Returns the number of transitions written.
    """
    buffer.add(obs=obs, next_obs=next_obs, action=action, reward=reward,
               done=done, infos=infos)
    if obs_sign is None:
        return 1
    buffer.add(
        obs=mirror_obs(obs, obs_sign),
        next_obs=mirror_obs(next_obs, obs_sign),
        action=mirror_action(action),
        reward=reward,
        done=done,
        infos=infos,
    )
    return 2

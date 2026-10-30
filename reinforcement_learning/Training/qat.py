"""Quantization building blocks for the rotary-pendulum SAC pipeline.

Three independent pieces:

1. Observation quantization (NumPy, env side)
   `FixedPointQuantizer` + `QuantizedObsWrapper` snap every observation
   dimension onto a signed fixed-point grid Q(int_bits).(frac_bits) with
   saturation. It is completely independent of `NORMILIZED_OBS` in
   pendulum_env.py: you only tell it each dimension's |max| (`abs_max`).
       * normalized obs -> abs_max = 1 everywhere -> int_bits = 0, so a
         word of N bits is 1 sign + (N-1) fractional bits.
       * raw obs        -> per-dimension abs_max -> int_bits > 0 where the
         physical range needs it (e.g. pen_vel +-32 rad/s -> 5 int bits).

2. Weight / activation fake quantization (PyTorch, policy side)
   `QuantLinear`, `QActivation`, `ActQuant`: straight-through-estimator
   fake quantization. Weights: symmetric, per-tensor (or per-channel),
   optionally power-of-two scales (=> shifts on the FPGA). Activations:
   EMA-tracked max-abs observer, signed/unsigned, power-of-two optional.

3. `QATSACPolicy`: SB3 SACPolicy whose ACTOR is quantization-aware. Critics
   stay float (they never get deployed). Parameter names match a stock SB3
   actor (latent_pi.0.weight, mu.weight, ...), so a float checkpoint can be
   loaded with strict=False and fine-tuned under quantization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.sac.policies import Actor, SACPolicy


# =============================================================================
# 1. Observation quantization
# =============================================================================

# Raw (physical) |max| per feature, used as the starting point before the env's
# obs_norm preset is applied. Names match pendulum_env._obs.
RAW_FEATURE_ABS_MAX = {
    "motor_pos": math.radians(145.0),
    "sin": 1.0,
    "cos": 1.0,
    "motor_vel": 6.0,
    "pen_vel": 32.0,
    "prev_act": 1.0,
}


def _feature_abs_max(raw_max: float, spec, gain: float) -> float:
    """|max| of one feature after the env's normalization spec and global gain
    (mirrors pendulum_env._apply_norm)."""
    if spec is None:
        m = raw_max
    else:
        kind = spec[0]
        if kind == "div":
            m = raw_max / spec[1]
        elif kind == "clipdiv":
            m = min(1.0, raw_max / spec[1])
        elif kind == "gain":
            m = raw_max * spec[1]
        elif kind == "asinh":
            m = 1.0  # output is clipped to [-1, 1]
        else:
            raise ValueError(f"unknown norm kind {kind}")
    return m * gain


def env_frame_layout(env: gym.Env) -> tuple[list[str], int]:
    """(feature names of ONE frame, history length), read from the env."""
    u = env.unwrapped
    names = ["motor_pos", "sin", "cos"]
    if u.obs_include_velocities:
        names += ["motor_vel", "pen_vel"]
    names.append("prev_act")
    return names, int(u.obs_history_len)


def default_obs_abs_max(env: gym.Env) -> np.ndarray:
    """Per-dimension |max| for the FULL observation vector (all stacked
    frames, oldest -> newest), derived from the env's obs_norm preset,
    obs_include_velocities and obs_history_len."""
    from reinforcement_learning.Simulation.pendulum_env import OBS_NORM_PRESETS
    u = env.unwrapped
    preset = OBS_NORM_PRESETS[u.obs_norm_name]
    feats, gain = preset.get("feats", {}), preset.get("gain", 1.0)
    names, hist = env_frame_layout(env)
    frame = [_feature_abs_max(RAW_FEATURE_ABS_MAX[n], feats.get(n), gain) for n in names]
    return np.tile(np.asarray(frame, dtype=np.float64), hist)


def _expand_per_dim(v, frame_len: int, total_len: int) -> np.ndarray:
    """Accept a scalar, a per-frame vector (tiled over history) or a full vector."""
    v = np.asarray(v, dtype=np.int64)
    if v.ndim == 0:
        return np.full(total_len, int(v), dtype=np.int64)
    if v.shape[0] == frame_len:
        return np.tile(v, total_len // frame_len)
    if v.shape[0] == total_len:
        return v
    raise ValueError(f"length {v.shape[0]} matches neither frame ({frame_len}) nor full obs ({total_len})")


def make_obs_quantizer(env: gym.Env, *, word_bits=None, frac_bits=None) -> "FixedPointQuantizer":
    """Build a quantizer that matches the env's observation layout.
    Give exactly one of word_bits (sign+int+frac) or frac_bits; each may be a
    scalar, one value per frame feature, or one per full-obs dimension."""
    if (word_bits is None) == (frac_bits is None):
        raise ValueError("pass exactly one of word_bits / frac_bits")
    abs_max = default_obs_abs_max(env)
    names, hist = env_frame_layout(env)
    n_total = abs_max.shape[0]
    if word_bits is not None:
        return FixedPointQuantizer.from_word_bits(
            abs_max, _expand_per_dim(word_bits, len(names), n_total))
    return FixedPointQuantizer(abs_max, _expand_per_dim(frac_bits, len(names), n_total))


def _int_bits(abs_max: float) -> int:
    """Integer bits (excluding sign) needed so that +-abs_max fits."""
    return max(0, int(math.ceil(math.log2(abs_max)))) if abs_max > 0 else 0


class FixedPointQuantizer:
    """Per-dimension signed fixed point, round-half-up, saturating.

    Representable range per dim: [-2^int_bits, 2^int_bits - 2^-frac_bits].
    (So +1.0 itself saturates to 1 - LSB when int_bits = 0.)
    """

    def __init__(self, abs_max, frac_bits):
        abs_max = np.asarray(abs_max, dtype=np.float64)
        frac = np.broadcast_to(np.asarray(frac_bits, dtype=np.int64), abs_max.shape).copy()
        self.abs_max = abs_max
        self.int_bits = np.array([_int_bits(a) for a in abs_max], dtype=np.int64)
        self.frac_bits = frac
        self.step = 2.0 ** (-frac.astype(np.float64))
        self.lo = -(2.0 ** self.int_bits)
        self.hi = 2.0 ** self.int_bits - self.step

    @classmethod
    def from_word_bits(cls, abs_max, word_bits) -> "FixedPointQuantizer":
        """Total word length per dim (sign + int + frac) is fixed; fractional
        bits are whatever is left after the integer part."""
        abs_max = np.asarray(abs_max, dtype=np.float64)
        w = np.broadcast_to(np.asarray(word_bits, dtype=np.int64), abs_max.shape)
        ib = np.array([_int_bits(a) for a in abs_max], dtype=np.int64)
        frac = w - 1 - ib
        if np.any(frac < 0):
            raise ValueError(f"word_bits {w.tolist()} too small for int_bits {ib.tolist()}")
        return cls(abs_max, frac)

    @property
    def word_bits(self) -> np.ndarray:
        return 1 + self.int_bits + self.frac_bits

    def describe(self) -> list[str]:
        return [f"Q{i}.{f}" for i, f in zip(self.int_bits, self.frac_bits)]

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        q = np.floor(x / self.step + 0.5) * self.step
        return np.clip(q, self.lo, self.hi)


class QuantizedObsWrapper(gym.ObservationWrapper):
    """Applies a FixedPointQuantizer to the final observation vector."""

    def __init__(self, env: gym.Env, quantizer: FixedPointQuantizer):
        super().__init__(env)
        if env.observation_space.shape != quantizer.abs_max.shape:
            raise ValueError(
                f"obs shape {env.observation_space.shape} != quantizer {quantizer.abs_max.shape}"
            )
        self.quantizer = quantizer

    def observation(self, obs):
        return self.quantizer(obs).astype(np.float32)


# =============================================================================
# 2. Fake quantization (STE)
# =============================================================================

@dataclass
class QATConfig:
    weight_bits: Optional[int] = 8     # None -> weights stay float
    act_bits: Optional[int] = 8        # None -> activations stay float
    bias_bits: Optional[int] = 16      # None -> bias stays float
    pow2_scales: bool = True           # round scales up to 2^k (shift-only requant)
    per_channel: bool = False          # per-output-channel weight scales
    act_momentum: float = 0.99         # EMA momentum of the activation observer


def _scale_from_max(m: th.Tensor, qmax: int, pow2: bool) -> th.Tensor:
    scale = m.clamp_min(1e-8) / qmax
    if pow2:
        scale = th.exp2(th.ceil(th.log2(scale)))  # ceil => range still covers m
    return scale


def _ste_quant(x: th.Tensor, scale: th.Tensor, qmin: int, qmax: int) -> th.Tensor:
    """Round-to-grid with clipped straight-through gradient (zero outside range)."""
    lo, hi = qmin * scale, qmax * scale
    xc = th.minimum(th.maximum(x, lo), hi)
    q = th.round(xc / scale) * scale
    return xc + (q - xc).detach()


class ActQuant(nn.Module):
    """Activation fake-quantizer with an EMA max-abs observer.

    The observer updates whenever the module is in train() mode (even while
    quantization is switched off, so it is calibrated by the time QAT starts).
    """

    def __init__(self, bits: Optional[int], signed: bool, momentum: float, pow2: bool):
        super().__init__()
        self.bits, self.signed, self.momentum, self.pow2 = bits, signed, momentum, pow2
        self.quant_on = True
        self.register_buffer("running_max", th.zeros(()))
        self.register_buffer("initialized", th.zeros((), dtype=th.bool))

    def _qrange(self) -> tuple[int, int]:
        if self.signed:
            qmax = 2 ** (self.bits - 1) - 1
            return -qmax, qmax
        return 0, 2 ** self.bits - 1

    def current_scale(self) -> th.Tensor:
        # .clone(): never hand autograd a tensor that a later forward will mutate in place
        return _scale_from_max(self.running_max.detach().clone(), self._qrange()[1], self.pow2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            with th.no_grad():
                cur = x.detach().abs().max()
                new = th.where(
                    self.initialized,
                    self.momentum * self.running_max + (1.0 - self.momentum) * cur,
                    cur,
                )
                self.running_max.copy_(new)
                self.initialized.fill_(True)
        if not self.quant_on or self.bits is None:
            return x
        qmin, qmax = self._qrange()
        q = _ste_quant(x, self.current_scale(), qmin, qmax)
        # th.where saves its condition for backward. SB3's SAC runs the actor several
        # times per gradient step (obs, then next_obs under no_grad) BEFORE backward, and
        # each training-mode forward updates the observer buffers in place. Using the
        # buffer directly raises "modified by an inplace operation"; a clone is safe.
        ready = self.initialized.detach().clone()
        return th.where(ready, q, x)  # pass-through until calibrated


class QuantLinear(nn.Linear):
    """nn.Linear with fake-quantized weights/bias and an optional output
    quantizer (used on the mu head)."""

    def __init__(self, in_features, out_features, bias=True, cfg: Optional[QATConfig] = None,
                 out_quant: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.cfg = cfg or QATConfig()
        self.quant_on = True
        self.out_q = (
            ActQuant(self.cfg.act_bits, True, self.cfg.act_momentum, self.cfg.pow2_scales)
            if out_quant else None
        )

    @classmethod
    def from_linear(cls, lin: nn.Linear, cfg: QATConfig, out_quant: bool = False) -> "QuantLinear":
        new = cls(lin.in_features, lin.out_features, bias=lin.bias is not None,
                  cfg=cfg, out_quant=out_quant)
        new.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            new.bias.data.copy_(lin.bias.data)
        return new

    def _w_scale(self) -> tuple[th.Tensor, int]:
        qmax = 2 ** (self.cfg.weight_bits - 1) - 1
        w = self.weight.detach().abs()
        m = w.amax(dim=1, keepdim=True) if self.cfg.per_channel else w.max()
        return _scale_from_max(m, qmax, self.cfg.pow2_scales), qmax

    def _b_scale(self) -> tuple[th.Tensor, int]:
        qmax = 2 ** (self.cfg.bias_bits - 1) - 1
        return _scale_from_max(self.bias.detach().abs().max(), qmax, self.cfg.pow2_scales), qmax

    def forward(self, x: th.Tensor) -> th.Tensor:
        w, b = self.weight, self.bias
        if self.quant_on:
            if self.cfg.weight_bits is not None:
                s, qmax = self._w_scale()
                w = _ste_quant(w, s, -qmax, qmax)
            if b is not None and self.cfg.bias_bits is not None:
                s, qmax = self._b_scale()
                b = _ste_quant(b, s, -qmax, qmax)
        y = F.linear(x, w, b)
        return self.out_q(y) if self.out_q is not None else y

    @th.no_grad()
    def integer_params(self):
        """(w_int, w_scale, b_int, b_scale) for export."""
        s, qmax = self._w_scale()
        w_int = th.clamp(th.round(self.weight / s), -qmax, qmax)
        b_int = b_scale = None
        if self.bias is not None and self.cfg.bias_bits is not None:
            b_scale, bq = self._b_scale()
            b_int = th.clamp(th.round(self.bias / b_scale), -bq, bq)
        return w_int, s, b_int, b_scale


class QActivation(nn.Module):
    """Wraps an activation module (same Sequential index) and quantizes its output."""

    def __init__(self, act: nn.Module, cfg: QATConfig):
        super().__init__()
        self.act = act
        self.q = ActQuant(cfg.act_bits, not isinstance(act, nn.ReLU),
                          cfg.act_momentum, cfg.pow2_scales)

    def forward(self, x):
        return self.q(self.act(x))


def set_quant_enabled(module: nn.Module, enabled: bool) -> None:
    for m in module.modules():
        if hasattr(m, "quant_on"):
            m.quant_on = enabled


# =============================================================================
# 3. SB3 integration
# =============================================================================

class QuantActor(Actor):
    """SB3 Actor with fake-quantized hidden layers and mu head.

    Only the mean path (latent_pi -> mu -> tanh) is what you deploy; the
    log_std head is training-only and stays float.
    """

    def __init__(self, *args, qcfg: QATConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise NotImplementedError("gSDE is not supported by QuantActor")
        layers = []
        for m in self.latent_pi:
            if isinstance(m, nn.Linear):
                layers.append(QuantLinear.from_linear(m, qcfg))
            else:
                layers.append(QActivation(m, qcfg))
        self.latent_pi = nn.Sequential(*layers)
        self.mu = QuantLinear.from_linear(self.mu, qcfg, out_quant=True)


class QATSACPolicy(SACPolicy):
    """SACPolicy with a QuantActor. Pass `qcfg=QATConfig(...)` via policy_kwargs."""

    def __init__(self, *args, qcfg: Optional[QATConfig] = None, **kwargs):
        # Must be set BEFORE super().__init__: it calls make_actor() internally.
        self.qcfg = qcfg or QATConfig()
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> QuantActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return QuantActor(**actor_kwargs, qcfg=self.qcfg).to(self.device)


class QATScheduleCallback(BaseCallback):
    """Delay fake-quantization: train in float until `enable_at_step`, then
    switch it on. Observers calibrate during the float phase. Put this FIRST
    in the callback list so EvalCallback sees the right mode."""

    def __init__(self, enable_at_step: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.enable_at_step = enable_at_step
        self._on = False

    def _apply(self, flag: bool) -> None:
        set_quant_enabled(self.model.policy.actor, flag)
        self._on = flag

    def _on_training_start(self) -> None:
        self._apply(self.model.num_timesteps >= self.enable_at_step)

    def _on_step(self) -> bool:
        if not self._on and self.model.num_timesteps >= self.enable_at_step:
            self._apply(True)
            if self.verbose:
                print(f"[QAT] fake-quantization enabled at step {self.model.num_timesteps}")
        return True


# =============================================================================
# Export
# =============================================================================

@th.no_grad()
def export_actor(actor: QuantActor, path: str, obs_quantizer: Optional[FixedPointQuantizer] = None) -> None:
    """Dump integer weights/biases, scales and activation formats to an .npz
    for the FPGA flow. Real value = int * scale (scales are powers of two when
    cfg.pow2_scales)."""
    layers = [m for m in actor.latent_pi if isinstance(m, QuantLinear)] + [actor.mu]
    acts = [m.q for m in actor.latent_pi if isinstance(m, QActivation)] + [actor.mu.out_q]
    out: dict[str, np.ndarray] = {}
    for i, (lin, aq) in enumerate(zip(layers, acts)):
        if lin.cfg.weight_bits is not None:
            w_int, w_s, b_int, b_s = lin.integer_params()
            out[f"l{i}_w_int"] = w_int.cpu().numpy().astype(np.int32)
            out[f"l{i}_w_scale"] = w_s.cpu().numpy()
            if b_int is not None:
                out[f"l{i}_b_int"] = b_int.cpu().numpy().astype(np.int32)
                out[f"l{i}_b_scale"] = b_s.cpu().numpy()
        else:
            out[f"l{i}_w"] = lin.weight.cpu().numpy()
            out[f"l{i}_b"] = lin.bias.cpu().numpy()
        if aq.bits is not None:
            out[f"l{i}_act_scale"] = aq.current_scale().cpu().numpy()
            out[f"l{i}_act_bits"] = np.array(aq.bits)
            out[f"l{i}_act_signed"] = np.array(aq.signed)
    if obs_quantizer is not None:
        out["obs_int_bits"] = obs_quantizer.int_bits
        out["obs_frac_bits"] = obs_quantizer.frac_bits
    np.savez(path, **out)
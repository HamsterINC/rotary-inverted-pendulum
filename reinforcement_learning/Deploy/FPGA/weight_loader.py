"""
weight_loader.py - push MLP weights from the PYNQ PS into the PL inference engine.

Hardware side: weight_loader_axi.sv (AXI4-Lite, 0x8004_0000, 128 KB) in front of a
double-buffered weight RAM. The PL always runs from the ACTIVE bank; this driver
writes the SHADOW bank, reads it back, and commits. The swap happens in the PL
only while the inference engine is idle, so a policy step never sees mixed
weights. Uploading during an episode is therefore safe too.

Typical use on the board, between episodes:

    from pynq import Overlay
    from weight_loader import WeightLoader, pack_weights

    ol = Overlay("board_top_AUPZU3.bit")          # .hwh next to it
    wl = WeightLoader()                           # or FpgaDeployClient(...).weights
    print(wl.shape)                               # network the bitstream was built for

    # after a QAT update (torch layout: W is (out, in), inputs in HW order)
    words = pack_weights(W1, b1, W2, b2, W3, b3)  # floats -> Q.FRAC_W, or pass int16 with quantized=True
    wl.load(words)                                # write + verify + commit

    words = actor_to_pl_words(model.actor)        # SB3 SAC actor: mu AND log_std heads

Network / memory layout (must match inference_chip_control.sv):
    6H -> H1 (ReLU) -> H2 (ReLU) -> 2, 16-bit fixed point with FRAC_W fractional
    bits. H = observation history (OBS_HISTORY), H1/H2 = hidden widths (HIDDEN1/
    HIDDEN2 in board_top, net-arch in config.json; multiples of 4).
    Each layer is stored per tile of 4 output neurons: one bias word, then one word
    per input column. A 64-bit word holds lanes {n3, n2, n1, n0}, lane k = neuron 4r+k
    in bits [16k+15:16k]. Layer 3 is padded to 4 neurons: lane 0 = mean (mu),
    lane 1 = log_std (SAC head, bitstream ID 0x57E1_0005). A 1-output W3 still
    packs (lane 1 = 0 -> log_std 0, std 1: only use it with exploration off).
    words = H1/4 (6H+1) + H2/4 (H1+1) + (H2+1)      (6-128-128-1: 4481)
    Banks larger than the 8192-word AXI window are written page by page (WPAGE).

Input column order seen by layer 1: H frames, oldest first, each frame in
    HW_INPUT_ORDER = (s1_pos, s2_sin, s2_cos, s1_vel, s2_vel, F_prev)
which is pendulum_env's frame order and stacking. If your model uses another
order within a frame, pass input_perm (6 indices) to pack_weights().

SAC head in the PL (sac_policy_head.sv, ID 0x57E1_0005 and later):
    action = tanh(mean + exp(clamp(log_std, -5, 2)) * eps),  eps ~ N(0, 1)
    With exploration off (DEPLOY_CTRL bit4 = 0) eps = 0, so action = tanh(mean),
    the same as SB3's deterministic action. SB3 clamps log_std to [-20, 2], the
    PL to [-5, 2] (std >= 0.0067). sac_action_reference() models the head bit-exactly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------- constants
WEIGHT_BASE = 0x8004_0000          # set by scripts/add_weight_axi_port.tcl
WEIGHT_RANGE = 0x2_0000
REG_CTRL = 0x1_0000                # W: bit0 commit / R: status
REG_ID = 0x1_0004
REG_WORDS = 0x1_0008               # [23:0] words used, [31:24] log2(bank words)
REG_CONFIG = 0x1_0020              # [7:0] obs history, [15:8] action smooth window, [31:16] mrad/s
REG_CONFIG2 = 0x1_0024             # [15:0] policy rate in Hz
REG_WPAGE = 0x1_0028               # weight window page
REG_NET = 0x1_002C                 # [11:0] hidden 1, [23:12] hidden 2
ID_VALUE = 0x57E1_0005              # bitstream with the SAC head (mean + log_std)
ID_LEGACY = 0x57E1_0004             # older bitstream: action = clamp(output 0, -1, 1)
REG_SAC_SEED = 0x1_0030            # W: reseed the noise LFSR (non-zero)
REG_SAC_DIAG0 = 0x1_0034           # R: last sample [15:0] mean, [31:16] log_std (Q.9)
REG_SAC_DIAG1 = 0x1_0038           # R: last sample [15:0] eps (Q3.12), [31:16] u (Q.9)
LOG_STD_MIN, LOG_STD_MAX = -5, 2   # sac_policy_head clamp
PAGE_WORDS = 8192                  # 64-bit words per 64 KB weight window

FRAC_W = 9                         # inference_chip FRAC_W (top_pipeline parameter)
FRAME = 6                          # features per observation frame
HW_INPUT_ORDER = ("s1_pos", "s2_sin", "s2_cos", "s1_vel", "s2_vel", "F_prev")
NUM_PE = 4


@dataclass(frozen=True)
class NetShape:
    """6*obs_history -> hidden1 -> hidden2 -> outputs (1: mean, 2: mean + log_std).
    `outputs` does not change the memory layout (layer 3 is always one 4-lane tile)."""
    obs_history: int = 1
    hidden1: int = 128
    hidden2: int = 128
    outputs: int = 2

    @property
    def inputs(self) -> int:
        return FRAME * self.obs_history

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (self.inputs, self.hidden1, self.hidden2, self.outputs)

    @property
    def words(self) -> int:
        return (self.hidden1 // 4) * (self.inputs + 1) + (self.hidden2 // 4) * (self.hidden1 + 1) \
            + (self.hidden2 + 1)

    def validate(self) -> "NetShape":
        if self.obs_history < 1 or self.hidden1 < 4 or self.hidden2 < 4 \
                or self.hidden1 % 4 or self.hidden2 % 4 or self.outputs not in (1, 2):
            raise ValueError(f"invalid network shape {self} (hidden widths: multiples of 4, 1 or 2 outputs)")
        return self

    def __str__(self) -> str:
        return f"{self.inputs}-{self.hidden1}-{self.hidden2}-{self.outputs} (obs history {self.obs_history})"


def words_used(obs_history: int = 1, hidden1: int = 128, hidden2: int = 128) -> int:
    return NetShape(obs_history, hidden1, hidden2).words


WORDS_USED = words_used()          # default 6-128-128-1 (kept for older callers)


# ----------------------------------------------------------------------------- fixed point
def quantize(x, frac: int = FRAC_W) -> np.ndarray:
    """float -> int16 with `frac` fractional bits (round to nearest, saturate).
    Use the same rounding in your QAT fake-quant (scale = 2**-frac) so that the
    network you train is the one the PL runs."""
    q = np.round(np.asarray(x, dtype=np.float64) * (1 << frac))
    return np.clip(q, -32768, 32767).astype(np.int16)


def dequantize(q, frac: int = FRAC_W) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / (1 << frac)


# ----------------------------------------------------------------------------- packing
def _pack_lanes(lanes: np.ndarray) -> np.ndarray:
    """(..., 4) int16 -> (...,) uint64, lane k in bits [16k+15:16k]."""
    u = lanes.astype(np.int16).view(np.uint16).astype(np.uint64)
    return u[..., 0] | (u[..., 1] << 16) | (u[..., 2] << 32) | (u[..., 3] << 48)


def _unpack_lanes(words: np.ndarray) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    lanes = [((w >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.uint16) for k in range(4)]
    return np.stack(lanes, axis=-1).view(np.int16)


def _pack_layer(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """W (out, in) int16, b (out,) int16 -> tiles of [bias word, in x weight words]."""
    out_f, in_f = W.shape
    pad = (-out_f) % NUM_PE
    W = np.pad(W, ((0, pad), (0, 0)))
    b = np.pad(b, (0, pad))
    tiles = W.shape[0] // NUM_PE
    Wt = W.reshape(tiles, NUM_PE, in_f).transpose(0, 2, 1)       # (tile, col, lane)
    bt = b.reshape(tiles, 1, NUM_PE)                              # (tile, 1, lane)
    return _pack_lanes(np.concatenate([bt, Wt], axis=1)).reshape(-1)


def shape_of(W1, W2, W3) -> NetShape:
    """Network shape from torch-layout weight matrices. W3 is (1, H2) (mean only)
    or (2, H2) (row 0 mean, row 1 log_std)."""
    in_f, h1, h2 = np.shape(W1)[1], np.shape(W1)[0], np.shape(W2)[0]
    if in_f % FRAME:
        raise ValueError(f"W1 has {in_f} inputs, expected a multiple of {FRAME}")
    n_out = np.shape(W3)[0]
    shape = NetShape(in_f // FRAME, h1, h2, n_out).validate()
    if np.shape(W2)[1] != h1 or tuple(np.shape(W3)) != (n_out, h2):
        raise ValueError(f"inconsistent layer shapes W1{np.shape(W1)} W2{np.shape(W2)} W3{np.shape(W3)}")
    return shape


def pack_weights(W1, b1, W2, b2, W3, b3, *, frac: int = FRAC_W, quantized: bool = False,
                 input_perm=None) -> np.ndarray:
    """Pack one actor network into the PL weight-RAM image (uint64[shape.words]).

    W* use the torch nn.Linear layout (out_features, in_features). The shape is
    taken from the matrices: W1 (H1, 6H), W2 (H2, H1), W3 (2, H2) / b3 (2,) with
    row 0 = mu and row 1 = log_std (or W3 (1, H2) / b3 (1,): mean only).
    quantized=False: arrays are floats, quantized here with `frac` fractional bits.
    quantized=True : arrays are already integer codes (int16 range).
    input_perm: per-frame index list so that a frame reordered by it is in
                HW_INPUT_ORDER, e.g. a model trained on (s1_pos, s1_vel, s2_cos,
                s2_sin, s2_vel, F_prev) needs input_perm = [0, 3, 2, 1, 4, 5].
                Applied to every frame.
    """
    params = [np.asarray(p) for p in (W1, b1, W2, b2, W3, b3)]
    shape = shape_of(params[0], params[2], params[4])
    for name, b, n in (("b1", params[1], shape.hidden1), ("b2", params[3], shape.hidden2), ("b3", params[5], shape.outputs)):
        if b.shape != (n,):
            raise ValueError(f"{name}: expected ({n},), got {b.shape}")
    if input_perm is not None:
        perm = np.concatenate([np.asarray(input_perm) + FRAME * f for f in range(shape.obs_history)])
        params[0] = params[0][:, perm]
    if quantized:
        params = [np.clip(np.round(p), -32768, 32767).astype(np.int16) for p in params]
    else:
        params = [quantize(p, frac) for p in params]
    W1, b1, W2, b2, W3, b3 = params
    words = np.concatenate([_pack_layer(W1, b1), _pack_layer(W2, b2), _pack_layer(W3, b3)])
    assert words.size == shape.words
    return words


def unpack_weights(words, shape: NetShape | None = None) -> tuple[np.ndarray, ...]:
    """Inverse of pack_weights (int16 codes): W1, b1, W2, b2, W3, b3 in torch layout.
    shape=None: 128-128 hidden layers, 2 outputs, history from the image length."""
    words = np.asarray(words, dtype=np.uint64)
    if shape is None:
        shape = default_shape_from_words(words.size)
    if words.size < shape.words:
        raise ValueError(f"{words.size} words, network {shape} needs {shape.words}")
    words = words[:shape.words]
    sizes = shape.sizes
    out, pos = [], 0
    for i in range(3):
        in_f, out_f = sizes[i], sizes[i + 1]
        tiles = -(-out_f // NUM_PE)
        n = tiles * (in_f + 1)
        blk = _unpack_lanes(words[pos:pos + n]).reshape(tiles, in_f + 1, NUM_PE)
        pos += n
        b = blk[:, 0, :].reshape(-1)[:out_f]
        W = blk[:, 1:, :].transpose(0, 2, 1).reshape(-1, in_f)[:out_f]
        out += [W, b]
    return tuple(out)


def default_shape_from_words(n_words: int, hidden1: int = 128, hidden2: int = 128,
                             outputs: int = 2) -> NetShape:
    """Observation history of an image with the given hidden widths (from its length)."""
    rest = n_words - (hidden2 // 4) * (hidden1 + 1) - (hidden2 + 1)
    in_f = rest // (hidden1 // 4) - 1
    shape = NetShape(max(1, in_f // FRAME), hidden1, hidden2, outputs)
    if in_f < FRAME or in_f % FRAME or shape.words != n_words:
        raise ValueError(f"{n_words} words is not a 6H-{hidden1}-{hidden2}-1 image; pass shape=")
    return shape


def load_mem_file(path) -> np.ndarray:
    with open(path) as f:
        return np.array([int(l, 16) for l in f if l.strip()], dtype=np.uint64)


def save_mem_file(path, words) -> None:
    with open(path, "w") as f:
        for w in np.asarray(words, dtype=np.uint64):
            f.write(f"{int(w):016x}\n")


# ----------------------------------------------------------------------------- reference model
def _sat16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767).astype(np.int64)


def _wrap32(x: np.ndarray) -> np.ndarray:
    return ((x + (1 << 31)) % (1 << 32)) - (1 << 31)


def reference_heads(words, obs, frac: int = FRAC_W, shape: NetShape | None = None):
    """Bit-exact model of inference_chip (IO_FRAC == FRAC_W).

    obs: int16 codes, shape (6H,) or (N, 6H): H frames oldest first, each in
         HW_INPUT_ORDER.
    shape: network shape (None: 128-128 hidden, history from the image length).
    returns: (mean, log_std) int16 codes (inference_result, inference_log_std),
             each shape () or (N,); log_std is 0 for a 1-output image.
    Per layer: acc = (bias << frac) + sum(a * w)  (32-bit wrap-around accumulator),
               out = sat16(acc >> frac), ReLU (acc < 0 -> 0) on layers 1 and 2.
    """
    if shape is not None and shape.outputs != 2:
        shape = NetShape(shape.obs_history, shape.hidden1, shape.hidden2, 2)
    W1, b1, W2, b2, W3, b3 = (p.astype(np.int64) for p in unpack_weights(words, shape))
    a = np.atleast_2d(np.asarray(obs, dtype=np.int64))
    for W, b, relu in ((W1, b1, True), (W2, b2, True), (W3, b3, False)):
        acc = _wrap32((b << frac)[None, :] + a @ W.T)
        out = _sat16(acc >> frac)
        a = np.where(acc < 0, 0, out) if relu else out
    mean, log_std = a[:, 0].astype(np.int16), a[:, 1].astype(np.int16)
    return (mean[0], log_std[0]) if np.ndim(obs) == 1 else (mean, log_std)


def reference_forward(words, obs, frac: int = FRAC_W, shape: NetShape | None = None) -> np.ndarray:
    """Network output 0 (the mean) only; see reference_heads()."""
    return reference_heads(words, obs, frac, shape)[0]


# SAC head tables: same formulas as Inference_Accelerator/scripts/gen_sac_luts.py
_EXP_LUT = np.minimum(0xFFFF, np.round(np.exp(-8 + (np.arange(512) + 0.5) / 32) * 4096)).astype(np.int64)
_TANH_LUT = np.minimum(0x7FFF, np.round(np.tanh((np.arange(512) + 0.5) / 128) * 32768)).astype(np.int64)


def sac_action_reference(mean, log_std, eps=0, frac: int = FRAC_W):
    """Bit-exact model of sac_policy_head (FRAC = frac, log_std clamp [-5, 2]).

    mean, log_std: int16 codes (Q.frac); eps: int16 Q3.12 noise code (0 =
    deterministic). Returns (action, u) int16 codes with u = mean + std * eps
    and action = tanh(u)."""
    m = np.asarray(mean, dtype=np.int64)
    ls = np.clip(np.asarray(log_std, dtype=np.int64), LOG_STD_MIN << frac, LOG_STD_MAX << frac)
    std = _EXP_LUT[(ls + (8 << frac)) >> (frac - 5)]                       # Q4.12
    psh = 24 - frac
    u = ((std * np.asarray(eps, dtype=np.int64) + (1 << (psh - 1))) >> psh) + m
    idx = np.minimum(np.abs(u) >> (frac - 7), 511)
    tsh = 15 - frac
    mag = (_TANH_LUT[idx] + (1 << (tsh - 1))) >> tsh if tsh else _TANH_LUT[idx]
    action = np.where(u < 0, -mag, mag)
    return action.astype(np.int16), np.clip(u, -32768, 32767).astype(np.int16)


# ----------------------------------------------------------------------------- SB3 / QAT actor export
def actor_float_params(actor, param_noise_std: float = 0.0, rng=None,
                       log_std: bool = True) -> tuple[np.ndarray, ...]:
    """(W1, b1, W2, b2, W3, b3) as floats from an SB3 SAC actor (stock Actor or
    Training.qat.QuantActor), i.e. latent_pi = [Linear, act, Linear, act], mu = Linear.
    log_std=True: W3/b3 stack [mu; log_std] (2 outputs) for the PL SAC head;
    log_std=False: mu only (1 output).

    QuantLinear layers with quantization on are exported with their fake-quantized
    values (what the QAT forward pass uses). param_noise_std > 0 adds Gaussian
    noise to every parameter (parameter-space exploration: the PL policy itself
    is deterministic).

    The log_std head stays float even in QuantActor, so it is rounded to Q.FRAC_W
    here. That only changes the exploration std, not the deterministic action.
    """
    import torch as th
    lins = [m for m in actor.latent_pi if isinstance(m, th.nn.Linear)] + [actor.mu]
    if len(lins) != 3:
        raise ValueError(f"actor has {len(lins)} linear layers, the PL network has 3")
    if log_std:
        if getattr(actor, "use_sde", False) or not isinstance(getattr(actor, "log_std", None), th.nn.Linear):
            raise ValueError("log_std export needs a state-dependent log_std Linear head (no gSDE)")
        lins.append(actor.log_std)
    rng = rng or np.random.default_rng()
    out = []
    with th.no_grad():
        for lin in lins:
            quant = (getattr(lin, "quant_on", False) and hasattr(lin, "integer_params")
                     and lin.cfg.weight_bits is not None)
            if quant:
                w_int, w_s, b_int, b_s = lin.integer_params()
                w = (w_int * w_s).cpu().numpy()
                b = (b_int * b_s).cpu().numpy() if b_int is not None else lin.bias.cpu().numpy()
            else:
                w, b = lin.weight.cpu().numpy(), lin.bias.cpu().numpy()
            w = w.astype(np.float64)
            b = b.astype(np.float64)
            if param_noise_std > 0.0:
                w = w + rng.normal(0.0, param_noise_std, w.shape)
                b = b + rng.normal(0.0, param_noise_std, b.shape)
            out += [w, b]
    if log_std:                                  # [mu; log_std] as one 2-output layer
        W_mu, b_mu, W_ls, b_ls = out[4:]
        out = out[:4] + [np.vstack([W_mu, W_ls]), np.concatenate([b_mu, b_ls])]
    return tuple(out)


def actor_to_pl_words(actor, param_noise_std: float = 0.0, rng=None, frac: int = FRAC_W,
                      warn: bool = True, log_std: bool = True) -> np.ndarray:
    """SB3 actor -> PL weight image (mu in lane 0, log_std in lane 1). Warns when
    the conversion to Q.frac loses information (values off the 2^-frac grid, or
    saturating at +-64). The log_std rows are not QAT-trained, so W3/b3 usually
    report rounding; that only affects the exploration std."""
    params = actor_float_params(actor, param_noise_std, rng, log_std=log_std)
    if warn and param_noise_std == 0.0:
        lsb = 2.0 ** -frac
        for name, p in zip(("W1", "b1", "W2", "b2", "W3", "b3"), params):
            q = quantize(p, frac).astype(np.float64) * lsb
            err = np.abs(q - p).max()
            if err > lsb / 2 + 1e-12:
                print(f"[weight_loader] {name}: saturates in Q.{frac} (max |x| = {np.abs(p).max():.3f})")
            elif err > 1e-9:
                print(f"[weight_loader] {name}: not on the 2^-{frac} grid, max rounding error {err:.2e}")
    return pack_weights(*params, frac=frac)


# ----------------------------------------------------------------------------- PL driver
@dataclass(frozen=True)
class PLConfig:
    """Policy settings baked into the bitstream (board_top localparams). They
    must match the checkpoint's config.json."""
    obs_history: int               # obs_history_len
    action_smooth_window: int      # action_smooth_window
    max_velocity_rad_s: float      # max_velocity_rad_s
    control_freq_hz: float         # control_freq_hz
    hidden1: int                   # net-arch[0]
    hidden2: int                   # net-arch[1]

    @property
    def shape(self) -> NetShape:
        return NetShape(self.obs_history, self.hidden1, self.hidden2)

    def mismatches(self, cfg: dict) -> list[str]:
        """Human-readable differences to a training config.json (keys absent
        from the config fall back to the pendulum_env / train_sac defaults)."""
        want = {
            "obs_history_len": (int(cfg.get("obs_history_len", 4)), self.obs_history),
            "action_smooth_window": (int(cfg.get("action_smooth_window", 4)), self.action_smooth_window),
            "max_velocity_rad_s": (float(cfg.get("max_velocity_rad_s", 5.0)), self.max_velocity_rad_s),
            "control_freq_hz": (float(cfg.get("control_freq_hz", 50.0)), self.control_freq_hz),
        }
        out = [f"{k}: checkpoint {c:g}, PL {p:g}" for k, (c, p) in want.items()
               if abs(float(c) - float(p)) > 1e-3]
        arch = [int(x) for x in cfg.get("net-arch", [256, 256])]
        if arch != [self.hidden1, self.hidden2]:
            out.append(f"net-arch: checkpoint {arch}, PL [{self.hidden1}, {self.hidden2}]")
        return out


def read_pl_config(mmio) -> PLConfig:
    c = mmio.read(REG_CONFIG)
    c2 = mmio.read(REG_CONFIG2)
    net = mmio.read(REG_NET)
    return PLConfig(obs_history=c & 0xFF, action_smooth_window=(c >> 8) & 0xFF,
                    max_velocity_rad_s=((c >> 16) & 0xFFFF) / 1000.0,
                    control_freq_hz=float(c2 & 0xFFFF),
                    hidden1=net & 0xFFF, hidden2=(net >> 12) & 0xFFF)


@dataclass
class WeightStatus:
    active_bank: int
    commit_pending: bool
    engine_idle: bool
    dropped_writes: int
    commits: int


class WeightLoader:
    """Driver for weight_loader_axi. Runs on the PYNQ board."""

    def __init__(self, base_addr: int = WEIGHT_BASE, mmio=None):
        if mmio is None:
            from pynq import MMIO       # imported here so packing works off-board
            mmio = MMIO(base_addr, WEIGHT_RANGE)
        self.mmio = mmio
        ident = self.mmio.read(REG_ID)
        if ident not in (ID_VALUE, ID_LEGACY):
            raise RuntimeError(f"weight_loader_axi ID {ident:#x} at {base_addr:#x}, driver expects "
                               f"{ID_VALUE:#x}. Is the current bitstream loaded?")
        self.ident = ident
        self.has_sac_head = ident == ID_VALUE      # PL samples tanh(mean + std*eps)
        words = self.mmio.read(REG_WORDS)
        self.words_used = words & 0xFFFFFF
        self.bank_words = 1 << (words >> 24)
        self.config = read_pl_config(self.mmio)
        self.shape = self.config.shape
        if self.shape.words != self.words_used:
            raise RuntimeError(f"PL reports {self.words_used} weight words, network {self.shape} "
                               f"needs {self.shape.words}")
        self.obs_history = self.shape.obs_history

    def status(self) -> WeightStatus:
        s = self.mmio.read(REG_CTRL)
        return WeightStatus(active_bank=s & 1, commit_pending=bool(s & 2), engine_idle=bool(s & 4),
                            dropped_writes=(s >> 8) & 0xFF, commits=s >> 16)

    def _wait_not_pending(self, timeout: float) -> WeightStatus:
        t0 = time.monotonic()
        while True:
            st = self.status()
            if not st.commit_pending:
                return st
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("weight commit still pending (inference engine never idle?)")

    def write_shadow(self, words: np.ndarray, timeout: float = 0.1) -> None:
        """Write the full image into the inactive bank (page by page)."""
        words = np.ascontiguousarray(words, dtype="<u8")
        if words.size > self.bank_words:
            raise ValueError("image larger than a bank")
        st = self._wait_not_pending(timeout)
        for page, start in enumerate(range(0, words.size, PAGE_WORDS)):
            self.mmio.write(REG_WPAGE, page)
            u32 = words[start:start + PAGE_WORDS].view("<u4")   # word n -> 8n (low), 8n+4 (high)
            self.mmio.array[0:u32.size] = u32
        self.mmio.write(REG_WPAGE, 0)
        if self.status().dropped_writes != st.dropped_writes:
            raise RuntimeError("PL dropped weight writes (commit raced the upload)")

    def read_shadow(self, n: int | None = None) -> np.ndarray:
        n = self.words_used if n is None else n
        out = np.empty(n, dtype="<u8")
        for page, start in enumerate(range(0, n, PAGE_WORDS)):
            k = min(PAGE_WORDS, n - start)
            self.mmio.write(REG_WPAGE, page)
            out[start:start + k] = np.array(self.mmio.array[0:2 * k], dtype="<u4").view("<u8")
        self.mmio.write(REG_WPAGE, 0)
        return out

    def commit(self, timeout: float = 0.1) -> WeightStatus:
        """Make the shadow bank active (takes effect between two inferences)."""
        before = self._wait_not_pending(timeout)
        self.mmio.write(REG_CTRL, 1)
        after = self._wait_not_pending(timeout)
        if after.active_bank == before.active_bank:
            raise RuntimeError("bank swap did not happen")
        return after

    def load(self, words: np.ndarray, verify: bool = True, timeout: float = 0.1) -> WeightStatus:
        """write shadow -> (read back + compare) -> commit."""
        words = np.asarray(words, dtype=np.uint64)
        if words.size != self.words_used:
            raise ValueError(f"image has {words.size} words, the PL network {self.shape} "
                             f"needs {self.words_used}")
        self.write_shadow(words, timeout)
        if verify:
            back = self.read_shadow(words.size)
            bad = np.flatnonzero(back != words)
            if bad.size:
                raise RuntimeError(f"weight readback mismatch at {bad.size} words, first at {bad[0]}")
        return self.commit(timeout)


# ----------------------------------------------------------------------------- self test (off-board)
def _selftest(mem_path: str, vectors_path: str | None = None, n: int = 64,
              shape: NetShape | None = None) -> None:
    words = load_mem_file(mem_path)
    shape = shape or default_shape_from_words(words.size)
    params = unpack_weights(words, shape)
    repacked = pack_weights(*params, quantized=True)
    assert np.array_equal(repacked, words), "pack(unpack(mem)) != mem"
    print(f"pack/unpack round trip OK ({words.size} words, network {shape})")

    rng = np.random.default_rng(0)
    obs = rng.integers(-2 << FRAC_W, 2 << FRAC_W, size=(n, shape.inputs)).astype(np.int16)
    res, ls = reference_heads(words, obs, shape=shape)
    print("reference mean    (first 8):", res[:8])
    print("reference log_std (first 8):", ls[:8], "" if np.any(ls) else "(all 0: image has no log_std head)")
    if vectors_path:
        with open(vectors_path, "w") as f:
            for o, r in zip(obs, res):
                f.write(" ".join(f"{int(v) & 0xFFFF:04x}" for v in (*o, r)) + "\n")
        print(f"wrote {n} test vectors to {vectors_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="weight packer self test (runs without PYNQ)")
    ap.add_argument("mem", help="unified_weights.mem")
    ap.add_argument("--vectors", help="write obs/expected-output vectors for an RTL testbench")
    ap.add_argument("--history", type=int, default=None)
    ap.add_argument("--hidden", type=int, nargs=2, default=None, metavar=("H1", "H2"))
    args = ap.parse_args()
    shp = None
    if args.history is not None or args.hidden is not None:
        h1, h2 = args.hidden or (128, 128)
        shp = NetShape(args.history or 1, h1, h2)
    _selftest(args.mem, args.vectors, shape=shp)

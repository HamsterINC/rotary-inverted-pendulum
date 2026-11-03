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

    # after a QAT update (torch layout: W is (out, in), inputs in HW order)
    words = pack_weights(W1, b1, W2, b2, W3, b3)  # floats -> Q.FRAC_W, or pass int16 with quantized=True
    wl.load(words)                                # write + verify + commit, ~10 ms

Network / memory layout (must match inference_chip_control.sv):
    6H -> 128 (ReLU) -> 128 (ReLU) -> 1, 16-bit fixed point with FRAC_W fractional
    bits, H = observation history (OBS_HISTORY in board_top, obs_history_len in sim).
    Each layer is stored per tile of 4 output neurons: one bias word, then one word
    per input column. A 64-bit word holds lanes {n3, n2, n1, n0}, lane k = neuron 4r+k
    in bits [16k+15:16k]. Layer 3 is padded to 4 neurons (only lane 0 is used).
    Layer bases: L1 = 0 (32 x (6H+1)), L2 = 32(6H+1) (32 x 129), L3 = L2 + 4128 (129)
    -> words_used(H) = 32(6H+1) + 4257   (H = 1: 4481, H = 4: 5057).

Input column order seen by layer 1: H frames, oldest first, each frame in
    HW_INPUT_ORDER = (s1_pos, s2_sin, s2_cos, s1_vel, s2_vel, F_prev)
which is pendulum_env's frame order and stacking. If your model uses another
order within a frame, pass input_perm (6 indices) to pack_weights().
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
REG_WORDS = 0x1_0008
ID_VALUE = 0x57E1_0002

FRAC_W = 9                         # inference_chip FRAC_W (top_pipeline parameter)
FRAME = 6                          # features per observation frame
HIDDEN = 128
HW_INPUT_ORDER = ("s1_pos", "s2_sin", "s2_cos", "s1_vel", "s2_vel", "F_prev")
NUM_PE = 4


def layer_sizes(obs_history: int = 1) -> tuple[int, int, int, int]:
    return (FRAME * obs_history, HIDDEN, HIDDEN, 1)


def words_used(obs_history: int = 1) -> int:
    return 32 * (FRAME * obs_history + 1) + 32 * (HIDDEN + 1) + (HIDDEN + 1)


def history_from_words(n_words: int) -> int:
    """Observation history H of a weight image / of the PL (from its word count)."""
    in_f = (n_words - 32 * (HIDDEN + 1) - (HIDDEN + 1)) // 32 - 1
    if in_f < FRAME or in_f % FRAME or words_used(in_f // FRAME) != n_words:
        raise ValueError(f"{n_words} words is not a valid 6H-128-128-1 image")
    return in_f // FRAME


WORDS_USED = words_used(1)         # H = 1 (kept for older callers)


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


def pack_weights(W1, b1, W2, b2, W3, b3, *, frac: int = FRAC_W, quantized: bool = False,
                 input_perm=None) -> np.ndarray:
    """Pack one actor network into the PL weight-RAM image (uint64[words_used(H)]).

    W* use the torch nn.Linear layout (out_features, in_features); W1 has 6*H
    columns (H stacked frames, oldest first), which sets H.
    quantized=False: arrays are floats, quantized here with `frac` fractional bits.
    quantized=True : arrays are already integer codes (int16 range).
    input_perm: per-frame index list so that a frame reordered by it is in
                HW_INPUT_ORDER, e.g. a model trained on (s1_pos, s1_vel, s2_cos,
                s2_sin, s2_vel, F_prev) needs input_perm = [0, 3, 2, 1, 4, 5].
                Applied to every frame.
    """
    params = [np.asarray(p) for p in (W1, b1, W2, b2, W3, b3)]
    in_f = params[0].shape[1]
    if in_f % FRAME:
        raise ValueError(f"W1 has {in_f} inputs, expected a multiple of {FRAME}")
    hist = in_f // FRAME
    if input_perm is not None:
        perm = np.concatenate([np.asarray(input_perm) + FRAME * f for f in range(hist)])
        params[0] = params[0][:, perm]
    if quantized:
        params = [np.clip(np.round(p), -32768, 32767).astype(np.int16) for p in params]
    else:
        params = [quantize(p, frac) for p in params]
    W1, b1, W2, b2, W3, b3 = params

    sizes = layer_sizes(hist)
    expect = [(sizes[i + 1], sizes[i]) for i in range(3)]
    for name, W, b, shp in (("W1", W1, b1, expect[0]), ("W2", W2, b2, expect[1]), ("W3", W3, b3, expect[2])):
        if W.shape != shp or b.shape != (shp[0],):
            raise ValueError(f"{name}: expected W{shp} b({shp[0]},), got W{W.shape} b{b.shape}")

    words = np.concatenate([_pack_layer(W1, b1), _pack_layer(W2, b2), _pack_layer(W3, b3)])
    assert words.size == words_used(hist)
    return words


def unpack_weights(words, obs_history: int | None = None) -> tuple[np.ndarray, ...]:
    """Inverse of pack_weights (int16 codes): W1, b1, W2, b2, W3, b3 in torch layout.
    obs_history=None: taken from the image length."""
    words = np.asarray(words, dtype=np.uint64)
    if obs_history is None:
        obs_history = history_from_words(words.size)
    words = words[:words_used(obs_history)]
    sizes = layer_sizes(obs_history)
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


def reference_forward(words, obs, frac: int = FRAC_W) -> np.ndarray:
    """Bit-exact model of inference_chip (IO_FRAC == FRAC_W).

    obs: int16 codes, shape (6H,) or (N, 6H): H frames oldest first, each in
         HW_INPUT_ORDER (H from the image length).
    returns: int16 inference_result, shape () or (N,).
    Per layer: acc = (bias << frac) + sum(a * w)  (32-bit wrap-around accumulator),
               out = sat16(acc >> frac), ReLU (acc < 0 -> 0) on layers 1 and 2.
    """
    W1, b1, W2, b2, W3, b3 = (p.astype(np.int64) for p in unpack_weights(words))
    a = np.atleast_2d(np.asarray(obs, dtype=np.int64))
    for W, b, relu in ((W1, b1, True), (W2, b2, True), (W3, b3, False)):
        acc = _wrap32((b << frac)[None, :] + a @ W.T)
        out = _sat16(acc >> frac)
        a = np.where(acc < 0, 0, out) if relu else out
    res = a[:, 0].astype(np.int16)
    return res[0] if np.ndim(obs) == 1 else res


# ----------------------------------------------------------------------------- SB3 / QAT actor export
def actor_float_params(actor, param_noise_std: float = 0.0, rng=None) -> tuple[np.ndarray, ...]:
    """(W1, b1, W2, b2, W3, b3) as floats from an SB3 SAC actor (stock Actor or
    Training.qat.QuantActor), i.e. latent_pi = [Linear, act, Linear, act], mu = Linear.

    QuantLinear layers with quantization on are exported with their fake-quantized
    values (what the QAT forward pass uses). param_noise_std > 0 adds Gaussian
    noise to every parameter (parameter-space exploration: the PL policy itself
    is deterministic).

    NOTE: SAC squashes the action with tanh(mu); the PL applies clamp(mu, -1, 1).
    The two agree only while |mu| is small.
    """
    import torch as th
    lins = [m for m in actor.latent_pi if isinstance(m, th.nn.Linear)] + [actor.mu]
    if len(lins) != 3:
        raise ValueError(f"actor has {len(lins)} linear layers, the PL network has 3")
    rng = rng or np.random.default_rng()
    out = []
    with th.no_grad():
        for lin in lins:
            if getattr(lin, "quant_on", False) and hasattr(lin, "integer_params")                     and lin.cfg.weight_bits is not None:
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
    return tuple(out)


def actor_to_pl_words(actor, param_noise_std: float = 0.0, rng=None, frac: int = FRAC_W,
                      warn: bool = True) -> np.ndarray:
    """SB3 actor -> PL weight image. Warns when the conversion to Q.frac loses
    information (values off the 2^-frac grid, or saturating at +-64)."""
    params = actor_float_params(actor, param_noise_std, rng)
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
REG_CONFIG = 0x1_0020              # [7:0] obs history, [15:8] action smooth window, [31:16] mrad/s
REG_CONFIG2 = 0x1_0024             # [15:0] policy rate in Hz


@dataclass(frozen=True)
class PLConfig:
    """Policy settings baked into the bitstream (board_top localparams). They
    must match the checkpoint's config.json."""
    obs_history: int               # obs_history_len
    action_smooth_window: int      # action_smooth_window
    max_velocity_rad_s: float      # max_velocity_rad_s
    control_freq_hz: float         # control_freq_hz

    def mismatches(self, cfg: dict) -> list[str]:
        """Human-readable differences to a training config.json (keys absent
        from the config fall back to the pendulum_env defaults)."""
        want = {
            "obs_history_len": (int(cfg.get("obs_history_len", 4)), self.obs_history),
            "action_smooth_window": (int(cfg.get("action_smooth_window", 4)), self.action_smooth_window),
            "max_velocity_rad_s": (float(cfg.get("max_velocity_rad_s", 5.0)), self.max_velocity_rad_s),
            "control_freq_hz": (float(cfg.get("control_freq_hz", 50.0)), self.control_freq_hz),
        }
        return [f"{k}: checkpoint {c:g}, PL {p:g}" for k, (c, p) in want.items()
                if abs(float(c) - float(p)) > 1e-3]


def read_pl_config(mmio) -> PLConfig:
    c = mmio.read(REG_CONFIG)
    c2 = mmio.read(REG_CONFIG2)
    return PLConfig(obs_history=c & 0xFF, action_smooth_window=(c >> 8) & 0xFF,
                    max_velocity_rad_s=((c >> 16) & 0xFFFF) / 1000.0,
                    control_freq_hz=float(c2 & 0xFFFF))


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
        if ident != ID_VALUE:
            raise RuntimeError(f"weight_loader_axi not found at {base_addr:#x} (ID={ident:#x}). "
                               "Is the right bitstream loaded?")
        words = self.mmio.read(REG_WORDS)
        self.words_used = words & 0xFFFF
        self.bank_words = words >> 16
        self.config = read_pl_config(self.mmio)
        if history_from_words(self.words_used) != self.config.obs_history:
            raise RuntimeError(f"PL reports {self.words_used} words but obs history "
                               f"{self.config.obs_history}")
        self.obs_history = self.config.obs_history

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
        """Write the full image into the inactive bank."""
        words = np.ascontiguousarray(words, dtype="<u8")
        if words.size > self.bank_words:
            raise ValueError("image larger than a bank")
        st = self._wait_not_pending(timeout)
        u32 = words.view("<u4")                          # word n -> 8n (low), 8n+4 (high)
        self.mmio.array[0:u32.size] = u32
        if self.status().dropped_writes != st.dropped_writes:
            raise RuntimeError("PL dropped weight writes (commit raced the upload)")

    def read_shadow(self, n: int | None = None) -> np.ndarray:
        n = self.words_used if n is None else n
        u32 = np.array(self.mmio.array[0:2 * n], dtype="<u4")
        return u32.view("<u8").copy()

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
            raise ValueError(f"image has {words.size} words (obs history "
                             f"{history_from_words(words.size)}), the PL network needs "
                             f"{self.words_used} (obs history {self.obs_history})")
        self.write_shadow(words, timeout)
        if verify:
            back = self.read_shadow(words.size)
            bad = np.flatnonzero(back != words)
            if bad.size:
                raise RuntimeError(f"weight readback mismatch at {bad.size} words, first at {bad[0]}")
        return self.commit(timeout)


# ----------------------------------------------------------------------------- self test (off-board)
def _selftest(mem_path: str, vectors_path: str | None = None, n: int = 64) -> None:
    words = load_mem_file(mem_path)
    hist = history_from_words(words.size)
    params = unpack_weights(words)
    repacked = pack_weights(*params, quantized=True)
    assert np.array_equal(repacked, words), "pack(unpack(mem)) != mem"
    print(f"pack/unpack round trip OK ({words.size} words, obs history {hist})")

    rng = np.random.default_rng(0)
    obs = rng.integers(-2 << FRAC_W, 2 << FRAC_W, size=(n, FRAME * hist)).astype(np.int16)
    res = reference_forward(words, obs)
    print("reference outputs (first 8):", res[:8])
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
    args = ap.parse_args()
    _selftest(args.mem, args.vectors)

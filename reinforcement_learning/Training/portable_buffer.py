"""Portable, pickle-free replay buffer export/import for SB3 off-policy models.

Why this exists
---------------
SB3's `model.save_replay_buffer()` pickles the whole buffer object. Arrays
pickled under NumPy 2.x reference the module `numpy._core`, which NumPy 1.x
(e.g. on the Zynq image) does not have, so unpickling fails with
`ModuleNotFoundError: No module named 'numpy._core'`.

This module writes the buffer as a directory of plain `.npy` files (one per
field) plus a `meta.json`. The `.npy` header format for simple numeric dtypes
has been stable since NumPy 1.0, no pickling is involved anywhere, and the
loader can memory-map the files so the Zynq doesn't need the whole buffer in
RAM twice.

Layout written by `export_buffer`:
    <out_dir>/
        observations.npy        (N, n_envs, *obs_shape)
        next_observations.npy   (N, n_envs, *obs_shape)
        actions.npy             (N, n_envs, action_dim)
        rewards.npy             (N, n_envs)
        dones.npy               (N, n_envs)
        timeouts.npy            (N, n_envs)
        meta.json               written last = completion marker

Transitions are stored in chronological order (oldest first), so a smaller
buffer on the target keeps the most recent experience.

Deliberately written for old Pythons/NumPys: no type annotations, no
NumPy-2-only APIs, pure numpy + json (SB3 is only imported by `convert`).

CLI:
    # Desktop (NumPy 2): convert an existing SB3 .pkl buffer
    python portable_buffer.py convert runs/x/replay_buffer.pkl runs/x/replay_buffer
    # Anywhere: inspect an exported buffer
    python portable_buffer.py info runs/x/replay_buffer
"""

import json
import os
import sys

import numpy as np

FORMAT_VERSION = 1
_FIELDS = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "timeouts",
)
_META = "meta.json"
_CHUNK = 50000  # rows copied per step on import (keeps peak RAM low)


def _chronological_index(size, pos, full):
    """Indices of the filled slots, oldest first."""
    if full:
        return np.concatenate([np.arange(pos, size), np.arange(0, pos)])
    return np.arange(0, pos)


def _check_supported(buffer):
    if isinstance(getattr(buffer, "observations", None), dict):
        raise ValueError("Dict observation spaces are not supported.")
    if getattr(buffer, "optimize_memory_usage", False):
        raise ValueError(
            "optimize_memory_usage=True buffers have no next_observations "
            "array and cannot be reordered safely; train with it off "
            "(the SB3 default)."
        )


def export_buffer(buffer, out_dir):
    """Write `buffer` (an SB3 ReplayBuffer or subclass) to `out_dir`.

    Returns the number of transitions written (per env slot).
    """
    _check_supported(buffer)
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Remove a stale completion marker first so a crash mid-export can't
    # leave a directory that looks complete.
    meta_path = os.path.join(out_dir, _META)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    idx = _chronological_index(buffer.buffer_size, buffer.pos, buffer.full)
    meta = {
        "format_version": FORMAT_VERSION,
        "n_transitions": int(len(idx)),
        "source_buffer_size": int(buffer.buffer_size),
        "n_envs": int(buffer.n_envs),
        "obs_shape": list(buffer.observations.shape[2:]),
        "action_shape": list(buffer.actions.shape[2:]),
        "buffer_class": type(buffer).__name__,
        "numpy_version": np.__version__,
        "fields": {},
    }
    # MirrorReplayBuffer carries the sign vector; keep it for a sanity check.
    obs_sign = getattr(buffer, "obs_sign", None)
    if obs_sign is not None:
        meta["obs_sign"] = np.asarray(obs_sign, dtype=np.float64).tolist()

    for name in _FIELDS:
        arr = getattr(buffer, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.dtype.hasobject:
            raise ValueError("Field %r has an object dtype; refusing to "
                             "export (would need pickle)." % name)
        # Force little-endian native numeric dtype and contiguous rows.
        out = np.ascontiguousarray(arr[idx], dtype=arr.dtype.newbyteorder("<"))
        np.save(os.path.join(out_dir, name + ".npy"), out, allow_pickle=False)
        meta["fields"][name] = {"dtype": out.dtype.str, "shape": list(out.shape)}
        del out

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta["n_transitions"]


def read_meta(in_dir):
    meta_path = os.path.join(os.fspath(in_dir), _META)
    if not os.path.exists(meta_path):
        raise IOError("%s missing: export incomplete or wrong directory."
                      % meta_path)
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported buffer format version %r"
                         % meta.get("format_version"))
    return meta


def load_buffer(buffer, in_dir, mmap=True, verbose=True):
    """Fill an existing (empty) SB3 ReplayBuffer from `in_dir`.

    `buffer` is typically `model.replay_buffer` right after `SAC.load(...)`.
    If the target buffer is smaller than the export, the most recent
    transitions are kept. Returns the number of transitions loaded.
    """
    _check_supported(buffer)
    in_dir = os.fspath(in_dir)
    meta = read_meta(in_dir)

    if meta["n_envs"] != buffer.n_envs:
        raise ValueError("n_envs mismatch: export has %d, target buffer has %d"
                         % (meta["n_envs"], buffer.n_envs))
    if list(buffer.observations.shape[2:]) != meta["obs_shape"]:
        raise ValueError("obs shape mismatch: export %s, target %s (check "
                         "obs_history_len / drop_velocity_obs)"
                         % (meta["obs_shape"], list(buffer.observations.shape[2:])))
    if list(buffer.actions.shape[2:]) != meta["action_shape"]:
        raise ValueError("action shape mismatch: export %s, target %s"
                         % (meta["action_shape"], list(buffer.actions.shape[2:])))

    tgt_sign = getattr(buffer, "obs_sign", None)
    if tgt_sign is not None and "obs_sign" in meta:
        if not np.allclose(np.asarray(tgt_sign, dtype=np.float64),
                           np.asarray(meta["obs_sign"])):
            print("WARNING: obs_sign differs between export and target buffer")

    n = int(meta["n_transitions"])
    cap = int(buffer.buffer_size)
    k = min(n, cap)
    start = n - k

    for name in _FIELDS:
        dst = getattr(buffer, name, None)
        if dst is None:
            continue
        path = os.path.join(in_dir, name + ".npy")
        if not os.path.exists(path):
            if name == "timeouts":  # older SB3 / handle_timeout off
                dst[:k] = 0
                continue
            raise IOError("Missing field file %s" % path)
        src = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if src.shape[0] != n:
            raise ValueError("%s has %d rows, meta says %d"
                             % (name, src.shape[0], n))
        for lo in range(0, k, _CHUNK):
            hi = min(lo + _CHUNK, k)
            dst[lo:hi] = src[start + lo:start + hi]  # casts dtype if needed
        del src

    buffer.pos = k % cap
    buffer.full = (k == cap)
    if verbose:
        print("Loaded %d transitions into %s (capacity %d)%s"
              % (k, type(buffer).__name__, cap,
                 "" if k == n else ", dropped %d oldest" % (n - k)))
    return k


def _cli(argv):
    if len(argv) >= 3 and argv[0] == "convert":
        from stable_baselines3.common.save_util import load_from_pkl
        buf = load_from_pkl(argv[1])
        n = export_buffer(buf, argv[2])
        print("Exported %d transitions to %s" % (n, argv[2]))
        return 0
    if len(argv) >= 2 and argv[0] == "info":
        print(json.dumps(read_meta(argv[1]), indent=2))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
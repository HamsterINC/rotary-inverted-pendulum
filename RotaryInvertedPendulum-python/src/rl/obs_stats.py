"""Find out which observation dimension a quantizer is destroying, and pick formats.

A uniform grid over the full +-1 normalized range wastes bits on signals that only
occupy a small part of it. Example: pen_vel / 32 rad/s while balancing is ~0.01, so
an 8-bit grid (step 2^-7 = 0.0078) leaves a handful of distinct codes: the velocity
the controller needs for fine balancing is simply not in the data any more.

This tool rolls out the env (unquantized; normalized or not, exactly as the env
emits it), then for each dimension reports, for the BALANCE phase (|theta| < 0.3)
and for everything:
  * spread (std, 99.9th percentile of |x|, max)
  * how many DISTINCT codes the balance phase uses at 8/10/12/14 bits
  * the smallest word width whose quantization noise (step/sqrt(12)) is below
    `--target-rel-err` of the balance-phase std, plus the abs_max it assumes
and prints flags to paste into train_sac_qat.py.

  python obs_stats.py --policy-zip runs/fp/last.zip --steps 30000 --obs-norm orig   # any train_sac flag
  python obs_stats.py --steps 30000        # random actions: no balance phase, only coverage

Needs torch/SB3/MuJoCo only for the rollout; analyze() is pure NumPy.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

NAMES = ["motor_pos", "cos_theta", "sin_theta", "motor_vel", "pen_vel", "prev_action"]
WORD_GRID = (8, 10, 12, 14)


def _ib(x: float) -> int:
    return int(math.ceil(math.log2(x))) if x > 0 else 0


def _codes(x: np.ndarray, int_bits: int, word: int) -> np.ndarray:
    step = 2.0 ** (int_bits - (word - 1))
    hi = 2 ** (word - 1) - 1
    return np.clip(np.floor(x / step + 0.5), -hi - 1, hi)


def analyze(obs: np.ndarray, balance: np.ndarray, target_rel_err: float = 0.1, out=print) -> dict:
    obs = np.asarray(obs, dtype=np.float64)
    has_bal = balance.sum() >= 50
    bal = obs[balance] if has_bal else obs
    if not has_bal:
        out("WARNING: (almost) no balance-phase samples; using all data. Pass --policy-zip "
            "of a policy that balances to size resolution requirements properly.\n")
    out(f"{len(obs)} samples, {int(balance.sum())} in balance phase (|theta|<0.3)\n")
    hdr = (f"{'dim':12s} {'std(bal)':>9s} {'p99.9|x|':>9s} {'max|x|':>8s} {'abs_max':>8s}  "
           + "".join(f"{'codes@' + str(w):>9s}" for w in WORD_GRID) + f" {'word bits':>10s}")
    out(hdr)
    out("-" * len(hdr))
    abs_max, words = [], []
    for j, name in enumerate(NAMES):
        a = np.abs(obs[:, j])
        p999, mx = float(np.quantile(a, 0.999)), float(a.max())
        # cos(theta) is flat near upright; its balance-phase std says nothing about the
        # information it carries (that lives in sin), so size it from the full range.
        std = float((obs if name == "cos_theta" else bal)[:, j].std())
        ib = _ib(p999)
        step_needed = max(target_rel_err * max(std, 1e-6) * math.sqrt(12.0), 1e-9)
        word = int(min(20, max(2, math.ceil(1 + ib - math.log2(step_needed)))))
        codes = [len(np.unique(_codes(bal[:, j], ib, w))) for w in WORD_GRID]
        out(f"{name:12s} {std:9.4f} {p999:9.4f} {mx:8.4f} {2.0 ** ib:8.4g}  "
            + "".join(f"{c:9d}" for c in codes) + f" {word:10d}")
        abs_max.append(2.0 ** ib)
        words.append(word)
    out("\n(codes@N = distinct quantized values the balance phase uses with an N-bit word and that "
        "abs_max;\n a dimension with only a few codes cannot support fine balancing.)")
    out("\nSuggested flags (per-dimension; word bits include the sign bit):")
    out(f"  --obs-abs-max {','.join(f'{v:g}' for v in abs_max)} "
        f"--obs-word-bits {','.join(str(w) for w in words)}")
    return dict(abs_max=abs_max, words=words)


def collect(policy_zip, steps, rest_argv, seed):
    import train_sac_qat as T
    args, _ = T.parse_all(rest_argv)
    extras = T._forwarded_extras(args)
    print(f"Env kwargs forwarded beyond the standard set: {extras or 'none'}")
    env = T.train_sac.make_env(
        domain_randomization=args.domain_randomization,
        dr_theta_bias_max_rad=args.dr_theta_bias_max_rad,
        **T._common_env_kwargs(args))()
    sac = T.SAC.load(policy_zip, device="cpu") if policy_zip else None
    rng = np.random.default_rng(seed)
    obs_l, bal_l = [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        obs_l.append(obs)
        bal_l.append(abs(env.unwrapped._theta_upright()) < 0.3)
        a = sac.predict(obs, deterministic=True)[0] if sac else rng.uniform(-1, 1, size=(1,))
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(obs_l), np.array(bal_l)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-zip", default=None, help="float (or QAT) SAC .zip that balances")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--target-rel-err", type=float, default=0.1,
                   help="allowed quantization noise as a fraction of the balance-phase std")
    p.add_argument("--seed", type=int, default=0)
    a, rest = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    obs, bal = collect(a.policy_zip, a.steps, [r for r in rest if r != "--"], a.seed)
    analyze(obs, bal, a.target_rel_err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
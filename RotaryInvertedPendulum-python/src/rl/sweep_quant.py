"""Sweep quantization levels and tabulate how much performance survives.

Each axis takes a list; "fp" means "leave this part in float".

  # one axis at a time, others float (cheapest way to find each knee)
  python sweep_quant.py --obs-word-bits fp 16 12 10 8 \
                        --weight-bits fp 8 6 4 --act-bits fp 8 6 \
                        --seeds 0 1 -- --total-steps 180000 --domain-randomization

  # full cross product
  python sweep_quant.py --mode grid --obs-word-bits 12 10 --weight-bits 8 6 --act-bits 8 \
                        -- --total-steps 180000

Everything after `--` is forwarded to train_sac_qat.py / train_sac.py.
Results are appended to sweep_results.csv (headless eval of last.zip and
best_model.zip, 10 deterministic episodes each, quantized env + policy).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import train_sac_qat as T

FP = "fp"


def _tok(v: str):
    return None if v.lower() == FP else int(v)


def build_configs(ow, wb, ab, mode):
    if mode == "grid":
        return list(itertools.product(ow, wb, ab))
    cfgs = [(None, None, None)]
    cfgs += [(o, None, None) for o in ow if o is not None]
    cfgs += [(None, w, None) for w in wb if w is not None]
    cfgs += [(None, None, a) for a in ab if a is not None]
    return list(dict.fromkeys(cfgs))  # de-dup, keep order


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs-word-bits", nargs="+", default=[FP])
    p.add_argument("--weight-bits", nargs="+", default=[FP])
    p.add_argument("--act-bits", nargs="+", default=[FP])
    p.add_argument("--mode", choices=["one-at-a-time", "grid"], default="one-at-a-time")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", default="sweep_results.csv")
    p.add_argument("--tag", default="sweep")
    sw, base = p.parse_known_args(argv if argv is not None else sys.argv[1:])
    base = [b for b in base if b != "--"]

    ow = [_tok(v) for v in sw.obs_word_bits]
    wb = [_tok(v) for v in sw.weight_bits]
    ab = [_tok(v) for v in sw.act_bits]
    configs = build_configs(ow, wb, ab, sw.mode)
    print(f"{len(configs)} configs x {len(sw.seeds)} seeds")

    out_path = Path(sw.out)
    new_file = not out_path.exists()
    fields = ["run", "obs_word_bits", "weight_bits", "act_bits", "seed", "which",
              "mean_return", "std_return", "upright_frac"]
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for (o, w, a), seed in itertools.product(configs, sw.seeds):
            name = f"{sw.tag}_o{o or 'fp'}_w{w or 'fp'}_a{a or 'fp'}_s{seed}"
            argv_run = list(base) + ["--run-name", name, "--seed", str(seed)]
            if o is not None:
                argv_run += ["--obs-word-bits", str(o)]
            if w is not None:
                argv_run += ["--weight-bits", str(w)]
            if a is not None:
                argv_run += ["--act-bits", str(a)]
            print(f"\n===== {name} =====")
            args, qargs = T.parse_all(argv_run)
            run_dir = T.train_qat(args, qargs)
            for which in ("last", "best_model"):
                path = run_dir / f"{which}.zip"
                if not path.exists():
                    continue
                m = T.evaluate_headless(path, args, qargs, sw.eval_episodes)
                writer.writerow(dict(run=name, obs_word_bits=o or FP, weight_bits=w or FP,
                                     act_bits=a or FP, seed=seed, which=which, **m))
                fh.flush()
                print(f"  {which}: {m}")
    print(f"\nResults in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

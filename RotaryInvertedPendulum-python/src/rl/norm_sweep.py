import argparse, subprocess, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--variants", nargs="+", default=["raw", "orig", "only_pv32", "only_pv8"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--total-steps", type=int, default=180_000)
ap.add_argument("--tag", default="norm")
args, extra = ap.parse_known_args()  # extra flags pass through to train_sac.py

runs = Path(__file__).resolve().parent / "runs"
for v in args.variants:
    for s in args.seeds:
        name = f"{args.tag}_{v}_s{s}"
        subprocess.run([sys.executable, "train_sac.py", "--obs-norm", v, "--seed", str(s),
                        "--run-name", name, "--total-steps", str(args.total_steps), *extra],
                       check=True)

print(f"\n{'variant':<18}{'n':>3}{'final eval':>16}{'best eval':>16}")
for v in args.variants:
    fin, best = [], []
    for s in args.seeds:
        f = runs / f"{args.tag}_{v}_s{s}" / "eval" / "evaluations.npz"
        if f.exists():
            r = np.load(f)["results"].mean(axis=1)  # mean over eval episodes per eval point
            fin.append(r[-1]); best.append(r.max())
    if fin:
        print(f"{v:<18}{len(fin):>3}{np.mean(fin):>10.1f}±{np.std(fin):<5.1f}{np.mean(best):>10.1f}±{np.std(best):<5.1f}")
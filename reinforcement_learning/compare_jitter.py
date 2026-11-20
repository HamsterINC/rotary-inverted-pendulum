"""Compare control-loop timing of fine-tune runs with different --inference.

    python compare_jitter.py runs/async_cpu_X runs/async_pl_X runs/async_baremetal_X
    python compare_jitter.py runs/async_* --mode eval --plot jitter.png
    python compare_jitter.py runs/async_X --mode eval --learner split   # busy vs idle

Every run's step_timing.csv (written by finetune_async.py) holds one row per
action with the PL-measured numbers, so the three inference paths are timed by
the same 100 MHz counters:

    period_ms   action -> action (the control period actually delivered)
    delay_ms    sample the policy used -> action applied (sensing-to-actuation)
    compute_us  inference only (pl: PL inference start -> action,
                baremetal: R5 obs build + MLP + accel write; empty for cpu)

Per run it reports the period mean / std / jitter % (std / mean), the
peak-to-peak and p1/p99 of the period error vs the nominal period, late ticks
(period > 1.5 x nominal), and the delay mean / std / p99 / max. --plot writes
period-error and delay histograms (needs matplotlib) and --out a CSV/JSON of
the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_run(run: Path, mode: str, skip_episodes: int, learner: str = "all") -> dict:
    path = run / "step_timing.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found (run predates step_timing.csv?)")
    period, delay, compute, episodes = [], [], [], set()
    inference = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if mode != "all" and row["mode"] != mode:
                continue
            # Older runs have no 'learner' column: they only match "all".
            if learner != "all" and row.get("learner") != learner:
                continue
            ep = int(row["episode"])
            if ep <= skip_episodes:
                continue
            inference = inference or row["inference"]
            episodes.add(ep)
            period.append(float(row["period_ms"]))
            delay.append(float(row["delay_ms"]) if row["delay_ms"] else np.nan)
            compute.append(float(row["compute_us"]) if row["compute_us"] else np.nan)
    src = run / "source.json"
    if inference is None and src.exists():
        inference = json.loads(src.read_text()).get("inference")
    control_hz = None
    cfg = run / "config.json"
    if cfg.exists():
        control_hz = json.loads(cfg.read_text()).get("control_freq_hz")
    if learner != "all":
        inference = f"{inference or '?'} [{learner}]"
    return {"run": run.name, "inference": inference or "?", "control_hz": control_hz,
            "period": np.array(period), "delay": np.array(delay),
            "compute": np.array(compute), "n_episodes": len(episodes)}


def summarize(d: dict, nominal_ms: float) -> dict:
    p, dl, c = d["period"], d["delay"], d["compute"]
    dl = dl[np.isfinite(dl)]
    c = c[np.isfinite(c)]
    out = {"run": d["run"], "inference": d["inference"], "episodes": d["n_episodes"],
           "n_actions": int(p.size)}
    if p.size:
        err_us = (p - nominal_ms) * 1e3
        out.update(
            period_mean_ms=float(p.mean()), period_std_us=float(p.std() * 1e3),
            jitter_pct=float(100 * p.std() / p.mean()),
            err_p1_us=float(np.percentile(err_us, 1)), err_p99_us=float(np.percentile(err_us, 99)),
            err_pp_us=float(err_us.max() - err_us.min()),
            late_ticks=int(np.sum(p > 1.5 * nominal_ms)),
        )
    if dl.size:
        out.update(delay_mean_ms=float(dl.mean()), delay_std_us=float(dl.std() * 1e3),
                   delay_p99_ms=float(np.percentile(dl, 99)), delay_max_ms=float(dl.max()))
    if c.size:
        out.update(compute_mean_us=float(c.mean()), compute_max_us=float(c.max()))
    return out


COLUMNS = [
    ("inference", "inference", "s"), ("episodes", "eps", "d"), ("n_actions", "actions", "d"),
    ("period_mean_ms", "period ms", ".3f"), ("period_std_us", "std us", ".1f"),
    ("jitter_pct", "jitter %", ".3f"), ("err_p1_us", "err p1 us", ".1f"),
    ("err_p99_us", "err p99 us", ".1f"), ("err_pp_us", "err p-p us", ".1f"),
    ("late_ticks", "late", "d"), ("delay_mean_ms", "delay ms", ".3f"),
    ("delay_std_us", "d.std us", ".1f"), ("delay_p99_ms", "d.p99 ms", ".3f"),
    ("delay_max_ms", "d.max ms", ".3f"), ("compute_mean_us", "comp us", ".1f"),
    ("compute_max_us", "comp max", ".1f"),
]


def print_table(rows: list[dict]) -> None:
    def cell(r, key, spec):
        v = r.get(key)
        return "-" if v is None else format(v, spec)
    cells = [[h for _, h, _ in COLUMNS]] + [[cell(r, k, s) for k, _, s in COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(COLUMNS))]
    for j, row in enumerate(cells):
        print("  ".join(v.rjust(w) for v, w in zip(row, widths)))
        if j == 0:
            print("  ".join("-" * w for w in widths))


def plot(runs: list[dict], nominal_ms: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for d in runs:
        err = (d["period"] - nominal_ms) * 1e3
        if err.size:
            ax1.hist(err, bins=120, histtype="step", linewidth=1.5, log=True,
                     label=f"{d['inference']} (std {err.std():.1f} us)")
        dl = d["delay"][np.isfinite(d["delay"])]
        if dl.size:
            ax2.hist(dl, bins=120, histtype="step", linewidth=1.5, log=True,
                     label=f"{d['inference']} (mean {dl.mean():.3f} ms)")
    ax1.set_xlabel(f"period error vs {nominal_ms:g} ms [us]")
    ax1.set_ylabel("actions")
    ax1.set_title("Control period jitter (PL counters)")
    ax2.set_xlabel("sample -> action delay [ms]")
    ax2.set_title("Sensing-to-actuation delay")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot -> {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="fine-tune run folders")
    p.add_argument("--mode", choices=("all", "train", "eval"), default="all",
                   help="episodes to include (eval = deterministic, fixed weights)")
    p.add_argument("--learner", choices=("all", "busy", "idle", "split"), default="all",
                   help="filter on whether the learner trained during the episode "
                        "(finetune_async.py --eval-learner). split = one busy and one "
                        "idle row per run, the training-independence test.")
    p.add_argument("--skip-episodes", type=int, default=0,
                   help="drop the first N episodes of every run (warm-up)")
    p.add_argument("--control-freq", type=float, default=None,
                   help="nominal rate [Hz] (default: config.json of the first run)")
    p.add_argument("--plot", type=Path, default=None, help="histogram PNG")
    p.add_argument("--out", type=Path, default=None,
                   help="write the table as .csv or .json")
    args = p.parse_args(argv)

    learners = ("busy", "idle") if args.learner == "split" else (args.learner,)
    runs = [load_run(r, args.mode, args.skip_episodes, lr)
            for r in args.runs for lr in learners]
    hz = args.control_freq or next((d["control_hz"] for d in runs if d["control_hz"]), None)
    if hz is None:
        print("ERROR: no control_freq_hz in config.json; pass --control-freq", file=sys.stderr)
        return 2
    rates = {d["control_hz"] for d in runs if d["control_hz"]}
    if len(rates) > 1:
        print(f"WARNING: runs use different control rates {sorted(rates)}; "
              f"errors are vs {hz:g} Hz", file=sys.stderr)
    nominal_ms = 1e3 / float(hz)
    rows = [summarize(d, nominal_ms) for d in runs]
    print(f"Loop timing, nominal period {nominal_ms:g} ms, episodes: {args.mode}"
          + (f", first {args.skip_episodes} skipped" if args.skip_episodes else "") + "\n")
    print_table(rows)
    if args.out:
        if args.out.suffix == ".json":
            args.out.write_text(json.dumps(rows, indent=2))
        else:
            present = {k for r in rows for k in r}
            fields = ["run"] + [k for k, _, _ in COLUMNS if k in present]
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        print(f"table -> {args.out}")
    if args.plot:
        plot(runs, nominal_ms, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

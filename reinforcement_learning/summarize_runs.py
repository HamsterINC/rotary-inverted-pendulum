"""One table across fine-tune runs: convergence, loop timing, learner load, balance.

    python summarize_runs.py                       # every run folder under runs/
    python summarize_runs.py runs/async_cpu_X runs/async_baremetal_Y
    python summarize_runs.py runs --balance-mode all --streak 3 --out summary.csv
    python summarize_runs.py --balance-mode last --last-n 10   # final performance
    python summarize_runs.py --balance-mode last --latex       # paste into LaTeX (booktabs)

Reads what finetune_async.py writes into each run folder (no rig needed):

    metrics.csv      per episode: mode (eval/train), balanced, time_to_balance_s,
                     frac_upright, dropped, rig_minutes, grad_steps
    step_timing.csv  per action, PL-measured: period_ms, delay_ms, compute_us
    timing.csv       per episode: learner_train_calls (+ fpga_* fallbacks)
    source.json / config.json   inference mode, control rate

Columns
    converged ep / min  first episode (of --converge-mode, default eval) that
                        caught and held the pendulum without dropping it, with
                        --streak consecutive such episodes (finetune_metrics
                        definition of "balanced"); rig minutes at that point
    jitter %            std / mean of the action->action period, all actions
    jitter nom %        same, only periods within 1 ms of nominal. Lost or
                        mis-latched strobes (PL clock-crossing bug in the old
                        low-level bitstream) show up as rare 30-40 ms periods
                        that dominate the plain std; this column ignores them
    jitter ms           std of the period (all actions)
    delay ms            mean sample -> action delay
    train calls/ep      mean learner_train_calls per episode (timing.csv)
    grad steps/ep       mean gradient steps per episode (metrics.csv)
    balanced %          share of --balance-mode episodes that caught and held
    upright %           mean frac_upright of those episodes (time with |theta|
                        under the threshold, whole episode)
    t balance s         mean time_to_balance_s over the episodes that caught
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOMINAL_TOL_MS = 1.0


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def find_runs(paths: list[Path]) -> list[Path]:
    """Run folders = folders with a metrics.csv; a parent folder expands to its runs."""
    runs: list[Path] = []
    for p in paths:
        if (p / "metrics.csv").exists():
            runs.append(p)
        elif p.is_dir():
            runs += sorted(d for d in p.iterdir() if (d / "metrics.csv").exists())
        else:
            print(f"WARNING: {p} is not a run folder", file=sys.stderr)
    return runs


def convergence(metrics: list[dict], mode: str, streak: int) -> tuple[float, float]:
    """(episode, rig_minutes) of the first `streak` consecutive held episodes."""
    eps = [m for m in metrics if mode == "all" or m.get("mode") == mode]
    run = 0
    for m in eps:
        held = _num(m.get("balanced")) == 1 and _num(m.get("dropped") or 0) == 0
        run = run + 1 if held else 0
        if run >= streak:
            return _num(m["episode"]), _num(m.get("rig_minutes"))
    return math.nan, math.nan


def timing_stats(run: Path, timing: list[dict], control_hz: float | None) -> dict:
    rows = _read_csv(run / "step_timing.csv")
    if rows:
        p = np.array([_num(r["period_ms"]) for r in rows])
        d = np.array([_num(r.get("delay_ms")) for r in rows])
        c = np.array([_num(r.get("compute_us")) for r in rows])
        p = p[np.isfinite(p)]
        nominal = 1e3 / control_hz if control_hz else float(np.median(p)) if p.size else math.nan
        nom = p[np.abs(p - nominal) < NOMINAL_TOL_MS]
        return {
            "jitter_pct": 100 * p.std() / p.mean() if p.size else math.nan,
            "jitter_nom_pct": 100 * nom.std() / nom.mean() if nom.size else math.nan,
            "jitter_ms": p.std() if p.size else math.nan,
            "delay_ms": np.nanmean(d) if np.isfinite(d).any() else math.nan,
            "compute_us": np.nanmean(c) if np.isfinite(c).any() else math.nan,
            "n_actions": int(p.size),
            "outliers": int(p.size - nom.size),
        }
    # older runs: per-episode FPGA summaries only (mean of per-episode values)
    col = lambda k: np.array([_num(r.get(k)) for r in timing])           # noqa: E731
    jp, ps, dl = col("fpga_jitter_pct"), col("fpga_period_std_ms"), col("fpga_delay_ms")
    mean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else math.nan  # noqa: E731
    return {"jitter_pct": mean(jp), "jitter_nom_pct": math.nan, "jitter_ms": mean(ps),
            "delay_ms": mean(dl), "compute_us": math.nan, "n_actions": 0, "outliers": 0}


def summarize(run: Path, args) -> dict:
    src = _read_json(run / "source.json")
    cfg = _read_json(run / "config.json")
    metrics = _read_csv(run / "metrics.csv")
    timing = _read_csv(run / "timing.csv")
    control_hz = cfg.get("control_freq_hz")

    conv_ep, conv_min = convergence(metrics, args.converge_mode, args.streak)
    if args.balance_mode == "last":
        bal = metrics[-args.last_n:]          # final episodes, eval and train alike
    else:
        bal = [m for m in metrics if args.balance_mode == "all" or m.get("mode") == args.balance_mode]
    balanced = np.array([_num(m.get("balanced")) == 1 and _num(m.get("dropped") or 0) == 0
                         for m in bal], dtype=float)
    upright = np.array([_num(m.get("frac_upright")) for m in bal])
    ttb = np.array([_num(m.get("time_to_balance_s")) for m in bal])
    calls = np.array([_num(r.get("learner_train_calls")) for r in timing])
    grads = np.array([_num(m.get("grad_steps")) for m in metrics])
    nanmean = lambda a: float(np.nanmean(a)) if a.size and np.isfinite(a).any() else math.nan  # noqa: E731

    out = {
        "run": run.name,
        "inference": src.get("inference", "?"),
        "episodes": len(metrics),
        "converged_ep": conv_ep,
        "converged_min": conv_min,
        **timing_stats(run, timing, control_hz),
        "train_calls_per_ep": nanmean(calls),
        "grad_steps_per_ep": nanmean(grads),
        "balanced_pct": 100 * balanced.mean() if balanced.size else math.nan,
        "upright_pct": 100 * nanmean(upright),
        "time_to_balance_s": nanmean(ttb),
        "n_balance_eps": len(bal),
    }
    return out


COLUMNS = [
    ("inference", "inference", "s"), ("episodes", "eps", "d"),
    ("converged_ep", "conv ep", ".0f"), ("converged_min", "conv min", ".1f"),
    ("jitter_pct", "jitter %", ".2f"), ("jitter_nom_pct", "jit nom %", ".3f"),
    ("jitter_ms", "jitter ms", ".3f"), ("delay_ms", "delay ms", ".3f"),
    ("outliers", "outliers", "d"),
    ("train_calls_per_ep", "calls/ep", ".1f"), ("grad_steps_per_ep", "grads/ep", ".1f"),
    ("balanced_pct", "balanced %", ".0f"), ("upright_pct", "upright %", ".1f"),
    ("time_to_balance_s", "t bal s", ".2f"),
]


def cell(row: dict, key: str, spec: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return format(v, spec)


def print_table(rows: list[dict], markdown: bool) -> None:
    head = ["run"] + [h for _, h, _ in COLUMNS]
    body = [[r["run"]] + [cell(r, k, s) for k, _, s in COLUMNS] for r in rows]
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join("---" for _ in head) + "|")
        for b in body:
            print("| " + " | ".join(b) + " |")
        return
    widths = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    print("  ".join(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(head, widths))))
    print("  ".join("-" * w for w in widths))
    for b in body:
        print("  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(b, widths))))


INFERENCE_LABELS = {"cpu": "CPU", "pl": "PL", "baremetal": "Bare metal"}

# (group, label, key, format); key "off_nominal_pct" is derived
LATEX_ROWS = [
    ("Convergence", "First held episode", "converged_ep", ".0f"),
    ("Convergence", "Rig time to convergence [min]", "converged_min", ".2f"),
    ("Loop timing", r"Period jitter, all actions [\%]", "jitter_pct", ".2f"),
    ("Loop timing", r"Period jitter, nominal periods [\%]", "jitter_nom_pct", ".2f"),
    ("Loop timing", "Period jitter (std) [ms]", "jitter_ms", ".3f"),
    ("Loop timing", "Sample-to-action delay [ms]", "delay_ms", ".3f"),
    ("Loop timing", r"Inference time [$\mu$s]", "compute_us", ".0f"),
    ("Loop timing", r"Off-nominal periods [\%]", "off_nominal_pct", ".2f"),
    ("Learner", "Training calls per episode", "train_calls_per_ep", ".2f"),
    ("Learner", "Gradient steps per episode", "grad_steps_per_ep", ".2f"),
    ("Balance", r"Episodes balanced [\%]", "balanced_pct", ".0f"),
    ("Balance", r"Time upright [\%]", "upright_pct", ".1f"),
    ("Balance", "Time to balance [s]", "time_to_balance_s", ".2f"),
]


def print_latex(rows: list[dict], args) -> None:
    """booktabs table, one column per run (labelled by inference mode when unique)."""
    for r in rows:
        n = r.get("n_actions") or 0
        r["off_nominal_pct"] = 100.0 * r.get("outliers", 0) / n if n else math.nan
    modes = [r["inference"] for r in rows]
    if len(set(modes)) == len(modes):
        heads = [INFERENCE_LABELS.get(m, m) for m in modes]
    else:
        heads = [r["run"].replace("_", r"\_") for r in rows]
    if args.balance_mode == "last":
        bal = f"last {args.last_n} episodes"
    else:
        bal = f"{args.balance_mode} episodes"
    conv = f"{args.converge_mode} episode" + (f", {args.streak} in a row" if args.streak > 1 else "")
    eps = sorted({r["episodes"] for r in rows})
    eps_txt = f"{eps[0]} episodes each" if len(eps) == 1 else "episodes: " + ", ".join(map(str, eps))

    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Fine-tuning on the rig per inference mode (" + eps_txt
          + r"). Convergence: first held " + conv + r"; balance over the " + bal
          + rf"; nominal periods within {NOMINAL_TOL_MS:g}\,ms of the control period.}}")
    print(r"  \label{tab:inference_comparison}")
    print(r"  \begin{tabular}{l" + "r" * len(rows) + "}")
    print(r"    \toprule")
    print("    & " + " & ".join(heads) + r" \\")
    group = None
    for g, label, key, spec in LATEX_ROWS:
        if g != group:
            print(r"    \midrule")
            print(rf"    \multicolumn{{{len(rows) + 1}}}{{l}}{{\textit{{{g}}}}} \\")
            group = g
        cells = [cell(r, key, spec) for r in rows]
        print(f"    {label} & " + " & ".join("--" if c == "-" else c for c in cells) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="*", default=[HERE / "runs"],
                   help="run folders, or folders containing them (default: runs/)")
    p.add_argument("--converge-mode", choices=("eval", "train", "all"), default="eval",
                   help="episodes that count for convergence (default eval: deterministic policy)")
    p.add_argument("--streak", type=int, default=1,
                   help="consecutive held episodes required for convergence")
    p.add_argument("--balance-mode", choices=("eval", "train", "all", "last"), default="eval",
                   help="episodes averaged for balanced / upright / time to balance "
                        "(last: the final --last-n episodes, eval and train)")
    p.add_argument("--last-n", type=int, default=10,
                   help="episodes for --balance-mode last (default 10)")
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    p.add_argument("--latex", action="store_true",
                   help="print a LaTeX (booktabs) table, one column per run")
    p.add_argument("--out", type=Path, default=None, help="also write .csv or .json")
    args = p.parse_args(argv)

    runs = find_runs(args.runs)
    if not runs:
        print("ERROR: no run folders (with metrics.csv) found", file=sys.stderr)
        return 2
    rows = [summarize(r, args) for r in runs]
    which = (f"the last {args.last_n}" if args.balance_mode == "last"
             else args.balance_mode)
    if args.latex:
        print_latex(rows, args)
    else:
        print(f"convergence: first {args.streak} held {args.converge_mode} episode(s); "
              f"balance stats over {which} episodes; jitter nom = periods within "
              f"{NOMINAL_TOL_MS:g} ms of nominal\n")
        print_table(rows, args.markdown)
    if args.out:
        if args.out.suffix == ".json":
            args.out.write_text(json.dumps(rows, indent=2, default=float))
        else:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"\ntable -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-episode convergence figure: one curve per platform (CPU Linux, CPU bare
metal, PL), from the fine-tune runs' metrics.csv.

    python plot_convergence.py saved_runs --out convergence.pdf
    python plot_convergence.py saved_runs --metric return --smooth 3
    python plot_convergence.py saved_runs --metric both --out convergence_both.pdf

Panels (top to bottom; with --metric both, one column per metric):
    deterministic evals  every --eval-every'th episode, fixed weights, the way
                         the policy is deployed (episode 1 = sim-only baseline)
    training episodes    exploring policy while it learns, rolling mean over
                         --smooth training episodes

Metrics:
    frac_upright  fraction of the episode with |θ| below the balance threshold
                  ("time upright", 0-1; default)
    return        episode return (same reward as in sim, Training/reward.py)
    mean_abs_action_balanced
                  mean |action| while balanced (control effort; episodes that
                  never balanced have no value and leave a gap)

Several runs of one platform (seeds): the line is the mean over the runs that
reached that episode, the band / error bars span min to max. Runs with fewer
episodes simply stop contributing. The platform comes from source.json
("inference"). Writes PDF (vector, for LaTeX) or PNG by --out's suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PLATFORMS = {"cpu": "CPU (Linux)", "baremetal": "CPU (bare metal)", "pl": "PL"}
PLATFORM_ORDER = ("cpu", "baremetal", "pl")
# Validated categorical slots 1-3 (dataviz reference palette, light mode) plus
# a secondary encoding (marker + line style) so identity is never colour alone
# and the figure survives greyscale printing.
STYLE = {
    "cpu":       dict(color="#2a78d6", marker="o", ls="-"),
    "baremetal": dict(color="#eb6834", marker="s", ls="--"),
    "pl":        dict(color="#1baf7a", marker="^", ls=":"),
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {"frac_upright": ("Time upright [fraction]", (0.0, 1.0)),
           "return": ("Episode return", None),
           "mean_abs_action_balanced": ("Mean |a| while balanced", (0.0, 1.0))}


def find_runs(paths: list[Path]) -> list[Path]:
    runs = []
    for p in paths:
        if (p / "metrics.csv").exists():
            runs.append(p)
        elif p.is_dir():
            runs += sorted(d for d in p.iterdir() if (d / "metrics.csv").exists())
        else:
            print(f"WARNING: {p} has no metrics.csv, skipped", file=sys.stderr)
    return runs


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_run(run: Path) -> dict:
    rows = list(csv.DictReader(open(run / "metrics.csv", newline="")))
    inference = "?"
    if (run / "source.json").exists():
        inference = json.loads((run / "source.json").read_text()).get("inference", "?")
    out = {"run": run.name, "inference": inference}
    for mode in ("train", "eval"):
        sel = [r for r in rows if r["mode"] == mode]
        out[mode] = {"episode": np.array([int(r["episode"]) for r in sel]),
                     **{m: np.array([_f(r.get(m)) for r in sel]) for m in METRICS}}
    return out


def rolling(y: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w finite values (NaN-safe), same length."""
    if w <= 1:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        win = y[max(0, i - w + 1):i + 1]
        win = win[np.isfinite(win)]
        if win.size:
            out[i] = win.mean()
    return out


def across_runs(series: list[tuple[np.ndarray, np.ndarray]]):
    """[(episodes, values)] per run -> episodes, mean, min, max, n over runs."""
    eps = np.array(sorted({int(e) for x, _ in series for e in x}))
    table = np.full((len(series), len(eps)), np.nan)
    for i, (x, y) in enumerate(series):
        idx = np.searchsorted(eps, x)
        table[i, idx] = y
    n = np.sum(np.isfinite(table), axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(table, axis=0)
            lo, hi = np.nanmin(table, axis=0), np.nanmax(table, axis=0)
    return eps, mean, lo, hi, n


def style_axis(ax, ylabel: str, ylim) -> None:
    ax.set_ylabel(ylabel, color=TEXT)
    if ylim:
        ax.set_ylim(ylim[0] - 0.03, ylim[1] + 0.03)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, width=0.6)


def direct_label(ax, x, y, text) -> None:
    """Queue the platform name for the end of its curve (placed by place_labels)."""
    ok = np.isfinite(y)
    if ok.any():
        i = np.where(ok)[0][-1]
        ax._pending_labels = getattr(ax, "_pending_labels", []) + [(x[i], y[i], text)]


def place_labels(ax, min_gap_frac: float = 0.075) -> None:
    """Draw queued end labels in text ink, nudged apart vertically so names of
    curves that end at the same height don't overlap."""
    items = sorted(getattr(ax, "_pending_labels", []), key=lambda t: t[1])
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ys = [it[1] for it in items]
    for k in range(1, len(ys)):                 # push up until spaced
        ys[k] = max(ys[k], ys[k - 1] + gap)
    overflow = ys[-1] - (hi - gap / 2)
    if overflow > 0:                            # shift the stack back down
        ys = [y - overflow for y in ys]
    for (x, y, text), ly in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(x, ly), textcoords="data",
                    va="center", ha="left", fontsize=7, color=TEXT,
                    xycoords="data", annotation_clip=False,
                    transform=ax.transData)
    ax._pending_labels = []


def draw(ax_eval, ax_train, data: dict, metric: str, smooth: int,
         show_runs: bool, label_ends: bool) -> None:
    ylabel, ylim = METRICS[metric]
    for inf, runs in data.items():
        st, name = STYLE.get(inf, dict(color=MUTED, marker="x", ls="-")), PLATFORMS.get(inf, inf)
        nr = len(runs)
        lab = f"{name} (n={nr})" if nr > 1 else name

        # Evals: mean over runs, min-max error bars, markers.
        ev = [(r["eval"]["episode"], r["eval"][metric]) for r in runs]
        x, m, lo, hi, _ = across_runs(ev)
        if x.size:
            yerr = np.vstack([m - lo, hi - m]) if nr > 1 else None
            ax_eval.errorbar(x, m, yerr=yerr, color=st["color"], marker=st["marker"],
                             ls=st["ls"], lw=1.5, ms=5, capsize=2, elinewidth=0.8,
                             mec="white", mew=0.8, label=lab)
            if label_ends:
                direct_label(ax_eval, x, m, name)

        # Training: per-run rolling mean, then mean + min-max band over runs.
        tr = [(r["train"]["episode"], rolling(r["train"][metric], smooth)) for r in runs]
        x, m, lo, hi, _ = across_runs(tr)
        if x.size:
            if show_runs:
                for xr, yr in tr:
                    ax_train.plot(xr, yr, color=st["color"], lw=0.6, alpha=0.35, ls=st["ls"])
            if nr > 1:
                ax_train.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)
            ax_train.plot(x, m, color=st["color"], ls=st["ls"], lw=1.6, label=lab,
                          marker=st["marker"], ms=3.5, markevery=max(1, len(x) // 8),
                          mec="white", mew=0.6)
            if label_ends:
                direct_label(ax_train, x, m, name)

    style_axis(ax_eval, ylabel, ylim)
    style_axis(ax_train, ylabel, ylim)
    ax_eval.set_title("Deterministic evaluation", fontsize=9, color=TEXT, loc="left")
    ax_train.set_title("Training episodes" + (f" (rolling mean over {smooth})" if smooth > 1 else ""),
                       fontsize=9, color=TEXT, loc="left")
    ax_train.set_xlabel("Episode on the rig", color=TEXT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+",
                   help="fine-tune run folders, or folders containing them")
    p.add_argument("--metric", choices=("frac_upright", "return", "mean_abs_action_balanced",
                                        "both"), default="frac_upright",
                   help="mean_abs_action_balanced = mean |action| over the balanced part "
                        "(control effort / chatter; NaN when the episode never balanced)")
    p.add_argument("--smooth", type=int, default=1,
                   help="rolling mean over N training episodes (1 = raw, default 5)")
    p.add_argument("--max-episode", type=int, default=30,
                   help="cut the x-axis here (e.g. to the shortest platform's run)")
    p.add_argument("--show-runs", action="store_true",
                   help="also draw each run's own training curve, thin")
    p.add_argument("--no-direct-labels", action="store_true",
                   help="legend only, no platform names at the curve ends")
    p.add_argument("--width", type=float, default=7.0,
                   help="figure width in inches (7 = two-column \\textwidth, 3.5 = column)")
    p.add_argument("--out", type=Path, default=Path("convergence.pdf"),
                   help="output file (.pdf or .png)")
    args = p.parse_args(argv)

    runs = [load_run(r) for r in find_runs(args.runs)]
    if not runs:
        print("ERROR: no run folder with metrics.csv found", file=sys.stderr)
        return 2
    order = {k: i for i, k in enumerate(PLATFORM_ORDER)}
    data: dict[str, list[dict]] = {}
    for r in sorted(runs, key=lambda r: (order.get(r["inference"], 99), r["run"])):
        data.setdefault(r["inference"], []).append(r)
    if args.max_episode:
        for rs in data.values():
            for r in rs:
                for mode in ("train", "eval"):
                    keep = r[mode]["episode"] <= args.max_episode
                    r[mode] = {k: v[keep] for k, v in r[mode].items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "pdf.fonttype": 42})

    metrics = ["frac_upright", "return"] if args.metric == "both" else [args.metric]
    fig, axes = plt.subplots(2, len(metrics), sharex=True, squeeze=False,
                             figsize=(args.width, 4.4))
    for j, metric in enumerate(metrics):
        draw(axes[0, j], axes[1, j], data, metric, args.smooth, args.show_runs,
             not args.no_direct_labels)
    # Room on the right for the direct labels.
    xmax = max(int(r[m]["episode"].max()) for rs in data.values() for r in rs
               for m in ("train", "eval") if r[m]["episode"].size)
    for ax in axes.flat:
        ax.set_xlim(0, xmax * (1.16 if not args.no_direct_labels else 1.02))
        # labels start just right of the curve end
        pend = getattr(ax, "_pending_labels", [])
        ax._pending_labels = [(x + 0.012 * xmax, y, t) for x, y, t in pend]
        place_labels(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"figure -> {args.out}")
    for inf, rs in data.items():
        print(f"  {PLATFORMS.get(inf, inf):<17} " + ", ".join(r["run"] for r in rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

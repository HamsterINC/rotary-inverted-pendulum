"""Control-loop timing per platform, during training vs eval, in one table.

    python summarize_timings.py runs/v2_cpu_s0 runs/v2_cpu_s1 runs/v2_pl_s0 runs/v2_baremetal_s0
    python summarize_timings.py runs --latex
    python summarize_timings.py runs --split-eval --tests --plot timing.png --out timing.csv

Reads every run's step_timing.csv (one row per action, PL-measured period and
sample->action delay, written by finetune_async.py) and fills ONE table:

    rows     platform (CPU Linux / CPU bare metal / PL) x condition (train / eval)
    columns  period mean, period std, jitter %, late ticks %,
             delay mean, delay std, delay p99, delay max

Each run gives one value per cell (pooled over all actions of that condition
in that run). With several runs of a platform (e.g. seeds) the cell shows
mean ± std across those runs; with one run, the value alone.

    late ticks  share of periods longer than nominal + --late-ms (default 1 ms)
    jitter %    period std / period mean

Conditions:
    train       exploring policy, learner training, mid-episode weight sync
    eval        deterministic, frozen weights. With --split-eval the evals are
                split by the 'learner' column of step_timing.csv:
                  eval busy  learner kept training during the episode
                  eval idle  no train() call overlapped the episode
                (finetune_async.py --eval-learner alternate). Eval busy vs eval
                idle isolates the training WORKLOAD; train vs eval adds weight
                sync and exploration.

--tests adds per-run Mann-Whitney U tests on per-episode values (period std,
mean delay, delay p99), since consecutive actions are not independent samples
but episodes are. A difference only counts as dependence when it is both
significant (p < --alpha) and larger than --min-effect-pct.
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
COND_ORDER = ("train", "eval", "eval busy", "eval idle", "eval ?")

# key, console header, latex header, format
COLUMNS = [
    ("period_ms", "period ms", r"Period [\si{\milli\second}]", ".3f"),
    ("period_std_us", "std us", r"Std [\si{\micro\second}]", ".0f"),
    ("jitter_pct", "jitter %", r"Jitter [\%]", ".2f"),
    ("late_pct", "late %", r"Late [\%]", ".2f"),
    ("delay_ms", "delay ms", r"Delay [\si{\milli\second}]", ".2f"),
    ("delay_std_us", "d.std us", r"Std [\si{\micro\second}]", ".0f"),
    ("delay_p99_ms", "d.p99 ms", r"p99 [\si{\milli\second}]", ".2f"),
    ("delay_max_ms", "d.max ms", r"Max [\si{\milli\second}]", ".2f"),
]

COMPARISONS = (
    ("eval busy", "eval idle", "training workload"),
    ("train", "eval busy", "weight sync + exploration"),
    ("train", "eval idle", "total (learning vs not)"),
    ("train", "eval", "train vs eval"),
)
EP_METRICS = (("period_std_us", "period std us"), ("delay_mean_ms", "delay ms"),
              ("delay_p99_ms", "delay p99 ms"))


# --------------------------------------------------------------------------- io

def find_runs(paths: list[Path]) -> list[Path]:
    runs = []
    for p in paths:
        if (p / "step_timing.csv").exists():
            runs.append(p)
        elif p.is_dir():
            runs += sorted(d for d in p.iterdir() if (d / "step_timing.csv").exists())
        else:
            print(f"WARNING: {p} has no step_timing.csv, skipped", file=sys.stderr)
    return runs


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_run(run: Path, skip_episodes: int, skip_steps: int, split_eval: bool) -> dict:
    """Per-action rows grouped into episodes, each tagged with its condition."""
    episodes: dict[int, dict] = {}
    inference = None
    with open(run / "step_timing.csv", newline="") as f:
        for row in csv.DictReader(f):
            ep = int(row["episode"])
            if ep <= skip_episodes or int(row["step"]) <= skip_steps:
                continue
            inference = inference or row.get("inference")
            cond = row["mode"]
            if cond == "eval" and split_eval:
                cond = f"eval {row.get('learner') or '?'}"
            e = episodes.setdefault(ep, {"cond": cond, "period": [], "delay": []})
            e["period"].append(_f(row["period_ms"]))
            e["delay"].append(_f(row.get("delay_ms")))
    for e in episodes.values():
        e["period"] = np.asarray(e["period"], dtype=float)
        e["delay"] = np.asarray(e["delay"], dtype=float)
    hz = None
    if (run / "config.json").exists():
        hz = json.loads((run / "config.json").read_text()).get("control_freq_hz")
    if inference is None and (run / "source.json").exists():
        inference = json.loads((run / "source.json").read_text()).get("inference")
    return {"run": run.name, "inference": inference or "?", "control_hz": hz,
            "episodes": episodes}


# ---------------------------------------------------------------------- metrics

def pooled(eps: list[dict], nominal_ms: float, late_ms: float) -> dict:
    """One run, one condition: metrics over all its actions."""
    p = np.concatenate([e["period"] for e in eps])
    d = np.concatenate([e["delay"] for e in eps])
    p, d = p[np.isfinite(p)], d[np.isfinite(d)]
    out = {"episodes": len(eps), "actions": int(p.size)}
    if p.size:
        out.update(period_ms=p.mean(), period_std_us=p.std() * 1e3,
                   jitter_pct=100 * p.std() / p.mean(),
                   late_pct=100 * np.mean(p > nominal_ms + late_ms))
    if d.size:
        out.update(delay_ms=d.mean(), delay_std_us=d.std() * 1e3,
                   delay_p99_ms=np.percentile(d, 99), delay_max_ms=d.max())
    return out


def per_episode(e: dict) -> dict:
    p = e["period"][np.isfinite(e["period"])]
    d = e["delay"][np.isfinite(e["delay"])]
    return {"period_std_us": p.std() * 1e3 if p.size else np.nan,
            "delay_mean_ms": d.mean() if d.size else np.nan,
            "delay_p99_ms": np.percentile(d, 99) if d.size else np.nan}


def aggregate(per_run: list[dict]) -> dict:
    """Several runs of one platform + condition -> mean, std (ddof=1) per metric."""
    out = {"runs": len(per_run),
           "episodes": int(sum(r["episodes"] for r in per_run)),
           "actions": int(sum(r["actions"] for r in per_run))}
    for key, *_ in COLUMNS:
        v = np.array([r.get(key, np.nan) for r in per_run], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = float(v.mean()) if v.size else float("nan")
        out[key + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return out


def mann_whitney(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def compare(groups: dict[str, list[dict]], alpha: float, min_effect_pct: float) -> list[dict]:
    rows = []
    for a, b, label in COMPARISONS:
        if a not in groups or b not in groups:
            continue
        ea = [per_episode(e) for e in groups[a]]
        eb = [per_episode(e) for e in groups[b]]
        row = {"comparison": f"{a} vs {b}", "tests": label, "n": f"{len(ea)}/{len(eb)}"}
        verdicts = []
        for key, _ in EP_METRICS:
            va = np.array([x[key] for x in ea])
            vb = np.array([x[key] for x in eb])
            pval = mann_whitney(va, vb)
            ma, mb = float(np.nanmedian(va)), float(np.nanmedian(vb))
            rel = 100 * (ma - mb) / mb if mb else float("nan")
            row[f"{key}_a"], row[f"{key}_b"] = ma, mb
            row[f"{key}_p"], row[f"{key}_rel_pct"] = pval, rel
            if np.isfinite(pval):
                verdicts.append(not (pval < alpha and abs(rel) > min_effect_pct))
        row["independent"] = "-" if not verdicts else ("yes" if all(verdicts) else "no")
        rows.append(row)
    return rows


# ----------------------------------------------------------------------- output

def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and not np.isfinite(v))


def _console_pm() -> str:
    try:
        "±".encode(sys.stdout.encoding or "ascii")
        return " ± "
    except (UnicodeEncodeError, LookupError):
        return " +- "          # e.g. a cp1252 Windows console


PM = _console_pm()


def cell(r: dict, key: str, spec: str, pm: str = PM) -> str:
    v, sd = r.get(key), r.get(key + "_sd")
    if not _ok(v):
        return "-"
    return format(v, spec) + (pm + format(sd, spec) if _ok(sd) else "")


def print_table(header: list[str], rows: list[list[str]], left: int = 2) -> None:
    cells = [header] + rows
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    for j, r in enumerate(cells):
        print("  " + "  ".join(v.ljust(w) if i < left else v.rjust(w)
                               for i, (v, w) in enumerate(zip(r, widths))))
        if j == 0:
            print("  " + "  ".join("-" * w for w in widths))


def print_summary(table: list[dict]) -> None:
    head = ["platform", "condition", "runs", "eps"] + [h for _, h, _, _ in COLUMNS]
    rows, last = [], None
    for r in table:
        plat = r["platform"] if r["platform"] != last else ""
        last = r["platform"]
        rows.append([plat, r["condition"], str(r["runs"]), str(r["episodes"])]
                    + [cell(r, k, s) for k, _, _, s in COLUMNS])
    print_table(head, rows)


def print_latex(table: list[dict], late_ms: float) -> None:
    n_runs = sorted({r["runs"] for r in table})
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Control-loop timing per platform during on-device learning "
          r"(train) and deterministic evaluation (eval), measured by the PL counters. "
          rf"Late: period $>$ nominal $+\;{late_ms:g}$\,ms. "
          + (r"Mean $\pm$ std over runs." if max(n_runs) > 1 else "") + "}")
    print(r"  \label{tab:timing}")
    print(r"  \small")
    print(r"  \begin{tabular}{ll" + "r" * len(COLUMNS) + "}")
    print(r"    \toprule")
    print(r"    & & \multicolumn{4}{c}{Control period} & \multicolumn{4}{c}{Sample-to-action delay} \\")
    print(r"    \cmidrule(lr){3-6} \cmidrule(lr){7-10}")
    print("    Platform & Condition & " + " & ".join(h for _, _, h, _ in COLUMNS) + r" \\")
    print(r"    \midrule")
    last = None
    for i, r in enumerate(table):
        if last is not None and r["platform"] != last:
            print(r"    \midrule")
        plat = r["platform"] if r["platform"] != last else ""
        last = r["platform"]
        vals = [cell(r, k, s, r" $\pm$ ") for k, _, _, s in COLUMNS]
        print(f"    {plat} & {r['condition']} & " + " & ".join(vals) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


def print_tests(results: list[dict]) -> None:
    for res in results:
        if not res["comparisons"]:
            continue
        print(f"\n  {res['platform']} - {res['run']}")
        head = ["comparison", "tests", "n"]
        for _, h in EP_METRICS:
            head += [h, "p"]
        head.append("independent")
        rows = []
        for c in res["comparisons"]:
            r = [c["comparison"], c["tests"], c["n"]]
            for key, _ in EP_METRICS:
                a, b, rel = c[key + "_a"], c[key + "_b"], c[key + "_rel_pct"]
                r += [f"{a:.2f} vs {b:.2f} ({rel:+.0f}%)" if _ok(rel) else "-",
                      f"{c[key + '_p']:.3f}" if _ok(c[key + "_p"]) else "-"]
            r.append(c["independent"])
            rows.append(r)
        print_table(head, rows)


def plot(results: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3.6 * len(results)), squeeze=False)
    for (ax1, ax2), res in zip(axes, results):
        labels, stds, delays = [], [], []
        for cond, eps in res["groups"].items():
            pe = [per_episode(e) for e in eps]
            labels.append(f"{cond}\n(n={len(pe)})")
            stds.append([x["period_std_us"] for x in pe])
            delays.append([x["delay_mean_ms"] for x in pe])
        for ax, data, ylabel in ((ax1, stds, "period std per episode [us]"),
                                 (ax2, delays, "mean delay per episode [ms]")):
            ax.boxplot(data)
            ax.set_xticks(range(1, len(labels) + 1), labels)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{res['platform']} - {res['run']}", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot -> {path}")


def write_out(table: list[dict], results: list[dict], path: Path) -> None:
    flat = [{"kind": "summary", **r} for r in table]
    for res in results:
        for cond, m in res["per_cond"].items():
            flat.append({"kind": "run", "platform": res["platform"], "condition": cond,
                         "run": res["run"], **m})
    if path.suffix == ".json":
        path.write_text(json.dumps(flat, indent=2, default=float))
    else:
        fields: list[str] = []
        for r in flat:
            fields += [k for k in r if k not in fields]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)
    print(f"table -> {path}")


# ------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+",
                   help="fine-tune run folders, or folders containing them")
    p.add_argument("--split-eval", action="store_true",
                   help="split eval into 'eval busy' / 'eval idle' (learner column)")
    p.add_argument("--skip-episodes", type=int, default=0,
                   help="drop the first N episodes of every run (warm-up)")
    p.add_argument("--skip-steps", type=int, default=1,
                   help="drop the first N actions of every episode (default 1: the "
                        "first period spans the episode start)")
    p.add_argument("--late-ms", type=float, default=1.0,
                   help="a tick is late when period > nominal + this [ms] (default 1)")
    p.add_argument("--control-freq", type=float, default=None,
                   help="nominal rate [Hz] (default: each run's config.json, else 50)")
    p.add_argument("--tests", action="store_true",
                   help="also print per-run Mann-Whitney tests between conditions")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-effect-pct", type=float, default=10.0,
                   help="--tests: a significant difference only breaks independence "
                        "when the medians differ by more than this (default 10 %%)")
    p.add_argument("--plot", type=Path, default=None,
                   help="per-episode boxplots per run and condition (PNG)")
    p.add_argument("--out", type=Path, default=None,
                   help="write summary + per-run rows as .csv or .json")
    p.add_argument("--latex", action="store_true", help="print a booktabs table (needs siunitx)")
    args = p.parse_args(argv)

    runs = find_runs(args.runs)
    if not runs:
        print("ERROR: no run folder with step_timing.csv found", file=sys.stderr)
        return 2

    results = []
    for run in runs:
        d = load_run(run, args.skip_episodes, args.skip_steps, args.split_eval)
        nominal_ms = 1e3 / float(args.control_freq or d["control_hz"] or 50.0)
        groups: dict[str, list[dict]] = {}
        for ep in sorted(d["episodes"]):
            groups.setdefault(d["episodes"][ep]["cond"], []).append(d["episodes"][ep])
        groups = {c: groups[c] for c in sorted(
            groups, key=lambda c: COND_ORDER.index(c) if c in COND_ORDER else 99)}
        results.append({
            "run": d["run"], "inference": d["inference"],
            "platform": PLATFORMS.get(d["inference"], d["inference"]),
            "nominal_ms": nominal_ms, "groups": groups,
            "per_cond": {c: pooled(eps, nominal_ms, args.late_ms) for c, eps in groups.items()},
            "comparisons": compare(groups, args.alpha, args.min_effect_pct) if args.tests else [],
        })

    rates = {round(r["nominal_ms"], 6) for r in results}
    if len(rates) > 1:
        print(f"WARNING: runs use different nominal periods {sorted(rates)} ms", file=sys.stderr)

    # One row per platform x condition, mean ± std over that platform's runs.
    inf_order = {k: i for i, k in enumerate(PLATFORM_ORDER)}
    table = []
    for inf in sorted({r["inference"] for r in results}, key=lambda k: (inf_order.get(k, 99), k)):
        mine = [r for r in results if r["inference"] == inf]
        conds = sorted({c for r in mine for c in r["per_cond"]},
                       key=lambda c: COND_ORDER.index(c) if c in COND_ORDER else 99)
        for cond in conds:
            per_run = [r["per_cond"][cond] for r in mine if cond in r["per_cond"]]
            table.append({"platform": PLATFORMS.get(inf, inf), "condition": cond,
                          **aggregate(per_run)})

    if args.latex:
        print_latex(table, args.late_ms)
    else:
        print(f"Loop timing per platform and condition (nominal "
              f"{', '.join(f'{x:g}' for x in sorted(rates))} ms, late = period > nominal + "
              f"{args.late_ms:g} ms; mean{PM}std over runs where runs > 1)\n")
        print_summary(table)
        print("\n  runs:")
        for res in sorted(results, key=lambda r: (inf_order.get(r["inference"], 99), r["run"])):
            print(f"    {res['platform']:<17} {res['run']}")
        if args.tests:
            print("\nPer-episode tests (Mann-Whitney U, alpha "
                  f"{args.alpha:g}, min effect {args.min_effect_pct:g}%):")
            print_tests(results)
    if args.out:
        write_out(table, results, args.out)
    if args.plot:
        plot(results, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

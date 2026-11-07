"""Sweep quantization levels and tabulate how much performance survives.

Each axis takes a list; "fp" means "leave this part in float".

Axes
  --norm-variants   observation normalization(s) to run each config under. "orig" passes
                    `--obs-norm orig` to the env; "raw" omits the flag (unnormalized).
  --obs-word-bits   TOTAL bits per observation dim (sign + int + frac). Compares at equal
                    hardware word width: raw dims lose fractional bits to their integer part.
  --obs-frac-bits   FRACTIONAL bits X per dim instead. Format is then Q0.X when normalized
                    and QY.X when raw (Y from the dim's range), i.e. equal resolution step
                    in each dimension's own units; raw words end up wider.
  --weight-bits / --act-bits   actor weight / activation bits

The quantizer range (abs_max) per dim is derived by qat.default_obs_abs_max from the
env's obs_norm preset (OBS_NORM_PRESETS in pendulum_env.py), so raw observations are
never squeezed into Q0.X. Frame layout: [motor_pos, sin, cos, motor_vel, pen_vel, prev_act].
Override a variant with --abs-max NAME=a,b,c,d,e,f.
The format actually used is stored in the CSV (obs_formats), e.g. raw pen_vel shows up as Q5.x.

Examples
  # one axis at a time, others float, both normalizations (cheapest way to find each knee)
  python -m Training.sweep_quant --mode grid --jobs 48 --obs-word-bits 16 12 8 4 --weight-bits 16 12 8 4 --act-bits 16 12 8 4 --seeds 0 1 2 3 4 -- --total-steps 180000 --domain-randomization

  # same fractional bits for both normalizations: Q0.X vs QY.X
  python sweep_quant.py --obs-frac-bits fp 11 9 7 --norm-variants orig raw -- --total-steps 180000

  # full cross product, 6 runs in parallel
  python sweep_quant.py --mode grid --jobs 6 --obs-word-bits 12 10 --weight-bits 8 6 --act-bits 8 \
                        -- --total-steps 180000 --device cpu

Everything after `--` is forwarded to train_sac_qat.py / train_sac.py (do not pass
--obs-norm there; --norm-variants controls it). Delay flags are train_sac's:
--transport-delay-s, --dr-transport-delay-min/max (residual delay after the boxcar). Results are appended to
sweep_results.csv (headless eval of last.zip and best_model.zip, quantized env + policy).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from multiprocessing.connection import wait as mp_wait
from pathlib import Path

# Imports are relative to reinforcement_learning/ (same as train_sac.py). sys.path is
# inherited by the spawned workers, so this also covers --jobs > 1.
_RL_ROOT = Path(__file__).resolve().parents[1]
if str(_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_ROOT))
LOG_DIR = Path(__file__).resolve().parent / "runs"

FP = "fp"
NORM_FLAG = "--obs-norm"
FIELDS = ["run", "obs_norm", "obs_mode", "obs_bits", "obs_formats", "weight_bits", "act_bits",
          "seed", "which", "mean_return", "std_return", "upright_frac"]


def _tok(v: str):
    return None if v.lower() == FP else int(v)


def build_configs(ob, wb, ab, mode):
    if mode == "grid":
        return list(itertools.product(ob, wb, ab))
    cfgs = [(None, None, None)]
    cfgs += [(o, None, None) for o in ob if o is not None]
    cfgs += [(None, w, None) for w in wb if w is not None]
    cfgs += [(None, None, a) for a in ab if a is not None]
    return list(dict.fromkeys(cfgs))  # de-dup, keep order


def _die_with_parent(parent_pid: int) -> None:
    import signal, threading
    # Linux: kernel sends SIGTERM to this worker the moment the parent exits (even kill -9).
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        except Exception:
            pass
    # Portable fallback (also covers the parent dying before prctl ran):
    # if we've been reparented, the sweep is gone, so exit.
    def watch():
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(5)
    threading.Thread(target=watch, daemon=True).start()


def _init_worker(parent_pid: int):
    _die_with_parent(parent_pid)
    import torch
    torch.set_num_threads(1)


def _run_one(task) -> list[dict]:
    """Train + evaluate one config. Top-level so it can run in a worker process."""
    import Training.train_sac_qat as T
    meta, argv_run, n_eval, log_path = task
    with contextlib.ExitStack() as st:
        if log_path:                                   # parallel mode: keep logs apart
            f = st.enter_context(open(log_path, "w", buffering=1))
            st.enter_context(contextlib.redirect_stdout(f))
            st.enter_context(contextlib.redirect_stderr(f))
        args, qargs = T.parse_all(argv_run)
        run_dir = T.train_qat(args, qargs)
        try:
            with open(run_dir / "config.json") as fh:
                fmts = json.load(fh)["quant"].get("obs_formats")
            meta = dict(meta, obs_formats=" ".join(fmts) if fmts else FP)
        except Exception:
            meta = dict(meta, obs_formats="?")
        rows = []
        for which in ("last", "best_model"):
            path = run_dir / f"{which}.zip"
            if not path.exists():
                continue
            m = T.evaluate_headless(path, args, qargs, n_eval)
            rows.append(dict(meta, which=which, **m))
        return rows


def _worker_main(task, result_path: str, parent_pid: int) -> None:
    """Entry point of one parallel run (its own process). Rows go to result_path as
    JSON; any Python error (startup included) is appended to the run's log and the
    process exits with code 1."""
    try:
        _init_worker(parent_pid)
        rows = _run_one(task)
    except BaseException:
        with open(task[3], "a") as f:
            f.write("\n" + traceback.format_exc())
        sys.exit(1)
    Path(result_path).write_text(json.dumps(rows))


def _describe_exit(code) -> str:
    """Why a worker process ended without a result."""
    if code is None or code >= 0:
        return f"exit code {code} (Python error, see the end of the log)"
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = {1: "SIGHUP", 9: "SIGKILL", 15: "SIGTERM"}.get(-code, f"signal {-code}")
    hint = {
        "SIGKILL": "usually the out-of-memory killer or a job-scheduler limit "
                   "(check `dmesg | grep -i -E 'oom|killed'`); try fewer --jobs",
        "SIGSEGV": "native crash (MuJoCo / torch); the log shows how far it got",
        "SIGTERM": "terminated from outside (scheduler, kill, shutdown)",
        "SIGHUP": "terminal closed; run the sweep under nohup / tmux / screen",
    }.get(name, "")
    return f"killed by {name}" + (f": {hint}" if hint else "")


def _run_parallel(tasks, jobs: int, on_rows, on_fail) -> None:
    """One fresh process per run, at most `jobs` at a time. A run that dies (OOM
    kill, segfault) only fails itself; the rest of the sweep keeps going."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # inherited by spawned workers
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")                    # safe with CUDA and MuJoCo
    pending = list(tasks)
    running = {}                                     # sentinel -> (name, process, result path)
    try:
        while pending or running:
            while pending and len(running) < jobs:
                t = pending.pop(0)
                name = t[0]["run"]
                res = LOG_DIR / f"{name}.result.json"
                res.unlink(missing_ok=True)
                proc = ctx.Process(target=_worker_main, args=(t, str(res), os.getpid()),
                                   name=name)
                proc.start()
                running[proc.sentinel] = (name, proc, res)
                print(f"started: {name} (log: {t[3]})", flush=True)
            for sentinel in mp_wait(list(running)):
                name, proc, res = running.pop(sentinel)
                proc.join()
                if proc.exitcode == 0 and res.exists():
                    on_rows(name, json.loads(res.read_text()))
                    res.unlink()
                    print(f"done: {name}", flush=True)
                else:
                    on_fail(name, f"{_describe_exit(proc.exitcode)}; log: {LOG_DIR / name}.log")
    finally:
        for _, proc, _ in running.values():          # Ctrl-C / parent error: stop the rest
            if proc.is_alive():
                proc.terminate()
                proc.join()


def _parse_abs_max(items) -> dict:
    """Explicit per-variant overrides only; other variants use the env-derived range."""
    table = {}
    for it in items or []:
        name, _, vals = it.partition("=")
        if len(vals.split(",")) != 6:
            raise SystemExit(f"--abs-max {it!r}: need NAME=six,comma,separated,values")
        table[name] = vals
    return table


def _open_results(path: Path):
    """Append to the CSV; if its header is from an older version, set the old file aside."""
    if path.exists():
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != FIELDS:
            bak = path.with_suffix(f".{time.strftime('%Y%m%d_%H%M%S')}.bak.csv")
            path.rename(bak)
            print(f"note: {path} had a different column layout; moved to {bak}")
    new_file = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
    return fh, writer


def build_tasks(sw, base):
    obs_mode = "frac" if sw.obs_frac_bits else "word"
    ob = [_tok(v) for v in (sw.obs_frac_bits or sw.obs_word_bits or [FP])]
    wb = [_tok(v) for v in sw.weight_bits]
    ab = [_tok(v) for v in sw.act_bits]
    abs_max = _parse_abs_max(sw.abs_max)
    configs = build_configs(ob, wb, ab, sw.mode)
    tasks = []
    for norm, (o, w, a), seed in itertools.product(sw.norm_variants, configs, sw.seeds):
        otag = "fp" if o is None else f"{obs_mode[0]}{o}"
        name = f"{sw.tag}_n{norm}_o{otag}_w{w or 'fp'}_a{a or 'fp'}_s{seed}"
        argv_run = list(base) + ["--run-name", name, "--seed", str(seed)]
        if norm != "raw":
            argv_run += [NORM_FLAG, norm]
        if o is not None:
            argv_run += ["--obs-frac-bits" if obs_mode == "frac" else "--obs-word-bits", str(o)]
            if norm in abs_max:
                argv_run += ["--obs-abs-max", abs_max[norm]]
        if w is not None:
            argv_run += ["--weight-bits", str(w)]
        if a is not None:
            argv_run += ["--act-bits", str(a)]
        log = None
        if sw.jobs > 1:
            LOG_DIR.mkdir(exist_ok=True)
            log = str(LOG_DIR / f"{name}.log")
        meta = dict(run=name, obs_norm=norm, obs_mode=obs_mode,
                    obs_bits=o if o is not None else FP,
                    weight_bits=w or FP, act_bits=a or FP, seed=seed)
        tasks.append((meta, argv_run, sw.eval_episodes, log))
    return tasks, len(configs), obs_mode


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obs-word-bits", nargs="+", default=None)
    p.add_argument("--obs-frac-bits", nargs="+", default=None)
    p.add_argument("--weight-bits", nargs="+", default=[FP])
    p.add_argument("--act-bits", nargs="+", default=[FP])
    p.add_argument("--norm-variants", nargs="+", default=["orig", "raw"],
                   help='"raw" = omit the flag; any other name (an OBS_NORM_PRESETS key, '
                        'e.g. "orig") is passed as --obs-norm NAME.')
    p.add_argument("--abs-max", nargs="*", default=None, metavar="NAME=a,b,c,d,e,f",
                   help="per-variant quantizer ranges (default: derived from the preset)")
    p.add_argument("--mode", choices=["one-at-a-time", "grid"], default="one-at-a-time")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", default="sweep_results_2.csv")
    p.add_argument("--tag", default="thebigsweep")
    p.add_argument("--jobs", type=int, default=1,
                   help="runs in parallel, one process each. Each run is mostly CPU-bound "
                        "(MuJoCo stepping + tiny 128x128 nets), so use about your physical "
                        "core count. Logs go to runs/<name>.log.")
    sw, base = p.parse_known_args(argv)
    base = [b for b in base if b != "--"]
    if NORM_FLAG in base:
        raise SystemExit(f"remove {NORM_FLAG} from the forwarded args; --norm-variants controls it")
    if sw.obs_word_bits and sw.obs_frac_bits:
        raise SystemExit("use either --obs-word-bits or --obs-frac-bits, not both")
    return sw, base


def main(argv=None) -> int:
    sw, base = parse(argv if argv is not None else sys.argv[1:])
    tasks, n_cfg, obs_mode = build_tasks(sw, base)
    print(f"{n_cfg} configs x {len(sw.seeds)} seeds x {len(sw.norm_variants)} normalizations "
          f"= {len(tasks)} runs, obs bits mode: {obs_mode}, jobs={sw.jobs}")

    failures = []
    fh, writer = _open_results(Path(sw.out))

    def record(name, rows):
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
            print(f"  {name} {r['which']}: return {r['mean_return']:.1f} "
                  f"upright {r['upright_frac']:.2f}  obs {r['obs_formats']}")
        fh.flush()

    def fail(name, why):
        print(f"FAILED: {name}: {why}", flush=True)
        failures.append((name, why))

    with fh:
        if sw.jobs <= 1:
            for t in tasks:
                print(f"\n===== {t[0]['run']} =====")
                try:
                    record(t[0]["run"], _run_one(t))
                except Exception as e:
                    traceback.print_exc()
                    fail(t[0]["run"], repr(e))
        else:
            _run_parallel(tasks, sw.jobs, record, fail)
    print(f"\nResults in {sw.out}")
    if failures:
        print(f"{len(failures)} failed:")
        for name, why in failures:
            print(f"  {name}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
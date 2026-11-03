"""Balance metrics shared by finetune_async.py and finetune_deploy.py.

Per episode, from the post-step states in order:
    balanced            caught: |θ| < theta_thresh held for hold_s consecutive seconds
    time_to_balance_s   start of that hold (swing-up + catch time)
    frac_upright        fraction of the episode with |θ| below threshold
    frac_upright_after_catch, theta_rms_deg, motor_pos_std_deg,
    mean_abs_action_balanced (balance quality / dither, after the catch)
    dropped             fell past horizontal after the catch
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

METRIC_KEYS = (
    "balanced", "time_to_balance_s", "frac_upright",
    "frac_upright_after_catch", "theta_rms_deg", "motor_pos_std_deg",
    "mean_abs_action_balanced", "dropped", "n_valid",
)


def wrap_pi(x: float) -> float:
    """Wrap angle into [-pi, pi] (same as pendulum_env._wrap_pi)."""
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def balance_metrics(theta, motor_pos, abs_action, control_freq: float,
                    theta_thresh: float, hold_s: float) -> dict:
    """Metrics from per-transition arrays: θ (0 = upright) and motor angle of
    the state AFTER each step, and |action| of that step. Times are nominal
    (step index / control rate)."""
    nan = float("nan")
    out = dict(balanced=0, time_to_balance_s=nan, frac_upright=nan,
               frac_upright_after_catch=nan, theta_rms_deg=nan,
               motor_pos_std_deg=nan, mean_abs_action_balanced=nan,
               dropped=0, n_valid=0)
    th = np.asarray(theta, dtype=np.float64)
    mp = np.asarray(motor_pos, dtype=np.float64)
    act = np.asarray(abs_action, dtype=np.float64)
    n = len(th)
    out["n_valid"] = n
    if n == 0:
        return out
    up = np.abs(th) < theta_thresh
    out["frac_upright"] = float(up.mean())

    # Catch = first run of hold_n consecutive upright steps.
    hold_n = max(1, int(round(hold_s * control_freq)))
    run, i_catch = 0, None
    for i, u in enumerate(up):
        run = run + 1 if u else 0
        if run >= hold_n:
            i_catch = i - hold_n + 1
            break
    if i_catch is None:
        return out

    th_a, mp_a, act_a, up_a = th[i_catch:], mp[i_catch:], act[i_catch:], up[i_catch:]
    out.update(
        balanced=1,
        # Transition i observes the state after step i+1.
        time_to_balance_s=(i_catch + 1) / control_freq,
        frac_upright_after_catch=float(up_a.mean()),
        theta_rms_deg=float(np.degrees(np.sqrt(np.mean(th_a[up_a] ** 2)))),
        motor_pos_std_deg=float(np.degrees(np.nanstd(mp_a[up_a]))),
        mean_abs_action_balanced=float(act_a[up_a].mean()),
        dropped=int(np.any(np.abs(th_a) > math.pi / 2)),
    )
    return out


def fmt(x, spec: str = ".2f") -> str:
    return "-" if (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def write_eval_summary(eval_history: list, run_dir: Path | None) -> None:
    """Print (and save to run_dir/eval_summary.json) baseline / best / final
    eval and the first eval that caught and held."""
    if not eval_history:
        return
    print("\nDeterministic eval episodes:")
    print("    ep  real_steps  rig_min    return  t_bal_s  upright  held")
    for e in eval_history:
        held = "yes" if e["balanced"] and not e["dropped"] else "no"
        print(f"  {e['episode']:4d}  {e['real_steps']:10d}  {e['rig_minutes']:7.1f}"
              f"  {e['return']:+8.1f}  {fmt(e['time_to_balance_s']):>7}"
              f"  {fmt(e['frac_upright']):>7}  {held:>4}")
    best = max(eval_history, key=lambda e: e["return"])
    first_held = next((e for e in eval_history
                       if e["balanced"] and not e["dropped"]), None)
    summary = {
        "baseline": eval_history[0],
        "best": best,
        "final": eval_history[-1],
        "first_held": first_held,
        "n_evals": len(eval_history),
    }
    if first_held is None:
        print("  No eval episode caught and held the pendulum yet.")
    else:
        print(f"  First eval that caught and held: episode {first_held['episode']} "
              f"after {first_held['real_steps']} real transitions "
              f"({first_held['rig_minutes']:.1f} rig minutes)")
    print(f"  Baseline return {eval_history[0]['return']:+.1f} → best "
          f"{best['return']:+.1f} (ep {best['episode']}) → final "
          f"{eval_history[-1]['return']:+.1f}")
    if run_dir is not None:
        with open(Path(run_dir) / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

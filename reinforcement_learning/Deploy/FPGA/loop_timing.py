"""Per-episode control-loop timing summary, shared by every inference path so
the timing.csv columns mean exactly the same thing for each of them.

Inputs come from the PL's 100 MHz counters:
  * cpu (FPGA_env + FpgaLowLevelClient): one LoopTiming per accel strobe
  * pl  (FPGA_deploy_env): the cycle stamps in each step-log record

Definitions (one "action" = one control step reaching the step generator):
  period    previous action -> this action
  delay     sample the policy used -> this action (sensing-to-actuation)
  n_samples PL samples (500 Hz) published since the previous action; one
            control period holds `samples_per_tick` of them
"""

from __future__ import annotations

import numpy as np

SUMMARY_KEYS = (
    "fpga_n_actions", "fpga_period_ms", "fpga_period_std_ms", "fpga_jitter_pct",
    "fpga_period_max_ms", "fpga_delay_ms", "fpga_delay_std_ms",
    "fpga_delay_p99_ms", "fpga_delay_max_ms",
    "fpga_n_late_ticks", "fpga_n_no_new_sample", "fpga_n_sync_misses",
)


def summarize_loop_timing(period_ms, delay_ms, n_samples, samples_per_tick: int, *,
                          n_actions: int, sync_misses: int | None = None) -> dict:
    """fpga_n_actions        actions measured
    fpga_period_ms        mean period
    fpga_period_std_ms    its standard deviation
    fpga_jitter_pct       period std / mean period * 100
    fpga_period_max_ms    longest period
    fpga_delay_ms         mean delay
    fpga_delay_std_ms     its standard deviation
    fpga_delay_p99_ms     99th percentile delay
    fpga_delay_max_ms     longest delay
    fpga_n_late_ticks     actions that saw more new samples than one control
                          period holds (the loop missed its sample)
    fpga_n_no_new_sample  actions with no new sample since the previous one
    fpga_n_sync_misses    PS sample waits that woke after their sample (cpu
                          only; 0 where no PS wait is involved)
    Statistics without data are left out (blank CSV cells).
    """
    per = np.asarray(period_ms, dtype=float)
    dly = np.asarray(delay_ms, dtype=float)
    dly = dly[np.isfinite(dly)]
    ns = np.asarray(n_samples, dtype=int)
    out: dict = {"fpga_n_actions": int(n_actions)}
    if per.size:
        out.update(
            fpga_period_ms=float(per.mean()),
            fpga_period_std_ms=float(per.std()),
            fpga_jitter_pct=float(100.0 * per.std() / per.mean()),
            fpga_period_max_ms=float(per.max()),
        )
    if dly.size:
        out.update(
            fpga_delay_ms=float(dly.mean()),
            fpga_delay_std_ms=float(dly.std()),
            fpga_delay_p99_ms=float(np.percentile(dly, 99)),
            fpga_delay_max_ms=float(dly.max()),
        )
    if ns.size:
        out.update(
            fpga_n_late_ticks=int(np.sum(ns > samples_per_tick)),
            fpga_n_no_new_sample=int(np.sum(ns == 0)),
        )
    if sync_misses is not None:
        out["fpga_n_sync_misses"] = int(sync_misses)
    return out

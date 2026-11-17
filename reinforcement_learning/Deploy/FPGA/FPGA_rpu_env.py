"""Real-rig env where the policy runs bare metal on the Cortex-R5F.

Same rig, bitstream and episode sequence as FPGA_env.RealRotaryInvertedPendulumEnv
(the --inference cpu env, which this class extends): Linux homes the motor,
waits for a still pendulum, tares and engages - then, instead of a Python
control loop, the R5 runs the episode (rpu_firmware/rpu_main.c) and Linux only
supervises: host heartbeat, mid-episode weight uploads (on_step), reading the
step log.

`collect_episode()` returns the same arrays as FpgaDeployEnv.collect_episode,
built from the R5 log: obs is exactly what the network saw, action exactly what
was applied (clipped policy action, as in the cpu path), and the reward is
Training/reward.compute_reward on the state read at the next tick, i.e.
identical to FPGA_env.observe_and_step.

Loop timing comes from the same PL latch (axi_gpio_meas) the cpu path reads,
so `rpu_loop_timing()` returns the cpu path's fpga_* keys and the three modes
compare one to one; plus rpu_compute_us (R5 cycles for obs + MLP + accel write).
"""

from __future__ import annotations

import math
import time

import numpy as np

from Deploy.FPGA.FPGA_env import RealRotaryInvertedPendulumEnv
from Deploy.FPGA.FPGA_lowlevel_client import (
    ACCEL_FULL_SCALE,
    BRAKE_ACCEL,
    CTRL_ACCEL_STB,
    MEAS_CNT_MAX,
    SAFE_LIMIT_RAD,
    SAMPLE_PERIOD_US,
)
from Deploy.FPGA.rpu_client import RpuClient, RpuError, RpuRecords, S
from Simulation.pendulum_env import MOTOR_LIMIT_RAD, MOTOR_SAFE_LIMIT_RAD, _wrap_pi

POLL_S = 0.002                 # rig-thread poll period while the R5 runs an episode
START_GRACE_S = 0.5            # episode wall-clock budget on top of max_steps * dt


def rpu_loop_timing(recs: RpuRecords, samples_per_tick: int, clock_hz: float,
                    cpu_hz: float) -> dict:
    """Per-episode loop timing from the R5 log, same keys and rules as
    FPGA_env.loop_timing_summary (first action after the pause and saturated
    periods dropped), plus:
        rpu_compute_us / rpu_compute_max_us   obs + MLP + accel write on the R5
    """
    m = (recs.flags & S.F_HAS_ACTION) != 0
    m &= (recs.flags & S.F_MEAS_VALID) != 0
    m &= (recs.flags & S.F_FIRST_ACTION) == 0
    period = recs.meas_period[m] & MEAS_CNT_MAX
    keep = period < MEAS_CNT_MAX
    period = period[keep]
    action = recs.meas_action[m][keep]
    sample = recs.sample_cyc[m][keep]
    out: dict = {"fpga_n_actions": int(period.size)}
    acts = recs.flags & S.F_HAS_ACTION != 0
    if acts.any():
        comp = recs.compute_cyc[acts] / cpu_hz * 1e6
        out.update(rpu_compute_us=float(comp.mean()), rpu_compute_max_us=float(comp.max()))
    if not period.size:
        return out
    per = period / clock_hz * 1e3
    n_samples = (action >> 24) & 0xFF
    dly = (((action & MEAS_CNT_MAX) - (sample & MEAS_CNT_MAX)) & MEAS_CNT_MAX) / clock_hz * 1e3
    out.update(
        fpga_period_ms=float(per.mean()),
        fpga_period_std_ms=float(per.std()),
        fpga_jitter_pct=float(100.0 * per.std() / per.mean()),
        fpga_period_max_ms=float(per.max()),
        fpga_n_late_ticks=int(np.sum(n_samples > samples_per_tick)),
        fpga_n_no_new_sample=int(np.sum(n_samples == 0)),
        fpga_delay_ms=float(dly.mean()),
        fpga_delay_std_ms=float(dly.std()),
        fpga_delay_p99_ms=float(np.percentile(dly, 99)),
        fpga_delay_max_ms=float(dly.max()),
    )
    return out


def rpu_step_timing(recs: RpuRecords, clock_hz: float, cpu_hz: float) -> dict:
    """Per-action timing arrays for step_timing.csv (same filtering as above)."""
    m = ((recs.flags & S.F_HAS_ACTION) != 0) & ((recs.flags & S.F_MEAS_VALID) != 0) \
        & ((recs.flags & S.F_FIRST_ACTION) == 0)
    period = recs.meas_period[m] & MEAS_CNT_MAX
    keep = period < MEAS_CNT_MAX
    action = recs.meas_action[m][keep]
    sample = recs.sample_cyc[m][keep]
    return {
        "step": recs.step[m][keep],
        "period_ms": period[keep] / clock_hz * 1e3,
        "delay_ms": (((action & MEAS_CNT_MAX) - (sample & MEAS_CNT_MAX)) & MEAS_CNT_MAX)
                    / clock_hz * 1e3,
        "compute_us": recs.compute_cyc[m][keep] / cpu_hz * 1e6,
    }


class RpuRotaryInvertedPendulumEnv(RealRotaryInvertedPendulumEnv):
    """FPGA_env with the control loop on the R5 (see module docstring)."""

    def __init__(self, *, rpu_firmware: str | None = None, rpu_shm_base: int | None = None,
                 rpu_rproc: str = "remoteproc0", rpu_host_timeout_ms: int = 1000,
                 dt_jitter_frac: float = 0.0, **kwargs):
        kwargs["sample_sync"] = True          # the R5 is always locked to the PL samples
        super().__init__(**kwargs)
        self.rpu_firmware = rpu_firmware
        self.rpu_shm_base = rpu_shm_base
        self.rpu_rproc = rpu_rproc
        self.rpu_host_timeout_ms = int(rpu_host_timeout_ms)
        self.dt_jitter_frac = float(dt_jitter_frac)
        self.rpu: RpuClient | None = None
        self.next_explore = False             # set by the rig worker in before_start
        self.next_seed = 1
        self.next_dt_jitter: float | None = None
        self._episode_counter = 0

    _CONFIG_KEYS = RealRotaryInvertedPendulumEnv._CONFIG_KEYS

    # ------------------------------------------------------------------ setup
    def _ensure_open(self):
        client = super()._ensure_open()
        if self.rpu is None:
            kw = {} if self.rpu_shm_base is None else {"shm_base": self.rpu_shm_base}
            self.rpu = RpuClient(client, firmware=self.rpu_firmware, rproc=self.rpu_rproc,
                                 host_timeout_ms=self.rpu_host_timeout_ms, **kw)
            print(f"[rpu_env] R5 firmware up: {self.rpu.cpu_hz / 1e6:.0f} MHz, "
                  f"log depth {self.rpu.log_depth}")
        return client

    def configure_policy(self, actor) -> None:
        """Network dims + observation layout -> R5 (once, after the model is built)."""
        from Deploy.FPGA.rpu_client import rpu_dims
        self._ensure_open()
        n_in, h1, h2 = rpu_dims(actor)
        frame = ["motor_pos", "sin", "cos"]
        if self.obs_include_velocities:
            frame += ["motor_vel", "pen_vel"]
        frame += ["prev_act"]
        self.rpu.configure(
            n_in=n_in, h1=h1, h2=h2, history=self.obs_history_len, frame_feats=frame,
            norm=self._obs_norm, samples_per_tick=self._samples_per_tick,
            sample_period_us=SAMPLE_PERIOD_US, max_accel=self.max_accel_rad_s2,
            sim_safe_limit=MOTOR_SAFE_LIMIT_RAD,
            term_limit=MOTOR_LIMIT_RAD if self.terminate_on_hard_stop else 0.0,
            safe_limit_raw=SAFE_LIMIT_RAD, brake_accel=BRAKE_ACCEL,
            accel_full_scale=ACCEL_FULL_SCALE)

    # ------------------------------------------------------------------ safety
    def _on_signal(self, *_):
        self._sigterm_received = True
        if self.rpu is not None:
            try:
                self.rpu.abort()              # R5 off the GPIOs before we disengage
            except Exception:
                pass
        super()._on_signal()

    def disengage_safely(self) -> None:
        if self.rpu is not None:
            try:
                self.rpu.abort()
            except Exception:
                pass
        super().disengage_safely()

    def close(self):
        if self.rpu is not None:
            try:
                self.rpu.close()              # abort, detach, free the window
            except Exception as exc:
                print(f"  [warn] closing the R5 window: {exc}")
            self.rpu = None
        super().close()

    # ------------------------------------------------------------------ episode
    def collect_episode(self, *, before_start=None, on_step=None) -> dict:
        """reset() (home, rest, tare, before_start, engage) -> R5 episode -> arrays."""
        _, reset_info = self.reset(options={"before_start": before_start})
        client, rpu = self._client, self.rpu
        ep_id = self._episode_counter
        self._episode_counter += 1
        jitter = self.dt_jitter_frac if self.next_dt_jitter is None else self.next_dt_jitter
        rpu.start_episode(ep_id, max_steps=self._max_steps, explore=self.next_explore,
                          seed=self.next_seed, dt_jitter=jitter,
                          ctrl_level=client._ctrl_state, ctrl_stb=CTRL_ACCEL_STB)
        parts: list[RpuRecords] = []
        deadline = time.monotonic() + self.episode_length_s + START_GRACE_S
        aborted = None
        try:
            while True:
                rpu.heartbeat()
                new = rpu.read_new()
                if new is not None:
                    parts.append(new)
                st = rpu.state()
                if st != S.ST_RUNNING:
                    new = rpu.read_new()
                    if new is not None:
                        parts.append(new)
                    if st == S.ST_ERROR:
                        aborted = f"R5 error {rpu.error()}"
                    break
                if self._sigterm_received:
                    aborted = "stop requested"
                    break
                if time.monotonic() > deadline:
                    aborted = "episode overran its wall-clock budget"
                    break
                if on_step is not None:
                    on_step()
                time.sleep(POLL_S)
        finally:
            if aborted is not None or self._sigterm_received:
                rpu.abort()
            # The R5 left accel 0 latched; Linux owns the motor again (homing next).
        if aborted:
            print(f"  [warn] R5 episode {ep_id} aborted: {aborted}")
        recs = RpuRecords.concat(parts) if parts else None
        if recs is not None:
            recs = self._episode_slice(recs, ep_id)
        return self._episode_arrays(recs, reset_info, aborted)

    @staticmethod
    def _episode_slice(recs: RpuRecords, ep_id: int) -> RpuRecords:
        keep = recs.episode == (ep_id & 0xFFFFFFFF)
        return RpuRecords(**{k: getattr(recs, k)[keep]
                             for k in RpuRecords.__dataclass_fields__})

    def _episode_arrays(self, recs: RpuRecords | None, reset_info: dict, aborted) -> dict:
        """R5 records (one per tick) -> transitions. Tick t holds (s_t, obs_t, a_t);
        transition t = (obs_t, a_t, r(a_t, s_{t+1}), obs_{t+1})."""
        O, A, R, O2, T, V, dts, phis, mps, brakes = [], [], [], [], [], [], [], [], [], []
        trunc = False
        n_rec = 0 if recs is None else len(recs)
        for i in range(max(0, n_rec - 1)):
            if not recs.flags[i] & S.F_HAS_ACTION or recs.step[i + 1] != recs.step[i] + 1:
                break                         # gap: the log is contiguous unless aborted
            a = float(recs.action[i])
            mp, phi = float(recs.motor_pos[i + 1]), float(recs.phi[i + 1])
            self._motor_vel = float(recs.motor_vel[i + 1])
            self._pen_vel = float(recs.pen_vel[i + 1])
            r = self._reward(a, mp, phi)      # Training/reward.py via FPGA_env
            last = recs.flags[i + 1]
            term = bool(last & S.F_TERMINATED)
            if term:
                r -= self.hard_stop_penalty
            O.append(recs.obs[i]); A.append(a); R.append(r); O2.append(recs.obs[i + 1])
            T.append(term); V.append(True)
            dts.append(((int(recs.time_us[i + 1]) - int(recs.time_us[i])) & 0xFFFFFFFF) * 1e-6)
            phis.append(phi); mps.append(mp)
            brakes.append(bool(recs.flags[i] & S.F_BRAKE))
            if term or last & S.F_TRUNCATED:
                trunc = bool(last & S.F_TRUNCATED) and not term
                break
        n = len(R)
        clock_hz = self._client.clock_frequency
        timing = (rpu_loop_timing(recs, self._samples_per_tick, clock_hz, self.rpu.cpu_hz)
                  if recs is not None and len(recs) else {})
        if recs is not None and len(recs):
            timing["fpga_n_sync_misses"] = 0          # busy-wait: no learnt clock offset
        steps = (rpu_step_timing(recs, clock_hz, self.rpu.cpu_hz)
                 if recs is not None and len(recs) else None)
        dim = self.observation_space.shape[0]
        return {
            "obs": np.asarray(O, dtype=np.float32).reshape(n, dim),
            "actions": np.asarray(A, dtype=np.float32).reshape(n, 1),
            "rewards": np.asarray(R, dtype=np.float32),
            "next_obs": np.asarray(O2, dtype=np.float32).reshape(n, dim),
            "terminated": np.asarray(T, dtype=bool),
            "valid": np.asarray(V, dtype=bool),
            "return": float(np.sum(R)),
            "length": n,
            "dt_s": np.asarray(dts, dtype=np.float64),
            "gap": np.zeros(n, dtype=bool),
            "brake": np.asarray(brakes, dtype=bool),
            "phi": np.asarray(phis, dtype=np.float64),
            "motor_pos": np.asarray(mps, dtype=np.float64),
            "truncated": bool(trunc),
            "aborted": aborted,
            "homed": bool(reset_info.get("homed")),
            "tared": bool(reset_info.get("tared")),
            "rest_wait_s": self.last_rest_wait_s,
            "pl_timing": timing,
            "step_timing": steps,
            "weight_versions": (np.unique(recs.weight_version[recs.flags & S.F_HAS_ACTION != 0])
                                if recs is not None and len(recs) else np.array([], np.uint32)),
        }


def episode_balance_inputs(ep: dict):
    """(theta from upright, motor_pos, |action|) for finetune_metrics.balance_metrics."""
    theta = np.array([_wrap_pi(p - math.pi) for p in ep["phi"]])
    return theta, ep["motor_pos"], np.abs(ep["actions"][:, 0])

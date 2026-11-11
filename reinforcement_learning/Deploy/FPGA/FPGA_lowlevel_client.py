"""FpgaLowLevelClient: PYNQ client mirroring the bare-metal C low-level server.

Behaviour matches the C firmware (ll_poll / handle_set_accel):
  * accel is given in rad/s^2, NaN-guarded, clamped to +-ACCEL_FULL_SCALE and
    normalised to Q8.8 (accel / ACCEL_FULL_SCALE) before being written
  * CTRL_ACCEL_STB is pulsed after every accepted accel write
  * rail brake at +-SAFE_LIMIT_RAD on the raw-frame motor position
  * accel commands are ignored while disengaged
  * tare waits for the PL to apply it before returning
  * state is returned with the server's output sign flip (optional)
  * every state read carries the PL cycle stamp of its sample (sample_cyc) and
    is checked for tearing (re-read if a new sample landed mid-read)
  * loop timing (optional): after every accel strobe the PL latches the
    action->action period and the action cycle stamp (axi_gpio_meas), so the
    delay is measured from the sample the caller actually used; see
    LoopTiming / timing_log
  * wait_for_sample(): sleep-then-spin wait for a PL sample, so a control
    loop can run locked to the 500 Hz sample clock
  * enter_realtime(): optional core pinning / SCHED_FIFO for the control thread
"""

from __future__ import annotations

import asyncio
import gc
import math
import os
import struct
import threading
import time
from pathlib import Path
from typing import NamedTuple

from pynq import Overlay, MMIO

# AXI GPIO register offsets
GPIO_CH1_DATA_OFFSET = 0x0000
GPIO_CH2_DATA_OFFSET = 0x0008

# Control bits (must match the C firmware / training_pipeline.sv)
CTRL_ENGAGE       = 0x01   # level
CTRL_TARE_ENCODER = 0x02   # pulse (rising edge)
CTRL_ACCEL_STB    = 0x04   # pulse after every accepted accel write

# Safety / scaling constants (must match the C firmware)
SAFE_LIMIT_RAD   = math.radians(125.0)   # 2.1816616 rad
BRAKE_ACCEL      = 150.0                 # rad/s^2
ACCEL_FULL_SCALE = 150.0                 # rad/s^2 at action = 1.0
TARE_SETTLE_S    = 0.005                 # > one 2 ms sample + pipeline

# PL sample clock (training_pipeline SAMPLE_HZ); time_us steps by this per sample
SAMPLE_HZ        = 500
SAMPLE_PERIOD_US = 1_000_000 // SAMPLE_HZ
SYNC_EDGE_MAX_S    = 200e-6   # sample edge must be pinned this tightly to learn the offset
SYNC_REF_MAX_AGE_S = 2.0      # older PL<->PS offsets are re-learnt (crystal drift)

# Loop-timing GPIO fields (must match training_pipeline.sv)
MEAS_CNT_BITS = 24
MEAS_CNT_MAX  = (1 << MEAS_CNT_BITS) - 1   # saturated = first action after reset/idle


def _to_int16(val: int) -> int:
    """Interpret an unsigned 16-bit register readout as signed."""
    val &= 0xFFFF
    return val - 0x10000 if val & 0x8000 else val

def _to_int32(val: int) -> int:
    """Interpret an unsigned 32-bit register readout as signed."""
    val &= 0xFFFFFFFF
    return val - 0x100000000 if val & 0x80000000 else val

def us_diff(a: int, b: int) -> int:
    """a - b for 32-bit wrapping PL microsecond stamps (signed result)."""
    return ((int(a) - int(b) + 0x80000000) & 0xFFFFFFFF) - 0x80000000

def _to_uint16(val: int) -> int:
    """Pack a signed 16-bit integer into an unsigned 16-bit word."""
    return struct.unpack("<H", struct.pack("<h", int(val)))[0]


def q16_to_float(raw_int16: int, frac_bits: int) -> float:
    return float(raw_int16) / float(1 << frac_bits)


def float_to_q16(val_float: float, frac_bits: int) -> int:
    raw = int(round(val_float * float(1 << frac_bits)))
    return max(-32768, min(32767, raw))


class LowLevelState(NamedTuple):
    """Decoded state (rad, rad/s)."""
    time_us: int
    motor_pos_rad: float
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float
    sample_cyc: int | None = None   # PL cycle stamp of this sample (None: old bitstream)


class LoopTiming(NamedTuple):
    """One PL measurement, latched at an accel strobe (units: PL clock cycles).

    period_cycles : previous action -> this action (loop period)
    delay_cycles  : sample used for this action -> this action, where "used"
                    is the last state returned by get_state()/step() before
                    the action (-1 if unknown)
    n_samples     : samples published since the previous action
                    (0: no new state since last action, 1: normal, >1: overrun)
    action_cyc    : PL cycle stamp of this action (low 24 bits)
    sample_cyc    : PL cycle stamp of the used sample (None if unknown)
    """
    seq: int
    period_cycles: int
    delay_cycles: int
    n_samples: int
    action_cyc: int
    sample_cyc: int | None
    clock_hz: float

    @property
    def period_s(self) -> float:
        return self.period_cycles / self.clock_hz

    @property
    def delay_s(self) -> float:
        return self.delay_cycles / self.clock_hz if self.delay_cycles >= 0 else float("nan")

    @property
    def saturated(self) -> bool:
        """True when the period overflowed (first action after reset or a long pause)."""
        return self.period_cycles >= MEAS_CNT_MAX


def enter_realtime(cpu: int | None = None, priority: int = 50, freeze_gc: bool = True) -> dict:
    """Best-effort soft-real-time setup for the CALLING thread (Linux only).

    cpu       : pin this thread to one core (None: leave affinity alone)
    priority  : SCHED_FIFO priority 1..99 (0: keep the normal scheduler);
                needs root / CAP_SYS_NICE, otherwise skipped
    freeze_gc : gc.freeze() so existing objects are never rescanned
                (garbage collection stays on)
    Returns what was applied; failures are reported, never raised.
    """
    done = {"cpu": None, "fifo": None, "gc_frozen": False}
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {int(cpu)})      # pid 0 = calling thread on Linux
            done["cpu"] = int(cpu)
        except OSError as e:
            print(f"  [rt] could not pin to CPU {cpu}: {e}")
    if priority > 0 and hasattr(os, "sched_setscheduler"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(int(priority)))
            done["fifo"] = int(priority)
        except OSError as e:
            print(f"  [rt] SCHED_FIFO {priority} not applied ({e}); run as root for it")
    if freeze_gc:
        gc.collect()
        gc.freeze()
        done["gc_frozen"] = True
    return done


class FpgaLowLevelClient:
    """Hardware client for the FPGA cores using 16-bit Q-format fixed point."""

    def __init__(
        self,
        bitstream_path: str | Path = "/home/xilinx/jupyter_notebooks/RIP/rotary-inverted-pendulum/reinforcement_learning/Deploy/FPGA/lowlevel_client_4x.xsa",
        download: bool = True,
        pos_frac_bits: int = 12,     # Q3.12 rad
        vel_frac_bits: int = 7,      # Q9.7  rad/s
        accel_frac_bits: int = 9,    # Q8.8  normalised action
        flip_sign: bool = True,      # match LowLevelServer::sendState() output flip
        safety: bool = True,         # NaN guard, clamp and rail brake
        measure_timing: bool = True, # read axi_gpio_meas after every accel strobe
        port: str | None = None,     # compatibility shim
        baud: int | None = None,     # compatibility shim
    ):
        self.clock_frequency = 1e8
        self.pos_frac_bits = pos_frac_bits
        self.vel_frac_bits = vel_frac_bits
        self.accel_frac_bits = accel_frac_bits
        self.flip_sign = flip_sign
        self.safety = safety

        bit_path = Path(bitstream_path).resolve()
        if not bit_path.exists():
            raise FileNotFoundError(f"Hardware bitstream/XSA not found at: {bit_path}")

        print(f"Programming FPGA overlay from: {bit_path.name} (download={download})")
        # PYNQ's device probe calls asyncio.get_event_loop(), which fails in a
        # non-main thread that has no loop yet; give such a thread its own.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        self.ol = Overlay(str(bit_path), download=download)

        try:
            gpio_step_addr = self.ol.ip_dict["axi_gpio_step"]["phys_addr"]
            gpio_pen_addr  = self.ol.ip_dict["axi_gpio_pen"]["phys_addr"]
            gpio_ctrl_addr = self.ol.ip_dict["axi_gpio_ctrl"]["phys_addr"]
            gpio_timing_addr = self.ol.ip_dict["axi_gpio_timing"]["phys_addr"]
        except KeyError as err:
            available = list(self.ol.ip_dict.keys())
            raise KeyError(
                f"Missing expected AXI GPIO IP: {err}. Discovered blocks: {available}"
            ) from err

        # Optional: older bitstreams have no measurement GPIO / sample stamp.
        self.gpio_meas = None
        if "axi_gpio_meas" in self.ol.ip_dict:
            self.gpio_meas = MMIO(self.ol.ip_dict["axi_gpio_meas"]["phys_addr"], 0x1000)
        elif measure_timing:
            print("  axi_gpio_meas not in overlay; loop-timing measurement disabled")
        timing_params = self.ol.ip_dict["axi_gpio_timing"].get("parameters", {}) or {}
        self._has_sample_cyc = (self.gpio_meas is not None
                                or str(timing_params.get("C_IS_DUAL", "0")) == "1")
        self.measure_timing = bool(measure_timing and self.gpio_meas is not None)
        self.timing_log: list[LoopTiming] = []   # valid (non-saturated) measurements
        self.last_timing: LoopTiming | None = None
        self._last_seq: int | None = None
        self._skip_next = True
        self._used_sample_cyc: int | None = None  # stamp of the last state handed out
        self._sync_ref: tuple[float, int] | None = None  # (perf_counter, time_us) pair
        self.n_torn_reads = 0                     # state reads retried for tearing
        self.n_sync_misses = 0                    # waits that woke after their sample

        self.gpio_step = MMIO(gpio_step_addr, 0x1000)
        self.gpio_pen  = MMIO(gpio_pen_addr,  0x1000)
        self.gpio_ctrl = MMIO(gpio_ctrl_addr, 0x1000)
        self.gpio_timing = MMIO(gpio_timing_addr, 0x1000)

        # RLock so public methods can call each other while holding the lock.
        self._lock = threading.RLock()
        self._ctrl_state = 0      # persistent (level) bits only; pulses never stored
        self._engaged = False

        # Known-safe initial state, like main() in the C firmware.
        self._hal_write_accel_raw(0)
        self._hal_write_ctrl(0)
        try:
            input("Home both encoders and then press enter.")
        except EOFError:
            print("  (no stdin; engaging in 2 s)")
            time.sleep(2.0)

    # ------------------------------------------------------------------
    # HAL
    # ------------------------------------------------------------------

    def _hal_read_raw(self) -> tuple[int, int, int, int, int]:
        """Return signed raw (mp, pp, mv, pv) register values."""
        mp = _to_int16(self.gpio_step.read(GPIO_CH1_DATA_OFFSET))
        mv = _to_int16(self.gpio_step.read(GPIO_CH2_DATA_OFFSET))
        pp = _to_int16(self.gpio_pen.read(GPIO_CH1_DATA_OFFSET))
        pv = _to_int16(self.gpio_pen.read(GPIO_CH2_DATA_OFFSET))
        t  = _to_int32(self.gpio_timing.read(GPIO_CH1_DATA_OFFSET))
        return mp, pp, mv, pv, t

    def _hal_read_stamp(self) -> tuple[int, int | None]:
        """(time_us, sample_cyc) of the sample currently on the GPIOs."""
        t = _to_int32(self.gpio_timing.read(GPIO_CH1_DATA_OFFSET))
        c = (self.gpio_timing.read(GPIO_CH2_DATA_OFFSET) & 0xFFFFFFFF
             if self._has_sample_cyc else None)
        return t, c

    def _hal_read_state(self) -> LowLevelState:
        """Read one consistent sample: stamp, data, stamp again; retry if a
        new sample was published in between (the PL updates every 2 ms)."""
        for _ in range(4):
            stamp0 = self._hal_read_stamp()
            mp = _to_int16(self.gpio_step.read(GPIO_CH1_DATA_OFFSET))
            mv = _to_int16(self.gpio_step.read(GPIO_CH2_DATA_OFFSET))
            pp = _to_int16(self.gpio_pen.read(GPIO_CH1_DATA_OFFSET))
            pv = _to_int16(self.gpio_pen.read(GPIO_CH2_DATA_OFFSET))
            stamp1 = self._hal_read_stamp()
            if stamp0 == stamp1:
                break
            self.n_torn_reads += 1
        t, c = stamp1
        self._used_sample_cyc = c
        s = -1.0 if self.flip_sign else 1.0
        return LowLevelState(
            time_us=t,
            motor_pos_rad=s * q16_to_float(mp, self.pos_frac_bits),
            pendulum_pos_rad=s * q16_to_float(pp, self.pos_frac_bits),
            motor_vel_rad_s=s * q16_to_float(mv, self.vel_frac_bits),
            pendulum_vel_rad_s=s * q16_to_float(pv, self.vel_frac_bits),
            sample_cyc=c,
        )

    def _hal_read_timing(self, used_sample_cyc: int | None = None) -> LoopTiming:
        period = self.gpio_meas.read(GPIO_CH1_DATA_OFFSET) & 0xFFFFFFFF
        action = self.gpio_meas.read(GPIO_CH2_DATA_OFFSET) & 0xFFFFFFFF
        action_cyc = action & MEAS_CNT_MAX
        delay = (-1 if used_sample_cyc is None
                 else (action_cyc - used_sample_cyc) & MEAS_CNT_MAX)
        return LoopTiming(
            seq=(period >> MEAS_CNT_BITS) & 0xFF,
            period_cycles=period & MEAS_CNT_MAX,
            delay_cycles=delay,
            n_samples=(action >> MEAS_CNT_BITS) & 0xFF,
            action_cyc=action_cyc,
            sample_cyc=used_sample_cyc,
            clock_hz=self.clock_frequency,
        )

    def _record_timing(self) -> None:
        """Fetch the measurement latched by the strobe just sent and log it."""
        used = self._used_sample_cyc
        m = self._hal_read_timing(used)
        if m.seq == self._last_seq:
            # The read overtook the strobe write on the bus; read once more.
            m = self._hal_read_timing(used)
            if m.seq == self._last_seq:
                return
        self._last_seq = m.seq
        self.last_timing = m
        if self._skip_next:
            # this period spans a pause (engage / log clear), not a loop step
            self._skip_next = False
        elif not m.saturated:
            self.timing_log.append(m)

    def _hal_write_accel_raw(self, accel_q: int) -> None:
        """Write an already-quantised Q8.8 value to ctrl ch2 (no strobe)."""
        self.gpio_ctrl.write(GPIO_CH2_DATA_OFFSET, _to_uint16(accel_q))

    def _hal_write_ctrl(self, c: int) -> None:
        self._ctrl_state = int(c) & 0xFFFF
        self.gpio_ctrl.write(GPIO_CH1_DATA_OFFSET, self._ctrl_state)

    def _pulse(self, bit: int) -> None:
        """Pulse a control bit without disturbing the persistent bits."""
        self.gpio_ctrl.write(GPIO_CH1_DATA_OFFSET, self._ctrl_state | bit)
        self.gpio_ctrl.write(GPIO_CH1_DATA_OFFSET, self._ctrl_state)

    def _apply_accel(self, accel: float) -> bool:
        """Safety-process, quantise, write and strobe an accel command (rad/s^2).

        Returns False if the command was ignored (not engaged).
        """
        if not self._engaged:
            return False

        a = float(accel)
        if self.safety:
            if a != a:                                   # NaN
                a = 0.0
            a = max(-ACCEL_FULL_SCALE, min(ACCEL_FULL_SCALE, a))

            # Rail brake on raw-frame motor position (before the sign flip).
            mp_raw, _, _, _, _ = self._hal_read_raw()
            mp_rad = q16_to_float(mp_raw, self.pos_frac_bits)
            if mp_rad >= SAFE_LIMIT_RAD:
                a = -BRAKE_ACCEL
            elif mp_rad <= -SAFE_LIMIT_RAD:
                a = BRAKE_ACCEL

        accel_q = float_to_q16(a / ACCEL_FULL_SCALE, self.accel_frac_bits)
        self._hal_write_accel_raw(accel_q)
        self._pulse(CTRL_ACCEL_STB)    # data write is ordered before the strobe
        if self.measure_timing:
            self._record_timing()
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> LowLevelState:
        with self._lock:
            return self._hal_read_state()

    def set_acceleration(self, accel: float | int) -> None:
        """Command acceleration in rad/s^2 (ignored while disengaged)."""
        with self._lock:
            self._apply_accel(accel)

    def step(self, accel: float | int) -> LowLevelState:
        with self._lock:
            self._apply_accel(accel)
            return self._hal_read_state()

    def get_timing(self) -> LoopTiming | None:
        """Read the PL loop-timing registers now (None if not available)."""
        if self.gpio_meas is None:
            return None
        with self._lock:
            return self._hal_read_timing(self._used_sample_cyc)

    def read_time_us(self) -> int:
        """PL time stamp (us) of the newest published sample."""
        return _to_int32(self.gpio_timing.read(GPIO_CH1_DATA_OFFSET))

    def wait_for_sample(self, target_time_us: int, *, spin_s: float = 0.001,
                        timeout_s: float = 0.1) -> int:
        """Block until the PL publishes the sample stamped target_time_us
        (half a sample period of tolerance) or a later one; returns the
        time_us seen. Returns at once if it is already there (caller is late).

        Sleeps until ~spin_s before the sample is due, then busy-polls the
        time_us register, so the wake-up error of time.sleep is absorbed and
        the sample is seen within a few microseconds. The PL<->perf_counter
        offset is learnt from the last sample edge seen while spinning, and
        only from an edge pinned down to < SYNC_EDGE_MAX_S (a GIL handoff or
        preemption mid-poll would otherwise bias it late). If a wait based on
        that offset still wakes after the sample, the offset was too late:
        it is dropped and the next wait spins long enough to learn it again.
        Offsets older than SYNC_REF_MAX_AGE_S (e.g. across a reset) are not
        trusted, so PS/PL crystal drift cannot build up.
        Raises TimeoutError (an OSError) if no such sample arrives in time.
        Does not take the client lock, so disengage from another thread is
        never blocked.
        """
        tol = SAMPLE_PERIOD_US // 2
        t = self.read_time_us()
        remaining_us = us_diff(target_time_us, t)
        if remaining_us <= tol:
            return t                      # caller is late; not a prediction error
        now = time.perf_counter()
        used_ref = False
        ref = self._sync_ref
        if ref is not None and now - ref[0] < SYNC_REF_MAX_AGE_S:
            due = ref[0] + us_diff(target_time_us, ref[1]) * 1e-6
            used_ref = True
        else:
            # the newest sample may already be up to one period old
            due = now + (remaining_us - SAMPLE_PERIOD_US) * 1e-6
        if due - spin_s > now:
            time.sleep(due - spin_s - now)
        hard_deadline = now + remaining_us * 1e-6 + timeout_s
        p_old = None                      # when the last old-value read started
        while True:
            p0 = time.perf_counter()
            t = self.read_time_us()
            p1 = time.perf_counter()
            if us_diff(t, target_time_us) >= -tol:
                if p_old is not None:
                    if p1 - p_old < SYNC_EDGE_MAX_S:
                        self._sync_ref = (0.5 * (p_old + p1), t)
                elif used_ref:
                    self._sync_ref = None # woke after the sample: relearn
                    self.n_sync_misses += 1
                return t
            p_old = p0
            if p1 > hard_deadline:
                raise TimeoutError(
                    f"no PL sample at time_us={target_time_us} (newest {t}); "
                    "is the PL running?")

    def clear_timing_log(self) -> None:
        """Drop logged measurements. The next action's period is dropped too,
        since it spans the pause before it."""
        with self._lock:
            self.timing_log.clear()
            self.n_sync_misses = 0
            self._skip_next = True

    def engage_motor(self) -> None:
        with self._lock:
            self._engaged = True
            self._skip_next = True        # first period after engage spans the pause
            self._hal_write_accel_raw(0)
            self._hal_write_ctrl(self._ctrl_state | CTRL_ENGAGE)

    def disengage_motor(self) -> None:
        with self._lock:
            self._engaged = False
            self._hal_write_accel_raw(0)
            self._hal_write_ctrl(self._ctrl_state & ~CTRL_ENGAGE)

    def disengage_safely(self) -> None:
        self.disengage_motor()

    def tare_pendulum(self) -> bool:
        """Zero the encoders; returns once the PL has applied the new zero."""
        with self._lock:
            self._pulse(CTRL_TARE_ENCODER)
            time.sleep(TARE_SETTLE_S)
        return True

    def reset(self) -> LowLevelState:
        with self._lock:
            self._hal_write_accel_raw(0)
            self.tare_pendulum()
            self.engage_motor()
            return self._hal_read_state()

    def wait_until_ready(self, *, retries: int = 10, retry_delay_s: float = 0.3) -> bool:
        """No UART handshake on-board; always ready once the overlay is loaded."""
        return True

    def close(self) -> None:
        self.disengage_motor()

    def __enter__(self) -> "FpgaLowLevelClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
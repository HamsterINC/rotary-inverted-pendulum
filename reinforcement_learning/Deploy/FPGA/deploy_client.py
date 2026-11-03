"""FpgaDeployClient: PYNQ client for the *deployment* bitstream, where the PL runs
the whole policy (sensors -> state -> MLP inference -> stepper) by itself.

The PS
  * engages / disengages the motor and tares (DEPLOY_CTRL register, the
    counterpart of the `en` GPIO in the training bitstream),
  * drives the motor itself while the policy is off: set_acceleration() works
    like FpgaLowLevelClient's (used for homing between episodes),
  * starts / stops the policy: while it runs, the PL decides the action,
  * reads the live state (AXI GPIO telemetry, updated every 2 ms),
  * reads the step log: one record per PL inference with exactly the
    observation the network saw and the action it produced (ring buffer of
    1024 records, so Python can fall behind without losing steps),
  * uploads new weights between episodes (double-buffered, see weight_loader.py).

Hardware: board_top_AUPZU3 (mode "Deployment") with weight_loader_axi on the
WEIGHT_AXI port (0x8004_0000) and the USB_Logger GPIOs.

Engage logic in the PL:   engaged = SW1 & (PS control ? PS engage : 1)
  -> SW1 must be ON (armed) for the PS to engage, and switching SW1 OFF always
     stops the motor. This client takes PS control on start-up with the motor
     disengaged, and keeps it on close() (releasing it while SW1 is on would
     hand control back to the switch and start the motor).

Episode phases (all under PS control):
    engage_motor()      driver on, policy OFF, PS acceleration (starts at 0)
    set_acceleration()  e.g. PD homing to motor angle 0
    disengage_motor()   driver off
    tare_pendulum()     zero motor angle AND pendulum (only on request)
    start_policy()      driver on + PL policy at 40 Hz (new episode in the log)
    disengage_motor()   end of episode
The motor angle and the PL rail brake (125 deg, also active while homing) use
the motor encoder, as in the training_pipeline bitstream. With SW1 alone (no
PS control) the old behaviour remains: tare on engage, policy always on.

All values are in the RAW PL frame, which is the sim frame used by
pendulum_env (FpgaLowLevelClient flips signs and FPGA_env un-flips them again).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from Deploy.FPGA.weight_loader import WEIGHT_BASE, WEIGHT_RANGE, WeightLoader, WeightStatus
except ImportError:                       # run from inside Deploy/FPGA
    from weight_loader import WEIGHT_BASE, WEIGHT_RANGE, WeightLoader, WeightStatus

# AXI GPIO register offsets
GPIO_CH1_DATA_OFFSET = 0x0000
GPIO_CH2_DATA_OFFSET = 0x0008

# weight_loader_axi registers (byte offsets in the WEIGHT_AXI window)
REG_DEPLOY_CTRL   = 0x1_000C     # bit0 engage, bit1 PS control, bit2 tare (pulse), bit3 policy
REG_DEPLOY_STATUS = 0x1_0010     # bit0 engaged, bit1 SW1 armed, bit2 policy active
REG_LOG_COUNT     = 0x1_0014     # records written since power-up
REG_LOG_INFO      = 0x1_0018     # [15:0] depth, [31:16] words per record
REG_PS_ACCEL      = 0x1_001C     # [15:0] action Q.9 while the policy is off; write = strobe
LOG_BASE          = 0x1_8000

CTRL_ENGAGE     = 0x1
CTRL_PS_CONTROL = 0x2
CTRL_TARE       = 0x4
CTRL_POLICY     = 0x8

# Fixed-point formats of the deployment pipeline (top_pipeline parameters)
OBS_FRAC_BITS    = 9             # FRAC_POS = FRAC_VEL = FRAC_W = 9 (sin/cos too)
ACTION_FRAC_BITS = 9             # FRAC_ACCEL: 512 = action 1.0
ACCEL_FULL_SCALE = 150.0         # rad/s^2 at action 1.0
MOTOR_ENC_CPR    = 4000          # motor encoder x4 counts/rev (calc_state_2 CPR)
PL_CONTROL_HZ    = 50.0          # default only; the bitstream reports its rate (weights.config)
SAFE_LIMIT_RAD   = 2.1816616     # PL rail brake (125 deg)

TARE_SETTLE_S    = 0.005         # > one 2 ms sample + pipeline


def _to_int16(val: int) -> int:
    val &= 0xFFFF
    return val - 0x10000 if val & 0x8000 else val


def _to_int32(val: int) -> int:
    val &= 0xFFFFFFFF
    return val - 0x100000000 if val & 0x80000000 else val


def q16_to_float(raw_int16: int, frac_bits: int = OBS_FRAC_BITS) -> float:
    return float(raw_int16) / float(1 << frac_bits)


class DeployState(NamedTuple):
    """Live state (sim frame). theta = 0 upright, phi = theta + pi."""
    motor_pos_rad: float
    sin_theta: float
    cos_theta: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float
    prev_action: float           # PL F_prev (last network output, unclipped)

    @property
    def theta(self) -> float:
        return math.atan2(self.sin_theta, self.cos_theta)


@dataclass(frozen=True)
class StepRecord:
    """One PL inference.

    obs     : the 6 network inputs, float, order = HW_INPUT_ORDER =
              [motor_pos, sin, cos, motor_vel, pen_vel, prev_act]
              (same order as pendulum_env with obs_norm="raw", history 1)
    action  : network output clipped to [-1, 1] (what the PL commanded,
              unless `brake` is set: then the rail brake overrode it)
    """
    index: int                   # global log index (LOG_COUNT - 1 when newest)
    obs: np.ndarray
    obs_raw: np.ndarray          # int16 codes
    action_raw: int              # int16 network output (Q.9)
    action: float
    brake: bool
    weight_bank: int
    clipped: bool
    time_us: int                 # PL time of the observation
    episode_step: int            # 0, 1, 2 ... within one engage
    motor_counts: int            # motor encoder counts since the last tare
    episode: int                 # PL engage counter

    @classmethod
    def from_words(cls, index: int, w: np.ndarray) -> "StepRecord":
        w = np.asarray(w, dtype=np.uint32)
        obs_raw = w[:3].view(np.int16).copy()          # w0..w2 -> 6 x int16 (low half first)
        action_raw = _to_int16(int(w[3]))
        flags = int(w[3]) >> 16
        a = action_raw / float(1 << ACTION_FRAC_BITS)
        return cls(
            index=index,
            obs=obs_raw.astype(np.float32) / float(1 << OBS_FRAC_BITS),
            obs_raw=obs_raw,
            action_raw=action_raw,
            action=float(min(1.0, max(-1.0, a))),
            brake=bool(flags & 1),
            weight_bank=(flags >> 1) & 1,
            clipped=bool(flags & 4),
            time_us=int(w[4]),
            episode_step=int(w[5]),
            motor_counts=_to_int32(int(w[6])),
            episode=int(w[7]),
        )


class LogOverrun(RuntimeError):
    """Requested records were already overwritten in the PL ring buffer."""


class FpgaDeployClient:
    """Hardware client for the on-PL policy (deployment bitstream)."""

    def __init__(
        self,
        bitstream_path: str | Path = "/home/xilinx/jupyter_notebooks/RIP/rotary-inverted-pendulum/reinforcement_learning/Deploy/FPGA/deployment.xsa",
        download: bool = True,
        weight_base: int | None = None,
        wait_for_arm: bool = True,
    ):
        from pynq import MMIO, Overlay

        bit_path = Path(bitstream_path).resolve()
        if not bit_path.exists():
            raise FileNotFoundError(f"Hardware bitstream/XSA not found at: {bit_path}")
        print(f"Programming FPGA overlay from: {bit_path.name} (download={download})")
        self.ol = Overlay(str(bit_path), download=download)

        try:
            gpio0 = self.ol.ip_dict["axi_gpio_0"]["phys_addr"]   # s1_pos, s1_vel
            gpio1 = self.ol.ip_dict["axi_gpio_1"]["phys_addr"]   # s2_cos, s2_sin
            gpio2 = self.ol.ip_dict["axi_gpio_2"]["phys_addr"]   # s2_vel, F_prev
        except KeyError as err:
            raise KeyError(f"Missing expected AXI GPIO IP: {err}. "
                           f"Discovered blocks: {list(self.ol.ip_dict)}") from err
        self.gpio_motor = MMIO(gpio0, 0x1000)
        self.gpio_trig  = MMIO(gpio1, 0x1000)
        self.gpio_pen   = MMIO(gpio2, 0x1000)

        if weight_base is None:
            weight_base = self._find_weight_base()
        self.mmio = MMIO(weight_base, WEIGHT_RANGE)
        self.weights = WeightLoader(mmio=self.mmio)          # checks the bitstream ID

        info = self.mmio.read(REG_LOG_INFO)
        self.log_depth = info & 0xFFFF
        self.log_words = info >> 16

        self._lock = threading.RLock()
        self._ctrl = 0
        # Take control with the motor off; from here on SW1 is only the arm switch.
        self._write_ctrl(CTRL_PS_CONTROL)

        if wait_for_arm and not self.is_armed():
            try:
                input("Switch SW1 ON (arm) - the motor stays off until engage_motor(). Press enter.")
            except EOFError:
                pass
            if not self.is_armed():
                print("  [warn] SW1 is still off: engage_motor() will have no effect")

    def _find_weight_base(self) -> int:
        for name, d in {**self.ol.ip_dict, **getattr(self.ol, "mem_dict", {})}.items():
            if "WEIGHT_AXI" in name.upper() and "phys_addr" in d:
                return int(d["phys_addr"])
        return WEIGHT_BASE

    # ------------------------------------------------------------------
    # HAL
    # ------------------------------------------------------------------

    def _write_ctrl(self, c: int) -> None:
        self._ctrl = c & (CTRL_ENGAGE | CTRL_PS_CONTROL | CTRL_POLICY)
        self.mmio.write(REG_DEPLOY_CTRL, self._ctrl)

    def _write_accel_q(self, q: int) -> None:
        """Write the PS action (Q.9). Every write strobes the step generator."""
        self.mmio.write(REG_PS_ACCEL, int(q) & 0xFFFF)

    def _read_raw(self) -> tuple[int, int, int, int, int, int]:
        mp = _to_int16(self.gpio_motor.read(GPIO_CH1_DATA_OFFSET))
        mv = _to_int16(self.gpio_motor.read(GPIO_CH2_DATA_OFFSET))
        co = _to_int16(self.gpio_trig.read(GPIO_CH1_DATA_OFFSET))
        si = _to_int16(self.gpio_trig.read(GPIO_CH2_DATA_OFFSET))
        pv = _to_int16(self.gpio_pen.read(GPIO_CH1_DATA_OFFSET))
        fp = _to_int16(self.gpio_pen.read(GPIO_CH2_DATA_OFFSET))
        return mp, si, co, mv, pv, fp

    # ------------------------------------------------------------------
    # Motor / tare
    # ------------------------------------------------------------------

    def engage_motor(self) -> None:
        """Driver on with the policy OFF: the PS drives the motor through
        set_acceleration() (starts at 0 = standing still). No tare."""
        with self._lock:
            self._write_accel_q(0)
            self._write_ctrl(CTRL_PS_CONTROL | CTRL_ENGAGE)

    def set_acceleration(self, accel: float | int) -> bool:
        """Command acceleration in rad/s^2 while the policy is off (homing).
        NaN-guarded and clamped to +-ACCEL_FULL_SCALE; the PL rail brake still
        overrides it. Returns False (ignored) when disengaged or the policy runs."""
        with self._lock:
            if self._ctrl & CTRL_POLICY or not self._ctrl & CTRL_ENGAGE:
                return False
            a = float(accel)
            if a != a:
                a = 0.0
            a = max(-ACCEL_FULL_SCALE, min(ACCEL_FULL_SCALE, a))
            q = int(round(a / ACCEL_FULL_SCALE * (1 << ACTION_FRAC_BITS)))
            self._write_accel_q(max(-32768, min(32767, q)))
            return True

    def start_policy(self) -> None:
        """Driver on + PL policy (40 Hz). Starts a new episode in the step log
        (episode counter +1, episode_step from 0, prev_action = 0)."""
        with self._lock:
            self._write_accel_q(0)
            self._write_ctrl(CTRL_PS_CONTROL | CTRL_ENGAGE | CTRL_POLICY)

    def stop_policy(self) -> None:
        """Policy off, driver stays on and the PS acceleration takes over at 0.
        NOTE: accel 0 keeps the current motor velocity; follow with homing or
        disengage_motor()."""
        with self._lock:
            self._write_accel_q(0)
            self._write_ctrl(self._ctrl & ~CTRL_POLICY)

    def disengage_motor(self) -> None:
        """Driver off, policy off (also ends an episode)."""
        with self._lock:
            self._write_ctrl(CTRL_PS_CONTROL)
            self._write_accel_q(0)

    def disengage_safely(self) -> None:
        try:
            self.disengage_motor()
        except Exception:
            pass

    def tare_pendulum(self) -> bool:
        """Zero motor angle and pendulum now (pendulum must hang still).
        Under PS control this is the ONLY tare (engaging does not tare)."""
        with self._lock:
            self.mmio.write(REG_DEPLOY_CTRL, self._ctrl | CTRL_TARE)
            time.sleep(TARE_SETTLE_S)
        return True

    def is_engaged(self) -> bool:
        return bool(self.mmio.read(REG_DEPLOY_STATUS) & 1)

    def is_armed(self) -> bool:
        return bool(self.mmio.read(REG_DEPLOY_STATUS) & 2)

    def is_policy_running(self) -> bool:
        return bool(self.mmio.read(REG_DEPLOY_STATUS) & 4)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> DeployState:
        """Live state from the 2 ms telemetry (also valid while disengaged).
        Individual 16-bit fields are read separately, so they can come from
        two consecutive samples; use the step log for training data."""
        with self._lock:
            mp, si, co, mv, pv, fp = self._read_raw()
        return DeployState(
            motor_pos_rad=q16_to_float(mp),
            sin_theta=q16_to_float(si),
            cos_theta=q16_to_float(co),
            motor_vel_rad_s=q16_to_float(mv),
            pendulum_vel_rad_s=q16_to_float(pv),
            prev_action=q16_to_float(fp, ACTION_FRAC_BITS),
        )

    # ------------------------------------------------------------------
    # Step log
    # ------------------------------------------------------------------

    def log_count(self) -> int:
        """Number of records the PL has written since power-up."""
        return self.mmio.read(REG_LOG_COUNT)

    def read_records(self, start: int, stop: int | None = None) -> list[StepRecord]:
        """Records with global index start <= i < stop (default: up to the newest)."""
        with self._lock:
            count = self.log_count()
            stop = count if stop is None else min(stop, count)
            if stop <= start:
                return []
            if count - start > self.log_depth:
                raise LogOverrun(f"records {start}..{count - self.log_depth - 1} were overwritten")
            base = LOG_BASE // 4
            n = self.log_words
            words = np.empty((stop - start, n), dtype=np.uint32)
            i = start
            while i < stop:                            # at most two slices (ring wrap)
                slot = i % self.log_depth
                k = min(stop - i, self.log_depth - slot)
                a = base + slot * n
                words[i - start:i - start + k] = np.array(self.mmio.array[a:a + k * n],
                                                           dtype=np.uint32).reshape(k, n)
                i += k
            # the PL may have lapped us while we were reading
            if self.log_count() - start > self.log_depth:
                raise LogOverrun("PL overwrote records while they were being read")
        return [StepRecord.from_words(start + j, words[j]) for j in range(len(words))]

    def latest_record(self) -> StepRecord | None:
        c = self.log_count()
        return self.read_records(c - 1, c)[0] if c else None

    def wait_for_records(self, after: int, timeout: float = 1.0, poll_s: float = 0.002) -> list[StepRecord]:
        """Block until at least one record with index >= `after` exists; return all of them."""
        t0 = time.monotonic()
        while self.log_count() <= after:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no new PL step (is SW1 armed and the motor engaged?)")
            time.sleep(poll_s)
        return self.read_records(after)

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def load_weights(self, words: np.ndarray, verify: bool = True) -> WeightStatus:
        """Upload a packed weight image (weight_loader.pack_weights /
        actor_to_pl_words) and make it active at the next inference boundary."""
        return self.weights.load(words, verify=verify)

    # ------------------------------------------------------------------
    # Lifetime
    # ------------------------------------------------------------------

    def wait_until_ready(self, **_) -> bool:
        return True

    def close(self) -> None:
        # Keep PS control (motor off): releasing it with SW1 on would re-engage.
        self.disengage_safely()

    def __enter__(self) -> "FpgaDeployClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

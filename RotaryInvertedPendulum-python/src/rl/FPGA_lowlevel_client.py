"""FpgaLowLevelClient: PYNQ client mirroring the bare-metal C low-level server.

Behaviour matches the C firmware (ll_poll / handle_set_accel):
  * accel is given in rad/s^2, NaN-guarded, clamped to +-ACCEL_FULL_SCALE and
    normalised to Q8.8 (accel / ACCEL_FULL_SCALE) before being written
  * CTRL_ACCEL_STB is pulsed after every accepted accel write
  * rail brake at +-SAFE_LIMIT_RAD on the raw-frame motor position
  * accel commands are ignored while disengaged
  * tare waits for the PL to apply it before returning
  * state is returned with the server's output sign flip (optional)
"""

from __future__ import annotations

import math
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


def _to_int16(val: int) -> int:
    """Interpret an unsigned 16-bit register readout as signed."""
    val &= 0xFFFF
    return val - 0x10000 if val & 0x8000 else val


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
    motor_pos_rad: float
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float


class FpgaLowLevelClient:
    """Hardware client for the FPGA cores using 16-bit Q-format fixed point."""

    def __init__(
        self,
        bitstream_path: str | Path = "/home/xilinx/jupyter_notebooks/RIP/rotary-inverted-pendulum/RotaryInvertedPendulum-FPGA/Training_pipeline.xsa",
        download: bool = True,
        pos_frac_bits: int = 12,     # Q3.12 rad
        vel_frac_bits: int = 7,      # Q9.7  rad/s
        accel_frac_bits: int = 9,    # Q8.8  normalised action
        flip_sign: bool = True,      # match LowLevelServer::sendState() output flip
        safety: bool = True,         # NaN guard, clamp and rail brake
        port: str | None = None,     # compatibility shim
        baud: int | None = None,     # compatibility shim
    ):
        self.pos_frac_bits = pos_frac_bits
        self.vel_frac_bits = vel_frac_bits
        self.accel_frac_bits = accel_frac_bits
        self.flip_sign = flip_sign
        self.safety = safety

        bit_path = Path(bitstream_path).resolve()
        if not bit_path.exists():
            raise FileNotFoundError(f"Hardware bitstream/XSA not found at: {bit_path}")

        print(f"Programming FPGA overlay from: {bit_path.name} (download={download})")
        self.ol = Overlay(str(bit_path), download=download)

        try:
            gpio_step_addr = self.ol.ip_dict["axi_gpio_step"]["phys_addr"]
            gpio_pen_addr  = self.ol.ip_dict["axi_gpio_pen"]["phys_addr"]
            gpio_ctrl_addr = self.ol.ip_dict["axi_gpio_ctrl"]["phys_addr"]
        except KeyError as err:
            available = list(self.ol.ip_dict.keys())
            raise KeyError(
                f"Missing expected AXI GPIO IP: {err}. Discovered blocks: {available}"
            ) from err

        self.gpio_step = MMIO(gpio_step_addr, 0x1000)
        self.gpio_pen  = MMIO(gpio_pen_addr,  0x1000)
        self.gpio_ctrl = MMIO(gpio_ctrl_addr, 0x1000)

        # RLock so public methods can call each other while holding the lock.
        self._lock = threading.RLock()
        self._ctrl_state = 0      # persistent (level) bits only; pulses never stored
        self._engaged = False

        # Known-safe initial state, like main() in the C firmware.
        self._hal_write_accel_raw(0)
        self._hal_write_ctrl(0)

    # ------------------------------------------------------------------
    # HAL
    # ------------------------------------------------------------------

    def _hal_read_raw(self) -> tuple[int, int, int, int]:
        """Return signed raw (mp, pp, mv, pv) register values."""
        mp = _to_int16(self.gpio_step.read(GPIO_CH1_DATA_OFFSET))
        mv = _to_int16(self.gpio_step.read(GPIO_CH2_DATA_OFFSET))
        pp = _to_int16(self.gpio_pen.read(GPIO_CH1_DATA_OFFSET))
        pv = _to_int16(self.gpio_pen.read(GPIO_CH2_DATA_OFFSET))
        return mp, pp, mv, pv

    def _hal_read_state(self) -> LowLevelState:
        mp, pp, mv, pv = self._hal_read_raw()
        s = -1.0 if self.flip_sign else 1.0
        return LowLevelState(
            motor_pos_rad=s * q16_to_float(mp, self.pos_frac_bits),
            pendulum_pos_rad=s * q16_to_float(pp, self.pos_frac_bits),
            motor_vel_rad_s=s * q16_to_float(mv, self.vel_frac_bits),
            pendulum_vel_rad_s=s * q16_to_float(pv, self.vel_frac_bits),
        )

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
            mp_raw, _, _, _ = self._hal_read_raw()
            mp_rad = q16_to_float(mp_raw, self.pos_frac_bits)
            if mp_rad >= SAFE_LIMIT_RAD:
                a = -BRAKE_ACCEL
            elif mp_rad <= -SAFE_LIMIT_RAD:
                a = BRAKE_ACCEL

        accel_q = float_to_q16(a / ACCEL_FULL_SCALE, self.accel_frac_bits)
        self._hal_write_accel_raw(accel_q)
        self._pulse(CTRL_ACCEL_STB)    # data write is ordered before the strobe
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

    def engage_motor(self) -> None:
        with self._lock:
            self._engaged = True
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
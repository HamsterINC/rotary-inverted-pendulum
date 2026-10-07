"""Python client for the LowLevelServer Arduino sketch.

Speaks the binary protocol defined at the top of
RotaryInvertedPendulum-arduino/LowLevelServer/LowLevelServer.ino:
    0x01 READY               -> board echoes 0x01
    0x02 GET_STATE           -> board returns 20 bytes:
                                  uint32 time_us
                                  float32 motor_pos_rad
                                  float32 pendulum_pos_rad
                                  float32 motor_vel_rad_s
                                  float32 pendulum_vel_rad_s
    0x03 SET_ACCEL + float   commanded angular acceleration in rad/s²
    0x04 ENGAGE_MOTOR
    0x05 DISENGAGE_MOTOR

Sign convention: LowLevelServer flips the sign of motor and pendulum
positions AND velocities on output. We pass the raw bytes through;
callers decide whether to apply additional transforms. The deploy code
(`run_policy.py`) and the real-env (`real_env.py`) un-flip on read so
the observation matches the sim convention.

This client is request-driven (one 20-byte read per get_state call).
Sustained bandwidth is tiny (~2 kB/s at 100 Hz), so 2 Mbaud is
comfortably over-provisioned.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

import serial


CMD_READY = 0x01
CMD_GET_STATE = 0x02
CMD_SET_ACCEL = 0x03  # was CMD_SET_TARGET (position-mode); now angular acceleration in rad/s²
CMD_ENGAGE_MOTOR = 0x04
CMD_DISENGAGE_MOTOR = 0x05
CMD_TARE_PENDULUM = 0x06

# time_us (uint32 LE), motor_pos, pen_pos, motor_vel, pen_vel (4× float32 LE).
_STATE_STRUCT = struct.Struct("<Iffff")
_STATE_SIZE = _STATE_STRUCT.size  # 20 bytes


@dataclass(frozen=True)
class State:
    time_us: int           # Arduino's micros() when the read was taken
    motor_pos_rad: float   # signed motor angle, server-flipped sign convention
    pendulum_pos_rad: float
    motor_vel_rad_s: float
    pendulum_vel_rad_s: float


class LowLevelClient:
    """Drives the LowLevelServer sketch over a serial port.

    Use as a context manager:

        with LowLevelClient(port) as c:
            c.wait_until_ready()
            c.engage_motor()
            c.set_acceleration(50.0)   # rad/s² — see RL_PLAN's accel-mode entry
            s = c.get_state()
            ...
            c.disengage_motor()  # also called automatically by __exit__
    """

    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.5):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None
        # Serial transactions are guarded by a lock so the client is safe to
        # use from multiple threads (e.g. an async control loop where one
        # thread does periodic get_state/set_acceleration while the main
        # thread may issue an emergency disengage_motor on signal).
        self._lock = threading.Lock()

    def __enter__(self) -> "LowLevelClient":
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Bootloader reset on serial open + setup() (which blocks on the AS5600
        # magnet) typically takes ~2 s.
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ser is not None:
            try:
                # Best-effort safe-state on exit: stop the motor.
                self.disengage_motor()
            except Exception:
                pass
            self._ser.close()
            self._ser = None

    @property
    def ser(self) -> serial.Serial:
        if self._ser is None:
            raise RuntimeError("LowLevelClient must be used as a context manager")
        return self._ser

    # --- Handshake -------------------------------------------------------

    def wait_until_ready(self, *, retries: int = 10, retry_delay_s: float = 0.3) -> bool:
        """Send READY until the board echoes it back. Returns True on success."""
        for _ in range(retries):
            with self._lock:
                self.ser.reset_input_buffer()
                self.ser.write(bytes([CMD_READY]))
                self.ser.flush()
                resp = self.ser.read(1)
            if resp == bytes([CMD_READY]):
                return True
            time.sleep(retry_delay_s)
        return False

    # --- Commands --------------------------------------------------------

    def get_state(self) -> State:
        """Request a state reading. Blocks until 12 bytes arrive or timeout."""
        with self._lock:
            self.ser.write(bytes([CMD_GET_STATE]))
            self.ser.flush()
            buf = self.ser.read(_STATE_SIZE)
        if len(buf) != _STATE_SIZE:
            raise RuntimeError(
                f"get_state: expected {_STATE_SIZE} bytes, got {len(buf)}. "
                "Is the LowLevelServer sketch flashed? Does the magnet detect?"
            )
        time_us, motor_rad, pen_rad, motor_vel, pen_vel = _STATE_STRUCT.unpack(buf)
        return State(time_us=int(time_us), motor_pos_rad=float(motor_rad),
                     pendulum_pos_rad=float(pen_rad),
                     motor_vel_rad_s=float(motor_vel),
                     pendulum_vel_rad_s=float(pen_vel))

    def set_acceleration(self, accel_rad_s2: float) -> None:
        """Command a new motor angular acceleration, in rad/s².

        Maps to FastAccelStepper's moveByAcceleration() with the matching
        int32 steps/s² on the firmware side. The stepper will accelerate
        smoothly toward the appropriate velocity rail (set once at boot via
        setSpeedInUs), passing through zero on direction reversal without
        any extra handling. Replaced the previous position-target command
        as of the accel-mode switch — see RL_PLAN.md.

        No `flush()` after `write()`: the 5-byte payload sits in the OS
        USB buffer until transmitted, but at our rate (50-100 Hz × 5 B =
        250-500 B/s versus a 2 Mbaud link) the buffer never approaches
        backing up, and the bytes are guaranteed on the wire before the
        next tick's `get_state` round-trip starts. Removes ~1 ms of
        macOS tcdrain overhead per tick. The other commands
        (engage/disengage/READY handshake) keep the flush because they
        run once per session, not per tick, and benefit from the
        deterministic completion-by-return semantics.
        """
        payload = bytes([CMD_SET_ACCEL]) + struct.pack("<f", float(accel_rad_s2))
        with self._lock:
            self.ser.write(payload)

    def engage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_ENGAGE_MOTOR]))
            self.ser.flush()

    def disengage_motor(self) -> None:
        with self._lock:
            self.ser.write(bytes([CMD_DISENGAGE_MOTOR]))
            self.ser.flush()

    def tare_pendulum(self, *, timeout_s: float = 0.5) -> bool:
        """Re-zero pen_position_rad to the AS5600's current reading.

        Used by `real_env.reset()` so each fine-tune episode samples a
        fresh bias from the rig's actual stiction-bounded rest
        distribution. The firmware atomically shifts every entry in its
        pendulum ring buffer by the current pen_position_rad, so
        velocity computation continues uninterrupted across the tare.

        Blocks until the Arduino acks (single-byte echo of the command)
        or `timeout_s` elapses. Returns True on success.
        """
        with self._lock:
            self.ser.write(bytes([CMD_TARE_PENDULUM]))
            self.ser.flush()
            resp = self.ser.read(1)
        return resp == bytes([CMD_TARE_PENDULUM])

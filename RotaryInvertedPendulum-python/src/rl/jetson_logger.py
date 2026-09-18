"""High-rate telemetry logger for Jetson embedded deployment."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any
import numpy as np


class JetsonTelemetryLogger:
    """Buffers and flushes synchronized hardware & simulation telemetry."""

    FIELDNAMES = [
        "time_s",
        "arm_pos_rad",
        "arm_vel_rad_s",
        "pendulum_pos_rad",
        "pendulum_angle_unwrapped",
        "control_action",
        "sim_cart_pos",
        "sim_pole_angle",
        "dyn_ghost_sim_pole_angle",
    ]

    def __init__(self, output_dir: str | Path = "/tmp", tag: str = "sysid_accel"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"{tag}_{timestamp}.csv"
        self.npz_path = self.output_dir / f"{tag}_{timestamp}.npz"

        self._records: list[dict[str, float]] = []
        self._prev_raw_pole: float | None = None
        self._wrap_offset: float = 0.0

    def unwrap_angle(self, raw_angle: float) -> float:
        """Continuous phase tracking across [-pi, pi] rolls."""
        if self._prev_raw_pole is None:
            self._prev_raw_pole = raw_angle
            return raw_angle

        diff = raw_angle - self._prev_raw_pole
        if diff > np.pi:
            self._wrap_offset -= 2.0 * np.pi
        elif diff < -np.pi:
            self._wrap_offset += 2.0 * np.pi

        self._prev_raw_pole = raw_angle
        return raw_angle + self._wrap_offset

    def log(
        self,
        t_s: float,
        arm_pos: float,
        arm_vel: float,
        pen_pos: float,
        accel_cmd: float,
        sim_cart_pos: float = 0.0,
        sim_pole_angle: float = 0.0,
        dyn_ghost_pole: float = 0.0,
    ) -> None:
        """Append one frame of real-time telemetry."""
        unwrapped_pen = self.unwrap_angle(pen_pos)
        record = {
            "time_s": float(t_s),
            "arm_pos_rad": float(arm_pos),
            "arm_vel_rad_s": float(arm_vel),
            "pendulum_pos_rad": float(pen_pos),
            "pendulum_angle_unwrapped": float(unwrapped_pen),
            "control_action": float(accel_cmd),
            "sim_cart_pos": float(sim_cart_pos),
            "sim_pole_angle": float(sim_pole_angle),
            "dyn_ghost_sim_pole_angle": float(dyn_ghost_pole),
        }
        self._records.append(record)

    def write_csv(self) -> Path:
        """Write all logged data to disk as CSV."""
        if not self._records:
            return self.csv_path

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._records)

        return self.csv_path

    def write_npz(self, **extra_arrays: Any) -> Path:
        """Write records to compressed NPZ archive with optional auxiliary arrays."""
        data_dict: dict[str, Any] = {}
        for key in self.FIELDNAMES:
            data_dict[key] = np.array([r[key] for r in self._records], dtype=np.float64)

        data_dict.update(extra_arrays)
        np.savez_compressed(self.npz_path, **data_dict)
        return self.npz_path

    def to_dataframe_dict(self) -> dict[str, np.ndarray]:
        """Convert current records to dict of 1D numpy arrays."""
        return {
            key: np.array([r[key] for r in self._records], dtype=np.float64)
            for key in self.FIELDNAMES
        }
"""Wait for a still pendulum between episodes (shared by FPGA_env and FPGA_deploy_env).

Still = every sample of the last `duration_s` has |pen_vel| < `vel_threshold`
AND all their angles lie within an `angle_range` band. The angle band catches
a small residual swing that the velocity check alone lets through (0.05 rad
amplitude peaks at ~0.3 rad/s), which would otherwise be tared in as an
angle offset. Returns as soon as that holds; gives up after `max_wait_s`.
With max_wait_s = 0 it is a single check window: tare only if the pendulum
happens to be still, never wait for it.
"""

from __future__ import annotations

import math
import time
from collections import deque

REST_THRESHOLD_RAD_S = 0.5
REST_ANGLE_RANGE_RAD = 0.03     # ~1.7 deg peak-to-peak
REST_DURATION_S = 1.0
REST_POLL_S = 0.02


def wait_until_still(read_angle_vel, *, max_wait_s: float,
                     vel_threshold: float = REST_THRESHOLD_RAD_S,
                     angle_range: float = REST_ANGLE_RANGE_RAD,
                     duration_s: float = REST_DURATION_S,
                     poll_s: float = REST_POLL_S,
                     should_stop=lambda: False) -> tuple[bool, float, str]:
    """Poll `read_angle_vel() -> (angle_rad, pen_vel_rad_s)` until still.

    Returns (rested, waited_s, detail). Raises KeyboardInterrupt if
    `should_stop()` becomes true while waiting.
    """
    need = max(2, int(round(duration_s / poll_s)))
    # Always allow one full check window (+ polling slack).
    max_wait_s = max(float(max_wait_s), need * poll_s + 3 * poll_s)
    window: deque = deque(maxlen=need)   # angles of the current still streak
    t0 = time.monotonic()
    last_vel, last_range = 0.0, float("nan")
    n_samples = 0
    # Time limit, but never fewer samples than one window (coarse sleep timers).
    while time.monotonic() - t0 < max_wait_s or n_samples < need + 1:
        n_samples += 1
        if should_stop():
            raise KeyboardInterrupt("stop requested while waiting for rest")
        angle, last_vel = read_angle_vel()
        if abs(last_vel) >= vel_threshold:
            window.clear()
        else:
            window.append(angle)
            if len(window) == need:
                ref = window[-1]         # wrap-safe spread around the newest angle
                d = [math.remainder(a - ref, 2.0 * math.pi) for a in window]
                last_range = max(d) - min(d)
                if last_range < angle_range:
                    return True, time.monotonic() - t0, ""
        time.sleep(poll_s)
    detail = f"last |pen_vel|={abs(last_vel):.3f} rad/s"
    if not math.isnan(last_range):
        detail += f", angle range {last_range:.3f} rad"
    return False, time.monotonic() - t0, detail

"""Shared sysid math helpers used by `sysid_wizard.py`.

This module is the single source of truth for:
  - the damped-pendulum free-swing fit (`fit_free_swing`)
  - elliptic-corrected friction derivation (`derive_pendulum_friction`)
  - aggregation across multiple free-swing runs
  - sanity-check / validation messages

Pendulum geometry (mass, COM, I_com) is *not* fit here — it comes from
the URDF via `pendulum_geometry`. The sysid pipeline only measures
quantities that genuinely vary per-rig: viscous and Coulomb friction.

Nothing here prints or prompts. The wizard owns all I/O.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks
from scipy.special import ellipk


GRAVITY = 9.81  # m/s^2


def _elliptic_period_factor(amplitude_rad: float) -> float:
    """Return T(amplitude) / T₀ for a simple gravitational pendulum.

    The exact period of a pendulum with peak amplitude θ_max is
        T(θ_max) = T₀ · (2/π) · K(sin²(θ_max/2))
    where K is the complete elliptic integral of the first kind and
    T₀ = 2π·sqrt(I/(m·g·d)) is the small-amplitude period. For θ_max → 0
    the factor → 1; at 90° (π/2) it's ≈ 1.18.

    Used to correct the measured period (taken at finite amplitude) to
    the small-amplitude T₀ that enters the inertia formula.
    """
    m = math.sin(amplitude_rad / 2.0) ** 2
    return float((2.0 / math.pi) * ellipk(m))

DEFAULT_FREE_SWING_DURATION_S: float = 10.0
DEFAULT_FREE_SWING_RUNS: int = 3


# ---------------------------------------------------------------------------
# Free-swing fit
# ---------------------------------------------------------------------------


def fit_free_swing(time_s: np.ndarray, pendulum_rad: np.ndarray) -> dict:
    """Fit damped-oscillator parameters to free-swing data.

    Period: average spacing between consecutive same-side extrema.

    Damping: fits a combined viscous + Coulomb friction model. The peak
    envelope of a pendulum with viscous (b·θ̇) and Coulomb (F_c·sign(θ̇))
    friction obeys
        |P_{n+1}| = r * |P_n| - delta_c
    per half-cycle, where r = exp(-π zeta / sqrt(1-zeta^2)) (viscous) and
    delta_c = 2 F_c / (m g l) (Coulomb step). This is linear in (r, delta_c)
    and fits trivially via least squares. Pure-viscous log-decrement is also
    computed for diagnostic comparison.
    """
    n = len(pendulum_rad)
    rest = float(np.mean(pendulum_rad[4 * n // 5:]))
    p = pendulum_rad - rest

    abs_max = float(np.max(np.abs(p)))
    if abs_max < 1e-3:
        raise RuntimeError(
            "Pendulum did not move appreciably during the recording. "
            "Did you displace it before release?"
        )
    prominence = max(0.02 * abs_max, 5e-3)

    pos_peaks, _ = find_peaks(p, prominence=prominence)
    neg_peaks, _ = find_peaks(-p, prominence=prominence)
    extrema = np.sort(np.concatenate([pos_peaks, neg_peaks]))
    if len(extrema) < 4:
        raise RuntimeError(
            f"Only {len(extrema)} oscillation extrema detected; need at least 4. "
            "Record longer or release with a larger initial angle."
        )

    extrema_t = time_s[extrema]
    extrema_v = p[extrema]

    # Per-pair period from same-side peaks (peak-to-peak across a full cycle).
    same_side_times = extrema_t[::2]
    same_side_amps = np.abs(extrema_v[::2])
    same_side = np.diff(same_side_times)
    period_s = float(np.mean(same_side))
    if period_s <= 0:
        raise RuntimeError("Failed to estimate a positive oscillation period.")
    omega_d = 2.0 * math.pi / period_s

    # Small-amplitude T₀ via the elliptic-integral correction. The measured
    # period T(θ) at finite amplitude θ is longer than T₀; without this
    # correction the derived inertia is biased upward at large amplitudes
    # (the small-amplitude formula I = m·g·d·T²/(4π²) needs T₀, not T(θ)).
    # We compute T₀ per consecutive same-side peak pair using the mean of
    # their amplitudes, then average. Falls back to the raw period if any
    # of the corrections fail (e.g., amplitude > π).
    pair_amps = (same_side_amps[:-1] + same_side_amps[1:]) / 2.0
    try:
        t0_estimates = np.array([
            period / _elliptic_period_factor(a)
            for period, a in zip(same_side, pair_amps)
            if a < math.pi
        ])
        small_amp_period_s = float(np.mean(t0_estimates)) if len(t0_estimates) else period_s
    except Exception:
        small_amp_period_s = period_s
    omega_n_small_amp = 2.0 * math.pi / small_amp_period_s

    log_amp = np.log(np.abs(extrema_v))
    A = np.vstack([extrema_t, np.ones_like(extrema_t)]).T
    slope, intercept = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    alpha_visc_only = -float(slope)
    if alpha_visc_only <= 0:
        raise RuntimeError(
            "Decay rate is non-positive; the recording does not show a "
            "decaying envelope. Check that the pendulum is freely swinging."
        )

    abs_peaks = np.abs(extrema_v)
    Pn = abs_peaks[:-1]
    Pn1 = abs_peaks[1:]
    M = np.vstack([Pn, np.ones_like(Pn)]).T
    sol, *_ = np.linalg.lstsq(M, Pn1, rcond=None)
    r_per_half = float(np.clip(sol[0], 1e-6, 1.0 - 1e-9))
    delta_c_per_half = float(max(-sol[1], 0.0))

    log_r = -math.log(r_per_half)
    zeta = log_r / math.pi
    for _ in range(8):
        zeta = log_r / math.pi * math.sqrt(max(1 - zeta * zeta, 1e-12))
    omega_n_combined = omega_d / math.sqrt(max(1 - zeta * zeta, 1e-12))

    pred_combined = np.empty_like(abs_peaks)
    pred_combined[0] = abs_peaks[0]
    for i in range(1, len(pred_combined)):
        pred_combined[i] = max(r_per_half * pred_combined[i - 1] - delta_c_per_half, 0.0)
    pred_visc_only = abs_peaks[0] * np.exp(-alpha_visc_only * (extrema_t - extrema_t[0]))
    env_rmse_combined = float(np.sqrt(np.mean((pred_combined - abs_peaks) ** 2)))
    env_rmse_visc_only = float(np.sqrt(np.mean((pred_visc_only - abs_peaks) ** 2)))

    return {
        "n_extrema_used": int(len(extrema)),
        "period_s": period_s,
        "small_amp_period_s": small_amp_period_s,
        "mean_pair_amplitude_rad": float(np.mean(pair_amps)) if len(pair_amps) else 0.0,
        "omega_d_rad_s": omega_d,
        "omega_n_rad_s": omega_n_combined,
        "omega_n_small_amp_rad_s": omega_n_small_amp,
        "zeta": zeta,
        "rest_angle_rad": rest,
        "decay_constant_s_inv": float(zeta * omega_n_combined),
        "initial_amplitude_rad": float(math.exp(intercept)),
        "r_per_half_cycle": r_per_half,
        "delta_c_per_half_cycle_rad": delta_c_per_half,
        "decay_constant_visc_only_s_inv": alpha_visc_only,
        "envelope_rmse_combined_rad": env_rmse_combined,
        "envelope_rmse_visc_only_rad": env_rmse_visc_only,
    }


def derive_pendulum_friction(fit: dict, *, gravity: float = GRAVITY) -> dict:
    """Derive viscous + Coulomb friction from a free-swing fit, using the
    URDF-defined pendulum geometry (mass, COM, I_com) as constants.

    Mass and COM no longer come from operator-typed measurements; they're
    geometric properties of the pendulum body, set by Onshape CAD and
    materialised in `urdf/model.urdf`. The sysid pipeline only measures
    things that genuinely vary per-rig (bearings, grease, temperature).

    Inertia formulas:
    - I_predicted = m·d² + I_com_swing — the CAD model.
    - I_measured  = m·g·d / ω_n_small² — from the small-amplitude period
      (elliptic-corrected from the finite-amplitude fit). Reported for
      sanity-check against I_predicted; a mismatch points at a stale
      URDF or a swap of pendulum geometry.

    Friction uses I_predicted (the CAD-constant pivot inertia) so the
    derived viscous/Coulomb numbers are decoupled from per-run period
    noise:
    - viscous b   = 2·α·I_predicted (decay envelope: b/(2I) = α).
    - Coulomb F_c = δ_c · m · g · d / 2 (Coulomb step per half-cycle).
    """
    # Local import keeps this module stdlib-only at import time.
    from pendulum_geometry import (
        PENDULUM_COM_M,
        PENDULUM_I_COM_SWING_KG_M2,
        PENDULUM_MASS_KG,
    )

    omega_n_small = fit.get("omega_n_small_amp_rad_s", fit["omega_n_rad_s"])
    decay = float(fit["decay_constant_s_inv"])  # = α = b/(2I)

    inertia_predicted = (
        PENDULUM_MASS_KG * PENDULUM_COM_M ** 2 + PENDULUM_I_COM_SWING_KG_M2
    )
    inertia_measured = (
        PENDULUM_MASS_KG * gravity * PENDULUM_COM_M / (omega_n_small ** 2)
    )
    viscous = 2.0 * decay * inertia_measured

    delta_c = float(fit.get("delta_c_per_half_cycle_rad", 0.0))
    coulomb = delta_c * PENDULUM_MASS_KG * gravity * PENDULUM_COM_M / 2.0

    return {
        "viscous_friction_N_m_s": viscous,
        "coulomb_friction_N_m": coulomb,
        "inertia_predicted_kg_m2": inertia_predicted,
        "inertia_measured_kg_m2": inertia_measured,
    }


# ---------------------------------------------------------------------------
# Aggregation across multiple free-swing runs
# ---------------------------------------------------------------------------

# Keys for which we report a median + stdev across runs and treat the median
# as the canonical value. All other keys come from the median run for
# interpretability (no point averaging "n_extrema_used", etc.).
_AGGREGATE_KEYS: tuple[str, ...] = (
    "period_s",
    "small_amp_period_s",
    "omega_d_rad_s",
    "omega_n_rad_s",
    "omega_n_small_amp_rad_s",
    "zeta",
    "decay_constant_s_inv",
    "r_per_half_cycle",
    "delta_c_per_half_cycle_rad",
)


def aggregate_free_swing_fits(fits: list[dict]) -> dict:
    """Combine N free-swing fits into a single canonical fit dict.

    Returns a dict that is schema-compatible with the single-run output of
    `fit_free_swing` (so downstream consumers keep working). The canonical
    value for each aggregate key is the median across runs; per-run lists
    and stdevs are exposed under `*_all_runs` / `*_stdev` for diagnostics.
    """
    if not fits:
        raise ValueError("aggregate_free_swing_fits called with no fits")
    if len(fits) == 1:
        out = dict(fits[0])
        out["n_runs"] = 1
        return out

    # Pick the run whose period is closest to the median as the donor of the
    # non-aggregate scalar diagnostics (rest_angle_rad, initial_amplitude_rad,
    # envelope RMSEs, etc.). The aggregate keys then overwrite with medians.
    periods = np.asarray([f["period_s"] for f in fits], dtype=float)
    median_idx = int(np.argmin(np.abs(periods - float(np.median(periods)))))
    out = dict(fits[median_idx])

    for k in _AGGREGATE_KEYS:
        values = np.asarray([f[k] for f in fits], dtype=float)
        out[k] = float(np.median(values))
        out[f"{k}_stdev"] = float(np.std(values, ddof=0))
        out[f"{k}_all_runs"] = [float(v) for v in values]

    out["n_runs"] = len(fits)
    out["median_run_index"] = median_idx
    return out


# ---------------------------------------------------------------------------
# Sanity-check / validation
# ---------------------------------------------------------------------------

@dataclass
class Warning_:
    """A sysid validation message. severity is 'info' | 'warn' | 'error'."""

    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def validate_free_swing(fit: dict, derived: dict) -> list[Warning_]:
    warnings: list[Warning_] = []
    period = float(fit["period_s"])
    zeta = float(fit["zeta"])
    inertia_measured = float(derived["inertia_measured_kg_m2"])
    inertia_predicted = float(derived["inertia_predicted_kg_m2"])

    if not (0.3 <= period <= 0.7):
        warnings.append(Warning_(
            "warn",
            f"period_s = {period:.3f}s is outside the expected 0.3-0.7s range. "
            "Wildly different means something is off with the geometry, the release, "
            "or the motor arm being free to wiggle. Did you hold the arm firmly?",
        ))
    if zeta <= 0:
        warnings.append(Warning_(
            "error",
            f"zeta = {zeta:.4f} is non-positive. The recording does not show a "
            "decaying envelope; the pendulum may not be swinging freely."
        ))
    elif zeta >= 0.1:
        warnings.append(Warning_(
            "warn",
            f"zeta = {zeta:.4f} is high (>0.1). Expected light damping (<0.1) — "
            "check that the motor arm is held still and the bearings spin freely."
        ))

    # Measured vs CAD-predicted pivot inertia — flag divergence as a hint
    # that the URDF is stale (pendulum geometry changed) or the recording
    # was not a clean free-swing.
    rel_err = abs(inertia_measured - inertia_predicted) / inertia_predicted
    if rel_err > 0.10:
        warnings.append(Warning_(
            "warn",
            f"I_measured = {inertia_measured:.3e} kg·m² differs from CAD "
            f"I_predicted = {inertia_predicted:.3e} by {rel_err*100:.1f}% "
            "(>10%). Either the pendulum geometry has changed and "
            "urdf/model.urdf is stale, or the recording is contaminated "
            "(motor arm not held firmly, magnet brushing the boss, etc.)."
        ))

    return warnings


# ---------------------------------------------------------------------------
# Control-rate suggestion (from docs/control_rate_selection.md)
# ---------------------------------------------------------------------------

def suggest_control_rate(pendulum_period_s: float, motor_rise_time_s: float) -> dict:
    """Return suggested control rate window and a recommended operating point.

    Window: [5 * f_n, 3 * BW_motor] where f_n = 1/period_s and the motor's
    effective bandwidth is approximated as BW_motor ≈ 1/rise_time_95 — the
    rule of thumb in `docs/control_rate_selection.md` (the rig's measured 64
    ms rise corresponds to ~16 Hz effective bandwidth, empirically validated
    against the 35 Hz operating point). Recommended rate biases toward the
    lower edge; max_action_delta_rad chosen so slew ≤ BW_motor × 0.2.
    """
    f_n = 1.0 / pendulum_period_s
    bw_motor = (1.0 / motor_rise_time_s) if motor_rise_time_s > 0 else float("nan")
    lower = 5.0 * f_n
    upper = 3.0 * bw_motor
    if not math.isfinite(upper) or upper <= lower:
        return {
            "f_n_hz": f_n,
            "bw_motor_hz": bw_motor,
            "window_lower_hz": lower,
            "window_upper_hz": upper,
            "feasible": False,
            "recommended_rate_hz": None,
            "recommended_max_action_delta_rad": None,
            "recommended_slew_rad_s": None,
        }
    # Bias toward the lower edge: lower + 0.4 × window width.
    recommended = lower + 0.4 * (upper - lower)
    slew_cap = bw_motor * 0.2
    delta = min(0.10, slew_cap / recommended) if recommended > 0 else 0.10
    return {
        "f_n_hz": f_n,
        "bw_motor_hz": bw_motor,
        "window_lower_hz": lower,
        "window_upper_hz": upper,
        "feasible": True,
        "recommended_rate_hz": recommended,
        "recommended_max_action_delta_rad": delta,
        "recommended_slew_rad_s": delta * recommended,
    }


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def write_params_json(
    path: str,
    *,
    pendulum_payload: dict | None = None,
    run_meta: dict | None = None,
) -> dict:
    """Write/merge the sysid_params.json the wizard produces.

    Preserves the top-level `pendulum` block read by `pendulum_env.py`.
    `run_meta` is informational only (timestamp, source files, fitter
    version); downstream code ignores it.
    """
    doc: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
    if pendulum_payload is not None:
        doc["pendulum"] = pendulum_payload
    if run_meta is not None:
        doc["run_meta"] = run_meta
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    return doc


def source_file_for_json(recording_path: str, out_json_path: str) -> str:
    """Path to a recording, expressed relative to the JSON's directory.

    Keeps the JSON portable across machines / repo clones.
    """
    out_dir = Path(out_json_path).resolve().parent
    rec_path = Path(recording_path).resolve()
    return os.path.relpath(rec_path, out_dir)


def format_warnings(warnings: Iterable[Warning_]) -> str:
    """Render warnings as a human-readable indented block."""
    return "\n".join(f"  {w}" for w in warnings)

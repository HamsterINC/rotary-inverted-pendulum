"""Single source of truth for the RL reward function.

Both `pendulum_env.py` (sim training) and `real_env.py` (rig fine-tuning)
call into `compute_reward()` so that the gradient signal during fine-tune
matches the basin SAC found in sim. The previous duplicated `_reward`
implementations drifted on 2026-05-21 (sim env got the stillness bonus,
real env didn't) and silently undid every fine-tune attempt because SAC's
real-rig gradient pushed the policy out of the bonus basin back toward
whatever the simpler canonical reward preferred. Don't repeat that
mistake — keep this function as the only place the reward math lives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Canonical Quanser quadratic cost weights + optional extras.

    Defaults match the 2026-05-16 canonical reward (5-term Quanser form
    with the accel-mode action-rate disable). Set the extras (`action_rate`,
    `motor_jerk`, `stillness_bonus`) to enable specific add-ons.
    """
    # Canonical Quanser quadratic-cost form: cost = sum of k·x²
    k_pen_vel: float = 0.001     # pendulum angular velocity penalty (small)
    k_motor_pos: float = 0.5     # motor position penalty (centring)
    k_motor_vel: float = 0.005   # motor angular velocity penalty (small)
    k_action: float = 0.20       # action magnitude penalty

    # Extras (default 0 = disabled, preserves canonical 5-term reward).
    k_action_rate: float = 0.0   # penalty on (a_t - a_{t-1})² ("command jerk")
    k_motor_jerk: float = 0.0    # penalty on (motor_vel_t - motor_vel_{t-1})²

    # Multiplicative stillness bonus: ADDS k · exp(-θ²/σ_θ²) · exp(-α̇²/σ_v²).
    # Targets Kapitza-style resonance balance: the product is only large
    # when BOTH theta and motor_vel are near zero. Kapitza has α̇ ≈ several
    # rad/s during balance and loses the bonus accordingly.
    k_stillness_bonus: float = 0.0
    sigma_theta: float = 0.3              # bonus active within ~17°
    sigma_motor_vel: float = 1.0          # full bonus only at α̇ < ~1 rad/s

    # Per-step ALIVE OFFSET (2026-07-21 audit finding F1). The canonical
    # all-negative quadratic cost combined with hard-stop *termination*
    # makes early termination attractive: hanging for a full episode costs
    # ≈ −4000, crashing into the rail costs ≈ −200..−500 total, and even the
    # best balancing episodes score ≈ −140 — so "drive into the wall" is a
    # strong local optimum, especially under DR when swing-up succeeds less
    # (observed as the stage-2/3 collapses). A constant per-step offset
    # ≥ the worst realistic per-step cost (~14 on this rig: θ²≈9.9 at
    # hanging + motor-pos@rail ≈2.8 + velocity/action terms ≈1) makes the
    # per-step reward non-negative, so terminating always forfeits value.
    # Gymnasium's Pendulum-v1 gets away without this only because it never
    # terminates; the Furuta sim-to-real literature (IEEE Access 2023) and
    # the Quanser/MATLAB QUBE example both carry a positive offset/alive
    # term. Default 0 preserves the canonical reward.
    k_alive_offset: float = 0.0           # recommended: 15.0

    # Velocity-gated UPRIGHT ALIVE bonus (Quanser/MATLAB-style constraint-
    # gated alive term): ADDS k when |θ| ≤ alive_theta_band AND
    # |θ̇| ≤ alive_pen_vel_limit. Unlike the raw cos/θ² shaping, a pendulum
    # swinging *through* upright at speed earns nothing here, so
    # spin-through farming is unprofitable. Gates match the honest balance
    # metrics in analyze_deploy.py (|θ| ≤ 15°, |θ̇| ≤ 2 rad/s) so training
    # optimises exactly what deployment certifies. Complements the
    # stillness bonus above, which gates on MOTOR velocity (anti-Kapitza)
    # rather than pendulum velocity (anti-spin). Default 0 (disabled).
    k_upright_alive: float = 0.0          # recommended: 5.0
    alive_theta_band: float = 0.2618      # 15°
    alive_pen_vel_limit: float = 2.0      # rad/s


def compute_reward(
    *,
    theta: float,
    pen_vel: float,
    motor_pos: float,
    motor_vel: float,
    action: float,
    prev_action: float,
    prev_motor_vel: float,
    weights: RewardWeights,
) -> float:
    """Reward = stillness_bonus - quadratic_cost.

    Sign convention: theta = 0 is upright (after wrapping), action ∈ [-1, 1],
    motor_vel and pen_vel are SI angular velocity (rad/s). The pendulum-angle
    quadratic term has implicit weight 1.0 — `RewardWeights` weights everything
    else relative to it.
    """
    action_delta = action - prev_action
    motor_vel_delta = motor_vel - prev_motor_vel
    cost = (
        theta * theta
        + weights.k_pen_vel * pen_vel * pen_vel
        + weights.k_motor_pos * motor_pos * motor_pos
        + weights.k_motor_vel * motor_vel * motor_vel
        + weights.k_action * action * action
        + weights.k_action_rate * action_delta * action_delta
        + weights.k_motor_jerk * motor_vel_delta * motor_vel_delta
    )
    if weights.k_stillness_bonus > 0.0:
        sigma_theta_sq = weights.sigma_theta * weights.sigma_theta
        sigma_motor_vel_sq = weights.sigma_motor_vel * weights.sigma_motor_vel
        upright_score = math.exp(-(theta * theta) / sigma_theta_sq)
        stillness_score = math.exp(-(motor_vel * motor_vel) / sigma_motor_vel_sq)
        bonus = weights.k_stillness_bonus * upright_score * stillness_score
    else:
        bonus = 0.0
    alive = weights.k_alive_offset
    if (weights.k_upright_alive > 0.0
            and abs(theta) <= weights.alive_theta_band
            and abs(pen_vel) <= weights.alive_pen_vel_limit):
        alive += weights.k_upright_alive
    return float(alive + bonus - cost)

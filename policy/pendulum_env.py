"""Gymnasium environment for the rotary inverted pendulum, parameterised
from the system-identification fits in a per-rig sysid_params_*.json.

Geometry (Furuta pendulum):
    - Arm rotates about a vertical axis, driven by a position-controlled
      "motor" (modelling the AccelStepper + driver as a stiff PD).
    - Pendulum hinges at the end of the arm, free-swinging in the vertical
      plane perpendicular to the arm direction. Hangs straight down at rest.

Observation (5-dim):
    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel]
    where theta = 0 is upright (so cos(theta) = 1 at the goal).

Action (1-dim, in [-1, 1]) :
    - commanded angular acceleration in
      [-max_accel, +max_accel] rad/s²; integrated to a capped velocity and
      then a position target fed to the PD actuator.
    

Reward:
    upright term      = (1 + cos(theta)) / 2     in [0, 1]
    motor-pos penalty = -k_pos * (motor_pos / motor_limit)^2
    motor-vel penalty = -k_vel * motor_vel^2
    action penalty    = -k_act * action^2
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pendulum_geometry import (
    ARM_COM_M,
    ARM_I_COM_SPIN_KG_M2,
    ARM_LENGTH_M,
    ARM_MASS_KG,
    PENDULUM_COM_M,
    PENDULUM_I_COM_SWING_KG_M2,
    PENDULUM_MASS_KG,
)
from reward import RewardWeights, compute_reward


HERE = Path(__file__).resolve().parent
# Fallback when no `params_path` is given. Named for the rig it describes
# rather than a generic `sysid_params.json`, so a multi-rig setup cannot
# silently train against whichever rig happened to be measured last. Pass
# `--params-path` for any other rig; see reference/python.md.
DEFAULT_PARAMS_PATH = HERE / "sysid_params.json"

# Hard-stop on the motor joint. Matches the lid-boss mechanical limit of ±135°,
# but we clamp the policy at ±125° so the policy never *commands* a stop hit.
MOTOR_LIMIT_RAD = math.radians(135.0)
MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)

NORMILIZED_OBS = True  # If True, observation is normalized to [-1, 1] range per dimension


# Observation-normalization presets. Each feature maps to (kind, *params);
# missing features pass through raw. "gain" multiplies every feature last.
OBS_NORM_PRESETS = {
    "raw": {},
    "orig": {"feats": {"motor_pos": ("clipdiv", MOTOR_LIMIT_RAD),
                       "motor_vel": ("clipdiv", 7.5),
                       "pen_vel":   ("clipdiv", 32.0)}},
    "only_mp":   {"feats": {"motor_pos": ("clipdiv", MOTOR_LIMIT_RAD)}},
    "only_mv":   {"feats": {"motor_vel": ("clipdiv", 7.5)}},
    "only_pv32": {"feats": {"pen_vel": ("clipdiv", 32.0)}},
    "only_pv8":  {"feats": {"pen_vel": ("clipdiv", 8.0)}},
    "only_pv4":  {"feats": {"pen_vel": ("div", 4.0)}},
    "only_pv_asinh1":  {"feats": {"pen_vel": ("asinh", 1.0, 32.0)}},
    "only_pv_asinh10": {"feats": {"pen_vel": ("asinh", 10.0, 32.0)}},
    "raw_x0.1": {"gain": 0.1},
    "raw_x0.3": {"gain": 0.3},
    "raw_x3":   {"gain": 3.0},
    "orig_sin10": {"feats": {"motor_pos": ("clipdiv", MOTOR_LIMIT_RAD),
                             "motor_vel": ("clipdiv", 7.5),
                             "pen_vel":   ("clipdiv", 32.0),
                             "sin":       ("gain", 10.0)}},
}


def _apply_norm(spec, x: float) -> float:
    if spec is None:
        return x
    kind = spec[0]
    if kind == "div":
        return x / spec[1]
    if kind == "clipdiv":
        return float(np.clip(x / spec[1], -1.0, 1.0))
    if kind == "gain":
        return x * spec[1]
    if kind == "asinh":
        s, xm = spec[1], spec[2]
        return float(np.clip(np.arcsinh(x / s) / np.arcsinh(xm / s), -1.0, 1.0))
    raise ValueError(f"unknown norm kind {kind}")

# Arm geometry, measured 2026-05-02 against the OnShape CAD + a kitchen
# scale. The arm is 65 mm from the stepper shaft to the pendulum joint
# (where a single 608 bearing now carries the pendulum-link shaft —
# rebuild dropped the second bearing from the motor-shaft end). Total
# arm mass 30 g; COM is measured at 35 mm from the motor shaft, slightly
# past mid-arm because the remaining bearing sits at the pendulum end.
# ARM_LENGTH_M = 0.065
# ARM_MASS_KG = 0.030
# ARM_COM_M = 0.035
# ARM_I_COM_SPIN_KG_M2 = 8.06e-6  # Cross-validated against sysid free-swing period
GRAVITY = 9.81

# AS5600 encoder resolution.
ENCODER_CPR = 16384
PENDULUM_LSB_RAD = 2.0 * math.pi / ENCODER_CPR
MAX_PENDULUM_VEL_RAD_S = 32.0  # Typical max swing-up velocity

# Firmware measurement model (see LowLevelServer.ino). The firmware keeps a
# 500 Hz ring buffer of encoder samples; GET_STATE returns positions from
# the newest sample and velocities as (newest − oldest)/Δt over a 5-sample
# (8 ms) window. Motor position is the step counter; the pendulum is the
# 12-bit AS5600. Quantised finite differences over the window reproduce the
# rig's velocity spikes mechanistically: ±1 pendulum LSB across the window
# ≈ 0.19 rad/s, ±1 motor step ≈ 0.12 rad/s at 1/32 (0.25 at 1/16, 0.49 at 1/8)
# — the same mechanism behind the ~±0.4 rad/s spikes measured on a stationary
# pendulum. Enabled via `firmware_obs_model`; when on it feeds both the
# observation AND the velocity-mode P-law (the real host uses the same
# GET_STATE read for both).
#
# MOTOR_MICROSTEPS must match the `MICROSTEPS` constant in the firmware
# sketches — it sets how coarsely the sim quantises the motor-position
# measurement. 1/32 is the one ratio both drivers reach from identical
# MS-pin levels (DRV8825 M0=H/M1=L/M2=H, TMC2209 MS1=H/MS2=L), so a single
# wiring and a single policy cover both carriers.
MOTOR_MICROSTEPS = 16
MOTOR_STEP_LSB_RAD = 2.0 * math.pi / (200.0 * MOTOR_MICROSTEPS)
FIRMWARE_VEL_WINDOW_S = 0.008
# Age of the state snapshot by the time the host acts on it: encoder sample
# age (0–2 ms at 500 Hz) plus the GET_STATE serial round-trip. Randomised
# per episode under DR; fixed nominal otherwise (and in replay).
OBS_STALENESS_NOMINAL_S = 0.004
DR_OBS_STALENESS_RANGE_S = (0.002, 0.010)

# Pendulum mass / COM / I_com_swing come from `pendulum_geometry.py`,
# which parses the URDF (single source of truth shared with Julia +
# MeshCat). MuJoCo applies parallel-axis from `body_ipos` automatically,
# so effective pivot inertia is m·d² + I_com_swing per episode. The
# previous point-mass DR approximation (I_com ≈ 0) systematically
# understated pivot inertia by ~25%; the CAD value (~8.06e-6 kg·m²) is
# cross-validated against the sysid free-swing period to within 1%.

# Domain-randomization ranges. Activated by `domain_randomization=True`. Bounds
# bracket the measured sysid values with conservative margins so the trained
# policy generalises across plausible real-system variation.
DR_PENDULUM_MASS_FRAC = 0.10            # ±10% on nominal mass (narrowed from
                                        # ±20% once CAD/sysid agreement on m·d
                                        # confirmed mass uncertainty is small)
DR_PENDULUM_COM_FRAC = 0.10             # ±10% on nominal COM distance
DR_PENDULUM_FRICTION_MULT_RANGE = (0.5, 2.0)
# Phase 2.5: tau and delay recalibrated from policy-driven trajectory
# fits (analyze_run.py + sim_vs_real.py on real-hardware logs). Real
# motor responds crisply (tau ~ 0) but with a fixed transport delay
# of ~30 ms (3 control steps at 100 Hz). Phase 2's wider tau range
# and zero-min delay were biased away from reality.
# Action-mode constants — see RL_PLAN's accel-mode entry. Action ∈ [-1, 1]
# maps to commanded *angular acceleration*; the firmware calls
# FastAccelStepper's moveByAcceleration() with the matching int32 steps/s².
# Velocity is the integral of accel, capped at MAX_VELOCITY_RAD_S; position
# is the integral of velocity, fed to the existing PD position actuator.
# Position-delta mode: action × MAX_ACTION_DELTA_RAD is the per-step
# motor-target increment (radians), integrated and clamped to the safety
# limit. 0.10 matches RLControl.ino's MAX_ACTION_DELTA_RAD. Position mode only.
#
# NB: the max commanded slew is MAX_ACTION_DELTA_RAD × control_freq_hz
# (3.5 rad/s at 0.10 × 35 Hz). LowLevelServer's boot-time speed cap
# (MOTOR_MIN_STEP_US ≈ 5 rad/s) also applies to its position-mode moveTo
# tracking, and the sim models no cap — keep the slew comfortably below
# 5 rad/s or off-board deploys will saturate in a way sim never showed.
# (RLControl.ino's own envelope is ~196 rad/s; not the binding limit.)
MAX_ACTION_DELTA_RAD = 0.10

# The rig's canonical control rate, and the env's default. Anything that wants
# to *state* the deployment rate should read this rather than restate it —
# `sysid_wizard.py` hard-coded 35 Hz here and kept printing it long after the
# stack moved to 50.
CONTROL_FREQ_HZ = 50.0

MAX_VELOCITY_RAD_S = 5.0
MAX_ACCEL_RAD_S2 = 150.0   # bumped from 100 after the first accel-mode
                            # deployment showed the policy saturating its
                            # accel command at ±99 repeatedly — needed more
                            # authority. The firmware envelope is much
                            # higher post-FastAccelStepper (100 kSteps/s² ≈
                            # 393 rad/s²); 150 is intentionally conservative
                            # to leave headroom against step-skipping.

# DR on the motor's effective acceleration envelope per episode (mimics
# real-rig variability in stepper torque headroom under different load).
# Shifted up to match the new MAX_ACCEL=150 — policy needs episodes where
# the envelope brackets the new cap, otherwise it never trains with full
# authority.
DR_MOTOR_ACCEL_RANGE_RAD_S2 = (110.0, 190.0)
# First-order action-lag time constant (continuous analogue of the
# integer-step queue). Models the laptop ↔ Arduino ↔ stepper-ISR pipeline
# as a low-pass filter on the commanded action. Real-rig measurement
# (2026-05-16, run_policy log) showed the motor follows a half-and-half
# mix of current and previous command — corresponds to tau ≈ control
# period (28.6 ms at 35 Hz). Default range brackets that case with margin
# on each side. See website/src/content/docs/reference/transport-delay.md.
DR_ACTION_LAG_TAU_RANGE_S = (0.0, 0.030)
# Position-mode motor-bandwidth model: first-order lag on the commanded
# position TARGET (distinct from the action-lag above, which lags the
# command). Captures the stepper's ramp-toward-target each tick. Position
# mode only.
DR_MOTOR_TAU_RANGE_S = (0.010, 0.030)
DR_OBS_NOISE_STD_POS_RAD = 0.005
# Per-episode pendulum θ-bias: models the rig's static-friction-bounded
# rest position. At firmware boot the encoder zeros at whatever angle the
# pendulum settled at, which is within ±F_c/(m·g·l) ≈ ±1.9° of true
# vertical-down. Without DR over this, sim trains the policy to drive
# observed θ → 0, which on the rig means physical θ → -bias — gravity
# then exerts a constant restoring torque, and the policy can never
# reach motor_vel=0 (the stillness bonus is unreachable on rig). Sampled
# once per episode when DR is on. Applied to the observation only (the
# physics, the reward, and the eval env are unchanged). 0.05 rad ≈ 2.9°
# brackets the measured rest band with headroom.
DR_THETA_BIAS_MAX_RAD = 0.05
# Per-episode base tilt: the rig sits on a table that is not perfectly
# level. Unlike the constant theta-bias above (encoder zero offset), a
# tilted base shifts TRUE upright by an amount that varies with arm
# position — gravity acquires a component in the pendulum's swing plane
# proportional to the tilt projected onto that plane, which sweeps
# sinusoidally as the arm rotates. Modelled exactly by rotating the
# gravity vector per episode: tilt angle uniform in [0, max], direction
# uniform in [0, 2pi). 0.017 rad ~= 1 deg covers a normal table.
DR_BASE_TILT_MAX_RAD = 0.017
# Velocity-observation noise. The firmware computes velocity as
# (newest − oldest)/Δt over an ~8 ms window of 12-bit AS5600 samples, so
# ±1 LSB of quantisation/I²C jitter alone produces spikes up to
# ~0.4 rad/s on a stationary pendulum (measured — see the rest-detection
# analysis in real_env.py). The previous 0.05 understated that by ~4–8×,
# training policies against far cleaner velocities than the rig delivers.
# 0.15 puts the measured spike level at ~2.7σ.
DR_OBS_NOISE_STD_VEL_RAD_S = 0.02
# With `firmware_obs_model` on, the quantised finite-difference produces the
# spike floor mechanistically, so the additive Gaussian drops to a residual
# covering what the mechanism doesn't (I²C jitter, sample-time jitter).
# Keeping 0.15 on top would double-count the quantisation noise.
DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S = 0.05
DR_CONTROL_DT_JITTER_FRAC = 0.05        # ±5% jitter on physics steps per control.
                                         # Empirically valuable: the legacy
                                         # variable-rate fine-tune (rate
                                         # bug, ~5 ms dt jitter from
                                         # gradient-update timing) produced
                                         # the calm "minimal action"
                                         # attractor. Strict timing without
                                         # this jitter pushed SAC into the
                                         # noisier "active correction"
                                         # attractor. See
                                         # website/src/content/docs/reference/control-rate.md
                                         # "calm vs active attractors".

# Motor-joint static + Coulomb friction (stiction). Real steppers have a
# detent torque that creates a dead zone the position actuator doesn't
# capture. Bracket includes 0 to maintain backward compatibility with
# Phase 2 policies that were trained without stiction.
DR_MOTOR_FRICTIONLOSS_RANGE_N_M = (0.0, 0.005)


@dataclass(frozen=True)
class PendulumParams:
    pendulum_mass_kg: float
    pendulum_com_m: float       # perpendicular distance, joint axis -> COM
    pendulum_inertia_kg_m2: float  # about the joint axis (m·d² + I_com_swing)
    pendulum_friction: float    # viscous, N·m·s
    pendulum_coulomb: float     # Coulomb (dry) friction torque, N·m

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PendulumParams":
        """Construct from a per-rig sysid_params_*.json. Mass / COM / pivot inertia come
        from the URDF (via `pendulum_geometry`) — those are geometric
        constants of the pendulum body, not per-rig measurements. Only the
        friction terms (which depend on bearings, grease, temperature)
        come from sysid.
        """
        path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
        with open(path) as f:
            doc = json.load(f)
        pen = doc["pendulum"]["derived"]
        return cls(
            pendulum_mass_kg=PENDULUM_MASS_KG,
            pendulum_com_m=PENDULUM_COM_M,
            pendulum_inertia_kg_m2=(
                PENDULUM_MASS_KG * PENDULUM_COM_M ** 2
                + PENDULUM_I_COM_SWING_KG_M2
            ),
            pendulum_friction=float(pen["viscous_friction_N_m_s"]),
            pendulum_coulomb=float(pen.get("coulomb_friction_N_m", 0.0)),
        )


def build_mjcf(p: PendulumParams) -> str:
    """Construct an MJCF model string parameterised by sysid params.

    Pendulum inertia decomposition: MJCF expects the inertia tensor in the
    body's *inertial* frame, expressed about the COM. Our pendulum body's
    frame x-axis is aligned with the joint's rotation axis (axis="1 0 0").
    For rotation about the joint axis through the joint origin, the
    parallel-axis theorem gives
        I_about_joint_x = I_com_xx + m * (perpendicular distance from x-axis to COM)^2
                       = I_com_xx + m * (y_com^2 + z_com^2)
    With COM at (0, 0, -d), perpendicular distance is d.
    We take I_com_xx from CAD (PENDULUM_I_COM_SWING_KG_M2) rather than
    back-computing I_axis − m·d² from sysid: the sysid I_axis is consistent
    with the CAD value to within 5%, and using the CAD constant keeps
    `(m, d, I_com)` independent so DR can sample m and d without dragging
    I_com along with the point-mass approximation. For the off-axis
    components: a rod-like pendulum extending along z has appreciable extent
    perpendicular to y, so iyy ≈ ixx; izz (length-axis) is near zero.
    Slight asymmetry doesn't affect 1-DOF swing dynamics but keeps MuJoCo's
    solver well-conditioned.

    Motor model: the real stepper, when engaged, holds its commanded
    position essentially rigidly against the tiny reaction torque from the
    swinging pendulum. We model this with a stiff position actuator
    (kp=10, critically damped against arm inertia).
    """
    m = p.pendulum_mass_kg
    d = p.pendulum_com_m
    I_com_swing = PENDULUM_I_COM_SWING_KG_M2 * 1.066
    # diaginertia[0] = ixx = swing-axis inertia about COM (THE one that matters)
    # diaginertia[1] = iyy ≈ ixx for a rod-like body with extent perpendicular to its length axis
    # diaginertia[2] = izz ≈ tiny (length-axis inertia)
    diag_inertia = (I_com_swing, I_com_swing, 1e-7)

    # Arm inertia about its COM (along the stepper rotation axis). After
    # the 1-bearing rebuild, the mass distribution is closer to a thin
    # rod than the previous "point masses at both ends" model; the rod
    # approximation m·L²/12 gives ~1.06e-5 kg·m² for the current 30 g /
    # 65 mm arm. Used for the MJCF body's diaginertia.
    # arm_I = ARM_MASS_KG * ARM_LENGTH_M ** 2 / 12.0
    arm_I = ARM_I_COM_SPIN_KG_M2

    # PD position-actuator gains. The motor joint sees the FULL effective
    # inertia (arm parallel-axis + pendulum mass at arm tip + pendulum
    # self-inertia about its own joint), not just arm_I; kv is critical
    # damping against that inertia.
    #
    # kp models the stepper's stiffness, and a stepper is a near-RIGID
    # position source: its holding torque (~0.4 N·m) dwarfs the pendulum's
    # reaction torque (~0.02 N·m), so the physical arm follows the step
    # profile essentially exactly. The old kp=100 was compliant enough
    # that the pendulum's reaction rang the sim arm at ±rad/s — plant
    # texture the rig doesn't have, which policies then overfit to. The
    # historical "kp ≥ 200 destabilises the integrator" ceiling was NOT a
    # timestep limit: it was the control-rate STAIRCASE target hammering
    # the servo with one large impulse per tick. With the target
    # interpolated at substep resolution (see step()), kp=1000 is stable
    # under implicitfast at the 1 ms timestep and tracks the commanded
    # path to ~2 mrad / ~0.08 rad/s.
    I_arm_about_motor = arm_I + ARM_MASS_KG * ARM_COM_M ** 2
    I_pen_at_arm_tip  = p.pendulum_mass_kg * ARM_LENGTH_M ** 2
    I_pen_self        = p.pendulum_mass_kg * p.pendulum_com_m ** 2 + PENDULUM_I_COM_SWING_KG_M2
    I_motor_joint     = I_arm_about_motor + I_pen_at_arm_tip + I_pen_self
    kp = 200.0
    kv = 2.0 * math.sqrt(kp * I_motor_joint)  # critical damping

    return f"""<?xml version="1.0"?>
<mujoco model="rotary_inverted_pendulum">
  <option timestep="0.001" gravity="0 0 -{GRAVITY}" integrator="implicitfast"/>

  <default>
    <joint armature="0" damping="0"/>
    <geom contype="0" conaffinity="0" rgba="0.6 0.6 0.6 1"/>
  </default>

  <worldbody>
    <light diffuse="0.7 0.7 0.7" pos="0 0 1" dir="0 0 -1"/>

    <body name="arm" pos="0 0 0">
      <joint name="motor_joint" type="hinge" axis="0 0 1"
             range="-{MOTOR_LIMIT_RAD} {MOTOR_LIMIT_RAD}" limited="true"
             damping="0.0001" frictionloss="0"/>
      <inertial pos="{ARM_COM_M} 0 0" mass="{ARM_MASS_KG}"
                diaginertia="1e-6 {arm_I} {arm_I}"/>
      <geom name="arm_visual" type="capsule"
            fromto="0 0 0 {ARM_LENGTH_M} 0 0" size="0.004"
            rgba="0.4 0.6 0.8 1"/>

      <body name="pendulum" pos="{ARM_LENGTH_M} 0 0">
        <joint name="pendulum_joint" type="hinge" axis="1 0 0"
               damping="{p.pendulum_friction}"
               frictionloss="{p.pendulum_coulomb}"/>
        <inertial pos="0 0 -{d}" mass="{m}"
                  diaginertia="{diag_inertia[0]} {diag_inertia[1]} {diag_inertia[2]}"/>
        <geom name="pendulum_visual" type="capsule"
              fromto="0 0 0 0 0 -{d * 2}" size="0.003"
              rgba="0.8 0.4 0.4 1"/>
        <geom name="pendulum_tip" type="sphere" pos="0 0 -{d * 2}"
              size="0.006" rgba="0.9 0.7 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="motor" joint="motor_joint"
              kp="{kp}" kv="{kv}"
              ctrlrange="-{MOTOR_SAFE_LIMIT_RAD} {MOTOR_SAFE_LIMIT_RAD}"/>
  </actuator>
</mujoco>
"""


class RotaryInvertedPendulumEnv(gym.Env):
    """Off-board MuJoCo simulation of the rotary inverted pendulum.

    Theta convention: theta = 0 means upright (pendulum points "up", along
    +z). The MuJoCo joint is initialised with theta = pi (pendulum hanging
    down) at reset, modulo a small noise.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        params_path: str | Path | None = None,
        control_freq_hz: float = CONTROL_FREQ_HZ,  # see website/src/content/docs/reference/control-rate.md
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,  # accel-mode: action × this = commanded angular accel
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,  # accel-mode: velocity saturation cap
        max_action_delta_rad: float = MAX_ACTION_DELTA_RAD,  # position-mode: action × this = per-step target delta
        episode_length_s: float = 8.0,
        # Fraction of episodes that start NEAR UPRIGHT instead of hanging.
        # With hanging-only resets the catch/balance skill only receives
        # gradient after a successful swing-up, which makes DR stage
        # transitions brittle (observed: stage-2/3 collapses where the
        # policy regresses to spinning/crashing). Mixing in near-upright
        # starts trains the catch directly (Quanser-style initial-state
        # randomisation; balance-first curricula in the Furuta RL
        # literature). 0.0 preserves the legacy hanging-only behaviour;
        # the eval env must pass 0.0 so best-model selection still scores
        # the full swing-up + balance task.
        upright_reset_frac: float = 0.0,
        # Number of past frames in the observation (1 = legacy single frame).
        # Each 6-dim frame already carries prev_action, so stacking K frames
        # gives the policy BOTH observation history (filter velocity noise,
        # infer the per-episode θ-bias from arm drift) and action history
        # (reason about the ~17 ms in-flight command) in one knob. Frames
        # concatenate oldest → newest; at reset the history is seeded with
        # K copies of the initial frame. Must match between training,
        # fine-tuning, and deployment (recorded in config.json).
        obs_history_len: int = 4,
        # Include velocities in each observation frame (legacy True). With
        # False the frame is [motor_pos, sin θ, cos θ, prev_action] and the
        # policy must derive velocities from the stacked position history —
        # theoretically cleanest: no hand-picked finite-difference window or
        # filter lag in the loop, and the sim/real observation gap shrinks
        # to the position channels (which sim already quantises under DR).
        # Requires obs_history_len >= 2 (a single positional frame is not
        # Markovian); recommended 4. Recorded in config.json (must-match).
        obs_include_velocities: bool = True,
        # Model the firmware's measurement pipeline instead of reading MuJoCo
        # state directly: positions quantised to encoder/step resolution,
        # velocities as finite differences over the firmware's 8 ms window,
        # the whole snapshot stale by `obs_staleness_s` — applied to the
        # observation AND the velocity-mode P-law feedback (the real host
        # uses one GET_STATE read for both). Sim-only realism knob: does not
        # change obs shape, so checkpoints remain loadable either way, but
        # policies should be trained and sim-evaluated with the same setting.
        firmware_obs_model: bool = False,
        obs_staleness_s: float | None = None,  # None → OBS_STALENESS_NOMINAL_S
        dr_obs_staleness_range_s: tuple[float, float] | None = None,
        # Weights tuned for the standard quadratic-cost reward (see _reward).
        # At worst-case the per-step cost reaches ~22 (most of which is the
        # θ² term, max ~9.87 at hanging-down).
        reward_motor_pos_weight: float = 0.5,    # at motor safety limit (±125°): ~2.4
        reward_motor_vel_weight: float = 0.005,  # at motor_vel ±12 rad/s: ~0.7
        reward_action_weight: float = 0.20,      # at |action|=1: 0.20
        # Was 0.05, but that was too weak — the active-correction attractor
        # (motor swings ±0.5 even when balanced) had ~identical reward to
        # the calm minimal-action attractor, so SAC converged on whichever
        # the optimizer landed on first. Bumping to 0.20 makes "use big
        # actions" cost ~0.20/step, vs 0.0005/step in calm — meaningful
        # preference for calm now. See website/src/content/docs/reference/control-rate.md.
        reward_pen_vel_weight: float = 0.001,    # at pen_vel ±30 rad/s: ~0.9 (was 0.005 — too punishing for swing-up)
        # Stillness bonus near upright. ADDS a non-negative bonus to the
        # canonical quadratic cost — bonus is shaped as
        #     k_bonus · exp(-θ²/σ_θ²) · exp(-α̇²/σ_v²)
        # so the bonus is only large when BOTH theta and motor_vel are
        # near zero simultaneously. Penalises Kapitza-style resonance
        # stabilisation specifically: a Kapitza policy has α̇ ≈ several
        # rad/s during balance and therefore loses the bonus, while a
        # corrective-feedback policy keeps motor_vel small and earns it.
        # Quadratic swing-up gradient is preserved by the additive
        # nature — bonus is ~0 far from upright, so swing-up dynamics
        # are unchanged.
        # Default None → 0 (disabled, current canonical reward).
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,        # bonus active within ~17°
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,  # full bonus only at α̇ < ~1 rad/s
        # Alive terms (audit F1, 2026-07-21). Offset makes the per-step
        # reward non-negative so terminating (hard-stop crash) always
        # forfeits value; velocity-gated upright bonus makes spin-through
        # farming unprofitable. None → 0 (canonical all-negative reward).
        # See reward.py for the full rationale and recommended values
        # (offset 15.0, upright alive 5.0).
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        render_mode: str | None = None,
        # --- Phase 2: realism / domain randomisation ---
        domain_randomization: bool = False,
        motor_max_accel_rad_s2: float | None = None,  # None => use max_accel_rad_s2
        action_lag_tau_s: float = 0.0,
        # Firmware-side action smoothing: the actuator receives the moving
        # average of the last N policy outputs (N=1 disables). A boxcar of
        # length 4 has exact nulls at rate/2 and rate/4 — precisely where
        # the learned PWM dither lives — while passing ≤3 Hz control content
        # almost untouched (gain 0.96) at a fixed 1.5-tick delay. Must match
        # ACTION_SMOOTH_WINDOW in RLControl.ino; the raw action still feeds
        # the reward and the observation's prev_action channel.
        action_smooth_window: int = 4,
        motor_tau_s: float = 0.0,  # position-mode fixed motor-bandwidth lag (non-DR)
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # DR range overrides for curriculum learning. None => use module
        # constants. Pass tuples to override per-instance.
        dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
        dr_action_lag_tau_range_s: tuple[float, float] | None = None,
        dr_motor_tau_range_s: tuple[float, float] | None = None,  # position-mode only
        dr_control_dt_jitter_frac: float | None = None,
        dr_theta_bias_max_rad: float | None = None,  # None → DR_THETA_BIAS_MAX_RAD
        dr_base_tilt_max_rad: float | None = None, # None → DR_BASE_TILT_MAX_RAD
        obs_norm: str = "raw",      
    ):  
        super().__init__()
        if obs_norm not in OBS_NORM_PRESETS:
            raise ValueError(f"obs_norm must be one of {list(OBS_NORM_PRESETS)}")
        self.obs_norm_name = obs_norm
        self._obs_norm = OBS_NORM_PRESETS[obs_norm]
        self.params = PendulumParams.load(params_path)
        self.control_freq_hz = control_freq_hz
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.max_velocity_rad_s = max_velocity_rad_s
        self.max_action_delta_rad = max_action_delta_rad
        self.episode_length_s = episode_length_s
        if not 0.0 <= upright_reset_frac <= 1.0:
            raise ValueError(f"upright_reset_frac must be in [0, 1], got {upright_reset_frac}")
        self.upright_reset_frac = float(upright_reset_frac)
        if int(obs_history_len) < 1:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
        self.obs_history_len = int(obs_history_len)
        self.obs_include_velocities = bool(obs_include_velocities)
        if not self.obs_include_velocities and self.obs_history_len < 2:
            raise ValueError(
                "obs_include_velocities=False needs obs_history_len >= 2 — "
                "velocities must be inferable from the position history"
            )
        self._obs_history: deque = deque(maxlen=self.obs_history_len)
        self.firmware_obs_model = bool(firmware_obs_model)
        self._fixed_obs_staleness_s = (
            float(obs_staleness_s) if obs_staleness_s is not None
            else OBS_STALENESS_NOMINAL_S
        )
        self._dr_obs_staleness_range_s = (
            dr_obs_staleness_range_s if dr_obs_staleness_range_s is not None
            else DR_OBS_STALENESS_RANGE_S
        )
        self._obs_staleness_s = self._fixed_obs_staleness_s
        # Ring of (motor_qpos, pendulum_qpos) captured every physics substep,
        # continuous across control steps within an episode. Sized to cover
        # the velocity window + worst-case staleness with margin.
        self._meas_ring: deque = deque(maxlen=32)
        self._prev_cmd_pos = 0.0
        self._meas_motor_pos = 0.0
        self._meas_pen_phi = 0.0
        self._meas_motor_vel = 0.0
        self._meas_pen_vel = 0.0
        # All reward terms live in a single RewardWeights dataclass so the
        # sim env and the real env (real_env.py) share one source of truth.
        # Add new terms to reward.py, not here. None at call site → use
        # the canonical default (matches the DR-range None-fallback pattern).
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
            k_alive_offset=(
                float(reward_alive_offset)
                if reward_alive_offset is not None else 0.0
            ),
            k_upright_alive=(
                float(reward_upright_alive_weight)
                if reward_upright_alive_weight is not None else 0.0
            ),
        )
        self._prev_action = 0.0
        self.render_mode = render_mode

        # Phase 2 config
        self.domain_randomization = domain_randomization
        self._fixed_motor_max_accel_rad_s2 = (
            float(motor_max_accel_rad_s2) if motor_max_accel_rad_s2 is not None
            else float(max_accel_rad_s2)
        )
        self._fixed_action_lag_tau_s = float(action_lag_tau_s)
        if int(action_smooth_window) < 1:
            raise ValueError(
                f"action_smooth_window must be >= 1, got {action_smooth_window}"
            )
        self.action_smooth_window = int(action_smooth_window)
        self._kick_ticks_left = 0
        self._kick_value = 0.0
        self._action_smooth_buf: deque = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )
        self._fixed_motor_tau_s = float(motor_tau_s)
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        self._fixed_motor_frictionloss = 0.0  # set by user via reset(options=) if desired
        # DR range overrides for curriculum learning
        self._dr_motor_accel_range_rad_s2 = (
            dr_motor_accel_range_rad_s2 if dr_motor_accel_range_rad_s2 is not None
            else DR_MOTOR_ACCEL_RANGE_RAD_S2
        )
        self._dr_action_lag_tau_range_s = (
            dr_action_lag_tau_range_s if dr_action_lag_tau_range_s is not None
            else DR_ACTION_LAG_TAU_RANGE_S
        )
        self._dr_motor_tau_range_s = (
            dr_motor_tau_range_s if dr_motor_tau_range_s is not None
            else DR_MOTOR_TAU_RANGE_S
        )
        self._dr_control_dt_jitter_frac = (
            float(dr_control_dt_jitter_frac)
            if dr_control_dt_jitter_frac is not None
            else DR_CONTROL_DT_JITTER_FRAC
        )
        self._dr_theta_bias_max_rad = (
            float(dr_theta_bias_max_rad)
            if dr_theta_bias_max_rad is not None
            else DR_THETA_BIAS_MAX_RAD
        )
        self._theta_bias_rad = 0.0  # sampled per-episode in reset()
        self._dr_base_tilt_max_rad = (
            float(dr_base_tilt_max_rad)
            if dr_base_tilt_max_rad is not None
            else DR_BASE_TILT_MAX_RAD
        )

        xml = build_mjcf(self.params)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Number of physics steps per control step.
        physics_dt = self.model.opt.timestep
        n_substeps = int(round((1.0 / control_freq_hz) / physics_dt))
        self._n_substeps = max(1, n_substeps)
        self._dt_control = self._n_substeps * physics_dt

        # Size the measurement ring for the velocity window plus worst-case
        # staleness at the actual physics dt (replaces the placeholder above).
        self._meas_win_substeps = max(1, int(round(FIRMWARE_VEL_WINDOW_S / physics_dt)))
        stale_max_s = max(self._fixed_obs_staleness_s, self._dr_obs_staleness_range_s[1])
        stale_max_substeps = int(math.ceil(stale_max_s / physics_dt))
        self._meas_ring = deque(maxlen=self._meas_win_substeps + stale_max_substeps + 2)

        self._max_steps = int(episode_length_s * control_freq_hz)
        self._step_count = 0
        # Accel-mode state: action commands accel; we integrate to vel (capped)
        # and then to position-target (fed to existing PD position actuator).
        self._motor_vel = 0.0      # commanded angular velocity, rad/s
        self._motor_target = 0.0   # integrated position target, rad
        self._motor_max_accel_rad_s2 = float(max_accel_rad_s2)  # set per-episode if DR on
        # Position-delta mode: first-order lag on the commanded target that
        # models the stepper's ramp-to-target bandwidth. `_lagged_target` is
        # the filter state (what the PD actuator actually chases); `_motor_tau_s`
        # is the per-episode time constant (DR) or the fixed default.
        self._lagged_target = 0.0
        self._motor_tau_s = 0.0
        self._action_queue: deque = deque()
        # Continuous action lag: first-order LP filter on commanded action.
        # `_action_lag_tau_s` set per-episode (DR) or from the fixed default.
        # `_lagged_action` is the filter's internal state, reset each episode.
        self._action_lag_tau_s = 0.0
        self._lagged_action = 0.0
        # Velocity-mode transport lag state: the P-law runs host-side with
        # no delay, so lag/queue apply to the accel COMMAND it sends, not to
        # the policy's velocity setpoint.
        self._lagged_accel_cmd = 0.0
        self._accel_cmd_queue: deque = deque()

        # Cache joint addresses (faster than name lookup each step).
        self._motor_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._motor_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._motor_dof_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        ]
        self._pen_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]
        self._pen_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]
        self._pen_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )
        self._pen_dof_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        ]
        self._noise_std_pos = 0.0  # set per-episode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        frame_high = (
            np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
            if self.obs_include_velocities
            else np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        )
        obs_high = np.tile(frame_high, self.obs_history_len)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        # Observation bounds are loose; SB3 ignores them for SAC but they
        # document the expected scale. Last dim is prev_action ∈ [-1, 1]
        # — gives the policy an implicit read on its own command pipeline,
        # which restores Markov property under action delay (POMDP→MDP).
        #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        self._viewer = None

    # --- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # --- Phase 2: per-episode randomisation ---
        if self.domain_randomization:
            self._sample_dr_params()
        else:
            # Use fixed values supplied at __init__ (or zeros).
            self._motor_max_accel_rad_s2 = self._fixed_motor_max_accel_rad_s2
            self._action_lag_tau_s = self._fixed_action_lag_tau_s
            self._motor_tau_s = self._fixed_motor_tau_s
            self._noise_std_pos = 0.0
            self._obs_staleness_s = self._fixed_obs_staleness_s
            self.model.opt.gravity[:] = (0.0, 0.0, -GRAVITY)
            # Reset model params to nominal in case a previous episode set them.
            self.model.dof_frictionloss[self._motor_dof_addr] = self._fixed_motor_frictionloss

        # Theta-bias DR is INDEPENDENT of --domain-randomization. Modelling
        # the rig's encoder calibration offset (static-friction rest band)
        # is always desirable for sim-to-real: even stage 1 (no action-lag
        # DR) needs bias robustness if the policy is going to fine-tune on
        # rig and find true upright. Eval env explicitly passes
        # `dr_theta_bias_max_rad=0.0` for a bias-free reference scenario.
        if self._dr_theta_bias_max_rad > 0.0:
            self._theta_bias_rad = float(self.np_random.uniform(
                -self._dr_theta_bias_max_rad, self._dr_theta_bias_max_rad
            ))
        else:
            self._theta_bias_rad = 0.0
        self._lagged_action = 0.0
        self._lagged_accel_cmd = 0.0
        self._kick_ticks_left = 0
        self._kick_value = 0.0
        self._action_smooth_buf = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )

        # Pendulum hangs down -> joint position pi (since theta=0 is upright,
        # and the joint is wired so theta = joint_pos = 0 means pendulum-down).
        # With our MJCF the pendulum geom points along -z at joint=0, which IS
        # hanging down. So joint=0 == hanging down == theta = pi.
        # Pendulum starts near hanging-down with small noise. Motor starts
        # at any position within the safe range — Quanser-style training
        # diversity so the policy learns to recover from EVERY starting
        # config, not just near-zero. Without this, the policy never
        # practises returning from the limit and gets stuck there at deploy
        # time. Magnitude 0.7 × safe limit (≈ ±88°) keeps reset clear of
        # the immediate ±125° clamp while covering most of the working
        # range.
        # A fraction of episodes start near upright (joint phi = pi ± 0.3,
        # gentle spin) so the catch/balance skill gets direct gradient
        # signal instead of only appearing after a successful swing-up.
        if (self.upright_reset_frac > 0.0
                and self.np_random.random() < self.upright_reset_frac):
            phi0 = math.pi + self.np_random.uniform(-0.3, 0.3)
            pen_vel0 = self.np_random.uniform(-0.5, 0.5)
        else:
            phi0 = self.np_random.uniform(-0.05, 0.05)
            pen_vel0 = 0.0
        self.data.qpos[self._motor_qpos_addr] = self.np_random.uniform(
            -0.7 * MOTOR_SAFE_LIMIT_RAD, 0.7 * MOTOR_SAFE_LIMIT_RAD
        )
        self.data.qpos[self._pen_qpos_addr] = phi0
        self.data.qvel[self._motor_qvel_addr] = 0.0
        self.data.qvel[self._pen_qvel_addr] = pen_vel0
        self._motor_target = float(self.data.qpos[self._motor_qpos_addr])
        self._lagged_target = self._motor_target  # start the lag at the true start pos
        self._prev_cmd_pos = self._motor_target   # substep interpolation anchor
        self._motor_vel = 0.0
        self._step_count = 0
        self._prev_action = 0.0
        self._obs_history.clear()  # _obs() reseeds it with the initial frame
        mujoco.mj_forward(self.model, self.data)
        if self.firmware_obs_model:
            self.seed_measurement_ring()
        return self._obs(), {}

    def seed_measurement_ring(self) -> None:
        """(Re)fill the measurement ring from the CURRENT joint state.

        Back-extrapolates positions with the current joint velocities so the
        first window finite-difference reads the true initial velocity
        (matters for near-upright resets, which start with spin — the real
        firmware's buffer holds genuine motion at engage). Also used by
        replay tooling after overwriting qpos/qvel directly.
        """
        motor0 = float(self.data.qpos[self._motor_qpos_addr])
        pen0 = float(self.data.qpos[self._pen_qpos_addr])
        motor_vel0 = float(self.data.qvel[self._motor_qvel_addr])
        pen_vel0 = float(self.data.qvel[self._pen_qvel_addr])
        self._prev_cmd_pos = motor0  # commanded path starts at the true position
        dt_p = self.model.opt.timestep
        n = self._meas_ring.maxlen
        self._meas_ring.clear()
        self._meas_ring.extend(
            (motor0 - motor_vel0 * (n - 1 - k) * dt_p,
             pen0 - pen_vel0 * (n - 1 - k) * dt_p)
            for k in range(n)
        )
        self._capture_measured_state()

    def _sample_dr_params(self) -> None:
        """Sample per-episode randomisation: physical params + lag/delay."""
        rng = self.np_random
        nominal = self.params

        # Pendulum mass / COM
        m = rng.uniform(
            nominal.pendulum_mass_kg * (1.0 - DR_PENDULUM_MASS_FRAC),
            nominal.pendulum_mass_kg * (1.0 + DR_PENDULUM_MASS_FRAC),
        )
        d = rng.uniform(
            nominal.pendulum_com_m * (1.0 - DR_PENDULUM_COM_FRAC),
            nominal.pendulum_com_m * (1.0 + DR_PENDULUM_COM_FRAC),
        )
        # Inertia about the swing axis through COM is a CAD-derived
        # geometric constant; MuJoCo applies parallel-axis from body_ipos
        # so the effective pivot inertia is m*d^2 + I_com_swing. Previous
        # versions hard-coded I_com ≈ 0 (point-mass approximation), which
        # systematically understated pivot inertia by ~25%.
        I_com_swing = PENDULUM_I_COM_SWING_KG_M2

        # Friction (multiplicative on nominal). Same multiplier for viscous
        # and Coulomb — both come from the same bearing and tend to vary
        # together with grease state, temperature, and ball seating.
        fric_mult = rng.uniform(*DR_PENDULUM_FRICTION_MULT_RANGE)
        friction = nominal.pendulum_friction * fric_mult
        coulomb = nominal.pendulum_coulomb * fric_mult

        # Apply to MuJoCo model. body_inertia is the diagonal inertia about
        # COM in the body frame; index 0 = ixx (swing axis through COM),
        # 1 = iyy ≈ ixx, 2 = izz (length-axis, near-zero).
        self.model.body_mass[self._pen_body_id] = m
        self.model.body_inertia[self._pen_body_id, 0] = I_com_swing
        self.model.body_inertia[self._pen_body_id, 1] = I_com_swing
        self.model.body_inertia[self._pen_body_id, 2] = 1e-7
        self.model.body_ipos[self._pen_body_id, 2] = -d
        self.model.dof_damping[self._pen_dof_addr] = friction
        self.model.dof_frictionloss[self._pen_dof_addr] = coulomb

        # Per-episode lag and delay (using instance-level overrides if set,
        # falling back to module constants otherwise — supports curriculum).
        self._motor_max_accel_rad_s2 = float(rng.uniform(*self._dr_motor_accel_range_rad_s2))
        self._action_lag_tau_s = float(rng.uniform(*self._dr_action_lag_tau_range_s))
        self._noise_std_pos = DR_OBS_NOISE_STD_POS_RAD
        # Base tilt: rotate gravity by a random small angle about a random
        # horizontal axis. The pendulum's swing-plane component of the
        # resulting horizontal gravity varies with arm position — the
        # position-dependent bias no constant-offset DR can express.
        tilt = float(rng.uniform(0.0, self._dr_base_tilt_max_rad))
        azim = float(rng.uniform(0.0, 2.0 * math.pi))
        self.model.opt.gravity[:] = (
            GRAVITY * math.sin(tilt) * math.cos(azim),
            GRAVITY * math.sin(tilt) * math.sin(azim),
            -GRAVITY * math.cos(tilt),
        )
        self._obs_staleness_s = (
            float(rng.uniform(*self._dr_obs_staleness_range_s))
            if self.firmware_obs_model else self._fixed_obs_staleness_s
        )

        # Per-episode stepper stiction. The lower bound includes 0 so that
        # episodes without any stiction can still appear during DR.
        self.model.dof_frictionloss[self._motor_dof_addr] = float(
            rng.uniform(*DR_MOTOR_FRICTIONLOSS_RANGE_N_M)
        )

    def kick(self, value: float, ticks: int = 2) -> None:
        """Drive the actuator at `value` for the next `ticks` control steps.

        The mirror of overriding a_cmd in run_policy.py, so the sim protocol
        and the rig protocol are the same experiment.
        """
        self._kick_value = float(np.clip(value, -1.0, 1.0))
        self._kick_ticks_left = max(0, int(ticks))

    def step(self, action):
        action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # --- Firmware action smoothing: boxcar over the last N outputs. ---
        # Runs FIRST because on the device it sits directly after
        # policy_forward, upstream of the velocity law and of everything the
        # transport lag/queue below model. `action` (raw) still feeds the
        # reward and prev_action observation channel; only the actuator
        # chain sees `smoothed_action`.
        if self.action_smooth_window > 1:
            self._action_smooth_buf.append(action)
            smoothed_action = sum(self._action_smooth_buf) / self.action_smooth_window
        else:
            smoothed_action = action

        # --- Disturbance injection (see disturbance.py). ---
        # Overrides the actuator command only. `action` stays the policy's own
        # output, so prev_action and the reward never see the kick: the policy
        # observes motion it did not cause, which is what makes this a
        # disturbance rather than an action it must own. Sits here, after the
        # boxcar, so the impulse reaches the motor at full amplitude instead of
        # being smeared over the window.
        if self._kick_ticks_left > 0:
            smoothed_action = self._kick_value
            self._kick_ticks_left -= 1

        # --- Continuous action lag: first-order LP filter on the action. ---
        # Models the laptop ↔ Arduino ↔ stepper-ISR pipeline as a low-pass.
        # The rational discretisation `alpha = dt / (tau + dt)` makes the
        # filter behave like a "fractional-step delay": tau=0 ⇒ alpha=1 ⇒
        # no lag; tau=dt ⇒ alpha=0.5 ⇒ half current + half previous (the
        # behaviour measured on the real rig — see website/src/content/docs/reference/transport-delay.md).
        # Replaces the legacy integer-step delay queue for continuous DR.
        if self._action_lag_tau_s > 0.0:
            dt_ctrl = 1.0 / self.control_freq_hz
            alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
            self._lagged_action = (
                (1.0 - alpha) * self._lagged_action + alpha * smoothed_action
            )
            lagged_action = self._lagged_action
        else:
            self._lagged_action = smoothed_action
            lagged_action = smoothed_action


        # --- Determine n_substeps for this tick (DR adds dt jitter). ---
        # Implements ~5% control-rate jitter as DR. Empirically protects
        # SAC from finding the "active correction" attractor that strict
        # timing alone allows. See website/src/content/docs/reference/control-rate.md.
        if self.domain_randomization and self._dr_control_dt_jitter_frac > 0.0:
            jitter = float(self.np_random.uniform(
                -self._dr_control_dt_jitter_frac, self._dr_control_dt_jitter_frac
            ))
            n_sub = max(1, int(round(self._n_substeps * (1.0 + jitter))))
        else:
            n_sub = self._n_substeps

        actual_dt_s = n_sub * self.model.opt.timestep

        
        accel_cmd = lagged_action * self.max_accel_rad_s2
        accel_cmd = float(np.clip(accel_cmd,
                                    -self._motor_max_accel_rad_s2,
                                    self._motor_max_accel_rad_s2))
        self._motor_vel = float(np.clip(
            self._motor_vel + accel_cmd * actual_dt_s,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        ))
        # Complementary drift correction, mirroring the host: pull the
        # commanded integrator gently toward the measured velocity so
        # rail clamps / skipped steps can't accumulate open-loop error.
        # λ = 0.1/tick lets ~10% of the measurement's quantisation
        # noise through — the residual dither the policy trains against.
       
        # Safety: zero velocity if we're at the safety rail and pushing outward.
        # Mirrors the firmware-side clamp on the real rig.
        if self._motor_target >= MOTOR_SAFE_LIMIT_RAD and self._motor_vel > 0.0:
            self._motor_vel = 0.0
        elif self._motor_target <= -MOTOR_SAFE_LIMIT_RAD and self._motor_vel < 0.0:
            self._motor_vel = 0.0
        self._motor_target = float(np.clip(
            self._motor_target + self._motor_vel * actual_dt_s,
            -MOTOR_SAFE_LIMIT_RAD,
            MOTOR_SAFE_LIMIT_RAD,
        ))
        cmd_end = self._motor_target

        # The stepper glides continuously through the control period, so the
        # PD target is interpolated at substep resolution. Feeding the
        # tick's final target as a per-tick staircase hammers the stiff
        # servo with one large impulse per control step — the actual source
        # of the historical "kp >= 200 is unstable" ceiling and of the
        # joint-velocity ringing the old compliant plant showed.
        cmd_start = self._prev_cmd_pos

        if self.firmware_obs_model:
            # Measurement sources mirror the rig's sensors. MOTOR: the
            # firmware reads its own step counter — the commanded/executed
            # trajectory — so the ring stores the commanded path (linear
            # from the previous target at the commanded velocity), NOT the
            # MuJoCo joint. The PD actuator is only our approximation of
            # the stepper's near-perfect execution; leaking its tracking
            # error into the measurement (and thence the P-law) creates a
            # sim-only inner-loop oscillation that destroys policies which
            # balance fine on the real rig. PENDULUM: the AS5600 measures
            # the physical joint, so its channel reads qpos.
            for k in range(n_sub):
                cmd_k = cmd_start + (cmd_end - cmd_start) * (k + 1) / n_sub
                self.data.ctrl[0] = cmd_k
                mujoco.mj_step(self.model, self.data)
                self._meas_ring.append((
                    cmd_k,
                    float(self.data.qpos[self._pen_qpos_addr]),
                ))
            # One measurement per control tick, exactly like the host's
            # GET_STATE: this snapshot feeds this step's observation and the
            # NEXT step's P-law feedback.
            self._capture_measured_state()
        else:
            for k in range(n_sub):
                self.data.ctrl[0] = cmd_start + (cmd_end - cmd_start) * (k + 1) / n_sub
                mujoco.mj_step(self.model, self.data)

        self._prev_cmd_pos = cmd_end
        self._step_count += 1

        motor_pos_now = float(self.data.qpos[self._motor_qpos_addr])
        terminated = False
        reward = self._reward(action)
        self._prev_action = float(action)
        if self.terminate_on_hard_stop and abs(motor_pos_now) >= MOTOR_LIMIT_RAD:
            terminated = True
            reward -= self.hard_stop_penalty

        truncated = self._step_count >= self._max_steps
        info = {
            "motor_pos": motor_pos_now,
            "phi": float(self.data.qpos[self._pen_qpos_addr]),
            "motor_target": self._motor_target,
            "motor_vel_cmd": self._motor_vel,
            "motor_max_accel_rad_s2": self._motor_max_accel_rad_s2,
            "action_lag_tau_s": self._action_lag_tau_s,
            "smoothed_action": smoothed_action,
            "motor_tau_s": self._motor_tau_s,
            "obs_staleness_s": self._obs_staleness_s if self.firmware_obs_model else 0.0,
            "motor_vel_meas": self._meas_motor_vel if self.firmware_obs_model else None,
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._viewer is None:
            from mujoco import viewer  # noqa: WPS433  (deferred import)
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # --- Internal helpers --------------------------------------------------

    def _theta_upright(self) -> float:
        """Pendulum angle measured from upright. theta=0 -> upright."""
        # MJCF: joint angle phi=0 is hanging-down. Upright is phi=pi.
        # theta = phi - pi  (so theta=0 at upright, theta=±pi at hanging down).
        phi = float(self.data.qpos[self._pen_qpos_addr])
        return _wrap_pi(phi - math.pi)

    def _capture_measured_state(self) -> None:
        """Model one GET_STATE read from the ring of physics-rate positions.

        Positions are quantised to the step-counter / AS5600 resolution and
        stale by `_obs_staleness_s`; velocities are finite differences of the
        quantised positions across the firmware's 8 ms window, so the
        LSB-scale spikes emerge from the same mechanism as on the rig.
        """
        dt_p = self.model.opt.timestep
        ring = self._meas_ring
        k_stale = int(round(self._obs_staleness_s / dt_p))
        idx_new = max(0, len(ring) - 1 - k_stale)
        idx_old = max(0, idx_new - self._meas_win_substeps)
        m_new, p_new = ring[idx_new]
        m_old, p_old = ring[idx_old]
        mq_new = round(m_new / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        mq_old = round(m_old / MOTOR_STEP_LSB_RAD) * MOTOR_STEP_LSB_RAD
        pq_new = round(p_new / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        pq_old = round(p_old / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
        span_s = max(1, idx_new - idx_old) * dt_p
        self._meas_motor_pos = mq_new
        self._meas_pen_phi = pq_new
        self._meas_motor_vel = (mq_new - mq_old) / span_s
        self._meas_pen_vel = (pq_new - pq_old) / span_s

    def _normalize_frame(self, vals: dict) -> np.ndarray:
        """Apply the active OBS_NORM_PRESETS entry to one observation frame."""
        feats = self._obs_norm.get("feats", {})
        g = self._obs_norm.get("gain", 1.0)
        return np.array(
            [_apply_norm(feats.get(k), v) * g for k, v in vals.items()],
            dtype=np.float32,
        )
    
    def _obs(self) -> np.ndarray:
        if self.firmware_obs_model:
            # Measured-state path: what the host actually reads back.
            motor_pos = self._meas_motor_pos
            phi = self._meas_pen_phi
            motor_vel = self._meas_motor_vel
            pen_vel = self._meas_pen_vel
        else:
            motor_pos = float(self.data.qpos[self._motor_qpos_addr])
            phi = float(self.data.qpos[self._pen_qpos_addr])
            motor_vel = float(self.data.qvel[self._motor_qvel_addr])
            pen_vel = float(self.data.qvel[self._pen_qvel_addr])

        if self.domain_randomization:
            if not self.firmware_obs_model:
                # Quantise pendulum angle to AS5600 LSB resolution (the
                # measured-state path already quantises at capture).
                phi = round(phi / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
            # Inject small position + velocity noise to mimic finite-diff jitter
            # and encoder noise on real hardware.
            rng = self.np_random
            motor_pos += rng.normal(0.0, self._noise_std_pos)
            phi += rng.normal(0.0, self._noise_std_pos)
            vel_noise_std = (
                DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S if self.firmware_obs_model
                else DR_OBS_NOISE_STD_VEL_RAD_S
            )
            motor_vel += rng.normal(0.0, vel_noise_std)
            pen_vel += rng.normal(0.0, vel_noise_std)

        # Apply per-episode theta-bias to the OBSERVATION only. Physics and
        # reward (which uses self._theta_upright() on raw qpos) are unbiased
        # — the policy must learn to find true upright through a biased
        # encoder reading. Bias is 0 in the non-DR / eval env.
        phi = phi + self._theta_bias_rad

        theta = _wrap_pi(phi - math.pi)

        # Build the raw feature dict in the same order as before:
        # [motor_pos, sin, cos, (motor_vel, pen_vel,) prev_act]
        vals = {
            "motor_pos": motor_pos,
            "sin": math.sin(theta),
            "cos": math.cos(theta),
        }
        if self.obs_include_velocities:
            vals["motor_vel"] = motor_vel
            vals["pen_vel"] = pen_vel
        vals["prev_act"] = float(np.clip(self._prev_action, -1.0, 1.0))

        frame = self._normalize_frame(vals)

        # Frame stacking: called exactly once per reset/step, so appending
        # here keeps the history in lockstep with the control ticks.
        if not self._obs_history:
            self._obs_history.extend([frame] * self.obs_history_len)
        else:
            self._obs_history.append(frame)
        if self.obs_history_len == 1:
            return frame
        return np.concatenate(self._obs_history)

    def _reward(self, action: float) -> float:
        # Standard quadratic-cost reward (Quanser QUBE-Servo / Furuta-pendulum
        # literature standard). All terms are quadratic in deviation from the
        # goal state (upright, still, centered, gentle controls). Reward is
        # purely non-positive, max 0 when fully balanced.
        #
        # Reference forms in the wild:
        #     r = γ - (θ² + C₁·θ̇² + C₂·α² + C₃·α̇² + C₄·a²)
        # We use γ=0; SAC handles negative rewards fine and the all-negative
        # signal makes "less negative" gradient toward upright unambiguous.
        # Penalising θ̇² *always* (not gated by upper-half) is what
        # discourages the "swing through upright forever" failure mode that
        # the previous multiplicative-gate reward couldn't suppress: even
        # spinning in the lower half costs reward, so the policy learns to
        # bleed off energy after a missed catch instead of continuing to
        # pump.
        return compute_reward(
            theta=self._theta_upright(),
            pen_vel=float(self.data.qvel[self._pen_qvel_addr]),
            motor_pos=float(self.data.qpos[self._motor_qpos_addr]),
            motor_vel=float(self.data.qvel[self._motor_qvel_addr]),
            action=action,
            weights=self._reward_weights,
        )


def _wrap_pi(x: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi

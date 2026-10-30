"""Gymnasium environment for the rotary inverted pendulum, parameterised
from the system-identification fits in a per-rig sysid_params_*.json.

Geometry (Furuta pendulum):
    - Arm rotates about a vertical axis, driven by a stepper that is modelled
      as a stiff PD position actuator tracking an integrated command path.
    - Pendulum hinges at the end of the arm, free-swinging in the vertical
      plane perpendicular to the arm direction. Hangs straight down at rest.

Observation (one frame, stacked `obs_history_len` times, oldest -> newest):
    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel, prev_action]
    (velocities omitted when obs_include_velocities=False), where theta = 0 is
    upright (so cos(theta) = 1 at the goal).

Action (1-dim, in [-1, 1]):
    commanded angular acceleration in [-max_accel, +max_accel] rad/s².

Actuator chain (mirrors the device, see LowLevelServer.ino / RLControl.ino):
    policy action
      -> firmware boxcar (moving average of last N actions)
      -> residual transport delay (fractional ticks, DR'd, jittered)
      -> [optional kick disturbance]
      -> accel clip (DR'd motor envelope)
      -> firmware rail brake (±150 rad/s² beyond ±125°)
      -> integrate to velocity (capped) -> integrate to position target
      -> PD actuator chases the target (interpolated at substep resolution)

Reward: see Training/reward.py (single source of truth shared with real_env).
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

from Simulation.pendulum_geometry import (
    ARM_COM_M,
    ARM_I_COM_SPIN_KG_M2,
    ARM_LENGTH_M,
    ARM_MASS_KG,
    PENDULUM_COM_M,
    PENDULUM_I_COM_SWING_KG_M2,
    PENDULUM_MASS_KG,
)
from Training.reward import RewardWeights, compute_reward


class FractionalDelay:
    """Pure transport delay of a float number of control ticks
    (linear interpolation between buffered samples). Unity gain at all
    frequencies, no IIR tail."""

    def __init__(self, action_dim: int, max_delay_steps: int = 8):
        self.n = max_delay_steps + 2
        self.buf = np.zeros((self.n, action_dim))

    def reset(self, a0):
        self.buf[:] = a0  # prefill: no startup transient

    def step(self, a, delay_steps: float):
        self.buf = np.roll(self.buf, 1, axis=0)
        self.buf[0] = a
        d = float(np.clip(delay_steps, 0.0, self.n - 2))
        i = int(d)
        f = d - i
        return (1.0 - f) * self.buf[i] + f * self.buf[i + 1]


HERE = Path(__file__).resolve().parents[1]
# Fallback when no `params_path` is given. Named for the rig it describes
# rather than a generic `sysid_params.json`, so a multi-rig setup cannot
# silently train against whichever rig happened to be measured last. Pass
# `--params-path` for any other rig; see reference/python.md.
DEFAULT_PARAMS_PATH = HERE / "sysid_params.json"

# Hard-stop on the motor joint. Matches the lid-boss mechanical limit of ±135°.
# The firmware starts braking at ±125° (MOTOR_SAFE_LIMIT_RAD), so a policy
# that keeps pushing outward overshoots the safe limit a little but should
# stop short of the hard stop.
MOTOR_LIMIT_RAD = math.radians(135.0)
MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)
# Brake authority beyond the safe limit. Must match MOTOR_BRAKE_ACCEL_STEPS_S2
# in LowLevelServer.ino (150 rad/s²).
MOTOR_BRAKE_ACCEL_RAD_S2 = 150.0


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


# Arm geometry lives in `pendulum_geometry.py` (measured 2026-05-02 against
# the OnShape CAD + a kitchen scale; 65 mm arm, 30 g, COM at 35 mm).
GRAVITY = 9.81

# Pendulum-angle encoder: AS5047P, 14-bit (16384 counts/rev), read over SPI.
ENCODER_CPR = 16384
PENDULUM_LSB_RAD = 2.0 * math.pi / ENCODER_CPR
MAX_PENDULUM_VEL_RAD_S = 32.0  # Typical max swing-up velocity

# Firmware measurement model (see LowLevelServer.ino). The firmware keeps a
# 500 Hz ring buffer of encoder samples; GET_STATE returns positions from
# the newest sample and velocities as (newest − oldest)/Δt over a 5-sample
# (8 ms) window. Motor position is the step counter (3200 steps/rev); the
# pendulum is the 14-bit AS5047P. Quantised finite differences over the
# window reproduce the rig's velocity spikes mechanistically:
#   ±1 pendulum LSB across the window ≈ 2π/16384 / 8 ms ≈ 0.048 rad/s
#   ±1 motor step   across the window ≈ 2π/3200  / 8 ms ≈ 0.25  rad/s
# (the earlier 0.19 rad/s figure was for a 12-bit encoder). Enabled via
# `firmware_obs_model`; when on it feeds the observation.
#
# MOTOR_MICROSTEPS must match the firmware: 200 full steps × 16 = 3200
# steps/rev, the same constant used for the rad/s² -> steps/s² conversion
# in CMD_SET_ACCEL.
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
# so effective pivot inertia is m·d² + I_com_swing per episode.
#
# Empirical scale on the CAD swing inertia. Used by BOTH build_mjcf and the
# DR sampler so DR and non-DR/eval episodes share one plant. (It used to be
# applied only in build_mjcf, giving DR episodes 6.6% less inertia.) Set to
# 1.0 to use the raw CAD value.
PENDULUM_I_COM_SCALE = 1.066

# Domain-randomization ranges. Activated by `domain_randomization=True`. Bounds
# bracket the measured sysid values with conservative margins so the trained
# policy generalises across plausible real-system variation.
DR_PENDULUM_MASS_FRAC = 0.10            # ±10% on nominal mass
DR_PENDULUM_COM_FRAC = 0.10             # ±10% on nominal COM distance
DR_PENDULUM_FRICTION_MULT_RANGE = (0.5, 2.0)

# Accel-mode constants. Action ∈ [-1, 1] maps to commanded *angular
# acceleration*; the firmware calls FastAccelStepper's moveByAcceleration()
# with the matching int32 steps/s². Velocity is the integral of accel,
# capped at MAX_VELOCITY_RAD_S (firmware: MOTOR_MIN_STEP_US = 392 µs/step
# ≈ 5 rad/s at 3200 steps/rev); position is the integral of velocity, fed to
# the PD position actuator.
MAX_ACTION_DELTA_RAD = 0.10  # legacy position-mode constant, unused

# The rig's canonical control rate, and the env's default. Anything that wants
# to *state* the deployment rate should read this rather than restate it.
CONTROL_FREQ_HZ = 50.0

MAX_VELOCITY_RAD_S = 5.0
MAX_ACCEL_RAD_S2 = 150.0   # bumped from 100 after the first accel-mode
                            # deployment showed the policy saturating its
                            # accel command at ±99 repeatedly. The firmware
                            # envelope is higher (100 kSteps/s² ≈ 393 rad/s²);
                            # 150 leaves headroom against step-skipping.

# DR on the motor's effective acceleration envelope per episode (mimics
# real-rig variability in stepper torque headroom under different load).
# Brackets MAX_ACCEL=150 so the policy still sees episodes with full authority.
DR_MOTOR_ACCEL_RANGE_RAD_S2 = (110.0, 190.0)

# RESIDUAL transport delay, in seconds, applied AFTER the firmware boxcar.
# Total end-to-end delay on the rig = boxcar ((N-1)/2 ticks, 30 ms at N=4 and
# 50 Hz) + this residual (serial/USB, FastAccelStepper's 8 ms forward-planning
# queue, ramp-generator period). Real-rig measurements (2026-05-16,
# transport-delay.md) showed ~30 ms; if that was taken WITH the boxcar active
# the boxcar already accounts for it and the residual is only the ~8-20 ms
# left over. Re-measure with the boxcar on, subtract (N-1)/2 ticks, and set
# `transport_delay_s` and this range around the result.
DR_TRANSPORT_DELAY_RANGE_S = (0.0, 0.020)

DR_OBS_NOISE_STD_POS_RAD = 0.005
# Per-episode pendulum θ-bias: models the rig's static-friction-bounded
# rest position. At firmware boot the encoder zeros at whatever angle the
# pendulum settled at, which is within ±F_c/(m·g·l) ≈ ±1.9° of true
# vertical-down. Without DR over this, sim trains the policy to drive
# observed θ → 0, which on the rig means physical θ → -bias — gravity
# then exerts a constant restoring torque, and the policy can never
# reach motor_vel=0. Applied to the observation only (the physics, the
# reward, and the eval env are unchanged). 0.05 rad ≈ 2.9° brackets the
# measured rest band with headroom.
DR_THETA_BIAS_MAX_RAD = 0.05
# Per-episode base tilt: the rig sits on a table that is not perfectly
# level. A tilted base shifts TRUE upright by an amount that varies with arm
# position (gravity acquires a component in the pendulum's swing plane that
# sweeps sinusoidally as the arm rotates). Modelled exactly by rotating the
# gravity vector per episode: tilt angle uniform in [0, max], direction
# uniform in [0, 2pi). 0.017 rad ~= 1 deg covers a normal table.
DR_BASE_TILT_MAX_RAD = 0.017
# Additive velocity-observation noise used when `firmware_obs_model` is OFF.
# (The quantised finite-difference of the firmware model produces the spike
# floor mechanistically, so the residual below is used when it is on.)
DR_OBS_NOISE_STD_VEL_RAD_S = 0.02
# With `firmware_obs_model` on, the additive Gaussian drops to a residual
# covering what the mechanism doesn't (SPI jitter, sample-time jitter).
# Keeping the larger value on top would double-count quantisation noise.
DR_OBS_NOISE_STD_VEL_RESIDUAL_RAD_S = 0.05
DR_CONTROL_DT_JITTER_FRAC = 0.05        # ±5% jitter on physics steps per control.
                                         # Empirically valuable: the legacy
                                         # variable-rate fine-tune (~5 ms dt
                                         # jitter from gradient-update timing)
                                         # produced the calm "minimal action"
                                         # attractor; strict timing pushed SAC
                                         # into the noisier "active correction"
                                         # attractor. See
                                         # website/src/content/docs/reference/control-rate.md

# Motor-joint static + Coulomb friction (stiction). Real steppers have a
# detent torque that creates a dead zone the position actuator doesn't
# capture. Bracket includes 0 to maintain backward compatibility.
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
    We take I_com_xx from CAD (PENDULUM_I_COM_SWING_KG_M2 × PENDULUM_I_COM_SCALE)
    rather than back-computing I_axis − m·d² from sysid, which keeps
    `(m, d, I_com)` independent so DR can sample m and d without dragging
    I_com along. For the off-axis components: a rod-like pendulum extending
    along z has appreciable extent perpendicular to y, so iyy ≈ ixx; izz
    (length-axis) is near zero. Slight asymmetry doesn't affect 1-DOF swing
    dynamics but keeps MuJoCo's solver well-conditioned.

    Motor model: the real stepper, when engaged, holds its commanded
    position essentially rigidly against the tiny reaction torque from the
    swinging pendulum. We model this with a stiff position actuator
    (kp=200, critically damped against the full motor-joint inertia).
    """
    m = p.pendulum_mass_kg
    d = p.pendulum_com_m
    I_com_swing = PENDULUM_I_COM_SWING_KG_M2 * PENDULUM_I_COM_SCALE
    # diaginertia[0] = ixx = swing-axis inertia about COM (THE one that matters)
    # diaginertia[1] = iyy ≈ ixx for a rod-like body with extent perpendicular to its length axis
    # diaginertia[2] = izz ≈ tiny (length-axis inertia)
    diag_inertia = (I_com_swing, I_com_swing, 1e-7)

    # Arm inertia about its COM (along the stepper rotation axis), from
    # pendulum_geometry (cross-validated against the sysid free-swing period).
    arm_I = ARM_I_COM_SPIN_KG_M2

    # PD position-actuator gains. The motor joint sees the FULL effective
    # inertia (arm parallel-axis + pendulum mass at arm tip + pendulum
    # self-inertia about its own joint), not just arm_I; kv is critical
    # damping against that inertia.
    #
    # kp models the stepper's stiffness: a near-RIGID position source whose
    # holding torque (~0.4 N·m) dwarfs the pendulum's reaction torque
    # (~0.02 N·m). The historical "kp >= 200 destabilises the integrator"
    # ceiling was not a timestep limit but the control-rate STAIRCASE target
    # hammering the servo; with the target interpolated at substep resolution
    # (see step()), kp=200 is stable under implicitfast at the 1 ms timestep.
    I_arm_about_motor = arm_I + ARM_MASS_KG * ARM_COM_M ** 2
    I_pen_at_arm_tip  = p.pendulum_mass_kg * ARM_LENGTH_M ** 2
    I_pen_self        = p.pendulum_mass_kg * p.pendulum_com_m ** 2 + I_com_swing
    I_motor_joint     = I_arm_about_motor + I_pen_at_arm_tip + I_pen_self
    kp = 200.0
    kv = 2.0 * math.sqrt(kp * I_motor_joint)  # critical damping

    # ctrlrange spans the HARD limit (not the safe limit) because the
    # firmware's rail brake lets the commanded path overshoot ±125° slightly
    # before it turns around; see step().
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
              ctrlrange="-{MOTOR_LIMIT_RAD} {MOTOR_LIMIT_RAD}"/>
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
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,  # action × this = commanded angular accel
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,  # velocity saturation cap
        max_action_delta_rad: float = MAX_ACTION_DELTA_RAD,  # DEPRECATED: ignored (position mode removed)
        episode_length_s: float = 8.0,
        # Fraction of episodes that start NEAR UPRIGHT instead of hanging.
        # With hanging-only resets the catch/balance skill only receives
        # gradient after a successful swing-up, which makes DR stage
        # transitions brittle. Mixing in near-upright starts trains the catch
        # directly. 0.0 preserves the legacy hanging-only behaviour; the eval
        # env must pass 0.0 so best-model selection still scores the full
        # swing-up + balance task.
        upright_reset_frac: float = 0.0,
        # Number of past frames in the observation (1 = legacy single frame).
        # Each frame carries prev_action, so stacking K frames gives the policy
        # BOTH observation history (filter velocity noise, infer the
        # per-episode θ-bias from arm drift) and action history (reason about
        # the in-flight command). Frames concatenate oldest → newest; at reset
        # the history is seeded with K copies of the initial frame. Must match
        # between training, fine-tuning, and deployment (config.json).
        obs_history_len: int = 4,
        # Include velocities in each observation frame (legacy True). With
        # False the frame is [motor_pos, sin θ, cos θ, prev_action] and the
        # policy must derive velocities from the stacked position history.
        # Requires obs_history_len >= 2. Recorded in config.json (must-match).
        obs_include_velocities: bool = True,
        # Model the firmware's measurement pipeline instead of reading MuJoCo
        # state directly: positions quantised to encoder/step resolution,
        # velocities as finite differences over the firmware's 8 ms window,
        # the whole snapshot stale by `obs_staleness_s`. Sim-only realism knob:
        # does not change obs shape, so checkpoints remain loadable either
        # way, but policies should be trained and sim-evaluated with the same
        # setting.
        firmware_obs_model: bool = False,
        obs_staleness_s: float | None = None,  # None → OBS_STALENESS_NOMINAL_S
        dr_obs_staleness_range_s: tuple[float, float] | None = None,
        # Weights tuned for the standard quadratic-cost reward (see _reward).
        reward_motor_pos_weight: float = 0.5,    # at motor safety limit (±125°): ~2.4
        reward_motor_vel_weight: float = 0.005,  # at motor_vel ±12 rad/s: ~0.7
        # Was 0.05, but that was too weak — the active-correction attractor
        # had ~identical reward to the calm minimal-action attractor. 0.20
        # makes "use big actions" cost ~0.20/step vs ~0.0005/step in calm.
        # See website/src/content/docs/reference/control-rate.md.
        reward_action_weight: float = 0.20,      # at |action|=1: 0.20
        reward_pen_vel_weight: float = 0.001,    # at pen_vel ±30 rad/s: ~0.9
        # Stillness bonus near upright: k_bonus · exp(-θ²/σ_θ²) · exp(-α̇²/σ_v²),
        # large only when BOTH theta and motor_vel are near zero. Penalises
        # Kapitza-style resonance stabilisation. Default None → 0 (disabled).
        reward_stillness_bonus_weight: float | None = None,
        reward_stillness_sigma_theta_rad: float = 0.3,        # bonus active within ~17°
        reward_stillness_sigma_motor_vel_rad_s: float = 1.0,  # full bonus only at α̇ < ~1 rad/s
        # Alive terms (audit F1, 2026-07-21). Offset makes the per-step
        # reward non-negative so terminating always forfeits value;
        # velocity-gated upright bonus makes spin-through farming
        # unprofitable. None → 0. See reward.py (recommended: offset 15.0,
        # upright alive 5.0).
        reward_alive_offset: float | None = None,
        reward_upright_alive_weight: float | None = None,
        render_mode: str | None = None,
        # --- Phase 2: realism / domain randomisation ---
        domain_randomization: bool = False,
        motor_max_accel_rad_s2: float | None = None,  # None => use max_accel_rad_s2
        # RESIDUAL transport delay (seconds) after the firmware boxcar; see
        # DR_TRANSPORT_DELAY_RANGE_S. Total sim delay = boxcar + this.
        transport_delay_s: float = 0.0,
        # Firmware-side action smoothing: the actuator receives the moving
        # average of the last N policy outputs (N=1 disables). A boxcar of
        # length 4 has exact nulls at rate/2 and rate/4 — precisely where
        # the learned PWM dither lives — while passing ≤3 Hz control content
        # almost untouched (gain 0.96) at a fixed 1.5-tick delay. Must match
        # ACTION_SMOOTH_WINDOW in RLControl.ino; the raw action still feeds
        # the reward and the observation's prev_action channel.
        action_smooth_window: int = 4,
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # DR range overrides for curriculum learning. None => use module
        # constants. Pass tuples to override per-instance.
        dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
        dr_transport_delay_range_s: tuple[float, float] | None = None,
        dr_control_dt_jitter_frac: float | None = None,
        dr_theta_bias_max_rad: float | None = None,  # None → DR_THETA_BIAS_MAX_RAD
        dr_base_tilt_max_rad: float | None = None,   # None → DR_BASE_TILT_MAX_RAD
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

        # --- Firmware measurement model state ---
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
        # Ring of (motor_cmd_pos, pendulum_qpos) captured every physics
        # substep, continuous across control steps within an episode. Resized
        # below once the physics timestep is known.
        self._meas_ring: deque = deque(maxlen=32)
        self._prev_cmd_pos = 0.0
        self._meas_motor_pos = 0.0
        self._meas_pen_phi = 0.0
        self._meas_motor_vel = 0.0
        self._meas_pen_vel = 0.0

        # All reward terms live in a single RewardWeights dataclass so the
        # sim env and the real env (real_env.py) share one source of truth.
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

        # --- Phase 2 config ---
        self.domain_randomization = domain_randomization
        self._fixed_motor_max_accel_rad_s2 = (
            float(motor_max_accel_rad_s2) if motor_max_accel_rad_s2 is not None
            else float(max_accel_rad_s2)
        )

        # Actuator chain: boxcar -> residual transport delay -> (kick) -> clip.
        if int(action_smooth_window) < 1:
            raise ValueError(
                f"action_smooth_window must be >= 1, got {action_smooth_window}"
            )
        self.action_smooth_window = int(action_smooth_window)
        self._action_smooth_buf: deque = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )
        self._delay = FractionalDelay(1, max_delay_steps=8)
        self._fixed_transport_delay_s = float(transport_delay_s)
        self._dr_transport_delay_range_s = (
            dr_transport_delay_range_s if dr_transport_delay_range_s is not None
            else DR_TRANSPORT_DELAY_RANGE_S
        )
        self._delay_s = self._fixed_transport_delay_s
        self._delay_jitter_steps = 0.1   # per-tick jitter, in control ticks (DR only)
        self._kick_ticks_left = 0
        self._kick_value = 0.0

        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        self._fixed_motor_frictionloss = 0.0  # set by user via reset(options=) if desired

        # DR range overrides for curriculum learning
        self._dr_motor_accel_range_rad_s2 = (
            dr_motor_accel_range_rad_s2 if dr_motor_accel_range_rad_s2 is not None
            else DR_MOTOR_ACCEL_RANGE_RAD_S2
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
        # staleness at the actual physics dt.
        self._meas_win_substeps = max(1, int(round(FIRMWARE_VEL_WINDOW_S / physics_dt)))
        stale_max_s = max(self._fixed_obs_staleness_s, self._dr_obs_staleness_range_s[1])
        stale_max_substeps = int(math.ceil(stale_max_s / physics_dt))
        self._meas_ring = deque(maxlen=self._meas_win_substeps + stale_max_substeps + 2)

        self._max_steps = int(episode_length_s * control_freq_hz)
        self._step_count = 0
        # Accel-mode state: action commands accel; we integrate to vel (capped)
        # and then to position-target (fed to the PD position actuator).
        self._motor_vel = 0.0      # commanded angular velocity, rad/s
        self._motor_target = 0.0   # integrated position target, rad
        self._motor_max_accel_rad_s2 = float(max_accel_rad_s2)  # set per-episode if DR on

        # Cache joint addresses (faster than name lookup each step).
        motor_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "motor_joint")
        pen_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_joint")
        self._motor_qpos_addr = self.model.jnt_qposadr[motor_jid]
        self._motor_qvel_addr = self.model.jnt_dofadr[motor_jid]
        self._motor_dof_addr = self.model.jnt_dofadr[motor_jid]
        self._pen_qpos_addr = self.model.jnt_qposadr[pen_jid]
        self._pen_qvel_addr = self.model.jnt_dofadr[pen_jid]
        self._pen_dof_addr = self.model.jnt_dofadr[pen_jid]
        self._pen_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pendulum"
        )
        self._noise_std_pos = 0.0  # set per-episode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation bounds are loose; SB3 ignores them for SAC but they
        # document the expected scale. The last entry of each frame is
        # prev_action ∈ [-1, 1] — gives the policy an implicit read on its
        # own command pipeline, restoring the Markov property under delay.
        n_frame = 6 if self.obs_include_velocities else 4
        obs_high = np.full(n_frame * self.obs_history_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

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
            self._delay_s = self._fixed_transport_delay_s
            self._noise_std_pos = 0.0
            self._obs_staleness_s = self._fixed_obs_staleness_s
            self.model.opt.gravity[:] = (0.0, 0.0, -GRAVITY)
            # Reset model params to nominal in case a previous episode set them.
            self.model.dof_frictionloss[self._motor_dof_addr] = self._fixed_motor_frictionloss

        # Theta-bias DR is INDEPENDENT of --domain-randomization: the rig's
        # encoder calibration offset (static-friction rest band) is always
        # worth modelling for sim-to-real. Eval env passes
        # `dr_theta_bias_max_rad=0.0` for a bias-free reference scenario.
        if self._dr_theta_bias_max_rad > 0.0:
            self._theta_bias_rad = float(self.np_random.uniform(
                -self._dr_theta_bias_max_rad, self._dr_theta_bias_max_rad
            ))
        else:
            self._theta_bias_rad = 0.0

        # Actuator-chain state (boxcar prefilled so early outputs aren't
        # scaled down by missing samples).
        self._delay.reset(0.0)
        self._kick_ticks_left = 0
        self._kick_value = 0.0
        self._action_smooth_buf = deque(
            [0.0] * self.action_smooth_window, maxlen=self.action_smooth_window
        )

        # Joint phi=0 is hanging down; upright is phi=pi.
        # Pendulum starts near hanging-down with small noise. Motor starts
        # at any position within ±0.7 × safe limit (≈ ±88°) — Quanser-style
        # diversity so the policy learns to recover from every starting
        # config, not just near-zero.
        # A fraction of episodes start near upright (phi = pi ± 0.3, gentle
        # spin) so the catch/balance skill gets direct gradient signal.
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
        """Sample per-episode randomisation: physical params + delay."""
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
        # Same swing inertia as build_mjcf (shared scale) so DR and non-DR
        # episodes differ only by what is actually randomised. MuJoCo applies
        # parallel-axis from body_ipos, so effective pivot inertia is
        # m*d^2 + I_com_swing.
        I_com_swing = PENDULUM_I_COM_SWING_KG_M2 * PENDULUM_I_COM_SCALE

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

        # Per-episode motor envelope and residual transport delay (instance
        # overrides if set, module constants otherwise — supports curriculum).
        self._motor_max_accel_rad_s2 = float(rng.uniform(*self._dr_motor_accel_range_rad_s2))
        self._delay_s = float(rng.uniform(*self._dr_transport_delay_range_s))
        self._noise_std_pos = DR_OBS_NOISE_STD_POS_RAD

        # Base tilt: rotate gravity by a random small angle about a random
        # horizontal axis. The swing-plane component of the resulting
        # horizontal gravity varies with arm position — the position-dependent
        # bias no constant-offset DR can express.
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

        Injected AFTER the boxcar and transport delay (full amplitude, no
        filter echo) but still subject to the accel clip. Raw `action`,
        prev_action and the reward never see it. The mirror of overriding
        a_cmd in run_policy.py, so the sim protocol and the rig protocol are
        the same experiment.
        """
        self._kick_value = float(np.clip(value, -1.0, 1.0))
        self._kick_ticks_left = max(0, int(ticks))

    def step(self, action):
        action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # --- Tick duration first: delay is a time-domain quantity. ---
        # Stochastic rounding so ±jitter fires even when n_substeps is small.
        if self.domain_randomization and self._dr_control_dt_jitter_frac > 0.0:
            jitter = float(self.np_random.uniform(
                -self._dr_control_dt_jitter_frac, self._dr_control_dt_jitter_frac))
            x = self._n_substeps * (1.0 + jitter)
            lo = int(np.floor(x))
            n_sub = max(1, lo + int(self.np_random.random() < (x - lo)))
        else:
            n_sub = self._n_substeps
        actual_dt_s = n_sub * self.model.opt.timestep

        # --- 1. Firmware boxcar (matches device; prefilled on reset). ---
        # `action` (raw) still feeds the reward and prev_action.
        self._action_smooth_buf.append(action)
        smoothed_action = sum(self._action_smooth_buf) / self.action_smooth_window

        # --- 2. Residual transport delay (fractional ticks, with jitter). ---
        d_steps = self._delay_s / actual_dt_s
        if self.domain_randomization:
            d_steps += float(self.np_random.uniform(-self._delay_jitter_steps,
                                                    self._delay_jitter_steps))
        lagged_action = float(self._delay.step(smoothed_action, max(0.0, d_steps))[0])

        # --- 3. Disturbance AFTER the delay: full amplitude, no filter echo. ---
        if self._kick_ticks_left > 0:
            lagged_action = self._kick_value
            self._kick_ticks_left -= 1

        # --- 4. Accel command -> envelope clip. ---
        accel_cmd = float(np.clip(
            lagged_action * self.max_accel_rad_s2,
            -self._motor_max_accel_rad_s2,
            self._motor_max_accel_rad_s2,
        ))

        # --- 5. Firmware rail brake: beyond ±125° (commanded position) the
        # firmware overrides the host's accel with ±150 rad/s² back toward the
        # centre. The motor decelerates over a short distance and can overshoot
        # the safe limit by ~v²/(2a) (≈ 5° from 5 rad/s) rather than stopping
        # dead, so the hard stop at ±135° is not hit for a cap-speed approach.
        if self._motor_target >= MOTOR_SAFE_LIMIT_RAD:
            accel_cmd = -MOTOR_BRAKE_ACCEL_RAD_S2
        elif self._motor_target <= -MOTOR_SAFE_LIMIT_RAD:
            accel_cmd = +MOTOR_BRAKE_ACCEL_RAD_S2

        # --- 6. Integrate accel -> capped velocity -> position target. ---
        self._motor_vel = float(np.clip(
            self._motor_vel + accel_cmd * actual_dt_s,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        ))
        self._motor_target = float(np.clip(
            self._motor_target + self._motor_vel * actual_dt_s,
            -MOTOR_LIMIT_RAD,
            MOTOR_LIMIT_RAD,
        ))
        cmd_end = self._motor_target

        # The stepper glides continuously through the control period, so the
        # PD target is interpolated at substep resolution. Feeding the
        # tick's final target as a per-tick staircase hammers the stiff servo
        # with one large impulse per control step.
        cmd_start = self._prev_cmd_pos

        if self.firmware_obs_model:
            # Measurement sources mirror the rig's sensors. MOTOR: the
            # firmware reads its own step counter — the commanded/executed
            # trajectory — so the ring stores the commanded path, NOT the
            # MuJoCo joint (leaking PD tracking error into the measurement
            # creates a sim-only inner-loop oscillation). PENDULUM: the
            # AS5047P measures the physical joint, so its channel reads qpos.
            for k in range(n_sub):
                cmd_k = cmd_start + (cmd_end - cmd_start) * (k + 1) / n_sub
                self.data.ctrl[0] = cmd_k
                mujoco.mj_step(self.model, self.data)
                self._meas_ring.append((
                    cmd_k,
                    float(self.data.qpos[self._pen_qpos_addr]),
                ))
            # One measurement per control tick, exactly like the host's
            # GET_STATE.
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
            "transport_delay_s": self._delay_s,
            "n_substeps": n_sub,
            "smoothed_action": smoothed_action,
            "lagged_action": lagged_action,
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

        Positions are quantised to the step-counter / AS5047P resolution and
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
                # Quantise pendulum angle to encoder LSB resolution (the
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
        # encoder reading. Bias is 0 in the eval env.
        phi = phi + self._theta_bias_rad

        theta = _wrap_pi(phi - math.pi)

        # Raw feature dict, in order:
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
        # literature). Penalising θ̇² *always* (not gated by upper-half)
        # discourages the "swing through upright forever" failure mode: even
        # spinning in the lower half costs reward, so the policy learns to
        # bleed off energy after a missed catch. See Training/reward.py.
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
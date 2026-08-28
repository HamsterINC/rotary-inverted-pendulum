"""Gymnasium environment for the rotary inverted pendulum, parameterised
from the system-identification fits in sysid_params.json.

Geometry (Furuta pendulum):
    - Arm rotates about a vertical axis, driven by a position-controlled
      "motor" (modelling the AccelStepper + driver as a stiff PD).
    - Pendulum hinges at the end of the arm, free-swinging in the vertical
      plane perpendicular to the arm direction. Hangs straight down at rest.

Observation (5-dim):
    [motor_pos, sin(theta), cos(theta), motor_vel, pendulum_vel]
    where theta = 0 is upright (so cos(theta) = 1 at the goal).

Action (1-dim, in [-1, 1]):
    Maps to a position-delta added to the motor's commanded target each
    control step.

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
    PENDULUM_COM_M,
    PENDULUM_I_COM_SWING_KG_M2,
    PENDULUM_MASS_KG,
)
from reward import RewardWeights, compute_reward


HERE = Path(__file__).resolve().parent
DEFAULT_PARAMS_PATH = HERE / "sysid_params.json"

# Hard-stop on the motor joint. Matches the lid-boss mechanical limit of ±135°,
# but we clamp the policy at ±125° so the policy never *commands* a stop hit.
MOTOR_LIMIT_RAD = math.radians(135.0)
MOTOR_SAFE_LIMIT_RAD = math.radians(125.0)

# Arm geometry, measured 2026-05-02 against the OnShape CAD + a kitchen
# scale. The arm is 65 mm from the stepper shaft to the pendulum joint
# (where a single 608 bearing now carries the pendulum-link shaft —
# rebuild dropped the second bearing from the motor-shaft end). Total
# arm mass 30 g; COM is measured at 35 mm from the motor shaft, slightly
# past mid-arm because the remaining bearing sits at the pendulum end.
ARM_LENGTH_M = 0.065
ARM_MASS_KG = 0.030
ARM_COM_M = 0.035

GRAVITY = 9.81

# AS5600 encoder resolution.

ENCODER_CPR = 2000
PENDULUM_LSB_RAD = 2.0 * math.pi / ENCODER_CPR
# Scaling constants (match these identically on your FPGA pipeline)
MAX_PENDULUM_VEL_RAD_S = 30.0  # Typical max swing-up velocity
# PENDULUM_LSB_RAD = 2.0 * math.pi / 4096.0

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
# Action transport delay (queue between policy decision and command landing
# on the motor). With accel-mode the stepper itself handles the smooth
# velocity ramp; this captures the laptop ↔ Arduino ↔ stepper-ISR pipeline
# delay. Position-mode era this was ~50 ms (1–3 steps at 35 Hz); post-
# accel-mode it's ~14 ms (~½ step) and better modelled by the action-lag
# range below. Kept at (0, 0) so it's a no-op unless a curriculum
# explicitly turns it on. See docs/transport_delay.md.
DR_ACTION_DELAY_STEPS_RANGE = (0, 0)
# First-order action-lag time constant (continuous analogue of the
# integer-step queue). Models the laptop ↔ Arduino ↔ stepper-ISR pipeline
# as a low-pass filter on the commanded action. Real-rig measurement
# (2026-05-16, run_policy log) showed the motor follows a half-and-half
# mix of current and previous command — corresponds to tau ≈ control
# period (28.6 ms at 35 Hz). Default range brackets that case with margin
# on each side. See docs/transport_delay.md.
DR_ACTION_LAG_TAU_RANGE_S = (0.0, 0.030)
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
DR_OBS_NOISE_STD_VEL_RAD_S = 0.05
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
                                         # docs/control_rate_selection.md
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
        """Construct from sysid_params.json. Mass / COM / pivot inertia come
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
    I_com_swing = PENDULUM_I_COM_SWING_KG_M2
    # diaginertia[0] = ixx = swing-axis inertia about COM (THE one that matters)
    # diaginertia[1] = iyy ≈ ixx for a rod-like body with extent perpendicular to its length axis
    # diaginertia[2] = izz ≈ tiny (length-axis inertia)
    diag_inertia = (I_com_swing, I_com_swing, 1e-7)

    # Arm inertia about its COM (along the stepper rotation axis). After
    # the 1-bearing rebuild, the mass distribution is closer to a thin
    # rod than the previous "point masses at both ends" model; the rod
    # approximation m·L²/12 gives ~1.06e-5 kg·m² for the current 30 g /
    # 65 mm arm. Used for the MJCF body's diaginertia.
    arm_I = ARM_MASS_KG * ARM_LENGTH_M ** 2 / 12.0

    # PD position-actuator gains. The motor joint sees the FULL effective
    # inertia (arm parallel-axis + pendulum mass at arm tip + pendulum
    # self-inertia about its own joint), not just arm_I. Originally we
    # set `kv = 2·√(kp · arm_I)` which used a value ~14× too small,
    # giving a severely-underdamped PD that under-tracked the integrated
    # accel-mode position target.
    #
    # Sysid_accel comparison (2026-05-16, pendulum held) showed sim
    # reaching only 73 % of real's peak motor velocity (3.49 vs 5.5 rad/s)
    # with kp=10. Sweeping kp + recomputing kv against the full joint
    # inertia gave kp=100 → sim peak v = 4.88 rad/s ≈ 89 % of real.
    # kp ≥ 200 destabilises the RK4 integrator at the env's 1 ms physics
    # timestep.
    I_arm_about_motor = arm_I + ARM_MASS_KG * ARM_COM_M ** 2
    I_pen_at_arm_tip  = p.pendulum_mass_kg * ARM_LENGTH_M ** 2
    I_pen_self        = p.pendulum_mass_kg * p.pendulum_com_m ** 2 + PENDULUM_I_COM_SWING_KG_M2
    I_motor_joint     = I_arm_about_motor + I_pen_at_arm_tip + I_pen_self
    kp = 100.0
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
        control_freq_hz: float = 35.0,  # canonical for this rig — see docs/control_rate_selection.md
        max_accel_rad_s2: float = MAX_ACCEL_RAD_S2,  # action × this = commanded angular accel
        max_velocity_rad_s: float = MAX_VELOCITY_RAD_S,  # velocity saturation cap
        episode_length_s: float = 8.0,
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
        # preference for calm now. See docs/control_rate_selection.md.
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
        reward_motor_jerk_weight: float | None = None,  # NEW (not in the Quanser
        # paper): penalty on (motor_vel_t - motor_vel_{t-1})², i.e. the physical
        # motor's change in angular velocity. Distinct from
        # `reward_action_rate_weight` which penalises change in *commanded*
        # action. Motor jerk targets the *observed* motor jitter, which is
        # what an observer of the rig actually sees. Default None → 0.0
        # (disabled), matching the canonical 5-term reward. Try 0.01 as a
        # gentle starting point when re-enabling.
        reward_action_rate_weight: float | None = None,  # disabled (= 0.0) after the accel-mode switch:
        # in position-mode `(a_t - a_{t-1})²` penalised target-position jitter
        # which aligned with smooth motor commands; in accel-mode the policy
        # MUST flip accel sign to balance, so this penalty fought the task
        # (training observed SAC's entropy collapsing into a low-reward basin
        # with actor_loss stuck at ~580 vs ~330 in position-mode). Setting
        # to 0 restores the canonical Quanser quadratic-cost form from the
        # reference paper. The accel-envelope DR + velocity cap already
        # bound motion smoothness physically.
        render_mode: str | None = None,
        # --- Phase 2: realism / domain randomisation ---
        domain_randomization: bool = False,
        motor_max_accel_rad_s2: float | None = None,  # None => use max_accel_rad_s2
        action_delay_steps: int = 0,
        action_lag_tau_s: float = 0.0,
        terminate_on_hard_stop: bool = True,
        hard_stop_penalty: float = 5.0,
        # DR range overrides for curriculum learning. None => use module
        # constants. Pass tuples to override per-instance.
        dr_motor_accel_range_rad_s2: tuple[float, float] | None = None,
        dr_action_delay_steps_range: tuple[int, int] | None = None,
        dr_action_lag_tau_range_s: tuple[float, float] | None = None,
        dr_control_dt_jitter_frac: float | None = None,
        dr_theta_bias_max_rad: float | None = None,  # None → DR_THETA_BIAS_MAX_RAD
    ):
        super().__init__()
        self.params = PendulumParams.load(params_path)
        self.control_freq_hz = control_freq_hz
        self.max_accel_rad_s2 = max_accel_rad_s2
        self.max_velocity_rad_s = max_velocity_rad_s
        self.episode_length_s = episode_length_s
        # All reward terms live in a single RewardWeights dataclass so the
        # sim env and the real env (real_env.py) share one source of truth.
        # Add new terms to reward.py, not here. None at call site → use
        # the canonical default (matches the DR-range None-fallback pattern).
        self._reward_weights = RewardWeights(
            k_pen_vel=reward_pen_vel_weight,
            k_motor_pos=reward_motor_pos_weight,
            k_motor_vel=reward_motor_vel_weight,
            k_action=reward_action_weight,
            k_action_rate=(
                float(reward_action_rate_weight)
                if reward_action_rate_weight is not None else 0.0
            ),
            k_motor_jerk=(
                float(reward_motor_jerk_weight)
                if reward_motor_jerk_weight is not None else 0.0
            ),
            k_stillness_bonus=(
                float(reward_stillness_bonus_weight)
                if reward_stillness_bonus_weight is not None else 0.0
            ),
            sigma_theta=float(reward_stillness_sigma_theta_rad),
            sigma_motor_vel=float(reward_stillness_sigma_motor_vel_rad_s),
        )
        self._prev_action = 0.0
        self._prev_motor_vel = 0.0  # tracked for the motor-jerk reward term
        self.render_mode = render_mode

        # Phase 2 config
        self.domain_randomization = domain_randomization
        self._fixed_motor_max_accel_rad_s2 = (
            float(motor_max_accel_rad_s2) if motor_max_accel_rad_s2 is not None
            else float(max_accel_rad_s2)
        )
        self._fixed_action_delay_steps = int(action_delay_steps)
        self._fixed_action_lag_tau_s = float(action_lag_tau_s)
        self.terminate_on_hard_stop = terminate_on_hard_stop
        self.hard_stop_penalty = float(hard_stop_penalty)
        self._fixed_motor_frictionloss = 0.0  # set by user via reset(options=) if desired
        # DR range overrides for curriculum learning
        self._dr_motor_accel_range_rad_s2 = (
            dr_motor_accel_range_rad_s2 if dr_motor_accel_range_rad_s2 is not None
            else DR_MOTOR_ACCEL_RANGE_RAD_S2
        )
        self._dr_action_delay_steps_range = (
            dr_action_delay_steps_range if dr_action_delay_steps_range is not None
            else DR_ACTION_DELAY_STEPS_RANGE
        )
        self._dr_action_lag_tau_range_s = (
            dr_action_lag_tau_range_s if dr_action_lag_tau_range_s is not None
            else DR_ACTION_LAG_TAU_RANGE_S
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

        xml = build_mjcf(self.params)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Path("model.xml").write_text(xml)
        # print("Saved model.xml")

        # Number of physics steps per control step.
        physics_dt = self.model.opt.timestep
        n_substeps = int(round((1.0 / control_freq_hz) / physics_dt))
        self._n_substeps = max(1, n_substeps)
        self._dt_control = self._n_substeps * physics_dt

        self._max_steps = int(episode_length_s * control_freq_hz)
        self._step_count = 0
        # Accel-mode state: action commands accel; we integrate to vel (capped)
        # and then to position-target (fed to existing PD position actuator).
        self._motor_vel = 0.0      # commanded angular velocity, rad/s
        self._motor_target = 0.0   # integrated position target, rad
        self._motor_max_accel_rad_s2 = float(max_accel_rad_s2)  # set per-episode if DR on
        self._action_delay_steps = 0
        self._action_queue: deque = deque()
        # Continuous action lag: first-order LP filter on commanded action.
        # `_action_lag_tau_s` set per-episode (DR) or from the fixed default.
        # `_lagged_action` is the filter's internal state, reset each episode.
        self._action_lag_tau_s = 0.0
        self._lagged_action = 0.0

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
        # Observation bounds are loose; SB3 ignores them for SAC but they
        # document the expected scale. Last dim is prev_action ∈ [-1, 1]
        # — gives the policy an implicit read on its own command pipeline,
        # which restores Markov property under action delay (POMDP→MDP).

        # Observation is now 5-dim: [motor_pos, theta_norm, motor_vel, pen_vel, prev_action]
        # obs_high = np.array(
        #     [MOTOR_LIMIT_RAD, 1.0, 1.0, 200.0, 200.0, 1.0], dtype=np.float32
        # )
        # obs_high = np.array(
        #             [MOTOR_LIMIT_RAD, 1.0, 200.0, 200.0, 1.0], dtype=np.float32
        #         )
        # self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

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
            self._action_delay_steps = self._fixed_action_delay_steps
            self._action_lag_tau_s = self._fixed_action_lag_tau_s
            self._noise_std_pos = 0.0
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
        self._action_queue = deque([0.0] * self._action_delay_steps,
                                   maxlen=max(1, self._action_delay_steps + 1))
        self._lagged_action = 0.0

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
        phi0 = self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[self._motor_qpos_addr] = self.np_random.uniform(
            -0.7 * MOTOR_SAFE_LIMIT_RAD, 0.7 * MOTOR_SAFE_LIMIT_RAD
        )
        self.data.qpos[self._pen_qpos_addr] = phi0
        self.data.qvel[self._motor_qvel_addr] = 0.0
        self.data.qvel[self._pen_qvel_addr] = 0.0
        self._motor_target = float(self.data.qpos[self._motor_qpos_addr])
        self._motor_vel = 0.0
        self._step_count = 0
        self._prev_action = 0.0
        self._prev_motor_vel = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

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
        self._action_delay_steps = int(rng.integers(
            self._dr_action_delay_steps_range[0],
            self._dr_action_delay_steps_range[1] + 1,
        ))
        self._action_lag_tau_s = float(rng.uniform(*self._dr_action_lag_tau_range_s))
        self._noise_std_pos = DR_OBS_NOISE_STD_POS_RAD

        # Per-episode stepper stiction. The lower bound includes 0 so that
        # episodes without any stiction can still appear during DR.
        self.model.dof_frictionloss[self._motor_dof_addr] = float(
            rng.uniform(*DR_MOTOR_FRICTIONLOSS_RANGE_N_M)
        )

    def step(self, action):
        action = float(np.clip(np.asarray(action).flatten()[0], -1.0, 1.0))

        # --- Continuous action lag: first-order LP filter on the action. ---
        # Models the laptop ↔ Arduino ↔ stepper-ISR pipeline as a low-pass.
        # The rational discretisation `alpha = dt / (tau + dt)` makes the
        # filter behave like a "fractional-step delay": tau=0 ⇒ alpha=1 ⇒
        # no lag; tau=dt ⇒ alpha=0.5 ⇒ half current + half previous (the
        # behaviour measured on the real rig — see docs/transport_delay.md).
        # Replaces the legacy integer-step delay queue for continuous DR.
        if self._action_lag_tau_s > 0.0:
            dt_ctrl = 1.0 / self.control_freq_hz
            alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
            self._lagged_action = (1.0 - alpha) * self._lagged_action + alpha * action
            lagged_action = self._lagged_action
        else:
            self._lagged_action = action
            lagged_action = action

        # --- Action delay queue (integer steps, off by default post-accel-mode). ---
        # Kept for back-compat / large-delay regimes; new training uses
        # action_lag_tau_s instead. When both are active they compose
        # (LP filter followed by integer queue).
        if self._action_delay_steps > 0:
            self._action_queue.append(lagged_action)
            delayed_action = float(self._action_queue.popleft())
        else:
            delayed_action = lagged_action

        # --- Determine n_substeps for this tick (DR adds dt jitter). ---
        # Implements ~5% control-rate jitter as DR. Empirically protects
        # SAC from finding the "active correction" attractor that strict
        # timing alone allows. See docs/control_rate_selection.md.
        if self.domain_randomization and self._dr_control_dt_jitter_frac > 0.0:
            jitter = float(self.np_random.uniform(
                -self._dr_control_dt_jitter_frac, self._dr_control_dt_jitter_frac
            ))
            n_sub = max(1, int(round(self._n_substeps * (1.0 + jitter))))
        else:
            n_sub = self._n_substeps
        actual_dt_s = n_sub * self.model.opt.timestep

        # --- Accel-mode integration: action → accel → velocity (capped) → pos target. ---
        # Mirrors FastAccelStepper's moveByAcceleration() behaviour. The
        # per-episode envelope clamp models the stepper's torque-limited
        # accel ceiling under varying load.
        accel_cmd = delayed_action * self.max_accel_rad_s2
        accel_cmd = float(np.clip(accel_cmd,
                                   -self._motor_max_accel_rad_s2,
                                   self._motor_max_accel_rad_s2))
        self._motor_vel = float(np.clip(
            self._motor_vel + accel_cmd * actual_dt_s,
            -self.max_velocity_rad_s,
            self.max_velocity_rad_s,
        ))
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
        self.data.ctrl[0] = self._motor_target

        for _ in range(n_sub):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        motor_pos_now = float(self.data.qpos[self._motor_qpos_addr])
        terminated = False
        reward = self._reward(action)
        self._prev_action = float(action)
        self._prev_motor_vel = float(self.data.qvel[self._motor_qvel_addr])
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
            "action_delay_steps": self._action_delay_steps,
            "action_lag_tau_s": self._action_lag_tau_s,
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

    def _obs(self) -> np.ndarray:
        motor_pos = float(self.data.qpos[self._motor_qpos_addr])
        phi = float(self.data.qpos[self._pen_qpos_addr])
        motor_vel = float(self.data.qvel[self._motor_qvel_addr])
        pen_vel = float(self.data.qvel[self._pen_qvel_addr])

        if self.domain_randomization:
            # Quantize to 2000 CPR ticks
            phi = round(phi / PENDULUM_LSB_RAD) * PENDULUM_LSB_RAD
            rng = self.np_random
            motor_pos += rng.normal(0.0, self._noise_std_pos)
            phi += rng.normal(0.0, self._noise_std_pos)
            motor_vel += rng.normal(0.0, DR_OBS_NOISE_STD_VEL_RAD_S)
            pen_vel += rng.normal(0.0, DR_OBS_NOISE_STD_VEL_RAD_S)

        phi = phi + self._theta_bias_rad
        theta = _wrap_pi(phi - math.pi)

        # --- Normalize all dims to [-1.0, 1.0] ---
        motor_pos_norm = np.clip(motor_pos / MOTOR_SAFE_LIMIT_RAD, -1.0, 1.0)
        theta_norm     = np.clip(theta / math.pi, -1.0, 1.0)
        motor_vel_norm = np.clip(motor_vel / self.max_velocity_rad_s, -1.0, 1.0)
        pen_vel_norm   = np.clip(pen_vel / MAX_PENDULUM_VEL_RAD_S, -1.0, 1.0)
        prev_act_norm  = np.clip(self._prev_action, -1.0, 1.0)

        return np.array(
            [motor_pos_norm, theta_norm, motor_vel_norm, pen_vel_norm, prev_act_norm],
            dtype=np.float32,
        )

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
            prev_action=self._prev_action,
            prev_motor_vel=self._prev_motor_vel,
            weights=self._reward_weights,
        )


def _wrap_pi(x: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi

# """Pendulum-link geometric constants, parsed from the URDF.

# Single source of truth for mass / COM / inertia of the pendulum body:
# - Onshape CAD is the authoring tool.
# - `urdf/model.urdf` is the exported, canonical robot description (also
#   consumed by Julia/MeshCat/RigidBodyDynamics for MPC + visualisation).
# - This module parses the URDF on import and exposes three constants:
#     PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2.

# These are *geometric* properties — set by the part shape, not by per-rig
# assembly. They do not vary across rebuilds, so they are not randomised.
# What does vary (friction, bearings, etc.) still goes through the sysid
# pipeline.

# Intentionally stdlib-only so the sysid workflow can import it without
# pulling in MuJoCo / gymnasium.
# """

# from __future__ import annotations

# import xml.etree.ElementTree as ET
# from pathlib import Path


# # Repo layout:
# #   <repo>/RotaryInvertedPendulum-python/src/rl/pendulum_geometry.py
# #   <repo>/urdf/model.urdf
# URDF_PATH = Path(__file__).resolve().parents[3] / "urdf" / "robot.urdf"


# def _load_pendulum_geometry(urdf_path: Path) -> tuple[float, float, float]:
#     """Parse pendulum-link mass, swing-axis COM distance, and I_com from URDF.

#     Returns (mass_kg, com_m, I_com_swing_kg_m2) where:
#     - mass_kg: total pendulum mass.
#     - com_m: perpendicular distance from the rotation axis (link x-axis,
#       matching the `arm_to_pendulum` joint axis) to the COM. Equals
#       sqrt(y² + z²) of the inertial origin — the x-component is *along*
#       the rotation axis and has no effect on swing dynamics (see URDF
#       note in `urdf/model.urdf`).
#     - I_com_swing_kg_m2: pendulum's moment of inertia about its own COM
#       along the swing axis (ixx of the inertia tensor at COM).
#     """
#     inertial = ET.parse(urdf_path).getroot().find(
#         "./link[@name='pendulum']/inertial"
#     )
#     if inertial is None:
#         raise RuntimeError(
#             f"Could not find <link name='pendulum'>/<inertial> in {urdf_path}"
#         )
#     mass = float(inertial.find("mass").get("value"))
#     xyz = [float(v) for v in inertial.find("origin").get("xyz").split()]
#     com_m = (xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
#     ixx = float(inertial.find("inertia").get("ixx"))
#     return mass, com_m, ixx


# PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2 = (
#     _load_pendulum_geometry(URDF_PATH)
# )


"""Pendulum and Arm geometric constants, parsed from the URDF.

Single source of truth for mass / COM / inertia of the pendulum and arm bodies:
- Onshape CAD is the authoring tool.
- `urdf/robot.urdf` (or `urdf/model.urdf`) is the exported canonical robot description
  (consumed by Python/MuJoCo, Julia/MeshCat/RigidBodyDynamics, and sysid scripts).
- This module parses the URDF on import and exposes constants for both bodies.

These are *geometric* properties — derived directly from CAD density, shape, and
fastener positions. They do not vary dynamically across episodes.

Intentionally stdlib-only so the sysid workflow can import it without
pulling in MuJoCo or gymnasium.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


# Repo layout:
#   <repo>/RotaryInvertedPendulum-python/src/rl/pendulum_geometry.py
#   <repo>/urdf/robot.urdf
URDF_PATH = Path(__file__).resolve().parents[3] / "urdf" / "robot.urdf"


@dataclass(frozen=True)
class ArmGeometry:
    length_m: float  # Pivot-to-pivot distance (motor axis -> pendulum joint)
    mass_kg: float  # Composite mass (arm + bearing + fasteners)
    com_m: float  # Perpendicular distance from motor axis to arm COM
    i_com_spin_kg_m2: float  # Moment of inertia about arm's COM along motor axis (izz)


@dataclass(frozen=True)
class PendulumGeometry:
    mass_kg: float  # Total pendulum link mass
    com_m: float  # Perpendicular distance from swing axis to pendulum COM
    i_com_swing_kg_m2: float  # Inertia about pendulum's COM along swing axis (ixx)


def _load_robot_geometry(
    urdf_path: Path,
) -> tuple[ArmGeometry, PendulumGeometry]:
    """Parse arm and pendulum link properties from the robot URDF."""
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

    root = ET.parse(urdf_path).getroot()

    # ---------------------------------------------------------------------
    # 1. Arm Link & Connecting Joint
    # ---------------------------------------------------------------------
    arm_inertial = root.find("./link[@name='arm']/inertial")
    if arm_inertial is None:
        raise RuntimeError(
            f"Could not find <link name='arm'>/<inertial> in {urdf_path}"
        )

    arm_mass = float(arm_inertial.find("mass").get("value"))
    arm_xyz = [
        float(v) for v in arm_inertial.find("origin").get("xyz").split()
    ]
    # In link 'arm', rotation about the motor axis is along the Z axis.
    # Distance from motor axis (0, 0) to COM is in the XY plane:
    arm_com_m = math.sqrt(arm_xyz[0] ** 2 + arm_xyz[1] ** 2)

    # Inertia about the motor's spin axis at the arm's COM (izz in arm link frame):
    arm_i_com_spin = float(arm_inertial.find("inertia").get("izz"))

    # Arm length: displacement from motor joint to pendulum joint.
    # Joint 'arm_to_pendulum' parent is 'arm', origin defines the offset.
    joint_arm_pen = root.find("./joint[@name='arm_to_pendulum']")
    if joint_arm_pen is None:
        raise RuntimeError(
            f"Could not find <joint name='arm_to_pendulum'> in {urdf_path}"
        )

    j_xyz = [
        float(v) for v in joint_arm_pen.find("origin").get("xyz").split()
    ]
    arm_length_m = math.sqrt(j_xyz[0] ** 2 + j_xyz[1] ** 2)

    arm_geo = ArmGeometry(
        length_m=arm_length_m,
        mass_kg=arm_mass,
        com_m=arm_com_m,
        i_com_spin_kg_m2=arm_i_com_spin,
    )

    # ---------------------------------------------------------------------
    # 2. Pendulum Link
    # ---------------------------------------------------------------------
    pen_inertial = root.find("./link[@name='pendulum']/inertial")
    if pen_inertial is None:
        raise RuntimeError(
            f"Could not find <link name='pendulum'>/<inertial> in {urdf_path}"
        )

    pen_mass = float(pen_inertial.find("mass").get("value"))
    pen_xyz = [
        float(v) for v in pen_inertial.find("origin").get("xyz").split()
    ]
    # Perpendicular distance from rotation axis to COM:
    # Joint swings about the hinge; in pendulum frame, offset perpendicular to swing axis:
    pen_com_m = math.sqrt(pen_xyz[1] ** 2 + pen_xyz[2] ** 2)

    # Inertia about swing axis at COM:
    pen_ixx = float(pen_inertial.find("inertia").get("ixx"))

    pen_geo = PendulumGeometry(
        mass_kg=pen_mass,
        com_m=pen_com_m,
        i_com_swing_kg_m2=pen_ixx,
    )

    return arm_geo, pen_geo


# Parse on import
_ARM_GEO, _PEN_GEO = _load_robot_geometry(URDF_PATH)

# --- Pendulum Constants (Backwards Compatible) ---
PENDULUM_MASS_KG = _PEN_GEO.mass_kg
PENDULUM_COM_M = _PEN_GEO.com_m
PENDULUM_I_COM_SWING_KG_M2 = _PEN_GEO.i_com_swing_kg_m2

# --- Arm Constants ---
ARM_LENGTH_M = _ARM_GEO.length_m
ARM_MASS_KG = _ARM_GEO.mass_kg
ARM_COM_M = _ARM_GEO.com_m
ARM_I_COM_SPIN_KG_M2 = _ARM_GEO.i_com_spin_kg_m2
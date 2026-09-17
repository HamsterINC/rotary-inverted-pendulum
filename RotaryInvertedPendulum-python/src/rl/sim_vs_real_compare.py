import time
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# =====================================================================
# Kinematic / Stepper Configuration (Matching Rig & Firmware)
# =====================================================================
MAX_ACCEL_RAD_S2 = 150.0
MOTOR_MAX_ACCEL_RAD_S2 = 150.0
MAX_VELOCITY_RAD_S = 5.0
MOTOR_SAFE_LIMIT_RAD = 1.2
SLOWDOWN_FACTOR = 4.0


def inject_ghost_pendulum(xml_path: str) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Could not find <worldbody> in XML model.")

    sim_base_body = None
    for body in worldbody.findall(".//body"):
        for joint in body.findall("joint"):
            if joint.get("name") == "motor_joint":
                sim_base_body = body
                break
        if sim_base_body is not None:
            break

    if sim_base_body is None:
        sim_base_body = worldbody.find("body")

    ghost_body = ET.fromstring(ET.tostring(sim_base_body))

    for elem in ghost_body.iter():
        if "name" in elem.attrib:
            elem.set("name", f"ghost_{elem.attrib['name']}")

        if elem.tag == "geom":
            elem.set("contype", "0")
            elem.set("conaffinity", "0")
            if "material" in elem.attrib:
                del elem.attrib["material"]
            elem.set("rgba", "0.2 0.9 0.4 0.35")
            elem.set("mass", "0")

        elif elem.tag in ("site", "camera", "light"):
            elem.set("name", f"ghost_{elem.attrib['name']}")

    worldbody.append(ghost_body)
    return ET.tostring(root, encoding="unicode")


# =====================================================================
# Main Execution
# =====================================================================

CSV_FILE = "telemetry_run_20260915_152631.csv"
OUTPUT_CSV = "telemetry_run_with_sim.csv"
df = pd.read_csv(CSV_FILE)
df["s2_angle"] = np.arctan2(df["s2_sin"], df["s2_cos"]) - 0.5 * np.pi

augmented_xml = inject_ghost_pendulum("model.xml")
model = mujoco.MjModel.from_xml_string(augmented_xml)
data = mujoco.MjData(model)

# 1. Retrieve Qpos (Position) and Dofadr (Velocity) Addresses
sim_cart_qpos_idx = model.joint("motor_joint").qposadr[0]
sim_pole_qpos_idx = model.joint("pendulum_joint").qposadr[0]
sim_cart_qvel_idx = model.joint("motor_joint").dofadr[0]
sim_pole_qvel_idx = model.joint("pendulum_joint").dofadr[0]

ghost_cart_qpos_idx = model.joint("ghost_motor_joint").qposadr[0]
ghost_pole_qpos_idx = model.joint("ghost_pendulum_joint").qposadr[0]

init_cart = float(df["s1_pos"].iloc[0])
init_pole = float(df["s2_angle"].iloc[0]) + np.pi

data.qpos[sim_cart_qpos_idx] = init_cart
data.qpos[sim_pole_qpos_idx] = init_pole
data.qpos[ghost_cart_qpos_idx] = init_cart
data.qpos[ghost_pole_qpos_idx] = init_pole

motor_vel = 0.0
motor_target = init_cart
data.ctrl[0] = motor_target
mujoco.mj_forward(model, data)

# 2. Buffers for Sim Telemetry
sim_pole_angles = []
sim_pole_vels = []
sim_cart_positions = []
sim_cart_vels = []

# 3. Visual Playback & Recording Loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Playing telemetry with Accel Integration & Ghost... Close to exit.")

    for i in range(len(df)):
        if not viewer.is_running():
            break

        step_start = time.time()
        row = df.iloc[i]

        if i > 0:
            actual_dt_s = (row["t_ms"] - df["t_ms"].iloc[i - 1]) / 1000.0
            if actual_dt_s <= 0:
                actual_dt_s = 0.025
        else:
            actual_dt_s = 0.025

        n_sub = max(1, int(round(actual_dt_s / model.opt.timestep)))

        # Stepper Kinematics Integration
        cmd_accel = float(row["inference"])
        accel_cmd = np.clip(
            cmd_accel * MAX_ACCEL_RAD_S2,
            -MOTOR_MAX_ACCEL_RAD_S2,
            MOTOR_MAX_ACCEL_RAD_S2,
        )

        motor_vel = float(
            np.clip(
                motor_vel + accel_cmd * actual_dt_s,
                -MAX_VELOCITY_RAD_S,
                MAX_VELOCITY_RAD_S,
            )
        )

        if motor_target >= MOTOR_SAFE_LIMIT_RAD and motor_vel > 0.0:
            motor_vel = 0.0
        elif motor_target <= -MOTOR_SAFE_LIMIT_RAD and motor_vel < 0.0:
            motor_vel = 0.0

        motor_target = float(
            np.clip(
                motor_target + motor_vel * actual_dt_s,
                -MOTOR_SAFE_LIMIT_RAD,
                MOTOR_SAFE_LIMIT_RAD,
            )
        )

        data.ctrl[0] = motor_target

        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        # Record Sim Telemetry (Angle & Angular Velocity)
        sim_pole_angles.append(float(data.qpos[sim_pole_qpos_idx]))
        sim_pole_vels.append(float(data.qvel[sim_pole_qvel_idx]))
        sim_cart_positions.append(float(data.qpos[sim_cart_qpos_idx]))
        sim_cart_vels.append(float(data.qvel[sim_cart_qvel_idx]))

        # Ghost Alignment & Kinematics
        data.qpos[ghost_cart_qpos_idx] = row["s1_pos"]
        data.qpos[ghost_pole_qpos_idx] = row["s2_angle"]

        mujoco.mj_kinematics(model, data)
        viewer.sync()

        target_frame_time = actual_dt_s * SLOWDOWN_FACTOR
        elapsed = time.time() - step_start
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)

# 4. Append to DataFrame and Export
logged_len = len(sim_pole_vels)
df_out = df.iloc[:logged_len].copy()

df_out["sim_pole_angle"] = sim_pole_angles
df_out["sim_pole_vel"] = sim_pole_vels
df_out["sim_cart_pos"] = sim_cart_positions
df_out["sim_cart_vel"] = sim_cart_vels

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Exported {logged_len} rows to {OUTPUT_CSV}")
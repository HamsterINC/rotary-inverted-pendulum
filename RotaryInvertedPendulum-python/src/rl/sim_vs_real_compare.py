from collections import deque
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from pendulum_env import RotaryInvertedPendulumEnv

# =====================================================================
# Configuration
# =====================================================================
MAX_ACCEL_RAD_S2 = 150.0
MOTOR_MAX_ACCEL_RAD_S2 = 150.0
MAX_VELOCITY_RAD_S = 10.0
MOTOR_SAFE_LIMIT_RAD = 3
SLOWDOWN_FACTOR = 10.0

FPGA = False
action_lag_tau_s = 0.05
action_delay_steps = 0

CSV_FILE = "logs/run_20260921_181307.csv"
OUTPUT_CSV = "logs/telemetry_run_with_sim.csv"
SB3_ZIP_PATH = "Saved_runs/1509.zip"


# =====================================================================
# Helper: Action Lag & Delay for the Replay Branch
# =====================================================================
class ActionDelayHandler:
    def __init__(
        self,
        control_freq_hz: float = 40.0,
        action_lag_tau_s: float = 0.0,
        action_delay_steps: int = 0,
        initial_action: float = 0.0,
    ):
        self.control_freq_hz = float(control_freq_hz)
        self._action_lag_tau_s = float(action_lag_tau_s)
        self._action_delay_steps = int(action_delay_steps)
        self.reset(initial_action)

    def reset(self, initial_action: float = 0.0):
        self._lagged_action = float(initial_action)
        if self._action_delay_steps > 0:
            self._action_queue = deque(
                [float(initial_action)] * self._action_delay_steps,
                maxlen=self._action_delay_steps + 1,
            )
        else:
            self._action_queue = deque()

    def process_action(self, cmd_accel: float) -> float:
        cmd_accel = float(cmd_accel)
        if self._action_lag_tau_s > 0.0:
            dt_ctrl = 1.0 / self.control_freq_hz
            alpha = dt_ctrl / (self._action_lag_tau_s + dt_ctrl)
            self._lagged_action = (1.0 - alpha) * self._lagged_action + alpha * cmd_accel
            lagged_action = self._lagged_action
        else:
            self._lagged_action = cmd_accel
            lagged_action = cmd_accel

        if self._action_delay_steps > 0:
            self._action_queue.append(lagged_action)
            return float(self._action_queue.popleft())
        return lagged_action


# =====================================================================
# Kinematic and Dynamic Ghost Cloner (with Position Actuators)
# =====================================================================
def inject_visual_ghosts(xml_path: str, ghost_defs: list[dict]) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

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

    actuator_elem = root.find("actuator")
    if actuator_elem is None:
        actuator_elem = ET.SubElement(root, "actuator")

    for g in ghost_defs:
        prefix = g["prefix"]
        rgba = g["rgba"]
        is_dynamic = g.get("dynamic_pole", False)
        ghost_body = ET.fromstring(ET.tostring(sim_base_body))

        for elem in ghost_body.iter():
            # 1. Prefix all named elements
            if "name" in elem.attrib:
                elem.set("name", f"{prefix}_{elem.attrib['name']}")

            # 2. Visual transparency and zero collisions (leaves CAD mass intact)
            if elem.tag == "geom":
                elem.set("contype", "0")
                elem.set("conaffinity", "0")
                if "material" in elem.attrib:
                    del elem.attrib["material"]
                elem.set("rgba", rgba)

            # 3. Inject identified low-damping only on the dynamic ghost's pendulum joint
            elif elem.tag == "joint":
                if is_dynamic and "pendulum_joint" in elem.attrib.get("name", ""):
                    elem.set("damping", "5.75086e-06")
                    elem.set("frictionloss", "7.74722e-08")

            elif elem.tag in ("site", "camera", "light"):
                elem.set("name", f"{prefix}_{elem.attrib['name']}")

        worldbody.append(ghost_body)

        # 4. Inject stiff PD servo for the dynamic ghost
        if is_dynamic:
            ET.SubElement(
                actuator_elem,
                "position",
                {
                    "name": f"{prefix}_actuator",
                    "joint": f"{prefix}_motor_joint",
                    "kp": "100",
                    "kv": "0.23",
                    "ctrlrange": "-10 10",
                },
            )

    return ET.tostring(root, encoding="unicode")


# =====================================================================
# Main Execution
# =====================================================================
df = pd.read_csv(CSV_FILE)

if FPGA:
    df["pendulum_angle_rad"] = np.arctan2(df["s2_sin"], df["s2_cos"]) - 0.5 * np.pi
    df["time_s"] = df["t_ms"] / 1000.0
    df = df.rename(columns={"s1_pos": "arm_pos_rad", "s1_vel": "arm_vel_rad_s", "inf_act": "control_action"})
else:
    df = df.rename(
        columns={
            "timestamp_s": "time_s",
            "arm_actual_vel_rad_s": "arm_vel_rad_s",
            "pendulum_pos_rad": "pendulum_angle_rad",
            "action_cmd": "control_action",
        }
    )

raw_angles_rad = df["pendulum_angle_rad"].to_numpy() * np.pi + np.pi
angle_rad_motor = df["arm_pos_rad"].to_numpy() 

df["pendulum_angle_unwrapped"] = np.unwrap(raw_angles_rad)
df["arm_pos_rad"] = angle_rad_motor
# 1. Canonical Gym Environment
env = RotaryInvertedPendulumEnv(
    render_mode=None,
    control_freq_hz=40.0,
    max_accel_rad_s2=MAX_ACCEL_RAD_S2,
    max_velocity_rad_s=MAX_VELOCITY_RAD_S,
    action_delay_steps=1,
    domain_randomization=False,
    dr_theta_bias_max_rad=0.0,
    action_lag_tau_s=0.03,
)

obs, _ = env.reset(seed=0)

init_cart = float(df["arm_pos_rad"].iloc[0])
init_pole = float(df["pendulum_angle_unwrapped"].iloc[0])

env.data.qpos[env._motor_qpos_addr] = init_cart
env.data.qpos[env._pen_qpos_addr] = init_pole
env.data.qvel[env._motor_qvel_addr] = 0.0
env.data.qvel[env._pen_qvel_addr] = 0.0
if hasattr(env, "_motor_target"):
    env._motor_target = init_cart
mujoco.mj_forward(env.model, env.data)
obs = env._obs()

# 2. Comparison Viewer Model
ghost_configs = [
    {"prefix": "ghost_hw", "rgba": "0.2 0.9 0.4 0.35", "dynamic_pole": False},   # Hardware Telemetry: Green
    {"prefix": "ghost_pol", "rgba": "0.2 0.4 0.95 0.5", "dynamic_pole": False},  # PPO Closed-Loop: Cyan
    {"prefix": "ghost_dyn", "rgba": "0.95 0.6 0.1 0.6", "dynamic_pole": True},   # HW Arm + Sim Pole: Orange
]
augmented_xml = inject_visual_ghosts("model.xml", ghost_configs)
model_comp = mujoco.MjModel.from_xml_string(augmented_xml)
data_comp = mujoco.MjData(model_comp)

# Cache Joint & Actuator IDs
sim_cart_qpos_idx = model_comp.joint("motor_joint").qposadr[0]
sim_pole_qpos_idx = model_comp.joint("pendulum_joint").qposadr[0]

hw_cart_qpos_idx = model_comp.joint("ghost_hw_motor_joint").qposadr[0]
hw_pole_qpos_idx = model_comp.joint("ghost_hw_pendulum_joint").qposadr[0]

pol_cart_qpos_idx = model_comp.joint("ghost_pol_motor_joint").qposadr[0]
pol_pole_qpos_idx = model_comp.joint("ghost_pol_pendulum_joint").qposadr[0]

dyn_cart_qpos_idx = model_comp.joint("ghost_dyn_motor_joint").qposadr[0]
dyn_pole_qpos_idx = model_comp.joint("ghost_dyn_pendulum_joint").qposadr[0]

sim_act_idx = model_comp.actuator("motor_joint").id if "motor_joint" in [model_comp.actuator(i).name for i in range(model_comp.nu)] else 0
dyn_act_idx = model_comp.actuator("ghost_dyn_actuator").id

# Initial Positions
for qpos_cart, qpos_pole in [
    (sim_cart_qpos_idx, sim_pole_qpos_idx),
    (hw_cart_qpos_idx, hw_pole_qpos_idx),
    (pol_cart_qpos_idx, pol_pole_qpos_idx),
    (dyn_cart_qpos_idx, dyn_pole_qpos_idx),
]:
    data_comp.qpos[qpos_cart] = init_cart
    data_comp.qpos[qpos_pole] = init_pole

data_comp.ctrl[sim_act_idx] = init_cart
data_comp.ctrl[dyn_act_idx] = init_cart
mujoco.mj_forward(model_comp, data_comp)

# 3. Load SB3 Policy
policy = PPO.load(SB3_ZIP_PATH, device="cpu")

sim_motor_vel = 0.0
sim_motor_target = init_cart
action_handler_sim = ActionDelayHandler(
    control_freq_hz=40.0,
    action_lag_tau_s=action_lag_tau_s,
    action_delay_steps=action_delay_steps,
    initial_action=float(df["control_action"].iloc[0]),
)

sim_pole_angles, sim_cart_positions = [], []
pol_pole_angles, pol_cart_positions = [], []
dyn_pole_angles = []
pol_action = []

with mujoco.viewer.launch_passive(model_comp, data_comp) as viewer:
    print("Viewer running: [Default: Replay Sim] | [Green: HW Replay] | [Cyan: PPO Env] | [Orange: HW Arm + Sim Pole]")

    for i in range(len(df)):
        if not viewer.is_running():
            break

        step_start = time.time()
        row = df.iloc[i]

        actual_dt_s = row["time_s"] - df["time_s"].iloc[i - 1] if i > 0 else 0.025
        if actual_dt_s <= 0:
            actual_dt_s = 0.025

        # -------------------------------------------------------------
        # 1. Closed-Loop Policy Step (via Canonical Gym Env)
        # -------------------------------------------------------------
        if i < 5:
            action = 0.0
        else :
            action = float(row["control_action"])
            #action, _= policy.predict(obs, deterministic=True)
        #action, _= policy.predict(obs, deterministic=True) 
        obs, reward, terminated, truncated, _ = env.step(action)

        # -------------------------------------------------------------
        # 2. Open-Loop Stepper Replay Step
        # -------------------------------------------------------------
        if i < 5:
            cmd_sim = 0.0
        else :
            cmd_sim = float(row["control_action"])
        filtered_sim_action = action_handler_sim.process_action(cmd_sim)
        accel_sim = np.clip(filtered_sim_action * MAX_ACCEL_RAD_S2, -MOTOR_MAX_ACCEL_RAD_S2, MOTOR_MAX_ACCEL_RAD_S2)
        sim_motor_vel = float(np.clip(sim_motor_vel + accel_sim * actual_dt_s, -MAX_VELOCITY_RAD_S, MAX_VELOCITY_RAD_S))

        if (sim_motor_target >= MOTOR_SAFE_LIMIT_RAD and sim_motor_vel > 0.0) or (
            sim_motor_target <= -MOTOR_SAFE_LIMIT_RAD and sim_motor_vel < 0.0
        ):
            sim_motor_vel = 0.0

        sim_motor_target = float(np.clip(sim_motor_target + sim_motor_vel * actual_dt_s, -MOTOR_SAFE_LIMIT_RAD, MOTOR_SAFE_LIMIT_RAD))
        data_comp.ctrl[sim_act_idx] = sim_motor_target

        # -------------------------------------------------------------
        # 3. Dynamic Physics via PD Actuator Setpoint Ramping
        # -------------------------------------------------------------
        target_hw_arm = float(row["arm_pos_rad"])
        start_hw_arm = float(data_comp.ctrl[dyn_act_idx])
        
        n_sub = max(1, int(round(actual_dt_s / model_comp.opt.timestep)))
        for sub_step in range(n_sub):
            s = (sub_step + 1) / n_sub
            # Smoothly ramp the actuator setpoint to eliminate impulsive velocity jumps
            data_comp.ctrl[dyn_act_idx] = start_hw_arm + s * (target_hw_arm - start_hw_arm)
            mujoco.mj_step(model_comp, data_comp)

        # -------------------------------------------------------------
        # 4. Synchronize Visual Poses
        # -------------------------------------------------------------
        # Hardware Telemetry Ghost (Pure Kinematics from CSV)
        data_comp.qpos[hw_cart_qpos_idx] = target_hw_arm
        data_comp.qpos[hw_pole_qpos_idx] = float(row["pendulum_angle_unwrapped"])

        # PPO Policy Ghost (Mirrored from Gym environment)
        data_comp.qpos[pol_cart_qpos_idx] = float(env.data.qpos[env._motor_qpos_addr])
        data_comp.qpos[pol_pole_qpos_idx] = float(env.data.qpos[env._pen_qpos_addr])

        # Dynamic Ghost: arm tracked via PD actuator; pole moved naturally by mj_step

        # Logging traces
        pol_action.append(float(action))
        sim_pole_angles.append(float(data_comp.qpos[sim_pole_qpos_idx]) )
        sim_cart_positions.append(float(data_comp.qpos[sim_cart_qpos_idx]))
        pol_pole_angles.append(float(env.data.qpos[env._pen_qpos_addr]))
        pol_cart_positions.append(float(env.data.qpos[env._motor_qpos_addr]))
        dyn_pole_angles.append(float(data_comp.qpos[dyn_pole_qpos_idx]))

        mujoco.mj_forward(model_comp, data_comp)
        viewer.sync()
        arm_err = abs(float(data_comp.qpos[dyn_cart_qpos_idx]) - target_hw_arm)
        if i % 20 == 0:
            print(f"Arm tracking error: {arm_err:.4f} rad ({np.degrees(arm_err):.2f} deg)")

        target_frame_time = actual_dt_s * SLOWDOWN_FACTOR
        elapsed = time.time() - step_start
        if elapsed < target_frame_time:
            time.sleep(target_frame_time - elapsed)

env.close()

logged_len = len(sim_pole_angles)
df_out = df.iloc[:logged_len].copy()
df_out["sim_cart_pos"] = sim_cart_positions
df_out["sim_pole_angle"] = sim_pole_angles
df_out["env_cart_pos"] = pol_cart_positions
df_out["env_pole_angle"] = pol_pole_angles
df_out["dyn_ghost_sim_pole_angle"] = dyn_pole_angles
df_out["env_action"] = pol_action
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Exported combined traces with smooth PD dynamic ghost to {OUTPUT_CSV}")
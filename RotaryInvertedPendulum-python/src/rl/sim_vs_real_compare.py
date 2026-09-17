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
MAX_VELOCITY_RAD_S = 5.0
MOTOR_SAFE_LIMIT_RAD = 1.2
SLOWDOWN_FACTOR = 4.0

FPGA = False
action_lag_tau_s = 0.0
action_delay_steps = 0

CSV_FILE = "logs/run_20260917_115604.csv"
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
# Kinematic Ghost Cloner (Visual Only)
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

    for g in ghost_defs:
        prefix = g["prefix"]
        rgba = g["rgba"]
        ghost_body = ET.fromstring(ET.tostring(sim_base_body))

        for elem in ghost_body.iter():
            if "name" in elem.attrib:
                elem.set("name", f"{prefix}_{elem.attrib['name']}")
            if elem.tag == "geom":
                elem.set("contype", "0")
                elem.set("conaffinity", "0")
                if "material" in elem.attrib:
                    del elem.attrib["material"]
                elem.set("rgba", rgba)
                elem.set("mass", "0")
            elif elem.tag in ("site", "camera", "light"):
                elem.set("name", f"{prefix}_{elem.attrib['name']}")

        worldbody.append(ghost_body)

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
    df = df.rename(columns={"timestamp_s": "time_s", "arm_actual_vel_rad_s": "arm_vel_rad_s", "pendulum_pos_rad": "pendulum_angle_rad", "action_cmd": "control_action"})

if df["pendulum_angle_rad"].abs().max() <= 1.05:
    raw_angles_rad = df["pendulum_angle_rad"].to_numpy() * np.pi + np.pi
else:
    raw_angles_rad = df["pendulum_angle_rad"].to_numpy()
df["pendulum_angle_unwrapped"] = np.unwrap(raw_angles_rad)

# 1. Instantiate the Canonical Gym Environment
env = RotaryInvertedPendulumEnv(
    render_mode=None,
    control_freq_hz=40.0,
    max_accel_rad_s2=MAX_ACCEL_RAD_S2,
    max_velocity_rad_s=MAX_VELOCITY_RAD_S,
    domain_randomization=False,
    dr_theta_bias_max_rad=0.0,
)

obs, _ = env.reset(seed=0)

# Synchronize env's initial state with the first CSV timestamp
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

# 2. Build the Comparison Viewer Model with 2 Kinematic Ghosts
ghost_configs = [
    {"prefix": "ghost_hw", "rgba": "0.2 0.9 0.4 0.35"},   # Hardware Replay: Green
    {"prefix": "ghost_pol", "rgba": "0.2 0.4 0.95 0.5"},   # PPO Env State: Cyan
]
augmented_xml = inject_visual_ghosts("model.xml", ghost_configs)
model_comp = mujoco.MjModel.from_xml_string(augmented_xml)
data_comp = mujoco.MjData(model_comp)

# Cache Joint Pointers for the Comparison Model
sim_cart_qpos_idx = model_comp.joint("motor_joint").qposadr[0]
sim_pole_qpos_idx = model_comp.joint("pendulum_joint").qposadr[0]
hw_cart_qpos_idx = model_comp.joint("ghost_hw_motor_joint").qposadr[0]
hw_pole_qpos_idx = model_comp.joint("ghost_hw_pendulum_joint").qposadr[0]
pol_cart_qpos_idx = model_comp.joint("ghost_pol_motor_joint").qposadr[0]
pol_pole_qpos_idx = model_comp.joint("ghost_pol_pendulum_joint").qposadr[0]
sim_act_idx = model_comp.actuator("motor_joint").id if "motor_joint" in [model_comp.actuator(i).name for i in range(model_comp.nu)] else 0

# Sync comparison model starting positions
data_comp.qpos[sim_cart_qpos_idx] = init_cart
data_comp.qpos[sim_pole_qpos_idx] = init_pole
data_comp.qpos[hw_cart_qpos_idx] = init_cart
data_comp.qpos[hw_pole_qpos_idx] = init_pole
data_comp.qpos[pol_cart_qpos_idx] = init_cart
data_comp.qpos[pol_pole_qpos_idx] = init_pole

data_comp.ctrl[sim_act_idx] = init_cart
mujoco.mj_forward(model_comp, data_comp)

# 3. Load SB3 Policy
policy = PPO.load(SB3_ZIP_PATH, device="cpu")

# Stepper Replay Setup
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
pol_action = []

with mujoco.viewer.launch_passive(model_comp, data_comp) as viewer:
    print("Viewer running: [Default: Replay Sim] | [Green: Telemetry HW] | [Cyan: PPO Env]")

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
        print(obs)
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        # -------------------------------------------------------------
        # 2. Open-Loop Stepper Replay Step
        # -------------------------------------------------------------
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

        n_sub = max(1, int(round(actual_dt_s / model_comp.opt.timestep)))
        for _ in range(n_sub):
            mujoco.mj_step(model_comp, data_comp)

        # -------------------------------------------------------------
        # 3. Synchronize All 3 Visual Bodies
        # -------------------------------------------------------------
        # Replay: Updated automatically by mj_step
        # HW Telemetry Ghost: Force from CSV
        data_comp.qpos[hw_cart_qpos_idx] = float(row["arm_pos_rad"]) * np.pi
        data_comp.qpos[hw_pole_qpos_idx] = float(row["pendulum_angle_unwrapped"])

        # PPO Policy Ghost: Mirror positions from env.data
        data_comp.qpos[pol_cart_qpos_idx] = float(env.data.qpos[env._motor_qpos_addr])
        data_comp.qpos[pol_pole_qpos_idx] = float(env.data.qpos[env._pen_qpos_addr])

        pol_action.append(float(action))
        sim_pole_angles.append(float(data_comp.qpos[sim_pole_qpos_idx]))
        sim_cart_positions.append(float(data_comp.qpos[sim_cart_qpos_idx]))
        pol_pole_angles.append(float(env.data.qpos[env._pen_qpos_addr]))
        pol_cart_positions.append(float(env.data.qpos[env._motor_qpos_addr]))

        mujoco.mj_kinematics(model_comp, data_comp)
        viewer.sync()

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
df_out["env_action"] = pol_action
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Exported combined traces using native environment to {OUTPUT_CSV}")
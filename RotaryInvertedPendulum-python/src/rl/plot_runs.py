import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv("logs/telemetry_run_with_sim.csv")

# Compute derivative columns upfront
df["sim_pole_vel_calc"] = np.gradient(df["sim_cart_pos"], df["time_s"])
df["sim_pole_acc_calc"] = np.gradient(df["sim_pole_vel_calc"], df["time_s"])
df["arm_acc_rad_s2"] = np.gradient(df["arm_vel_rad_s"], df["time_s"])
df["env_pole_vel_calc"] = np.gradient(df["env_cart_pos"], df["time_s"])
df["env_pole_acc_calc"] = np.gradient(df["env_pole_vel_calc"], df["time_s"])

# Create a figure with 1 row and 3 columns side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ==========================================
# 1. First Plot: Positions
# ==========================================
axes[0].plot(
    df["arm_pos_rad"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="b",
    label="Arm Pos",
)
axes[0].plot(
    df["sim_cart_pos"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="r",
    label="Sim Cart",
)
axes[0].plot(
    df["env_cart_pos"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="g",
    label="Env Cart",
)
axes[0].plot(
    df["control_action"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="orange",
    label="Control",
)

axes[0].set_title("Positions", fontsize=14)
axes[0].set_xlabel("Sample Index", fontsize=12)
axes[0].set_ylabel("Position Value", fontsize=12)
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()


# ==========================================
# 2. Second Plot: Velocities
# ==========================================
axes[1].plot(
    df["arm_vel_rad_s"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="b",
    label="Arm Vel",
)
axes[1].plot(
    df["sim_pole_vel_calc"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="r",
    label="Sim Pole Vel",
)
axes[1].plot(
    df["env_pole_vel_calc"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="g",
    label="Env Pole Vel",
)
axes[1].plot(
    df["control_action"] * 10,
    marker="o",
    linestyle="-",
    markersize=3,
    color="orange",
    label="Control x10",
)

axes[1].set_title("Velocities", fontsize=14)
axes[1].set_xlabel("Sample Index", fontsize=12)
axes[1].set_ylabel("Velocity Value", fontsize=12)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend()


# ==========================================
# 3. Third Plot: Accelerations
# ==========================================
axes[2].plot(
    df["arm_acc_rad_s2"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="b",
    label="Arm Acc",
)
axes[2].plot(
    df["sim_pole_acc_calc"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="r",
    label="Sim Pole Acc",
)
axes[2].plot(
    df["env_pole_acc_calc"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="g",
    label="Env Pole Vel",
)
axes[2].plot(
    df["control_action"] * 100,
    marker="o",
    linestyle="-",
    markersize=3,
    color="orange",
    label="Control x100",
)

axes[2].set_title("Accelerations", fontsize=14)
axes[2].set_xlabel("Sample Index", fontsize=12)
axes[2].set_ylabel("Acceleration Value", fontsize=12)
axes[2].grid(True, linestyle="--", alpha=0.6)
axes[2].legend()


# Display all subplots together
plt.tight_layout()
plt.show()
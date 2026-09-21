import matplotlib.pyplot as plt
import pandas as pd

# Load the data (assuming you saved the text above into a file named 'data.csv')
df = pd.read_csv("logs/telemetry_run_with_sim.csv")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    df["arm_pos_rad"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="b",
)
plt.plot(
    df["sim_cart_pos"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="r",
)

# Add titles and labels
plt.title("Arm Position vs. Simulated Pole Angle", fontsize=14)
plt.xlabel("Arm Position (rad) [arm_pos_rad]", fontsize=12)
plt.ylabel("Simulated Pole Angle (rad) [sim_pole_angle]", fontsize=12)

# Add grid for readability
plt.grid(True, linestyle="--", alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Load the data (assuming you saved the text above into a file named 'data.csv')
df = pd.read_csv("logs/telemetry_run_with_sim.csv")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(
    df["control_action"],
    marker="o",
    linestyle="-",
    markersize=3,
    color="b",
)


# Add titles and labels
plt.title("Arm Position vs. Simulated Pole Angle", fontsize=14)
plt.xlabel("Arm Position (rad) [arm_pos_rad]", fontsize=12)
plt.ylabel("Simulated Pole Angle (rad) [sim_pole_angle]", fontsize=12)

# Add grid for readability
plt.grid(True, linestyle="--", alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()
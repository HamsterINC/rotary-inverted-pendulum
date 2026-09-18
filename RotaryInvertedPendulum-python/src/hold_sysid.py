import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 1. Load Raw CSV Data
df = pd.read_csv("rl/clamped_pendulum_decay.csv")

# -------------------------------------------------------------
# Active Indices Detection & Pre-Release Trimming
# -------------------------------------------------------------
initial_angle = df["pendulum_angle_unwrapped"].iloc[0]

# Detect when the pendulum deviates by more than 0.05 rad (~3 degrees)
active_indices = np.where(np.abs(df["pendulum_angle_unwrapped"] - initial_angle) > 0.05)[0]

if len(active_indices) > 0:
    # Start 2 frames before detection to catch the exact start of movement
    start_idx = max(0, active_indices[0] - 2)
    df_trimmed = df.iloc[start_idx:].copy()
    
    # Zero-shift the time vector
    df_trimmed["time_s"] = df_trimmed["time_s"] - df_trimmed["time_s"].iloc[0]
    df_trimmed.reset_index(drop=True, inplace=True)
    print(f"Trimmed {start_idx} stationary samples. Recording begins at actual release.")
else:
    df_trimmed = df.copy()
    print("No stationary delay detected; using full dataset.")

t_exp = df_trimmed["time_s"].values
alpha_exp = df_trimmed["pendulum_angle_unwrapped"].values

if "pendulum_vel_rad_s" in df_trimmed.columns:
    dalpha_exp = df_trimmed["pendulum_vel_rad_s"].values
else:
    dalpha_exp = np.gradient(alpha_exp, t_exp)

y0 = [alpha_exp[0], dalpha_exp[0]]

# -------------------------------------------------------------
# 2. ODE Definition
# -------------------------------------------------------------
def pendulum_ode(t, y, omega_sq, beta_v, beta_c):
    alpha, dalpha = y
    # Restoring torque + linear viscous damping + smoothed Coulomb friction
    ddalpha = -omega_sq * np.sin(alpha) - beta_v * dalpha - beta_c * np.tanh(40.0 * dalpha)
    return [dalpha, ddalpha]

def simulate_rollout(params, t_eval):
    omega_sq, beta_v, beta_c = params
    sol = solve_ivp(
        fun=lambda t, y: pendulum_ode(t, y, omega_sq, beta_v, beta_c),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    return sol.y[0]

def objective(params):
    try:
        alpha_sim = simulate_rollout(params, t_exp)
        return np.mean((alpha_sim - alpha_exp) ** 2)
    except Exception:
        return 1e6

# -------------------------------------------------------------
# 3. Optimization with Corrected Bounds
# -------------------------------------------------------------
# omega_sq ~ 179 rad^2/s^2 (corresponds to ~2.13 Hz)
# beta_v   ~ 0.15 - 0.25 (keeps the wave alive past 7 seconds)
# beta_c   ~ 0.01 (small dry bearing friction)
p0 = [150.0, 0.15, 0.01]

bounds = [
    (120.0, 205.0),   # Unlocks the ~2.1 Hz oscillation frequency
    (0.001, 0.30),    # Caps damping so the simulation doesn't die early
    (0.000, 0.03)     # Coulomb threshold
]

print("Running optimization...")
res = minimize(objective, p0, bounds=bounds, method="L-BFGS-B")
omega_sq_fit, beta_v_fit, beta_c_fit = res.x

print("\n--- Identified System Parameters ---")
print(f"omega_sq (m*g*l / J_p) : {omega_sq_fit:.4f} rad^2/s^2")
print(f"Natural Frequency      : {np.sqrt(omega_sq_fit):.2f} rad/s ({np.sqrt(omega_sq_fit)/(2*np.pi):.2f} Hz)")
print(f"beta_v   (c_v / J_p)    : {beta_v_fit:.4f} 1/s")
print(f"beta_c   (f_c / J_p)    : {beta_c_fit:.4f} rad/s^2")

# -------------------------------------------------------------
# 4. Plot Comparison
# -------------------------------------------------------------
alpha_fitted = simulate_rollout(res.x, t_exp)

plt.figure(figsize=(11, 5))
plt.plot(t_exp, alpha_exp, label="Real Hardware (Active Movement)", color="black", alpha=0.8)
plt.plot(t_exp, alpha_fitted, "--", label="Calibrated Model Simulation", color="crimson", linewidth=1.5)
plt.xlabel("Time (s)")
plt.ylabel("Pendulum Angle (rad)")
plt.title("Clamped Pendulum Validation: Trimmed Active Motion")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
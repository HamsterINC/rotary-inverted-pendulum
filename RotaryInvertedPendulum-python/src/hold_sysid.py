import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 1. Load and trim idle delay
df = pd.read_csv("rl/clamped_pendulum_decay.csv")
angle_0 = df["pendulum_angle_unwrapped"].iloc[0]
active_indices = np.where(np.abs(df["pendulum_angle_unwrapped"] - angle_0) > 0.05)[0]

if len(active_indices) > 0:
    start_idx = max(0, active_indices[0] - 2)
    df = df.iloc[start_idx:].copy()
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    df.reset_index(drop=True, inplace=True)

t_exp = df["time_s"].values
alpha_exp = df["pendulum_angle_unwrapped"].values

if "pendulum_vel_rad_s" in df.columns:
    dalpha_exp = df["pendulum_vel_rad_s"].values
else:
    dalpha_exp = np.gradient(alpha_exp, t_exp)

y0 = [alpha_exp[0], dalpha_exp[0]]


# 2. ODE function
def pendulum_ode(t, y, omega_sq, beta_v, beta_c):
    alpha, dalpha = y
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
        rtol=1e-5,
        atol=1e-7,
    )
    return sol.y[0]


def objective(params):
    try:
        alpha_sim = simulate_rollout(params, t_exp)
        return np.mean((alpha_sim - alpha_exp) ** 2)
    except Exception:
        return 1e6


# 3. Optimize with bounded frequency
p0 = [116.0, 0.08, 0.04]
bounds = [(90.0, 150.0), (0.001, 1.0), (0.0, 0.5)]

res = minimize(objective, p0, bounds=bounds, method="L-BFGS-B")
omega_sq_fit, beta_v_fit, beta_c_fit = res.x

print("\n--- Corrected Identification Results ---")
print(f"omega_sq : {omega_sq_fit:.4f} rad^2/s^2 (Freq = {np.sqrt(omega_sq_fit)/(2*np.pi):.2f} Hz)")
print(f"beta_v   : {beta_v_fit:.4f} 1/s")
print(f"beta_c   : {beta_c_fit:.4f} rad/s^2")

# 4. Plot aligned fit
alpha_fitted = simulate_rollout(res.x, t_exp)
plt.figure(figsize=(10, 5))
plt.plot(t_exp, alpha_exp, label="Real Hardware (Trimmed)", color="black", alpha=0.7)
plt.plot(t_exp, alpha_fitted, "--", label="Corrected Model Simulation", color="crimson")
plt.xlabel("Time (s)")
plt.ylabel("Pendulum Angle (rad)")
plt.title("Clamped Pendulum System Identification (Aligned & Corrected)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 1. Load the Clamped Pendulum Dataset
# CSV should contain: time_s, pendulum_angle_unwrapped, pendulum_vel_rad_s
df = pd.read_csv("rl/clamped_pendulum_decay.csv")

t_exp = df["time_s"].values
alpha_exp = df["pendulum_angle_unwrapped"].values

# If velocity is not in the CSV, compute via central difference:
if "pendulum_vel_rad_s" in df.columns:
    dalpha_exp = df["pendulum_vel_rad_s"].values
else:
    dalpha_exp = np.gradient(alpha_exp, t_exp)

# Initial state observed at t=0
y0 = [alpha_exp[0], dalpha_exp[0]]

# 2. Define Parametric ODE System
def pendulum_ode(t, y, omega_sq, beta_v, beta_c):
    """
    dy[0]/dt = y[1]
    dy[1]/dt = - omega_sq * sin(y[0]) - beta_v * y[1] - beta_c * tanh(50 * y[1])
    """
    alpha, dalpha = y
    ddalpha = -omega_sq * np.sin(alpha) - beta_v * dalpha - beta_c * np.tanh(50.0 * dalpha)
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
    return sol.y[0], sol.y[1]

# 3. Objective Function (Residual Sum of Squares)
def objective(params):
    omega_sq, beta_v, beta_c = params
    try:
        alpha_sim, dalpha_sim = simulate_rollout(params, t_exp)
        # Weighted loss prioritizing angle tracking and velocity damping envelope
        loss = np.mean((alpha_sim - alpha_exp)**2) + 0.05 * np.mean((dalpha_sim - dalpha_exp)**2)
        return loss
    except Exception:
        return 1e6

# 4. Initial Estimates & Optimization
# For a ~0.2m to 0.3m rod hanging down: omega_n ~ sqrt(g / L) ~ sqrt(9.81 / 0.25) ~ 6.2 rad/s => omega_sq ~ 40
initial_guess = [38.0, 0.15, 0.05]
bounds = [
    (1.0, 200.0),   # omega_sq (rad^2/s^2)
    (0.0001, 5.0),  # beta_v (1/s)
    (0.0, 2.0)      # beta_c (rad/s^2)
]

print("Running optimization on clamped decay trajectory...")
res = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

omega_sq_fit, beta_v_fit, beta_c_fit = res.x
print(f"\n--- Identified Normalized Parameters ---")
print(f"omega_sq (m*g*l / J_p) : {omega_sq_fit:.4f} rad^2/s^2")
print(f"beta_v   (c_v / J_p)    : {beta_v_fit:.4f} 1/s")
print(f"beta_c   (f_c / J_p)    : {beta_c_fit:.4f} rad/s^2")

# 5. Extract Physical SI Units (Using Measured Mass and Length)
# Measure total pole mass on a scale (e.g., m_p = 0.12 kg)
# Measure distance from pivot to center of mass (e.g., l_com = 0.18 m)
m_p_measured = 0.12       # kg
l_com_measured = 0.18     # m
g = 9.81

# Compute physical inertia and friction constants
J_p_identified = (m_p_measured * g * l_com_measured) / omega_sq_fit
c_v_identified = beta_v_fit * J_p_identified
f_c_identified = beta_c_fit * J_p_identified

print(f"\n--- Converted Physical Parameters (SI) ---")
print(f"Pivot Inertia (J_p) : {J_p_identified:.6f} kg*m^2")
print(f"Viscous Damping (c_v): {c_v_identified:.6f} N*m*s/rad")
print(f"Coulomb Friction (f_c): {f_c_identified:.6f} N*m")

# 6. Plot Fit vs. Real Telemetry
alpha_fitted, _ = simulate_rollout(res.x, t_exp)

plt.figure(figsize=(10, 5))
plt.plot(t_exp, alpha_exp, label="Real Hardware (Clamped)", color="black", alpha=0.7)
plt.plot(t_exp, alpha_fitted, "--", label="Model Simulation", color="crimson")
plt.xlabel("Time (s)")
plt.ylabel("Pendulum Angle (rad)")
plt.title("Clamped Pendulum System Identification Validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
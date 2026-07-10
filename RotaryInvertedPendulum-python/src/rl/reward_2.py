import jax.numpy as jnp


def compute_reward(
    theta,
    pen_vel,
    motor_pos,
    motor_vel,
    action,
    prev_action=0.0,
    *,
    motor_limit=2.18,
    k_theta=10.0,
    k_pen_vel=0.001,
    k_motor_pos=0.5,
    k_motor_vel=0.005,
    k_action=0.2,
    k_action_rate=0.0,
):
    """
    Quadratic reward for Furuta pendulum.

    theta = 0 is upright.
    """

    reward = -(
        k_theta * theta**2
        + k_pen_vel * pen_vel**2
        + k_motor_pos * (motor_pos / motor_limit)**2
        + k_motor_vel * motor_vel**2
        + k_action * action**2
        + k_action_rate * (action - prev_action)**2
    )

    return reward
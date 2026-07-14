# env.py

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

from flax import struct


# ============================================================
# Parameters
# ============================================================

MODEL_PATH = "model.xml"

CONTROL_FREQ = 35
PHYSICS_DT = 0.001

N_SUBSTEPS = int(
    round((1.0 / CONTROL_FREQ) / PHYSICS_DT)
)

MAX_ACCEL = 150.0
MAX_VEL = 5.0

MOTOR_LIMIT = 2.18

EPISODE_LENGTH = 280
NUM_ENVS = 2048


# ============================================================
# Load MJX model
# ============================================================

mj_model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

mjx_model = mjx.put_model(
    mj_model
)


# ============================================================
# Environment state
# ============================================================

@struct.dataclass
class EnvState:

    data: mjx.Data

    motor_target: jnp.ndarray
    motor_velocity: jnp.ndarray

    prev_action: jnp.ndarray

    step_count: jnp.ndarray
    episode_return: jnp.ndarray  # running sum of reward within the current episode


# ============================================================
# Utilities
# ============================================================

def wrap_angle(x):

    return (
        (x + jnp.pi)
        % (2.0 * jnp.pi)
    ) - jnp.pi



# ============================================================
# Physics
# ============================================================

def physics_step(data):

    def body(_, state):
        return mjx.step(
            mjx_model,
            state
        )

    return jax.lax.fori_loop(
        0,
        N_SUBSTEPS,
        body,
        data
    )



# ============================================================
# Reset
# ============================================================

def reset(
    rng,
    env_params=None
):

    rng_motor, rng_theta = jax.random.split(
        rng
    )


    data = mjx.make_data(
        mjx_model
    )


    motor_pos = jax.random.uniform(
        rng_motor,
        (),
        minval=-1.5,
        maxval=1.5
    )


    theta0 = jax.random.uniform(
        rng_theta,
        (),
        minval=-0.05,
        maxval=0.05
    )


    qpos = data.qpos

    qpos = qpos.at[0].set(
        motor_pos
    )

    qpos = qpos.at[1].set(
        theta0
    )


    data = data.replace(

        qpos=qpos,

        qvel=jnp.zeros_like(
            data.qvel
        )

    )


    state = EnvState(

        data=data,

        motor_target=motor_pos,

        motor_velocity=jnp.array(
            0.0
        ),

        prev_action=jnp.array(
            0.0
        ),

        step_count=jnp.array(
            0
        ),
        episode_return=jnp.array(0.0),

    )


    obs = get_obs(
        state
    )


    return obs, state



# ============================================================
# Observation
# ============================================================

def get_obs(state):

    qpos = state.data.qpos
    qvel = state.data.qvel


    motor_pos = qpos[0]


    theta = wrap_angle(
        qpos[1] - jnp.pi
    )


    return jnp.array([

        motor_pos,

        jnp.sin(theta),

        jnp.cos(theta),

        qvel[0],

        qvel[1],

        state.prev_action

    ])



# ============================================================
# Reward
# ============================================================

def get_reward(
    state,
    action
):

    qpos = state.data.qpos
    qvel = state.data.qvel


    theta = wrap_angle(
        qpos[1] - jnp.pi
    )


    reward = -(

        theta**2

        +0.5*qpos[0]**2

        +0.005*qvel[0]**2

        +0.001*qvel[1]**2

        +0.2*action**2

    )


    return reward



# ============================================================
# Single environment step
# ============================================================

def step(
    rng,
    state,
    action,
    env_params=None
):

    action = jnp.squeeze(action)
    action = jnp.clip(
        action,
        -1.0,
        1.0
    )

    acceleration = (
        action
        *
        MAX_ACCEL
    )

    velocity = jnp.clip(
        state.motor_velocity
        +
        acceleration / CONTROL_FREQ,
        -MAX_VEL,
        MAX_VEL
    )

    target = jnp.clip(
        state.motor_target
        +
        velocity / CONTROL_FREQ,
        -MOTOR_LIMIT,
        MOTOR_LIMIT
    )

    data = state.data.replace(
        ctrl=jnp.array([target])
    )

    data = physics_step(data)

    stepped_state = EnvState(
        data=data,
        motor_target=target,
        motor_velocity=velocity,
        prev_action=action,
        step_count=state.step_count + 1,
        episode_return=state.episode_return,  # updated just below
    )

    reward = get_reward(stepped_state, action)

    new_episode_return = state.episode_return + reward
    stepped_state = stepped_state.replace(episode_return=new_episode_return)

    done = stepped_state.step_count >= EPISODE_LENGTH

    # Auto-reset: if this episode just ended, swap in a fresh reset() state
    # so the NEXT call to step() starts a brand-new episode instead of
    # continuing to evolve the pendulum indefinitely past termination.
    obs_reset, state_reset = reset(rng, env_params)
    obs_stepped = get_obs(stepped_state)

    def select(reset_val, stepped_val):
        return jnp.where(done, reset_val, stepped_val)

    new_state = jax.tree_util.tree_map(select, state_reset, stepped_state)
    obs = jax.tree_util.tree_map(select, obs_reset, obs_stepped)

    info = {
        "returned_episode": done,
        "returned_episode_returns": new_episode_return,
        "timestep": stepped_state.step_count,
    }

    return (
        obs,
        new_state,
        reward,
        done,
        info
    )



# ============================================================
# Vectorized PureJaxRL interface
# ============================================================

def batch_reset(rngs, env_params=None):
    obs, states = jax.vmap(
        reset, in_axes=(0, None)
    )(rngs, env_params)
    return obs, states


def batch_step(rngs, states, actions, env_params=None):
    obs, states, reward, done, info = jax.vmap(
        step, in_axes=(0, 0, 0, None)
    )(rngs, states, actions, env_params)
    return obs, states, reward, done, info



# ============================================================
# Spaces
# ============================================================

def observation_space(
    env_params=None
):

    class Space:

        shape = (6,)

    return Space()



def action_space(
    env_params=None
):

    class Space:

        shape = (1,)

    return Space()
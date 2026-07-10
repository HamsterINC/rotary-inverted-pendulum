# train.py

import jax
import wandb

from env import (
    batch_reset,
    batch_step,
)

from purejaxrl.purejaxrl.ppo_continuous_action import make_train


# --------------------------------------------------
# PPO configuration
# --------------------------------------------------

config = {

    # environment
    "NUM_ENVS": 1024,
    "NUM_STEPS": 500,

    # training
    "TOTAL_TIMESTEPS": 10_000_000,

    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,

    # PPO parameters
    "LR": 3e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,

    "CLIP_EPS": 0.2,

    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,

    "MAX_GRAD_NORM": 0.5,

    # network
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,

}


# --------------------------------------------------
# Build PureJaxRL trainer
# --------------------------------------------------

train = make_train(
    config=config,
)


# --------------------------------------------------
# Run PPO
# --------------------------------------------------

rng = jax.random.PRNGKey(0)

out = train(rng)


# --------------------------------------------------
# Save parameters
# --------------------------------------------------

params = out["runner_state"].params


import pickle

with open(
    "ppo_rotary_pendulum.pkl",
    "wb"
) as f:
    pickle.dump(
        params,
        f
    )
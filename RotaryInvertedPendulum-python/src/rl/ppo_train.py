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
    "NUM_ENVS": 8192,
    "NUM_STEPS": 280,

    # training
    "TOTAL_TIMESTEPS": 200_000_000,

    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 512,

    # PPO parameters
    "LR": 5e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,

    "CLIP_EPS": 0.2,

    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,

    "MAX_GRAD_NORM": 0.5,

    # network
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "DEBUG": True,

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

params = out["runner_state"][0].params


import pickle

with open(
    "ppo_rotary_pendulum.pkl",
    "wb"
) as f:
    pickle.dump(
        params,
        f
    )
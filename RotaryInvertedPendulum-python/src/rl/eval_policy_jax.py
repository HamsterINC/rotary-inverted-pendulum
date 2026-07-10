import pickle

import jax
import jax.numpy as jnp
import numpy as np

import env
from purejaxrl.purejaxrl.ppo_continuous_action import ActorCritic  # adjust import path if needed


# ============================================================
# Load trained params
# ============================================================

with open("ppo_rotary_pendulum.pkl", "rb") as f:
    params = pickle.load(f)

network = ActorCritic(env.action_space().shape[0], activation="tanh")


# ============================================================
# Rollout a single episode (deterministic: use the mean action,
# not a sampled one, so evaluation isn't noisy).
#
# IMPORTANT: this is jitted and uses lax.scan instead of a Python
# for-loop, so JAX/XLA compiles it exactly ONCE instead of once per
# timestep. An un-jitted Python loop calling env.step() 280 times
# forces 280 separate CPU compiles of MJX's solver, which is what
# was exhausting memory (the "LLVM compilation error: Cannot
# allocate memory" spam).
# ============================================================

def _rollout_step(carry, _, deterministic):
    rng, obs, state = carry

    pi, value = network.apply(params, obs)

    if deterministic:
        action = pi.mean()  # mode of a Gaussian == mean
    else:
        rng, act_rng = jax.random.split(rng)
        action = pi.sample(seed=act_rng)

    rng, step_rng = jax.random.split(rng)
    obs, state, reward, done, info = env.step(step_rng, state, action)

    carry = (rng, obs, state)
    return carry, (obs, reward)


def _make_rollout_fn(deterministic):
    def rollout_fn(rng):
        obs, state = env.reset(rng)
        carry = (rng, obs, state)
        step_fn = lambda c, x: _rollout_step(c, x, deterministic)
        carry, (obs_history, reward_history) = jax.lax.scan(
            step_fn, carry, None, length=env.EPISODE_LENGTH
        )
        return jnp.sum(reward_history), obs_history, reward_history

    return jax.jit(rollout_fn)

jax.config.update("jax_log_compiles", True)

_rollout_deterministic = _make_rollout_fn(deterministic=True)
_rollout_stochastic = _make_rollout_fn(deterministic=False)


def rollout(rng, deterministic=True):
    fn = _rollout_deterministic if deterministic else _rollout_stochastic
    total_reward, obs_history, reward_history = fn(rng)
    return float(total_reward), np.array(obs_history), np.array(reward_history)


# ============================================================
# Run evaluation over several episodes (different init conditions)
# ============================================================

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    n_eval_episodes = 10

    rewards = []
    for i in range(n_eval_episodes):
        rng, eval_rng = jax.random.split(rng)
        total_reward, obs_hist, reward_hist = rollout(eval_rng, deterministic=True)
        rewards.append(total_reward)
        print(f"Episode {i}: total reward = {total_reward:.3f}")

    print(f"\nMean reward over {n_eval_episodes} episodes: {np.mean(rewards):.3f}")
    print(f"Std reward: {np.std(rewards):.3f}")

    # Optional: plot the theta (pendulum angle) trajectory of the last episode
    try:
        import matplotlib.pyplot as plt

        sin_theta = obs_hist[:, 1]
        cos_theta = obs_hist[:, 2]
        theta = np.arctan2(sin_theta, cos_theta)

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(theta)
        axes[0].set_ylabel("pendulum angle (rad, 0 = upright)")
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.5)

        axes[1].plot(reward_hist)
        axes[1].set_ylabel("reward")
        axes[1].set_xlabel("timestep")

        plt.tight_layout()
        plt.savefig("eval_rollout.png", dpi=150)
        print("\nSaved trajectory plot to eval_rollout.png")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")
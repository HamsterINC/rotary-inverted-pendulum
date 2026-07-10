import jax
import jax.numpy as jnp
import env  # your env.py

rng = jax.random.PRNGKey(0)

# Test 1: single env (no vmap), but through your actual reset()/step(), not raw mjx calls
obs, state = env.reset(rng)
print("reset qpos:", state.data.qpos)
action = jnp.array(0.0)
for i in range(5):
    obs, state, reward, done, info = env.step(rng, state, action)
print("Test 1 (single env, your reset/step) OK")

# Test 2: vmap over batch of 2, but IDENTICAL initial state (no randomness across batch)
rngs = jnp.stack([rng, rng])  # same rng twice -> identical states
obs, state = env.batch_reset(rngs)
print("Test 2 batched identical reset OK, qpos:", state.data.qpos)
action = jnp.zeros((2, 1))[..., 0]
step_rngs = jnp.stack([rng, rng])
obs, state, reward, done, info = env.batch_step(step_rngs, state, action, None)
print("Test 2 (vmap, identical states) OK")

# Test 3: vmap over batch of 2 with DIFFERENT random initial states (like your real sanity check)
rngs = jax.random.split(rng, 2)
obs, state = env.batch_reset(rngs)
print("Test 3 reset qpos:", state.data.qpos)
action = jnp.zeros((2,))
step_rngs = jax.random.split(rng, 2)
obs, state, reward, done, info = env.batch_step(step_rngs, state, action, None)
print("Test 3 (vmap, different random states) OK")

rngs = jax.random.split(rng, 2)
obs, state = env.batch_reset(rngs)
action = jnp.zeros((2, 1))   # shape (N,1), matching what PPO actually sends
step_rngs = jax.random.split(rng, 2)
obs, state, reward, done, info = env.batch_step(step_rngs, state, action, None)
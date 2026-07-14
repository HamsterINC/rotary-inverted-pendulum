"""
Convert a JAX/Flax `.pkl` checkpoint into a Stable-Baselines3-loadable `.zip`.
No environment rollouts happen here -- we only need the observation/action
*spaces* to construct a policy with the right layer shapes, then we copy your
JAX weights into it and let `model.save()` do the zipping.

USAGE
-----
1. Run with --inspect to print JAX param shapes vs. torch policy shapes.
2. Fill in JAX_TO_TORCH_MAPPING below.
3. Run without --inspect to produce the .zip.
"""

import argparse
import pickle
import sys

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

ALGOS = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}


# ---------------------------------------------------------------------------
# Fake env: exists only to carry observation_space/action_space so SB3 can
# size the policy's layers. No reset/step logic is ever actually exercised.
# ---------------------------------------------------------------------------
class SpacesOnlyEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Load + flatten the JAX/Flax pkl
# ---------------------------------------------------------------------------
def load_jax_params(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "params" in obj:
        obj = obj["params"]
    return obj


def flatten_params(params, prefix=""):
    flat = {}
    for k, v in params.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_params(v, key))
        else:
            flat[key] = np.asarray(v)
    return flat


# ---------------------------------------------------------------------------
# Build an SB3 policy with matching shapes (edit net_arch/activation to mirror
# your JAX net's hidden sizes and activation function)
# ---------------------------------------------------------------------------
def build_sb3_model(obs_dim, act_dim, algo="PPO", net_arch=None, activation_fn=torch.nn.Tanh,
                     discrete=False):
    if net_arch is None:
        net_arch = dict(pi=[256, 256], vf=[256, 256]) # PPO/A2C style; adjust for SAC/TD3

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    if discrete:
        act_space = gym.spaces.Discrete(act_dim)
    else:
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    fake_env = DummyVecEnv([lambda: SpacesOnlyEnv(obs_space, act_space)])
    policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn)
    model = ALGOS[algo]("MlpPolicy", fake_env, policy_kwargs=policy_kwargs, verbose=0)
    return model


# ---------------------------------------------------------------------------
# Mapping you fill in after --inspect
# ---------------------------------------------------------------------------
# (jax_flat_key, torch_state_dict_key, needs_transpose)
# transpose=True for Dense "kernel" -> nn.Linear "weight" (Flax kernel is
# (in, out); torch Linear weight is (out, in)). Biases: transpose=False.
JAX_TO_TORCH_MAPPING = [
    # Policy (actor) hidden layers
    ("Dense_0/kernel", "mlp_extractor.policy_net.0.weight", True),
    ("Dense_0/bias",   "mlp_extractor.policy_net.0.bias",   False),
    ("Dense_1/kernel", "mlp_extractor.policy_net.2.weight", True),
    ("Dense_1/bias",   "mlp_extractor.policy_net.2.bias",   False),
    # Policy final action head (NOT under mlp_extractor)
    ("Dense_2/kernel", "action_net.weight", True),
    ("Dense_2/bias",   "action_net.bias",   False),

    # Value (critic) hidden layers
    ("Dense_3/kernel", "mlp_extractor.value_net.0.weight", True),
    ("Dense_3/bias",   "mlp_extractor.value_net.0.bias",   False),
    ("Dense_4/kernel", "mlp_extractor.value_net.2.weight", True),
    ("Dense_4/bias",   "mlp_extractor.value_net.2.bias",   False),
    # Value final head
    ("Dense_5/kernel", "value_net.weight", True),
    ("Dense_5/bias",   "value_net.bias",   False),

    # Gaussian log std (state-independent), no transpose — it's a flat vector
    ("log_std", "log_std", False),
]


def apply_mapping(jax_flat, torch_state_dict, mapping):
    new_state_dict = dict(torch_state_dict)
    missing, shape_mismatch = [], []

    for jax_key, torch_key, transpose in mapping:
        if jax_key not in jax_flat:
            missing.append(("jax", jax_key))
            continue
        if torch_key not in torch_state_dict:
            missing.append(("torch", torch_key))
            continue

        tensor = torch.tensor(jax_flat[jax_key], dtype=torch_state_dict[torch_key].dtype)
        if transpose:
            tensor = tensor.T

        if tensor.shape != torch_state_dict[torch_key].shape:
            shape_mismatch.append((jax_key, torch_key, tuple(tensor.shape), tuple(torch_state_dict[torch_key].shape)))
            continue

        new_state_dict[torch_key] = tensor

    if missing:
        print("WARNING - missing keys:", missing)
    if shape_mismatch:
        print("WARNING - shape mismatches (not applied):")
        for m in shape_mismatch:
            print("  ", m)

    return new_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--act-dim", type=int, required=True)
    parser.add_argument("--discrete", action="store_true")
    parser.add_argument("--algo", default="PPO", choices=list(ALGOS.keys()))
    parser.add_argument("--out", default="converted_model.zip")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    jax_params = load_jax_params(args.pkl)
    jax_flat = flatten_params(jax_params)

    model = build_sb3_model(args.obs_dim, args.act_dim, algo=args.algo, discrete=args.discrete)
    torch_state_dict = model.policy.state_dict()

    if args.inspect:
        print("\n=== JAX params (flattened) ===")
        for k, v in jax_flat.items():
            print(f"  {k}: {v.shape}")
        print("\n=== Torch policy state_dict ===")
        for k, v in torch_state_dict.items():
            print(f"  {k}: {tuple(v.shape)}")
        print("\nFill in JAX_TO_TORCH_MAPPING above, then rerun without --inspect.")
        sys.exit(0)

    if not JAX_TO_TORCH_MAPPING:
        print("ERROR: JAX_TO_TORCH_MAPPING is empty. Run with --inspect first.")
        sys.exit(1)

    new_state_dict = apply_mapping(jax_flat, torch_state_dict, JAX_TO_TORCH_MAPPING)
    model.policy.load_state_dict(new_state_dict)
    model.save(args.out)
    print(f"Saved SB3-compatible zip to {args.out}")


if __name__ == "__main__":
    main()
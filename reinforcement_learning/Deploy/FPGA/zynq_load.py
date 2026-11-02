"""Load a desktop-trained SAC checkpoint on the Zynq for fine-tuning,
without touching SB3's cloudpickled metadata (the `data` entry of the zip).

Restores everything fine-tuning needs, not just the network weights:
  - policy.pth               actor, critics AND target critics
  - actor/critic optimizers  Adam moment estimates (avoids a bad first few
                             hundred updates from cold Adam statistics)
  - ent_coef optimizer       "
  - log_ent_coef             the LEARNED entropy temperature. A fresh SAC
                             starts at ent_coef = 1.0, typically 10-100x the
                             trained value, which floods the actions with
                             exploration noise at the start of fine-tuning.
and rebuilds the model with the training hyperparameters (net_arch, gamma,
mirror buffer) instead of SB3 defaults.

Usage on the Zynq:
    import json
    cfg = json.load(open("runs/<run>/config.json"))
    model = load_sac_for_finetune("runs/<run>/last.zip", real_env, cfg,
                                  buffer_dir="runs/<run>/replay_buffer")
    model.learn(total_timesteps=..., reset_num_timesteps=False)
"""

import io
import zipfile

import torch
from stable_baselines3 import SAC

from Training.portable_buffer import load_buffer


def _read_pth(archive, name, device):
    # No weights_only kwarg: old torch on the Zynq doesn't accept it, and
    # newer torch defaults to weights_only=True, which these files satisfy.
    with archive.open(name) as f:
        return torch.load(io.BytesIO(f.read()), map_location=device)


def load_sac_for_finetune(zip_path, env, run_config=None, buffer_dir=None,
                          buffer_size=200000, learning_rate=3e-4,
                          learning_starts=0, gamma=None, use_sde=False,
                          mirror=None, device="cpu", verbose=1):
    """Build a SAC matching the training setup and load the checkpoint.

    run_config: dict from the run's config.json (net-arch, control_freq_hz,
        mirror_augment, obs_history_len, obs_include_velocities). Values
        not present fall back to the train_sac.py defaults.
    gamma: override. Default = 0.99 ** (35 / control_freq_hz), the same
        constant-horizon rule train_sac.py uses. Pass it explicitly if you
        trained with --gamma.
    use_sde: must match training (--use-sde), since it changes the actor's
        parameter shapes. Not recorded in config.json.
    mirror: None = follow run_config["mirror_augment"].
    buffer_size: capacity on the Zynq; the most recent transitions are kept
        if the export is larger. ~200 B/transition for a 24-dim obs.
    """
    cfg = run_config or {}
    net_arch = [int(x) for x in cfg.get("net-arch", [256, 256])]
    control_freq = float(cfg.get("control_freq_hz", 50.0))
    if gamma is None:
        gamma = 0.99 ** (35.0 / control_freq)
    if mirror is None:
        mirror = bool(cfg.get("mirror_augment", False))

    kwargs = dict(
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=gamma,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        use_sde=use_sde,
        policy_kwargs=dict(net_arch=net_arch),
        device=device,
        verbose=verbose,
    )
    if mirror:
        import Training.symmetry as symmetry
        kwargs["replay_buffer_class"] = symmetry.MirrorReplayBuffer
        kwargs["replay_buffer_kwargs"] = {"obs_sign": symmetry.obs_signs(
            obs_history_len=int(cfg.get("obs_history_len", 4)),
            obs_include_velocities=bool(cfg.get("obs_include_velocities", True)),
        )}

    model = SAC("MlpPolicy", env, **kwargs)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())

        # Strict: a net_arch / use_sde mismatch fails loudly here.
        model.policy.load_state_dict(_read_pth(z, "policy.pth", device))

        for attr, fname in (
            (model.actor.optimizer, "actor.optimizer.pth"),
            (model.critic.optimizer, "critic.optimizer.pth"),
            (model.ent_coef_optimizer, "ent_coef_optimizer.pth"),
        ):
            if attr is not None and fname in names:
                attr.load_state_dict(_read_pth(z, fname, device))
            elif verbose:
                print("note: %s not in checkpoint, optimizer starts cold" % fname)

        if "pytorch_variables.pth" in names:
            pv = _read_pth(z, "pytorch_variables.pth", device)
            if "log_ent_coef" in pv and model.log_ent_coef is not None:
                # In-place copy keeps the tensor the optimizer is bound to.
                with torch.no_grad():
                    model.log_ent_coef.copy_(pv["log_ent_coef"].to(device))
        elif verbose:
            print("WARNING: no pytorch_variables.pth, ent_coef reset to 1.0")

    if verbose:
        print("Loaded %s: net_arch=%s gamma=%.4f ent_coef=%.4g mirror=%s"
              % (zip_path, net_arch, gamma,
                 float(torch.exp(model.log_ent_coef.detach())), mirror))

    if buffer_dir is not None:
        load_buffer(model.replay_buffer, buffer_dir)

    return model
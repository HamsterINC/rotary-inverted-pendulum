"""Per-run training-config provenance.

`train_sac.py` and `finetune_async.py` write a `config.json` into the run
directory recording the knobs that MUST match between training, fine-tuning,
and deployment (action mode, control rate, action scaling, reward extras).
Both action modes share identical obs/action spaces, so a checkpoint loaded
with the wrong flags fails silently — behaving nonsensically on the rig
instead of erroring. The deploy/fine-tune entrypoints call `check_config()`
to compare their CLI args against the checkpoint's recorded config and abort
on mismatch (`--ignore-config-mismatch` downgrades the abort to a warning).

Checkpoints predating this file have no config.json; validation is skipped.
"""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_NAME = "config.json"

# Keys recorded for provenance but NOT enforced by check_config: they do not
# change the observation/action layout or the objective, so a checkpoint
# trained one way can legitimately be resumed or deployed the other way.
# `mirror_augment` is the case in point — turning mirror data augmentation on
# for a later curriculum stage is a normal thing to want, and aborting on it
# would just teach people to pass --ignore-config-mismatch, which would then
# hide the mismatches that DO matter.
# `params_path` is provenance for a second reason: it is a filesystem path, so
# an absolute path recorded on one machine (or a re-run from a different cwd)
# would false-trip a comparison that is really about which rig's measured
# friction the sim used. It is inherited downstream rather than enforced.
PROVENANCE_ONLY_KEYS = frozenset({"mirror_augment", "motor_microsteps",
                                  "params_path"})

# How many directory levels above the policy file to search for config.json.
# Covers run_dir/last.zip (1), run_dir/checkpoints/sac_x.zip (2), and
# run_dir/distill_h16_aug/student.pt (2).
_SEARCH_LEVELS = 3


def save_run_config(run_dir: str | Path, config: dict) -> Path:
    path = Path(run_dir) / CONFIG_NAME
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def find_run_config(policy_path: str | Path) -> dict | None:
    """Walk up from a checkpoint path looking for the run's config.json."""
    p = Path(policy_path).resolve()
    for parent in list(p.parents)[:_SEARCH_LEVELS]:
        cfg = parent / CONFIG_NAME
        if cfg.exists():
            with open(cfg) as f:
                return json.load(f)
    return None


def check_config(policy_path: str | Path, expected: dict, *,
                 ignore: bool = False) -> None:
    """Compare CLI-derived values against the policy's recorded config.

    `expected` maps config keys to the values the caller is about to run
    with; keys absent from the saved config are skipped. Exits with an
    error on any mismatch unless `ignore` is True (then warns and
    continues). No-op when the checkpoint has no config.json.
    """
    saved = find_run_config(policy_path)
    if saved is None:
        return
    mismatches = []
    for key, cli_val in expected.items():
        if key in PROVENANCE_ONLY_KEYS:
            continue
        if key not in saved or saved[key] is None or cli_val is None:
            continue
        saved_val = saved[key]
        if isinstance(cli_val, (int, float)) and isinstance(saved_val, (int, float)):
            same = abs(float(cli_val) - float(saved_val)) < 1e-9
        else:
            same = cli_val == saved_val
        if not same:
            mismatches.append(
                f"  {key}: policy trained with {saved_val!r}, CLI gives {cli_val!r}"
            )
    if not mismatches:
        return
    msg = (f"config mismatch against the policy's {CONFIG_NAME}:\n"
           + "\n".join(mismatches))
    if ignore:
        print(f"WARNING: {msg}\n  (--ignore-config-mismatch given; continuing)")
    else:
        raise SystemExit(
            f"ERROR: {msg}\n"
            "Fix the flags to match the training config, or pass "
            "--ignore-config-mismatch to run anyway."
        )

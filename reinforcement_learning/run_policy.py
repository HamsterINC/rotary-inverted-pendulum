"""Run a trained policy on the real rig for several episodes, WITHOUT learning.

Same episode structure as finetune_async.py (PD-home the motor, stillness
check + tare, then one --episode-length-s episode, default 8 s, repeated
--episodes times), but every episode is a deterministic evaluation with the
loaded weights: no replay buffers are loaded, no gradient steps are taken and
no checkpoints are written.

This wrapper only forwards its arguments with --no-learn:
    python run_policy.py runs/<run> --inference cpu --port /dev/ttyPS1 --episodes 10
is the same as
    python finetune_async.py runs/<run> --inference cpu --port /dev/ttyPS1 --episodes 10 --no-learn

All finetune_async.py options still apply (--inference cpu|pl|baremetal,
--checkpoint, --control-freq, --episode-length-s, --run-name, ...).
Output goes to runs/run_<inference>_<source>_<timestamp>/ with metrics.csv
(mode = eval for every episode), timing.csv, step_timing.csv,
eval_summary.json and run_summary.json.
"""

import sys

from finetune_async import main

if __name__ == "__main__":
    raise SystemExit(main(["--no-learn", *sys.argv[1:]]))

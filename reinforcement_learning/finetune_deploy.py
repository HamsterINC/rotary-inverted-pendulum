"""PL-inference fine-tuning now lives in finetune_async.py (--inference pl), so
CPU and PL runs share one code path and one set of metrics.

This wrapper only forwards its arguments with --inference pl:
    python finetune_deploy.py runs/<run> --param-noise 0.01 --episodes 50
is the same as
    python finetune_async.py runs/<run> --inference pl --param-noise 0.01 --episodes 50
"""

import sys

from finetune_async import main

if __name__ == "__main__":
    raise SystemExit(main(["--inference", "pl", *sys.argv[1:]]))

"""Bare-metal R5 inference + Linux learner: finetune_async.py --inference baremetal.

    python finetune_baremetal.py runs/<run> --episodes 50
is the same as
    python finetune_async.py runs/<run> --inference baremetal --episodes 50

Load rpu_policy.elf on the R5 first (JTAG / XSCT, or --rpu-firmware with
remoteproc).

See Deploy/FPGA/rpu_firmware/README.md for building and loading the firmware.
"""

import sys

from finetune_async import main

if __name__ == "__main__":
    raise SystemExit(main(["--inference", "baremetal", *sys.argv[1:]]))

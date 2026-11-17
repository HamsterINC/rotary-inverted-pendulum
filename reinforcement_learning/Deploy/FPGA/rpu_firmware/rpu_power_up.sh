#!/bin/sh
# rpu_power_up.sh - power up Cortex-R5 #0 through the PMU firmware (run on the
# board, once per boot, BEFORE load_rpu.tcl on the PC):
#
#   sudo sh Deploy/FPGA/rpu_firmware/rpu_power_up.sh
#
# PYNQ powers the RPU off at boot. Powering it with raw PMU_GLOBAL.REQ_PWRUP
# writes leaves the R5's AXI ports dead (every access outside TCM aborts, and
# nobody outside reaches its TCM); waking it through the PMU firmware's PM API
# does the full sequence. The R5 starts executing TCM right away, so have
# load_rpu.tcl waiting on the PC (it halts the core as soon as it runs, then
# zeroes TCM and loads the firmware). Linux keeps the TCM banks and the core
# requested until the next reboot.
set -e
P=/sys/kernel/debug/zynqmp-firmware/pm
mountpoint -q /sys/kernel/debug || mount -t debugfs none /sys/kernel/debug
[ -w "$P" ] || { echo "no $P (needs CONFIG_ZYNQMP_FIRMWARE_DEBUG and root)"; exit 1; }

status() { echo "pm_get_node_status $1" > "$P"; cat "$P" | grep -E "Status|Usage" | tr -s ' \n' ' '; echo; }

case "$(status 7)" in
    *"Status: 1"*) echo "R5 #0 is already powered (node 7): nothing to do"; exit 0 ;;
esac

# TCM_0_A, TCM_0_B, TCM_1_A, TCM_1_B (lock-step R5 #0 uses all four)
for n in 15 16 17 18; do echo "pm_request_node $n 1 100 2" > "$P"; done
# wake RPU_0, boot address 0 (low vectors, TCM), blocking ack
echo "pm_request_wakeup 7 1 0 2" > "$P"
echo "R5 #0 (node 7): $(status 7)"

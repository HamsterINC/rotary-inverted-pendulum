# load_rpu.tcl - start rpu_policy.elf on Cortex-R5 #0 while PYNQ Linux keeps running.
#
#   xsdb load_rpu.tcl <path/to/rpu_policy.elf>
#   (or in the XSDB console:  set argv {<elf>}; source load_rpu.tcl)
#
# Needed once per board boot (the firmware lives in TCM). Only R5 #0 is reset;
# the A53s / Linux and the PL are not touched. Steps:
#   1. if R5 #0 has no power: wait until rpu_power_up.sh (run on the board)
#      has woken it through the PMU firmware, and halt it the moment it runs.
#      (Do not power it with raw PMU_GLOBAL.REQ_PWRUP writes: the core then
#      runs but its AXI ports are dead.)
#   2. low exception vectors (RPU_0_CFG.VINITHI = 0) so faults land in the
#      firmware's own vector table instead of OCM (ARM Trusted Firmware),
#   3. reset R5 #0 and zero all its TCM with 64-bit writes (fresh TCM has no
#      valid ECC; reading an unwritten word aborts),
#   4. clear the OCM mailbox (so a magic word left by an earlier load does not
#      count), download + run, then check the mailbox magic and heartbeat
#      through the AXI port, the same way Linux reads it.
#
# The image must be linked with rpu_boot.S / rpu_crt0.S (fixed BSP startup
# code); with the stock 2024.1 BSP copies the non-secure R5 aborts before main
# (see README.md).

if {[llength $argv] < 1} {
    error "usage: xsdb load_rpu.tcl <rpu_policy.elf>"
}
set elf [file normalize [lindex $argv 0]]

set PMU_PWR_STATE        0xFFD80100
set PWR_RPU_TCM          0xF400        ;# RPU island (bit 10) + TCM0A/0B/1A/1B (bits 12-15)
set WAKE_TIMEOUT_S       300
set RPU_GLBL_CNTL        0xFF9A0000
set RPU_0_CFG            0xFF9A0100
set MBOX                 0xFFFCFF00    ;# attach mailbox in OCM (rpu_shared.h MBOX_ADDR)
set MBOX_MAGIC           0x4D424F58

proc r5_state {} {
    if {[regexp {Cortex-R5 #0 \(([^)]*)\)} [targets] -> s]} { return $s }
    return "absent"
}

connect

# 1. power: woken by the PMU firmware (rpu_power_up.sh on the board)
set s [r5_state]
if {[string match -nocase "*power*" $s] || [string match -nocase "*reset*" $s]} {
    puts "R5 #0: $s. Run on the board now:  sudo sh rpu_power_up.sh  (waiting ${WAKE_TIMEOUT_S} s)"
    flush stdout
    set deadline [expr {[clock seconds] + $WAKE_TIMEOUT_S}]
    while {![string match -nocase "*running*" [set s [r5_state]]]} {
        if {[clock seconds] > $deadline} { error "R5 #0 not woken (state: $s)" }
        after 2
    }
    targets -set -filter {name =~ "*R5*#0"}
    stop                                   ;# it runs uninitialised TCM until here
    puts "R5 #0 woken and halted"
}
targets -set -nocase -filter {name =~ "*PSU*"}
set pwr [mrd -value $PMU_PWR_STATE]
if {($pwr & $PWR_RPU_TCM) != $PWR_RPU_TCM} {
    error [format "RPU/TCM not powered: PWR_STATE=0x%08X (run rpu_power_up.sh after a reboot)" $pwr]
}

# 2. low vectors (keep the other RPU_0_CFG bits)
set cfg [mrd -value $RPU_0_CFG]
mwr $RPU_0_CFG [expr {$cfg & ~0x4}]
set split [expr {([mrd -value $RPU_GLBL_CNTL] & 0x8) != 0}]
set tcm_dwords [expr {$split ? 0x2000 : 0x4000}]   ;# 64 KiB per TCM type split, 128 KiB lock-step

# 3. reset R5 #0 only, zero its TCM through the core
targets -set -filter {name =~ "*R5*#0"}
catch {stop}
rst -processor -clear-registers
mwr -size d 0x0 0 $tcm_dwords
mwr -size d 0x20000 0 $tcm_dwords

# 4. clear the mailbox, load and run
targets -set -nocase -filter {name =~ "*PSU*"}
mwr $MBOX 0 16
targets -set -filter {name =~ "*R5*#0"}
dow $elf
con
after 500

targets -set -nocase -filter {name =~ "*PSU*"}
set w [mrd -value $MBOX 4]
after 100
set hb2 [mrd -value [expr {$MBOX + 8}]]
targets -set -filter {name =~ "*R5*#0"}
stop
set pc [expr {"0x[lindex [rrd -nv pc] end]"}]
con
set magic [lindex $w 0]
puts [format "R5 #0: pc=0x%08X  mailbox magic=0x%08X fw=%d heartbeat=%d -> %d" \
      $pc $magic [lindex $w 1] [lindex $w 2] $hb2]
disconnect
if {$magic != $MBOX_MAGIC || $hb2 == [lindex $w 2]} {
    error "firmware did not come up (no mailbox magic or frozen heartbeat); see README.md bring-up notes"
}
puts "rpu_policy running on R5 #0 ([expr {$split ? {split} : {lock-step}}] mode)"

# Bare-metal policy on the Cortex-R5F (`--inference baremetal`)

The third inference path for `finetune_async.py`, next to `cpu` (Python control
loop on Linux) and `pl` (MLP in the fabric):

```
 A53 x4 — PYNQ Linux                        R5F core 0 — bare metal (this folder)
 ─────────────────────                      ───────────────────────────────────────
 learner (main thread, --learner-cpu N)     busy-wait for PL sample k·N (2 ms clock)
 rig thread: home, rest, tare, engage  ──►  read state (AXI GPIO) → obs → MLP (fp32)
   CMD_START, heartbeat, weight uploads     → SAC action → rail rules → accel strobe
   read step log, reward (reward.py)   ◄──  → PL timing latch → log record
            ▲                                         │
            └──── shared DDR window (rpu_shared.h) ───┘
```

The R5 runs **the cpu path's control loop, in C, on a core with nothing else on
it**, using the same low-level bitstream and the same PL counters
(`axi_gpio_meas`) for timing. That makes the three modes directly comparable:

```bash
python compare_jitter.py runs/async_cpu_X runs/async_pl_X runs/async_baremetal_X --plot jitter.png
```

## Files

| File | What |
|---|---|
| `rpu_shared.h` | Shared-memory contract. `Deploy/FPGA/rpu_client.py` parses its `#define`s at import, so the two sides can't drift. |
| `policy_core.c/.h` | Obs normalisation, MLP (ReLU, `[mu; log_std]` head), SAC sampler. No BSP calls. |
| `rpu_main.c` | Control loop, PL I/O, weight-bank swap, log, watchdog. |
| `lscript.ld` | Everything in TCM (code in ATCM, data/stack in BTCM). |
| `rpu_boot.S`, `rpu_crt0.S` | Copies of the Vitis 2024.1 R5 `boot.S` / `xil-crt0.S` with a BSP bug fixed (see "Bring-up notes"). Linked instead of the BSP's copies. |
| `load_rpu.tcl` | XSDB script: power up the RPU + TCM, low vectors, zero TCM, load, verify. |
| `test_policy_core.py` | Host test: builds `policy_core.c` with gcc, checks it against numpy/SB3 maths, syntax-checks `rpu_main.c`, checks the layout. |

## Shared window

The host allocates the 2 MiB window per run with `pynq.allocate` (contiguous,
uncached on Linux), so the stock PYNQ image needs no device-tree change. It
passes the window's address to the R5 through a 64-byte **attach mailbox** in
OCM at `0xFFFCFF00` (same address for both sides; see "Bring-up notes"):

1. The firmware boots, writes `MBOX_MAGIC`, bumps `MB_HEARTBEAT` and touches
   no DDR at all until a window is attached.
2. The host zeroes the window, writes `MB_SHM_BASE` and `MB_LEASE_MS`, then
   bumps `MB_ATTACH_SEQ`. The R5 checks the address (512 KiB aligned, below
   2 GiB), maps the uncached part with a free MPU region, initialises the
   control block and acks with `MB_ACK_SEQ` / `MB_ATTACHED_BASE`.
3. A host thread bumps `MB_LEASE` every `lease_ms / 8`. Linux frees the window
   when the process exits, so if the lease stalls for `lease_ms` (default
   2000; the process was killed), the R5 disengages the motor if an episode is
   running and drops the window without writing to it again.
4. `RpuClient.close()` (called by the env and at exit) aborts any episode,
   detaches (`MB_SHM_BASE = 0`), and only then frees the buffer.

While idle and attached, the R5 writes the window only in answer to a host
request (command ack, weight-bank ack).

Window layout (base = the attached address):

| Offset | Size | R5 caching | Content |
|---|---|---|---|
| `0x000000` | 512 KiB | cached, invalidated on swap | weight bank 0 (float32) |
| `0x080000` | 512 KiB | cached, invalidated on swap | weight bank 1 |
| `0x100000` | 4 KiB | non-cacheable (MPU) | control block |
| `0x101000` | 384 KiB | non-cacheable (MPU) | step-log ring, 1024 × 96 words |

Linux writes the inactive bank and commits it with `C_WB_REQ_VERSION`. The R5
switches banks only between two inferences, in the slack before it waits for
the next sample, so a swap never adds to the sample→action delay.

## Build (Vitis)

1. New platform from the **low-level** XSA (the one with `axi_gpio_meas`,
   e.g. `low_level_client_timing.xsa`), processor `psu_cortexr5_0`, OS `standalone`.
2. New empty C application on that platform, domain
   `standalone_psu_cortexr5_0` (the build log must show `armr5-none-eabi-gcc`;
   `aarch64` means the A53 domain was picked). Add `policy_core.c`,
   `policy_core.h`, `rpu_main.c`, `rpu_shared.h`, `rpu_boot.S`, `rpu_crt0.S`.
   The 2024.1 app CMake only collects `.c` files, so append to the end of
   `src/UserConfig.cmake` (outside the USER SETTINGS block):
   ```cmake
   set(_sources "${CMAKE_CURRENT_SOURCE_DIR}/rpu_boot.S;${CMAKE_CURRENT_SOURCE_DIR}/rpu_crt0.S")
   ```
   Check after the build: `armr5-none-eabi-objdump -d rpu_policy.elf | grep -c "ldr	r0, \[r0\]"`
   prints 3 when the fixed startup code is linked.
3. Replace the generated `lscript.ld` with this one. If your Vitis version's
   template uses different symbol names, keep the generated one and only move
   the sections: `.vectors`/`.text` → `psu_r5_atcm_0`, everything else →
   `psu_r5_btcm_0`, start `psu_r5_btcm_0` at `0x20040` (length `0xFFC0`;
   the first 64 bytes are the mailbox), and add the `.resource_table` section.
4. Linker flags: add `-lm`. Compiler: `-O2`. In `UserConfig.cmake` write the
   linker-script path with forward slashes
   (`set(USER_LINKER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/lscript.ld")`); a
   Windows path with backslashes makes Vitis fall back to a missing
   `linker_files/lscript_r5.ld.in`.
5. Build → `rpu_policy.elf`.

Host check (no Vitis needed):

```bash
python Deploy/FPGA/rpu_firmware/test_policy_core.py
```

## Loading the firmware (once per board power-up)

The stock PYNQ image has the R5 remoteproc driver only as a module and no R5
device-tree node, so the R5 is woken through the PMU firmware from Linux and
the firmware is loaded over JTAG from the PC (board USB-JTAG connected).
Linux keeps running; only R5 #0 is reset. Once per board boot:

1. On the PC, start the loader. It waits while R5 #0 has no power:
   ```bash
   C:/Xilinx/Vitis/2024.1/bin/xsdb.bat Deploy/FPGA/rpu_firmware/load_rpu.tcl <path>/rpu_policy.elf
   ```
2. On the board, wake the R5 through the PMU firmware:
   ```bash
   sudo sh Deploy/FPGA/rpu_firmware/rpu_power_up.sh
   ```
3. The loader halts the core as soon as it runs, zeroes TCM, loads the image
   and ends with `rpu_policy running on R5 #0` once the mailbox magic and a
   moving heartbeat are seen. The cable can be unplugged afterwards.

Reloading a new image in the same boot is step 1 alone (the R5 stays powered).

Do not use the IDE's Run/Debug with "Reset entire system" or "Program FPGA"
ticked, or `rst -system`: that resets the PS and kills Linux. The firmware
lives in TCM, so it survives host runs and bitstream reloads but not a power
cycle or reboot: run the script once per boot, **before** starting the host.

Then start the host without `--rpu-firmware`.

### Bring-up notes (why the script does what it does)

Found on the AUP-ZU3 with the stock PYNQ image:

- **RPU powered off.** Linux powers off unclaimed islands at boot.
  `pm_request_node` on the TCM banks alone leaves them off; the R5 has to be
  woken (`pm_request_wakeup 7`), which `rpu_power_up.sh` does. Powering the
  island with raw `PMU_GLOBAL.REQ_PWRUP_*` writes instead makes the core run,
  but its AXI ports stay dead: every R5 access outside TCM (DDR, OCM, LPD
  registers) takes an external abort and nobody outside reaches its TCM, with
  XPPU off and XMPUs at their defaults. The PMU wake starts the core on
  uninitialised TCM, so `load_rpu.tcl` waits for it and halts it at once.
- **Lock-step.** PYNQ's boot leaves the RPU in lock-step (R5 #1 shows
  "Lock Step Mode"); R5 #0 then sees 128 KiB ATCM at `0x0` and 128 KiB BTCM
  at `0x20000`. The image and mailbox fit either mode.
- **High vectors.** `RPU_0_CFG.VINITHI` is 1, so exceptions vector to
  `0xFFFF0000` in OCM, where ARM Trusted Firmware lives. The script clears
  it before the reset.
- **TCM ECC.** Freshly powered TCM has no valid ECC; reading a word that was
  never written aborts (DFSR `0x19`). The script zeroes all TCM through the
  core with 64-bit writes before `dow`.
- **BSP bug.** The R5 runs non-secure, and the 2024.1 SDT boot code checks
  whether it may touch the secure-only RPU registers with
  `ldr r0,=XCortexr5_ConfigTable; ands r0,#1`, which tests the table's
  address, not its `access-val` word. So `boot.S` and `xil-crt0.S` read
  `RPU_GLBL_CNTL` / `RPU_0_CFG` / `RPU_0_PWRCTL` and take a synchronous
  external abort (DFSR `0x1008`) before `main`. `rpu_boot.S` and
  `rpu_crt0.S` load the word first. (`standalone_lockstep_mode_debug=true`
  in the BSP settings only removes the first of the three accesses.)
- **PL clock crossing (affects cpu and baremetal timing).** The AXI GPIOs run
  on `clk_pl_0`, the low-level core (`training_pipeline` in the Vivado
  project's `lowlevelserver.sv`) on the board oscillator `sys_clk`, declared
  asynchronous in `zu3.xdc`. `ctrl` entered unsynchronised, so ~8% of accel
  strobes were lost or mis-latched (40 / 29 / 21 ms "periods", ticks without
  a timing latch; the misses recur at the beat of the two clocks). Fixed in
  the HDL: `ctrl` is two-flop synchronised and `accel_cmd` is taken only on
  the synchronised strobe (cleared while disengaged). Needs a rebuilt
  low-level bitstream. The latch read back (PL to PS) is not synchronised
  either, so `rpu_main.c read_meas` and `FpgaLowLevelClient._hal_read_timing`
  accept it only when two reads in a row agree. Until the new bitstream is in
  use, compare runs on the std of the nominal ticks (|period - 20 ms| < 1 ms)
  rather than the overall std.
- **Mailbox in OCM.** The attach mailbox sits at `0xFFFCFF00` (OCM area used
  only by FSBL during boot; ARM Trusted Firmware is at `0xFFFEA000+`), mapped
  non-cacheable on the R5 by its own MPU region. It was in TCM first, but with
  the raw power-up nobody outside could reach TCM; OCM works either way.

## Optional: fixed window and remoteproc

Only needed if you prefer them over the defaults above.

**Fixed window**: reserve 2 MiB `no-map` in the boot device tree and pass its
address with `--rpu-shm-base` instead of letting the host allocate one.
Device-tree fragment:

```dts
/dts-v1/;
/plugin/;
/ {
    fragment@0 {
        target-path = "/reserved-memory";
        __overlay__ {
            #address-cells = <2>;
            #size-cells = <2>;
            rpu_shm: rpu_shm@3ee00000 {
                no-map;
                reg = <0x0 0x3ee00000 0x0 0x200000>;
            };
        };
    };
};
```

Reserved memory is applied at boot (on PYNQ the device tree is inside
`image.ub`), not by a runtime overlay. Check with `cat /proc/iomem`: the
range must not be listed as `System RAM`.

**remoteproc**: if `ls /sys/class/remoteproc` shows `remoteproc0` (needs the
`zynqmp_r5_remoteproc` module loaded and an R5 node with TCM banks in the
device tree), `--rpu-firmware rpu_policy.elf` copies the ELF to
`/lib/firmware` and starts it (needs root). The image ships an empty resource
table, so drivers that require one accept it.

The host waits for the mailbox magic and a moving heartbeat before it does
anything.

## Run

```bash
sudo python finetune_async.py runs/<run> --inference baremetal \
    --control-freq 50 --learner-cpu 1 --episodes 50
```

For a fair comparison, give all three modes the same `--control-freq`,
`--learner-cpu`, `--weight-sync`, episode count and `--eval-every`, then
compare with `--mode eval` (deterministic, fixed weights) and `--mode train`.

## What the R5 does per tick (mirrors the cpu path)

1. Busy-poll `time_us` until the sample `N` samples after the last used one
   (`N = round(dt / 2 ms)`, ± `--dt-jitter-frac` like `AsyncControlLoop`).
   If that sample is already out, it is late: take the newest one.
2. Read a consistent sample (stamp, data, stamp; retry if torn).
3. Build the frame exactly like `FPGA_env._build_obs` (same norm presets,
   history seeded with the first frame, first frame has `motor_vel = 0`).
4. MLP → `mu`, `log_std` (clamped to SB3's [-20, 2]) → `tanh(mu + std·eps)`
   when exploring, `tanh(mu)` for evals.
5. `FPGA_env.apply_action`'s rail rule, then `FpgaLowLevelClient`'s clamp,
   rail brake and Q-format, write + strobe.
6. Read back the PL timing latch, log the record (state, obs, action, timing,
   R5 cycles).
7. Episode ends on `max_steps` or `|motor_pos| ≥ 135°`: final record with the
   last state, accel 0, `ST_DONE`. Linux homes the motor next.

## Safety

- Linux engages the motor and owns it between episodes. The R5 only writes
  the accel register and the strobe while `ST_RUNNING`.
- Ctrl-C / disengage on Linux first sets `C_ABORT` and waits for the R5 to
  leave `ST_RUNNING`, then disengages. On an abort the R5 never pulses the
  ctrl register again (which would rewrite the engage bit).
- Host watchdog: if the rig thread stops bumping `C_HOST_HEARTBEAT` for
  `--rpu-host-timeout-ms` (default 1000), the R5 disengages the motor itself
  and reports `ERR_HOST_LOST`.
- Lease: if the host process dies (no `MB_LEASE` renewal for `lease_ms`), the
  R5 disengages if mid-episode and stops using the window Linux freed.
- SW1 and the PL limits work as before.

## Hardware status

Verified on the AUP-ZU3 (stock PYNQ image, Vitis 2024.1): the firmware
builds, is loaded by `load_rpu.tcl`, runs in lock-step on R5 #0 and publishes
the mailbox magic and heartbeat in OCM; after the PMU wake the R5 reaches
DDR, OCM and the LPD registers. `RpuClient` on the board (with the low-level
overlay loaded) attaches a `pynq.allocate` window (got `0x78500000`), reads
the control block the R5 initialised there, and detaches cleanly. The
low-level bitstream's GPIOs sit at `0x80000000`-`0x80040000` (HPM0_LPD).

Run the host under `sudo` with PYNQ's environment, e.g.
`sudo bash -c 'source /etc/profile.d/xrt_setup.sh; source /etc/profile.d/pynq_venv.sh; <venv>/bin/python finetune_async.py ...'`
(plain `sudo python` finds no XRT device, and `pynq.allocate` needs an
overlay loaded first, which the env does).

Not yet verified: the R5 driving the low-level bitstream's AXI GPIOs during
an episode, and whether the forward
pass fits in the period (`rpu_compute_us` in timing.csv; about 1–3 ms
expected for 256×256, because the ~0.3 MB of weights stream from DDR on every
tick).

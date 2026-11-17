/*
 * rpu_shared.h - shared-memory contract between the bare-metal policy on the
 * Cortex-R5F (rpu_policy.c) and the Linux learner (Deploy/FPGA/rpu_client.py).
 *
 * Keep this file and the constants at the top of rpu_client.py in lockstep:
 * every offset below has a twin there (same name, prefix-free).
 *
 * Physical layout. The window is allocated by Linux at run time (pynq.allocate,
 * contiguous, uncached on Linux) and handed to the firmware through the attach
 * mailbox in OCM (MBOX_* below), so no device-tree reservation is
 * needed. SHM_BASE = the address the host attached, SHM_ALIGN aligned:
 *
 *   SHM_BASE + 0x000000  weight bank 0   (512 KiB, float32)   R5: cacheable
 *   SHM_BASE + 0x080000  weight bank 1   (512 KiB, float32)   R5: cacheable
 *   SHM_BASE + 0x100000  control block   (4 KiB, u32 words)   R5: non-cacheable
 *   SHM_BASE + 0x101000  step log ring   (LOG_DEPTH records)  R5: non-cacheable
 *
 * The R5 never writes a weight bank, and invalidates a bank's cache lines when
 * it switches to it, so Linux can write the inactive bank through its uncached
 * mapping. The control block and the log are non-cacheable on the R5 (MPU
 * region) and uncached on Linux, so plain volatile accesses suffice.
 *
 * Linux frees the window when its process exits, so the firmware only touches
 * it while attached: the host bumps MB_LEASE from a background thread, and if
 * that stops (process killed) the R5 disengages if mid-episode and detaches.
 * While idle the R5 only writes the window in answer to a host request.
 *
 * All numbers are little-endian 32-bit words; "f" fields hold IEEE float32.
 */
#ifndef RPU_SHARED_H
#define RPU_SHARED_H

#define SHM_SIZE            0x00200000u
#define SHM_ALIGN           0x00080000u    /* the uncached MPU region must be size-aligned */
#define SHM_R5_LIMIT        0x80000000u    /* window must lie in low DDR (R5 32-bit view) */

/* ---- attach mailbox: 64 bytes of on-chip RAM (OCM), same address for the R5
 * and Linux. Not in R5 TCM: on PYNQ, Linux (and the JTAG AXI port) get bus
 * errors on the TCM's global address. 0xFFFCFF00 lies in the area FSBL uses
 * during boot only (OCM 0xFFFC0000-0xFFFE9FFF; ARM Trusted Firmware sits at
 * 0xFFFEA000+). The R5 maps MBOX_MPU_BYTES around it non-cacheable. -------- */
#define MBOX_ADDR           0xFFFCFF00u
#define MBOX_BYTES          0x40u
#define MBOX_MPU_BYTES      0x100u         /* uncached MPU region, MBOX_ADDR is aligned to it */
#define MBOX_MAGIC          0x4D424F58u    /* "MBOX" */
/* firmware -> host */
#define MB_MAGIC            0
#define MB_FW_VERSION       1
#define MB_HEARTBEAT        2      /* +1 per main-loop pass (attached or not) */
#define MB_STATE            3      /* MBS_* */
#define MB_ACK_SEQ          4      /* = MB_ATTACH_SEQ once taken */
#define MB_ATTACHED_BASE    5      /* window in use, 0: none */
#define MB_ERROR            6      /* MBE_* of the last attach / detach */
/* host -> firmware */
#define MB_ATTACH_SEQ       8      /* host +1 after writing the two words below */
#define MB_SHM_BASE         9      /* physical window base, 0 = detach */
#define MB_LEASE_MS         10     /* detach if MB_LEASE stalls this long (0: never, max 8000) */
#define MB_LEASE            11     /* host +1 periodically while the window is alive */

#define MBS_WAITING         0
#define MBS_ATTACHED        1

#define MBE_NONE            0
#define MBE_BAD_BASE        1      /* 0, misaligned or not below SHM_R5_LIMIT */
#define MBE_LEASE_LOST      2      /* host stopped renewing: window dropped */

#define SHM_BANK0_OFF       0x000000u
#define SHM_BANK1_OFF       0x080000u
#define SHM_BANK_BYTES      0x080000u
#define SHM_BANK_WORDS      (SHM_BANK_BYTES / 4u)
#define SHM_CTRL_OFF        0x100000u
#define SHM_LOG_OFF         0x101000u
#define SHM_UNCACHED_OFF    0x100000u      /* start of the R5 non-cacheable MPU region */
#define SHM_UNCACHED_BYTES  0x080000u      /* 512 KiB, aligned to its size */

#define RPU_MAGIC           0x52505531u    /* "RPU1" */
#define RPU_FW_VERSION      3u

/* ---- control block: word offsets ------------------------------------------ */
/* firmware -> host */
#define C_MAGIC             0
#define C_FW_VERSION        1
#define C_HEARTBEAT         2      /* +1 per PL sample / control tick in an episode */
#define C_STATE             3      /* ST_* */
#define C_ERROR             4      /* ERR_* of the last failed episode */
#define C_CPU_HZ            5      /* R5 clock: unit of compute_cyc */
#define C_CMD_ACK           8      /* = C_CMD_SEQ once the command was taken */
#define C_LOG_COUNT         10     /* records written since boot */
#define C_LOG_DEPTH         11
#define C_LOG_WORDS         12
#define C_EPISODE           13     /* episode id of the running/last episode */
#define C_EP_STEPS          14     /* control ticks done in it */
#define C_WB_ACTIVE_BANK    86
#define C_WB_ACTIVE_VERSION 87
#define C_BANK_WORDS_MAX    90
#define C_N_LATE            96     /* ticks whose sample was already old (whole run) */
#define C_N_TORN            97     /* state reads retried for tearing */
#define C_N_MEAS_RETRY      98     /* meas reads that had to wait for the strobe */
#define C_N_SWAPS           99

/* host -> firmware: commands */
#define C_CMD_SEQ           6      /* host +1 to issue C_CMD */
#define C_CMD               7      /* CMD_* */
#define C_ABORT             9      /* host sets 1 to stop an episode now */
#define C_HOST_HEARTBEAT    15     /* host +1 while it supervises an episode */
#define C_HOST_TIMEOUT_MS   76     /* 0: no host watchdog (max 8000) */
#define C_HAS_SAMPLE_CYC    77     /* 1: axi_gpio_timing ch2 holds the sample cycle stamp */

/* host -> firmware: episode configuration (read at CMD_START) */
#define C_GPIO_STEP         16     /* AXI GPIO base addresses (R5 view = phys) */
#define C_GPIO_PEN          17
#define C_GPIO_CTRL         18
#define C_GPIO_TIMING       19
#define C_GPIO_MEAS         20     /* 0: no measurement GPIO */
#define C_POS_FRAC          21
#define C_VEL_FRAC          22
#define C_ACCEL_FRAC        23
#define C_ACCEL_FULL_SCALE  24     /* f  rad/s^2 at Q action 1.0 */
#define C_MAX_ACCEL         25     /* f  rad/s^2 at policy action 1.0 */
#define C_SAFE_LIMIT_RAW    26     /* f  client rail brake (raw PL frame) */
#define C_BRAKE_ACCEL       27     /* f  brake accel beyond it */
#define C_SIM_SAFE_LIMIT    28     /* f  env rail: no outward accel beyond it */
#define C_TERM_LIMIT        29     /* f  |motor_pos| ending the episode (0: off) */
#define C_CTRL_LEVEL        30     /* ctrl GPIO level bits while running (engage) */
#define C_CTRL_STB          31     /* accel strobe bit */
#define C_SAMPLES_PER_TICK  32
#define C_SAMPLE_PERIOD_US  33
#define C_MAX_STEPS         34
#define C_HISTORY           35
#define C_N_FRAME           36     /* 6 (velocities) or 4 */
#define C_EXPLORE           37     /* 1: tanh(mu + std*eps), 0: tanh(mu) */
#define C_SEED              38     /* non-zero */
#define C_DT_JITTER         39     /* f  tick interval x uniform(1-f, 1+f) */
#define C_EPISODE_ID        40
#define C_OBS_GAIN          41     /* f */
#define C_FRAME_FEAT        42     /* [8] FEAT_* per frame slot */
#define C_NORM_KIND         50     /* [8] NORM_* per frame slot */
#define C_NORM_P1           58     /* [8] f */
#define C_NORM_P2           66     /* [8] f */
#define C_LOG_STD_MIN       74     /* f */
#define C_LOG_STD_MAX       75     /* f */

/* host -> firmware: network (dims fixed while an episode runs) */
#define C_N_IN              80
#define C_H1                81
#define C_H2                82
#define C_N_OUT             83     /* 2: [mu, log_std] */
#define C_WB_REQ_BANK       84
#define C_WB_REQ_VERSION    85     /* written LAST: commits the bank */

#define CTRL_WORDS          128

/* states / commands / errors */
#define ST_IDLE     0
#define ST_RUNNING  1
#define ST_DONE     2
#define ST_ERROR    3

#define CMD_NONE    0
#define CMD_START   1

#define ERR_NONE        0
#define ERR_CONFIG      1      /* bad GPIO address / dims / history */
#define ERR_NO_WEIGHTS  2      /* no bank committed yet */
#define ERR_HOST_LOST   3      /* host heartbeat stalled: motor disengaged */
#define ERR_NO_SAMPLE   4      /* PL stopped publishing samples */
#define ERR_HOST_GONE   5      /* lease lost mid-episode: disengaged, window dropped */

/* frame features (order of FPGA_env._build_obs) */
#define FEAT_MOTOR_POS  0
#define FEAT_SIN        1
#define FEAT_COS        2
#define FEAT_MOTOR_VEL  3
#define FEAT_PEN_VEL    4
#define FEAT_PREV_ACT   5

/* pendulum_env._apply_norm kinds */
#define NORM_NONE       0
#define NORM_DIV        1      /* x / p1 */
#define NORM_CLIPDIV    2      /* clip(x / p1, -1, 1) */
#define NORM_GAIN       3      /* x * p1 */
#define NORM_ASINH      4      /* clip(asinh(x / p1) / p2, -1, 1), p2 = asinh(xm / p1) */

/* ---- step log record: word offsets ---------------------------------------- */
#define LOG_DEPTH       1024
#define LOG_WORDS       96
#define MAX_OBS         64     /* obs_dim = history * n_frame must fit */
#define MAX_HIDDEN      512

#define R_EPISODE       0
#define R_STEP          1      /* tick index in the episode (0 = first obs) */
#define R_FLAGS         2      /* F_* */
#define R_WEIGHT_VER    3      /* weight version that produced the action */
#define R_TIME_US       4      /* PL time stamp of the state in this record */
#define R_SAMPLE_CYC    5      /* PL cycle stamp of that sample */
#define R_MOTOR_POS     6      /* f  sim frame (= raw PL frame) */
#define R_PHI           7      /* f  tared pendulum angle, 0 = hanging */
#define R_MOTOR_VEL     8      /* f */
#define R_PEN_VEL       9      /* f */
#define R_ACTION        10     /* f  policy action in [-1, 1] (the transition's action) */
#define R_MEAN          11     /* f  network mu */
#define R_LOG_STD       12     /* f  clamped log_std */
#define R_EPS           13     /* f  N(0,1) draw (0 when deterministic) */
#define R_ACCEL_CMD     14     /* f  rad/s^2 written to the PL (after rail rules) */
#define R_MEAS_PERIOD   15     /* raw axi_gpio_meas ch1 after the strobe */
#define R_MEAS_ACTION   16     /* raw axi_gpio_meas ch2 after the strobe */
#define R_COMPUTE_CYC   17     /* R5 cycles: obs build + forward + action write */
#define R_LATE_US       18     /* how late the tick's sample already was (>= 0) */
#define R_WAIT_CYC      19     /* R5 cycles spent waiting for the sample */
#define R_OBS           32     /* f[MAX_OBS] network input of this tick */

#define F_HAS_ACTION    0x001  /* an action was applied at this tick */
#define F_BRAKE         0x002  /* client rail brake overrode the accel */
#define F_RAIL_ZERO     0x004  /* env rail zeroed an outward accel */
#define F_EXPLORED      0x008
#define F_LATE          0x010  /* the sample this tick used was not fresh */
#define F_TERMINATED    0x020  /* last record: |motor_pos| >= term limit */
#define F_TRUNCATED     0x040  /* last record: max steps reached */
#define F_MEAS_VALID    0x080  /* R_MEAS_* belong to this tick's strobe */
#define F_FIRST_ACTION  0x100  /* period spans the pause before the episode */
/* An aborted episode (host abort, lost host, no sample) ends without a
 * terminal record; ST_ERROR / C_ERROR say why. */

#endif /* RPU_SHARED_H */

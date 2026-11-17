/*
 * rpu_main.c - bare-metal SAC policy on the Cortex-R5F (Zynq UltraScale+ RPU).
 *
 * The A53 cluster runs PYNQ Linux and the SAC learner (finetune_async.py
 * --inference baremetal). This core only does inference, with nothing else on
 * it: no OS, no interrupts, no scheduler. Per control tick it
 *
 *     busy-waits for the PL sample N samples after the previous one
 *     -> reads the state from the low-level bitstream's AXI GPIOs
 *     -> builds the observation exactly like FPGA_env._build_obs
 *     -> runs the actor MLP in float32 and samples the SAC action
 *     -> applies FPGA_env.apply_action's rail rule and FpgaLowLevelClient's
 *        clamp / rail brake, writes the Q-format accel and strobes it
 *     -> reads back the PL loop-timing latch (axi_gpio_meas)
 *     -> logs one record (state, obs, action, timing) to shared memory.
 *
 * So it is the CPU inference path (async_control.AsyncControlLoop over
 * FPGA_env) moved to a dedicated bare-metal core, on the same bitstream and
 * timed by the same PL counters - the third point of the cpu / pl / baremetal
 * jitter comparison.
 *
 * Linux owns everything between episodes (homing, rest check, tare, engage)
 * and starts an episode with CMD_START once the motor is engaged at accel 0.
 * Rewards are computed on Linux from the logged states (Training/reward.py
 * stays the single source of truth). Weights arrive double-buffered (see
 * rpu_shared.h); a new bank is taken between two inferences only.
 *
 * The shared window lives in DDR that Linux allocates per run; the host
 * passes its address through the attach mailbox in OCM
 * (MBOX_*, rpu_shared.h) and keeps a lease on it. Without an attached window
 * the firmware touches no DDR at all.
 *
 * Build: Vitis standalone application for psu_cortexr5_0 on the low-level
 * XSA, sources policy_core.c + rpu_main.c, linker script lscript.ld (all in
 * TCM). See README.md.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "xil_cache.h"
#include "xil_io.h"
#include "xil_mpu.h"
#include "xparameters.h"
#include "xreg_cortexr5.h"

#include "policy_core.h"
#include "rpu_shared.h"

#if defined(XPAR_CPU_CORE_CLOCK_FREQ_HZ)          /* 2023.2+ system-device-tree BSP */
#define CPU_HZ ((uint32_t)XPAR_CPU_CORE_CLOCK_FREQ_HZ)
#elif defined(XPAR_CPU_CORTEXR5_0_CPU_CLK_FREQ_HZ)
#define CPU_HZ ((uint32_t)XPAR_CPU_CORTEXR5_0_CPU_CLK_FREQ_HZ)
#else
#error "R5 clock frequency not found in xparameters.h"
#endif

#define PI_F             3.14159265358979323846f
#define GPIO_CH1         0x0u
#define GPIO_CH2         0x8u
#define MEAS_CNT_MASK    0x00FFFFFFu
#define NO_SAMPLE_US     100000      /* no new PL sample for this long -> ERR_NO_SAMPLE */
#define TORN_RETRIES     4
#define MEAS_RETRIES     8

/* ---- empty remoteproc resource table (older zynqmp_r5 drivers want one) ---- */
struct rsc_table {
    uint32_t ver, num, reserved[2];
};
__attribute__((section(".resource_table"), used, aligned(8)))
const struct rsc_table resources = {1u, 0u, {0u, 0u}};

/* ---- shared memory -------------------------------------------------------- */
static uintptr_t shm;                        /* attached window base, 0: none */
#define MBOX ((volatile uint32_t *)MBOX_ADDR)
#define CTRL ((volatile uint32_t *)(shm + SHM_CTRL_OFF))
#define LOG  ((volatile uint32_t *)(shm + SHM_LOG_OFF))

/* The BSP's xpseudo_asm_gcc.h (pulled in by xil_io.h) defines these already. */
#ifndef dmb
#define dmb() __asm__ volatile("dmb" ::: "memory")
#endif
#ifndef dsb
#define dsb() __asm__ volatile("dsb" ::: "memory")
#endif

static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float ctrl_f(int i) { return u2f(CTRL[i]); }

/* ---- R5 PMU cycle counter --------------------------------------------------- */
static void pmu_init(void)
{
    uint32_t v;
    __asm__ volatile("mrc p15, 0, %0, c9, c12, 0" : "=r"(v));
    v |= 0x1u | 0x4u;                                   /* enable, reset CCNT */
    v &= ~0x8u;                                         /* count every cycle */
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(v));
    __asm__ volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(0x80000000u)); /* CCNT on */
}

static inline uint32_t cycles(void)
{
    uint32_t v;
    __asm__ volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(v));
    return v;
}

/* ---- episode configuration snapshot ---------------------------------------- */
typedef struct {
    uint32_t gpio_step, gpio_pen, gpio_ctrl, gpio_timing, gpio_meas;
    int has_sample_cyc;
    float pos_scale, vel_scale;              /* 2^-frac */
    int accel_frac;
    float accel_full_scale, max_accel, safe_limit_raw, brake_accel;
    float sim_safe_limit, term_limit;
    uint32_t ctrl_level, ctrl_stb;
    int samples_per_tick, sample_period_us, max_steps;
    int history, n_frame, explore;
    uint32_t seed, episode_id;
    float dt_jitter, obs_gain, ls_min, ls_max;
    int feat[8], norm_kind[8];
    float norm_p1[8], norm_p2[8];
    net_dims_t dims;
    uint32_t host_timeout_cyc;               /* 0: no watchdog */
} cfg_t;

typedef struct {
    int32_t time_us;
    uint32_t sample_cyc;
    float mp, pp, mv, pv;                    /* raw PL frame = sim frame */
} state_t;

/* ---- module state ---------------------------------------------------------- */
static cfg_t cfg;
static int have_weights;
static uint32_t active_bank, active_version;
static uint32_t log_count;
static uint32_t last_meas_seq = 0xFFFFFFFFu;
static uint32_t host_hb_seen, host_hb_cyc;
static int abort_reason;                     /* ERR_* or -1 for a host abort */
static uint32_t mpu_region;                  /* MPU slot of the uncached part of the window */
static uint32_t lease_seen, lease_cyc, lease_timeout_cyc;

static float hist[MAX_OBS];                  /* stacked frames, oldest first */
static float obs[MAX_OBS];
static float h1buf[MAX_HIDDEN], h2buf[MAX_HIDDEN];

static inline int16_t sext16(uint32_t v) { return (int16_t)(v & 0xFFFFu); }
static inline int32_t us_diff(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

static const float *bank_ptr(uint32_t bank)
{
    return (const float *)(shm + (bank ? SHM_BANK1_OFF : SHM_BANK0_OFF));
}

/* Take a newly committed weight bank (only ever between two inferences). */
static void service_weight_swap(void)
{
    uint32_t req = CTRL[C_WB_REQ_VERSION];
    dmb();
    if (req == active_version && have_weights)
        return;
    if (req == 0u)                           /* nothing committed yet */
        return;
    uint32_t bank = CTRL[C_WB_REQ_BANK] & 1u;
    /* Lines of this bank may still be cached from when it was last active. */
    Xil_DCacheInvalidateRange((INTPTR)bank_ptr(bank), SHM_BANK_BYTES);
    active_bank = bank;
    active_version = req;
    have_weights = 1;
    CTRL[C_WB_ACTIVE_BANK] = bank;
    dmb();
    CTRL[C_WB_ACTIVE_VERSION] = req;         /* ack: host may now write the other bank */
    CTRL[C_N_SWAPS] = CTRL[C_N_SWAPS] + 1u;
}

static int load_cfg(cfg_t *c)
{
    c->gpio_step = CTRL[C_GPIO_STEP];
    c->gpio_pen = CTRL[C_GPIO_PEN];
    c->gpio_ctrl = CTRL[C_GPIO_CTRL];
    c->gpio_timing = CTRL[C_GPIO_TIMING];
    c->gpio_meas = CTRL[C_GPIO_MEAS];
    c->has_sample_cyc = CTRL[C_HAS_SAMPLE_CYC] != 0u;
    c->pos_scale = ldexpf(1.0f, -(int)CTRL[C_POS_FRAC]);
    c->vel_scale = ldexpf(1.0f, -(int)CTRL[C_VEL_FRAC]);
    c->accel_frac = (int)CTRL[C_ACCEL_FRAC];
    c->accel_full_scale = ctrl_f(C_ACCEL_FULL_SCALE);
    c->max_accel = ctrl_f(C_MAX_ACCEL);
    c->safe_limit_raw = ctrl_f(C_SAFE_LIMIT_RAW);
    c->brake_accel = ctrl_f(C_BRAKE_ACCEL);
    c->sim_safe_limit = ctrl_f(C_SIM_SAFE_LIMIT);
    c->term_limit = ctrl_f(C_TERM_LIMIT);
    c->ctrl_level = CTRL[C_CTRL_LEVEL];
    c->ctrl_stb = CTRL[C_CTRL_STB];
    c->samples_per_tick = (int)CTRL[C_SAMPLES_PER_TICK];
    c->sample_period_us = (int)CTRL[C_SAMPLE_PERIOD_US];
    c->max_steps = (int)CTRL[C_MAX_STEPS];
    c->history = (int)CTRL[C_HISTORY];
    c->n_frame = (int)CTRL[C_N_FRAME];
    c->explore = CTRL[C_EXPLORE] != 0u;
    c->seed = CTRL[C_SEED];
    c->dt_jitter = ctrl_f(C_DT_JITTER);
    c->episode_id = CTRL[C_EPISODE_ID];
    c->obs_gain = ctrl_f(C_OBS_GAIN);
    c->ls_min = ctrl_f(C_LOG_STD_MIN);
    c->ls_max = ctrl_f(C_LOG_STD_MAX);
    for (int k = 0; k < 8; ++k) {
        c->feat[k] = (int)CTRL[C_FRAME_FEAT + k];
        c->norm_kind[k] = (int)CTRL[C_NORM_KIND + k];
        c->norm_p1[k] = ctrl_f(C_NORM_P1 + k);
        c->norm_p2[k] = ctrl_f(C_NORM_P2 + k);
    }
    c->dims.n_in = (int)CTRL[C_N_IN];
    c->dims.h1 = (int)CTRL[C_H1];
    c->dims.h2 = (int)CTRL[C_H2];
    c->dims.n_out = (int)CTRL[C_N_OUT];
    uint32_t to_ms = CTRL[C_HOST_TIMEOUT_MS];
    if (to_ms > 8000u)
        to_ms = 8000u;                       /* 32-bit cycle counter wraps after ~8 s */
    c->host_timeout_cyc = to_ms * (CPU_HZ / 1000u);

    if (!c->gpio_step || !c->gpio_pen || !c->gpio_ctrl || !c->gpio_timing)
        return ERR_CONFIG;
    if (c->n_frame != 4 && c->n_frame != 6)
        return ERR_CONFIG;
    if (c->history < 1 || c->history * c->n_frame != c->dims.n_in || c->dims.n_in > MAX_OBS)
        return ERR_CONFIG;
    if (c->dims.h1 < 1 || c->dims.h1 > MAX_HIDDEN || c->dims.h2 < 1 || c->dims.h2 > MAX_HIDDEN
        || c->dims.n_out != 2 || net_words(&c->dims) > SHM_BANK_WORDS)
        return ERR_CONFIG;
    if (c->samples_per_tick < 1 || c->sample_period_us < 1 || c->max_steps < 1
        || c->accel_full_scale <= 0.0f || c->accel_frac < 0 || c->accel_frac > 15)
        return ERR_CONFIG;
    for (int k = 0; k < c->n_frame; ++k)
        if (c->feat[k] < FEAT_MOTOR_POS || c->feat[k] > FEAT_PREV_ACT)
            return ERR_CONFIG;
    if (!have_weights)
        return ERR_NO_WEIGHTS;
    return ERR_NONE;
}

/* ---- PL I/O ------------------------------------------------------------- */

static void read_stamp(int32_t *t, uint32_t *cyc)
{
    *t = (int32_t)Xil_In32(cfg.gpio_timing + GPIO_CH1);
    *cyc = cfg.has_sample_cyc ? Xil_In32(cfg.gpio_timing + GPIO_CH2) : 0u;
}

/* One consistent sample: stamp, data, stamp; retry if the PL published a new
 * sample in between (FpgaLowLevelClient._hal_read_state). */
static void read_state(state_t *s)
{
    int32_t t0, t1;
    uint32_t c0, c1, mp, mv, pp, pv;
    for (int i = 0; i < TORN_RETRIES; ++i) {
        read_stamp(&t0, &c0);
        mp = Xil_In32(cfg.gpio_step + GPIO_CH1);
        mv = Xil_In32(cfg.gpio_step + GPIO_CH2);
        pp = Xil_In32(cfg.gpio_pen + GPIO_CH1);
        pv = Xil_In32(cfg.gpio_pen + GPIO_CH2);
        read_stamp(&t1, &c1);
        if (t0 == t1 && c0 == c1)
            break;
        CTRL[C_N_TORN] = CTRL[C_N_TORN] + 1u;
    }
    s->time_us = t1;
    s->sample_cyc = c1;
    s->mp = (float)sext16(mp) * cfg.pos_scale;
    s->pp = (float)sext16(pp) * cfg.pos_scale;
    s->mv = (float)sext16(mv) * cfg.vel_scale;
    s->pv = (float)sext16(pv) * cfg.vel_scale;
}

static void write_accel_q(int32_t q, int strobe)
{
    Xil_Out32(cfg.gpio_ctrl + GPIO_CH2, (uint32_t)q & 0xFFFFu);
    if (strobe) {
        dsb();                               /* data before the strobe */
        Xil_Out32(cfg.gpio_ctrl + GPIO_CH1, cfg.ctrl_level | cfg.ctrl_stb);
        Xil_Out32(cfg.gpio_ctrl + GPIO_CH1, cfg.ctrl_level);
    }
    dsb();
}

/* FPGA_env.apply_action (rail rule on the last observed motor_pos) followed by
 * FpgaLowLevelClient._apply_accel (NaN guard, clamp, rail brake on a fresh
 * raw read, Q-format, strobe). Returns the accel written, sets *flags. */
static float apply_accel(float action, float mp_last, uint32_t *flags)
{
    float a = action * cfg.max_accel;
    if (mp_last >= cfg.sim_safe_limit && a > 0.0f) {
        a = 0.0f;
        *flags |= F_RAIL_ZERO;
    } else if (mp_last <= -cfg.sim_safe_limit && a < 0.0f) {
        a = 0.0f;
        *flags |= F_RAIL_ZERO;
    }
    if (a != a)
        a = 0.0f;
    a = clampf(a, -cfg.accel_full_scale, cfg.accel_full_scale);
    float mp_raw = (float)sext16(Xil_In32(cfg.gpio_step + GPIO_CH1)) * cfg.pos_scale;
    if (mp_raw >= cfg.safe_limit_raw) {
        a = -cfg.brake_accel;
        *flags |= F_BRAKE;
    } else if (mp_raw <= -cfg.safe_limit_raw) {
        a = cfg.brake_accel;
        *flags |= F_BRAKE;
    }
    float qf = roundf(a / cfg.accel_full_scale * (float)(1 << cfg.accel_frac));
    int32_t q = (int32_t)clampf(qf, -32768.0f, 32767.0f);
    write_accel_q(q, 1);
    return a;
}

/* Loop-timing latch of the strobe just sent (FpgaLowLevelClient._record_timing). */
static int read_meas(uint32_t *period, uint32_t *action)
{
    if (!cfg.gpio_meas)
        return 0;
    for (int i = 0; i < MEAS_RETRIES; ++i) {
        *period = Xil_In32(cfg.gpio_meas + GPIO_CH1);
        *action = Xil_In32(cfg.gpio_meas + GPIO_CH2);
        uint32_t seq = (*period >> 24) & 0xFFu;
        if (seq != last_meas_seq) {
            last_meas_seq = seq;
            return 1;
        }
        CTRL[C_N_MEAS_RETRY] = CTRL[C_N_MEAS_RETRY] + 1u;
    }
    return 0;
}

/* ---- supervision ---------------------------------------------------------- */

/* The host process renews MB_LEASE from a background thread. When that stops
 * (process killed), Linux may already have freed the window. */
static int lease_expired(void)
{
    if (!lease_timeout_cyc)
        return 0;
    uint32_t l = MBOX[MB_LEASE], now = cycles();
    if (l != lease_seen) {
        lease_seen = l;
        lease_cyc = now;
        return 0;
    }
    return now - lease_cyc > lease_timeout_cyc;
}

/* Non-zero when the episode must stop now. Sets abort_reason. */
static int must_stop(void)
{
    if (lease_expired()) {
        abort_reason = ERR_HOST_GONE;
        return 1;
    }
    if (CTRL[C_ABORT]) {
        abort_reason = -1;
        return 1;
    }
    if (cfg.host_timeout_cyc) {
        uint32_t hb = CTRL[C_HOST_HEARTBEAT];
        uint32_t now = cycles();
        if (hb != host_hb_seen) {
            host_hb_seen = hb;
            host_hb_cyc = now;
        } else if (now - host_hb_cyc > cfg.host_timeout_cyc) {
            abort_reason = ERR_HOST_LOST;
            return 1;
        }
    }
    return 0;
}

/* Busy-wait for the sample stamped `target` (half a period of tolerance) or a
 * later one (FpgaLowLevelClient.wait_for_sample without the sleep). Returns how
 * late that sample already was (us, >= 0) or -1 to stop. */
static int32_t wait_for_sample(int32_t target)
{
    int32_t tol = cfg.sample_period_us / 2;
    int32_t t = (int32_t)Xil_In32(cfg.gpio_timing + GPIO_CH1);
    if (us_diff(target, t) <= tol)
        return us_diff(t, target) > 0 ? us_diff(t, target) : 0;   /* already there: late */
    int32_t last_t = t;
    uint32_t last_change = cycles();
    const uint32_t no_sample_cyc = (uint32_t)NO_SAMPLE_US * (CPU_HZ / 1000000u);
    for (;;) {
        t = (int32_t)Xil_In32(cfg.gpio_timing + GPIO_CH1);
        if (us_diff(t, target) >= -tol)
            return 0;
        if (t != last_t) {
            last_t = t;
            last_change = cycles();
            CTRL[C_HEARTBEAT] = CTRL[C_HEARTBEAT] + 1u;
            if (must_stop())                 /* checked once per 2 ms sample */
                return -1;
        } else if (cycles() - last_change > no_sample_cyc) {
            abort_reason = ERR_NO_SAMPLE;
            return -1;
        }
    }
}

/* ---- observation ------------------------------------------------------------- */

static void build_frame(const state_t *s, float motor_vel, float pen_vel, float prev,
                        float *out)
{
    float th = wrap_pi(s->pp - PI_F);
    float vals[6];
    vals[FEAT_MOTOR_POS] = s->mp;
    vals[FEAT_SIN] = sinf(th);
    vals[FEAT_COS] = cosf(th);
    vals[FEAT_MOTOR_VEL] = motor_vel;
    vals[FEAT_PEN_VEL] = pen_vel;
    vals[FEAT_PREV_ACT] = clampf(prev, -1.0f, 1.0f);
    for (int k = 0; k < cfg.n_frame; ++k)
        out[k] = apply_norm(cfg.norm_kind[k], cfg.norm_p1[k], cfg.norm_p2[k],
                            vals[cfg.feat[k]]) * cfg.obs_gain;
}

/* Push a frame (history seeded with copies of the first frame, as in FPGA_env). */
static void push_frame(const float *frame, int first)
{
    int nf = cfg.n_frame, n = cfg.dims.n_in;
    if (first) {
        for (int h = 0; h < cfg.history; ++h)
            memcpy(&hist[h * nf], frame, (size_t)nf * sizeof(float));
    } else {
        memmove(hist, hist + nf, (size_t)(n - nf) * sizeof(float));
        memcpy(&hist[n - nf], frame, (size_t)nf * sizeof(float));
    }
}

/* ---- log ------------------------------------------------------------------- */

typedef struct {
    int step;
    uint32_t flags;
    const state_t *s;
    float action, mean, log_std, eps, accel;
    uint32_t meas_period, meas_action, compute_cyc, wait_cyc;
    int32_t late_us;
} rec_t;

static void log_record(const rec_t *r)
{
    volatile uint32_t *w = LOG + (log_count % LOG_DEPTH) * LOG_WORDS;
    w[R_EPISODE] = cfg.episode_id;
    w[R_STEP] = (uint32_t)r->step;
    w[R_FLAGS] = r->flags;
    w[R_WEIGHT_VER] = active_version;
    w[R_TIME_US] = (uint32_t)r->s->time_us;
    w[R_SAMPLE_CYC] = r->s->sample_cyc;
    w[R_MOTOR_POS] = f2u(r->s->mp);
    w[R_PHI] = f2u(r->s->pp);
    w[R_MOTOR_VEL] = f2u(r->s->mv);
    w[R_PEN_VEL] = f2u(r->s->pv);
    w[R_ACTION] = f2u(r->action);
    w[R_MEAN] = f2u(r->mean);
    w[R_LOG_STD] = f2u(r->log_std);
    w[R_EPS] = f2u(r->eps);
    w[R_ACCEL_CMD] = f2u(r->accel);
    w[R_MEAS_PERIOD] = r->meas_period;
    w[R_MEAS_ACTION] = r->meas_action;
    w[R_COMPUTE_CYC] = r->compute_cyc;
    w[R_LATE_US] = (uint32_t)r->late_us;
    w[R_WAIT_CYC] = r->wait_cyc;
    for (int i = 0; i < cfg.dims.n_in; ++i)
        w[R_OBS + i] = f2u(obs[i]);
    dmb();
    log_count++;
    CTRL[C_LOG_COUNT] = log_count;           /* publish after the record */
}

/* ---- episode ------------------------------------------------------------- */

static void finish(int state, int err)
{
    CTRL[C_ERROR] = (uint32_t)err;
    dmb();
    CTRL[C_STATE] = (uint32_t)state;
}

static void run_episode(void)
{
    abort_reason = 0;
    service_weight_swap();
    int err = load_cfg(&cfg);
    if (err != ERR_NONE) {
        finish(ST_ERROR, err);
        return;
    }
    CTRL[C_EP_STEPS] = 0u;
    CTRL[C_ERROR] = ERR_NONE;
    host_hb_seen = CTRL[C_HOST_HEARTBEAT];
    host_hb_cyc = cycles();

    rng_t noise, jitter;
    rng_seed(&noise, cfg.seed);
    rng_seed(&jitter, cfg.seed ^ 0xA5A5A5A5u);

    /* First observation = FPGA_env.reset(): motor_vel 0, measured pen_vel. */
    state_t s;
    float frame[8];
    read_state(&s);
    build_frame(&s, 0.0f, s.pv, 0.0f, frame);
    push_frame(frame, 1);

    int step = 0, stopped = 0;
    uint32_t wait_cyc = 0;
    int32_t late_us = 0;
    for (;;) {
        /* ---- inference + action for tick `step` ---- */
        uint32_t c0 = cycles();
        rec_t r;
        memset(&r, 0, sizeof r);
        r.step = step;
        r.s = &s;
        r.flags = F_HAS_ACTION | (step == 0 ? F_FIRST_ACTION : 0u)
                  | (late_us > 0 ? F_LATE : 0u) | (cfg.explore ? F_EXPLORED : 0u);
        memcpy(obs, hist, (size_t)cfg.dims.n_in * sizeof(float));
        float out[2];
        mlp_forward(bank_ptr(active_bank), &cfg.dims, obs, h1buf, h2buf, out);
        r.mean = out[0];
        r.action = sac_action(out[0], out[1], cfg.ls_min, cfg.ls_max, cfg.explore,
                              &noise, &r.eps, &r.log_std);
        if (CTRL[C_ABORT]) {                 /* last check before touching the motor */
            abort_reason = -1;
            stopped = 1;
            break;
        }
        r.accel = apply_accel(r.action, s.mp, &r.flags);
        r.compute_cyc = cycles() - c0;
        if (read_meas(&r.meas_period, &r.meas_action))
            r.flags |= F_MEAS_VALID;
        r.late_us = late_us;
        r.wait_cyc = wait_cyc;
        log_record(&r);
        float prev_action = r.action;

        /* ---- weights for the next inference, in the slack before the wait ---- */
        service_weight_swap();

        /* ---- wait for the next tick's sample ---- */
        int n = cfg.samples_per_tick;
        if (cfg.dt_jitter > 0.0f) {
            float j = (2.0f * rng_uniform(&jitter) - 1.0f) * cfg.dt_jitter;
            n = (int)lroundf((float)cfg.samples_per_tick * (1.0f + j));
            if (n < 1)
                n = 1;
        }
        int32_t target = s.time_us + n * cfg.sample_period_us;
        uint32_t w0 = cycles();
        late_us = wait_for_sample(target);
        wait_cyc = cycles() - w0;
        if (late_us < 0) {
            stopped = 1;
            break;
        }
        if (late_us > 0)
            CTRL[C_N_LATE] = CTRL[C_N_LATE] + 1u;
        read_state(&s);
        step++;
        CTRL[C_EP_STEPS] = (uint32_t)step;
        CTRL[C_HEARTBEAT] = CTRL[C_HEARTBEAT] + 1u;

        /* ---- next observation (FPGA_env.observe_and_step) ---- */
        build_frame(&s, s.mv, s.pv, prev_action, frame);
        push_frame(frame, 0);
        int term = cfg.term_limit > 0.0f && fabsf(s.mp) >= cfg.term_limit;
        int trunc = step >= cfg.max_steps;
        if (term || trunc) {
            rec_t last;
            memset(&last, 0, sizeof last);
            last.step = step;
            last.s = &s;
            last.flags = (term ? F_TERMINATED : 0u) | (trunc ? F_TRUNCATED : 0u)
                         | (late_us > 0 ? F_LATE : 0u);
            last.late_us = late_us;
            last.wait_cyc = wait_cyc;
            memcpy(obs, hist, (size_t)cfg.dims.n_in * sizeof(float));
            log_record(&last);
            break;
        }
        if (must_stop()) {
            stopped = 1;
            break;
        }
    }

    if (stopped && abort_reason == ERR_HOST_GONE) {
        /* Host process gone: disengage and do not write the window again
         * (it may be freed already). main() detaches. */
        write_accel_q(0, 0);
        Xil_Out32(cfg.gpio_ctrl + GPIO_CH1, 0u);
        dsb();
    } else if (stopped && abort_reason == ERR_HOST_LOST) {
        write_accel_q(0, 0);
        Xil_Out32(cfg.gpio_ctrl + GPIO_CH1, 0u);  /* nobody supervises: disengage */
        dsb();
        finish(ST_ERROR, ERR_HOST_LOST);
    } else if (stopped) {
        /* Host abort: it disengages itself, so never pulse the ctrl register
         * (that would rewrite the engage level). */
        write_accel_q(0, 0);
        finish(abort_reason > 0 ? ST_ERROR : ST_DONE, abort_reason > 0 ? abort_reason : ERR_NONE);
    } else {
        write_accel_q(0, 1);                 /* accel 0 until Linux homes the motor */
        finish(ST_DONE, ERR_NONE);
    }
}

/* ---- window attach / detach ------------------------------------------------ */

static void detach(void)
{
    if (!shm)
        return;
    Xil_DisableMPURegionByRegNum(mpu_region);
    shm = 0;
    have_weights = 0;
    active_version = 0;
    lease_timeout_cyc = 0;
    dsb();
}

/* Take the window at `base` and initialise its control block. Returns MBE_*. */
static int attach(uint32_t base, uint32_t lease_ms, uint32_t *seen_cmd)
{
    if (base == 0u || (base & (SHM_ALIGN - 1u)) || base >= SHM_R5_LIMIT
        || SHM_R5_LIMIT - base < SHM_SIZE)
        return MBE_BAD_BASE;
    detach();
    /* Bank lines may still be cached from an earlier window at this address. */
    Xil_DCacheInvalidateRange((INTPTR)base, SHM_UNCACHED_OFF);
    /* Control block + log: non-cacheable. Weight banks stay cacheable (the
     * forward pass streams ~0.3 MB per tick) and are invalidated on a swap. */
    Xil_SetMPURegionByRegNum(mpu_region, (INTPTR)(base + SHM_UNCACHED_OFF),
                             SHM_UNCACHED_BYTES, NORM_NSHARED_NCACHE | PRIV_RW_USER_RW);
    shm = base;

    log_count = 0;
    CTRL[C_STATE] = ST_IDLE;
    CTRL[C_ERROR] = ERR_NONE;
    CTRL[C_CPU_HZ] = CPU_HZ;
    CTRL[C_LOG_COUNT] = 0u;
    CTRL[C_LOG_DEPTH] = LOG_DEPTH;
    CTRL[C_LOG_WORDS] = LOG_WORDS;
    CTRL[C_BANK_WORDS_MAX] = SHM_BANK_WORDS;
    CTRL[C_WB_ACTIVE_VERSION] = 0u;
    CTRL[C_N_LATE] = CTRL[C_N_TORN] = CTRL[C_N_MEAS_RETRY] = CTRL[C_N_SWAPS] = 0u;
    *seen_cmd = CTRL[C_CMD_SEQ];             /* the host zeroed the window: 0 */
    CTRL[C_CMD_ACK] = *seen_cmd;
    CTRL[C_WB_REQ_VERSION] = 0u;
    CTRL[C_ABORT] = 0u;
    CTRL[C_FW_VERSION] = RPU_FW_VERSION;
    dmb();
    CTRL[C_MAGIC] = RPU_MAGIC;

    if (lease_ms > 8000u)
        lease_ms = 8000u;                    /* 32-bit cycle counter wraps after ~8 s */
    lease_timeout_cyc = lease_ms * (CPU_HZ / 1000u);
    lease_seen = MBOX[MB_LEASE];
    lease_cyc = cycles();
    return MBE_NONE;
}

static void publish_attach_state(int err)
{
    MBOX[MB_ERROR] = (uint32_t)err;
    MBOX[MB_ATTACHED_BASE] = (uint32_t)shm;
    MBOX[MB_STATE] = shm ? MBS_ATTACHED : MBS_WAITING;
    dmb();
}

int main(void)
{
    pmu_init();
    /* Free MPU slots sit above the BSP's own regions, and higher numbers take
     * priority, so ours override the BSP's cacheable OCM / DDR regions. */
    mpu_region = Xil_GetNextMPURegion();
    /* The mailbox in OCM is cacheable in the BSP's map; Linux must see every
     * word at once, so give it its own non-cacheable region first. */
    Xil_SetMPURegionByRegNum(mpu_region, (INTPTR)MBOX_ADDR, MBOX_MPU_BYTES,
                             NORM_NSHARED_NCACHE | PRIV_RW_USER_RW);
    mpu_region = Xil_GetNextMPURegion();     /* next free slot: the window */

    publish_attach_state(MBE_NONE);
    uint32_t seen_attach = MBOX[MB_ATTACH_SEQ];  /* ignore a request from a previous load */
    MBOX[MB_ACK_SEQ] = seen_attach;
    MBOX[MB_FW_VERSION] = RPU_FW_VERSION;
    dmb();
    MBOX[MB_MAGIC] = MBOX_MAGIC;             /* host waits for this */

    uint32_t seen_seq = 0;
    for (;;) {
        MBOX[MB_HEARTBEAT] = MBOX[MB_HEARTBEAT] + 1u;

        uint32_t a = MBOX[MB_ATTACH_SEQ];
        if (a != seen_attach) {              /* attach (base != 0) or detach (0) */
            seen_attach = a;
            dmb();
            uint32_t base = MBOX[MB_SHM_BASE];
            int err = MBE_NONE;
            if (base)
                err = attach(base, MBOX[MB_LEASE_MS], &seen_seq);
            else
                detach();
            publish_attach_state(err);
            MBOX[MB_ACK_SEQ] = a;
            continue;
        }
        if (!shm)
            continue;                        /* no window: touch no DDR */
        if (lease_expired()) {
            detach();
            publish_attach_state(MBE_LEASE_LOST);
            continue;
        }

        service_weight_swap();
        uint32_t seq = CTRL[C_CMD_SEQ];
        if (seq != seen_seq) {
            seen_seq = seq;
            dmb();
            uint32_t cmd = CTRL[C_CMD];
            if (cmd == CMD_START) {
                /* State and episode id before the ack, so the host never
                 * mistakes the previous episode's ST_DONE for this one's. */
                CTRL[C_EPISODE] = CTRL[C_EPISODE_ID];
                CTRL[C_STATE] = ST_RUNNING;
                dmb();
                CTRL[C_CMD_ACK] = seq;
                run_episode();
                if (abort_reason == ERR_HOST_GONE) {
                    detach();
                    publish_attach_state(MBE_LEASE_LOST);
                }
            } else {
                CTRL[C_CMD_ACK] = seq;
            }
        }
    }
}

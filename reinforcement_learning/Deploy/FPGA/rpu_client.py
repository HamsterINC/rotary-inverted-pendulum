"""RpuClient: Linux side of the bare-metal policy on the Cortex-R5F.

Zynq UltraScale+ split for `finetune_async.py --inference baremetal`:

    A53 cluster (PYNQ Linux)              R5F core 0 (bare metal, rpu_firmware/)
    ------------------------              ---------------------------------------
    SAC learner (main thread)             control loop, locked to the PL's 2 ms
    rig thread: homing / rest / tare,       samples: state -> obs -> MLP -> action
      starts episodes, uploads weights,     -> accel strobe -> timing latch -> log
      reads the step log, computes reward

Both sides talk through a 2 MiB uncached DDR window (rpu_shared.h is the
contract; its #defines are parsed at import so the offsets can never drift
apart). The window is allocated here with pynq.allocate, so the stock PYNQ
image needs no device-tree reservation; its address goes to the firmware
through the attach mailbox in OCM (MBOX_ADDR). A background thread
renews a lease in that mailbox: if this process dies (and Linux frees the
window), the R5 disengages if mid-episode and stops touching the window.
close() detaches cleanly before the buffer is freed.

The R5 drives the motor through the same AXI GPIOs as FpgaLowLevelClient, so
it needs the low-level bitstream loaded and the motor engaged by Linux before
an episode starts.

    client = RpuClient(lowlevel_client)             # allocates + attaches the window
    client.configure(obs_layout..., dims...)        # once per run
    client.load_weights(flat_params)                # double-buffered, any time
    client.start_episode(episode_id, max_steps=..., explore=..., seed=...)
    while client.state() == ST_RUNNING: client.heartbeat(); client.read_new()
    client.abort()                                  # stop now (then disengage)
    client.close()                                  # detach, free the window

Firmware loading: start it over JTAG / XSCT before the host (firmware=None),
or RpuFirmware.start("rpu_policy.elf") through remoteproc when the image has
an R5 remoteproc node. Load the firmware first: the mailbox lives in R5 TCM,
which may be powered down while no firmware has run.
"""

from __future__ import annotations

import atexit
import math
import re
import shutil
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARED_HEADER = HERE / "rpu_firmware" / "rpu_shared.h"


def _parse_header(path: Path) -> dict[str, int]:
    """Integer #defines of rpu_shared.h (expressions of earlier names allowed)."""
    defs: dict[str, int] = {}
    pat = re.compile(r"^\s*#define\s+([A-Z_][A-Z0-9_]*)\s+(.+?)\s*(/\*.*)?$")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        name, expr = m.group(1), m.group(2)
        expr = re.sub(r"\b(0x[0-9A-Fa-f]+|\d+)[uU]\b", r"\1", expr)
        try:
            defs[name] = int(eval(expr, {"__builtins__": {}}, dict(defs)))
        except Exception:
            pass                              # include guards etc.
    return defs


_H = _parse_header(SHARED_HEADER)
S = SimpleNamespace(**_H)                     # S.C_STATE, S.R_OBS, S.F_*, S.ST_*, ...

ERROR_NAMES = {_H[k]: k for k in _H if k.startswith("ERR_")}
MBOX_ERROR_NAMES = {_H[k]: k for k in _H if k.startswith("MBE_")}
FEATURES = {"motor_pos": S.FEAT_MOTOR_POS, "sin": S.FEAT_SIN, "cos": S.FEAT_COS,
            "motor_vel": S.FEAT_MOTOR_VEL, "pen_vel": S.FEAT_PEN_VEL,
            "prev_act": S.FEAT_PREV_ACT}
SB3_LOG_STD_MIN, SB3_LOG_STD_MAX = -20.0, 2.0    # stable_baselines3.sac.policies
MEAS_CNT_MASK = 0xFFFFFF


def _f2u(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _u2f(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint32).view(np.float32)


class RpuError(RuntimeError):
    pass


@dataclass
class RpuRecords:
    """A block of step-log records as columns (sim frame, see rpu_shared.h)."""
    index: np.ndarray        # global record index
    episode: np.ndarray
    step: np.ndarray
    flags: np.ndarray
    weight_version: np.ndarray
    time_us: np.ndarray      # int64 (unwrapped later by the caller if needed)
    sample_cyc: np.ndarray
    motor_pos: np.ndarray
    phi: np.ndarray
    motor_vel: np.ndarray
    pen_vel: np.ndarray
    action: np.ndarray
    mean: np.ndarray
    log_std: np.ndarray
    eps: np.ndarray
    accel_cmd: np.ndarray
    meas_period: np.ndarray
    meas_action: np.ndarray
    compute_cyc: np.ndarray
    late_us: np.ndarray
    wait_cyc: np.ndarray
    obs: np.ndarray          # (n, obs_dim) float32

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def from_words(cls, start: int, w: np.ndarray, obs_dim: int) -> "RpuRecords":
        f = _u2f(w)
        return cls(
            index=np.arange(start, start + len(w)),
            episode=w[:, S.R_EPISODE].copy(), step=w[:, S.R_STEP].astype(np.int64),
            flags=w[:, S.R_FLAGS].copy(), weight_version=w[:, S.R_WEIGHT_VER].copy(),
            time_us=w[:, S.R_TIME_US].astype(np.int64), sample_cyc=w[:, S.R_SAMPLE_CYC].copy(),
            motor_pos=f[:, S.R_MOTOR_POS].copy(), phi=f[:, S.R_PHI].copy(),
            motor_vel=f[:, S.R_MOTOR_VEL].copy(), pen_vel=f[:, S.R_PEN_VEL].copy(),
            action=f[:, S.R_ACTION].copy(), mean=f[:, S.R_MEAN].copy(),
            log_std=f[:, S.R_LOG_STD].copy(), eps=f[:, S.R_EPS].copy(),
            accel_cmd=f[:, S.R_ACCEL_CMD].copy(),
            meas_period=w[:, S.R_MEAS_PERIOD].copy(), meas_action=w[:, S.R_MEAS_ACTION].copy(),
            compute_cyc=w[:, S.R_COMPUTE_CYC].copy(),
            late_us=w[:, S.R_LATE_US].view(np.int32).astype(np.int64),
            wait_cyc=w[:, S.R_WAIT_CYC].copy(),
            obs=f[:, S.R_OBS:S.R_OBS + obs_dim].copy(),
        )

    @staticmethod
    def concat(parts: list["RpuRecords"]) -> "RpuRecords":
        names = RpuRecords.__dataclass_fields__.keys()
        return RpuRecords(**{k: np.concatenate([getattr(p, k) for p in parts]) for k in names})


class RpuFirmware:
    """Start / stop the R5 image through Linux remoteproc."""

    def __init__(self, rproc: str = "remoteproc0", firmware_dir: str = "/lib/firmware"):
        self.sys = Path("/sys/class/remoteproc") / rproc
        self.firmware_dir = Path(firmware_dir)

    def state(self) -> str:
        return (self.sys / "state").read_text().strip()

    def start(self, elf: str | Path, restart: bool = True) -> None:
        elf = Path(elf).resolve()
        if not self.sys.exists():
            raise RpuError(f"{self.sys} not found: the device tree has no R5 remoteproc "
                           "node (see Deploy/FPGA/rpu_firmware/README.md)")
        if self.state() == "running":
            if not restart:
                return
            (self.sys / "state").write_text("stop")
        target = self.firmware_dir / elf.name
        if target.resolve() != elf:
            shutil.copyfile(elf, target)
        (self.sys / "firmware").write_text(elf.name)
        (self.sys / "state").write_text("start")
        print(f"[rpu] {self.sys.name}: started {elf.name} ({self.state()})")

    def stop(self) -> None:
        if self.sys.exists() and self.state() == "running":
            (self.sys / "state").write_text("stop")


class RpuClient:
    """Shared-memory driver for rpu_firmware (see module docstring)."""

    def __init__(self, lowlevel_client=None, *, shm_base: int | None = None,
                 mem=None, mbox=None, firmware: str | Path | None = None,
                 rproc: str = "remoteproc0", boot_timeout_s: float = 2.0,
                 host_timeout_ms: int = 1000, lease_ms: int = 2000):
        """lowlevel_client: FpgaLowLevelClient with the overlay loaded (GPIO
        addresses, Q formats).
        shm_base: None allocates the window with pynq.allocate (stock image);
        an address uses that fixed, device-tree-reserved window via /dev/mem.
        mem / mbox: objects with a uint32 `.array` standing in for the window
        and the mailbox (tests; mem needs shm_base).
        firmware: ELF to start through remoteproc first (None: already running).
        lease_ms: the R5 drops the window if this process stops renewing it."""
        if firmware is not None:
            RpuFirmware(rproc).start(firmware)
        if mbox is None:
            from pynq import MMIO
            mbox = MMIO(S.MBOX_ADDR, S.MBOX_BYTES)
        self.mb = mbox.array
        self.ll = lowlevel_client
        self.host_timeout_ms = int(host_timeout_ms)
        self.lease_ms = int(lease_ms)
        self.obs_dim = 0
        self.dims: tuple[int, int, int, int] | None = None
        self._req_version = 0
        self._hb = 0
        self._next_index = 0
        self._buf = None
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        self._attached = False
        self._lost = False
        self._wait_for_mailbox(boot_timeout_s)

        if mem is None:
            if shm_base is None:
                mem, shm_base = self._allocate_window()
            else:
                from pynq import MMIO
                mem = MMIO(int(shm_base), S.SHM_SIZE)
        elif shm_base is None:
            raise ValueError("mem= needs shm_base= (the address the R5 should use)")
        self.base = int(shm_base)
        self.mem = mem
        self.a = mem.array                    # uint32 view of the whole window
        self.ctrl_base = S.SHM_CTRL_OFF // 4
        self.log_base = S.SHM_LOG_OFF // 4
        self.bank_base = (S.SHM_BANK0_OFF // 4, S.SHM_BANK1_OFF // 4)
        try:
            self._attach(boot_timeout_s)
        except BaseException:
            self._free_window()
            raise
        atexit.register(self.close)
        self.cpu_hz = float(self.c(S.C_CPU_HZ))
        self.log_depth = self.c(S.C_LOG_DEPTH)
        self.log_words = self.c(S.C_LOG_WORDS)
        if self.log_words != S.LOG_WORDS:
            raise RpuError(f"firmware log record has {self.log_words} words, "
                           f"rpu_shared.h says {S.LOG_WORDS}: rebuild the firmware")

    # ------------------------------------------------------------------ raw access
    def c(self, i: int) -> int:
        return int(self.a[self.ctrl_base + i])

    def cf(self, i: int) -> float:
        return float(_u2f([self.c(i)])[0])

    def w(self, i: int, v: int) -> None:
        self.a[self.ctrl_base + i] = int(v) & 0xFFFFFFFF

    def wf(self, i: int, v: float) -> None:
        self.w(i, _f2u(float(v)))

    def m(self, i: int) -> int:
        return int(self.mb[i])

    def mw(self, i: int, v: int) -> None:
        self.mb[i] = int(v) & 0xFFFFFFFF

    # ------------------------------------------------------------------ attach
    def _wait_for_mailbox(self, timeout_s: float) -> None:
        t0 = time.monotonic()
        while self.m(S.MB_MAGIC) != S.MBOX_MAGIC:
            if time.monotonic() - t0 > timeout_s:
                raise RpuError(f"no R5 firmware (mailbox magic 0x{self.m(S.MB_MAGIC):08x} at "
                               f"0x{S.MBOX_ADDR:08x}); load rpu_policy.elf on "
                               "psu_cortexr5_0 first (XSCT, see rpu_firmware/README.md)")
            time.sleep(0.01)
        hb = self.m(S.MB_HEARTBEAT)
        time.sleep(0.005)
        if self.m(S.MB_HEARTBEAT) == hb:
            raise RpuError("R5 mailbox magic present but heartbeat frozen: the firmware "
                           "is halted or crashed (reload it)")
        fw = self.m(S.MB_FW_VERSION)
        if fw != S.RPU_FW_VERSION:
            raise RpuError(f"firmware version {fw}, host expects {S.RPU_FW_VERSION}: "
                           "rebuild / reload rpu_policy.elf")

    def _allocate_window(self):
        """Contiguous, uncached CMA buffer; the window is its SHM_ALIGN-aligned part."""
        from pynq import allocate
        buf = allocate(shape=((S.SHM_SIZE + S.SHM_ALIGN) // 4,), dtype=np.uint32,
                       cacheable=False)
        phys = int(buf.physical_address)
        base = (phys + S.SHM_ALIGN - 1) & ~(S.SHM_ALIGN - 1)
        if base + S.SHM_SIZE > S.SHM_R5_LIMIT:
            buf.freebuffer()
            raise RpuError(f"pynq.allocate returned 0x{phys:x}, above the R5's low-DDR "
                           f"limit 0x{S.SHM_R5_LIMIT:x}")
        self._buf = buf
        off = (base - phys) // 4
        return SimpleNamespace(array=buf[off:off + S.SHM_SIZE // 4]), base

    def _request(self, base: int, timeout_s: float) -> bool:
        """Attach (base != 0) or detach (0); True once the R5 acknowledged."""
        seq = (self.m(S.MB_ATTACH_SEQ) + 1) & 0xFFFFFFFF
        self.mw(S.MB_SHM_BASE, base)
        self.mw(S.MB_LEASE_MS, self.lease_ms)
        self.mw(S.MB_ATTACH_SEQ, seq)
        t0 = time.monotonic()
        while self.m(S.MB_ACK_SEQ) != seq:
            if time.monotonic() - t0 > timeout_s:
                return False
            time.sleep(0.0005)
        return True

    def _attach(self, timeout_s: float) -> None:
        self.a[:] = 0                         # firmware reads a clean control block
        self._lost = False
        self._lease_stop.clear()
        self._lease_thread = threading.Thread(target=self._renew_lease, name="rpu-lease",
                                              daemon=True)
        self._lease_thread.start()
        if not self._request(self.base, timeout_s):
            self._stop_lease()
            raise RpuError("R5 did not answer the attach request (firmware stuck in "
                           "an episode?)")
        err = self.m(S.MB_ERROR)
        if err != S.MBE_NONE or self.m(S.MB_ATTACHED_BASE) != self.base:
            self._stop_lease()
            raise RpuError(f"R5 refused window 0x{self.base:08x}: "
                           f"{MBOX_ERROR_NAMES.get(err, err)}")
        self._attached = True
        t0 = time.monotonic()
        while self.c(S.C_MAGIC) != S.RPU_MAGIC:   # written by the R5 on attach
            if time.monotonic() - t0 > timeout_s:
                raise RpuError("R5 attached but its control block never appeared: the "
                               "R5 and Linux see different memory at "
                               f"0x{self.base:08x}")
            time.sleep(0.001)
        print(f"[rpu] window 0x{self.base:08x} attached (lease {self.lease_ms} ms)")

    def _renew_lease(self) -> None:
        period = max(0.02, self.lease_ms / 1000.0 / 8)
        n = self.m(S.MB_LEASE)
        while not self._lease_stop.is_set():
            n = (n + 1) & 0xFFFFFFFF
            self.mw(S.MB_LEASE, n)
            self._lease_stop.wait(period)

    def _stop_lease(self) -> None:
        self._lease_stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=1.0)
            self._lease_thread = None

    def _free_window(self) -> None:
        if self._buf is not None:
            try:
                self._buf.freebuffer()
            except Exception:
                pass
            self._buf = None

    def close(self) -> None:
        """Abort a running episode, detach, free the window. Idempotent."""
        if not self._attached:
            self._stop_lease()
            self._free_window()
            return
        detached = False
        try:
            if not self._lost and self.c(S.C_STATE) == S.ST_RUNNING:
                self.abort()
            detached = self._request(0, 0.5)
        finally:
            self._attached = False
            self._stop_lease()
            if not detached and self._buf is not None:
                # The R5 did not confirm: wait for its lease to run out before
                # Linux may reuse the memory.
                print("[rpu] R5 did not confirm the detach; waiting for its lease to expire")
                time.sleep(self.lease_ms / 1000.0 * 1.5)
            self._free_window()

    def attached(self) -> bool:
        """False once the R5 dropped the window (lease lost) or after close()."""
        if self._attached and not self._lost and self.m(S.MB_ATTACHED_BASE) != self.base:
            self._lost = True
        return self._attached and not self._lost

    # ------------------------------------------------------------------ status
    def state(self) -> int:
        if not self.attached():               # the R5 no longer updates the window
            return S.ST_ERROR
        return self.c(S.C_STATE)

    def error(self) -> str:
        if self._lost:
            return ("ERR_HOST_GONE (R5 dropped the window: lease lost, "
                    f"{MBOX_ERROR_NAMES.get(self.m(S.MB_ERROR), self.m(S.MB_ERROR))})")
        return ERROR_NAMES.get(self.c(S.C_ERROR), str(self.c(S.C_ERROR)))

    def log_count(self) -> int:
        return self.c(S.C_LOG_COUNT)

    def heartbeat(self) -> None:
        """Host watchdog: call at least every host_timeout_ms during an episode."""
        self._hb = (self._hb + 1) & 0xFFFFFFFF
        self.w(S.C_HOST_HEARTBEAT, self._hb)

    def counters(self) -> dict:
        return {"n_late": self.c(S.C_N_LATE), "n_torn": self.c(S.C_N_TORN),
                "n_meas_retry": self.c(S.C_N_MEAS_RETRY), "n_swaps": self.c(S.C_N_SWAPS)}

    # ------------------------------------------------------------------ configuration
    def configure(self, *, n_in: int, h1: int, h2: int, history: int,
                  frame_feats: list[str], norm: dict, samples_per_tick: int,
                  sample_period_us: int, max_accel: float, sim_safe_limit: float,
                  term_limit: float, safe_limit_raw: float, brake_accel: float,
                  accel_full_scale: float) -> None:
        """Static settings for the run (written while the R5 is idle).

        frame_feats: names of the frame slots in order (FPGA_env._build_obs).
        norm: an OBS_NORM_PRESETS entry ({"feats": {name: spec}, "gain": g}).
        """
        if self.state() == S.ST_RUNNING:
            raise RpuError("configure() while an episode runs")
        ll = self.ll
        if ll is None:
            raise RpuError("configure() needs the low-level client (GPIO addresses)")
        n_frame = len(frame_feats)
        if n_in != history * n_frame or n_in > S.MAX_OBS:
            raise RpuError(f"obs dim {n_in} != history {history} x frame {n_frame} "
                           f"(max {S.MAX_OBS})")
        if max(h1, h2) > S.MAX_HIDDEN:
            raise RpuError(f"hidden width {max(h1, h2)} > firmware MAX_HIDDEN {S.MAX_HIDDEN}")
        ip = ll.ol.ip_dict
        addr = lambda name: int(ip[name]["phys_addr"])                               # noqa: E731
        self.w(S.C_GPIO_STEP, addr("axi_gpio_step"))
        self.w(S.C_GPIO_PEN, addr("axi_gpio_pen"))
        self.w(S.C_GPIO_CTRL, addr("axi_gpio_ctrl"))
        self.w(S.C_GPIO_TIMING, addr("axi_gpio_timing"))
        self.w(S.C_GPIO_MEAS, addr("axi_gpio_meas") if "axi_gpio_meas" in ip else 0)
        for name in ("axi_gpio_step", "axi_gpio_pen", "axi_gpio_ctrl", "axi_gpio_timing"):
            if addr(name) >> 32:
                raise RpuError(f"{name} at 0x{addr(name):x} is above 4 GiB: not "
                               "reachable from the R5")
        self.w(S.C_HAS_SAMPLE_CYC, int(bool(ll._has_sample_cyc)))
        self.w(S.C_POS_FRAC, ll.pos_frac_bits)
        self.w(S.C_VEL_FRAC, ll.vel_frac_bits)
        self.w(S.C_ACCEL_FRAC, ll.accel_frac_bits)
        self.wf(S.C_ACCEL_FULL_SCALE, accel_full_scale)
        self.wf(S.C_MAX_ACCEL, max_accel)
        self.wf(S.C_SAFE_LIMIT_RAW, safe_limit_raw if ll.safety else 1e9)
        self.wf(S.C_BRAKE_ACCEL, brake_accel)
        self.wf(S.C_SIM_SAFE_LIMIT, sim_safe_limit)
        self.wf(S.C_TERM_LIMIT, term_limit)
        self.w(S.C_SAMPLES_PER_TICK, samples_per_tick)
        self.w(S.C_SAMPLE_PERIOD_US, sample_period_us)
        self.w(S.C_HISTORY, history)
        self.w(S.C_N_FRAME, n_frame)
        feats = norm.get("feats", {})
        self.wf(S.C_OBS_GAIN, float(norm.get("gain", 1.0)))
        kinds = {None: S.NORM_NONE, "div": S.NORM_DIV, "clipdiv": S.NORM_CLIPDIV,
                 "gain": S.NORM_GAIN, "asinh": S.NORM_ASINH}
        for k in range(8):
            name = frame_feats[k] if k < n_frame else None
            spec = feats.get(name) if name else None
            kind = kinds[spec[0] if spec else None]
            p1 = float(spec[1]) if spec else 0.0
            p2 = math.asinh(spec[2] / spec[1]) if spec and spec[0] == "asinh" else 0.0
            self.w(S.C_FRAME_FEAT + k, FEATURES[name] if name else 0)
            self.w(S.C_NORM_KIND + k, kind)
            self.wf(S.C_NORM_P1 + k, p1)
            self.wf(S.C_NORM_P2 + k, p2)
        self.wf(S.C_LOG_STD_MIN, SB3_LOG_STD_MIN)
        self.wf(S.C_LOG_STD_MAX, SB3_LOG_STD_MAX)
        self.w(S.C_N_IN, n_in)
        self.w(S.C_H1, h1)
        self.w(S.C_H2, h2)
        self.w(S.C_N_OUT, 2)
        self.w(S.C_HOST_TIMEOUT_MS, self.host_timeout_ms)
        self.obs_dim = n_in
        self.dims = (n_in, h1, h2, 2)

    # ------------------------------------------------------------------ weights
    def load_weights(self, params: np.ndarray, timeout_s: float = 0.2) -> int:
        """Write a flat float32 weight image (actor_to_rpu_params) to the
        inactive bank and commit it; the R5 takes it before its next inference.
        Returns the new weight version."""
        params = np.ascontiguousarray(params, dtype=np.float32).ravel()
        if self.dims is not None:
            n_in, h1, h2, n_out = self.dims
            need = h1 * n_in + h1 + h2 * h1 + h2 + n_out * h2 + n_out
            if params.size != need:
                raise RpuError(f"weight image has {params.size} floats, dims need {need}")
        if params.size > S.SHM_BANK_WORDS:
            raise RpuError(f"weight image ({params.size} floats) > bank ({S.SHM_BANK_WORDS})")
        # The previous commit must have been taken, so the inactive bank is free.
        t0 = time.monotonic()
        while self.c(S.C_WB_ACTIVE_VERSION) != self._req_version:
            if time.monotonic() - t0 > timeout_s:
                raise RpuError("R5 did not take the previous weight bank in time")
            time.sleep(0.0005)
        active = self.c(S.C_WB_ACTIVE_BANK) if self._req_version else 1
        bank = 1 - active
        base = self.bank_base[bank]
        self.a[base:base + params.size] = params.view(np.uint32)
        self._req_version += 1
        self.w(S.C_WB_REQ_BANK, bank)
        self.w(S.C_WB_REQ_VERSION, self._req_version)
        return self._req_version

    def read_bank(self, bank: int, n: int) -> np.ndarray:
        base = self.bank_base[bank]
        return _u2f(np.array(self.a[base:base + n]))

    # ------------------------------------------------------------------ episodes
    def start_episode(self, episode_id: int, *, max_steps: int, explore: bool,
                      seed: int, dt_jitter: float = 0.0, ctrl_level: int,
                      ctrl_stb: int, timeout_s: float = 0.1) -> None:
        """Hand the (engaged, accel 0) motor to the R5 for one episode."""
        if not self.attached():
            raise RpuError("R5 window not attached (lease lost or closed)")
        if self.dims is None:
            raise RpuError("configure() first")
        if self._req_version == 0:
            raise RpuError("load_weights() first")
        self.w(S.C_MAX_STEPS, max_steps)
        self.w(S.C_EXPLORE, int(bool(explore)))
        self.w(S.C_SEED, (int(seed) & 0xFFFFFFFF) or 1)
        self.wf(S.C_DT_JITTER, dt_jitter)
        self.w(S.C_EPISODE_ID, episode_id)
        self.w(S.C_CTRL_LEVEL, ctrl_level)
        self.w(S.C_CTRL_STB, ctrl_stb)
        self.w(S.C_ABORT, 0)
        self.heartbeat()
        self._next_index = self.log_count()
        seq = (self.c(S.C_CMD_SEQ) + 1) & 0xFFFFFFFF
        self.w(S.C_CMD, S.CMD_START)
        self.w(S.C_CMD_SEQ, seq)
        t0 = time.monotonic()
        while self.c(S.C_CMD_ACK) != seq:
            if time.monotonic() - t0 > timeout_s:
                raise RpuError("R5 did not acknowledge S.CMD_START")
            time.sleep(0.0002)

    def abort(self, timeout_s: float = 0.05) -> bool:
        """Stop a running episode before the motor is touched by anyone else.
        Returns True once the R5 has left S.ST_RUNNING."""
        if not self.attached():
            return True                       # the R5 dropped the window and disengaged
        self.w(S.C_ABORT, 1)
        t0 = time.monotonic()
        while self.c(S.C_STATE) == S.ST_RUNNING:
            if time.monotonic() - t0 > timeout_s:
                return False
            time.sleep(0.0005)
        return True

    def read_new(self) -> RpuRecords | None:
        """Records written since the last call (or since start_episode)."""
        start = self._next_index
        count = self.log_count()
        if count <= start:
            return None
        if count - start > self.log_depth:
            raise RpuError(f"step log overrun: records {start}..{count - self.log_depth - 1} lost")
        n, depth, lw = count - start, self.log_depth, self.log_words
        words = np.empty((n, lw), dtype=np.uint32)
        i = start
        while i < count:                       # at most two slices (ring wrap)
            slot = i % depth
            k = min(count - i, depth - slot)
            a = self.log_base + slot * lw
            words[i - start:i - start + k] = np.array(self.a[a:a + k * lw],
                                                      dtype=np.uint32).reshape(k, lw)
            i += k
        if self.log_count() - start > depth:
            raise RpuError("R5 overwrote records while they were being read")
        self._next_index = count
        return RpuRecords.from_words(start, words, self.obs_dim)


# ---------------------------------------------------------------------- weights
def actor_to_rpu_params(actor) -> np.ndarray:
    """SB3 SAC actor -> flat float32 image for the R5:
    W1[h1][n_in] b1 W2[h2][h1] b2 W3[2][h2] b3, W3/b3 = [mu; log_std].
    The firmware hard-codes ReLU hidden layers and SB3's log_std clamp."""
    import torch as th
    from Deploy.FPGA.weight_loader import actor_float_params
    acts = [m for m in actor.latent_pi if not isinstance(m, th.nn.Linear)]
    if any(not isinstance(m, th.nn.ReLU) for m in acts):
        raise ValueError(f"R5 firmware implements ReLU only, actor has {acts}")
    W1, b1, W2, b2, W3, b3 = actor_float_params(actor, log_std=True)
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3]).astype(np.float32)


def rpu_dims(actor) -> tuple[int, int, int]:
    """(n_in, h1, h2) of an SB3 SAC actor."""
    import torch as th
    lins = [m for m in actor.latent_pi if isinstance(m, th.nn.Linear)]
    if len(lins) != 2:
        raise ValueError(f"R5 firmware runs 2 hidden layers, actor has {len(lins)}")
    return lins[0].in_features, lins[0].out_features, lins[1].out_features


def reference_forward(params: np.ndarray, dims: tuple[int, int, int], obs: np.ndarray):
    """numpy twin of policy_core.mlp_forward: returns (mu, log_std) per row."""
    n_in, h1, h2 = dims
    p = np.asarray(params, dtype=np.float32)
    o = 0
    def take(n):
        nonlocal o
        out = p[o:o + n]
        o += n
        return out
    W1 = take(h1 * n_in).reshape(h1, n_in); b1 = take(h1)
    W2 = take(h2 * h1).reshape(h2, h1); b2 = take(h2)
    W3 = take(2 * h2).reshape(2, h2); b3 = take(2)
    x = np.atleast_2d(np.asarray(obs, dtype=np.float32))
    y = np.maximum(x @ W1.T + b1, 0)
    y = np.maximum(y @ W2.T + b2, 0)
    out = y @ W3.T + b3
    return out[:, 0], out[:, 1]

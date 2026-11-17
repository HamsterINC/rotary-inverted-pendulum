"""Host test of the R5 firmware's math and of the shared-memory contract.

    python Deploy/FPGA/rpu_firmware/test_policy_core.py

1. Builds policy_core.c with the host gcc as a shared library and checks, via
   ctypes, the MLP forward pass against rpu_client.reference_forward (numpy),
   wrap_pi against pendulum_env._wrap_pi, apply_norm against _apply_norm, the
   SAC action against tanh(mu + exp(clamp(log_std)) * eps), and that the noise
   generator draws N(0, 1).
2. Syntax-checks rpu_main.c against stub BSP headers (catches C errors without
   Vitis; the ARM asm itself is only checked by the real build).
3. Checks that rpu_client parses every #define of rpu_shared.h and that the
   layout fits (log ring inside the uncached region, banks hold a 256x256 net).
4. Runs RpuClient's attach / lease / detach protocol against a Python model of
   the firmware's main loop (mailbox + window as numpy arrays).
No board, torch or mujoco needed.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[2]
sys.path.insert(0, str(RL_ROOT))
sys.modules.setdefault("pynq", types.ModuleType("pynq"))      # rpu_client imports it lazily

from Deploy.FPGA.rpu_client import RpuClient, RpuError, S, reference_forward  # noqa: E402

PI = math.pi


def wrap_pi_ref(x: float) -> float:          # pendulum_env._wrap_pi
    return ((x + PI) % (2.0 * PI)) - PI


def apply_norm_ref(spec, x: float) -> float:  # pendulum_env._apply_norm
    if spec is None:
        return x
    kind = spec[0]
    if kind == "div":
        return x / spec[1]
    if kind == "clipdiv":
        return float(np.clip(x / spec[1], -1.0, 1.0))
    if kind == "gain":
        return x * spec[1]
    if kind == "asinh":
        s, xm = spec[1], spec[2]
        return float(np.clip(np.arcsinh(x / s) / np.arcsinh(xm / s), -1.0, 1.0))
    raise ValueError(kind)


class Dims(ctypes.Structure):
    _fields_ = [("n_in", ctypes.c_int), ("h1", ctypes.c_int),
                ("h2", ctypes.c_int), ("n_out", ctypes.c_int)]


class Rng(ctypes.Structure):
    _fields_ = [("s", ctypes.c_uint32 * 4), ("has_spare", ctypes.c_int),
                ("spare", ctypes.c_float)]


def build(tmp: Path) -> ctypes.CDLL:
    lib = tmp / ("policy_core.dll" if os.name == "nt" else "policy_core.so")
    cmd = ["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
           "-o", str(lib), str(HERE / "policy_core.c"), "-lm"]
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(str(lib))


def test_core(lib: ctypes.CDLL) -> None:
    F = ctypes.POINTER(ctypes.c_float)
    lib.mlp_forward.argtypes = [F, ctypes.POINTER(Dims), F, F, F, F]
    lib.net_words.argtypes = [ctypes.POINTER(Dims)]
    lib.net_words.restype = ctypes.c_uint32
    lib.wrap_pi.argtypes = [ctypes.c_float]
    lib.wrap_pi.restype = ctypes.c_float
    lib.apply_norm.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.apply_norm.restype = ctypes.c_float
    lib.rng_seed.argtypes = [ctypes.POINTER(Rng), ctypes.c_uint32]
    lib.rng_normal.argtypes = [ctypes.POINTER(Rng)]
    lib.rng_normal.restype = ctypes.c_float
    lib.sac_action.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                               ctypes.c_int, ctypes.POINTER(Rng), F, F]
    lib.sac_action.restype = ctypes.c_float

    rng = np.random.default_rng(0)
    for n_in, h1, h2 in ((24, 256, 256), (6, 128, 128), (16, 64, 32), (7, 5, 3)):
        d = Dims(n_in, h1, h2, 2)
        n = h1 * n_in + h1 + h2 * h1 + h2 + 2 * h2 + 2
        assert lib.net_words(ctypes.byref(d)) == n
        params = (rng.standard_normal(n) * (1.0 / math.sqrt(max(n_in, h1)))).astype(np.float32)
        h1b = np.zeros(h1, np.float32)
        h2b = np.zeros(h2, np.float32)
        out = np.zeros(2, np.float32)
        worst = 0.0
        for _ in range(50):
            x = (rng.standard_normal(n_in) * 2).astype(np.float32)
            lib.mlp_forward(params.ctypes.data_as(F), ctypes.byref(d), x.ctypes.data_as(F),
                            h1b.ctypes.data_as(F), h2b.ctypes.data_as(F), out.ctypes.data_as(F))
            mu, ls = reference_forward(params, (n_in, h1, h2), x)
            worst = max(worst, abs(out[0] - mu[0]), abs(out[1] - ls[0]))
        assert worst < 1e-4, f"mlp_forward {n_in}x{h1}x{h2}: max error {worst}"
        print(f"  mlp_forward {n_in:>2}x{h1:>3}x{h2:>3}: max |C - numpy| = {worst:.2e}")

    worst = max(abs(lib.wrap_pi(x) - wrap_pi_ref(np.float32(x)))
                for x in np.linspace(-20, 20, 4001))
    # near +-pi the float32 result may land on the other branch (same angle)
    for x in np.linspace(-20, 20, 4001):
        c, r = lib.wrap_pi(x), wrap_pi_ref(float(np.float32(x)))
        diff = abs(c - r)
        assert diff < 1e-4 or abs(diff - 2 * PI) < 1e-4, (x, c, r)
    print(f"  wrap_pi: ok (max diff {worst:.2e}, incl. +-pi branch)")

    specs = [(S.NORM_NONE, None), (S.NORM_DIV, ("div", 4.0)),
             (S.NORM_CLIPDIV, ("clipdiv", 7.5)), (S.NORM_GAIN, ("gain", 10.0)),
             (S.NORM_ASINH, ("asinh", 1.0, 32.0)), (S.NORM_ASINH, ("asinh", 10.0, 32.0))]
    for kind, spec in specs:
        p1 = spec[1] if spec else 0.0
        p2 = math.asinh(spec[2] / spec[1]) if spec and spec[0] == "asinh" else 0.0
        for x in np.linspace(-50, 50, 1001):
            c, r = lib.apply_norm(kind, p1, p2, x), apply_norm_ref(spec, float(np.float32(x)))
            assert abs(c - r) < 1e-5 * max(1.0, abs(r)), (spec, x, c, r)
    print("  apply_norm: all OBS_NORM_PRESETS kinds match")

    r = Rng()
    lib.rng_seed(ctypes.byref(r), 12345)
    z = np.array([lib.rng_normal(ctypes.byref(r)) for _ in range(200_000)])
    assert abs(z.mean()) < 0.01 and abs(z.std() - 1) < 0.01, (z.mean(), z.std())
    assert np.isfinite(z).all()
    print(f"  rng_normal: mean {z.mean():+.4f}, std {z.std():.4f}, |z|max {np.abs(z).max():.2f}")

    eps, ls_out = ctypes.c_float(), ctypes.c_float()
    for mu, ls in ((0.3, -1.0), (-2.0, 5.0), (1.0, -30.0)):
        lib.rng_seed(ctypes.byref(r), 7)
        a = lib.sac_action(mu, ls, -20.0, 2.0, 1, ctypes.byref(r),
                           ctypes.byref(eps), ctypes.byref(ls_out))
        lsc = min(2.0, max(-20.0, ls))
        assert abs(ls_out.value - lsc) < 1e-6
        assert abs(a - math.tanh(mu + math.exp(lsc) * eps.value)) < 1e-5
        a_det = lib.sac_action(mu, ls, -20.0, 2.0, 0, ctypes.byref(r),
                               ctypes.byref(eps), ctypes.byref(ls_out))
        assert eps.value == 0.0 and abs(a_det - math.tanh(mu)) < 1e-6
    print("  sac_action: stochastic and deterministic match SB3's squashed Gaussian")


STUB_HEADERS = {
    # Mirrors names the real BSP defines, so clashes show up here too.
    "xil_types.h": "#include <stdint.h>\ntypedef uint32_t u32; typedef uint64_t u64; "
                   "typedef int16_t s16; typedef int32_t s32; typedef intptr_t INTPTR;\n",
    "xil_io.h": '#include "xil_types.h"\nstatic inline u32 Xil_In32(INTPTR a){return *(volatile u32*)a;}\n'
                "static inline void Xil_Out32(INTPTR a, u32 v){*(volatile u32*)a=v;}\n"
                '#define dsb() __asm__ __volatile__ ("" : : : "memory")\n'
                '#define dmb() __asm__ __volatile__ ("" : : : "memory")\n',
    "xil_cache.h": '#include "xil_types.h"\nvoid Xil_DCacheInvalidateRange(INTPTR a, INTPTR l);\n',
    "xil_mpu.h": '#include "xil_types.h"\nu32 Xil_GetNextMPURegion(void);\n'
                 "u32 Xil_SetMPURegionByRegNum(u32 n, INTPTR a, u64 s, u32 attr);\n"
                 "u32 Xil_DisableMPURegionByRegNum(u32 n);\n",
    "xreg_cortexr5.h": "#define NORM_NSHARED_NCACHE 0x8u\n#define PRIV_RW_USER_RW 0x300u\n",
    "xparameters.h": "#define XPAR_CPU_CORE_CLOCK_FREQ_HZ 499999908\n",
}


def test_main_syntax(tmp: Path) -> None:
    inc = tmp / "stub_bsp"
    inc.mkdir()
    for name, text in STUB_HEADERS.items():
        (inc / name).write_text(text)
    # -fsyntax-only: parses and type-checks; the ARM asm strings are not assembled.
    subprocess.run(["gcc", "-fsyntax-only", "-std=gnu99", "-Wall", "-Wextra", "-Werror",
                    "-Wno-int-to-pointer-cast", "-Wno-pointer-to-int-cast",
                    "-I", str(inc), "-I", str(HERE), str(HERE / "rpu_main.c")], check=True)
    print("  rpu_main.c: compiles against stub BSP headers (-Wall -Wextra -Werror)")


def test_layout() -> None:
    assert S.SHM_UNCACHED_OFF % S.SHM_ALIGN == 0, "aligned base must keep the MPU region aligned"
    # mailbox in the FSBL-only part of OCM (below ARM Trusted Firmware at 0xFFFEA000)
    assert 0xFFFC0000 <= S.MBOX_ADDR and S.MBOX_ADDR + S.MBOX_MPU_BYTES <= 0xFFFEA000
    assert S.MBOX_ADDR % S.MBOX_MPU_BYTES == 0 and S.MBOX_BYTES <= S.MBOX_MPU_BYTES
    mb = [v for k, v in vars(S).items() if k.startswith("MB_")]
    assert max(mb) * 4 < S.MBOX_BYTES and len(mb) == len(set(mb)), "mailbox offsets clash"
    log_end = S.SHM_LOG_OFF + S.LOG_DEPTH * S.LOG_WORDS * 4
    assert S.SHM_UNCACHED_OFF <= S.SHM_CTRL_OFF and log_end <= S.SHM_UNCACHED_OFF + S.SHM_UNCACHED_BYTES
    assert S.SHM_UNCACHED_OFF % S.SHM_UNCACHED_BYTES == 0, "MPU region must be size-aligned"
    assert S.SHM_CTRL_OFF + S.CTRL_WORDS * 4 <= S.SHM_LOG_OFF
    assert S.R_OBS + S.MAX_OBS <= S.LOG_WORDS
    assert S.SHM_BANK1_OFF >= S.SHM_BANK0_OFF + S.SHM_BANK_BYTES
    assert S.SHM_BANK1_OFF + S.SHM_BANK_BYTES <= S.SHM_UNCACHED_OFF
    n256 = 256 * 24 + 256 + 256 * 256 + 256 + 2 * 256 + 2
    assert n256 <= S.SHM_BANK_WORDS
    ctrl = [v for k, v in vars(S).items() if k.startswith("C_")]
    assert max(ctrl) < S.CTRL_WORDS and len(ctrl) == len(set(ctrl)), "control-block offsets clash"
    rec = [v for k, v in vars(S).items() if k.startswith("R_")]
    assert len(rec) == len(set(rec)), "record offsets clash"
    print(f"  layout: log ends at +0x{log_end:x}, 256x256 net uses "
          f"{100 * n256 / S.SHM_BANK_WORDS:.0f}% of a bank, {len(ctrl)} control words")


class FakeFirmware:
    """Python model of rpu_main.c's main loop: attach / detach / lease /
    weight-bank ack. Window and mailbox are plain numpy arrays."""

    def __init__(self, window: np.ndarray):
        self.mb = np.zeros(S.MBOX_BYTES // 4, np.uint32)
        self.win = window
        self.shm = 0
        self.stop = threading.Event()
        self.mb[S.MB_FW_VERSION] = S.RPU_FW_VERSION
        self.mb[S.MB_MAGIC] = S.MBOX_MAGIC
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def ctrl(self, i, v=None):
        k = S.SHM_CTRL_OFF // 4 + i
        if v is None:
            return int(self.win[k])
        self.win[k] = v

    def run(self):
        mb, seen, lease_seen, lease_t, timeout = self.mb, 0, 0, 0.0, 0.0
        while not self.stop.is_set():
            mb[S.MB_HEARTBEAT] += 1
            if mb[S.MB_ATTACH_SEQ] != seen:
                seen = int(mb[S.MB_ATTACH_SEQ])
                base, err = int(mb[S.MB_SHM_BASE]), S.MBE_NONE
                if base == 0:
                    self.shm = 0
                elif base % S.SHM_ALIGN or base + S.SHM_SIZE > S.SHM_R5_LIMIT:
                    err = S.MBE_BAD_BASE
                else:
                    self.shm = base
                    for i, v in ((S.C_CPU_HZ, 500_000_000), (S.C_LOG_DEPTH, S.LOG_DEPTH),
                                 (S.C_LOG_WORDS, S.LOG_WORDS), (S.C_FW_VERSION, S.RPU_FW_VERSION),
                                 (S.C_MAGIC, S.RPU_MAGIC)):
                        self.ctrl(i, v)
                    timeout = min(int(mb[S.MB_LEASE_MS]), 8000) / 1000.0
                    lease_seen, lease_t = int(mb[S.MB_LEASE]), time.monotonic()
                mb[S.MB_ERROR] = err
                mb[S.MB_ATTACHED_BASE] = self.shm
                mb[S.MB_STATE] = S.MBS_ATTACHED if self.shm else S.MBS_WAITING
                mb[S.MB_ACK_SEQ] = seen
            elif self.shm:
                if mb[S.MB_LEASE] != lease_seen:
                    lease_seen, lease_t = int(mb[S.MB_LEASE]), time.monotonic()
                elif timeout and time.monotonic() - lease_t > timeout:
                    self.shm = 0
                    mb[S.MB_ERROR] = S.MBE_LEASE_LOST
                    mb[S.MB_ATTACHED_BASE] = 0
                    mb[S.MB_STATE] = S.MBS_WAITING
                    continue
                req = self.ctrl(S.C_WB_REQ_VERSION)
                if req and req != self.ctrl(S.C_WB_ACTIVE_VERSION):
                    self.ctrl(S.C_WB_ACTIVE_BANK, self.ctrl(S.C_WB_REQ_BANK))
                    self.ctrl(S.C_WB_ACTIVE_VERSION, req)
            time.sleep(0.0002)


def test_attach_protocol() -> None:
    ns = types.SimpleNamespace
    win = np.full(S.SHM_SIZE // 4, 0xDEADBEEF, np.uint32)       # garbage until attached
    fw = FakeFirmware(win)
    base = 0x10000000
    try:
        cl = RpuClient(None, mem=ns(array=win), mbox=ns(array=fw.mb), shm_base=base,
                       lease_ms=200)
        assert cl.attached() and fw.shm == base and cl.state() == S.ST_IDLE
        assert win[S.SHM_BANK0_OFF // 4] == 0, "host must zero the window before attaching"
        assert cl.cpu_hz == 5e8
        assert cl.load_weights(np.ones(10, np.float32)) == 1
        assert cl.load_weights(np.ones(10, np.float32)) == 2     # waits for the bank ack
        time.sleep(0.5)                                          # > lease: thread renews it
        assert cl.attached(), "lease thread did not keep the window"
        cl.close()
        assert fw.shm == 0 and fw.mb[S.MB_ATTACHED_BASE] == 0 and not cl.attached()
        cl.close()                                               # idempotent

        cl = RpuClient(None, mem=ns(array=win), mbox=ns(array=fw.mb), shm_base=base,
                       lease_ms=200)
        cl._stop_lease()                                         # as if the process died
        time.sleep(0.5)
        assert fw.shm == 0 and fw.mb[S.MB_ERROR] == S.MBE_LEASE_LOST
        assert not cl.attached() and cl.state() == S.ST_ERROR and "HOST_GONE" in cl.error()
        assert cl.abort() is True
        try:
            cl.start_episode(1, max_steps=1, explore=False, seed=1, ctrl_level=1, ctrl_stb=2)
            raise AssertionError("start_episode on a dropped window")
        except RpuError:
            pass
        cl.close()

        try:
            RpuClient(None, mem=ns(array=win), mbox=ns(array=fw.mb), shm_base=base + 0x1000)
            raise AssertionError("misaligned window accepted")
        except RpuError as exc:
            assert "MBE_BAD_BASE" in str(exc)
    finally:
        fw.stop.set()
    print("  attach / lease / detach protocol: ok (incl. lease loss, bad base)")


def main() -> int:
    # ignore_cleanup_errors: Windows keeps the loaded DLL locked until exit
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        tmp = Path(t)
        print("policy_core.c (host build):")
        test_core(build(tmp))
        print("rpu_main.c:")
        test_main_syntax(tmp)
    print("rpu_shared.h:")
    test_layout()
    print("rpu_client.py:")
    test_attach_protocol()
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

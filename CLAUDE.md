# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope: work only in `reinforcement_learning/`

All active work lives in `reinforcement_learning/`. It holds the simulation,
training, deployment, and system-identification (sysid) code for a learned
swing-up + balance controller for a rotary inverted (Furuta) pendulum.

Do **not** read, edit, or take guidance from the other top-level folders
(`RotaryInvertedPendulum-arduino/`, `RotaryInvertedPendulum-julia/`,
`RotaryInvertedPendulum-python/`, `RotaryInvertedPendulum-FPGA/`, `esp32/`,
`policy/`, `docs/`, `New folder/`, etc.). They are legacy or describe older
hardware (AS5600 encoder, NEMA17, Arduino Nano). Docstrings inside
`reinforcement_learning/` sometimes still point at those old paths
(e.g. `RotaryInvertedPendulum-arduino/LowLevelServer/...`,
`src/rl/...`) — treat those references as stale; the live equivalents are
in `reinforcement_learning/`.

The only file outside the folder that the code depends on is the URDF in
`urdf/robot.urdf` (see "Where physical parameters live").

## Hardware

- **Controller**: ESP32 (primary). An FPGA (PYNQ) target exists as an alternative.
- **Encoders**: AS5047P magnetic encoders (14-bit, 16384 CPR, SPI) on **both**
  the pendulum shaft and the motor shaft.
- **Motor**: SF2421 stepper.
- **Driver**: TMC2209 (STEP/DIR/EN), driven by `FastAccelStepper`.
  200 full steps × 16 microsteps = 3200 steps/rev.
- **Serial link**: 1,000,000 baud (ESP32 `LowLevelServer.ino` and the Python clients).
- **Motor travel**: hard limit ±135°, firmware rail brake beyond ±125°.

## Folder layout

```
reinforcement_learning/
├── Simulation/
│   ├── pendulum_env.py        # Gymnasium + MuJoCo env, domain randomisation, actuator model
│   └── pendulum_geometry.py   # parses urdf/robot.urdf → pendulum + arm mass/COM/inertia
├── Training/
│   ├── train_sac.py           # SB3 SAC trainer → Training/runs/<run_name>/
│   ├── train_sac_qat.py       # quantization-aware variant (adds quant flags on top of train_sac)
│   ├── qat.py                 # obs fixed-point quantizer, fake-quant layers, QATSACPolicy
│   ├── sweep_quant.py         # sweeps quantization bit widths, tabulates performance
│   ├── reward.py              # single source of truth for the reward (sim AND real)
│   └── symmetry.py            # mirror symmetry (augmentation + asymmetry metric)
├── Deploy/
│   ├── ESP32/
│   │   ├── LowLevelServer.ino # tethered firmware: binary serial protocol, accel-mode motor
│   │   ├── RLControl.ino      # standalone on-device policy (MLP weights in policy_weights*.h)
│   │   ├── StepperUtils.h     # step ↔ angle helpers
│   │   ├── lowlevel_client.py # Python client for LowLevelServer
│   │   ├── real_env.py        # Gymnasium env backed by the real rig (fine-tuning)
│   │   └── client.jl          # Julia client (legacy)
│   └── FPGA/
│       ├── FPGA_lowlevel_client.py  # PYNQ client mirroring the C low-level server
│       ├── FPGA_env.py              # real-rig env for the FPGA target
│       └── *.xsa / *.xclbin         # bitstreams / hardware handoffs
├── finetune_async.py          # threaded SAC fine-tuning on the real rig (strict control rate)
├── run_config.py              # writes/validates config.json per run (train ↔ finetune ↔ deploy)
├── sysid_wizard.py            # free-swing sysid: collect → fit → validate → sysid_params.json
├── sysid_core.py              # sysid math (free-swing fit, friction derivation)
├── sysid_accel.py             # accel-mode step/chirp response, real vs sim replay
├── accel_lag_moving_probe.py  # measures firmware accel-command lag from a moving baseline
├── sim_vs_real.py             # replays a real action log in sim and compares trajectories
├── analyze_deploy.py          # balance / chatter metrics from a deploy log
├── sysid_params.json          # measured per-rig friction (written by sysid_wizard)
├── sysid_runs/                # raw sysid recordings (.npz, plots, metadata)
└── accel_lag_moving_runs/     # lag-probe recordings
```

Scripts select the ESP32 or FPGA client at import time
(`Deploy.ESP32.lowlevel_client` vs `Deploy.FPGA.FPGA_lowlevel_client`).

Imports are mixed: some modules use `from Simulation...` / `from Training...`
(expects `reinforcement_learning/` on `sys.path`), others use
`from reinforcement_learning.Simulation...` (expects the repo root). Check
the imports of the script you run and launch it from the matching directory.

## Workflow

1. **Sysid** on the rig → `sysid_wizard.py` (friction from free swings) and
   `sysid_accel.py` / `accel_lag_moving_probe.py` (motor response and lag).
   Results feed `sysid_params.json` and the DR ranges in `pendulum_env.py`.
2. **Train in sim** → `Training/train_sac.py` (or `train_sac_qat.py` for a
   quantized policy targeting the FPGA/ESP32).
3. **Fine-tune on the rig** → `finetune_async.py` using `Deploy/ESP32/real_env.py`.
4. **Deploy** → tethered via `LowLevelServer.ino` + the Python client, or
   standalone via `RLControl.ino` with exported weights.
5. **Analyse** → `analyze_deploy.py`, `sim_vs_real.py`.

## Invariants that must hold

- **Reward**: only in `Training/reward.py`. Both `pendulum_env.py` and
  `real_env.py` call `compute_reward()`. Never duplicate the reward math.
- **Run config**: `run_config.py` records action mode, control rate, action
  scaling, obs layout, and reward extras in each run's `config.json`.
  Fine-tune and deploy entrypoints call `check_config()` and abort on mismatch.
  A checkpoint run with the wrong flags fails silently on the rig, so don't
  bypass this with `--ignore-config-mismatch` unless there's a reason.
- **Control rate**: training, fine-tuning, and deployment must use the same
  `--control-freq` (training default 50 Hz; `RLControl.ino` runs at 50 Hz).
  Note `finetune_async.py` defaults differ — pass the rate explicitly.
- **Sim ↔ firmware actuator chain**: `pendulum_env.py` models the firmware
  (action boxcar, transport delay, accel clip, ±125° rail brake, velocity cap,
  8 ms finite-difference velocity window, encoder/step quantisation). If you
  change `LowLevelServer.ino` or `RLControl.ino`, update the sim model to match
  (and vice versa).
- **Sign convention**: `LowLevelServer` flips motor and pendulum signs on
  output; `real_env.py` un-flips on read so observations match the sim.
- **Observation frame**: `[motor_pos, sin θ, cos θ, motor_vel, pen_vel, prev_action]`,
  stacked `obs_history_len` frames, θ = 0 upright. `symmetry.py` relies on this layout.

## Where physical parameters live

One source of truth per parameter class; changing a number anywhere else is a bug.

- **Pendulum and arm geometry** (mass, COM, inertia, arm length): authored in
  Onshape → exported to `urdf/robot.urdf` → parsed by
  `Simulation/pendulum_geometry.py` → consumed by `pendulum_env.py` and
  `sysid_core.py`.
- **Per-rig friction** (viscous + Coulomb): measured by `sysid_wizard.py`,
  stored in `sysid_params.json`, loaded by `pendulum_env.py`.
- **Hardware / firmware constants** (encoder CPR, microsteps, motor limits,
  max velocity/accel, DR ranges): module constants at the top of
  `Simulation/pendulum_env.py`, with the ESP32 firmware
  (`Deploy/ESP32/LowLevelServer.ino`) as the upstream source for motor-side values.

## Serial protocol (`Deploy/ESP32/LowLevelServer.ino`)

Single-byte commands at 1,000,000 baud:

| Byte | Command | Payload |
|------|---------|---------|
| `0x01` | READY | board echoes `0x01` |
| `0x02` | GET_STATE | returns 20 bytes: `uint32 time_us`, `float32 motor_pos`, `pendulum_pos`, `motor_vel`, `pendulum_vel` (rad, rad/s) |
| `0x03` | SET_ACCEL | `float32` angular acceleration in rad/s² |
| `0x04` | ENGAGE_MOTOR | – |
| `0x05` | DISENGAGE_MOTOR | – |
| `0x06` | TARE_PENDULUM | re-zero pendulum angle |

The firmware samples encoders every 2 ms and computes velocities by finite
difference over a 5-sample window.

## Dependencies

Python: `stable-baselines3`, `gymnasium`, `mujoco`, `torch`, `numpy`,
`scipy`, `pyserial`; `pynq` for the FPGA target.
Arduino (ESP32): `FastAccelStepper`, `AS5047P`.

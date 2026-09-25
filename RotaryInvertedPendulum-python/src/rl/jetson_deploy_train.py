from __future__ import annotations

import math
import time
import threading
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import Jetson.GPIO as GPIO
import spidev
from JetsonPWM import JetsonPWM
from stable_baselines3 import PPO

# ==========================================
# Constants
# ==========================================
TARGET_HZ = 40.0
PERIOD = 1.0 / TARGET_HZ
CONTROL_PERIOD = PERIOD
STEP_UPDATE_HZ = 100.0
STEP_UPDATE_PERIOD = 1.0 / STEP_UPDATE_HZ

STEPS_PER_REV = 3200
ARM_RAD_PER_STEP = (2.0 * math.pi) / STEPS_PER_REV
ARM_SAFE_LIMIT_RAD = math.radians(125.0)
ARM_MAX_SAFE_STEPS = int(ARM_SAFE_LIMIT_RAD / ARM_RAD_PER_STEP)
MAX_ACCEL_RAD_S2 = 150.0
MAX_VELOCITY_RAD_S = 5.0
ARM_MAX_DELTA_STEPS = (MAX_VELOCITY_RAD_S / ARM_RAD_PER_STEP) * PERIOD
PWM_MIN_FREQ_HZ = 40.0

PEND_ENCODER_RESOLUTION = 16384
PEND_LSB_RAD = (2.0 * math.pi) / PEND_ENCODER_RESOLUTION
MAX_PENDULUM_VEL_RAD_S = 30.0
PEND_MAX_DELTA_TICKS = (MAX_PENDULUM_VEL_RAD_S / PEND_LSB_RAD) * PERIOD

DIR_PIN = 16
EN_PIN = 22

SB3_ZIP_PATH = "./Saved_runs/2509-2.zip"

# ==========================================
# Shared Hardware State (Used by Background Thread)
# ==========================================
stop_event = threading.Event()
state_lock = threading.Lock()

shared_accel_cmd = 0.0
arm_current_steps = 0
motor_vel_rad_s = 0.0
motor_target_rad = 0.0

# SPI Setup
spi_pendulum = spidev.SpiDev()
spi_pendulum.open(0, 0)
spi_pendulum.max_speed_hz = 100_000
spi_pendulum.mode = 1
spi_pendulum.bits_per_word = 8

def read_raw_ticks(spi_device) -> int:
    spi_device.xfer2([0xFF, 0xFF])
    data = spi_device.xfer2([0xC0, 0x00])
    raw = (data[0] << 8) | data[1]
    return raw & 0x3FFF

def process_encoder(current_ticks: int, prev_ticks: int):
    norm_angle = (current_ticks - 5000) / 8192
    norm_angle = (norm_angle + 1.0) % 2.0 - 1.0
    angle_rad = norm_angle * math.pi

    delta = current_ticks - prev_ticks
    if delta > 8192: delta -= 16384
    elif delta < -8192: delta += 16384
    
    norm_vel = delta / float(PEND_MAX_DELTA_TICKS)
    norm_vel = max(-1.0, min(1.0, norm_vel))
    return angle_rad, norm_vel

# ==========================================
# 100Hz Background Stepper Thread
# ==========================================
def step_update_loop():
    global arm_current_steps, motor_vel_rad_s, motor_target_rad

    pwm_step = JetsonPWM(chip=0, channel=0)
    pwm_step.start(initial_freq_hz=PWM_MIN_FREQ_HZ)

    print("[Step Thread] Starting 100 Hz kinematics/PWM loop.")
    next_tick = time.perf_counter()

    try:
        while not stop_event.is_set():
            with state_lock:
                accel_cmd = shared_accel_cmd
                current_steps = arm_current_steps

            dt = STEP_UPDATE_PERIOD

            motor_vel_rad_s += accel_cmd * dt
            motor_vel_rad_s = max(-MAX_VELOCITY_RAD_S, min(MAX_VELOCITY_RAD_S, motor_vel_rad_s))

            if motor_target_rad >= ARM_SAFE_LIMIT_RAD and motor_vel_rad_s > 0.0:
                motor_vel_rad_s = 0.0
                motor_target_rad = ARM_SAFE_LIMIT_RAD
            elif motor_target_rad <= -ARM_SAFE_LIMIT_RAD and motor_vel_rad_s < 0.0:
                motor_vel_rad_s = 0.0
                motor_target_rad = -ARM_SAFE_LIMIT_RAD

            motor_target_rad += motor_vel_rad_s * dt
            motor_target_rad = max(-ARM_SAFE_LIMIT_RAD, min(ARM_SAFE_LIMIT_RAD, motor_target_rad))

            target_pos_steps = int(round(motor_target_rad / ARM_RAD_PER_STEP))
            target_pos_steps = max(-ARM_MAX_SAFE_STEPS, min(ARM_MAX_SAFE_STEPS, target_pos_steps))

            step_error = target_pos_steps - current_steps

            if step_error != 0:
                is_forward = step_error < 0
                GPIO.output(DIR_PIN, GPIO.HIGH if is_forward else GPIO.LOW)

                max_steps_this_update = max(1, int(math.floor(MAX_VELOCITY_RAD_S * dt / ARM_RAD_PER_STEP)))
                actual_steps = max(-max_steps_this_update, min(max_steps_this_update, step_error))

                new_current_steps = current_steps + actual_steps
                new_current_steps = max(-ARM_MAX_SAFE_STEPS, min(ARM_MAX_SAFE_STEPS, new_current_steps))

                freq = abs(actual_steps) / dt
                freq = max(PWM_MIN_FREQ_HZ, freq)

                pwm_step.change_frequency(freq)

                with state_lock:
                    arm_current_steps = new_current_steps
            else:
                pwm_step.pause()
                if abs(motor_target_rad - current_steps * ARM_RAD_PER_STEP) < ARM_RAD_PER_STEP * 0.5:
                    motor_target_rad = current_steps * ARM_RAD_PER_STEP

            next_tick += STEP_UPDATE_PERIOD
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    finally:
        pwm_step.stop()
        print("[Step Thread] Stopped.")

# ==========================================
# Custom Gymnasium Environment for Real Hardware
# ==========================================
class RealPendulumEnv(gym.Env):
    """Custom Environment that interfaces directly with physical hardware."""
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self.prev_arm_steps = 0
        self.prev_pend_ticks = -read_raw_ticks(spi_pendulum)
        self.prev_action = 0.0
        self.step_count = 0
        self.next_tick = time.perf_counter()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        global shared_accel_cmd, arm_current_steps

        print("\n[ENV] Episode finished. Homing arm smoothly to center...")
        
        # ==========================================
        # 1. Homing Sequence (Runs locally in reset)
        # ==========================================
        with state_lock:
            prev_steps = arm_current_steps
            
        last_time = time.perf_counter()
        
        while True:
            # Read current state
            with state_lock:
                curr_steps = arm_current_steps
                
            # Exit condition: We are at center (within 2 steps) AND not moving
            if abs(curr_steps) <= 2 and curr_steps == prev_steps:
                break
                
            # Calculate accurate time step for velocity estimation
            curr_time = time.perf_counter()
            dt = curr_time - last_time
            if dt <= 0.001: 
                dt = 0.025
                
            # Calculate position and velocity
            current_pos_rad = curr_steps * ARM_RAD_PER_STEP
            current_vel_rad_s = ((curr_steps - prev_steps) * ARM_RAD_PER_STEP) / dt
            
            # --- PD Controller for Smooth Acceleration ---
            # Kp (Proportional) acts like a spring pulling the arm to 0.0.
            # Kd (Derivative) acts like friction, slowing it down so it doesn't overshoot.
            Kp = 30.0  
            Kd = 10.0  
            
            accel_cmd = (0.0 - current_pos_rad) * Kp - (current_vel_rad_s * Kd)
            
            # Cap the homing acceleration to a very gentle 20 rad/s^2 (max is 150)
            accel_cmd = max(-20.0, min(20.0, accel_cmd))
            
            # Send the gentle acceleration command to the background thread
            with state_lock:
                shared_accel_cmd = accel_cmd
                
            # Update trackers and sleep to run at roughly 40Hz
            prev_steps = curr_steps
            last_time = curr_time
            time.sleep(0.025)

        # ------------------------------------------
        # 2. Stop completely & let pendulum settle
        # ------------------------------------------
        with state_lock:
            shared_accel_cmd = 0.0

        print("[ENV] Arm homed! Waiting 3 seconds for pendulum to stop swinging...")
        time.sleep(3.0)

        # ------------------------------------------
        # 3. Standard Environment Reset Tracker Update
        # ------------------------------------------
        self.prev_action = 0.0
        self.step_count = 0
        with state_lock:
            self.prev_arm_steps = arm_current_steps 
            
        self.prev_pend_ticks = -read_raw_ticks(spi_pendulum)
        
        # Get initial observation for the new episode
        obs = self._get_obs(0.0)
        self.next_tick = time.perf_counter() + CONTROL_PERIOD
        
        return obs, {}

    def step(self, action):
        global shared_accel_cmd
        self.step_count += 1

        # 1. Enforce 40Hz timing
        sleep_time = self.next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.next_tick = time.perf_counter() + CONTROL_PERIOD

        # 2. Apply Action to physical hardware
        action_val = float(action[0])
        action_val = max(-1.0, min(1.0, action_val))
        
        accel_cmd = action_val * MAX_ACCEL_RAD_S2
        with state_lock:
            shared_accel_cmd = accel_cmd

        # 3. Read physical states
        obs = self._get_obs(action_val)
        arm_pos, cos_p, sin_p, arm_vel, pend_vel, _ = obs
        
        pend_rad = math.atan2(sin_p, cos_p) 

        # 4. Calculate Reward 
        # Max reward (+1) when upright, min reward (-1) when hanging straight down.
        reward = math.cos(pend_rad) - 0.05 * abs(arm_pos) - 0.1 * abs(action_val)

        # 5. Determine Termination (Failure conditions)
        terminated = False
        truncated = False
        
        # ONLY terminate if the arm hits the physical safety limit
        with state_lock:
            current_steps = arm_current_steps
        if abs(current_steps) >= ARM_MAX_SAFE_STEPS - 50:
            terminated = True
            reward -= 10.0 # Heavy penalty for hitting the edge
            
        # Truncate episode exactly when PPO buffer fills (2048 steps = 51.2 seconds)
        if self.step_count >= 2048:
            truncated = True

        self.prev_action = action_val
        
        # New Gym API expects 5 variables: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, {}

    def _get_obs(self, current_action):
        with state_lock:
            observed_arm_steps = arm_current_steps

        # Arm calculations
        arm_delta_steps = observed_arm_steps - self.prev_arm_steps
        self.prev_arm_steps = observed_arm_steps
        norm_arm_pos = max(-1.0, min(1.0, observed_arm_steps / float(STEPS_PER_REV / 2)))
        norm_arm_vel = max(-1.0, min(1.0, arm_delta_steps / float(ARM_MAX_DELTA_STEPS)))

        # Pendulum calculations
        pend_ticks = -read_raw_ticks(spi_pendulum)
        pend_rad, norm_pend_vel = process_encoder(pend_ticks, self.prev_pend_ticks)
        self.prev_pend_ticks = pend_ticks

        obs = np.array([
            norm_arm_pos,
            math.cos(pend_rad),
            math.sin(pend_rad),
            norm_arm_vel,
            norm_pend_vel,
            current_action
        ], dtype=np.float32)
        
        return obs

# ==========================================
# Main Training Entry Point
# ==========================================
if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(EN_PIN, GPIO.OUT, initial=GPIO.LOW)

    # 1. Start Background Hardware Thread
    step_thread = threading.Thread(target=step_update_loop, daemon=True, name="StepUpdate100Hz")
    step_thread.start()

    try:
        # 2. Instantiate Environment
        env = RealPendulumEnv()

        # 3. Load existing pre-trained model
        print(f"Loading base model from {SB3_ZIP_PATH}...")
        model = PPO.load(SB3_ZIP_PATH, env=env, device="cpu")
        
        # Lower learning rate slightly so it doesn't instantly forget its balancing behavior 
        # while thrashing around trying to figure out swing-up.
        model.learning_rate = 1e-4

        # 4. Start real-world training!
        print("\nStarting Real-World Training...")
        
        # 51,200 timesteps = 25 updates (roughly 21 minutes of real-world time)
        # You can safely increase this to 102400 (50 cycles) if you have time.
        model.learn(total_timesteps=51200, reset_num_timesteps=False)

        # 5. Save the newly fine-tuned model
        save_path = "./Saved_runs/real_world_swingup.zip"
        model.save(save_path)
        print(f"Real-world training complete. Saved to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Stopping hardware...")
    
    finally:
        # Clean shutdown
        stop_event.set()
        step_thread.join(timeout=2.0)
        GPIO.setup(EN_PIN, GPIO.HIGH)  # Disable stepper driver
        GPIO.cleanup()
        spi_pendulum.close()
        print("Shutdown clean.")
import os
import time

class JetsonPWM:
    def __init__(self, chip=0, channel=0):
        self.chip_path = f"/sys/class/pwm/pwmchip{chip}"
        self.pwm_path = f"{self.chip_path}/pwm{channel}"
        self.channel = channel
        
        self._export_pwm()
        
        # Open raw file descriptors (O_RDWR / O_WRONLY) without Python buffering
        self._period_fd = os.open(f"{self.pwm_path}/period", os.O_RDWR)
        self._duty_fd = os.open(f"{self.pwm_path}/duty_cycle", os.O_RDWR)
        self._enable_fd = os.open(f"{self.pwm_path}/enable", os.O_WRONLY)

        # Read the current hardware period
        os.lseek(self._period_fd, 0, os.SEEK_SET)
        raw = os.read(self._period_fd, 32).decode().strip()
        self.current_period_ns = int(raw) if raw else 10_000_000

        self.is_enabled = False

    def _export_pwm(self):
        if not os.path.exists(self.pwm_path):
            with open(f"{self.chip_path}/export", "w") as f:
                f.write(str(self.channel))
            time.sleep(0.1)

    def _write_fd(self, fd, val_str):
        """Rewinds, truncates, and directly writes to sysfs without buffer residue."""
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{val_str}\n".encode())

    def start(self, initial_freq_hz=100):
        """Initializes duty and period safely with output disabled, then enables."""
        period_ns = int(1_000_000_000 / initial_freq_hz)
        duty_ns = (period_ns // 2) - 100  # -100ns margin prevents hardware rounding race

        # Disable first to set initial baseline cleanly
        self._write_fd(self._enable_fd, "0")
        self._write_fd(self._period_fd, str(period_ns))
        self._write_fd(self._duty_fd, str(duty_ns))

        # Re-read actual period assigned by hardware clock ticks
        os.lseek(self._period_fd, 0, os.SEEK_SET)
        actual = os.read(self._period_fd, 32).decode().strip()
        self.current_period_ns = int(actual)

        # Enable
        self._write_fd(self._enable_fd, "1")
        self.is_enabled = True

    def change_frequency(self, freq_hz):
        """
        Dynamically updates the frequency without disabling output.
        Enforces duty_cycle <= period ordering using low-level OS writes.
        """
        if not self.is_enabled:
            self.start(freq_hz)
            return

        new_period_ns = int(1_000_000_000 / freq_hz)
        # 100ns safety buffer avoids kernel rejection if period snaps down slightly
        new_duty_ns = max(1, (new_period_ns // 2) - 100)

        if new_period_ns < self.current_period_ns:
            # Frequency increasing: period shrinks -> duty_cycle MUST drop first
            self._write_fd(self._duty_fd, str(new_duty_ns))
            self._write_fd(self._period_fd, str(new_period_ns))
        else:
            # Frequency decreasing: period expands -> period MUST expand first
            self._write_fd(self._period_fd, str(new_period_ns))
            self._write_fd(self._duty_fd, str(new_duty_ns))

        self.current_period_ns = new_period_ns

    def stop(self):
        """Disables channel and unexports."""
        try:
            self._write_fd(self._enable_fd, "0")
        except Exception:
            pass

        try:
            os.close(self._period_fd)
            os.close(self._duty_fd)
            os.close(self._enable_fd)
        except Exception:
            pass

        try:
            with open(f"{self.chip_path}/unexport", "w") as f:
                f.write(str(self.channel))
        except Exception:
            pass
        self.is_enabled = False
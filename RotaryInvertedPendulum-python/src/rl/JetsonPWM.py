import os
import time

class JetsonPWM:
    def __init__(self, chip=0, channel=0):
        self.chip_path = f"/sys/class/pwm/pwmchip{chip}"
        self.pwm_path = f"{self.chip_path}/pwm{channel}"
        self.channel = channel
        
        self._export_pwm()
        
        # Keep file descriptors open for low latency updates
        self._period_fd = open(f"{self.pwm_path}/period", "r+")
        self._duty_fd = open(f"{self.pwm_path}/duty_cycle", "r+")
        self._enable_fd = open(f"{self.pwm_path}/enable", "w")

        # Read the current period from sysfs or initialize default
        try:
            self._period_fd.seek(0)
            raw = self._period_fd.read().strip()
            self.current_period_ns = int(raw) if raw else 10_000_000
        except (ValueError, IOError):
            self.current_period_ns = 10_000_000

        self.is_enabled = False

    def _export_pwm(self):
        if not os.path.exists(self.pwm_path):
            with open(f"{self.chip_path}/export", "w") as f:
                f.write(str(self.channel))
            time.sleep(0.1)

    def start(self, initial_freq_hz=100):
        """Configures initial frequency, sets 50% duty cycle, and enables output."""
        period_ns = int(1_000_000_000 / initial_freq_hz)
        duty_ns = period_ns // 2

        # Safe initial write while disabled
        self._enable_fd.seek(0)
        self._enable_fd.write("0\n")
        self._enable_fd.flush()

        self._period_fd.seek(0)
        self._period_fd.write(f"{period_ns}\n")
        self._period_fd.flush()

        self._duty_fd.seek(0)
        self._duty_fd.write(f"{duty_ns}\n")
        self._duty_fd.flush()

        self.current_period_ns = period_ns

        # Turn ON
        self._enable_fd.seek(0)
        self._enable_fd.write("1\n")
        self._enable_fd.flush()
        self.is_enabled = True

    def change_frequency(self, freq_hz):
        """
        Dynamically updates the output frequency while remaining enabled.
        Enforces: duty_cycle <= period at all times to prevent EINVAL.
        """
        if not self.is_enabled:
            self.start(freq_hz)
            return

        new_period_ns = int(1_000_000_000 / freq_hz)
        new_duty_ns = new_period_ns // 2

        if new_period_ns < self.current_period_ns:
            # Frequency increasing: shrink duty_cycle first
            self._duty_fd.seek(0)
            self._duty_fd.write(f"{new_duty_ns}\n")
            self._duty_fd.flush()

            self._period_fd.seek(0)
            self._period_fd.write(f"{new_period_ns}\n")
            self._period_fd.flush()
        else:
            # Frequency decreasing: expand period first
            self._period_fd.seek(0)
            self._period_fd.write(f"{new_period_ns}\n")
            self._period_fd.flush()

            self._duty_fd.seek(0)
            self._duty_fd.write(f"{new_duty_ns}\n")
            self._duty_fd.flush()

        self.current_period_ns = new_period_ns

    def stop(self):
        """Disables output and releases file descriptors."""
        try:
            self._enable_fd.seek(0)
            self._enable_fd.write("0\n")
            self._enable_fd.flush()
        except Exception:
            pass

        self._period_fd.close()
        self._duty_fd.close()
        self._enable_fd.close()

        # Unexport
        try:
            with open(f"{self.chip_path}/unexport", "w") as f:
                f.write(str(self.channel))
        except Exception:
            pass
        self.is_enabled = False
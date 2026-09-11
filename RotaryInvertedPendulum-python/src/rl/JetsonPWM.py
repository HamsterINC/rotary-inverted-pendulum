import os
import time

class JetsonPWM:
    def __init__(self, chip=0, channel=0):
        self.chip_path = f"/sys/class/pwm/pwmchip{chip}"
        self.pwm_path = f"{self.chip_path}/pwm{channel}"
        self.channel = channel
        
        self._export_pwm()
        self.current_period_ns = self._read_int("period", default=10_000_000)
        self.is_enabled = False

    def _export_pwm(self):
        if not os.path.exists(self.pwm_path):
            with open(f"{self.chip_path}/export", "w") as f:
                f.write(str(self.channel))
            time.sleep(0.1)

    def _write_sysfs(self, node, val):
        """Single-shot open/write avoids seek/buffer corruption in Tegra sysfs."""
        path = f"{self.pwm_path}/{node}"
        with open(path, "w") as f:
            f.write(f"{int(val)}\n")

    def _read_int(self, node, default=0):
        path = f"{self.pwm_path}/{node}"
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return default

    def start(self, initial_freq_hz=100):
        """Initializes with output off to establish the baseline safely."""
        period_ns = int(1_000_000_000 / initial_freq_hz)
        # Use 45% duty cycle to avoid edge-case hardware rounding overflows
        duty_ns = int(period_ns * 0.45)

        # 1. Disable while configuring initial state
        self._write_sysfs("enable", 0)
        # 2. Reset duty cycle to 0 first so ANY new period is accepted
        self._write_sysfs("duty_cycle", 0)
        # 3. Set target period
        self._write_sysfs("period", period_ns)
        # 4. Set target duty cycle
        self._write_sysfs("duty_cycle", duty_ns)
        # 5. Turn ON
        self._write_sysfs("enable", 1)

        # Sync with actual hardware-snapped period
        self.current_period_ns = self._read_int("period", default=period_ns)
        self.is_enabled = True

    def change_frequency(self, freq_hz):
        """
        Dynamically updates frequency every 40 ms while keeping enable=1.
        """
        if not self.is_enabled:
            self.start(freq_hz)
            return

        new_period_ns = int(1_000_000_000 / freq_hz)
        # 45% duty gives plenty of breathing room for hardware divisor rounding
        new_duty_ns = int(new_period_ns * 0.45)

        try:
            if new_period_ns < self.current_period_ns:
                # Frequency increasing: period shrinks.
                # Lower duty cycle first, then shrink period.
                self._write_sysfs("duty_cycle", new_duty_ns)
                self._write_sysfs("period", new_period_ns)
            else:
                # Frequency decreasing: period expands.
                # Expand period first, then raise duty cycle.
                self._write_sysfs("period", new_period_ns)
                self._write_sysfs("duty_cycle", new_duty_ns)

            self.current_period_ns = new_period_ns

        except OSError as e:
            # Fallback if kernel snaps period lower than new_duty_ns:
            # Drop duty to 1, apply period, then restore duty.
            self._write_sysfs("duty_cycle", 1)
            self._write_sysfs("period", new_period_ns)
            self._write_sysfs("duty_cycle", new_duty_ns)
            self.current_period_ns = self._read_int("period", default=new_period_ns)

    def stop(self):
        """Disables output and releases the channel."""
        try:
            self._write_sysfs("enable", 0)
        except Exception:
            pass

        try:
            with open(f"{self.chip_path}/unexport", "w") as f:
                f.write(str(self.channel))
        except Exception:
            pass
        self.is_enabled = False
import os
import time

class HardwarePWMStep:
    def __init__(self, chip: int = 0, channel: int = 0):
        self.base_path = f"/sys/class/pwm/pwmchip{chip}/pwm{channel}"
        
        # Export PWM channel if not already exported
        if not os.path.exists(self.base_path):
            try:
                with open(f"/sys/class/pwm/pwmchip{chip}/export", "w") as f:
                    f.write(str(channel))
            except OSError:
                pass  # Already exported or handled by udev
            time.sleep(0.05)

        self.period_file = open(os.path.join(self.base_path, "period"), "w")
        self.duty_file = open(os.path.join(self.base_path, "duty_cycle"), "w")
        self.enable_file = open(os.path.join(self.base_path, "enable"), "w")

        # Start disabled
        self.stop()

    def set_frequency(self, freq_hz: float):
        """Sets output frequency with a 50% duty cycle. Stops if freq == 0."""
        if freq_hz <= 0:
            self.stop()
            return

        period_ns = int(1_000_000_000 / freq_hz)
        duty_ns = period_ns // 2

        # Disable temporarily before adjusting to avoid period < duty_cycle kernel errors
        self.enable_file.seek(0)
        self.enable_file.write("0\n")
        self.enable_file.flush()

        self.period_file.seek(0)
        self.period_file.write(f"{period_ns}\n")
        self.period_file.flush()

        self.duty_file.seek(0)
        self.duty_file.write(f"{duty_ns}\n")
        self.duty_file.flush()

        self.enable_file.seek(0)
        self.enable_file.write("1\n")
        self.enable_file.flush()

    def stop(self):
        self.enable_file.seek(0)
        self.enable_file.write("0\n")
        self.enable_file.flush()

    def close(self):
        self.stop()
        self.period_file.close()
        self.duty_file.close()
        self.enable_file.close()
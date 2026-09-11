import time
import Jetson.GPIO as GPIO

DIR_PIN = 31
STEP_PIN = 33

GPIO.setmode(GPIO.BOARD)
GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(STEP_PIN, GPIO.OUT, initial=GPIO.LOW)

print("Pulsing slowly (20 steps/sec)...")
try:
    for i in range(100):
        GPIO.output(STEP_PIN, GPIO.HIGH)
        time.sleep(0.025)  # 25 ms HIGH
        GPIO.output(STEP_PIN, GPIO.LOW)
        time.sleep(0.025)  # 25 ms LOW
    print("Done.")
finally:
    GPIO.cleanup()
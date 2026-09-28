#include <FastAccelStepper.h>
#include <SPI.h>
#include <AS5047P.h>
#include <Wire.h>

#include "StepperUtils.h"

// Communication speed
const long BAUD_RATE = 2000000;

// Command bytes
#define CMD_READY 0x01
#define CMD_GET_STATE 0x02
#define CMD_SET_ACCEL 0x03   // was CMD_SET_TARGET (position-mode); now angular accel (rad/s²)
#define CMD_ENGAGE_MOTOR 0x04
#define CMD_DISENGAGE_MOTOR 0x05
#define CMD_TARE_PENDULUM 0x06   // re-zero pen_position_rad to current AS5600 reading

// Pin assignments. STEP must be on pin 9 (Timer1 OC1A on ATmega328) for
// FastAccelStepper. DIR and ENABLE can be any digital pin.
#define DIR_PIN 2
#define STEP_PIN 9
#define ENABLE_PIN 5

// Accel-mode envelope. See pendulum_env.py for the corresponding sim
// constants. The velocity cap below corresponds to MAX_VELOCITY_RAD_S
// = 5 rad/s: 5 × (1600 steps/rev / 2π) ≈ 1273 steps/s ⇒ ~785 µs/step.
const uint32_t MOTOR_MIN_STEP_US = 392; //785;  // ≈ 5 rad/s

// Position safety limit (matches MOTOR_SAFE_LIMIT_RAD on the Python side,
// ±125°). Past the rail the firmware actively brakes (commands a fixed
// opposing accel) so the motor decelerates even if the host stops
// sending commands (USB hiccup, host hang). Clamping to zero instead
// would just let moveByAcceleration(0, true) coast the motor past the
// rail at peak velocity.
const int32_t MOTOR_SAFE_LIMIT_STEPS = (int32_t)((125.0f * PI / 180.0f) *
                                                  (3200.0f / (2.0f * PI)));
// Brake authority when past the rail. 150 rad/s² matches the
// pendulum_env.py MAX_ACCEL_RAD_S2 — strong enough to bleed off the
// 5 rad/s vel cap within ~33 ms.
const int32_t MOTOR_BRAKE_ACCEL_STEPS_S2 =
    (int32_t)(150.0f * (3200.0f / (2.0f * PI)));

// Encoder samples are kept in a ring buffer updated at 500 Hz; GET_STATE
// returns velocity computed as (newest - oldest)/Δt over a window of 5
// samples = 4 inter-sample gaps = 8 ms. Window halved from 10 → 5 on
// 2026-05-20 to cut ~9 ms of observation lag — the largest tunable
// component of the rig's policy-loop lag. Tradeoff: ~√2× noisier
// per-sample velocity estimate. See docs/transport_delay.md.
const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  BUFFER_SIZE      = 16;
const uint8_t  VEL_WINDOW       = 5;
// Discard impossibly-large per-sample wraps as I²C glitches. Real
// pendulum tops out at ~50 rad/s, so in one 2 ms sample period the
// raw AS5600 reading can change by at most 50·0.002·4096/(2π) ≈ 65 LSB.
// Anything more is almost certainly a corrupted I²C transaction —
// keep the previous reading rather than letting it pollute the
// accumulator with a spurious ±2π wrap.
const long PEN_RAW_MAX_DELTA_LSB = 500;

static int32_t motor_step_buf[BUFFER_SIZE];   // raw stepper position (steps)
static float   pen_rad_buf[BUFFER_SIZE];      // accumulated pendulum angle (rad)
static uint32_t time_us_buf[BUFFER_SIZE];     // sample timestamps (µs)
static uint8_t  buf_head = 0;                 // next write index
static bool     buf_filled = false;           // becomes true after first full lap
static uint32_t last_sample_us = 0;

// Continuously-tracked pendulum angle (independent of GET_STATE cadence so
// rapid wraparounds aren't missed).
static long    pen_raw_prev   = -1;           // -1 = first read sentinel
static float   pen_position_rad = 0.0f;

// State variables
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5047P AS5047P(10);

// `motor_engaged` is only touched from loop() / handleCommand() — no ISR
// access — so `volatile` would only mislead future readers. Plain bool.
bool motor_engaged = false;

// Function prototypes
void handleCommand();
void sendState();
void sampleState();
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s);

long ZERO_OFFSET_RAW =0;

void setup()
{
    Serial.begin(BAUD_RATE);
    if (!AS5047P.initSPI()) {
      Serial.println("AS5047P init failed!");
      while (1);
    } 

    delay(100); // Give the sensor a moment to stabilize

    // 1. Read the current position at startup and set it as the zero point
    long ZERO_OFFSET_RAW = AS5047P.readAngleRaw();
  
    Serial.print("Physical zero offset captured at: ");
    Serial.println(ZERO_OFFSET_RAW); 

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        while (true) { /* halt: STEP_PIN is not Timer1 OC1A/OC1B */ }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);

    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        while (true) {}
    }
    // Shrink the forward-planning window from the library's 20 ms default
    // to its documented minimum (8 ms = two cyclic-task periods). Default
    // adds ~20 ms of lag between a new accel command and any change in
    // emitted step intervals, because step pulses already queued can't be
    // retroactively edited. Measured (accel_step_probe.py) total firmware
    // lag at default: ~37 ms → expected ~15 ms after this change. At our
    // top step rate (≈1270 steps/s) 8 ms still queues ~10 pulses, well
    // above the "stepper starves and stalls at full speed" failure mode
    // the library warns about.
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    while (!Serial) { ; }
    // while (!as5600.detectMagnet()) { delay(500); }
    // Equivalent to !detectMagnet()
    // Reads the diagnostic register to check the "Magnet Low" (MAGL) error flag
    AS5047P_Types::DIAAGC_t diagnostics = AS5047P.read_DIAAGC(nullptr, true);
    
    while (diagnostics.data.values.MAGL == 1) {
      Serial.println("Magnet not detected or too far away! Please adjust.");
      delay(500);
      // Read the register again to update the status for the next loop iteration
      diagnostics = AS5047P.read_DIAAGC(nullptr, true); 
  }

    last_sample_us = micros();
}

void loop()
{
    uint32_t now_us = micros();
    if ((uint32_t)(now_us - last_sample_us) >= SAMPLE_PERIOD_US)
    {
        last_sample_us = now_us;
        sampleState();
    }
    if (Serial.available() > 0)
    {
        handleCommand();
    }
}

/*
 * Append a (time, motor_position, pendulum_position) sample to the ring
 * buffer and update the pendulum wraparound accumulator.
 */
void sampleState()
{
    int32_t motor_step = stepper->getCurrentPosition();
    long raw = AS5047P.readAngleRaw(); 
    
    if (pen_raw_prev < 0)
    {
        // --- FIRST RUN INITIALIZATION ---
        pen_raw_prev = raw;
        
        // Apply the zero offset ONLY to the initial startup position
        long initial_raw = raw; //- ZERO_OFFSET_RAW;
        
        // Wrap the starting math cleanly between 0 and 16383
        if (initial_raw < 0) initial_raw += 16384;
        
        // Set the starting absolute radian position
        pen_position_rad = (float)initial_raw * (TWO_PI / 16384.0f);
        
        // Optional: If you want the pendulum to start between -PI and PI 
        // instead of 0 to 2PI, uncomment the next line:
        // if (pen_position_rad > PI) pen_position_rad -= TWO_PI;
    }
    else
    {
        // --- CONTINUOUS DELTA TRACKING ---
        long delta = raw - pen_raw_prev;
        
        // AS5047P 14-bit wraparound handling (16384 steps)
        if (delta >  8192) delta -= 16384;
        if (delta < -8192) delta += 16384;
        
        // Glitch rejection (SPI is much more robust than I2C, but still good to have)
        if (delta > PEN_RAW_MAX_DELTA_LSB || delta < -PEN_RAW_MAX_DELTA_LSB)
        {
            // Glitch detected: Do nothing. Skip updating pen_raw_prev and position.
        }
        else
        {
            // Convert the clean 14-bit step delta directly to radians
            pen_position_rad += (float)delta * (TWO_PI / 16384.0f);
            pen_raw_prev = raw;
        }
    }

    motor_step_buf[buf_head] = motor_step;
    pen_rad_buf[buf_head]    = pen_position_rad;
    time_us_buf[buf_head]    = last_sample_us;
    buf_head = (buf_head + 1) % BUFFER_SIZE;
    if (buf_head == 0) buf_filled = true;
}

/*
 * Compute velocity over the most recent VEL_WINDOW samples as
 * (newest - oldest)/Δt. Returns 0 until the buffer holds enough samples.
 */
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s)
{
    uint8_t n_samples = buf_filled ? BUFFER_SIZE : buf_head;
    if (n_samples < VEL_WINDOW)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1)            % BUFFER_SIZE);
    uint8_t oldest = (uint8_t)((buf_head + BUFFER_SIZE - VEL_WINDOW)   % BUFFER_SIZE);

    uint32_t t_new = time_us_buf[newest];
    uint32_t t_old = time_us_buf[oldest];
    float dt_s = (float)((uint32_t)(t_new - t_old)) * 1e-6f;
    if (dt_s <= 0.0f)
    {
        *motor_vel_rad_s = 0.0f;
        *pen_vel_rad_s   = 0.0f;
        return;
    }

    int32_t motor_step_delta = motor_step_buf[newest] - motor_step_buf[oldest];
    *motor_vel_rad_s = ((float)motor_step_delta * ((2.0f * PI) / 3200.0f)) / dt_s;

    float pen_delta = pen_rad_buf[newest] - pen_rad_buf[oldest];
    *pen_vel_rad_s = pen_delta / dt_s;
}

/*
 * Handle incoming commands.
 */
void handleCommand()
{
    uint8_t command = Serial.read();

    switch (command)
    {
    case CMD_READY:
        Serial.write(CMD_READY);
        break;

    case CMD_GET_STATE:
        sendState();
        break;

    case CMD_SET_ACCEL:
        {
            // Read 4 bytes (size of float) with a short timeout. If a byte
            // gets dropped — UART overrun under stepper EMI, or a flipped
            // command byte that desyncs the parser — readBytes returns
            // short and we bail. The next command re-syncs.
            Serial.setTimeout(5);
            float accel_rad_s2;
            size_t n = Serial.readBytes((char *)&accel_rad_s2, sizeof(float));
            Serial.setTimeout(1000);  // restore Stream default
            if (n != sizeof(float)) break;

            if (!motor_engaged) break;

            // Convert rad/s² to steps/s² (3200 microsteps per revolution).
            // moveByAcceleration takes int32_t.
            int32_t accel_steps_s2 =
                (int32_t)(accel_rad_s2 * (3200.0f / (2.0f * PI)));

            // Position-limit safety: past the rail, ignore the host's
            // command and actively brake instead. Just zeroing accel here
            // is dangerous — with allow_reverse=true, an accel of 0 means
            // "hold current speed", so a motor at +5 rad/s heading
            // outbound would coast indefinitely if the host went quiet.
            // Brake with a fixed opposing accel so we decelerate
            // regardless of host liveness.
            int32_t cur_pos = stepper->getCurrentPosition();
            if (cur_pos >= MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = -MOTOR_BRAKE_ACCEL_STEPS_S2;
            }
            else if (cur_pos <= -MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = +MOTOR_BRAKE_ACCEL_STEPS_S2;
            }

            // allow_reverse=true makes the library decelerate smoothly through
            // zero when the sign of accel opposes the current velocity — no
            // state machine needed on our side.
            stepper->moveByAcceleration(accel_steps_s2, true);
        }
        break;

    case CMD_ENGAGE_MOTOR:
        motor_engaged = true;
        stepper->enableOutputs();
        // Start in zero-accel state. moveByAcceleration(0, true) means
        // "hold current speed", and since we've just enabled the driver
        // the stepper is at rest — so this leaves it at rest until the
        // host sends its first CMD_SET_ACCEL.
        // Note: ENGAGE does not re-zero `getCurrentPosition()` or
        // `pen_position_rad`. Position counters persist across
        // engage/disengage cycles (and across episodes) by design — the
        // host is responsible for tracking the frame.
        stepper->moveByAcceleration(0, true);
        break;

    case CMD_DISENGAGE_MOTOR:
        motor_engaged = false;
        // forceStop() drains the Timer1 step queue immediately; without this,
        // queued steps would continue advancing the firmware position counter
        // even though the driver's enable pin is HIGH.
        stepper->forceStop();
        stepper->disableOutputs();
        break;

    case CMD_TARE_PENDULUM:
        // Re-zero pen_position_rad to the current AS5600 reading. Used by
        // real_env.reset() after the pendulum has settled at rest so that
        // each fine-tune episode samples a fresh bias from the rig's
        // static-friction-bounded rest distribution (±1.9°). Without this,
        // all fine-tune episodes share the firmware-boot bias and the
        // policy overfits to that single calibration offset.
        //
        // Shift pen_position_rad AND every entry in pen_rad_buf by the
        // current pen_position_rad. The buffer's relative deltas (used
        // by computeVelocities) are preserved, so velocity calculation
        // continues uninterrupted across the tare. Disable interrupts so
        // sampleState() doesn't run mid-update with a partially-shifted
        // buffer.
        noInterrupts();
        {
            float offset = pen_position_rad;
            for (uint8_t i = 0; i < BUFFER_SIZE; i++)
            {
                pen_rad_buf[i] -= offset;
            }
            pen_position_rad = 0.0f;
        }
        interrupts();
        Serial.write(CMD_TARE_PENDULUM);  // ack
        break;

    default:
        break;
    }
}

/*
 * Send the current state of the system:
 * - Current time in microseconds   (4 bytes, uint32)
 * - Stepper motor position in rad  (4 bytes, float)
 * - Pendulum position in rad       (4 bytes, float)
 * - Stepper motor velocity in rad/s (4 bytes, float)
 * - Pendulum velocity in rad/s     (4 bytes, float)
 */
void sendState()
{
    // Take the most recent buffer entry as the instantaneous position,
    // and compute velocity from the (newest - oldest)/Δt regression
    // window. The timestamp is the sample time of `newest` — not a
    // fresh micros() — so the host gets a self-consistent (t, pos, vel)
    // tuple it can time-align without inheriting up-to-one-sample of
    // bias.
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    uint32_t current_time = time_us_buf[newest];
    float motor_position_radians = stepsToRadians(motor_step_buf[newest]);
    float pendulum_position_radians = pen_rad_buf[newest];

    float motor_velocity_rad_s, pendulum_velocity_rad_s;
    computeVelocities(&motor_velocity_rad_s, &pendulum_velocity_rad_s);

    // Flip the signs of the motor and pendulum positions / velocities to
    // match the sim-frame convention the Python clients expect.
    motor_position_radians    *= -1;
    pendulum_position_radians *= -1;
    motor_velocity_rad_s      *= -1;
    pendulum_velocity_rad_s   *= -1;

    // Pack and send the data
    Serial.write((byte *)&current_time, sizeof(current_time));
    Serial.write((byte *)&motor_position_radians, sizeof(motor_position_radians));
    Serial.write((byte *)&pendulum_position_radians, sizeof(pendulum_position_radians));
    Serial.write((byte *)&motor_velocity_rad_s, sizeof(motor_velocity_rad_s));
    Serial.write((byte *)&pendulum_velocity_rad_s, sizeof(pendulum_velocity_rad_s));
}

#include <FastAccelStepper.h>
#include <SPI.h>
#include <AS5047P.h>
#include <Wire.h>

#include "StepperUtils.h"

// Communication speed
const long BAUD_RATE = 1000000;

// Command bytes
#define CMD_READY 0x01
#define CMD_GET_STATE 0x02
#define CMD_SET_ACCEL 0x03   // was CMD_SET_TARGET (position-mode); now angular accel (rad/s²)
#define CMD_ENGAGE_MOTOR 0x04
#define CMD_DISENGAGE_MOTOR 0x05
#define CMD_TARE_PENDULUM 0x06   // re-zero pen_position_rad to current AS5600 reading

// ESP32 SPI Pins (Standard VSPI)
#define SCK_PIN  18
#define MISO_PIN 19
#define MOSI_PIN 23
#define CS_PIN   5

// ESP32 Safe Stepper Pins. 
// FastAccelStepper on ESP32 uses MCPWM/RMT, so it is NOT restricted to specific pins.
#define DIR_PIN 26
#define STEP_PIN 27
#define ENABLE_PIN 25

// Accel-mode envelope. See pendulum_env.py for the corresponding sim
// constants. The velocity cap below corresponds to MAX_VELOCITY_RAD_S
// = 5 rad/s: 5 × (1600 steps/rev / 2π) ≈ 1273 steps/s ⇒ ~785 µs/step.
const uint32_t MOTOR_MIN_STEP_US = 392; //785;  // ≈ 5 rad/s

// Position safety limit (matches MOTOR_SAFE_LIMIT_RAD on the Python side, ±125°).
const int32_t MOTOR_SAFE_LIMIT_STEPS = (int32_t)((125.0f * PI / 180.0f) *
                                                 (3200.0f / (2.0f * PI)));
// Brake authority when past the rail.
const int32_t MOTOR_BRAKE_ACCEL_STEPS_S2 =
    (int32_t)(150.0f * (3200.0f / (2.0f * PI)));

const uint16_t SAMPLE_PERIOD_US = 2000;
const uint8_t  BUFFER_SIZE      = 16;
const uint8_t  VEL_WINDOW       = 5;
const long PEN_RAW_MAX_DELTA_LSB = 500;

static int32_t motor_step_buf[BUFFER_SIZE];   // raw stepper position (steps)
static float   pen_rad_buf[BUFFER_SIZE];      // accumulated pendulum angle (rad)
static uint32_t time_us_buf[BUFFER_SIZE];     // sample timestamps (µs)
static uint8_t  buf_head = 0;                 // next write index
static bool     buf_filled = false;           // becomes true after first full lap
static uint32_t last_sample_us = 0;

// Continuously-tracked pendulum angle
static long    pen_raw_prev   = -1;           // -1 = first read sentinel
static float   pen_position_rad = 0.0f;

// State variables
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *stepper = NULL;
AS5047P AS5047P(CS_PIN); // Initialize with ESP32 CS pin

bool motor_engaged = false;

// Function prototypes
void handleCommand();
void sendState();
void sampleState();
void computeVelocities(float* motor_vel_rad_s, float* pen_vel_rad_s);

long ZERO_OFFSET_RAW = 0;

void setup()
{
    Serial.begin(BAUD_RATE);
    // 1. Wait for the Serial Monitor to actually connect!
    while (!Serial) { ; } 
    delay(1000); // Give the USB port 1 extra second to stabilize

    Serial.println("\n--- ESP32 Booting ---");
    
    // Explicitly initialize SPI for ESP32 before starting the sensor
    SPI.begin(SCK_PIN, MISO_PIN, MOSI_PIN, CS_PIN);
    
    if (!AS5047P.initSPI()) {
      Serial.println("AS5047P init failed!");
      while (1);
    } 

    delay(100); // Give the sensor a moment to stabilize

    // 1. Read the current position at startup and set it as the zero point
    ZERO_OFFSET_RAW = AS5047P.readAngleRaw();
  
    Serial.print("Physical zero offset captured at: ");
    Serial.println(ZERO_OFFSET_RAW); 

    engine.init();
    stepper = engine.stepperConnectToPin(STEP_PIN);
    if (!stepper)
    {
        while (true) { /* halt: Failed to connect stepper to pin */ }
    }
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(ENABLE_PIN);
    stepper->setAutoEnable(false);
    stepper->setDelayToEnable(50);

    int8_t rc_speed = stepper->setSpeedInUs(MOTOR_MIN_STEP_US);
    if (rc_speed != 0)
    {
        while (true) {}
    }
    
    stepper->setForwardPlanningTimeInMs(8);
    stepper->disableOutputs();

    while (!Serial) { ; }
    
    AS5047P_Types::DIAAGC_t diagnostics = AS5047P.read_DIAAGC(nullptr, true);
    
    while (diagnostics.data.values.MAGL == 1) {
      Serial.println("Magnet not detected or too far away! Please adjust.");
      delay(500);
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

void sampleState()
{
    int32_t motor_step = stepper->getCurrentPosition();
    long raw = AS5047P.readAngleRaw(); 
    
    if (pen_raw_prev < 0)
    {
        pen_raw_prev = raw;
        long initial_raw = raw;
        if (initial_raw < 0) initial_raw += 16384;
        pen_position_rad = (float)initial_raw * (TWO_PI / 16384.0f);
    }
    else
    {
        long delta = raw - pen_raw_prev;
        
        // AS5047P 14-bit wraparound handling
        if (delta >  8192) delta -= 16384;
        if (delta < -8192) delta += 16384;
        
        if (delta > PEN_RAW_MAX_DELTA_LSB || delta < -PEN_RAW_MAX_DELTA_LSB)
        {
            // Glitch detected: Do nothing
        }
        else
        {
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
            Serial.setTimeout(5);
            float accel_rad_s2;
            size_t n = Serial.readBytes((char *)&accel_rad_s2, sizeof(float));
            Serial.setTimeout(1000); 
            if (n != sizeof(float)) break;

            if (!motor_engaged) break;

            int32_t accel_steps_s2 =
                (int32_t)(accel_rad_s2 * (3200.0f / (2.0f * PI)));

            int32_t cur_pos = stepper->getCurrentPosition();
            if (cur_pos >= MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = -MOTOR_BRAKE_ACCEL_STEPS_S2;
            }
            else if (cur_pos <= -MOTOR_SAFE_LIMIT_STEPS)
            {
                accel_steps_s2 = +MOTOR_BRAKE_ACCEL_STEPS_S2;
            }

            stepper->moveByAcceleration(accel_steps_s2, true);
        }
        break;

    case CMD_ENGAGE_MOTOR:
        motor_engaged = true;
        stepper->enableOutputs();
        stepper->moveByAcceleration(0, true);
        break;

    case CMD_DISENGAGE_MOTOR:
        motor_engaged = false;
        stepper->forceStop();
        stepper->disableOutputs();
        break;

    case CMD_TARE_PENDULUM:
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
        Serial.write(CMD_TARE_PENDULUM); 
        break;

    default:
        break;
    }
}

void sendState()
{
    uint8_t newest = (uint8_t)((buf_head + BUFFER_SIZE - 1) % BUFFER_SIZE);
    uint32_t current_time = time_us_buf[newest];
    float motor_position_radians = stepsToRadians(motor_step_buf[newest]);
    float pendulum_position_radians = pen_rad_buf[newest];

    float motor_velocity_rad_s, pendulum_velocity_rad_s;
    computeVelocities(&motor_velocity_rad_s, &pendulum_velocity_rad_s);

    motor_position_radians    *= -1;
    pendulum_position_radians *= -1;
    motor_velocity_rad_s      *= -1;
    pendulum_velocity_rad_s   *= -1;

    Serial.write((byte *)&current_time, sizeof(current_time));
    Serial.write((byte *)&motor_position_radians, sizeof(motor_position_radians));
    Serial.write((byte *)&pendulum_position_radians, sizeof(pendulum_position_radians));
    Serial.write((byte *)&motor_velocity_rad_s, sizeof(motor_velocity_rad_s));
    Serial.write((byte *)&pendulum_velocity_rad_s, sizeof(pendulum_velocity_rad_s));
}
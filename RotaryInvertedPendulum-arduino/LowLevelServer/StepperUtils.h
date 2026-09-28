#ifndef STEPPER_UTILS_H
#define STEPPER_UTILS_H

#include <Arduino.h>   // PI, round()
#include <stdint.h>

const int32_t microstepsPerRev = 3200; // 200 steps * 8 microsteps

// `int` is 16-bit on AVR — anything beyond ±32767 steps (~20 revolutions)
// would silently wrap. Use int32_t end-to-end so the conversion helpers
// match FastAccelStepper's int32_t step counts.

// Convert steps to degrees
float stepsToDegrees(int32_t steps)
{
  return (steps / float(microstepsPerRev)) * 360.0;
}

// Convert steps to radians
float stepsToRadians(int32_t steps)
{
  return (steps / float(microstepsPerRev)) * (2 * PI);
}

// Convert degrees to steps
int32_t degreesToSteps(float degrees)
{
  return (int32_t)round((degrees / 360.0) * microstepsPerRev);
}

// Convert radians to steps
int32_t radiansToSteps(float radians)
{
  return (int32_t)round((radians / (2 * PI)) * microstepsPerRev);
}

#endif

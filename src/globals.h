#ifndef GLOBALS_H
#define GLOBALS_H

#include <chrono>
#include <cstddef>

// Declaration of global variables
extern std::chrono::time_point<std::chrono::system_clock> startTime;
extern std::chrono::time_point<std::chrono::system_clock> endTime;
extern double time_copy;

// Function to measure time difference
double getTimeDifference();

#endif

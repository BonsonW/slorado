#ifndef GLOBALS_H
#define GLOBALS_H

#include <chrono>
#include <cstddef>

// Declaration of global variables
extern std::chrono::time_point<std::chrono::system_clock> startTime;
extern std::chrono::time_point<std::chrono::system_clock> endTime;
extern double_t time_copy;

// Function to measure time difference
double_t getTimeDifference();

#endif

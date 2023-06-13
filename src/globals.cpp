#include "globals.h"

// Definition of global variables
std::chrono::time_point<std::chrono::system_clock> startTime;
std::chrono::time_point<std::chrono::system_clock> endTime;
double_t time_copy;

// Function to measure time difference in seconds
double_t getTimeDifference() {
    std::chrono::duration<double> timeSpan = endTime - startTime;
    return timeSpan.count();
}

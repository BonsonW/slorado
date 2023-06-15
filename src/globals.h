#ifndef GLOBALS_H
#define GLOBALS_H

#include <chrono>
#include <cstddef>

// Declaration of global variables
extern double startTime;
extern double endTime;

extern double subStartTime;
extern double subEndTime;

extern double time_forward;
extern double forward_l62;
extern double forward_l159;
extern double forward_l469;
extern double forward_l5136;
extern double forward_l577;
extern double forward_l642;

extern double x_flip;
extern double rnn1;
extern double rnn2;
extern double rnn3;
extern double rnn4;
extern double rnn5;


// Function to measure time difference
double getTimeDifference();

double getSubTimeDifference();

#endif

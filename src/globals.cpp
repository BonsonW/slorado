#include "globals.h"
#include "misc.h"

double startTime;
double endTime;

double subStartTime;
double subEndTime;

double time_forward;
double forward_l62;
double forward_l159;
double forward_l469;
double forward_l5136;
double forward_l577;
double forward_l642;

double x_flip;
double rnn1;
double rnn2;
double rnn3;
double rnn4;
double rnn5;

// Function to measure time difference in seconds
double getTimeDifference() {
    return endTime - startTime;
}

double getSubTimeDifference() {
    return subEndTime - subStartTime;
}
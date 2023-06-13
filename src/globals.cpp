#include "globals.h"
#include "misc.h"

double startTime;
double endTime;

double time_forward;
extern double forward_l62;
extern double forward_l159;
extern double forward_l469;
extern double forward_l510;
extern double forward_l577;
extern double forward_l642;

// Function to measure time difference in seconds
double getTimeDifference() {
    return endTime - startTime;
}

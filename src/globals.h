#ifndef GLOBALS_H
#define GLOBALS_H

#include <chrono>
#include <cstddef>

// Declaration of global variables
extern bool isCUDA;

extern double startTime;
extern double endTime;

extern double subStartTime;
extern double subEndTime;

extern double subStartTimev2;
extern double subEndTimev2;

extern double time_forward;
extern double forward_l62;
extern double forward_l159;
extern double forward_l469;
extern double forward_l536;
extern double forward_l577;
extern double forward_l642;

extern double x_flipt;
extern double rnn1t;
extern double rnn2t;
extern double rnn3t;
extern double rnn4t;
extern double rnn5t;

extern double rnn1tt1;
extern double rnn1th1;
extern double rnn1ty1;
extern double rnn1tflip;

//isCUDA
extern double CudaCallerT;
extern double NCudaCallerT;
extern double NNTaskT;
extern double call_chunksT;
extern double cuda_thread_fnT;
extern double SubCudaCallerT;




// Function to measure time difference
double getTimeDifference();

double getSubTimeDifference();

double getSubTimeDifferencev2();

#endif

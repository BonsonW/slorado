#include "globals.h"
#include "misc.h"

bool isCUDA = false;

double startTime;
double endTime;

double subStartTime;
double subEndTime;

double subStartTimev2;
double subEndTimev2;

double time_forward;
double forward_l62;
double forward_l159;
double forward_l469;
double forward_l536;
double forward_l577;
double forward_l642;
double cudaLSTM;

double x_flipt;
double rnn1t;
double rnn2t;
double rnn3t;
double rnn4t;
double rnn5t;

double rnn1tt1;
double rnn1th1;
double rnn1ty1;
double rnn1tflip;

//isCUDA
double CudaCallerT;
double CudaCallerT1;
double CudaCallerT2;
double CudaCallerT3;
double CudaCallerT4;
double CudaCallerT5;
double NCudaCallerT;
double NNTaskT;
double call_chunksT;
double cuda_thread_fnT;
double SubCudaCallerT;

double matMul;
double cublasGemmExT;
double assertT;

// Function to measure time difference in seconds
double getTimeDifference() {
    return endTime - startTime;
}

double getSubTimeDifference() {
    return subEndTime - subStartTime;
}

double getSubTimeDifferencev2() {
    return subEndTimev2 - subStartTimev2;
}
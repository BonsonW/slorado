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
extern double cudaLSTM;

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
extern double CudaCallerT1;
extern double CudaCallerT2;
extern double CudaCallerT3;
extern double CudaCallerT4;
extern double CudaCallerT5;
extern double load_crf_modelT;
extern double load_crf_modelT1;
extern double NCudaCallerT;
extern double NNTaskT;
extern double call_chunksT;
extern double cuda_thread_fnT;
extern double cuda_thread_fnT2;
extern double cuda_thread_fnT3;
extern double cuda_thread_fnT4;
extern double cuda_thread_fnT5;
extern double cuda_thread_fnT6;
extern double SubCudaCallerT;
extern double forward_cublasT;
extern double forward_cublasT2;
extern double forward_cublasT3;
extern double cudaLSTMImplT;
extern double convolutionImplT;
extern double cudaLSTMStackImplT;
extern double matmul_f16T;
extern double host_transpose_f16T;
extern double rnnIterate;
extern double forLoopRest;
extern double state_bufT;
extern double weights_cpuT;
extern double weightsT;
extern double biasT;
extern double transposed_weightsT;
extern double weightCPUcalls;
extern double cont;
extern double ncont;
// extern auto transWeights;

extern torch::Tensor weightsTrans;

extern double NNTaskT0;
extern double NNTaskT1;
extern double NNTaskT2;

extern double matMul;
extern double cublasGemmExT;
extern double assertT;

// Function to measure time difference
double getTimeDifference();

double getSubTimeDifference();

double getSubTimeDifferencev2();

#endif

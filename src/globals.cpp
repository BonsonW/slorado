#include "globals.h"
#include "misc.h"
#include <torch/torch.h>

bool isCUDA = false;

double subStartTime;
double subEndTime;
double subStartTimev2;
double subEndTimev2;
double time_forward;
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
double CudaCallerT;
double CudaCallerT1;
double CudaCallerT2;
double CudaCallerT3;
double CudaCallerT4;
double CudaCallerT5;
double load_crf_modelT;
double load_crf_modelT1;
double NCudaCallerT;
double NNTaskT;
double call_chunksT;
double cuda_thread_fnT;
double cuda_thread_fnT2;
double cuda_thread_fnT3;
double cuda_thread_fnT4;
double cuda_thread_fnT5;
double cuda_thread_fnT6;
double SubCudaCallerT;
double forward_cublasT;
double forward_cublasT2;
double forward_cublasT3;
double cudaLSTMImplT;
double convolutionImplT;
double cudaLSTMStackImplT;
double matmul_f16T;
double host_transpose_f16T;
double rnnIterate;
double forLoopRest;
double state_bufT;
double weights_cpuT;
double weightsT;
double biasT;
double transposed_weightsT;
double weightCPUcalls;
double cont;
double ncont;

std::vector<at::Tensor> transposedRNNWeights;
std::vector<at::Tensor> GPUWeights;

torch::Tensor rnn1WeightsT;
torch::Tensor rnn2WeightsT;
torch::Tensor rnn3WeightsT;
torch::Tensor rnn4WeightsT;
torch::Tensor rnn5WeightsT;
bool setTrans = false;

double matMul;
double cublasGemmExT;
double assertT;
double NNTaskT0;
double NNTaskT1;
double NNTaskT2;
double beam_searchT;               
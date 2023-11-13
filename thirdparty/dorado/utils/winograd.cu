#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <cublas_v2.h>
#include <torch/torch.h>

#define BLOCK_THRESHOLD 1280000
#define TILE_X 32
#define TILE_Y 16
const float alpha = 1.0f; // Scaling factor for matrix A
const float beta = 1.0f;  // Scaling factor for matrix B
const float nbeta = -1.0f;
const float zbeta = 0.0f;
const dim3 blockDim(TILE_X * TILE_Y);
const dim3 gridDim((32/TILE_X) * (32/TILE_Y));

__global__ void computeC12 (float* C, float* m1 ,float* m2 ,float* m5 ,float* m6 , int width , int subWidth, int height, int subHeight )
{
    int tx = threadIdx.x; 
    /*This line obtains the x-coordinate (column) of the current thread within a thread block.*/
    int ty = threadIdx.y;
    /* This line obtains the y-coordinate (row) of the current thread within a thread block.*/
    int row = blockIdx.y * TILE_Y + ty;
    /* This line calculates the global row index in the original matrix for the current thread block using the blockIdx.y and threadIdx.y values. The variable 'TILE_Y' seems to represent the height of the tile.*/
    int column = blockIdx.x * TILE_X + tx;
    /* This line calculates the global column index in the original matrix for the current thread block using the blockIdx.x and threadIdx.x values. The variable 'TILE_X' seems to represent the width of the tile.*/
    __shared__ float as[ TILE_Y ][ TILE_X ];
    /* This line declares a shared memory array 'as' with dimensions TILE_Y by TILE_X. Shared memory is used for cooperative thread block-level data sharing.*/
    float Csub ; /*to store the intermediate sum.*/

    as[ty ][ tx ]= m1 [( row )* subWidth + column ];
    Csub =as[ty ][ tx ];
    as[ty ][ tx ]= m2 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m5 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m6 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    C[( row )* width + column ]= Csub;

}
__global__ void computeC11 (float* C,float* m2 ,float* m3,int width , int subWidth , int height, int subHeight)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_Y + ty;
    int column = blockIdx.x * TILE_X + tx;
    __shared__ float as[ TILE_Y ][ TILE_X ];
    float Csub;

    as[ty ][ tx ]= m2 [( row )* subWidth + column ];
    Csub =as[ty ][ tx ];
    as[ty ][ tx ]= m3 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    C[( row )* width + column ]= Csub;

}

__global__ void computeC21 (float* C,float* m1, float* m2, float* m4, float* m7 , int width , int subWidth, int height, int subHeight ){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_Y + ty;
    int column = blockIdx.x * TILE_X + tx;
    __shared__ float as[ TILE_Y ][ TILE_X ];
    float Csub ;

    as[ty ][ tx ]= m1 [( row )* subWidth + column ];
    Csub =as[ty ][ tx ];
    as[ty ][ tx ]= m2 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m4 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m7 [( row )* subWidth + column ];
    Csub -= as[ty ][ tx ];
    C[( row )* width + column ]= Csub;

}
__global__ void computeC22 (float* C,float* m1,float* m2, float* m4, float* m5 , int width , int subWidth, int height, int subHeight )
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_Y + ty;
    int column = blockIdx.x * TILE_X + tx;
    __shared__ float as[ TILE_Y ][ TILE_X ];
    float Csub ;

    as[ty ][ tx ]= m1 [( row )* subWidth + column ];
    Csub =as[ty ][ tx ];
    as[ty ][ tx ]= m2 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m4 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    as[ty ][ tx ]= m5 [( row )* subWidth + column ];
    Csub += as[ty ][ tx ];
    C[( row )* width + column ]= Csub;

}


__global__ void mergeSubmatrices(float* submatrix0, float* submatrix1, float* submatrix2, float* submatrix3, float* finalMatrix,int N, int M)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    finalMatrix[y * M + x] = submatrix0[y * N + x];
    finalMatrix[y * M + x + N] = submatrix1[y * N + x];
    finalMatrix[(y + N) * M + x] = submatrix2[y * N + x];
    finalMatrix[(y + N) * M + x + N] = submatrix3[y * N + x];
}


void winograd_mm(torch::Tensor const  &A,torch::Tensor const &B,torch::Tensor &C){

     auto start_time_mem_alloc = std::chrono::high_resolution_clock::now();
        //A, B, C are memory pointers on host
        // rowsB = colsA
        int rowsA = A.size(0);
        int colsA = A.size(1);
	int rowsB = B.size(0);
        int colsB = B.size(1);
        int subRowsA=rowsA/2;
        int subColsA=colsA/2;
        int subRowsB=rowsB/2;
        int subColsB=colsB/2;
     
     float *d_A11=0,*d_A12=0,*d_A21=0,*d_A22=0, *d_B11=0, *d_B12=0, *d_B21=0, *d_B22=0, *d_m1=0,*d_m2=0, *d_m3=0, *d_m4=0, *d_m5=0, *d_m6=0, *d_m7=0, *d_S1=0, *d_S2=0, *d_S3=0, *d_S4=0, *d_S5=0, *d_S6=0, *d_S7=0, *d_S8=0;
     float *d_C11=0, *d_C12=0, *d_C21=0, *d_C22=0, *d_C=0;
     cudaMalloc((void**)&d_A11, subRowsA* subColsA * sizeof(float));
     cudaMalloc((void**)&d_A12, subRowsA* subColsA * sizeof(float));
     cudaMalloc((void**)&d_A21, subRowsA* subColsA * sizeof(float));
     cudaMalloc((void**)&d_A22, subRowsA* subColsA * sizeof(float));

     cudaMalloc((void**)&d_B11, subRowsB* subColsB * sizeof(float));
     cudaMalloc((void**)&d_B12, subRowsB* subColsB * sizeof(float));
     cudaMalloc((void**)&d_B21, subRowsB* subColsB * sizeof(float));
     cudaMalloc((void**)&d_B22, subRowsB* subColsB * sizeof(float));


     auto start_time_split = std::chrono::high_resolution_clock::now();

     // Split the original matrix into four equal submatrices
     cublasSetMatrix(subRowsA, subColsA, sizeof(float), (float *)A.data_ptr<float>(), colsA, d_A11, subColsA);
     cublasSetMatrix(subRowsA, subColsA, sizeof(float), (float *)A.data_ptr<float>()+ subColsA, colsA, d_A12, subColsA);
     cublasSetMatrix(subRowsA, subColsA, sizeof(float), (float *)A.data_ptr<float>()+ subRowsA * colsA, colsA, d_A21, subColsA);
     cublasSetMatrix(subRowsA, subColsA, sizeof(float), (float *)A.data_ptr<float>()+ subRowsA * colsA+ subColsA,colsA, d_A22, subColsA);

     cublasSetMatrix(subRowsB, subColsB, sizeof(float), (float *)B.data_ptr<float>(), colsB, d_B11, subColsB);
     cublasSetMatrix(subRowsB, subColsB, sizeof(float), (float *)B.data_ptr<float>()+ subColsB, colsB, d_B12, subColsB);
     cublasSetMatrix(subRowsB, subColsB, sizeof(float), (float *)B.data_ptr<float>()+ subRowsB * colsB, colsB, d_B21, subColsB);
     cublasSetMatrix(subRowsB, subColsB, sizeof(float), (float *)B.data_ptr<float>()+ subRowsB * colsB+ subColsB, colsB, d_B22, subColsB);

     auto end_time_split = std::chrono::high_resolution_clock::now();
     cublasHandle_t handle;
     cublasStatus_t cudaStatus=cublasCreate(&handle);

     cudaMalloc((void**)&d_S1, subRowsA*subColsA*sizeof(float));
     cudaMalloc((void**)&d_S2, subRowsA*subColsA*sizeof(float));
     cudaMalloc((void**)&d_S3, subRowsA*subColsA*sizeof(float));
     cudaMalloc((void**)&d_S4, subRowsA*subColsA*sizeof(float));
     cudaMalloc((void**)&d_S5, subRowsB*subColsB*sizeof(float));
     cudaMalloc((void**)&d_S6, subRowsB*subColsB*sizeof(float));
     cudaMalloc((void**)&d_S7, subRowsB*subColsB*sizeof(float));
     cudaMalloc((void**)&d_S8, subRowsB*subColsB*sizeof(float));

     cudaMalloc((void**)&d_C11, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_C12, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_C21, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_C22, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_C, rowsA*colsB*sizeof(float));

     auto end_time_mem_alloc = std::chrono::high_resolution_clock::now();

     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsA, subColsA,
                          &alpha,
                          d_A21, subRowsA,
                          &beta,
                          d_A22, subRowsA,
                          d_S1, subRowsA);
     // add(A21,A22,S1);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsA, subColsA,
                          &alpha,
                          d_A11, subRowsA,
                          &nbeta,
                          d_A21, subRowsA,
                          d_S3, subRowsA);
     // add(A11,A21,S3);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsB, subColsB,
                          &alpha,
                          d_B12, subRowsB,
                          &nbeta,
                          d_B11, subRowsB,
                          d_S5, subRowsB);
     //sub(B12,B11,S5);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsB, subColsB,
                          &alpha,
                          d_B22, subRowsB,
                          &nbeta,
                          d_B12, subRowsB,
                          d_S7, subRowsB);
     // sub(B22,B12,S7);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsA, subColsA,
                          &alpha,
                          d_S1, subRowsA,
                          &nbeta,
                          d_A11, subRowsA,
                          d_S2, subRowsA);
     // sub(S1,A11,S2);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsB, subColsB,
                          &alpha,
                          d_B22, subRowsB,
                          &nbeta,
                          d_S5, subRowsB,
                          d_S6, subRowsB);
     // sub(B22,S5,S6);
     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsA, subColsA,
                          &alpha,
                          d_A12, subRowsA,
                          &nbeta,
                          d_S2, subRowsA,
                          d_S4, subRowsA);
    // sub(A12,S2,S4);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          subRowsB, subColsB,
                          &alpha,
                          d_S6, subRowsB,
                          &nbeta,
                          d_B21, subRowsB,
                          d_S8, subRowsB);

     cudaMalloc((void**)&d_m1, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m2, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m3, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m4, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m5, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m6, subRowsA*subColsB*sizeof(float));
     cudaMalloc((void**)&d_m7, subRowsA*subColsB*sizeof(float));

    auto end_time_sgeam = std::chrono::high_resolution_clock::now();
    auto start_time_sgemm=std::chrono::high_resolution_clock::now(), end_time_sgemm=std::chrono::high_resolution_clock::now(), start_time_recursive=std::chrono::high_resolution_clock::now() ,end_time_recursive=std::chrono::high_resolution_clock::now();
    if (rowsA <= BLOCK_THRESHOLD){
	start_time_sgemm = std::chrono::high_resolution_clock::now();
        //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_S6, subColsB, d_S2, subColsA, &zbeta, d_m1, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_B11, subColsB, d_A11, subColsA, &zbeta, d_m2, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_B21, subColsB, d_A12, subColsA, &zbeta, d_m3, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_S7, subColsB, d_S3, subColsA, &zbeta, d_m4, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_S5, subColsB, d_S1, subColsA, &zbeta, d_m5, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_B22, subColsB, d_S4, subColsA, &zbeta, d_m6, subColsB);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, subColsB, subRowsA, subColsA, &alpha, d_S8, subColsB, d_A22, subColsA, &zbeta, d_m7, subColsB);
	end_time_sgemm = std::chrono::high_resolution_clock::now();
   } else {
	/*	
	//start_time_recursive = std::chrono::high_resolution_clock::now();
        winograd_mm(d_S2, d_S6, d_m1, subRowsA, subColsA, subRowsB, subColsB);
        winograd_mm(d_A11, d_B11, d_m2, subRowsA, subColsA,subRowsB,subColsB);
        winograd_mm(d_A12, d_B21, d_m3, subRowsA, subColsA,subRowsB,subColsB);
        winograd_mm(d_S3, d_S7, d_m4, subRowsA, subColsA,subRowsB,subColsB);
        winograd_mm(d_S1, d_S5, d_m5, subRowsA, subColsA,subRowsB,subColsB);
        winograd_mm(d_S4, d_B22, d_m6, subRowsA, subColsA,subRowsB,subColsB);
        winograd_mm(d_A22, d_S8, d_m7, subRowsA, subColsA,subRowsB,subColsB);
	//end_time_recursive = std::chrono::high_resolution_clock::now();
	*/
    }
    auto start_time_free = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    cudaFree(d_A11);
    cudaFree(d_A12);
    cudaFree(d_A21);
    cudaFree(d_A22);
    cudaFree(d_B11);
    cudaFree(d_B12);
    cudaFree(d_B21);
    cudaFree(d_B22);
    cudaFree(d_S1);
    cudaFree(d_S2);
    cudaFree(d_S3);
    cudaFree(d_S4);
    cudaFree(d_S5);
    cudaFree(d_S6);
    cudaFree(d_S7);
    cudaFree(d_S8);

    auto start_time_kernels = std::chrono::high_resolution_clock::now();

    computeC11<<<gridDim,blockDim>>> (d_C11, d_m2, d_m3, colsB, subColsB, rowsA, subRowsA);
    computeC12<<<gridDim,blockDim>>> (d_C12, d_m1, d_m2 , d_m5 , d_m6, colsB, subColsB, rowsA, subRowsA);
    computeC21<<<gridDim,blockDim>>> (d_C21, d_m1 , d_m2 , d_m4, d_m7, colsB, subColsB, rowsA, subRowsA);
    computeC22<<<gridDim,blockDim>>> (d_C22, d_m1, d_m2, d_m4, d_m5 , colsB, subColsB, rowsA, subRowsA);

    auto end_time_kernels = std::chrono::high_resolution_clock::now();

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_m3);
    cudaFree(d_m4);
    cudaFree(d_m5);
    cudaFree(d_m6);
    cudaFree(d_m7);

    auto start_time_merge = std::chrono::high_resolution_clock::now();
    // Launch the CUDA kernel to merge submatrices
    dim3 threadsPerBlock(subRowsA, subRowsA);
    mergeSubmatrices<<<16,threadsPerBlock>>>(d_C11, d_C12, d_C21, d_C22, d_C, subRowsA, rowsA);

    //cudaMemcpy(C, d_C, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);
    //convert d_C to deviceTensor
    torch::Tensor deviceTensor = torch::from_blob(d_C, {rowsA,colsB});

    C= deviceTensor.to(torch::kCPU);
    auto end_time_merge = std::chrono::high_resolution_clock::now();

    auto elapsed_time_mem_alloc = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_mem_alloc  - end_time_split)+std::chrono::duration_cast<std::chrono::milliseconds>(start_time_split - start_time_mem_alloc);
    auto elapsed_time_split = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_split  - start_time_split);
    auto elapsed_time_sgeam = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_sgeam - end_time_mem_alloc);
    auto elapsed_time_sgemm = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_sgemm - start_time_sgemm);
    //auto elapsed_time_recursive = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_recursive   - start_time_recursive );
    auto elapsed_time_kernels = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_kernels - start_time_kernels );
    auto elapsed_time_merge = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_merge - start_time_merge);
    auto elapsed_time_free = std::chrono::duration_cast<std::chrono::milliseconds>(start_time_merge - end_time_kernels)+std::chrono::duration_cast<std::chrono::milliseconds>(start_time_kernels  - start_time_free);

    std::cout << "Elapsed time mem alloc: " << elapsed_time_mem_alloc.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time split: " << elapsed_time_split.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time sgeam: " << elapsed_time_sgeam.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time sgemm: " << elapsed_time_sgemm.count() << " milliseconds" << std::endl;
    //std::cout << "Elapsed time recursive: " << elapsed_time_recursive.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time kernels: " << elapsed_time_kernels.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time merge: " << elapsed_time_merge.count() << " milliseconds" << std::endl;
    std::cout << "Elapsed time free: " << elapsed_time_free.count() << " milliseconds" << std::endl;

}

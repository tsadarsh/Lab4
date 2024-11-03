// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const float* A, const float* B, float* C, int N) 
{
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "blocksPerGrid = " << blocksPerGrid << std::endl;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

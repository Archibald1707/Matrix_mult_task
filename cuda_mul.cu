#ifndef CUDA_MUL_H
#define CUDA_MUL_H

#include "Matrix.h"
#include "FlatMatrix.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Matrix multiply_cuda(const Matrix& A, const Matrix& B) {
    int N = A.rows;
    size_t size = N * N * sizeof(double);

    double* h_A = new double[N * N];
    double* h_B = new double[N * N];
    double* h_C = new double[N * N];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = A[i][j];
            h_B[i * N + j] = B[i][j];
        }

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    Matrix C(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = h_C[i * N + j];

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

__global__ void matrixMulKernelFlat(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

FlatMatrix multiply_flat_cuda(const FlatMatrix& A, const FlatMatrix& B) {
    int N = A.rows;
    size_t size = N * N * sizeof(double);

    const double* h_A = A.data.data();
    const double* h_B = B.data.data();

    double* d_A;
    double* d_B;
    double* d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    matrixMulKernelFlat<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    double* h_C = new double[N * N];
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    FlatMatrix C(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C(i, j) = h_C[i * N + j];

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

__global__ void matrixMulKernelTransposed(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[col * N + k];
        }
        C[row * N + col] = sum;
    }
}

FlatMatrix multiply_flat_transposed_cuda(FlatMatrix& A, FlatMatrix& B) {
    int N = A.rows;
    size_t size = N * N * sizeof(double);

    const double* h_A = A.data.data();
    const double* h_B = B.data.data();
    double* h_C = new double[N * N];

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
    matrixMulKernelTransposed<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    FlatMatrix C(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C(i, j) = h_C[i * N + j];

    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include "header/uvp.h"

// CUDA Kernel to compute F
__global__ void calculate_f_kernel(double Re, double GX, double gamma, double dt, double dx, double dy, int imax, int jmax, double *U, double *V, double *F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= imax - 1 && j <= jmax) {
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        double du2dx = 1.0 / dx * ((U[i * jmax + j] + U[(i + 1) * jmax + j]) * (U[i * jmax + j] + U[(i + 1) * jmax + j]) / 4.0 - (U[(i - 1) * jmax + j] + U[i * jmax + j]) * (U[(i - 1) * jmax + j] + U[i * jmax + j]) / 4.0) + gamma * 1.0 / dx * (fabs(U[i * jmax + j] + U[(i + 1) * jmax + j]) * (U[i * jmax + j] - U[(i + 1) * jmax + j]) / 4.0 - fabs(U[(i - 1) * jmax + j] + U[i * jmax + j]) * (U[(i - 1) * jmax + j] - U[i * jmax + j]) / 4.0);
        double duvdy = 1.0 / dy * ((V[i * jmax + j] + V[(i + 1) * jmax + j]) * (U[i * jmax + j] + U[i * jmax + j + 1]) / 4.0 - (V[i * jmax + j - 1] + V[(i + 1) * jmax + j - 1]) * (U[i * jmax + j - 1] + U[i * jmax + j]) / 4.0) + gamma * 1.0 / dy * (fabs(V[i * jmax + j] + V[(i + 1) * jmax + j]) * (U[i * jmax + j] - U[i * jmax + j + 1]) / 4.0 - fabs(V[i * jmax + j - 1] + V[(i + 1) * jmax + j - 1]) * (U[i * jmax + j - 1] - U[i * jmax + j]) / 4.0);
        double laplace_u = (U[(i + 1) * jmax + j] - 2.0 * U[i * jmax + j] + U[(i - 1) * jmax + j]) / dx2 + (U[i * jmax + j + 1] - 2.0 * U[i * jmax + j] + U[i * jmax + j - 1]) / dy2;
        F[i * jmax + j] = U[i * jmax + j] + dt * ((1.0 / Re) * laplace_u - du2dx - duvdy + GX);
    }
}

// CUDA Kernel to compute G
__global__ void calculate_g_kernel(double Re, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double *U, double *V, double *G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i <= imax && j <= jmax - 1) {
        double dx2 = dx * dx;
        double dy2 = dy * dy;
        double duvdx = 1.0 / dx * ((U[i * jmax + j] + U[i * jmax + j + 1]) * (V[i * jmax + j] + V[(i + 1) * jmax + j]) / 4.0 - (U[(i - 1) * jmax + j] + U[(i - 1) * jmax + j + 1]) * (V[(i - 1) * jmax + j] + V[i * jmax + j]) / 4.0) + gamma * 1.0 / dx * (fabs(U[i * jmax + j] + U[i * jmax + j + 1]) * (V[i * jmax + j] - V[(i + 1) * jmax + j]) / 4.0 - fabs(U[(i - 1) * jmax + j] + U[(i - 1) * jmax + j + 1]) * (V[(i - 1) * jmax + j] - V[i * jmax + j]) / 4.0);
        double dv2dy = 1.0 / dy * ((V[i * jmax + j] + V[i * jmax + j + 1]) * (V[i * jmax + j] + V[i * jmax + j + 1]) / 4.0 - (V[i * jmax + j - 1] + V[i * jmax + j]) * (V[i * jmax + j - 1] + V[i * jmax + j]) / 4.0) + gamma * 1.0 / dy * (fabs(V[i * jmax + j] + V[i * jmax + j + 1]) * (V[i * jmax + j] - V[i * jmax + j + 1]) / 4.0 - fabs(V[i * jmax + j - 1] + V[i * jmax + j]) * (V[i * jmax + j - 1] - V[i * jmax + j]) / 4.0);
        double laplace_v = (V[(i + 1) * jmax + j] - 2.0 * V[i * jmax + j] + V[(i - 1) * jmax + j]) / dx2 + (V[i * jmax + j + 1] - 2.0 * V[i * jmax + j] + V[i * jmax + j - 1]) / dy2;
        G[i * jmax + j] = V[i * jmax + j] + dt * ((1.0 / Re) * laplace_v - duvdx - dv2dy + GY);
    }
}

// Host function to calculate F and G using CUDA
extern "C" void calculate_fg(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G) {
    double *d_U, *d_V, *d_F, *d_G;
    int size = (imax + 2) * (jmax + 2) * sizeof(double);

    // Allocate device memory
    cudaMalloc((void**)&d_U, size);
    cudaMalloc((void**)&d_V, size);
    cudaMalloc((void**)&d_F, size);
    cudaMalloc((void**)&d_G, size);

    // Copy host data to device
    cudaMemcpy(d_U, U[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, G[0], size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imax + threadsPerBlock.x - 1) / threadsPerBlock.x, (jmax + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernels
    calculate_f_kernel<<<numBlocks, threadsPerBlock>>>(Re, GX, gamma, dt, dx, dy, imax, jmax, d_U, d_V, d_F);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete before proceeding
    calculate_g_kernel<<<numBlocks, threadsPerBlock>>>(Re, GY, gamma, dt, dx, dy, imax, jmax, d_U, d_V, d_G);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete before proceeding

    // Check for any errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host
    cudaMemcpy(F[0], d_F, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(G[0], d_G, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(U[0], d_U, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(V[0], d_V, size, cudaMemcpyDeviceToHost);

    // Free device memory`
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_F);
    cudaFree(d_G);
}

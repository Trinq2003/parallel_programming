#include "cuda_kernels.h"
#include <stdio.h>
#include <math.h>

__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void calculate_fg_kernel(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double *U, double *V, double *F, double *G, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double dx2 = dx * dx;
    double dy2 = dy * dy;

    int color = (i % 2) + (j % 2) * 2;

    if (color != phase) return;

    // Calculate F
    if (i > 0 && i < imax && j > 0 && j <= jmax) {
        double du2dx = 1.0 / dx * ((U[i * (jmax + 2) + j] + U[(i + 1) * (jmax + 2) + j]) * (U[i * (jmax + 2) + j] + U[(i + 1) * (jmax + 2) + j]) / 4.0 -
                                   (U[(i - 1) * (jmax + 2) + j] + U[i * (jmax + 2) + j]) * (U[(i - 1) * (jmax + 2) + j] + U[i * (jmax + 2) + j]) / 4.0) +
                        gamma * 1.0 / dx * (fabs(U[i * (jmax + 2) + j] + U[(i + 1) * (jmax + 2) + j]) * (U[i * (jmax + 2) + j] - U[(i + 1) * (jmax + 2) + j]) / 4.0 -
                                            fabs(U[(i - 1) * (jmax + 2) + j] + U[i * (jmax + 2) + j]) * (U[(i - 1) * (jmax + 2) + j] - U[i * (jmax + 2) + j]) / 4.0);
        double duvdy = 1.0 / dy * ((V[i * (jmax + 2) + j] + V[(i + 1) * (jmax + 2) + j]) * (U[i * (jmax + 2) + j] + U[i * (jmax + 2) + j + 1]) / 
4.0 - (V[i * (jmax + 2) + j - 1] + V[(i + 1) * (jmax + 2) + j - 1]) * (U[i * (jmax + 2) + j - 1] + U[i * (jmax + 2) + j]) / 4.0) +
                        gamma * 1.0 / dy * (fabs(V[i * (jmax + 2) + j] + V[(i + 1) * (jmax + 2) + j]) * (U[i * (jmax + 2) + j] - U[i * (jmax + 2) + j + 1]) / 4.0 -
                                            fabs(V[i * (jmax + 2) + j - 1] + V[(i + 1) * (jmax + 2) + j - 1]) * (U[i * (jmax + 2) + j - 1] - U[i * (jmax + 2) + j]) / 4.0);
        double laplace_u = (U[(i + 1) * (jmax + 2) + j] - 2.0 * U[i * (jmax + 2) + j] + U[(i - 1) * (jmax + 2) + j]) / dx2 +
                           (U[i * (jmax + 2) + j + 1] - 2.0 * U[i * (jmax + 2) + j] + U[i * (jmax + 2) + j - 1]) / dy2;
        F[i * (jmax + 2) + j] = U[i * (jmax + 2) + j] + dt * ((1.0 / Re) * laplace_u - du2dx - duvdy + GX);
    }

    // Calculate G
    if (i > 0 && i <= imax && j > 0 && j < jmax) {
        double duvdx = 1.0 / dx * ((U[i * (jmax + 2) + j] + U[i * (jmax + 2) + j + 1]) * (V[i * (jmax + 2) + j] + V[(i + 1) * (jmax + 2) + j]) / 4.0 -
                                   (U[(i - 1) * (jmax + 2) + j] + U[(i - 1) * (jmax + 2) + j + 1]) * (V[(i - 1) * (jmax + 2) + j] + V[i * (jmax + 2) + j]) / 4.0) +
                        gamma * 1.0 / dx * (fabs(U[i * (jmax + 2) + j] + U[i * (jmax + 2) + j + 1]) * (V[i * (jmax + 2) + j] - V[(i + 1) * (jmax + 2) + j]) / 4.0 -
                                            fabs(U[(i - 1) * (jmax + 2) + j] + U[(i - 1) * (jmax + 2) + j + 1]) * (V[(i - 1) * (jmax + 2) + j + 1] - V[i * (jmax + 2) + j]) / 4.0);
        double dv2dy = 1.0 / dy * ((V[i * (jmax + 2) + j] + V[i * (jmax + 2) + j + 1]) * (V[i * (jmax + 2) + j] + V[i * (jmax + 2) + j + 1]) / 4.0 -
                                   (V[i * (jmax + 2) + j - 1] + V[i * (jmax + 2) + j]) * (V[i * (jmax + 2) + j - 1] + V[i * (jmax + 2) + j]) / 4.0) +
                        gamma * 1.0 / dy * (fabs(V[i * (jmax + 2) + j] + V[i * (jmax + 2) + j + 1]) * (V[i * (jmax + 2) + j] - V[i * (jmax + 2) + j + 1]) / 4.0 -
                                            fabs(V[i * (jmax + 2) + j - 1] + V[i * (jmax + 2) + j]) * (V[i * (jmax + 2) + j - 1] - V[i * (jmax + 2) + j]) / 4.0);
        double laplace_v = (V[(i + 1) * (jmax + 2) + j] - 2.0 * V[i * (jmax + 2) + j] + V[(i - 1) * (jmax + 2) + j]) / dx2 +
                           (V[i * (jmax + 2) + j + 1] - 2.0 * V[i * (jmax + 2) + j] + V[i * (jmax + 2) + j - 1]) / dy2;
        G[i * (jmax + 2) + j] = V[i * (jmax + 2) + j] + dt * ((1.0 / Re) * laplace_v - duvdx - dv2dy + GY);
    }
}

__global__ void sor_kernel(double omg, double dx, double dy, int imax, int jmax, double *P, double *RS, double *residual, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ double s_residual;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_residual = 0.0;
    }
    __syncthreads();

    double rdx2 = 1.0 / (dx * dx);
    double rdy2 = 1.0 / (dy * dy);
    double beta = 1.0 / (2.0 * (rdx2 + rdy2));

    int color = (i % 2) + (j % 2) * 2;

    if (color != phase) return;

    if (i > 0 && i <= imax && j > 0 && j <= jmax) {
        double p_old = P[i * (jmax + 2) + j];
        P[i * (jmax + 2) + j] = (1.0 - omg) * p_old +
                                omg * beta * ((P[(i + 1) * (jmax + 2) + j] + P[(i - 1) * (jmax + 2) + j]) * rdx2 +
                                              (P[i * (jmax + 2) + j + 1] + P[i * (jmax + 2) + j - 1]) * rdy2 -
                                              RS[i * (jmax + 2) + j]);

        double res = (P[(i + 1) * (jmax + 2) + j] - 2.0 * P[i * (jmax + 2) + j] + P[(i - 1) * (jmax + 2) + j]) * rdx2 +
                     (P[i * (jmax + 2) + j + 1] - 2.0 * P[i * (jmax + 2) + j] + P[i * (jmax + 2) + j - 1]) * rdy2 -
                     RS[i * (jmax + 2) + j];
        atomicAddDouble(&s_residual, res);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAddDouble(residual, s_residual);
    }
}

void calculate_fg_cuda(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double *U, double *V, double *F, double *G) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imax + threadsPerBlock.x - 1) / threadsPerBlock.x, (jmax + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int phase = 0; phase < 4; ++phase) {
        calculate_fg_kernel<<<numBlocks, threadsPerBlock>>>(Re, GX, GY, gamma, dt, dx, dy, imax, jmax, U, V, F, G, phase);
        cudaDeviceSynchronize();
    }
}

void sor_cuda(double omg, double dx, double dy, int imax, int jmax, double *P, double *RS, double *residual) {
    dim

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "header/init.h"
#include "header/uvp.h"
#include "header/boundary_val.h"
#include "header/sor.h"
#include "header/visual.h"
#include "header/helper.h"
#include "cuda/cuda_kernels.h"

// Macro to check CUDA errors
#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if (e != cudaSuccess) {                                          \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
               cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

int main() {
    double xlength = 1.0, ylength = 1.0;
    int imax = 50, jmax = 50;
    double t = 0, tend = 50.0, dt = 0.05, tau = 0.5, dt_value = 0.5;
    double eps = 0.001, omg = 1.7, gamma = 0.5;
    double Re = 100, GX = 0.0, GY = 0.0;
    double UI = 0.0, VI = 0.0, PI = 0.0;
    int itermax = 100;
    double dx = xlength / imax, dy = ylength / jmax;
    double **U, **V, **P, **F, **G, **RS;
    int n = 0, it = 0;
    double res;

    // Memory allocation on host
    U = matrix(0, imax + 1, 0, jmax + 1);
    V = matrix(0, imax + 1, 0, jmax + 1);
    P = matrix(0, imax + 1, 0, jmax + 1);
    F = matrix(0, imax + 1, 0, jmax + 1);
    G = matrix(0, imax + 1, 0, jmax + 1);
    RS = matrix(0, imax + 1, 0, jmax + 1);

    // Memory allocation on device
    double *d_U, *d_V, *d_P, *d_F, *d_G, *d_RS, *d_residual;
    size_t size = (imax + 2) * (jmax + 2) * sizeof(double);
    cudaMalloc((void**)&d_U, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_V, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_P, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_F, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_G, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_RS, size);
    // cudaCheckError();
    cudaMalloc((void**)&d_residual, sizeof(double));
    // cudaCheckError();

    // Initialization
    init_uvp(UI, VI, PI, imax, jmax, U, V, P);
    cudaMemcpy(d_U, U[0], size, cudaMemcpyHostToDevice);
    // cudaCheckError();
    cudaMemcpy(d_V, V[0], size, cudaMemcpyHostToDevice);
    // cudaCheckError();
    cudaMemcpy(d_P, P[0], size, cudaMemcpyHostToDevice);
    // cudaCheckError();

    while (t < tend) {
        calculate_dt(Re, tau, &dt, dx, dy, imax, jmax, U, V);
        boundaryvalues(imax, jmax, U, V, P, F, G);

        // Copy updated boundary values to the device
        cudaMemcpy(d_U, U[0], size, cudaMemcpyHostToDevice);
        // cudaCheckError();
        cudaMemcpy(d_V, V[0], size, cudaMemcpyHostToDevice);
        // cudaCheckError();
        cudaMemcpy(d_F, F[0], size, cudaMemcpyHostToDevice);
        // cudaCheckError();
        cudaMemcpy(d_G, G[0], size, cudaMemcpyHostToDevice);
        // cudaCheckError();

        // Calculate F and G on the device
        calculate_fg_cuda(Re, GX, GY, gamma, dt, dx, dy, imax, jmax, d_U, d_V, d_F, d_G);

        // Copy results back to host
        cudaMemcpy(F[0], d_F, size, cudaMemcpyDeviceToHost);
        // cudaCheckError();
        cudaMemcpy(G[0], d_G, size, cudaMemcpyDeviceToHost);
        // cudaCheckError();

        calculate_rs(dt, dx, dy, imax, jmax, F, G, RS);
        cudaMemcpy(d_RS, RS[0], size, cudaMemcpyHostToDevice);
        // cudaCheckError();

        it = 0;
        do {
            sor_cuda(omg, dx, dy, imax, jmax, d_P, d_RS, d_residual);
            cudaMemcpy(&res, d_residual, sizeof(double), cudaMemcpyDeviceToHost);
            // cudaCheckError();
            res = sqrt(fmax(res, 0.0) / (imax * jmax));
            it++;
        } while (it < itermax && res > eps);

        // Copy updated pressure values back to the host for the next time step
        cudaMemcpy(P[0], d_P, size, cudaMemcpyDeviceToHost);
        // cudaCheckError();

        calculate_uv(dt, dx, dy, imax, jmax, U, V, F, G, P);

        if (n % (int)(dt_value / dt) == 0) {
            write_vtkFile("cavity", n, xlength, ylength, imax, jmax, dx, dy, U, V, P);
        }

        t += dt;
        n++;
    }

    // Free memory on host
    free_matrix(U, 0, imax + 1, 0, jmax + 1);
    free_matrix(V, 0, imax + 1, 0, jmax + 1);
    free_matrix(P, 0, imax + 1, 0, jmax + 1);
    free_matrix(F, 0, imax + 1, 0, jmax + 1);
    free_matrix(G, 0, imax + 1, 0, jmax + 1);
    free_matrix(RS, 0, imax + 1, 0, jmax + 1);

    // Free memory on device
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_P);
    cudaFree(d_F);
    cudaFree(d_G);
    cudaFree(d_RS);
    cudaFree(d_residual);

    return 0;
}

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void calculate_fg_cuda(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double *U, double *V, double *F, double *G);
void sor_cuda(double omg, double dx, double dy, int imax, int jmax, double *P, double *RS, double *residual);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H

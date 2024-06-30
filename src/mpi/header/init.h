#ifndef INIT_H
#define INIT_H

void init_uvp(double UI, double VI, double PI, int imax, int jmax, double **U, double **V, double **P);
double **allocate_2d_array(int nx, int ny);

#endif // INIT_H
#include "header/init.h"
#include <stdlib.h>

void init_uvp(double UI, double VI, double PI, int imax, int jmax, double **U, double **V, double **P) {
    int i, j;
    for (i = 0; i <= imax + 1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
            U[i][j] = UI;
            V[i][j] = VI;
            P[i][j] = PI;
        }
    }
}

double **allocate_2d_array(int nx, int ny) {
    double **array = (double **)malloc(nx * sizeof(double *));
    for (int i = 0; i < nx; i++) {
        array[i] = (double *)malloc(ny * sizeof(double));
    }
    return array;
}


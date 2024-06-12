#include "header/init.h"

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

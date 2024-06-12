#include "header/helper.h"
#include <stdlib.h>

double **matrix(int nrl, int nrh, int ncl, int nch) {
    double **m = (double **)malloc((nrh - nrl + 1) * sizeof(double *));
    for (int i = nrl; i <= nrh; i++) {
        m[i] = (double *)malloc((nch - ncl + 1) * sizeof(double));
    }
    return m;
}

void free_matrix(double **m, int nrl, int nrh, int ncl, int nch) {
    for (int i = nrl; i <= nrh; i++) {
        free(m[i]);
    }
    free(m);
}

#include "header/helper.h"
#include <stdlib.h>

double **matrix(int nrl, int nrh, int ncl, int nch) {
    int rows = nrh - nrl + 1;
    int cols = nch - ncl + 1;
    double **m = (double **)malloc(rows * sizeof(double *));
    m[0] = (double *)malloc(rows * cols * sizeof(double));
    for (int i = 1; i < rows; i++) {
        m[i] = m[0] + i * cols;
    }
    return m;
}

void free_matrix(double **m, int nrl, int nrh, int ncl, int nch) {
    free(m[0]);
    free(m);
}

double **allocate_2d_array(int rows, int cols) {
    double **array = (double **)malloc(rows * sizeof(double *));
    array[0] = (double *)malloc(rows * cols * sizeof(double));
    for (int i = 1; i < rows; i++) {
        array[i] = array[0] + i * cols;
    }
    return array;
}

void free_2d_array(double **array) {
    free(array[0]);
    free(array);
}
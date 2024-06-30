#ifndef HELPER_H
#define HELPER_H

double **matrix(int nrl, int nrh, int ncl, int nch);
void free_matrix(double **m, int nrl, int nrh, int ncl, int nch);

double **allocate_2d_array(int rows, int cols);
void free_2d_array(double **array);

#endif // HELPER_H

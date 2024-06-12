#include <math.h>
#include "header/sor.h"

void sor(double omg, double dx, double dy, int imax, int jmax, double **P, double **RS, double *res)
{
    int i, j;
    double rdx2 = 1.0 / (dx * dx);
    double rdy2 = 1.0 / (dy * dy);
    double beta = 1.0 / (2.0 * (rdx2 + rdy2));
    double residual = 0.0;

    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            P[i][j] = (1.0 - omg) * P[i][j] +
                      omg * beta * ((P[i + 1][j] + P[i - 1][j]) * rdx2 + (P[i][j + 1] + P[i][j - 1]) * rdy2 - RS[i][j]);
        }
    }

    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            residual += ((P[i + 1][j] - 2 * P[i][j] + P[i - 1][j]) * rdx2 +
                         (P[i][j + 1] - 2 * P[i][j] + P[i][j - 1]) * rdy2 -
                         RS[i][j]) *
                        ((P[i + 1][j] - 2 * P[i][j] + P[i - 1][j]) * rdx2 +
                         (P[i][j + 1] - 2 * P[i][j] + P[i][j - 1]) * rdy2 -
                         RS[i][j]);
        }
    }
    *res = sqrt(fmax(residual, 0.0) / (imax * jmax));
}

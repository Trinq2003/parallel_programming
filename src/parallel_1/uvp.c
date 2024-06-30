#include <math.h>
#include "header/uvp.h"

// void calculate_fg(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G)
// {
//     int i, j;
//     double dx2 = dx * dx;
//     double dy2 = dy * dy;

//     for (i = 1; i <= imax - 1; i++)
//     {
//         for (j = 1; j <= jmax; j++)
//         {
//             double du2dx = 1.0 / dx * ((U[i][j] + U[i + 1][j]) * (U[i][j] + U[i + 1][j]) / 4.0 - (U[i - 1][j] + U[i][j]) * (U[i - 1][j] + U[i][j]) / 4.0) + gamma * 1.0 / dx * (fabs(U[i][j] + U[i + 1][j]) * (U[i][j] - U[i + 1][j]) / 4.0 - fabs(U[i - 1][j] + U[i][j]) * (U[i - 1][j] - U[i][j]) / 4.0);
//             double duvdy = 1.0 / dy * ((V[i][j] + V[i + 1][j]) * (U[i][j] + U[i][j + 1]) / 4.0 - (V[i][j - 1] + V[i + 1][j - 1]) * (U[i][j - 1] + U[i][j]) / 4.0) + gamma * 1.0 / dy * (fabs(V[i][j] + V[i + 1][j]) * (U[i][j] - U[i][j + 1]) / 4.0 - fabs(V[i][j - 1] + V[i + 1][j - 1]) * (U[i][j - 1] - U[i][j]) / 4.0);
//             double laplace_u = (U[i + 1][j] - 2.0 * U[i][j] + U[i - 1][j]) / dx2 + (U[i][j + 1] - 2.0 * U[i][j] + U[i][j - 1]) / dy2;
//             F[i][j] = U[i][j] + dt * ((1.0 / Re) * laplace_u - du2dx - duvdy + GX);
//         }
//     }

//     for (i = 1; i <= imax; i++)
//     {
//         for (j = 1; j <= jmax - 1; j++)
//         {
//             double duvdx = 1.0 / dx * ((U[i][j] + U[i][j + 1]) * (V[i][j] + V[i + 1][j]) / 4.0 - (U[i - 1][j] + U[i - 1][j + 1]) * (V[i - 1][j] + V[i][j]) / 4.0) + gamma * 1.0 / dx * (fabs(U[i][j] + U[i][j + 1]) * (V[i][j] - V[i + 1][j]) / 4.0 - fabs(U[i - 1][j] + U[i - 1][j + 1]) * (V[i - 1][j + 1] - V[i][j]) / 4.0);
//             double dv2dy = 1.0 / dy * ((V[i][j] + V[i][j + 1]) * (V[i][j] + V[i][j + 1]) / 4.0 - (V[i][j - 1] + V[i][j]) * (V[i][j - 1] + V[i][j]) / 4.0) + gamma * 1.0 / dy * (fabs(V[i][j] + V[i][j + 1]) * (V[i][j] - V[i][j + 1]) / 4.0 - fabs(V[i][j - 1] + V[i][j]) * (V[i][j - 1] - V[i][j]) / 4.0);
//             double laplace_v = (V[i + 1][j] - 2.0 * V[i][j] + V[i - 1][j]) / dx2 + (V[i][j + 1] - 2.0 * V[i][j] + V[i][j - 1]) / dy2;
//             G[i][j] = V[i][j] + dt * ((1.0 / Re) * laplace_v - duvdx - dv2dy + GY);
//         }
//     }
// }

void calculate_rs(double dt, double dx, double dy, int imax, int jmax, double **F, double **G, double **RS)
{
    int i, j;
    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            RS[i][j] = ((F[i][j] - F[i - 1][j]) / dx + (G[i][j] - G[i][j - 1]) / dy) / dt;
        }
    }
}

void calculate_uv(double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G, double **P) {
    for (int i = 1; i <= imax - 1; i++) {
        for (int j = 1; j <= jmax; j++) {
            U[i][j] = F[i][j] - dt / dx * (P[i + 1][j] - P[i][j]);
        }
    }

    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j <= jmax - 1; j++) {
            V[i][j] = G[i][j] - dt / dy * (P[i][j + 1] - P[i][j]);
        }
    }
}


void calculate_dt(double Re, double tau, double *dt, double dx, double dy, int imax, int jmax, double **U, double **V) {
    double umax = 0.0;
    double vmax = 0.0;

    for (int i = 1; i <= imax; i++) {
        for (int j = 1; j <= jmax; j++) {
            if (fabs(U[i][j]) > umax) {
                umax = fabs(U[i][j]);
            }
            if (fabs(V[i][j]) > vmax) {
                vmax = fabs(V[i][j]);
            }
        }
    }

    double dt1 = Re / 2.0 * (dx * dx * dy * dy) / (dx * dx + dy * dy);
    double dt2 = dx / umax;
    double dt3 = dy / vmax;

    if (umax == 0) dt2 = 1e10;
    if (vmax == 0) dt3 = 1e10;

    *dt = tau * fmin(dt1, fmin(dt2, dt3));
}

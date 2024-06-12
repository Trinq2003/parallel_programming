#include "header/boundary_val.h"

void boundaryvalues(int imax, int jmax, double **U, double **V, double **P, double **F, double **G) {
    int i, j;

    // Apply boundary conditions for U
    for (i = 0; i <= imax + 1; i++) {
        U[i][0] = 0.0;               // Bottom boundary
        U[i][jmax + 1] = 0.0;        // Top boundary
    }

    // Apply boundary conditions for V
    for (j = 0; j <= jmax + 1; j++) {
        V[0][j] = 0.0;               // Left boundary
        V[imax + 1][j] = 0.0;        // Right boundary
    }

    // Additional boundary conditions for U
    for (i = 1; i <= imax; i++) {
        U[i][jmax + 1] = 2.0 - U[i][jmax]; // Top boundary specific condition
    }

    // Apply Neumann boundary conditions for P
    for (j = 1; j <= jmax; j++) {
        P[0][j] = P[1][j];           // Left boundary
        P[imax + 1][j] = P[imax][j]; // Right boundary
    }

    for (i = 1; i <= imax; i++) {
        P[i][0] = P[i][1];           // Bottom boundary
        P[i][jmax + 1] = P[i][jmax]; // Top boundary
    }

    // Apply Neumann boundary conditions for F and G
    for (j = 1; j <= jmax; j++) {
        F[0][j] = U[0][j];           // Left boundary
        F[imax][j] = U[imax][j];     // Right boundary
    }

    for (i = 1; i <= imax; i++) {
        G[i][0] = V[i][0];           // Bottom boundary
        G[i][jmax] = V[i][jmax];     // Top boundary
    }
}

#include <stdio.h>
#include <stdlib.h>
#include "header/init.h"
#include "header/uvp.h"
#include "header/boundary_val.h"
#include "header/sor.h"
#include "header/visual.h"
#include "header/helper.h"

int main() {
    double xlength = 1.0, ylength = 1.0;
    int imax = 50, jmax = 50;
    double t = 0, tend = 50.0, dt = 0.05, tau = 0.5, dt_value = 0.5;
    double eps = 0.001, omg = 1.7, gamma = 0.5;
    double Re = 100, GX = 0.0, GY = 0.0;
    double UI = 0.0, VI = 0.0, PI = 0.0;
    int itermax = 100;
    double dx = xlength / imax, dy = ylength / jmax;
    double **U, **V, **P, **F, **G, **RS;
    int n = 0, it = 0;
    double res;

    // Memory allocation
    U = matrix(0, imax + 1, 0, jmax + 1);
    V = matrix(0, imax + 1, 0, jmax + 1);
    P = matrix(0, imax + 1, 0, jmax + 1);
    F = matrix(0, imax + 1, 0, jmax + 1);
    G = matrix(0, imax + 1, 0, jmax + 1);
    RS = matrix(0, imax + 1, 0, jmax + 1);

    // Initialization
    init_uvp(UI, VI, PI, imax, jmax, U, V, P);

    while (t < tend) {
        calculate_dt(Re, tau, &dt, dx, dy, imax, jmax, U, V);
        boundaryvalues(imax, jmax, U, V, P, F, G);
        calculate_fg(Re, GX, GY, gamma, dt, dx, dy, imax, jmax, U, V, F, G);
        calculate_rs(dt, dx, dy, imax, jmax, F, G, RS);

        it = 0;
        do {
            sor(omg, dx, dy, imax, jmax, P, RS, &res);
            it++;
        } while (it < itermax && res > eps);

        calculate_uv(dt, dx, dy, imax, jmax, U, V, F, G, P);

        if (n % (int)(dt_value / dt) == 0) {
            write_vtkFile("cavity", n, xlength, ylength, imax, jmax, dx, dy, U, V, P);
        }

        t += dt;
        n++;
    }

    // Free memory
    free_matrix(U, 0, imax + 1, 0, jmax + 1);
    free_matrix(V, 0, imax + 1, 0, jmax + 1);
    free_matrix(P, 0, imax + 1, 0, jmax + 1);
    free_matrix(F, 0, imax + 1, 0, jmax + 1);
    free_matrix(G, 0, imax + 1, 0, jmax + 1);
    free_matrix(RS, 0, imax + 1, 0, jmax + 1);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "header/init.h"
#include "header/uvp.h"
#include "header/boundary_val.h"
#include "header/sor.h"
#include "header/visual.h"
#include "header/helper.h"

int main() {
    double xlength = 1.0, ylength = 1.0;
    int imax = 20, jmax = 20;
    double t = 0, tend = 5.0, dt = 0.05, tau = 0.5, dt_value = 0.5;
    double eps = 0.001, omg = 1.7, gamma = 0.5;
    double Re = 0.1, GX = 0.0, GY = 0.0;
    double UI = 0.0, VI = 0.0, PI = 0.0;
    int itermax = 100;
    double dx = xlength / imax, dy = ylength / jmax;
    double **U, **V, **P, **F, **G, **RS;
    int n = 0, it = 0;
    double res;

    // Memory allocation
    U = allocate_2d_array(imax + 2, jmax + 2);
    V = allocate_2d_array(imax + 2, jmax + 2);
    P = allocate_2d_array(imax + 2, jmax + 2);
    F = allocate_2d_array(imax + 2, jmax + 2);
    G = allocate_2d_array(imax + 2, jmax + 2);
    RS = allocate_2d_array(imax + 2, jmax + 2);

    // Initialization
    init_uvp(UI, VI, PI, imax, jmax, U, V, P);

    // Start timing
    clock_t start_time = clock();

    while (t < tend) {
        calculate_dt(Re, tau, &dt, dx, dy, imax, jmax, U, V);
        boundaryvalues(imax, jmax, U, V, P, F, G);
        calculate_fg(Re, GX, GY, gamma, dt, dx, dy, imax, jmax, U, V, F, G); // CUDA-accelerated function
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

    // End timing
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Time taken to complete: %f seconds\n", time_taken);

    // Free memory
    free_2d_array(U);
    free_2d_array(V);
    free_2d_array(P);
    free_2d_array(F);
    free_2d_array(G);
    free_2d_array(RS);

    return 0;
}

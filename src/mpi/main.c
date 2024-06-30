#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "header/init.h"
#include "header/uvp.h"
#include "header/boundary_val.h"
#include "header/sor.h"
#include "header/visual.h"
#include "header/helper.h"

#define NX 100  // Number of cells in x direction
#define NY 100  // Number of cells in y direction
#define TIMESTEPS 1000
void update_cells(double **P, int start_x, int end_x, int start_y, int end_y);
void exchange_boundaries(double **P, int grid_size_x, int grid_size_y, int rank, int size);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int grid_size_x = NX / 2;  // Assuming 2x2 grid of processors
    int grid_size_y = NY / 2;

    double **P = allocate_2d_array(NX, NY);
    // Initialize P and other variables here...

    for (int t = 0; t < TIMESTEPS; t++) {
        // Phase 1
        if (rank % 2 == 0) update_cells(P, 0, grid_size_x, 0, grid_size_y);
        MPI_Barrier(MPI_COMM_WORLD);
        // Exchange boundaries
        exchange_boundaries(P, grid_size_x, grid_size_y, rank, size);

        // Phase 2
        if (rank % 2 == 1) update_cells(P, 0, grid_size_x, 0, grid_size_y);
        MPI_Barrier(MPI_COMM_WORLD);
        // Exchange boundaries
        exchange_boundaries(P, grid_size_x, grid_size_y, rank, size);

        // Phase 3
        if (rank % 2 == 0) update_cells(P, grid_size_x, NX, grid_size_y, NY);
        MPI_Barrier(MPI_COMM_WORLD);
        // Exchange boundaries
        exchange_boundaries(P, grid_size_x, grid_size_y, rank, size);

        // Phase 4
        if (rank % 2 == 1) update_cells(P, grid_size_x, NX, grid_size_y, NY);
        MPI_Barrier(MPI_COMM_WORLD);
        // Exchange boundaries
        exchange_boundaries(P, grid_size_x, grid_size_y, rank, size);
    }

    // Clean up
    for (int i = 0; i < NX; i++) {
        free(P[i]);
    }
    free(P);

    MPI_Finalize();
    return 0;
}

void update_cells(double **P, int start_x, int end_x, int start_y, int end_y) {
    for (int i = start_x; i < end_x; i++) {
        for (int j = start_y; j < end_y; j++) {
            // Update P based on neighbors
            P[i][j] = (P[i-1][j] + P[i+1][j] + P[i][j-1] + P[i][j+1]) / 4.0;
        }
    }
}

void exchange_boundaries(double **P, int grid_size_x, int grid_size_y, int rank, int size) {
    MPI_Status status;
    int up, down, left, right;

    up = (rank - 2 + size) % size;
    down = (rank + 2) % size;
    left = (rank - 1 + size) % size;
    right = (rank + 1) % size;

    // Send and receive data from neighboring processes
    // Example for sending to the right and receiving from the left
    MPI_Sendrecv(&P[grid_size_x-1][0], grid_size_y, MPI_DOUBLE, right, 0, 
                 &P[0][0], grid_size_y, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &status);

    // Similarly, handle other directions (up, down, left, right)
}
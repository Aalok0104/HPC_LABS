#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#include <omp.h>

void interpolation(double *mesh_value, Points *points) {

    const int nx = GRID_X;
    const int ny = GRID_Y;
    const int total_size = nx * ny;

    const double dx = 1.0 / (nx - 1);
    const double dy = 1.0 / (ny - 1);

    // Reset mesh
    #pragma omp parallel for
    for (int idx = 0; idx < total_size; idx++) {
        mesh_value[idx] = 0.0;
    }

    int num_threads = omp_get_max_threads();

    double **local_mesh = (double **) malloc(num_threads * sizeof(double *));
    for (int t = 0; t < num_threads; t++) {
        local_mesh[t] = (double *) calloc(total_size, sizeof(double));
    }

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local = local_mesh[tid];

#pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {

            double xi = points[p].x;
            double yi = points[p].y;
            double fi = 1.0;

            int j = (int)(xi / dx);
            int i = (int)(yi / dy);

            if (j >= nx - 1) j = nx - 2;
            if (i >= ny - 1) i = ny - 2;

            double Xi = j * dx;
            double Yj = i * dy;

            double lx = xi - Xi;
            double ly = yi - Yj;

            double w_ij     = (dx - lx) * (dy - ly);
            double w_i1j    = lx * (dy - ly);
            double w_ij1    = (dx - lx) * ly;
            double w_i1j1   = lx * ly;

            double inv_area = 1.0 / (dx * dy);

            w_ij   *= inv_area;
            w_i1j  *= inv_area;
            w_ij1  *= inv_area;
            w_i1j1 *= inv_area;

            local[i * nx + j]           += w_ij * fi;
            local[i * nx + (j + 1)]     += w_i1j * fi;
            local[(i + 1) * nx + j]     += w_ij1 * fi;
            local[(i + 1) * nx + j + 1] += w_i1j1 * fi;
        }
    }

    
    #pragma omp parallel for
    for (int idx = 0; idx < total_size; idx++) {
        double sum = 0.0;
        for (int t = 0; t < num_threads; t++) {
            sum += local_mesh[t][idx];
        }
        mesh_value[idx] = sum;
    }

    // Free memory
    for (int t = 0; t < num_threads; t++) {
        free(local_mesh[t]);
    }
    free(local_mesh);
}

// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}

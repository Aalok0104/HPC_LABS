#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

extern int GRID_X, GRID_Y, NUM_Points;
extern double dx, dy;

double min_val, max_val;

// ================= INTERPOLATION =================
void interpolation(double *mesh_value, Points *points) {

    int size = GRID_X * GRID_Y;

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        mesh_value[i] = 0.0;

    int nthreads = omp_get_max_threads();

    double **local_grid = (double **)malloc(nthreads * sizeof(double *));
    for (int t = 0; t < nthreads; t++) {
        local_grid[t] = (double *)calloc(size, sizeof(double));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *grid = local_grid[tid];

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {

            if (points[p].is_void) continue;

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x / dx);
            int j = (int)(y / dy);

            if (i >= GRID_X - 1) i = GRID_X - 2;
            if (j >= GRID_Y - 1) j = GRID_Y - 2;

            double X = i * dx;
            double Y = j * dy;

            double lx = x - X;
            double ly = y - Y;

            double w00 = (dx - lx) * (dy - ly);
            double w10 = ly * (dx - lx);
            double w01 = lx * (dy - ly);
            double w11 = lx * ly;

            int idx = j * GRID_X + i;

            grid[idx] += w00;
            grid[idx + 1] += w01;
            grid[idx + GRID_X] += w10;
            grid[idx + GRID_X + 1] += w11;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        for (int t = 0; t < nthreads; t++) {
            mesh_value[i] += local_grid[t][i];
        }
    }

    for (int t = 0; t < nthreads; t++)
        free(local_grid[t]);
    free(local_grid);
}

// ================= NORMALIZATION =================
void normalization(double *mesh_value) {

    int size = GRID_X * GRID_Y;

    min_val = 1e18;
    max_val = -1e18;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < size; i++) {
        if (mesh_value[i] < min_val) min_val = mesh_value[i];
        if (mesh_value[i] > max_val) max_val = mesh_value[i];
    }

    double range = max_val - min_val;
    if (range == 0) range = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}

// ================= MOVER =================
void mover(double *mesh_value, Points *points) {

    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i >= GRID_X - 1) i = GRID_X - 2;
        if (j >= GRID_Y - 1) j = GRID_Y - 2;

        double X = i * dx;
        double Y = j * dy;

        double lx = x - X;
        double ly = y - Y;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = ly * (dx - lx);
        double w01 = lx * (dy - ly);
        double w11 = lx * ly;

        int idx = j * GRID_X + i;

        double F =
            w00 * mesh_value[idx] +
            w01 * mesh_value[idx + 1] +
            w10 * mesh_value[idx + GRID_X] +
            w11 * mesh_value[idx + GRID_X + 1];

        points[p].x += F * dx;
        points[p].y += F * dy;

        if (points[p].x < 0 || points[p].x > 1 ||
            points[p].y < 0 || points[p].y > 1) {
            points[p].is_void = true;
        }
    }
}

// ================= DENORMALIZATION =================
void denormalization(double *mesh_value) {

    int size = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0) range = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * range / 2.0 + min_val;
    }
}

// ================= VOID COUNT =================
long long int void_count(Points *points) {

    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }

    return voids;
}

// ================= SAVE MESH =================
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

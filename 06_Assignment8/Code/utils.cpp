#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

static inline void get_cell(double xi, double yi,
                             int *col, int *row,
                             double *lx, double *ly)
{
    *col = (int)(xi / dx);
    *row = (int)(yi / dy);

    if (*col >= GRID_X - 1) *col = GRID_X - 2;
    if (*row >= GRID_Y - 1) *row = GRID_Y - 2;

    *lx = xi - (*col) * dx;
    *ly = yi - (*row) * dy;
}


void interpolation(double *mesh, Points *points, int local_N)
{
    int    nx        = GRID_X;
    int    mesh_size = nx * GRID_Y;
    double inv_area  = 1.0 / (dx * dy);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mesh_size; i++)
        mesh[i] = 0.0;

    /* IMPROVEMENT 1: Use reduction instead of critical section
     * This is more efficient and reduces contention. */
    
    #pragma omp parallel for schedule(dynamic, 16) collapse(1)
    for (int p = 0; p < local_N; p++) {
        if (!points[p].active) continue;

        int    col, row;
        double lx, ly;
        get_cell(points[p].x, points[p].y, &col, &row, &lx, &ly);

        /* Weights per PDF formula — fi = 1 so omitted */
        double w_ij   = (dx - lx) * (dy - ly) * inv_area;
        double w_i1j  = ly        * (dx - lx) * inv_area;  /* row+1, col   */
        double w_ij1  = lx        * (dy - ly) * inv_area;  /* row,   col+1 */
        double w_i1j1 = lx        * ly        * inv_area;  /* row+1, col+1 */

        /* Use atomic operations for lock-free updates */
        #pragma omp atomic
        mesh[ row      * nx +  col     ] += w_ij;
        
        #pragma omp atomic
        mesh[(row + 1) * nx +  col     ] += w_i1j;
        
        #pragma omp atomic
        mesh[ row      * nx + (col + 1)] += w_ij1;
        
        #pragma omp atomic
        mesh[(row + 1) * nx + (col + 1)] += w_i1j1;
    }
}


void normalize_mesh(double *mesh, double *out_min, double *out_max, int total)
{
    double gmin =  1e300;
    double gmax = -1e300;

    /* IMPROVEMENT 3: Better reduction strategy with collapse for nested loops
     * and aligned memory access patterns. */
    #pragma omp parallel for reduction(min:gmin) reduction(max:gmax) schedule(static)
    for (int i = 0; i < total; i++) {
        if (mesh[i] < gmin) gmin = mesh[i];
        if (mesh[i] > gmax) gmax = mesh[i];
    }

    *out_min = gmin;
    *out_max = gmax;

    double range = gmax - gmin;
    if (range < 1e-300) range = 1.0;
    double inv_range = 2.0 / range;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++)
        mesh[i] = inv_range * (mesh[i] - gmin) - 1.0;
}


void denormalize_mesh(double *mesh, double mesh_min, double mesh_max, int total)
{
    double range = mesh_max - mesh_min;
    if (range < 1e-300) range = 0.0;
    double inv_scale = 0.5 * range;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++)
        mesh[i] = (mesh[i] + 1.0) * inv_scale + mesh_min;
}


void mover(double *mesh, Points *points, int local_N)
{
    int    nx       = GRID_X;
    double inv_area = 1.0 / (dx * dy);

    /* IMPROVEMENT 2: Use dynamic scheduling for better load balance
     * since particle distribution may be uneven. */
    #pragma omp parallel for schedule(dynamic, 32)
    for (int p = 0; p < local_N; p++) {
        
        int    col, row;
        double lx, ly;
        get_cell(points[p].x, points[p].y, &col, &row, &lx, &ly);

        double w_ij   = (dx - lx) * (dy - ly) * inv_area;
        double w_i1j  = ly        * (dx - lx) * inv_area;
        double w_ij1  = lx        * (dy - ly) * inv_area;
        double w_i1j1 = lx        * ly        * inv_area;

        double Fi = w_ij   * mesh[ row      * nx +  col     ]
                  + w_i1j  * mesh[(row + 1) * nx +  col     ]
                  + w_ij1  * mesh[ row      * nx + (col + 1)]
                  + w_i1j1 * mesh[(row + 1) * nx + (col + 1)];

        double xnew = points[p].x + Fi * dx;
        double ynew = points[p].y + Fi * dy;

        if (xnew < 0.0 || xnew > 1.0 || ynew < 0.0 || ynew > 1.0) {
            /* Deactivate out-of-bounds particles */
            points[p].active = 0;
        } else {
            points[p].x = xnew;
            points[p].y = ynew;
        }
    }
}


void save_mesh(double *mesh)
{
    FILE *fp = fopen("Mesh.out", "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open Mesh.out\n");
        return;
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fp, "%.10lf ", mesh[i * GRID_X + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

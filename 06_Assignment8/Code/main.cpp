#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

/* Global simulation parameters (defined here, extern in headers) */
int    GRID_X, GRID_Y, NX, NY;
int    NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0)
            printf("Usage: %s input.bin [threads]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (argc >= 3) {
        int threads = atoi(argv[2]);
        omp_set_num_threads(threads);
    }

    /* ---- Read header on rank 0 ---- */
    FILE *file = NULL;
    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) {
            fprintf(stderr, "Error: cannot open %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fread(&NX,         sizeof(int), 1, file);
        fread(&NY,         sizeof(int), 1, file);
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter,    sizeof(int), 1, file);
    }

    MPI_Bcast(&NX,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter,    1, MPI_INT, 0, MPI_COMM_WORLD);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx     = 1.0 / NX;
    dy     = 1.0 / NY;

    /* -----------------------------------------------------------------
     * Particle distribution — handle remainder so no particles dropped.
     * ----------------------------------------------------------------- */
    int base_N    = NUM_Points / size;
    int remainder = NUM_Points % size;
    int local_N   = base_N + (rank < remainder ? 1 : 0);

    /* Counts and displacements for MPI_Scatterv (bytes) */
    int *counts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int n     = base_N + (r < remainder ? 1 : 0);
            counts[r] = n * (int)sizeof(Points);
            displs[r] = offset;
            offset   += counts[r];
        }
    }
    int my_byte_count = local_N * (int)sizeof(Points);

    Points *local_points = (Points *)malloc(local_N * sizeof(Points));
    double *local_mesh   = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    double *global_mesh  = (double *)calloc(GRID_X * GRID_Y, sizeof(double));

    double total_interp_time = 0.0;
    double total_mover_time  = 0.0;

    /* -----------------------------------------------------------------
     * Main iteration loop.
     * Each iteration reads a fresh batch of NUM_Points particles from
     * the binary file (the file stores Maxiter consecutive batches).
     * ----------------------------------------------------------------- */
    Points *all_points = NULL;

	if (rank == 0) {
	    all_points = (Points *)malloc(NUM_Points * sizeof(Points));
	    read_points(file, all_points, NUM_Points);
	}

	// Scatter once
	MPI_Scatterv(all_points, counts, displs, MPI_BYTE,
		     local_points, my_byte_count, MPI_BYTE,
		     0, MPI_COMM_WORLD);

	if (rank == 0) free(all_points);

	
	double *temp_mesh = (double *)malloc(GRID_X * GRID_Y * sizeof(double));

	// THEN loop
	for (int iter = 0; iter < Maxiter; iter++) {
        #pragma omp parallel for schedule(static)
	for (int i = 0; i < GRID_X * GRID_Y; i++) {
	    local_mesh[i] = 0.0;
	    global_mesh[i] = 0.0;
	    temp_mesh[i] = 0.0;
	}

        /* --- Interpolation: local particles -> local mesh --- */
        double t0 = MPI_Wtime();
        interpolation(local_mesh, local_points, local_N);
        double t1 = MPI_Wtime();
        total_interp_time += (t1 - t0);

        /* --- Reduce local meshes to global mesh --- */
        MPI_Allreduce(local_mesh, global_mesh,
                      GRID_X * GRID_Y,
                      MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        /* --- Normalize global mesh to [-1, 1] --- */
        double g_min, g_max;
        normalize_mesh(global_mesh, &g_min, &g_max, GRID_X * GRID_Y);

        /* --- Mover: update local particle positions --- */
        double t2 = MPI_Wtime();
        mover(global_mesh, local_points, local_N);
        double t3 = MPI_Wtime();
        total_mover_time += (t3 - t2);

        /* --- Denormalize mesh back to original range --- */
        denormalize_mesh(global_mesh, g_min, g_max, GRID_X * GRID_Y);
    }

    free(temp_mesh);

    /* -----------------------------------------------------------------
     * Output
     * ----------------------------------------------------------------- */
    if (rank == 0) {
        save_mesh(global_mesh);
        printf("Total interpolation time = %.6lf sec\n", total_interp_time);
        printf("Total mover time         = %.6lf sec\n", total_mover_time);
        printf("Total time               = %.6lf sec\n",
               total_interp_time + total_mover_time);
    }

    free(local_points);
    free(local_mesh);
    free(global_mesh);
    if (rank == 0) {
        free(counts);
        free(displs);
        fclose(file);
    }

    MPI_Finalize();
    return 0;
}

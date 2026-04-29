#include <stdio.h>
#include <stdlib.h>
#include "init.h"

void initializepoints(Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x      = (double)rand() / RAND_MAX;
        points[i].y      = (double)rand() / RAND_MAX;
        points[i].active = 1;
    }
}

/* Read n particles from binary file. Format per particle: x (double), y (double). */
void read_points(FILE *file, Points *points, int n) {
    for (int i = 0; i < n; i++) {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
        points[i].active = 1;
    }
}

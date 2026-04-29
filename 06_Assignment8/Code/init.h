#ifndef INIT_H
#define INIT_H

#include <stdio.h>

typedef struct {
    double x, y;
    int    active;
} Points;

extern int    GRID_X, GRID_Y, NX, NY;
extern int    NUM_Points, Maxiter;
extern double dx, dy;

void read_points(FILE *file, Points *points, int n);
void initializepoints(Points *points);

#endif

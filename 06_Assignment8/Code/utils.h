#ifndef UTILS_H
#define UTILS_H

#include "init.h"

void interpolation(double *mesh_value, Points *points, int local_N);
void mover(double *mesh_value, Points *points, int local_N);
void normalize_mesh(double *mesh, double *out_min, double *out_max, int total);
void denormalize_mesh(double *mesh, double mesh_min, double mesh_max, int total);
void save_mesh(double *mesh_value);

#endif

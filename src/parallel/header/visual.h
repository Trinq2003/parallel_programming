#ifndef VISUAL_H
#define VISUAL_H

#include <stdio.h>

void write_vtkFile(char *szProblem, int timeStepNumber, double xlength, double ylength, int imax, int jmax, double dx, double dy, double **U, double **V, double **P);
void write_vtkFileHeader(FILE *fp, int imax, int jmax, double dx, double dy);
void write_vtkPointCoordinates(FILE *fp, int imax, int jmax, double dx, double dy);

#endif // VISUAL_H

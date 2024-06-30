#ifndef UVP_H
#define UVP_H

#ifdef __cplusplus
extern "C" {
#endif

void calculate_dt(double Re, double tau, double *dt, double dx, double dy, int imax, int jmax, double **U, double **V);
void calculate_fg(double Re, double GX, double GY, double gamma, double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G);
void calculate_rs(double dt, double dx, double dy, int imax, int jmax, double **F, double **G, double **RS);
void calculate_uv(double dt, double dx, double dy, int imax, int jmax, double **U, double **V, double **F, double **G, double **P);

#ifdef __cplusplus
}
#endif

#endif // UVP_H

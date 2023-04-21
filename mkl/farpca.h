#pragma once

#include "mkl.h"
#include "matrix_vector_functions.h"

double csr_norm_fro2(mat_csr *A);

double matrix_norm_fro2(mat* A);

void randQB_EI_PCA_dense(mat *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize);

void randQB_EI_PCA(mat_csr *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize);

void farPCA(mat_csr *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize);

void farPCA_dense(mat *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize);

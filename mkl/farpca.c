#include "farpca.h"
#include "math.h"

/*[L, ~] = lu(A) as in MATLAB*/
void LUfactorization(mat *A, mat *L)
{
    matrix_copy(L, A);
    MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*L->nrows);    
    LAPACKE_dgetrf (LAPACK_COL_MAJOR, L->nrows, L->ncols, L->d, L->nrows, ipiv);    
    long long i,j;
    #pragma omp parallel private(i,j) 
    {
    #pragma omp for     
        for(i=0;i<L->ncols;i++)
        {
            for(j=0;j<i;j++)
            {
                L->d[i*L->nrows+j] = 0;
            }
            L->d[i*L->nrows+i] = 1;
        }
    }
    
    {    
        for(i=L->ncols-1;i>=0;i--)
        {
            int ipi = ipiv[i]-1;
            for(j=0;j<L->ncols;j++)
            {
                double temp = L->d[j*L->nrows+ipi];
                L->d[j*L->nrows+ipi] = L->d[j*L->nrows+i];
                L->d[j*L->nrows+i] = temp;
            }
        }
    }
    free(ipiv);
}

void eigSVD(mat* A, mat *U, mat *S, mat *V)
{   
    matrix_transpose_matrix_mult(A, A, V);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', V->ncols, V->d, V->ncols, S->d);
    mat *V1 = matrix_new(V->ncols, V->ncols);
    matrix_copy(V1, V);
    MKL_INT i, j;
    #pragma omp parallel shared(V1,S) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V1->ncols; i++)
        {
            S->d[i] = sqrt(S->d[i]);
            for(j=0; j<V1->nrows;j++)
            {           
                V1->d[i*V1->nrows+j] /= S->d[i];
            }
        }
    }
    mat *Uc = matrix_new(U->nrows, U->ncols);
    matrix_matrix_mult(A, V1, Uc);
    matrix_copy(U, Uc);
    matrix_delete(Uc);
    matrix_delete(V1);
}

double csr_norm_fro2(mat_csr *A)
{
    double normA=0;
    MKL_INT i;
    #pragma omp parallel shared(A,normA) private(i) 
    {
    #pragma omp for reduction(+:normA)
    for(i=0;i<A->nnz;i++)
        normA += (A->values[i]*A->values[i]);
    }
    return normA;
}

double matrix_norm_fro2(mat *A){
    MKL_INT i;
    MKL_INT len = A->nrows;
    len *= A->ncols;
    double normA = 0;
    #pragma omp parallel shared(A,normA) private(i) 
    {
    #pragma omp for reduction(+:normA)
    for(i=0; i<len; i++){
        normA += A->d[i]*A->d[i];
    }
    }
    return normA;
}

void powerIteration(mat_csr *A, mat *Y, mat *W, mat *Yt, mat* Wt, mat *ZL)
{
    if(Y->ncols==0)
    {
        csr_matrix_matrix_mult(A, Wt, Yt);
        csr_matrix_transpose_matrix_mult(A, Yt, Wt);
    }
    else
    {
        mat *temp = matrix_new(Y->ncols, Yt->ncols);
        matrix_transpose_matrix_mult(W, Wt, temp);
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', ZL->nrows, temp->ncols, ZL->d, ZL->nrows, temp->d, temp->nrows);
        csr_matrix_matrix_mult(A, Wt, Yt);
        csr_matrix_transpose_matrix_mult(A, Yt, Wt);
        double alpha, beta;
        alpha = -1.0; beta = 1.0;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Y->nrows, temp->ncols, Y->ncols, alpha, Y->d, Y->nrows, temp->d, temp->nrows, beta, Yt->d, Yt->nrows);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, W->nrows, temp->ncols, W->ncols, alpha, W->d, W->nrows, temp->d, temp->nrows, beta, Wt->d, Wt->nrows);
        matrix_delete(temp);
    }
}

void shiftedPowerIteration(mat_csr *A, mat *Y, mat *W, mat *Yt, mat* Wt, mat *ZL, double alpha)
{
    if(Y->ncols==0)
    {
        csr_matrix_matrix_mult(A, Wt, Yt);
        csrAt_mult_B_minus_dC(A, Yt, Wt, alpha);
    }
    else
    {
        mat *temp = matrix_new(Y->ncols, Yt->ncols);
        matrix_transpose_matrix_mult(W, Wt, temp);
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', ZL->nrows, temp->ncols, ZL->d, ZL->nrows, temp->d, temp->nrows);
        csr_matrix_matrix_mult(A, Wt, Yt);
        csrAt_mult_B_minus_dC(A, Yt, Wt, alpha);
        double alpha, beta;
        alpha = -1.0; beta = 1.0;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, W->nrows, temp->ncols, W->ncols, alpha, W->d, W->nrows, temp->d, temp->nrows, beta, Wt->d, Wt->nrows);
        matrix_delete(temp);
    }
}

void shiftedPowerIteration_dense(mat *A, mat *Y, mat *W, mat *Yt, mat* Wt, mat *ZL, double alpha)
{
    if(Y->ncols==0)
    {
        matrix_matrix_mult(A, Wt, Yt);
        At_mult_B_minus_dC(A, Yt, Wt, alpha);
    }
    else
    {
        mat *temp = matrix_new(Y->ncols, Yt->ncols);
        matrix_transpose_matrix_mult(W, Wt, temp);
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', ZL->nrows, temp->ncols, ZL->d, ZL->nrows, temp->d, temp->nrows);
        matrix_matrix_mult(A, Wt, Yt);
        At_mult_B_minus_dC(A, Yt, Wt, alpha);
        double alpha, beta;
        alpha = -1.0; beta = 1.0;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, W->nrows, temp->ncols, W->ncols, alpha, W->d, W->nrows, temp->d, temp->nrows, beta, Wt->d, Wt->nrows);
        matrix_delete(temp);
    }
}


void matrix_combination(mat *ZT, mat *Z)
{
    MKL_INT i, j;
    for(i=0;i<Z->ncols;i++)
        for(j=0;j<Z->nrows;j++)
            ZT->d[i*ZT->nrows+j] = Z->d[i*Z->nrows+j];
    for(i=Z->nrows;i<ZT->nrows;i++)
        for(j=0;j<Z->ncols;j++)
            ZT->d[i+j*ZT->ncols] = ZT->d[j+i*ZT->ncols];
}

void SVD_producer(mat *Y, mat *W, mat *Z, mat *T, mat **U, mat **S, mat **V)
{
    mat *D = matrix_new(Z->ncols, 1);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', Z->ncols, Z->d, Z->ncols, D->d);
    mat *V1 = matrix_new(Z->ncols, Z->ncols);
    matrix_copy(V1, Z);
    MKL_INT i, j;
    #pragma omp parallel shared(V1,D) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V1->ncols; i++)
        {
            D->d[i] = sqrt(D->d[i]);
            for(j=0; j<V1->nrows;j++)
            {           
                V1->d[i*V1->nrows+j] /= D->d[i];
            }
        }
    }
    matrix_delete(Z);
    mat *Tt = matrix_new(T->nrows, T->ncols);
    matrix_transpose_matrix_mult(V1, T, Tt);
    matrix_matrix_mult(Tt, V1, T);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', T->ncols, T->d, T->ncols, D->d);
    mat *V2 = matrix_new(T->ncols, T->ncols);
    matrix_copy(V2, T);
    
    #pragma omp parallel shared(V2,D) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V2->ncols; i++)
        {
            D->d[i] = sqrt(D->d[i]);
            for(j=0; j<V2->nrows;j++)
            {           
                V2->d[i*V2->nrows+j] /= D->d[i];
            }
        }
    }
    
    mat *Vt = matrix_new(T->nrows, T->ncols);
    matrix_matrix_mult(V1, T, Vt);
    (*U) = matrix_new(Y->nrows, Y->ncols);
    matrix_matrix_mult(Y, Vt, *U);
    matrix_delete(Y);
    matrix_delete(T);
    (*V) = matrix_new(W->nrows, W->ncols);
    matrix_matrix_mult(V1, V2, Vt);
    matrix_matrix_mult(W, Vt, *V);
    matrix_delete(W);
    (*S) = matrix_new((*U)->ncols, 1);
    for(i=0; i<D->nrows; i++)
        (*S)->d[i] = D->d[i];
    matrix_delete(D);
    matrix_delete(V1);
    matrix_delete(V2);
    matrix_delete(Vt);
}

void randQB_EI_PCA(mat_csr *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize)
{
    if(maxsize % block != 0) maxsize = maxsize/block*block;
    int maxiter = maxsize/block;
    mat *Q = matrix_new(A->nrows, maxsize);
    mat *B = matrix_new(A->ncols, maxsize);
    mat *Qt = matrix_new(A->nrows, block);
    mat *Bt = matrix_new(A->ncols, block);
    mat *Y = matrix_new(A->nrows, block);
    mat *W = matrix_new(A->ncols, block);
    int i,j;
    Q->ncols = 0;
    B->ncols = 0;
    double normA = csr_norm_fro2(A);
    double temp = 0;
    VSLStreamStatePtr stream;
    vslNewStream( &stream, BRNG,  time(NULL) );
    for(i=0;i<maxiter;i++)
    {
        initialize_random_matrix_double(Bt, &stream);
        mat *T = matrix_new(Q->ncols, block);
        csr_matrix_matrix_mult(A, Bt, Qt);
        if (i > 0)
        {
           matrix_transpose_matrix_mult(B, Bt, T);
           matrix_matrix_mult(Q, T, Y);
           matrix_sub(Qt, Y); 
        }
        QR_factorization_getQ_inplace(Qt);
        if(p>0)
            for(j=1;j<=p;j++)
            {
                csr_matrix_transpose_matrix_mult(A, Qt, Bt);
                if (i > 0)
                {
                    matrix_transpose_matrix_mult(Q, Qt, T);
                    matrix_matrix_mult(B, T, W);
                    matrix_sub(Bt, W);
                }
                QR_factorization_getQ_inplace(Bt);
                csr_matrix_matrix_mult(A, Bt, Qt);
                if (i > 0)
                {
                    matrix_transpose_matrix_mult(B, Bt, T);
                    matrix_matrix_mult(Q, T, Y);
                    matrix_sub(Qt, Y); 
                }
                QR_factorization_getQ_inplace(Qt);
            }
        if (i > 0)
        {
            matrix_transpose_matrix_mult(Q, Qt, T);
            matrix_matrix_mult(Q, T, Y);
            matrix_sub(Qt, Y);
            QR_factorization_getQ_inplace(Qt);
        }
        csr_matrix_transpose_matrix_mult(A, Qt, Bt);
        matrix_delete(T);
        int inds[block];
        for (j = 0; j < block; ++j)
            inds[j] = j+i*block;
        Q->ncols += block;
        B->ncols += block;
        matrix_set_selected_columns(Q, inds, Qt);
        matrix_set_selected_columns(B, inds, Bt);
        temp += matrix_norm_fro2(Bt);
        if((normA-temp)<normA*tol*tol)
        {
            break;
        }
    }
    matrix_delete(Qt);
    matrix_delete(Bt);
    matrix_delete(Y);
    matrix_delete(W);
    (*U) = matrix_new(A->nrows, Q->ncols);
    (*S) = matrix_new(Q->ncols, 1);
    (*V) = matrix_new(A->ncols, Q->ncols);
    mat *T = matrix_new(Q->ncols, Q->ncols);
    eigSVD(B, *V, *S, T);
    matrix_delete(B);
    matrix_matrix_mult(Q, T, *U);
    matrix_delete(Q);
    matrix_delete(T);
}

void randQB_EI_PCA_dense(mat *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize)
{
    if(maxsize % block != 0) maxsize = maxsize/block*block;
    int maxiter = maxsize/block;
    mat *Q = matrix_new(A->nrows, maxsize);
    mat *B = matrix_new(A->ncols, maxsize);
    mat *Qt = matrix_new(A->nrows, block);
    mat *Bt = matrix_new(A->ncols, block);
    mat *Y = matrix_new(A->nrows, block);
    mat *W = matrix_new(A->ncols, block);
    int i,j;
    Q->ncols = 0;
    B->ncols = 0;
    double normA = matrix_norm_fro2(A);
    double temp = 0;
    VSLStreamStatePtr stream;
    vslNewStream( &stream, BRNG,  time(NULL) );
    for(i=0;i<maxiter;i++)
    {
        initialize_random_matrix_double(Bt, &stream);
        mat *T = matrix_new(Q->ncols, block);
        matrix_matrix_mult(A, Bt, Qt);
        if (i > 0)
        {
           matrix_transpose_matrix_mult(B, Bt, T);
           matrix_matrix_mult(Q, T, Y);
           matrix_sub(Qt, Y); 
        }
        QR_factorization_getQ_inplace(Qt);
        if(p>0)
            for(j=1;j<=p;j++)
            {
                matrix_transpose_matrix_mult(A, Qt, Bt);
                if (i > 0)
                {
                    matrix_transpose_matrix_mult(Q, Qt, T);
                    matrix_matrix_mult(B, T, W);
                    matrix_sub(Bt, W);
                }
                QR_factorization_getQ_inplace(Bt);
                matrix_matrix_mult(A, Bt, Qt);
                if (i > 0)
                {
                    matrix_transpose_matrix_mult(B, Bt, T);
                    matrix_matrix_mult(Q, T, Y);
                    matrix_sub(Qt, Y); 
                }
                QR_factorization_getQ_inplace(Qt);
            }
        if (i > 0)
        {
            matrix_transpose_matrix_mult(Q, Qt, T);
            matrix_matrix_mult(Q, T, Y);
            matrix_sub(Qt, Y);
            QR_factorization_getQ_inplace(Qt);
        }
        matrix_transpose_matrix_mult(A, Qt, Bt);
        matrix_delete(T);
        int inds[block];
        for (j = 0; j < block; ++j)
            inds[j] = j+i*block;
        Q->ncols += block;
        B->ncols += block;
        matrix_set_selected_columns(Q, inds, Qt);
        matrix_set_selected_columns(B, inds, Bt);
        temp += matrix_norm_fro2(Bt);
        if((normA-temp)<normA*tol*tol)
        {
            break;
        }
    }
    matrix_delete(Qt);
    matrix_delete(Bt);
    matrix_delete(Y);
    matrix_delete(W);
    (*U) = matrix_new(A->nrows, Q->ncols);
    (*S) = matrix_new(Q->ncols, 1);
    (*V) = matrix_new(A->ncols, Q->ncols);
    mat *T = matrix_new(Q->ncols, Q->ncols);
    eigSVD(B, *V, *S, T);
    matrix_delete(B);
    matrix_matrix_mult(Q, T, *U);
    matrix_delete(Q);
    matrix_delete(T);
}

void farPCA(mat_csr *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize)
{
    if(maxsize % block != 0) maxsize = maxsize/block*block;
    int maxiter = maxsize/block;
    mat *Y = matrix_new(A->nrows, maxsize);
    mat *W = matrix_new(A->ncols, maxsize);
    mat *Yt = matrix_new(A->nrows, block);
    mat *Wt = matrix_new(A->ncols, block);
    int i, j;
    Y->ncols = 0;
    W->ncols = 0;
    double normA = csr_norm_fro2(A);
    VSLStreamStatePtr stream;
    vslNewStream( &stream, BRNG,  time(NULL) );
    mat *Z;
    mat *T;
    mat *ZL;
    mat *St = matrix_new(block, 1);
    mat *Vt = matrix_new(block, block);
    double alpha = 0;
    
    for(i=0;i<maxiter;i++)
    {
        initialize_random_matrix_double(Wt, &stream);
        int inds[block];
        for(j=0;j<block;j++)
            inds[j] = i*block+j;
        alpha = 0;
        for(j=0;j<=p;j++)
        {
            if(j < p)
                shiftedPowerIteration(A, Y, W, Yt, Wt, ZL, alpha);
            if(j==p)
            { 
                csr_matrix_matrix_mult(A, Wt, Yt);
                csr_matrix_transpose_matrix_mult(A, Yt, Wt);
                break;
            }
            else
            {
                eigSVD(Wt, Wt, St, Vt);
                if (j>0 && alpha < St->d[0]) alpha = (alpha + St->d[0])/2;
            }
        }
        Y->ncols += block;
        matrix_set_selected_columns(Y, inds, Yt);
        W->ncols += block;
        matrix_set_selected_columns(W, inds, Wt);
        if(i>0)
        {
            mat* Z1 = matrix_new(Y->ncols, block);
            matrix_transpose_matrix_mult(Y, Yt, Z1);
            mat* ZT = matrix_new(Y->ncols, Y->ncols);
            matrix_set_selected_columns(ZT, inds, Z1);
            matrix_combination(ZT, Z);
            matrix_delete(Z);
            Z = ZT;
            matrix_transpose_matrix_mult(W, Wt, Z1);
            mat* TT = matrix_new(W->ncols, W->ncols);
            matrix_set_selected_columns(TT, inds, Z1);
            matrix_delete(Z1);
            matrix_combination(TT, T);
            matrix_delete(T);
            T = TT;
        }
        else
        {
            Z = matrix_new(Y->ncols, Y->ncols);
            matrix_transpose_matrix_mult(Y, Y, Z);
            T = matrix_new(W->ncols, W->ncols);
            matrix_transpose_matrix_mult(W, W, T);
        }
        mat *NT = matrix_new(Y->ncols, Y->ncols);
        if(i>0)
            matrix_delete(ZL);
        ZL = matrix_new(Z->nrows, Z->ncols);
        linear_solver(Z, T, NT, ZL);
        double normB = 0;
        for(j=0;j<NT->ncols;j++)
            normB += NT->d[j*NT->ncols+j];
        matrix_delete(NT);
        if(normA-normB < tol*tol*normA)
        {
            break;
        }
    }
    matrix_delete(Yt);
    matrix_delete(Wt);
    matrix_delete(St);
    matrix_delete(Vt);
    matrix_delete(ZL);
    SVD_producer(Y, W, Z, T, U, S, V);
}

void farPCA_dense(mat *A, mat **U, mat **S, mat **V, int block, int p, double tol, int maxsize)
{
    if(maxsize % block != 0) maxsize = maxsize/block*block;
    int maxiter = maxsize/block;
    mat *Y = matrix_new(A->nrows, maxsize);
    mat *W = matrix_new(A->ncols, maxsize);
    mat *Yt = matrix_new(A->nrows, block);
    mat *Wt = matrix_new(A->ncols, block);
    int i, j;
    Y->ncols = 0;
    W->ncols = 0;
    double normA = matrix_norm_fro2(A);
    
    VSLStreamStatePtr stream;
    vslNewStream( &stream, BRNG,  time(NULL) );
    mat *Z;
    mat *T;
    mat *ZL;
    mat *St = matrix_new(block, 1);
    mat *Vt = matrix_new(block, block);
    double alpha = 0;
    
    for(i=0;i<maxiter;i++)
    {
        initialize_random_matrix_double(Wt, &stream);
        int inds[block];
        for(j=0;j<block;j++)
            inds[j] = i*block+j;
        alpha = 0;
        for(j=0;j<=p;j++)
        {
            if(j < p)
                shiftedPowerIteration_dense(A, Y, W, Yt, Wt, ZL, alpha);
            if(j==p)
            { 
                matrix_matrix_mult(A, Wt, Yt);
                matrix_transpose_matrix_mult(A, Yt, Wt);
                break;
            }
            else
            {
                eigSVD(Wt, Wt, St, Vt);
                if (j>0 && alpha < St->d[0]) alpha = (alpha + St->d[0])/2;
            }
        }
        Y->ncols += block;
        matrix_set_selected_columns(Y, inds, Yt);
        W->ncols += block;
        matrix_set_selected_columns(W, inds, Wt);
        if(i>0)
        {
            mat* Z1 = matrix_new(Y->ncols, block);
            matrix_transpose_matrix_mult(Y, Yt, Z1);
            mat* ZT = matrix_new(Y->ncols, Y->ncols);
            matrix_set_selected_columns(ZT, inds, Z1);
            matrix_combination(ZT, Z);
            matrix_delete(Z);
            Z = ZT;
            matrix_transpose_matrix_mult(W, Wt, Z1);
            mat* TT = matrix_new(W->ncols, W->ncols);
            matrix_set_selected_columns(TT, inds, Z1);
            matrix_delete(Z1);
            matrix_combination(TT, T);
            matrix_delete(T);
            T = TT;
        }
        else
        {
            Z = matrix_new(Y->ncols, Y->ncols);
            matrix_transpose_matrix_mult(Y, Y, Z);
            T = matrix_new(W->ncols, W->ncols);
            matrix_transpose_matrix_mult(W, W, T);
        }
        mat *NT = matrix_new(Y->ncols, Y->ncols);
        if(i>0)
            matrix_delete(ZL);
        ZL = matrix_new(Z->nrows, Z->ncols);
        linear_solver(Z, T, NT, ZL);
        double normB = 0;
        for(j=0;j<NT->ncols;j++)
            normB += NT->d[j*NT->ncols+j];
        matrix_delete(NT);
        if(normA-normB < tol*tol*normA)
        {
            break;
        }
    }
    matrix_delete(Yt);
    matrix_delete(Wt);
    matrix_delete(St);
    matrix_delete(Vt);
    matrix_delete(ZL);
    SVD_producer(Y, W, Z, T, U, S, V);
}

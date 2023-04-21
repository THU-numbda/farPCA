/* high level matrix/vector functions using Intel MKL for blas */

#include "matrix_vector_functions.h"
#include "mkl_scalapack.h"


mat * matrix_new(int nrows, int ncols)
{
    mat *M = malloc(sizeof(mat));
    MKL_INT Size = nrows;
    Size *= ncols;
    M->d = (double*)calloc(Size, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}


vec * vector_new(int nrows)
{
    vec *v = malloc(sizeof(vec));
    v->d = (double*)calloc(nrows,sizeof(double));
    v->nrows = nrows;
    return v;
}


void matrix_delete(mat *M)
{
    free(M->d);
    free(M);
}


void vector_delete(vec *v)
{
    free(v->d);
    free(v);
}


void matrix_set_element(mat *M, int row_num, int col_num, double val){
    MKL_INT index = col_num;
    index *= M->nrows;
    index += row_num;
    M->d[index] = val;
}


double matrix_get_element(mat *M, int row_num, int col_num){
    MKL_INT index = col_num;
    index *= M->nrows;
    index += row_num;
    return M->d[index];
}


void vector_set_element(vec *v, int row_num, double val){
    v->d[row_num] = val;
}


double vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}


void matrix_print(mat * M){
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%.8f  ", val);
        }
        printf("\n");
    }
}


void vector_print(vec * v){
    int i;
    double val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%f\n", val);
    }
}


void matrix_copy(mat *D, mat *S){
    MKL_INT i;
    MKL_INT length = S->nrows;
    length *= S->ncols;
    //#pragma omp parallel for
    #pragma omp parallel shared(D,S) private(i) 
    {
    #pragma omp for 
    for(i=0; i<length; i++){
        D->d[i] = S->d[i];
    }
    }
}


void matrix_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_set_col(mat *M, int j, vec *column_vec){
    MKL_INT i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp for
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}


void matrix_get_col(mat *M, int j, vec *column_vec){
    MKL_INT i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp parallel for
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


void matrix_get_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        vector_set_element(row_vec,j,matrix_get_element(M,i,j));
    }
    }
}


void matrix_set_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        matrix_set_element(M,i,j,vector_get_element(row_vec,j));
    }
    }
}


void matrix_get_selected_rows(mat *M, int *inds, mat *Mr){
    int i;
    vec *row_vec; 
    #pragma omp parallel shared(M,Mr,inds) private(i,row_vec) 
    {
    #pragma omp parallel for
    for(i=0; i<(Mr->nrows); i++){
        row_vec = vector_new(M->ncols); 
        matrix_get_row(M,inds[i],row_vec);
        matrix_set_row(Mr,i,row_vec);
        vector_delete(row_vec);
    }
    }
}

double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval){
    long secs_used, micros_used;
    secs_used=(end_timeval.tv_sec - start_timeval.tv_sec);
    micros_used= ((secs_used*1000000) + end_timeval.tv_usec) - (start_timeval.tv_usec);
    return (micros_used/1e6); 
}


void initialize_random_matrix_double(mat *M, VSLStreamStatePtr* stream){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;
    float a=0.0,sigma=1.0;
    long long N = m;
    N *= n;
    //VSLStreamStatePtr stream;
    

    //vslNewStream( &stream, BRNG,  time(NULL) );
    vdRngGaussian( METHOD, *stream, N, M->d, a, sigma );
}


void matrix_get_selected_columns(mat *M, int *inds, mat *Mc){
    MKL_INT i;
    vec *col_vec = vector_new(M->nrows);
    for(i=0; i<(Mc->ncols); i++){
        matrix_get_col(M,inds[i],col_vec);
        matrix_set_col(Mc,i,col_vec);
    }
    vector_delete(col_vec);
}

/* M(:,inds) = Mc */
void matrix_set_selected_columns(mat *M, int *inds, mat *Mc){
    int i;
    vec *col_vec= vector_new(M->nrows);
    for(i=0; i<(Mc->ncols); i++){
        matrix_get_col(Mc,i,col_vec);
        matrix_set_col(M,inds[i],col_vec);
    }
    vector_delete(col_vec);
}


mat_coo* coo_matrix_new(int nrows, int ncols, int capacity) {
    mat_coo *M = (mat_coo*)malloc(sizeof(mat_coo));
    M->values = (double*)calloc(capacity, sizeof(double));
    M->rows = (MKL_INT*)calloc(capacity, sizeof(MKL_INT));
    M->cols = (MKL_INT*)calloc(capacity, sizeof(MKL_INT));
    M->nnz = 0;
    M->nrows = nrows; M->ncols = ncols;
    M->capacity = capacity;
    return M;
}

void compact_QR_factorization(mat *M, mat *Q, mat *R){
    int i,j,m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);

    mat *R_full = matrix_new(m,n);
    matrix_copy(R_full,M);
    //vec *tau = vector_new(n);
    vec *tau = vector_new(k);
    // get R
    //printf("get R..\n");
    //LAPACKE_dgeqrf(CblasColMajor, m, n, R_full->d, n, tau->d);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, R_full->nrows, R_full->ncols, R_full->d, R_full->ncols, tau->d);
    
    for(i=0; i<k; i++){
        for(j=0; j<k; j++){
            if(j>=i){
                matrix_set_element(R,i,j,matrix_get_element(R_full,i,j));
            }
        }
    }

    // get Q
    matrix_copy(Q,R_full); 
    //printf("dorgqr..\n");
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->nrows, Q->ncols, min(Q->ncols,Q->nrows), Q->d, Q->ncols, tau->d);


    // clean up
    matrix_delete(R_full);
    vector_delete(tau);
}

void linear_solve_Uxb(mat *A, mat *b) {
    LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N',  //unclear
        b->nrows,
        b->ncols, 
        A->d,
        A->ncols,
        b->d,
        b->ncols
    );
}

/*AX = B*/
void linear_solver2(mat* A, mat* B, mat* X)
{
    mat *Q = matrix_new(A->nrows, A->ncols);
    mat *R = matrix_new(A->ncols, A->ncols);
    compact_QR_factorization(A, Q, R);
    matrix_transpose_matrix_mult(Q, B, X);
    linear_solve_Uxb(R, X); //unclear
    matrix_delete(Q);
    matrix_delete(R);
}

int linear_solver(mat* A, mat* B, mat* X, mat *L)
{
    //L = matrix_new(A->nrows, A->ncols);
    matrix_copy(X, B);
    matrix_copy(L, A);
    int t = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', L->nrows, L->d, L->nrows);
    double tmax = L->d[0];
    double tmin = L->d[0];
    int i;
    for(i=1;i<A->nrows;i++)
    {
        if(L->d[i*A->nrows+i]>tmax) tmax = L->d[i*A->nrows+i];
        if(L->d[i*A->nrows+i]<tmin) tmin = L->d[i*A->nrows+i];
    }
    double ratio = tmax/tmin;
    //printf("ratio:%.10f, %.10f, %.10f\n", ratio, tmax, tmin);
    //printf("%.10f ", ratio);
    if(t>0) return 1;
    LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', L->nrows, X->ncols, L->d, L->nrows, X->d, X->nrows);
    //matrix_print(L);
    //matrix_delete(L);
    return 0;
}


void coo_matrix_delete(mat_coo *M) {
    free(M->values);
    free(M->cols);
    free(M->rows);
    free(M);
}


void coo_matrix_print(mat_coo *M) {
    int i;
    for (i = 0; i < M->nnz; i++) {
        printf("(%d, %d: %f), ", *(M->rows+i), *(M->cols+i), *(M->values+i));
    }
    printf("\n");
}


void csr_matrix_delete(mat_csr *M) {
    free(M->values);
    free(M->cols);
    free(M->pointerB);
    free(M->pointerE);
    free(M);
}


void csr_matrix_print(mat_csr *M) {
    int i;
    printf("values: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%f ", M->values[i]);
    }
    printf("\ncolumns: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%d ", M->cols[i]);
    }
    printf("\npointerB: ");
    for (i = 0; i < M->nrows; i++) {
        printf("%d\t", M->pointerB[i]);
    }
    printf("\npointerE: ");
    for (i = 0; i < M->nrows; i++) {
        printf("%d\t", M->pointerE[i]);
    }
    printf("\n");
}


mat_csr* csr_matrix_new() {
    mat_csr *M = (mat_csr*)malloc(sizeof(mat_csr));
    return M;
}


void csr_init_from_coo(mat_csr *D, mat_coo *M) {
    D->nrows = M->nrows; 
    D->ncols = M->ncols;
    D->pointerB = (MKL_INT*)malloc(D->nrows*sizeof(MKL_INT));
    D->pointerE = (MKL_INT*)malloc(D->nrows*sizeof(MKL_INT));
    D->cols = (MKL_INT*)calloc(M->nnz, sizeof(MKL_INT));
    D->nnz = M->nnz;
    
    D->values = (double*)malloc(M->nnz * sizeof(double));
    memcpy(D->values, M->values, M->nnz * sizeof(double));
    
    MKL_INT current_row, cursor=0;
    for (current_row = 0; current_row < D->nrows; current_row++) {
        D->pointerB[current_row] = cursor+1;
        while (cursor < M->nnz && M->rows[cursor]-1 == current_row) {
            D->cols[cursor] = M->cols[cursor];
            cursor++;
        }
        D->pointerE[current_row] = cursor+1;
    }
}


void csr_matrix_matrix_mult(mat_csr *A, mat *B, mat *C) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "N";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}


void csr_matrix_transpose_matrix_mult(mat_csr *A, mat *B, mat *C) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "T";
    double alpha = 1.0, beta = 0.0;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}


void At_mult_B_minus_dC(mat* A, mat* B, mat* C, double d)
{
    double alpha, beta;
    alpha = 1.0; beta = -1.0*d;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);

}

void csrAt_mult_B_minus_dC(mat_csr *A, mat* B, mat *C, double d) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "T";
    double alpha = 1.0, beta = -1.0*d;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}


void csrA_mult_B_minus_dC(mat_csr *A, mat* B, mat *C, double d) {
    /* void mkl_dcsrmm (
        const char *transa , const MKL_INT *m , const MKL_INT *n , 
        const MKL_INT *k , const double *alpha , const char *matdescra , 
        const double *val , const MKL_INT *indx , const MKL_INT *pntrb , 
        const MKL_INT *pntre , const double *b , const MKL_INT *ldb , 
        const double *beta , double *c , const MKL_INT *ldc );
    */
    char * transa = "N";
    double alpha = 1.0, beta = -1.0*d;
    const char *matdescra = "GXXF";
    mkl_dcsrmm(transa, &(A->nrows), &(C->ncols),
        &(A->ncols), &alpha, matdescra,
        A->values, A->cols, A->pointerB,
        A->pointerE, B->d, &(B->nrows),
        &beta, C->d, &(C->nrows));
}


void QR_factorization_getQ_inplace(mat *Q) {
    
    
    // printf("k1\n");
    MKL_INT i,j,m,n,k;
    m = Q->nrows; n = Q->ncols;
    k = min(m,n);
    //MKL_INT *jpvt = (MKL_INT*)malloc(n*sizeof(MKL_INT));
    vec *tau = vector_new(k);
    
    // check memory allocation
    // printf("k1b\n");
    // for (i=0; i++; i<m) {
    //     for (j=0; j++; j<n) {
    //         matrix_set_element(Q, i, j, matrix_get_element(Q, i, j));
    //     }
    // }
/* 
BUG DETECTED! the dgeqrf call raises segmentation fault occasionally.
the arguments passed to it seems to be fine. probably it's due to bug 
internal to MKL.

To reproduce the bug: call qr_bug_reproduce() in main.c 
*/ 
    // printf("k2 m=%d,n=%d,size=%d,tau=%d\n", m, n, sizeof(Q->d), k);
    // LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q->d, m, tau->d);
    //LAPACKE_dgeqpf(LAPACK_COL_MAJOR, m, n, Q->d, m, jpvt, tau->d);
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Q->d, m, tau->d);
    // printf("k2b\n");
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, Q->d, m, tau->d);
    // printf("k3\n");
    // clean up
    vector_delete(tau);
    //free(jpvt);
    // printf("k4\n");
    
}

void matrix_sub(mat *A, mat *B){
    MKL_INT i;
    MKL_INT len = A->nrows;
    len *= A->ncols;
    //#pragma omp parallel for
    #pragma omp parallel shared(A,B,len) private(i) 
    {
    #pragma omp for 
    for(i=0; i<len; i++){
        A->d[i] = A->d[i] - B->d[i];
    }
    }
}

void singular_value_decomposition(mat *M, mat *U, mat *S, mat *Vt){
    MKL_INT m,n,k;
    m = M->nrows; n = M->ncols;
    k = min(m,n);
    vec * work = vector_new(2*max(3*min(m, n)+max(m, n), 5*min(m,n)));

    LAPACKE_dgesvd( LAPACK_COL_MAJOR, 'S', 'S', m, n, M->d, m, S->d, U->d, m, Vt->d, k, work->d );

    vector_delete(work);
}

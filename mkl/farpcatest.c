#include "matrix_vector_functions.h"
#include "farpca.h"
#include "string.h"

MKL_INT m;
MKL_INT n;


void dense_test()
{
    FILE* fid=fopen("image1.dat", "r");
    m = 9504;
    n = 4752;
    mat* D = matrix_new(m, n);
    int i;
    unsigned char c;
    int len = m*n;
    for(i=0;i<m*n;i++)
    {
        fread(&c, sizeof(unsigned char), 1, fid);
        D->d[i] = 1.0*c/255;
    }
    //fread(D->d, sizeof(double), m*n, fid);
    struct timeval start_timeval, end_timeval;
    
    mat *U, *S, *V;
   
    double normA = matrix_norm_fro2(D); 
    double tol = 0.1;
    int b = 47;    
  
    //warm satrt 
    farPCA_dense(D, &U, &S, &V, b, 5, tol, 10*b);
    randQB_EI_PCA_dense(D, &U, &S, &V, b, 5, tol, 10*b);

    int j;
    printf("Result on Image, b = %d, tol = %f\n", b, tol);

    for(i=1;i<=5;i+=4)
    {
        gettimeofday(&start_timeval, NULL);
        farPCA_dense(D, &U, &S, &V, b, i, tol, 10*b);
        gettimeofday(&end_timeval, NULL);
    
        printf("farPCA p = %d\n", i);
        printf("Total Time: %f\n", get_seconds_frac(start_timeval,end_timeval));
        double temp = 0;
        for(j=S->nrows-1;j>=0;j--)
        {
            temp+= S->d[j]*S->d[j];
            if(normA-temp<normA*tol*tol)
            {
                printf("Initial rank k: %d\nTruncated rank r: %d\n\n", S->nrows, S->nrows-j);
                break;
            }
        }            

        matrix_delete(U);
        matrix_delete(S);
        matrix_delete(V);
    }
    for(i=1;i<=5;i+=4)
    {
        gettimeofday(&start_timeval, NULL);
        randQB_EI_PCA_dense(D, &U, &S, &V, b, i, tol, 10*b);
        gettimeofday(&end_timeval, NULL);
   
        printf("randQB_EI p = %d\n", i);
        printf("Total Time: %f\n", get_seconds_frac(start_timeval,end_timeval));
        double temp = 0;
        for(j=S->nrows-1;j>=0;j--)
        {
            temp+= S->d[j]*S->d[j];
            if(normA-temp<normA*tol*tol)
            {
                printf("Initial rank k: %d\nTruncated rank r: %d\n\n", S->nrows, S->nrows-j);
                break;
            }
        } 
 
        matrix_delete(U);
        matrix_delete(S);
        matrix_delete(V);
    }

}

void sparse_test()
{
    FILE* fid;
    fid = fopen("SNAP.dat", "r");
    m = 82168;
    n = 82168;
    int nnz = 948464;

    mat_coo *A = coo_matrix_new(m, n, nnz);

    A->nnz = nnz;
    long long i;
    for(i=0;i<A->nnz;i++)
    {
        int ii, jj;
        double kk;
        fscanf(fid, "%d %d %lf", &ii, &jj, &kk);
        A->rows[i] = (MKL_INT)ii;
        A->cols[i] = (MKL_INT)jj;
        A->values[i] = kk;
    }
    mat_csr* D = csr_matrix_new();
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);
    

    struct timeval start_timeval, end_timeval;
    
    mat *U, *S, *V;
    
    double normA = csr_norm_fro2(D);
    int j;
    int b = 821;
    double tol = 0.5;

    printf("Result on SNAP, b = %d, tol = %f\n", b, tol);

    for(i=1;i<=5;i+=4)
    {
        gettimeofday(&start_timeval, NULL);
        farPCA(D, &U, &S, &V, b, i, tol, 10*b);
        gettimeofday(&end_timeval, NULL);
    
        printf("farPCA p = %d\n", i);
        printf("Total Time: %f\n", get_seconds_frac(start_timeval,end_timeval));
        double temp = 0;
        for(j=S->nrows-1;j>=0;j--)
        {
            temp+= S->d[j]*S->d[j];
            if(normA-temp<normA*tol*tol)
            {
                printf("Initial rank k: %d\nTruncated rank r: %d\n\n", S->nrows, S->nrows-j);
                break;
            }
        }            

        matrix_delete(U);
        matrix_delete(S);
        matrix_delete(V);
    }
    for(i=1;i<=5;i+=4)
    {
        gettimeofday(&start_timeval, NULL);
        randQB_EI_PCA(D, &U, &S, &V, b, i, tol, 10*b);
        gettimeofday(&end_timeval, NULL);
   
        printf("randQB_EI p = %d\n", i);
        printf("Total Time: %f\n", get_seconds_frac(start_timeval,end_timeval));
        double temp = 0;
        for(j=S->nrows-1;j>=0;j--)
        {
            temp+= S->d[j]*S->d[j];
            if(normA-temp<normA*tol*tol)
            {
                printf("Initial rank k: %d\nTruncated rank r: %d\n\n", S->nrows, S->nrows-j);
                break;
            }
        } 
 
        matrix_delete(U);
        matrix_delete(S);
        matrix_delete(V);
    }
    
}

int main(int argc, char const *argv[])
{
    dense_test();
    //sparse_test();
    return 0;
}

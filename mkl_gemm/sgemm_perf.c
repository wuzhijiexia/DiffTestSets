//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2018 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example measures performance of computing the real matrix product 
*   C=alpha*A*B+beta*C using Intel(R) MKL function dgemm, where A, B, and C are 
*   matrices and alpha and beta are double precision scalars. 
*
*   In this simple example, practices such as memory management, data alignment, 
*   and I/O that are necessary for good programming style and high Intel(R) MKL 
*   performance are omitted to improve readability.
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 50

int main()
{
    float *A, *B, *C;
    int m, n, k;
    int i, j, r;
    float alpha, beta;
    double s_initial, s_elapsed;
    double m_flop;

    const int setNum = 8;
    int M[setNum] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int K[setNum] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int N[setNum] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    
    for(j = 0; j < setNum; j++){
        m = M[j];
        k = K[j];
        n = N[j];
        
        printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
                " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
        alpha = 1.0; beta = 0.0;

        // Allocating memory for matrices aligned on 64-byte boundary for better
        // performance
        A = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
        B = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
        C = (float *)mkl_malloc( m*n*sizeof( float ), 64 );
        if (A == NULL || B == NULL || C == NULL) {
            printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
            mkl_free(A);
            mkl_free(B);
            mkl_free(C);
            return 1;
        }

        for (i = 0; i < (m*k); i++) {
            A[i] = (float)(i+1);
        }

        for (i = 0; i < (k*n); i++) {
            B[i] = (float)(-i-1);
        }

        for (i = 0; i < (m*n); i++) {
            C[i] = 0.0;
        }

        // Making the first run of matrix product using Intel(R) MKL dgemm function
        // via CBLAS interface to get stable run time measurements
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, k, alpha, A, k, B, n, beta, C, n);

        // Measuring performance of matrix product using Intel(R) MKL dgemm function
        // via CBLAS interface
        s_initial = dsecnd();
        for (r = 0; r < LOOP_COUNT; r++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                        m, n, k, alpha, A, k, B, n, beta, C, n);
        }
        s_elapsed = (dsecnd() - s_initial) / LOOP_COUNT;
        m_flop = 1.0*m*(k+k-1)*n/1e3/1e3;

        printf (" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
                " == at %.3f milliseconds == "
                " == at %.3f gflops \n\n", (s_elapsed * 1000), m_flop/s_elapsed/1e3);
        
        // Deallocating memory
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }
    
    return 0;
}

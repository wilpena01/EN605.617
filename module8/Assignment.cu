#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <iostream>
#include "Utilities.h"

using namespace std;

 int  main () 
 {
    cublasInit();

    int H = 3;
    int W = H; 
    int HW = H*W;

    float *A = (float*)malloc(HW*sizeof(float));
    float *B = (float*)malloc(HW*sizeof(float));
    float *C = (float*)malloc(HW*sizeof(float));

    float *h_A = (float*)malloc(HW*sizeof(float));
    float *h_B = (float*)malloc(HW*sizeof(float));
    float *h_C = (float*)malloc(HW*sizeof(float));

    initMat(A,H,W); equalMat(h_A,A,HW);
    initMat(B,H,W); equalMat(h_B,B,HW);

    mulMat(h_A,h_B,H,W,h_C);
 
    float* g_A; float* g_B; float* g_C;

    /*ALLOCATE ON THE DEVICE*/
    cublasAlloc(HW,sizeof(float),(void**)&g_A);
    cublasAlloc(HW,sizeof(float),(void**)&g_B);
    cublasAlloc(HW,sizeof(float),(void**)&g_C);

    /*SET MATRIX*/
    cublasSetMatrix(H,W,sizeof(float),A,H,g_A,H);
    cublasSetMatrix(H,W,sizeof(float),B,H,g_B,H);
  
    /*KERNEL*/
    cublasSgemm('n','n',H,W,W,1,g_A,H,g_B,H,0,g_C,H);
    cublasGetError();
    cublasGetMatrix(H,W,sizeof(float),g_C,H,C,H);

    /* PERFORMANCE OUTPUT*/

    printf("\nMatriz A:\n");
    printMat(A,W,H);
    printf("\nMatriz B:\n");
    printMat(B,W,H);
    printf("\nMatriz C:\n");
    printMat(C,W,H);

    free( A );  
    free( B );  
    free ( C );
    cublasFree(g_A);
    cublasFree(g_B);
    cublasFree(g_C);

    /* Shutdown */
    cublasShutdown();

		return EXIT_SUCCESS;


  }

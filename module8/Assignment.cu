#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <iostream>
#include <chrono>
#include "Utilities.h"

using namespace std;
using namespace std::chrono;

void mulMatAnalysis(float *A, float *B, float *C, int H, int W)
{
    int HW = H*W;
    float *h_A = (float*)malloc(HW*sizeof(float));
    float *h_B = (float*)malloc(HW*sizeof(float));
    float *h_C = (float*)malloc(HW*sizeof(float));

    equalMat(h_A,A,HW); equalMat(h_B,B,HW);
    
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

    free( h_A );  
    free( h_B );
    free( h_C );
    cublasFree(g_A);
    cublasFree(g_B);
    cublasFree(g_C);

}
 int  main () 
 {
    cublasInit();
    int H = 3;
    int W = H; 
    int HW = H*W;

    float *A = (float*)malloc(HW*sizeof(float));
    float *B = (float*)malloc(HW*sizeof(float));
    float *C = (float*)malloc(HW*sizeof(float));

    initMat(A,H,W); 
    initMat(B,H,W); 

    auto start = high_resolution_clock::now();
    mulMatAnalysis(A,B,C,H,W);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    free( A ); 
    free( B ); 
    free( C );

    /* Shutdown */
    cublasShutdown();
    printf("time = %f\n",duration.count());

		return 0;


  }

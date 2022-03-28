#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <iostream>
#include <chrono>
#include <cufft.h>
#include "Utilities.h"

using namespace std;
using namespace std::chrono;

typedef float2 Complex;

void mulMatAnalysis(float *A, float *B, float *C, int H, int W)
{
    int HW = H*W;
    float *h_A = (float*)malloc(HW*sizeof(float));
    float *h_B = (float*)malloc(HW*sizeof(float));
    float *h_C = (float*)malloc(HW*sizeof(float));

    equalMat(h_A,A,HW); equalMat(h_B,B,HW);
    
    auto start = high_resolution_clock::now();
    mulMat(h_A,h_B,H,W,h_C);
    auto stop = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop - start);

    float* g_A; float* g_B; float* g_C;

    /*ALLOCATE ON THE DEVICE*/
    cublasAlloc(HW,sizeof(float),(void**)&g_A);
    cublasAlloc(HW,sizeof(float),(void**)&g_B);
    cublasAlloc(HW,sizeof(float),(void**)&g_C);

    /*SET MATRIX*/
    cublasSetMatrix(H,W,sizeof(float),A,H,g_A,H);
    cublasSetMatrix(H,W,sizeof(float),B,H,g_B,H);
  
    /*KERNEL*/
    start = high_resolution_clock::now();
    cublasSgemm('n','n',H,W,W,1,g_A,H,g_B,H,0,g_C,H);
    stop = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop - start);
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

    string str[] = {"cuBlas"};
    outputTime(duration1, duration2, str);

}

__global__ 
void ComplexMUL(Complex *a, Complex *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    a[i].x = a[i].x * b[i].x - a[i].y*b[i].y;
    a[i].y = a[i].x * b[i].y + a[i].y*b[i].x;
}

void test()
{

    int N = 5;
    int SIZE = N*N;


    Complex *fg = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fg[i].x = 1;
        fg[i].y = 0;
    }
    Complex *fig = new Complex[SIZE];
    for (int i = 0; i < SIZE; i++){
        fig[i].x = 1; // 
        fig[i].y = 0;
    }
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << fg[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
    for (int i = 0; i < N * N; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << fig[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;

    int mem_size = sizeof(Complex)* SIZE;


    cufftComplex *d_signal;
    checkCudaErrors(cudaMalloc((void **) &d_signal, mem_size)); 
    checkCudaErrors(cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice));

    cufftComplex *d_filter_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));
    checkCudaErrors(cudaMemcpy(d_filter_kernel, fig, mem_size, cudaMemcpyHostToDevice));

    // cout << d_signal[1].x << endl;
    // CUFFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);

    // Transform signal and filter
    printf("Transforming signal cufftExecR2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD);

    printf("Launching Complex multiplication<<< >>>\n");
    ComplexMUL <<< N, N >> >(d_signal, d_filter_kernel);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    Complex *result = new Complex[SIZE];
    cudaMemcpy(result, d_signal, sizeof(Complex)*SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i = i + N)
    {
        for (int j=0; j < N; j++){
            cout << result[i+j].x << " ";
        }
        cout << endl;
    }

    delete result, fg, fig;
    cufftDestroy(plan);
    //cufftDestroy(plan2);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);

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

    mulMatAnalysis(A,B,C,H,W);
    test();

    free( A ); 
    free( B ); 
    free( C );

    /* Shutdown */
    cublasShutdown();

		return 0;


  }

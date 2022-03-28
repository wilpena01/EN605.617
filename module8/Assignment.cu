#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <iostream>
#include <chrono>
#include <cufft.h>
#include "Utilities.h"

using namespace std;
using namespace std::chrono;

#define H  = 3;
#define W  = H;
#define HW = H*W;
typedef float2 Complex;

void mulMatAnalysis(float *A, float *B, float *C)
{
    
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

    free( h_A );  cublasFree(g_A);
    free( h_B );  cublasFree(g_B);
    free( h_C );  cublasFree(g_C);

    string str[] = {"cuBlas"};
    outputTime(duration1, duration2, str);
}

__global__ 
void ComplexMUL(Complex *mat1, Complex *mat2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    mat1[i].x = mat1[i].x * mat2[i].x - mat1[i].y*mat2[i].y;
    mat1[i].y = mat1[i].x * mat2[i].y + mat1[i].y*mat2[i].x;
}

void runcuFFT()
{
    Complex *fg = new Complex[HW];
    for (int i = 0; i < HW; i++){
        fg[i].x = 1;
        fg[i].y = 0;
    }
    Complex *fig = new Complex[HW];
    for (int i = 0; i < HW; i++){
        fig[i].x = 1; // 
        fig[i].y = 0;
    }
    for (int i = 0; i < H * W; i = i + H)
    {
        for (int j=0; j < W; j++){
            cout << fg[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;
    for (int i = 0; i < H * W; i = i + H)
    {
        for (int j=0; j < W; j++){
            cout << fig[i+j].x << " ";
        }
        cout << endl;
    }
    cout << "----------------" << endl;

    int mem_size = sizeof(Complex)* HW;


    cufftComplex *d_signal;
    cudaMalloc((void **) &d_signal, mem_size); 
    cudaMemcpy(d_signal, fg, mem_size, cudaMemcpyHostToDevice);

    cufftComplex *d_filter_kernel;
    cudaMalloc((void **)&d_filter_kernel, mem_size);
    cudaMemcpy(d_filter_kernel, fig, mem_size, cudaMemcpyHostToDevice);

    // cout << d_signal[1].x << endl;
    // CUFFT plan
    cufftHandle plan;
    cufftPlan2d(&plan, H, H, CUFFT_C2C);

    // Transform signal and filter
    printf("Transforming signal cufftExecR2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
    cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD);

    printf("Launching Complex multiplication<<< >>>\n");
    ComplexMUL <<< H, H >> >(d_signal, d_filter_kernel);

    // Transform signal back
    printf("Transforming signal back cufftExecC2C\n");
    cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);

    Complex *result = new Complex[HW];
    cudaMemcpy(result, d_signal, sizeof(Complex)*HW, cudaMemcpyDeviceToHost);

    for (int i = 0; i < HW; i = i + H)
    {
        for (int j=0; j < W; j++){
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

    float *A = (float*)malloc(HW*sizeof(float));
    float *B = (float*)malloc(HW*sizeof(float));
    float *C = (float*)malloc(HW*sizeof(float));

    initMat(A,H,W); 
    initMat(B,H,W); 

    mulMatAnalysis(A,B,C);
    runcuFFT();

    free( A ); 
    free( B ); 
    free( C );

    /* Shutdown */
    cublasShutdown();

		return 0;


  }

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

void printMat(float*P,int uWP,int uHP){
  //printf("\n %f",P[1]);
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",i);
  }
}

 int  main (int argc, char** argv) {
    cublasStatus status;
    int i,j;
    cublasInit();

    int H = 9, W=9;

    float *A = (float*)malloc(H*W*sizeof(float));
    float *B = (float*)malloc(H*W*sizeof(float));
    float *C = (float*)malloc(H*W*sizeof(float));
    if (A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (B == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
        return EXIT_FAILURE;
    }
    if (C == 0) {
      fprintf (stderr, "!!!! host memory allocation error (A)\n");
      return EXIT_FAILURE;
    }


    for (i=0;i<H;i++)
      for (j=0;j<W;j++)
        A[index(i,j,H)] = rand()%100; 
    for (i=0;i<H;i++)
      for (j=0;j<W;j++)
        B[index(i,j,H)] = rand()%100; 
 
    float* g_A; float* g_B; float* g_C;

    /*ALLOCATE ON THE DEVICE*/
    status=cublasAlloc(H*W,sizeof(float),(void**)&g_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(H*W,sizeof(float),(void**)&g_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasAlloc(H*W,sizeof(float),(void**)&g_C);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    /*SET MATRIX*/
    status=cublasSetMatrix(H,W,sizeof(float),A,H,g_A,H);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    status=cublasSetMatrix(H,W,sizeof(float),B,H,g_B,H);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (A)\n");
      return EXIT_FAILURE;
    }

    /*KERNEL*/
    cublasSgemm('n','n',H,W,W,1,g_A,H,g_B,H,0,g_C,H);

    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! kernel execution error.\n");
      return EXIT_FAILURE;
    }
    cublasGetMatrix(H,W,sizeof(float),g_C,H,C,H);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device read error (A)\n");
      return EXIT_FAILURE;
    }


    /* PERFORMANCE OUTPUT*/

    printf("\nMatriz A:\n");
    printMat(A,W,H);
    printf("\nMatriz B:\n");
    printMat(B,W,H);
    printf("\nMatriz C:\n");
    printMat(C,W,H);

    free( A );  free( B );  free ( C );
    status = cublasFree(g_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (A)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(g_B);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (B)\n");
      return EXIT_FAILURE;
    }
    status = cublasFree(g_C);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! memory free error (C)\n");
      return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    if (argc > 1) {
      if (!strcmp(argv[1], "-noprompt") ||!strcmp(argv[1], "-qatest") ){
        return EXIT_SUCCESS;
      }
    } 
    else{
      printf("\nPress ENTER to exit...\n");
      getchar();
    }

		return EXIT_SUCCESS;


  }

//
//  assignment3.cu
//  assignment3
//
//  Created by Wilson on 2/10/22.
//


#include <iostream>
#include <chrono>
#include <vector>
#include "Utilities.h"
#include "ADD.h"

using namespace std;
using namespace std::chrono;



void run_Funs(unsigned int *gpu_arr1, unsigned int *gpu_arr2, 
         unsigned int numBlocks, unsigned int blockSize)
{
	RESULT addR; const unsigned int ARRAY_SIZE = numBlocks * blockSize;
	
	Topadd(gpu_arr1, gpu_arr1, numBlocks, blockSize, &addR);
	output(&addR, ARRAY_SIZE);

}

void main_Pegeable(unsigned int totalThreads, unsigned int  blockSize, unsigned int numBlocks)
{
	const unsigned int ARRAY_SIZE = totalThreads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int *cpu_arr1, *cpu_arr2;

	cpu_arr1 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_arr2 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);
	
	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	
	cudaMalloc((void **)&gpu_arr1,      ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2,      ARRAY_SIZE_IN_BYTES);

	init(cpu_arr1, cpu_arr2, ARRAY_SIZE);	

	cudaMemcpy(gpu_arr1, cpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_arr2, cpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
					  
	run_Funs(gpu_arr1, gpu_arr2, numBlocks, blockSize);	

	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);								  
	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	free(cpu_arr1);
	free(cpu_arr2);
}

void main_Pinned(unsigned int totalThreads, unsigned int  blockSize, unsigned int numBlocks)
{
	const unsigned int ARRAY_SIZE = totalThreads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int *cpu_arr1, *cpu_arr2;

	cudaHostAlloc((unsigned int *)&cpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaHostAlloc((unsigned int *)&cpu_arr2, ARRAY_SIZE_IN_BYTES);

	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	
	cudaMalloc((void **)&gpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2, ARRAY_SIZE_IN_BYTES);

	init(cpu_arr1, cpu_arr2, ARRAY_SIZE);	

	cudaMemcpy(gpu_arr1, cpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_arr2, cpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
					  
	run_Funs(gpu_arr1, gpu_arr2, numBlocks, blockSize);	

	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);								  
	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	cudaFreeHost(cpu_arr1);
	cudaFreeHost(cpu_arr2);
}

int main(int argc, char** argv)
{
	// read command line arguments
	unsigned int totalThreads = 64;
	unsigned int blockSize    = 32;
	unsigned int numBlocks    = 8;
	
	if (argc >= 2) {
        
        sscanf(argv[1], "%d", &totalThreads);
	}
	if (argc >= 3) {
        sscanf(argv[2], "%d", &blockSize);
	}

	numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) 
	{
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		cout<<"Warning: Total thread count is not evenly divisible by the block size\n";
		cout<<"The total number of threads will be rounded up to "<< totalThreads<<endl;
	}
	
	main_Pegeable(totalThreads, blockSize, numBlocks);

	
	return EXIT_SUCCESS;
}
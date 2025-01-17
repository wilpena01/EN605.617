//
//  assignment4.cu
//  assignment4
//
//  Created by Wilson on 2/18/22.
//


#include <iostream>
#include <chrono>
#include <vector>
#include "Utilities.h"
#include "ADD.h"
#include "SUB.h"
#include "MUL.h"
#include "MOD.h"

using namespace std;
using namespace std::chrono;



void run_Funs(unsigned int *gpu_arr1, unsigned int *gpu_arr2, 
         unsigned int numBlocks, unsigned int blockSize)
{
	RESULT addR, subR, mulR, modR; 
	const unsigned int ARRAY_SIZE = numBlocks * blockSize;
	
	//Do the four mathematical calculation and output
	//the result
	Topadd(gpu_arr1, gpu_arr2, numBlocks, blockSize, &addR);
	Topsub(gpu_arr1, gpu_arr2, numBlocks, blockSize, &subR);
	Topmul(gpu_arr1, gpu_arr2, numBlocks, blockSize, &mulR);
	Topmod(gpu_arr1, gpu_arr2, numBlocks, blockSize, &modR); 
	cudaDeviceSynchronize();
	output(gpu_arr1, gpu_arr2, &addR, &subR, &mulR, &modR, ARRAY_SIZE);
}

void main_Pegeable(unsigned int totalThreads, unsigned int numBlocks, 
				   unsigned int blockSize)
{
	const unsigned int ARRAY_SIZE = totalThreads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int *cpu_arr1, *cpu_arr2, *gpu_arr1, *gpu_arr2;

	cpu_arr1 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_arr2 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);	
	cudaMalloc((void **)&gpu_arr1,      ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2,      ARRAY_SIZE_IN_BYTES);

	init(cpu_arr1, cpu_arr2, ARRAY_SIZE);	

	cudaMemcpy(gpu_arr1, cpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_arr2, cpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
					  
	run_Funs(gpu_arr1, gpu_arr2, numBlocks, blockSize);	
	
	//free GPU and CPU memory
	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);								  
	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	free(cpu_arr1);
	free(cpu_arr2);
}

void main_Pinned(unsigned int totalThreads, unsigned int numBlocks, 
				 unsigned int blockSize)
{
	const unsigned int ARRAY_SIZE = totalThreads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int *cpu_arr1, *cpu_arr2, *gpu_arr1, *gpu_arr2;

	cudaMallocHost((unsigned int **)&cpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMallocHost((unsigned int **)&cpu_arr2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2, ARRAY_SIZE_IN_BYTES);

	init(cpu_arr1, cpu_arr2, ARRAY_SIZE);	

	cudaMemcpy(gpu_arr1, cpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_arr2, cpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	run_Funs(gpu_arr1, gpu_arr2, numBlocks, blockSize);	

	//free GPU and CPU memory
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

	//launch the main_Pegleble() or main_main_Pinned()
	// and measure the execution time
	float delta1 = 0, delta2 = 0;
	cudaEvent_t start = get_time();	
	main_Pegeable(totalThreads, numBlocks, blockSize); 
	cudaEvent_t stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta1, start, stop);

	start = get_time();	
	main_Pinned(totalThreads, numBlocks, blockSize); 
	stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta2, start, stop);
	outputTime(delta1,delta2);


	return EXIT_SUCCESS;
}
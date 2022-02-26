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



void run_Funs(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	RESULT addR, subR, mulR, modR; 
	const UInt32 ARRAY_SIZE = numBlocks * blockSize;
	
	//Do the four mathematical calculation and output
	//the result
	Topadd(gpu_arr1, gpu_arr2, numBlocks, blockSize, &addR);
	Topsub(gpu_arr1, gpu_arr2, numBlocks, blockSize, &subR);
	Topmul(gpu_arr1, gpu_arr2, numBlocks, blockSize, &mulR);
	Topmod(gpu_arr1, gpu_arr2, numBlocks, blockSize, &modR); 
	cudaDeviceSynchronize();
	output(gpu_arr1, gpu_arr2, &addR, &subR, &mulR, &modR, ARRAY_SIZE);
}

void main_Pegeable(UInt32 totalThreads, UInt32 numBlocks, 
				   UInt32 blockSize)
{
	const UInt32 ARRAY_SIZE = totalThreads;
	UInt32 ARRAY_SIZE_IN_BYTES  = (sizeof(UInt32) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	UInt32 *cpu_arr1, *cpu_arr2, *gpu_arr1, *gpu_arr2;

	cpu_arr1 = (UInt32 *)malloc(ARRAY_SIZE_IN_BYTES);
	cpu_arr2 = (UInt32 *)malloc(ARRAY_SIZE_IN_BYTES);	
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

void main_Pinned(UInt32 totalThreads, UInt32 numBlocks, 
				 UInt32 blockSize)
{
	const UInt32 ARRAY_SIZE = totalThreads;
	UInt32 ARRAY_SIZE_IN_BYTES  = (sizeof(UInt32) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	UInt32 *cpu_arr1, *cpu_arr2, *gpu_arr1, *gpu_arr2;

	cudaMallocHost((UInt32 **)&cpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMallocHost((UInt32 **)&cpu_arr2, ARRAY_SIZE_IN_BYTES);
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
	UInt32 totalThreads = 12;
	UInt32 blockSize    = 12;
	UInt32 numBlocks    = 1;
	
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
	main_Pegeable(totalThreads, numBlocks, blockSize); 
	return EXIT_SUCCESS;
}
//
//  assignment5.cu
//  assignment5
//
//  Created by Wilson on 2/25/22.
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

RESULT run_add(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	//call the add module using stream and no stream
	//and record the time using events
	RESULT addR;
	float delta1, delta2;
	cudaEvent_t start = get_time();

	Topadd_stream(gpu_arr1, gpu_arr2, numBlocks, blockSize, &addR);
	
	cudaEvent_t stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta1, start, stop);

	start = get_time();

	Topadd(gpu_arr1, gpu_arr2, numBlocks, blockSize, &addR);
	stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta2, start, stop);
	cout<<"Addition Execution Time";
	outputTime(delta1,delta2);
	return addR;
}

RESULT run_sub(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	//call the sub module using stream and no stream
	//and record the time using events
	RESULT subR;
	float delta1, delta2;
	cudaEvent_t start = get_time();

	Topsub_stream(gpu_arr1, gpu_arr2, numBlocks, blockSize, &subR);
	
	cudaEvent_t stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta1, start, stop);

	start = get_time();

	Topsub(gpu_arr1, gpu_arr2, numBlocks, blockSize, &subR);
	stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta2, start, stop);
	cout<<"Subtraction Execution Time";
	outputTime(delta1,delta2);

	return subR;
}

RESULT run_mul(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	//call the mul module using stream and no stream
	//and record the time using events
	RESULT mulR;
	float delta1, delta2;
	cudaEvent_t start = get_time();

	Topmul_stream(gpu_arr1, gpu_arr2, numBlocks, blockSize, &mulR);
	
	cudaEvent_t stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta1, start, stop);

	start = get_time();

	Topmul(gpu_arr1, gpu_arr2, numBlocks, blockSize, &mulR);

	stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta2, start, stop);
	cout<<"Multiplication Execution Time";
	outputTime(delta1,delta2);

	return mulR;
}

RESULT run_mod(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	//call the mod module using stream and no stream
	//and record the time using events
	RESULT modR;
	float delta1, delta2;
	cudaEvent_t start = get_time();

	Topmod_stream(gpu_arr1, gpu_arr2, numBlocks, blockSize, &modR);
	
	cudaEvent_t stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta1, start, stop);

	start = get_time();

	Topmod(gpu_arr1, gpu_arr2, numBlocks, blockSize, &modR);
	stop = get_time();	
	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&delta2, start, stop);
	cout<<"Modulo Execution Time";
	outputTime(delta1,delta2);

	return modR;
}

void run_Funs(UInt32 *gpu_arr1, UInt32 *gpu_arr2, 
         UInt32 numBlocks, UInt32 blockSize)
{
	//call the four mathematical calculation and output
	//the result
	const UInt32 ARRAY_SIZE = numBlocks * blockSize;
	RESULT addR, subR, mulR, modR; 

	addR = run_add(gpu_arr1, gpu_arr2, numBlocks, blockSize);
	subR = run_sub(gpu_arr1, gpu_arr2, numBlocks, blockSize);
	mulR = run_mul(gpu_arr1, gpu_arr2, numBlocks, blockSize);
	modR = run_mod(gpu_arr1, gpu_arr2, numBlocks, blockSize);
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

int main()
{
	
	UInt32 totalThreads = 12;
	UInt32 blockSize    = 12;
	UInt32 numBlocks    = 1;

	main_Pinned(totalThreads, numBlocks, blockSize); 


	return EXIT_SUCCESS;
}
#ifndef ADD_H
#define ADD_H

#include "Utilities.h"

__global__
void add_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *Result,
			 unsigned int *Block, unsigned int *Thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	Result[thread_idx] = arr1[thread_idx] + arr2[thread_idx];
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;
}

void Topadd(unsigned int *gpu_arr1, unsigned int *gpu_arr2,unsigned int num_blocks, 
              unsigned int num_threads, RESULT *finalResult)
{
	const unsigned int ARRAY_SIZE     = num_blocks * num_threads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	unsigned int cpu_addResult[ARRAY_SIZE];
	unsigned int cpu_addBlock[ARRAY_SIZE];
	unsigned int cpu_addThread[ARRAY_SIZE];	
	
	unsigned int *gpu_addResult;
	unsigned int *gpu_addBlock;
	unsigned int *gpu_addThread;

	cudaMalloc((void **)&gpu_addResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addBlock,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addThread, ARRAY_SIZE_IN_BYTES);

	add_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_addResult, 
										 gpu_addBlock, gpu_addThread);

	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addBlock,  gpu_addBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addThread, gpu_addThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_addResult);
	cudaFree(gpu_addBlock);
	cudaFree(gpu_addThread);

	pushResult(cpu_addResult, cpu_addBlock, cpu_addThread, finalResult, ARRAY_SIZE);
}

#endif
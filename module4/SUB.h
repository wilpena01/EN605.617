#ifndef SUB_H
#define SUB_H

#include "Utilities.h"

__global__
void sub_arr(unsigned int *arr1, unsigned int *arr2, int *Result,
			 unsigned int *Block, unsigned int *Thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	Result[thread_idx] = static_cast<int>(arr1[thread_idx] - arr2[thread_idx]);
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;
}

void Topsub(unsigned int *gpu_arr1, unsigned int *gpu_arr2,unsigned int num_blocks, 
              unsigned int num_threads, RESULT *finalResult)
{
    //Preparation to do the subtraction in the kernel
	const unsigned int ARRAY_SIZE     = num_blocks * num_threads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
    unsigned int ARRAY_SIZE_IN_BYTES1 = (sizeof(unsigned int) * (ARRAY_SIZE));
	         int cpu_Result[ARRAY_SIZE];
	unsigned int cpu_Block[ARRAY_SIZE];
	unsigned int cpu_Thread[ARRAY_SIZE];	
	
	         int *gpu_Result;
	unsigned int *gpu_Block;
	unsigned int *gpu_Thread;

	cudaMalloc((void **)&gpu_Result, ARRAY_SIZE_IN_BYTES1);
	cudaMalloc((void **)&gpu_Block,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Thread, ARRAY_SIZE_IN_BYTES);

	sub_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_Result, 
										 gpu_Block, gpu_Thread);

	cudaMemcpy(cpu_Result, gpu_Result, ARRAY_SIZE_IN_BYTES1,cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Block,  gpu_Block,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Thread, gpu_Thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_Result);
	cudaFree(gpu_Block);
	cudaFree(gpu_Thread);

	pushResult(cpu_Result, cpu_Block, cpu_Thread, finalResult, ARRAY_SIZE);
}

#endif
#ifndef MUL_H
#define MUL_H

#include "Utilities.h"

__global__
void mul_arr(UInt32 *arr1, UInt32 *arr2, UInt32 *Result,
			 UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	Result[thread_idx] = static_cast<int>(arr1[thread_idx] * arr2[thread_idx]);
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;
}

__global__
void mul_arr_shared(UInt32 *arr1, UInt32 *arr2, UInt32 *Result,
			 UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ UInt32 g_input1;
	__shared__ UInt32 g_input2;

	copy_data_to_shared(arr1,arr2,g_input1,g_input2,thread_idx);
	
	Result[thread_idx] = g_input1 * g_input2;
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;

	__syncthreads();
}

void runKernerMul(UInt32 *gpu_arr1, UInt32 *gpu_arr2, UInt32 num_blocks, 
                  UInt32 num_threads, UInt32 *gpu_Result, UInt32 *gpu_Block,
			      UInt32 *gpu_Thread)
{

	float delta1 = 0, delta2=0;
	cudaEvent_t start1 = get_time();
	mul_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_Result, 
										 gpu_Block, gpu_Thread);
	cudaEvent_t stop1 = get_time();	
	cudaEventSynchronize(stop1);	
	cudaEventElapsedTime(&delta1, start1, stop1);


	cudaEvent_t start2 = get_time();
	mul_arr_shared<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_Result, 
										 gpu_Block, gpu_Thread);
	cudaEvent_t stop2 = get_time();	
	cudaEventSynchronize(stop2);	
	cudaEventElapsedTime(&delta2, start2, stop2);

	cout<<"Multiplication Elapse Time:\n";
	outputTime(delta1,delta2);

}

void Topmul(UInt32 *gpu_arr1, UInt32 *gpu_arr2,UInt32 num_blocks, 
              UInt32 num_threads, RESULT *finalResult)
{
    //Preparation to do the multiplication in the kernel
	const UInt32 ARRAY_SIZE     = num_blocks * num_threads;
	UInt32 ARRAY_SIZE_IN_BYTES  = (sizeof(UInt32) * (ARRAY_SIZE));
	UInt32 cpu_Result[ARRAY_SIZE], cpu_Block[ARRAY_SIZE], cpu_Thread[ARRAY_SIZE];	
	UInt32 *gpu_Result, *gpu_Block, *gpu_Thread;

	cudaMalloc((void **)&gpu_Result, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Block,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Thread, ARRAY_SIZE_IN_BYTES);

	runKernerMul(gpu_arr1, gpu_arr2, num_blocks, num_threads, gpu_Result, 
			     gpu_Block, gpu_Thread);

	cudaMemcpy(cpu_Result, gpu_Result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Block,  gpu_Block,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Thread, gpu_Thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_Result);
	cudaFree(gpu_Block);
	cudaFree(gpu_Thread);

	pushResult(cpu_Result, cpu_Block, cpu_Thread, finalResult, ARRAY_SIZE);
}

#endif
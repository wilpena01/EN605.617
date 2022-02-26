#ifndef SUB_H
#define SUB_H

#include "Utilities.h"
#include <string>

__global__
void sub_Const(UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	UInt32 result = (Input1 + thread_idx) - Input2;
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;	
}

__global__
void sub_literal(UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	UInt32 result = (5 + thread_idx) - 5;
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;	
}

__global__
void sub_arr(UInt32 *arr1, UInt32 *arr2, Int32 *Result,
			 UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	Result[thread_idx] = static_cast<int>(arr1[thread_idx] - arr2[thread_idx]);
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;
}

__global__
void sub_arr_shared(UInt32 *arr1, UInt32 *arr2, Int32 *Result,
			 UInt32 *Block, UInt32 *Thread)
{
	const UInt32 thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ UInt32 g_input1;
	__shared__ UInt32 g_input2;

	copy_data_to_shared(arr1,arr2,g_input1,g_input2,thread_idx);
	
	Result[thread_idx] = g_input1 - g_input2;
	Block[thread_idx]  = blockIdx.x;
	Thread[thread_idx] = threadIdx.x;

	__syncthreads();
}

void subRunsharedMem(UInt32 *gpu_arr1, UInt32 *gpu_arr2, UInt32 num_blocks, 
                  UInt32 num_threads, Int32 *gpu_Result, UInt32 *gpu_Block,
			      UInt32 *gpu_Thread)
{

	float delta1 = 0, delta2=0;
	cudaEvent_t start1 = get_time();
	sub_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_Result, 
										 gpu_Block, gpu_Thread);
	cudaEvent_t stop1 = get_time();	
	cudaEventSynchronize(stop1);	
	cudaEventElapsedTime(&delta1, start1, stop1);


	cudaEvent_t start2 = get_time();
	sub_arr_shared<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_Result, 
										 gpu_Block, gpu_Thread);
	cudaEvent_t stop2 = get_time();	
	cudaEventSynchronize(stop2);	
	cudaEventElapsedTime(&delta2, start2, stop2);

	string str[] ={"global", "shared"};
	outputTime(delta1,delta2, str);
}

void subRunConstMem(UInt32 num_blocks, UInt32 num_threads, 
				 UInt32 *gpu_Block, UInt32 *gpu_Thread)
{
	float delta1 = 0, delta2=0;
	cudaEvent_t start1 = get_time();
	sub_literal<<<num_blocks, num_threads>>>(gpu_Block, gpu_Thread);
	cudaEvent_t stop1 = get_time();	
	cudaEventSynchronize(stop1);	
	cudaEventElapsedTime(&delta1, start1, stop1);

	cudaEvent_t start2 = get_time();
	sub_Const<<<num_blocks, num_threads>>>(gpu_Block, gpu_Thread);
	cudaEvent_t stop2 = get_time();	
	cudaEventSynchronize(stop2);	
	cudaEventElapsedTime(&delta2, start2, stop2);

	string str[] ={"literal", "constant"};
	outputTime(delta1,delta2, str);

}

void Topsub(UInt32 *gpu_arr1, UInt32 *gpu_arr2,UInt32 num_blocks, 
              UInt32 num_threads, RESULT *finalResult)
{
    //Preparation to do the subtraction in the kernel
	const UInt32 ARRAY_SIZE     = num_blocks * num_threads;
	UInt32 ARRAY_SIZE_IN_BYTES  = (sizeof(UInt32) * (ARRAY_SIZE));
    UInt32 ARRAY_SIZE_IN_BYTES1 = (sizeof(UInt32) * (ARRAY_SIZE));
	Int32  cpu_Result[ARRAY_SIZE];
	UInt32 cpu_Block[ARRAY_SIZE], cpu_Thread[ARRAY_SIZE];	
	
	 Int32 *gpu_Result;
	UInt32 *gpu_Block, *gpu_Thread;

	cudaMalloc((void **)&gpu_Result, ARRAY_SIZE_IN_BYTES1);
	cudaMalloc((void **)&gpu_Block,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Thread, ARRAY_SIZE_IN_BYTES);

	cout<<"Addition Elapse Time:\n";
	subRunsharedMem(gpu_arr1, gpu_arr2, num_blocks, num_threads, gpu_Result, 
			     gpu_Block, gpu_Thread);
	subRunConstMem(num_blocks, num_threads, gpu_Block, gpu_Thread);
	cout<<"\n######################################\n";
	cudaMemcpy(cpu_Result, gpu_Result, ARRAY_SIZE_IN_BYTES1,cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Block,  gpu_Block,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_Thread, gpu_Thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_Result);
	cudaFree(gpu_Block);
	cudaFree(gpu_Thread);

	pushResult(cpu_Result, cpu_Block, cpu_Thread, finalResult, ARRAY_SIZE);
}

#endif
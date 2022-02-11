//
//  assignment3.cpp
//  assignment3
//
//  Created by Wilson on 2/10/22.
//


#include <iostream>
#include <time>

using namespace std;

#define ARRAY_SIZE 64
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
#define ARRAY_SIZE_IN_BYTES1 (sizeof(int) * (ARRAY_SIZE))


__global__
void init(unsigned int *arr1, unsigned int *arr2, 
		  unsigned int *r1, int *r2, unsigned int *r3, unsigned int *r4)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	arr1[thread_idx] = thread_idx;
	arr2[thread_idx] = thread_idx % 4;	
	
	r1[thread_idx]   = 0;
	r2[thread_idx]   = 0;
	r3[thread_idx]   = 0;
	r4[thread_idx]   = 0;

	
}
__global__
void add_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] + arr2[thread_idx];
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}


__global__
void sub_arr(unsigned int *arr1, unsigned int *arr2, int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] - arr2[thread_idx];
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

__global__
void mul_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] * arr2[thread_idx];
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

__global__
void mod_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(arr2[thread_idx]>0)
		result[thread_idx] = arr1[thread_idx] % arr2[thread_idx];
	else
		result[thread_idx] = 99999999;
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void main_sub0()
{

	/* Declare  statically two arrays of ARRAY_SIZE each */
	unsigned int cpu_arr1[ARRAY_SIZE];
	unsigned int cpu_arr2[ARRAY_SIZE];
	unsigned int cpu_addResult[ARRAY_SIZE];
         	 int cpu_subResult[ARRAY_SIZE];
	unsigned int cpu_mulResult[ARRAY_SIZE];
	unsigned int cpu_modResult[ARRAY_SIZE];
	unsigned int cpu_addBlock[ARRAY_SIZE];
	unsigned int cpu_addThread[ARRAY_SIZE];	
	unsigned int cpu_subBlock[ARRAY_SIZE];
	unsigned int cpu_subThread[ARRAY_SIZE];	
	unsigned int cpu_mulBlock[ARRAY_SIZE];
	unsigned int cpu_mulThread[ARRAY_SIZE];	
	unsigned int cpu_modBlock[ARRAY_SIZE];
	unsigned int cpu_modThread[ARRAY_SIZE];	


	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	unsigned int *gpu_addResult;
	         int *gpu_subResult;
	unsigned int *gpu_mulResult;
	unsigned int *gpu_modResult;
	unsigned *int gpu_addBlock[ARRAY_SIZE];
	unsigned *int gpu_addThread[ARRAY_SIZE];	
	unsigned *int gpu_subBlock[ARRAY_SIZE];
	unsigned *int gpu_subThread[ARRAY_SIZE];	
	unsigned *int gpu_mulBlock[ARRAY_SIZE];
	unsigned *int gpu_mulThread[ARRAY_SIZE];	
	unsigned *int gpu_modBlock[ARRAY_SIZE];
	unsigned *int gpu_modThread[ARRAY_SIZE];	

	cudaMalloc((void **)&gpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subResult, ARRAY_SIZE_IN_BYTES1);
	cudaMalloc((void **)&gpu_mulResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addBlock, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subBlock, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_mulBlock, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_mulThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modBlock, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modThread, ARRAY_SIZE_IN_BYTES);
		
	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addBlock, gpu_addBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addThread,gpu_addThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subBlock, gpu_subBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subThread,gpu_subThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulBlock, gpu_mulBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulThread,gpu_mulThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modBlock, gpu_modBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modThread,gpu_modThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);


	const unsigned int numthread_per_block = 16;
	const unsigned int num_blocks = ARRAY_SIZE/numthread_per_block;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Execute init kernel */
	init<<<num_blocks, num_threads>>>(gpu_arr1,      gpu_arr2, 
									  gpu_addResult, gpu_subResult,
									  gpu_mulResult, gpu_modResult);
									  
	/* Execute init kernel */
	add_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_addResult, 
										 gpu_addBlock, gpu_addThread);
										 
	sub_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_subResult, 
										 gpu_subBlock, gpu_subThread);
										 	
	mul_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_mulResult, 
										 gpu_mulBlock, gpu_mulThread);
										 								                
	mod_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_modResult, 
										 gpu_modBlock, gpu_modThread);
										 
										  
	/* Free the arrays on the GPU as now we're done with them */

	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addBlock, gpu_addBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addThread, gpu_addThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subBlock, gpu_subBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subThread, gpu_subThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulBlock, gpu_mulBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulThread, gpu_mulThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modBlock, gpu_modBlock, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modThread, gpu_modThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	cudaFree(gpu_addResult);
	cudaFree(gpu_subResult);
	cudaFree(gpu_mulResult);
	cudaFree(gpu_modResult);
	cudaFree(gpu_addBlock);
	cudaFree(gpu_addThread);
	cudaFree(gpu_subBlock);
	cudaFree(gpu_subThread);
	cudaFree(gpu_mulBlock);
	cudaFree(gpu_mulThread);
	cudaFree(gpu_modBlock);
	cudaFree(gpu_modThread);
	
	
	
	/* Iterate through the arrays and print */
	cout<<"######################################"<<endl;
	cout<<"blocks = "<<num_blocks<<"\tThreads = "<<num_threads<<endl;
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cout<<"Array1["<<i<<"] = "<<cpu_arr1[i]
		<<"\tArray2["<<i<<"] = "<<cpu_arr2[i]
		<<"\tresult["<<i<<"] = "<<cpu_modResult[i]<<endl;
	}
	cout<<"######################################"<<endl;

	


	/* Iterate through the arrays and print */
	//for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	//{
	//	cout<<("Thread: %2u - Block: %2u\n",cpu_thread[i],cpu_block[i]);
	//}
}

int main()
{
	main_sub0();

	return EXIT_SUCCESS;
}

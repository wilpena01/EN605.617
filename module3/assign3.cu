//
//  assignment3.cpp
//  assignment3
//
//  Created by Wilson on 2/10/22.
//


#include <iostream>

using namespace std;

#define ARRAY_SIZE 64
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
#define ARRAY_SIZE_IN_BYTES1 (sizeof(int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_arr1[ARRAY_SIZE];
unsigned int cpu_arr2[ARRAY_SIZE];


unsigned int cpu_addResult[ARRAY_SIZE];
         int cpu_subResult[ARRAY_SIZE];
unsigned int cpu_mulResult[ARRAY_SIZE];
unsigned int cpu_modResult[ARRAY_SIZE];

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
void add_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] + arr2[thread_idx];
	
	//block[thread_idx] = blockIdx.x;
	//thread[thread_idx] = threadIdx.x;
}

void main_sub0()
{

	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	
	unsigned int *gpu_addResult;
	         int *gpu_subResult;
	unsigned int *gpu_mulResult;
	unsigned int *gpu_modResult;

	cudaMalloc((void **)&gpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2, ARRAY_SIZE_IN_BYTES);
	
	cudaMalloc((void **)&gpu_addResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subResult, ARRAY_SIZE_IN_BYTES1);
	cudaMalloc((void **)&gpu_mulResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modResult, ARRAY_SIZE_IN_BYTES);
	
	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);


	const unsigned int numthread_per_block = 16;
	const unsigned int num_blocks = ARRAY_SIZE/numthread_per_block;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Execute init kernel */
	init<<<num_blocks, num_threads>>>(gpu_arr1,      gpu_arr2, 
									  gpu_addResult, gpu_subResult,
									  gpu_mulResult, gpu_modResult);
									  
	/* Execute init kernel */
	add_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_addResult);								                
									  
	/* Free the arrays on the GPU as now we're done with them */

	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	cudaFree(gpu_addResult);
	cudaFree(gpu_subResult);
	cudaFree(gpu_mulResult);
	cudaFree(gpu_modResult);
	
	
	/* Iterate through the arrays and print */
	cout<<"######################################"<<endl;
	cout<<"blocks = "<<num_blocks<<"\tThreads = "<<num_threads<<endl;
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cout<<"Array1["<<i<<"] = "<<cpu_arr1[i]
		<<"\tArray2["<<i<<"] = "<<cpu_arr2[i]
		<<"\tresult["<<i<<"] = "<<cpu_addResult[i]<<endl;
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

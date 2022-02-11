//
//  assignment3.cpp
//  assignment3
//
//  Created by Wilson on 2/10/22.
//

#include <iostream>

using namespace std;

#define ARRAY_SIZE 256
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
double cpu_arr1[ARRAY_SIZE];
double cpu_arr2[ARRAY_SIZE];
double cpu_result[ARRAY_SIZE];

__global__
void init(double *arr1, double *arr2, double *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	arr1[thread_idx] = thread_idx;
	arr2[thread_idx] = thread_idx % 3;
	result[thread_idx] = 0;
	
	
}
__global__
void add_arr(double *arr1, double *arr2, double *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] + arr2[thread_idx];
	
	//block[thread_idx] = blockIdx.x;
	//thread[thread_idx] = threadIdx.x;
}

__global__
void sub_arr(double *arr1, double *arr2, double *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] - arr2[thread_idx];
	
	//block[thread_idx] = blockIdx.x;
	//thread[thread_idx] = threadIdx.x;
}

__global__
void mul_arr(double *arr1, double *arr2, double *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] * arr2[thread_idx];
	
	//block[thread_idx] = blockIdx.x;
	//thread[thread_idx] = threadIdx.x;
}

__global__
void mod_arr(double *arr1, double *arr2, double *result)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[thread_idx] = arr1[thread_idx] % arr2[thread_idx];
	
	//block[thread_idx] = blockIdx.x;
	//thread[thread_idx] = threadIdx.x;
}

void main_sub0()
{

	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	unsigned int *gpu_result;

	cudaMalloc((void **)&gpu_arr1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_result, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	const unsigned int numthread_per_block = 16;
	const unsigned int num_blocks = ARRAY_SIZE/numthread_per_block;
	const unsigned int num_threads = ARRAY_SIZE/num_blocks;

	/* Execute init kernel */
	init<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_result);

	/* Iterate through the arrays and print */
	cout<<"######################################"<<endl;
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cout<<"Array1["<<i<<"] = "<<gpu_arr1[i]
		<<"\tArray2["<<i<<"] = "<<<gpu_arr1[i]
		<<"result["<<i<<"] = "<<gpu_result[i]<<endl;
	}
	cout<<"######################################"<<endl;

	/* Execute init kernel */
	add_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_result);

	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_result, gpu_result, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	cudaFree(gpu_result);

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

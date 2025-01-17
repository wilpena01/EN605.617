//
//  assignment3.cu
//  assignment3
//
//  Created by Wilson on 2/10/22.
//


#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;


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
void mul_branch(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (thread_idx%2 == 0)
		result[thread_idx] = arr1[thread_idx] * arr2[thread_idx];
	else
		result[thread_idx] = 99999999;  //invalid number
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

__global__
void mod_arr(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int *block, unsigned int *thread)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(arr2[thread_idx] != 0)
		result[thread_idx] = arr1[thread_idx] % arr2[thread_idx];
	else
		result[thread_idx] = 99999999;  //invalid number
	
	block[thread_idx] = blockIdx.x;
	thread[thread_idx] = threadIdx.x;
}

void host_mul_branch(unsigned int *arr1, unsigned int *arr2, unsigned int *result,
			 unsigned int arraySize)
{
	for (int i=0; i<arraySize; i++)	
	{
		if (i%2 == 0)
			result[i] = arr1[i] * arr2[i];
		else
			result[i] = 99999999;  //invalid number
	}
}

void main_sub0(const unsigned int ARRAY_SIZE, const unsigned int num_threads, 
               const unsigned int num_blocks)
{

	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	unsigned int ARRAY_SIZE_IN_BYTES1 = (sizeof(int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int cpu_arr1[ARRAY_SIZE];
	unsigned int cpu_arr2[ARRAY_SIZE];
	unsigned int cpu_addResult[ARRAY_SIZE];
	unsigned int cpu_addBlock[ARRAY_SIZE];
	unsigned int cpu_addThread[ARRAY_SIZE];	
         	 int cpu_subResult[ARRAY_SIZE];
    unsigned int cpu_subBlock[ARRAY_SIZE];
	unsigned int cpu_subThread[ARRAY_SIZE];	
	unsigned int cpu_mulResult[ARRAY_SIZE];
	unsigned int cpu_mulBlock[ARRAY_SIZE];
	unsigned int cpu_mulThread[ARRAY_SIZE];	
	unsigned int cpu_modResult[ARRAY_SIZE];
	unsigned int cpu_modBlock[ARRAY_SIZE];
	unsigned int cpu_modThread[ARRAY_SIZE];	
	unsigned int cpu_brResult[ARRAY_SIZE];
	unsigned int cpu_brBlock[ARRAY_SIZE];
	unsigned int cpu_brThread[ARRAY_SIZE];	

	unsigned int cpu_hostbrResult[ARRAY_SIZE];	
	
	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	unsigned int *gpu_addResult;
	unsigned int *gpu_addBlock;
	unsigned int *gpu_addThread;
	         int *gpu_subResult;
	unsigned int *gpu_subBlock;
	unsigned int *gpu_subThread;
	unsigned int *gpu_mulResult;
	unsigned int *gpu_mulBlock;
	unsigned int *gpu_mulThread;	
	unsigned int *gpu_modResult;
	unsigned int *gpu_modBlock;
	unsigned int *gpu_modThread;	
	unsigned int *gpu_brResult;
	unsigned int *gpu_brBlock;
	unsigned int *gpu_brThread;		

	/* allocate memory for GPU based params */
	cudaMalloc((void **)&gpu_arr1,      ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2,      ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addBlock,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_addThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subResult, ARRAY_SIZE_IN_BYTES1);
	cudaMalloc((void **)&gpu_subBlock,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_subThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_mulResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_mulBlock,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_mulThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modResult, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modBlock,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_modThread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_brResult,  ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_brBlock,   ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_brThread,  ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_arr1,      gpu_arr1,      ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_arr2,      gpu_arr2,      ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addBlock,  gpu_addBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_addThread, gpu_addThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1,cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subBlock,  gpu_subBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subThread, gpu_subThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulBlock,  gpu_mulBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_mulThread, gpu_mulThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modBlock,  gpu_modBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modThread, gpu_modThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_subResult, gpu_brResult,  ARRAY_SIZE_IN_BYTES1,cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modBlock,  gpu_brBlock,   ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_modThread, gpu_brThread,  ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);



 	
 	
	/* Execute kernels */
	init<<<num_blocks, num_threads>>>(gpu_arr1,      gpu_arr2, 
									  gpu_addResult, gpu_subResult,
									  gpu_mulResult, gpu_modResult);
									  
	add_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_addResult, 
										 gpu_addBlock, gpu_addThread);
										 
	sub_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_subResult, 
										 gpu_subBlock, gpu_subThread);
										 
	auto start1 = high_resolution_clock::now();	
									 	
	mul_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_mulResult, 
										 gpu_mulBlock, gpu_mulThread);
										 
	auto stop1 = high_resolution_clock::now();	
	auto start2 = high_resolution_clock::now();		
								 
	mul_branch<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_brResult, 
										 gpu_brBlock, gpu_brThread);
	
	auto stop2 = high_resolution_clock::now();			
							 								                
	mod_arr<<<num_blocks, num_threads>>>(gpu_arr1, gpu_arr2, gpu_modResult, 
										 gpu_modBlock, gpu_modThread);
	cudaDeviceSynchronize();
										 
	
	
	auto duration1 = duration_cast<microseconds>(stop1 - start1);
	auto duration2 = duration_cast<microseconds>(stop2 - start2);
	  
	/* Free the arrays on the GPU as now we're done with them */
	cudaMemcpy(cpu_arr1,      gpu_arr1,      ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2,      gpu_arr2,      ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addResult, gpu_addResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addBlock,  gpu_addBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_addThread, gpu_addThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subResult, gpu_subResult, ARRAY_SIZE_IN_BYTES1,cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subBlock,  gpu_subBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_subThread, gpu_subThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulResult, gpu_mulResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulBlock,  gpu_mulBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_mulThread, gpu_mulThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modResult, gpu_modResult, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modBlock,  gpu_modBlock,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_modThread, gpu_modThread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_brResult,  gpu_brResult,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_brBlock,   gpu_brBlock,   ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_brThread,  gpu_brThread,  ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
	cudaFree(gpu_addResult);
	cudaFree(gpu_addBlock);
	cudaFree(gpu_addThread);
	cudaFree(gpu_subResult);
	cudaFree(gpu_subBlock);
	cudaFree(gpu_subThread);
	cudaFree(gpu_mulResult);
	cudaFree(gpu_mulBlock);
	cudaFree(gpu_mulThread);
	cudaFree(gpu_modResult);
	cudaFree(gpu_modBlock);
	cudaFree(gpu_modThread);
	cudaFree(gpu_brResult);
	cudaFree(gpu_brBlock);
	cudaFree(gpu_brThread);
	
	auto stop3     = high_resolution_clock::now();	
	host_mul_branch(cpu_arr1, cpu_arr2, cpu_hostbrResult, ARRAY_SIZE);
	auto start3    = high_resolution_clock::now();	
	auto duration3 = duration_cast<microseconds>(stop2 - start2);

	//output the capture data
	for(unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cout<<"Array1["<<i<<"] = "<<cpu_arr1[i]<<"\nArray2["<<i<<"]  = "<<cpu_arr2[i]
		
		<<"\nAdd["<<i<<"] = "<<cpu_addResult[i]<<"\taddBock["<<i<<"] = "<<cpu_addBlock[i]
		<<"\taddThread["<<i<<"] = "<<cpu_addThread[i]<<"\n"
		
		
		<<"Sub["<<i<<"] = "<<cpu_subResult[i]<<"\tsubBock["<<i<<"] = "<<cpu_subBlock[i]
		<<"\tsubThread["<<i<<"] = "<<cpu_subThread[i]<<"\n"
		
		
		<<"Mul["<<i<<"] = "<<cpu_mulResult[i]<<"\tmulBock["<<i<<"] = "<<cpu_mulBlock[i]
		<<"\tmulThread["<<i<<"] = "<<cpu_mulThread[i]<<"\n"
		
		
		<<"Mod["<<i<<"] = "<<cpu_modResult[i]<<"\tmodBock["<<i<<"] = "<<cpu_modBlock[i]
		<<"\tmodThread["<<i<<"] = "<<cpu_modThread[i]<<"\n"
		
		
		<<"MulB["<<i<<"] = "<<cpu_brResult[i]<<"\tBr_Bock["<<i<<"] = "<<cpu_brBlock[i]
		<<"\tBr_Thread["<<i<<"] = "<<cpu_brThread[i]<<"\n"
		
		<<"\n######################################\n";

	}

		/* Iterate through the arrays and print */
	cout<<"\nTotal # of Threads = "<<ARRAY_SIZE
	      <<"\nNumber of threads per block = "<<num_threads
	      <<"\nTotal # of blocks = "<<num_blocks
	      <<"\nElapsed Mul time is = "<< duration1.count() << " milliseconds"
	      <<"\nElapsed Mul Branched time is = "<< duration2.count() << " milliseconds"
	      <<"\nElapsed Host Mul time is = "<< duration3.count() << " milliseconds\n"
	      <<"\n######################################\n";
}

int main(int argc, char** argv)
{
	// read command line arguments
	unsigned int totalThreads = 256;
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
	
	main_sub0(totalThreads, blockSize, numBlocks);
	
	return EXIT_SUCCESS;
}
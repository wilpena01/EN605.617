//
//  assignment3.cu
//  assignment3
//
//  Created by Wilson on 2/10/22.
//


#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

struct RESULT
{
	vector<int> result;
	vector<unsigned int> blockId;
	vector<unsigned int> threadId;
};

void pushResult(unsigned int *cpu_addResult, unsigned int *cpu_addBlock, 
unsigned int *cpu_addThread, RESULT *finalResult, unsigned int ARRAY_SIZE)
{
	for(int i=0; i< ARRAY_SIZE; i++)
	{
		finalResult->result.push_back(cpu_addResult[i]);
		finalResult->blockId.push_back(cpu_addBlock[i]);
		finalResult->threadId.push_back(cpu_addThread[i]);
	}

}
void output(RESULT *outadd, unsigned int arraySize)
{
	for(int i=0; i<arraySize; i++)
	{
		cout<<"Add["<<i<<"] = "<<outadd->result.pop_back()
		    <<"\tBlockId["<<i<<"] = "<<outadd->blockId.pop_back()
			<<"\tThreadId["<<i<<"] = "<<outadd->threadId.pop_back()
			<<endl;
	}
}

__global__
void init(unsigned int *arr1, unsigned int *arr2)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	arr1[thread_idx] = thread_idx;
	arr2[thread_idx] = thread_idx % 4;	
}

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

void run_Funs(unsigned int *gpu_arr1, unsigned int *gpu_arr2, 
         unsigned int numBlocks, unsigned int blockSize)
{
	RESULT addR; const unsigned int ARRAY_SIZE = numBlocks * blockSize;
	
	Topadd(gpu_arr1, gpu_arr1, numBlocks, blockSize, &addR);
	output(&addR,ARRAY_SIZE);

}


void submain(unsigned int totalThreads, unsigned int  blockSize, unsigned int numBlocks)
{
	const unsigned int ARRAY_SIZE = totalThreads;
	unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (ARRAY_SIZE));
	unsigned int ARRAY_SIZE_IN_BYTES1 = (sizeof(int) * (ARRAY_SIZE));
	
	/* Declare  statically arrays of ARRAY_SIZE each */
	unsigned int cpu_arr1[ARRAY_SIZE];
	unsigned int cpu_arr2[ARRAY_SIZE];
	
	/* Declare pointers for GPU based params */
	unsigned int *gpu_arr1;
	unsigned int *gpu_arr2;
	
	cudaMalloc((void **)&gpu_arr1,      ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_arr2,      ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(cpu_arr1,      gpu_arr1,      ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(cpu_arr2,      gpu_arr2,      ARRAY_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	
	/* Execute kernels */
	init<<<numBlocks, blockSize>>>(gpu_arr1, gpu_arr2);
									  
	run_Funs(gpu_arr1, gpu_arr2, numBlocks, blockSize);
									  
	cudaMemcpy(cpu_arr1,      gpu_arr1,      ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2,      gpu_arr2,      ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);								  
	cudaFree(gpu_arr1);
	cudaFree(gpu_arr2);
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
	
	submain(totalThreads, blockSize, numBlocks);

	
	return EXIT_SUCCESS;
}
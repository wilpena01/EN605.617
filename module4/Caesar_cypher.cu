#include <stdio.h> 
#include <stdlib.h>
#include "Utilities.h"

/* 
 * The maximum and minimum integer values of the range of printable characters 
 * in the ASCII alphabet. Used by encrypt kernel to wrap adjust values to that 
 * ciphertext is always printable. 
 */ 
#define MAX_PRINTABLE 64 
#define MIN_PRINTABLE 128 
#define NUM_ALPHA MAX_PRINTABLE - MIN_PRINTABLE



unsigned int get_data_from_file(unsigned int *cpu_text, unsigned int *cpu_key,
								FILE *input_fp, FILE *key_fp, int array_size)
{
	//import data from the msg file and the key file
	int array_size_in_bytes = (sizeof(unsigned int) * (array_size));
	char temp;
	unsigned int idx=0; 
	while ( fscanf(input_fp, "%c", &temp ) == 1 && idx<array_size )  
	{ 
		cpu_text[idx] = static_cast<unsigned int>(temp);
		idx++;
	}

	fscanf(key_fp,"%c", &temp);
	memset(cpu_key,static_cast<unsigned int>(temp),array_size_in_bytes);

	return idx;
}
void print_results(unsigned int *cpu_text, unsigned int *cpu_key, 
			  unsigned int *cpu_result, int array_size, float duration)
{
	// print the result of the encryption
	printf("Pageable Transfer- Duration: %fmsn\n", duration); 
	cout<<"\nmsg: ";
	for(int i=0; i<array_size; i++)
	{
		cout<<static_cast<char>(cpu_text[i]);
	}
	cout<<"\nkey msg: "<<static_cast<char>(cpu_key[1]);
	
	cout<<"\nencrypted msg: ";
	for(int i=0; i<array_size; i++)
	{
		cout<<static_cast<char>(cpu_result[i]);
	}
	cout<<endl;
	
}
__global__ void encrypt(unsigned int *text, unsigned int *key, unsigned int *result) 
{ /* Calculate the current index */ 

	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	 /* 
	  * Adjust value of text and key to be based at 0 
	  * Printable ASCII starts at MIN_PRINTABLE, but 0 start is easier to work with 
	  */ 
	char adjusted_text = static_cast<char>(text[idx] - MIN_PRINTABLE); 
	char adjusted_key = static_cast<char>(key[idx] - MIN_PRINTABLE);

	 /* The cipher character is the text char added to the key char modulo the number of chars in the alphabet*/ 
	char cipherchar = (adjusted_text + adjusted_key) % (NUM_ALPHA);

	 /* adjust back to normal ascii (starting at MIN_PRINTABLE) and save to result */ 
	result[idx] = static_cast<unsigned int>(cipherchar + MIN_PRINTABLE);
}
void freeData(unsigned int *gpu_text, unsigned int *gpu_key, unsigned int *gpu_result,
		 unsigned int *cpu_text, unsigned int *cpu_key, unsigned int *cpu_result)
{
	/* Free the GPU memory */ 
	 cudaFree(gpu_text);
	 cudaFree(gpu_key);
	 cudaFree(gpu_result);

	 /* Free the CPU memory */ 
	 free(cpu_text);
	 free(cpu_key);
	 free(cpu_result);
}

float run_funs(unsigned int *gpu_text, unsigned int *gpu_key, unsigned int *gpu_result,
		 	   unsigned int num_blocks, unsigned int num_threads)
{
	//run the kerner and time the function
	 float duration         = 0; 
	 cudaEvent_t start_time = get_time();

	 encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

	 cudaEvent_t end_time   = get_time(); 
	 cudaEventSynchronize(end_time); 
	 cudaEventElapsedTime(&duration, start_time, end_time);

	 return duration;
}

void pageable_transfer_execution(int array_size, int threads_per_block, FILE *input_fp, FILE *key_fp) 
{ /* Calculate the size of the array*/ 

	int array_size_in_bytes  = (sizeof(unsigned int) * (array_size));
	unsigned int *cpu_text   = (unsigned int *) malloc(array_size_in_bytes); 
	unsigned int *cpu_key    = (unsigned int *) malloc(array_size_in_bytes); 
	unsigned int *cpu_result = (unsigned int *) malloc(array_size_in_bytes);

	/* Read characters from the input and key files into the text and key arrays */ 
	unsigned int idx = 0;
	idx = get_data_from_file(cpu_text, cpu_key,input_fp,key_fp, array_size);

	 
	 unsigned int *gpu_text, *gpu_key, *gpu_result;
	 cudaMalloc((void **)&gpu_text, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_key, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_result, array_size_in_bytes);

	 /* Copy the CPU memory to the GPU memory */ 
	 cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice); 
	 cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

	 /* Designate the number of blocks and threads */ 
	 const unsigned int num_blocks = array_size/threads_per_block; 
	 const unsigned int num_threads = array_size/num_blocks;

	 /* Execute the encryption kernel and keep track of start and end time for duration */ 
	 float dx = 0; 
	 dx = run_funs(gpu_text, gpu_key, gpu_result, num_blocks, num_threads);

	 /* Copy the changed GPU memory back to the CPU */ 
	 cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);
	 print_results(cpu_text, cpu_key, cpu_result, idx, dx);

	/* Free the CPU & GPU memory */ 
	freeData(gpu_text, gpu_key, gpu_result, 
			 cpu_text, cpu_key, cpu_result);

}


void print_usage(char *name) {
	printf("Usage: %s <total_num_threads> <threads_per_block> <input_file> <key_file>\n", name); 
}

/** 
  * Performs simple setup functions before calling the pageable_transfer() 
  * function. * Makes sure the files are valid, handles opening and closing of file pointers. 
  */ void pageable_transfer(int num_threads, int threads_per_block) { 

	FILE *input_fp, *key_fp;
	input_fp = fopen("msg.txt","r");
	key_fp = fopen("key.txt","r");

	if(input_fp != NULL && key_fp != NULL)
	{
		/* Perform the pageable transfer */ 
		pageable_transfer_execution(num_threads, threads_per_block, input_fp, key_fp);           
	}
 	
 	fclose(input_fp); fclose(key_fp); 
}



	/** 
	  * Entry point for excution. Checks command line arguments and 
	  */
int main(int argc, char *argv[]) { 
	
	// read command line arguments
	unsigned int totalThreads = 64;
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

 	printf("\n"); 
	/* Perform the pageable transfer */ 
	pageable_transfer(totalThreads, blockSize);

 	return EXIT_SUCCESS; 
}

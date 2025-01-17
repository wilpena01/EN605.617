#ifndef UTILITIES_H
#define UTILITIES_H
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

struct RESULT
{
    //structure used to store the final
    //result of each mathematical
    //calculation
	vector<int> result;
	vector<unsigned int> blockId;
	vector<unsigned int> threadId;
};

void init(unsigned int *arr1, unsigned int *arr2 , unsigned int ARRAY_SIZE)
{
    //initialize the input arrays.
	for(unsigned int i = 0; i<ARRAY_SIZE; i++)
	{
		arr1[i] = i;
		arr2[i] = i % 4;
	}
}

template <typename T>
void pushResult(T *cpu_Result, unsigned int *cpu_Block, 
unsigned int *cpu_Thread, RESULT *finalResult, unsigned int ARRAY_SIZE)
{
    //puch the kerner result into the finalresult structure
	for(int i=0; i< ARRAY_SIZE; i++)
	{
		finalResult->result.push_back(cpu_Result[i]);
		finalResult->blockId.push_back(cpu_Block[i]);
		finalResult->threadId.push_back(cpu_Thread[i]);
	}

}

void output(unsigned int *gpu_arr1, unsigned int *gpu_arr2, RESULT *outadd, 
    RESULT *outsub, RESULT *outmul, RESULT *outmod, unsigned int arraySize)
{
    //output the result of every calculation
    unsigned int *in1, *in2;
    unsigned int ARRAY_SIZE_IN_BYTES  = (sizeof(unsigned int) * (arraySize));

    in1 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);
	in2 = (unsigned int *)malloc(ARRAY_SIZE_IN_BYTES);

    cudaMemcpy(in1, gpu_arr1 , ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(in2, gpu_arr2 , ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    	//output the capture data
	for(unsigned int i = 0; i < arraySize; i++)
	{
		cout<<"Array1["<<i<<"] = "<<in1[i]<<"\tArray2["<<i<<"]  = "<<in2[i]
		
		<<"\nAdd["<<i<<"] = "<<outadd->result.at(i)<<"\taddBock["<<i<<"] = "<<outadd->blockId.at(i)
		<<"\taddThread["<<i<<"] = "<<outadd->threadId.at(i)<<"\n"
		
		
		<<"Sub["<<i<<"] = "<<outsub->result.at(i)<<"\tsubBock["<<i<<"] = "<<outsub->blockId.at(i)
		<<"\tsubThread["<<i<<"] = "<<outsub->threadId.at(i)<<"\n"
		
		
		<<"Mul["<<i<<"] = "<<outmul->result.at(i)<<"\tmulBock["<<i<<"] = "<<outmul->blockId.at(i)
		<<"\tmulThread["<<i<<"] = "<<outmul->threadId.at(i)<<"\n"
		
		
		<<"Mod["<<i<<"] = "<<outmod->result.at(i)<<"\tmodBock["<<i<<"] = "<<outmod->blockId.at(i)
		<<"\tmodThread["<<i<<"] = "<<outmod->threadId.at(i)<<"\n"
		
		<<"\n######################################\n";

	}
    free(in1);
    free(in2);
}


void outputTime(float duration1, float duration2)
{
    /* print the duratino */
	cout<<"\nElapsed Time using Pegeable memory allocation = "<< duration1<< " msn"
        <<"\nElapsed Time using Pinned memory allocation = "<< duration2<< " msn"
	    <<"\n######################################\n";
}

__host__ cudaEvent_t get_time(void)
{
    //get the current time.
	cudaEvent_t time;
	cudaEventCreate(&time);
	cudaEventRecord(time);
	return time;
}

#endif
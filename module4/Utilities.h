#ifndef UTILITIES_H
#define UTILITIES_H
#include <iostream>

using namespace std;

struct RESULT
{
	vector<int> result;
	vector<unsigned int> blockId;
	vector<unsigned int> threadId;
};

void init(unsigned int *arr1, unsigned int *arr2, unsigned int ARRAY_SIZE)
{
	for(unsigned int i = 0; i<ARRAY_SIZE; i++)
	{
		arr1[i] = i;
		arr2[i] = i % 4;	
	}
}

void pushResult(unsigned int *cpu_Result, unsigned int *cpu_Block, 
unsigned int *cpu_Thread, RESULT *finalResult, unsigned int ARRAY_SIZE)
{
	for(int i=0; i< ARRAY_SIZE; i++)
	{
		finalResult->result.push_back(cpu_Result[i]);
		finalResult->blockId.push_back(cpu_Block[i]);
		finalResult->threadId.push_back(cpu_Thread[i]);
        /*
        finalResult->result.insert(finalResult->result.begin()     + i,cpu_Result[i]);
		finalResult->blockId.insert(finalResult->blockId.begin()   + i, cpu_Block[i]);
		finalResult->threadId.insert(finalResult->threadId.begin() + i,cpu_Thread[i]);
        */
	}

}

void output(unsigned int *inp1, unsigned int *inp2, RESULT *outadd, RESULT *outsub, 
            RESULT *outmul, RESULT *outmod, unsigned int arraySize)
{
    unsigned int *in1, *in2;
    cudaMemcpy(cpu_arr1, gpu_arr1, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_arr2, gpu_arr2, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    	//output the capture data
	for(unsigned int i = 0; i < arraySize; i++)
	{
		cout<<"Array1["<<i<<"] = "<<in1[i]<<"\nArray2["<<i<<"]  = "<<in2[i]
		
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
}



#endif
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
		finalResult->result.insert(finalResult->result.begin()     + i,cpu_Result[i]);
		finalResult->blockId.insert(finalResult->blockId.begin()   + i,cpu_Block[i]);
		finalResult->threadId.insert(finalResult->threadId.begin() + i,cpu_Thread[i]);
	}

}

void output(RESULT *outadd, unsigned int arraySize)
{
	for(int i=0; i<arraySize; i++)
	{
		cout<<"Add["<<i<<"] = "<<outadd->result.at(i)
		    <<"\tBlockId["<<i<<"] = "<<outadd->blockId.at(i)
			<<"\tThreadId["<<i<<"] = "<<outadd->threadId.at(i)
			<<endl;
	}
}



#endif
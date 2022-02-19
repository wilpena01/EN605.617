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
		cout<<"Add["<<i<<"] = "<<outadd->result.at(i)
		    <<"\tBlockId["<<i<<"] = "<<outadd->blockId.at(i)
			<<"\tThreadId["<<i<<"] = "<<outadd->threadId.at(i)
			<<endl;
	}
}

#endif
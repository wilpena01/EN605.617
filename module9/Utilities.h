#ifndef UTILITIES_H
#define UTILITIES_H
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <thrust/host_vector.h>


using namespace std::chrono;
using namespace std;
typedef unsigned int UInt32;
typedef int Int32;

struct RESULT
{
    //structure used to store the final
    //result of each mathematical
    //calculation
	vector<int> result;
	vector<UInt32> blockId;
	vector<UInt32> threadId;
};


void init(UInt32 *arr1, UInt32 *arr2 , UInt32 ARRAY_SIZE)
{
    //initialize the input arrays.
	for(UInt32 i = 0; i<ARRAY_SIZE; i++)
	{
		arr1[i] = (unsigned int) rand() % 10;
		arr2[i] = (unsigned int) rand() % 10;
	}
}

template <typename T>
void pushResult(T *cpu_Result, UInt32 *cpu_Block, 
UInt32 *cpu_Thread, RESULT *finalResult, UInt32 ARRAY_SIZE)
{
    //puch the kerner result into the finalresult structure
	for(int i=0; i< ARRAY_SIZE; i++)
	{
		finalResult->result.push_back(static_cast<int>(cpu_Result[i]));
		finalResult->blockId.push_back(cpu_Block[i]);
		finalResult->threadId.push_back(cpu_Thread[i]);
	}

}

void output(UInt32 *gpu_arr1, UInt32 *gpu_arr2, RESULT *outadd, 
    RESULT *outsub, RESULT *outmul, RESULT *outmod, UInt32 arraySize)
{
    //output the result of every calculation
    UInt32 *in1, *in2;
    UInt32 ARRAY_SIZE_IN_BYTES  = (sizeof(UInt32) * (arraySize));

    in1 = (UInt32 *)malloc(ARRAY_SIZE_IN_BYTES);
	in2 = (UInt32 *)malloc(ARRAY_SIZE_IN_BYTES);

    cudaMemcpy(in1, gpu_arr1 , ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy(in2, gpu_arr2 , ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

    	//output the capture data
	for(UInt32 i = 0; i < arraySize; i++)
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


void outputTime(microseconds duration1, 
				microseconds duration2, string *str)
{
    /* print the duratino */
	cout<<"Elapsed Time using "<<str[0]<<" = "<< duration1.count()<< " microsecond"
        <<"\nElapsed Time Not using "<<str[0]<<" = "<< duration2.count()<< " microsecond\n\n";
	    
}

void outputTime(float duration1, float duration2)
{
    /* print the duratino */
	cout<<"\nElapsed Time using multiple stream = "<< duration1<< " microsecond"
        <<"\nElapsed Time NOT using multiple streams = "<< duration2<< " microsecond\n\n";
	    
}

void outputTimeReg(float duration1, float duration2, UInt32 *str)
{
    /* print the duratino */
	cout<<"\nElapsed Time with "<<str[0]<<" input size and register memory allocation = "<< duration1<< " micro"
        <<"\nElapsed Time with "<<str[1]<<" input size and register memory allocation = "<< duration2<< " msn\n";
	    
}

void outputTime(float duration1, float duration2, 
				float duration3, float duration4)
{
    /* print the duratino */
	cout<<"\nElapsed Time using global memory allocation = "<< duration1<< " msn"
        <<"\nElapsed Time using shared memory allocation = "<< duration2<< " msn"
		<<"\nElapsed Time using literal memory allocation = "<< duration3<< " msn"
        <<"\nElapsed Time using constant memory allocation = "<< duration4<< " msn"
	    <<"\n######################################\n";
}

void outputVec(thrust::host_vector<int> vec)
{
	for(int i = 0; i != vec.size(); i++)
	{
		cout<<vec[i]<<" ";
	}
	cout<<endl;
}

inline int index(int i, int j, int k) 
{
	return (((j)*(k))+(i));
}

void printMat(float*mat,int W,int H)
{
  //printf("\n %f",P[1]);
  for(int i=0;i<H;i++)
  {
      cout<<"\n";
      for(int j=0;j<W;j++)
          cout<<mat[index(i,j,H)]<<"\t";
  }
}

void mulMat(float *mat1, float* mat2,int H, int W, float *rslt ) 
{
    cout << "Multiplication of given two matrices is:\n" << endl;
 
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            rslt[index(i,j,H)] = 0;
 
            for (int k = 0; k < W; k++) {
                rslt[index(i,j,H)] += mat1[index(i,k,H)] * mat2[index(k,j,H)];
            }
 
            cout << rslt[index(i,j,H)] << "\t";
        }
 
        cout << endl;
    }
}



#endif
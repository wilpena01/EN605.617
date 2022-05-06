/*
 * Utilities.h
 *
 *  Created on: Apr 10, 2022
 *      Author: Wilson
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <iostream>
#include <vector>
using namespace std;
typedef unsigned int uint32;

struct RESULT
{
    //structure used to store the final
    //result of each mathematical
    //calculation
	vector<int> result;
	vector<int> blockId;
	vector<int> threadId;
};
// Defining Structures pixfreq
template<unsigned int N>
struct pixfreq
{
   int intensity, larrloc, rarrloc;
   float Freq;
   struct pixfreq<N> *left, *right;
   char code[N];
};


// Defining Structures
// huffcode
struct huffcode
{
   int intensity, arrloc;
   float Freq;

};

// function to find fibonacci number
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n - 1) + fib(n - 2);
}

int index(int H, int W, int k) 
{
	return ((W*k)+H);
}


int codelen(char* code)
{
    // function to calculate word length
   int l = 0;
   while (*(code + l) != '\0')
      l++;
   return l;
}

void PrintHuffmanCode(pixfreq<25> *pix_freq, int nodes)
{
   // Printing Huffman Codes
   printf("Huffmann Codes::\n\n");
   printf("pixel values -> Code\n\n");
   for (int i = 0; i < nodes; i++) 
   {
      if (snprintf(NULL, 0, "%d", pix_freq[i].intensity) == 2)
         printf("  %d    -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
      else
         printf(" %d  -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
   }
}

void calBitLength(pixfreq<25> *pix_freq, int nodes)
{
   // Calculating Average Bit Length
   float avgbitnum = 0;
   for (int i = 0; i < nodes; i++)
      avgbitnum += pix_freq[i].Freq * codelen(pix_freq[i].code);
   printf("Average number of bits:: %f", avgbitnum);

}


void outputResult(int *Result, int* Block, int* Thread, int idx)
{
   for(int i=0; i<idx; i++)
   {
      std::cout<<"r = "<<Result[i]<<"\tblock = "<<Block[i]<<"\tThread = "<<Thread[i]<<std::endl;;
   }  
}

void  allocHost(int *&cpu_Result,int *&cpu_Block,int *&cpu_Thread,int *&hist, 
                int *&totalnodes,int *&nodes, const int &HistSize_Byte)
{
   cpu_Result  = (int *)malloc(HistSize_Byte);
	cpu_Block   = (int *)malloc(HistSize_Byte);
   cpu_Thread  = (int *)malloc(HistSize_Byte);
   hist        = (int *)malloc(HistSize_Byte);
   totalnodes  = (int *)malloc(sizeof(int));
   nodes       = (int *)malloc(sizeof(int));
}

void allocDevice(int* &g_image, int* &g_width, int* &g_height, int* &g_hist, 
                 int* &g_nodes, int* &g_totalnodes, int* &gpu_Result, 
                 int* &gpu_Block, int* &gpu_Thread, int IMAGE_SIZE_IN_BYTES,
                 int HistSize_Byte)
{
   cudaMalloc((void **)&g_image,       IMAGE_SIZE_IN_BYTES);
   cudaMalloc((void **)&g_width,       sizeof(int));
   cudaMalloc((void **)&g_height,      sizeof(int));
   cudaMalloc((void **)&g_hist,        HistSize_Byte);
   cudaMalloc((void **)&g_nodes,       sizeof(int));
   cudaMalloc((void **)&g_totalnodes,  sizeof(int));
   cudaMalloc((void **)&gpu_Result,    HistSize_Byte);
   cudaMalloc((void **)&gpu_Block,     HistSize_Byte);
   cudaMalloc((void **)&gpu_Thread,    HistSize_Byte);
}

void HostToDevice(int* &g_image, int* &g_width, int* &g_height, int* &g_hist,
                  int* &image, int &width, int &height, int* &hist, 
                  int* &g_nodes, int* &g_totalnodes, int* &nodes, int* &totalnodes, 
                  int IMAGE_SIZE_IN_BYTES, int HistSize_Byte)
{
   cudaMemcpy(g_image,      image,       IMAGE_SIZE_IN_BYTES, cudaMemcpyHostToDevice);
   cudaMemcpy(g_width,      &width,      sizeof(int),         cudaMemcpyHostToDevice);
   cudaMemcpy(g_height,     &height,     sizeof(int),         cudaMemcpyHostToDevice);
   cudaMemcpy(g_hist,       hist,        HistSize_Byte,       cudaMemcpyHostToDevice);
   cudaMemcpy(g_nodes,      &nodes,      sizeof(int),         cudaMemcpyHostToDevice);
   cudaMemcpy(g_totalnodes, &totalnodes, sizeof(int),         cudaMemcpyHostToDevice);
}
#endif /* UTILITIES_H_ */

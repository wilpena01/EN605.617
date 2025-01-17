#ifndef COMPRESS_HELPER_CU_H_
#define COMPRESS_HELPER_CU_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>

#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "ImageIO.h"
#include "Exceptions.h"
#include <npp.h>
#include "Compress_Helper.h"

using namespace std;

__device__ int   shared_hist[256];
__device__ float shared_prob;
__device__ int   shared_temp;
__device__ int   shared_node;
__device__ int   shared_totalnode;

void readBMPFILE_cu(int &width, int &height, int* &image)
{
   // load bmp image
   int i, j, temp = 0, offset = 2, bpp = 0;
   char file[] = "Lena.bmp";
   long bmpS = 0, bmpoff = 0;
   FILE* inputImage;

   inputImage = fopen(file, "rb");
   if (inputImage == NULL)
   {
      cout<<"Error input File!!"<<endl;
      exit(1);
   }
   else
   {
      fseek(inputImage, offset, SEEK_SET);
      fread(&bmpS, 4, 1, inputImage); offset = 10;
      fseek(inputImage, offset, SEEK_SET);
      fread(&bmpoff, 4, 1, inputImage);
      fseek(inputImage, 18, SEEK_SET);
      fread(&width, 4, 1, inputImage);
      fread(&height, 4, 1, inputImage);
      fseek(inputImage, 2, SEEK_CUR);
      fread(&bpp, 2, 1, inputImage);
      fseek(inputImage, bmpoff, SEEK_SET);

      // Creating Image array
      image = (int*)malloc(height * width * sizeof(int));

      // Reading the inputImage
      // into the Image Array
      for (i = 0; i < height; i++)
      {
         for (j = 0; j < width; j++)
         {
            int idx = (i*height) + j;
            fread(&temp, 3, 1, inputImage);
            temp = temp & 0x0000FF;
            image[idx] = static_cast<int>(temp);
         }
      }
   }
   fclose(inputImage);
}

__global__ 
void initHist_cu(int* hist, int *Result, int *Block, int *Thread)
{
   //Initialized all global variables on the device.
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   shared_hist[idx] = 0;
   hist[idx] = 0;
   Result[idx] = 0;
   shared_node = 0;
   shared_prob = 1.0;
   shared_temp = 1;
   Block[idx]  = blockIdx.x;
	Thread[idx] = threadIdx.x;
}

__global__
void ocurrence_cu(int* image)
{
   // Finding the probability
   // of occurrence
   const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

   atomicAdd((shared_hist+image[thread_idx]),1);
   __syncthreads();
}

__global__
void nonZero_ocurrence_cu(int *node, int *Result, int *Block, int *Thread)
{
   // Finding number of
   // non-zero occurrences
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (shared_hist[idx] != 0)
      atomicAdd(&shared_node,1);
    __syncthreads();

   *node = shared_node;
   Result[idx] = *node;
   Block[idx]  = blockIdx.x+5;
	Thread[idx] = threadIdx.x;

}

__global__
void minProp_cu(int* width, int* height, int *Result, int *Block, int *Thread)
{
   // Calculating minimum probability
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   int ptemp = 100000 * shared_hist[idx] / (*height * *width);

   if (ptemp > 0)
      atomicMin(&shared_temp,ptemp);
   __syncthreads();

   shared_prob = shared_temp/100000.0; 
   Result[idx] = ptemp;
   Block[idx]  = blockIdx.x+10;
	Thread[idx] = threadIdx.x;

}

int MaxLength_cu(float p)
{
   // Calculating max length
   int i = 0;
   while ((1 / p) > fib(i))
   {
      i++;
   }

   return i;
}

//done
__global__
void totalNode(int *totalnode, int *node, int *Result, int *Block, int *Thread)
{
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
   *node = shared_node;
   *totalnode = 2 * *node - 1;

   Result[idx] = *totalnode;
   Block[idx]  = blockIdx.x+10;
	Thread[idx] = threadIdx.x;
}

__global__
void InitStruct_cu(pixfreq<25> *pix_freq, huffcode* huffcodes, 
                  int *height, int *width, int *Result, int *Block, int *Thread)
{
     // Initializing
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   __shared__ int jj;

   if(idx == 24)
      jj = 0;
   __syncthreads();
   int MaxSize = *height * *width;
   float currProb;
   int k=0;

   if (shared_hist[idx] != 0)
   {
      k = atomicAdd(&jj,1);
      huffcodes[k].intensity = idx;
      pix_freq[k].intensity = idx;
      huffcodes[k].arrloc = k;
      currProb = shared_hist[idx] / (float)MaxSize;
      pix_freq[k].Freq = currProb;
      huffcodes[k].Freq = currProb;
      pix_freq[k].left = NULL;
      pix_freq[k].right = NULL;
      pix_freq[k].code[0] = '\0';
      
   }
   
   Result[idx] = k;
   Block[idx]  = blockIdx.x+12;
	Thread[idx] = threadIdx.x;
}

__global__
void sortHist_cu(huffcode *huffcodes, int* nodes, int *Result, int *Block, int *Thread)
{
   // Sorting the histogram
   int n, k;
   huffcode huff;

   // Sorting probability
   for (n = 0; n < *nodes; n++)
   {
      for (k = n + 1; k < *nodes; k++)
      {
         if (huffcodes[n].Freq < huffcodes[k].Freq)
         {
               huff = huffcodes[n];
               huffcodes[n] = huffcodes[k];
               huffcodes[k] = huff;
         }
      }
   Result[n] = static_cast<int>(huffcodes[n].Freq*1000000);
   Block[n]  = blockIdx.x+n;
   Thread[n] = threadIdx.x;
} 
   
}

__global__
void BuildTree_cu(pixfreq<25> *pix_freq, huffcode* huffcodes, int *nodes, int *Result, int *Block, int *Thread)
{
   // Building Huffman Tree
   float totalprob;
   int totalpix,z;
   int i = 0, j = 0;
   int n_node = *nodes;

   while (i < *nodes - 1)
   {

      // Adding the lowest two probabilities
      totalprob = huffcodes[*nodes - i - 1].Freq + huffcodes[*nodes - i - 2].Freq;
      totalpix = huffcodes[*nodes - i - 1].intensity + huffcodes[*nodes - i - 2].intensity;

      // Appending to the pix_freq Array
      pix_freq[n_node].intensity = totalpix;
      pix_freq[n_node].Freq = totalprob;
      pix_freq[n_node].left = &pix_freq[huffcodes[*nodes - i - 2].arrloc];
      pix_freq[n_node].right = &pix_freq[huffcodes[*nodes - i - 1].arrloc];
      pix_freq[n_node].code[0] = '\0';
      z = 0;

      // Sorting and Updating the
      // huffcodes array simultaneously
      // New position of the combined node
      while (totalprob <= huffcodes[z].Freq)
         z++;

      // Inserting the new node
      // in the huffcodes array
      for (j = *nodes; j >= 0; j--)
      {
         if (j == z)
         {
               huffcodes[j].intensity = totalpix;
               huffcodes[j].Freq = totalprob;
               huffcodes[j].arrloc = n_node;
         }
         else if (j > z)

         // Shifting the nodes below
         // the new node by 1
         // For inserting the new node
         // at the updated position k
         huffcodes[j] = huffcodes[j - 1];

      }
   Result[i] = i;
   Block[i]  = blockIdx.x+84;
   Thread[i] = threadIdx.x;

   i = i + 1;
   n_node = n_node + 1;

   }


}


__device__
void stradd_cu(char* strptr, char* pcode, char add)
{
   // function to add the words
   int i = 0;
   while (*(pcode + i) != '\0')
   {
      *(strptr + i) = *(pcode + i);
      i++;
   }
   if (add != '2')
   {
      strptr[i] = add;
      strptr[i + 1] = '\0';
   }
   else
      strptr[i] = '\0';
}

__global__
void AssignCode_cu(pixfreq<25> *pix_freq, int *nodes, int *totalnodes, int *Result, int *Block, int *Thread)
{
   // Assigning Code
   int n;
   char left = '0';
   char right = '1';
   for (n = *totalnodes - 1; n >= *nodes; n--)
   {
      if (pix_freq[n].left != NULL)
         stradd_cu(pix_freq[n].left->code, pix_freq[n].code, left);
      if (pix_freq[n].right != NULL)
         stradd_cu(pix_freq[n].right->code, pix_freq[n].code, right);
   Result[n] = n;
   Block[n]  = blockIdx.x+44;
   Thread[n] = threadIdx.x;
   }
    
}

__global__
void copy_data_from_shared(int *hist, int *Result, int *Block, int *Thread)
{
   unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   hist[idx]   = shared_hist[idx];
   Result[idx] = hist[idx];
   Block[idx]  = blockIdx.x;
   Thread[idx] = threadIdx.x;
}
/*
void ocurrence(int* hist, int** image, int width, int height)
{
    // Finding the probability
    // of occurrence
    int i,j;
   
    for (i = 0; i < 256; i++)
        hist[i] = 0;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            hist[image[i][j]] += 1;
        }
    }
}*/

#endif /* COMPRESS_HELPER_CU_H_ */
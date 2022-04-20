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

void LoadImagePGM(int &width, int &height, int** &image_cl)
{
   int i,j;

   string name = "Lena.bmp";
   FILE* inputfile = fopen(name.c_str(), "rb");

   if (inputfile == NULL)
   {
      cout << "assignmentNPP unable to open: <" << name.data() << ">" << endl;
      fclose(inputfile);
      exit(EXIT_FAILURE);
   }
   else
   {
      cout << "assignmentNPP opened: <" << name.data() << "> successfully!" << endl;
   }
      

   string result = name;
   std::string::size_type dot = result.rfind('.');

   if (dot != std::string::npos)
   {
      result = result.substr(0, dot);
   }

   // declare a host image object for an 8-bit grayscale image
   npp::ImageCPU_8u_C1 hostImage;
   
   // load gray-scale image from disk  
   npp::loadImage(name, hostImage);
  
cout<<"\n\nentre aqui<<\n\n";
   height = hostImage.height();
   width = hostImage.width();


   cout<<"height = "<<height<<"\twidth"<<width<<endl;
   // Creating Image array
   image_cl = (int**)malloc(height * sizeof(int*));

   for (i = 0; i < height; i++)
   {
      image_cl[i] = (int*)malloc(width * sizeof(int*));
   }

   for(i=0; i<height; i++)
      for(j=0; j<width;j++)
         image_cl[i][j] = static_cast<int>(*hostImage.data(i,j));

   fclose(inputfile);
}


//done
__global__ 
void initHist_cu(int* hist)
{
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
   hist[idx] = 0;
}


//done i think
__global__
void ocurrence_cu(int* hist, int* image)
{
   // Finding the probability
   // of occurrence
   const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

   hist[image[thread_idx]] += 1;
   __syncthreads();
}


//done I think
__global__
void nonZero_ocurrence_cu(int* hist, int *node)
{
   // Finding number of
   // non-zero occurrences
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (hist[idx] != 0)
      *node += 1;
    __syncthreads();

}

//done i think
__global__
void minProp_cu(float* p, int* hist, int* width, int* height)
{
    // Calculating minimum probability
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   float ptemp;
   ptemp = (hist[idx] / (static_cast<float>(*height * *width)));
   if (ptemp > 0 && ptemp <= *p)
      *p = ptemp;

   __syncthreads();

}

int MaxLength_cu(float p)
{
    // Calculating max length
    // of code word
    int i = 0;
    while ((1 / p) > fib(i))
    {
        i++;
    }

    return i;
}

//done
__global__
void totalNode(int* totalnode, int* nodes)
{
   *totalnode = 2 * *nodes - 1;
   __syncthreads();
}

//done
__global__
void InitStruct_cu(pixfreq<25> *pix_freq, huffcode* huffcodes, 
                int* hist, int height, int width)
{
     // Initializing
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   int i; int j=0;
   int totpix = height * width;
   float tempprob;

   if (hist[idx] != 0)
   {

      // pixel intensity value
      huffcodes[j].intensity = idx;
      pix_freq[j].intensity = idx;

      // location of the node
      // in the pix_freq array
      huffcodes[j].arrloc = j;

      // probability of occurrence
      tempprob = (float)hist[idx] / (float)totpix;
      pix_freq[j].Freq = tempprob;
      huffcodes[j].Freq = tempprob;

      // Declaring the child of leaf
      // node as NULL pointer
      pix_freq[j].left = NULL;
      pix_freq[j].right = NULL;

      // initializing the code
      // word as end of line
      pix_freq[j].code[0] = '\0';
      j++;
   }
   

}

void sortHist_cu(huffcode* huffcodes, int nodes)
{
     // Sorting the histogram
    int i, j;
    huffcode temphuff;

    // Sorting w.r.t probability
    // of occurrence
    for (i = 0; i < nodes; i++)
    {
        for (j = i + 1; j < nodes; j++)
        {
            if (huffcodes[i].Freq < huffcodes[j].Freq)
            {
                temphuff = huffcodes[i];
                huffcodes[i] = huffcodes[j];
                huffcodes[j] = temphuff;
            }
        }
    }
}

void BuildTree_cu(pixfreq<25> *pix_freq, huffcode* huffcodes, int nodes)
{
    // Building Huffman Tree
    float sumprob;
    int sumpix,i;
    int n = 0, k = 0;
    int nextnode = nodes;

    // Since total number of
    // nodes in Huffman Tree
    // is 2*nodes-1
    while (n < nodes - 1)
    {

        // Adding the lowest two probabilities
        sumprob = huffcodes[nodes - n - 1].Freq + huffcodes[nodes - n - 2].Freq;
        sumpix = huffcodes[nodes - n - 1].intensity + huffcodes[nodes - n - 2].intensity;

        // Appending to the pix_freq Array
        pix_freq[nextnode].intensity = sumpix;
        pix_freq[nextnode].Freq = sumprob;
        pix_freq[nextnode].left = &pix_freq[huffcodes[nodes - n - 2].arrloc];
        pix_freq[nextnode].right = &pix_freq[huffcodes[nodes - n - 1].arrloc];
        pix_freq[nextnode].code[0] = '\0';
        i = 0;

        // Sorting and Updating the
        // huffcodes array simultaneously
        // New position of the combined node
        while (sumprob <= huffcodes[i].Freq)
            i++;

        // Inserting the new node
        // in the huffcodes array
        for (k = nodes; k >= 0; k--)
        {
            if (k == i)
            {
                huffcodes[k].intensity = sumpix;
                huffcodes[k].Freq = sumprob;
                huffcodes[k].arrloc = nextnode;
            }
            else if (k > i)

            // Shifting the nodes below
            // the new node by 1
            // For inserting the new node
            // at the updated position k
            huffcodes[k] = huffcodes[k - 1];

        }
        n += 1;
        nextnode += 1;
    }





}

void AssignCode_cu(pixfreq<25> *pix_freq, int nodes, int totalnodes)
{
       // Assigning Code through
    // backtracking
    int i;
    char left = '0';
    char right = '1';
    for (i = totalnodes - 1; i >= nodes; i--)
    {
        if (pix_freq[i].left != NULL)
            strconcat(pix_freq[i].left->code, pix_freq[i].code, left);
        if (pix_freq[i].right != NULL)
            strconcat(pix_freq[i].right->code, pix_freq[i].code, right);
    }
}


#endif /* COMPRESS_HELPER_CU_H_ */
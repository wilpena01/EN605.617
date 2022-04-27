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

__device__ int shared_hist[256];
__shared__ int shared_node;
__shared__ float shared_prob;
__shared__ int shared_temp;

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
            cout<<"\n\nentre aqui<<\n\n";


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

void readBMPFILE_cu(int &width, int &height, int* &image)
{
      int i, j;
      char filename[] = "Lena.bmp";
      int offset, bpp = 0;
      long bmpsize = 0, bmpdataoff = 0;
      int temp = 0;
   // Reading the BMP File
      FILE* image_file;

      image_file = fopen(filename, "rb");
      if (image_file == NULL)
      {
         printf("Error Opening File!!");
         exit(1);
      }
      else
      {

         // Set file position of the
         // stream to the beginning
         // Contains file signature
         // or ID "BM"
         offset = 0;

         // Set offset to 2, which
         // contains size of BMP File
         offset = 2;

         fseek(image_file, offset, SEEK_SET);

         // Getting size of BMP File
         fread(&bmpsize, 4, 1, image_file);

         // Getting offset where the
         // pixel array starts
         // Since the information is
         // at offset 10 from the start,
         // as given in BMP Header
         offset = 10;

         fseek(image_file, offset, SEEK_SET);

         // Bitmap data offset
         fread(&bmpdataoff, 4, 1, image_file);

         // Getting height and width of the image
         // Width is stored at offset 18 and
         // height at offset 22, each of 4 bytes
         fseek(image_file, 18, SEEK_SET);

         fread(&width, 4, 1, image_file);

         fread(&height, 4, 1, image_file);

         // Number of bits per pixel
         fseek(image_file, 2, SEEK_CUR);

         fread(&bpp, 2, 1, image_file);

         // Setting offset to start of pixel data
         fseek(image_file, bmpdataoff, SEEK_SET);

         // Creating Image array
         image = (int*)malloc(height * width * sizeof(int));

         // Reading the BMP File
         // into Image Array
         for (i = 0; i < height; i++)
         {
            for (j = 0; j < width; j++)
            {
               int idx = (i*height) + j;
               fread(&temp, 3, 1, image_file);

               // the Image is a
               // 24-bit BMP Image
               temp = temp & 0x0000FF;
               //if(idx>height*width-5)
               //cout<<"idx = "<<idx<<"\t";
               image[idx] = static_cast<int>(temp);
            }
         }
      }
      fclose(image_file);
}

__device__ 
void copy_data_to_hist(int value, int idx)
{
	//copy from global to shared memory
	shared_hist[idx] = value;
}

__global__ 
void copy_data_from_shared(int* hist, int *Result, int *Block, int *Thread)
{
	//copy from global to shared memory
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;


	hist[idx] = shared_hist[idx];
   Result[idx] = idx;
   Block[idx]  = blockIdx.x+1;
	Thread[idx] = threadIdx.x;
}



//done
__global__ 
void initHist_cu(int* hist, int *Result, int *Block, int *Thread)
{
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   //copy_data_to_hist(0,idx);
   //copy_data_from_hist(hist,idx);
   shared_hist[idx] = 0;
   hist[idx] = 0;
   Result[idx] = 0;
   shared_node = 0;
   shared_prob = 1.0;
   shared_temp = 1;
   Block[idx]  = blockIdx.x;
	Thread[idx] = threadIdx.x;
  // __syncthreads();
}


//done
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

//done 
__global__
void nonZero_ocurrence_cu(int *Result, int *Block, int *Thread)
{
   // Finding number of
   // non-zero occurrences
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (shared_hist[idx] != 0)
      atomicAdd(&shared_node,1);
    __syncthreads();

   Result[idx] = shared_hist[idx] ;
   Block[idx]  = blockIdx.x+5;
	Thread[idx] = threadIdx.x;

}

//done i think
__global__
void minProp_cu(int* width, int* height, int *Result, int *Block, int *Thread)
{
    // Calculating minimum probability
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   int ptemp = 100 * shared_hist[idx] / (*height * *width);

   if (ptemp > 0)
      atomicMin(&shared_temp,ptemp);
   __syncthreads();
   shared_prob = shared_temp/100.0; 
   Result[idx] = shared_hist[idx] ;
   Block[idx]  = blockIdx.x+10;
	Thread[idx] = threadIdx.x;

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
                int* hist, int *height, int *width)
{
     // Initializing
   int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

   int j=0;
   int totpix = *height * *width;
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
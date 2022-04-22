// C Code for
// Image Compression
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
#include "Compress_Helper_cu.h"


using namespace std;

// Driver code
void compressionDriver()
{
   int width, height;
   int** image;
   int hist[256];
   int nodes, maxcodelen, totalnodes;
   float p = 1.0; 
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;

   readBMPFILE(width, height, image);
   ocurrence(hist, image, width, height);
   nonZero_ocurrence(hist, nodes);
   minProp(p, hist, width, height);
   maxcodelen = MaxLength(p) - 3;

   totalnodes = 2 * nodes - 1;
   pix_freq = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * totalnodes);
   huffcodes = (struct huffcode*)malloc(sizeof(struct huffcode) * nodes);

   InitStruct(pix_freq, huffcodes, hist, height, width);
   sortHist(huffcodes, nodes);
   BuildTree(pix_freq, huffcodes, nodes);
   AssignCode(pix_freq, nodes, totalnodes);
   PrintHuffmanCode(pix_freq, nodes);
   calBitLength(pix_freq, nodes);
   delete[] image; image = NULL;
}

void compressionDriver_CL()
{
   const int HistSize = 256;
   const int HistSize_Byte = sizeof(int) * HistSize;
   int width, height;
   int MaxSize;
   int* image;
   int* hist;
   int nodes = 0;
   int maxcodelen, totalnodes;
   float p = 1.0; 
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;

   const int hist_num_blocks     = 1;
   const int hist_num_threads    = HistSize;

   const int image_num_blocks    = 512;
   const int image_num_threads   = 512;

   int* g_image;
   int* g_width, *g_height, *g_nodes, *g_totalnodes;
   int* g_MaxSize;
   int* g_hist;
   float* g_p;
   pixfreq<25>* g_pix_freq;
   huffcode* g_huffcodes;

   int *gpu_Result, *gpu_Block, *gpu_Thread;
   int *cpu_Result;
   int *cpu_Block;
   int *cpu_Thread;
   
   readBMPFILE_cu(width, height, image);
   MaxSize = width * height;
/*
   for(int i=0; i<height; i++)
   {
      for(int j=0; j<width; j++)
      {
         if(image[i][j]>256)
            cout<<"image ="<<image[i][j]<<"   ";
      }
   }

*/

   int IMAGE_SIZE_IN_BYTES = sizeof(int) * MaxSize;

   cpu_Result = (int *)malloc(HistSize_Byte);
	cpu_Block = (int *)malloc(HistSize_Byte);
   cpu_Thread = (int *)malloc(HistSize_Byte);
   int* image2 = (int *)malloc(IMAGE_SIZE_IN_BYTES);
   hist = (int *)malloc(HistSize_Byte);

   cudaMalloc((void **)&g_image,       IMAGE_SIZE_IN_BYTES);
   cudaMalloc((void **)&g_width,       sizeof(int));
   cudaMalloc((void **)&g_height,      sizeof(int));
   cudaMalloc((void **)&g_hist,        HistSize_Byte);
   cudaMalloc((void **)&g_nodes,       sizeof(int));
   cudaMalloc((void **)&g_p,           sizeof(float));
   cudaMalloc((void **)&g_totalnodes,  sizeof(int));
   cudaMalloc((void **)&g_MaxSize,     sizeof(int));
   cudaMalloc((void **)&gpu_Result,    HistSize_Byte);
   cudaMalloc((void **)&gpu_Block,     HistSize_Byte);
   cudaMalloc((void **)&gpu_Thread,    HistSize_Byte);

   cudaMemcpy(g_image,      image,       IMAGE_SIZE_IN_BYTES,   cudaMemcpyHostToDevice);
   cudaMemcpy(g_width,      &width,      sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_height,     &height,     sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_hist,       hist,        HistSize_Byte,         cudaMemcpyHostToDevice);
   cudaMemcpy(g_nodes,      &nodes,      sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_p,          &p,          sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_totalnodes, &totalnodes, sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_MaxSize,    &MaxSize,    sizeof(int),           cudaMemcpyHostToDevice);
   //cudaMemcpy(gpu_Result,   cpu_Result,  HistSize_Byte,           cudaMemcpyHostToDevice);
   //cudaMemcpy(gpu_Block,    cpu_Block,   HistSize_Byte,           cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_Thread,   cpu_Thread,  HistSize_Byte,           cudaMemcpyHostToDevice);


  // cout<<"heiht = "<<height<<"\twidth = "<<width<<endl;
   initHist_cu<<<hist_num_blocks, hist_num_threads>>>(g_hist, gpu_Result, gpu_Block, gpu_Thread);
               
               cudaMemcpy(cpu_Result, gpu_Result, HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Block,  gpu_Block,  HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Thread, gpu_Thread, HistSize_Byte, cudaMemcpyDeviceToHost);
              // outputResult(cpu_Result, cpu_Block, cpu_Thread, 256);



   cudaMemcpy(image2,        g_image,       IMAGE_SIZE_IN_BYTES,  cudaMemcpyDeviceToHost);
   
   for(int i=0; i<width*height; i++)
   {
         if(image2[i]==241)
            cout<<"image = "<<image2[i]<<"\n";
      
   }
   //cout<<"heiht = "<<height<<"\twidth = "<<width<<endl;

   //cudaMemcpy(hist,        g_hist,       HistSize_Byte,  cudaMemcpyDeviceToHost);

   //for(int i=0; i<256; i++)
   //   cout<<"hist["<<i<<"] ="<<hist[i]<<"   ";

   //ocurrence_cu(hist, image, width, height)   ;
   ocurrence_cu<<<image_num_blocks,image_num_threads>>>(g_image);
   //ocurrence_cu<<<image_num_blocks,image_num_threads>>>(g_image);
   cudaDeviceSynchronize();
   //cudaMemcpy(g_hist,       hist,        HistSize_Byte,         cudaMemcpyHostToDevice);

   copy_data_from_shared<<<hist_num_blocks, hist_num_threads>>>(g_hist, gpu_Result, gpu_Block, gpu_Thread);



               cudaMemcpy(cpu_Result, gpu_Result, HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Block,  gpu_Block,  HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Thread, gpu_Thread, HistSize_Byte, cudaMemcpyDeviceToHost);
               //outputResult(cpu_Result, cpu_Block, cpu_Thread, 256);

   //for(int i=0; i<256; i++)
   //   cout<<"hist["<<i<<"] ="<<hist[i]<<"\n";

   nonZero_ocurrence_cu<<<hist_num_blocks, hist_num_threads>>>(gpu_Result, gpu_Block, gpu_Thread);

               cudaMemcpy(cpu_Result, gpu_Result, HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Block,  gpu_Block,  HistSize_Byte, cudaMemcpyDeviceToHost);
               cudaMemcpy(cpu_Thread, gpu_Thread, HistSize_Byte, cudaMemcpyDeviceToHost);
               outputResult(cpu_Result, cpu_Block, cpu_Thread, 256);












   minProp_cu<<<hist_num_blocks, hist_num_threads>>>(g_p, g_hist,g_width,g_height );
   //maxcodelen = MaxLength_cu(p) - 3;
   totalNode<<<1,1>>>(g_totalnodes,g_nodes);

   pix_freq  = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * totalnodes);
   huffcodes = (struct huffcode*)malloc(sizeof(struct huffcode) * nodes);

   cudaMalloc((void **)&g_pix_freq,   sizeof(pixfreq<25>*) * totalnodes);
   cudaMalloc((void **)&g_huffcodes,  sizeof(struct huffcode) * nodes);

   InitStruct_cu<<<hist_num_blocks, hist_num_threads>>>(g_pix_freq, g_huffcodes, g_hist, g_height, g_width);
/*
   cudaMemcpy(image,        g_image,      IMAGE_SIZE_IN_BYTES,   cudaMemcpyDeviceToHost);
   cudaMemcpy(&width,       g_width,      sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(&height,      g_height,     sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(&hist,        g_hist,       HistSize*sizeof(int),  cudaMemcpyDeviceToHost);
   cudaMemcpy(&nodes,       g_nodes,      sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(&p,           g_p,          sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(&totalnodes,  g_totalnodes, sizeof(int),           cudaMemcpyDeviceToHost);

*/
   sortHist_cu(huffcodes, nodes);
   BuildTree_cu(pix_freq, huffcodes, nodes);
   AssignCode_cu(pix_freq, nodes, totalnodes);
   PrintHuffmanCode(pix_freq, nodes);
   calBitLength(pix_freq, nodes);
   delete[] image; image = NULL;

   cudaFree(g_image);
   cudaFree(g_width);
   cudaFree(g_height);
   cudaFree(g_hist);
   cudaFree(g_nodes);
   cudaFree(g_totalnodes);
   cudaFree(g_pix_freq);
   cudaFree(g_huffcodes);
   cudaFree(g_p);
   cudaFree(g_MaxSize);
   cudaFree(gpu_Result);
	cudaFree(gpu_Block);
	cudaFree(gpu_Thread);
   delete[] cpu_Result;
	delete[] cpu_Block;
	delete[] cpu_Thread;
   delete[] image2;
   delete[] hist;

}

int main()
{
   cout<<"Using Local CPU"<<endl;
   //compressionDriver();

   cout<<"\n\nUsing GPU"<<endl;
   compressionDriver_CL();

   return 0;

}

   // Encode the Image
   //int pix_val;
   //int l;

   // Writing the Huffman encoded
   // Image into a text file

   /*ofstream imagehuff; 
   imagehuff.open ("encoded_image.bin", ios::out | ios::app | ios::binary);
   cout<<"bien aqui<"<<endl;

   for (i = 0; i < height; i++)
   {
   for (j = 0; j < width; j++)
   {
      pix_val = image[i][j];
      cout<<"image[" <<i<<"]["<<j<<"] ="<<image[i][j]<<" ";
      for (l = 0; l < nodes; l++)
      {
         if (pix_val == pix_freq[l].intensity)
            imagehuff<< pix_freq[l].code;
      }
   }
   cout<<endl;
   }
   */
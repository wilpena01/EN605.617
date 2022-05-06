// C Code for
// Image Compression
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>
#include <chrono>

#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "ImageIO.h"
#include "Exceptions.h"
#include <npp.h>

#include "Compress_Helper.h"
#include "Compress_Helper_cu.h"

using namespace std;
using namespace std::chrono;

// Driver code
void compressionDriver()
{
   int width, height;
   int** image;
   int hist[256];
   int nodes, totalnodes;
   float p = 1.0; 
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;

   readBMPFILE(width, height, image);
   ocurrence(hist, image, width, height);
   nonZero_ocurrence(hist, nodes);
   minProp(p, hist, width, height);
   //maxcodelen = MaxLength(p) - 3; in this case 25

   totalnodes = 2 * nodes - 1;
   pix_freq = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * totalnodes);
   huffcodes = (struct huffcode*)malloc(sizeof(struct huffcode) * nodes);

   InitStruct(pix_freq, huffcodes, hist, height, width);
   sortHist(huffcodes, nodes);
   BuildTree(pix_freq, huffcodes, nodes);
   AssignCode(pix_freq, nodes, totalnodes);
   PrintHuffmanCode(pix_freq, nodes);
   calBitLength(pix_freq, nodes);
   free(image);
}

void compressionDriver_cu()
{
   const int HistSize            = 256;
   const int image_num_blocks    = 512;
   const int image_num_threads   = 512;
   const int hist_num_blocks     = 1;
   const int hist_num_threads    = HistSize;
   const int HistSize_Byte       = sizeof(int) * HistSize;

   int width, height, MaxSize, *image, *hist, *totalnodes, *nodes;
   int* g_image, *g_width, *g_height, *g_nodes, *g_totalnodes, *g_hist;
   int *gpu_Result, *gpu_Block, *gpu_Thread;
   int *cpu_Result, *cpu_Block, *cpu_Thread;

   pixfreq<25> *pix_freq;
   pixfreq<25> *g_pix_freq;
   huffcode    *g_huffcodes;
   
   readBMPFILE_cu(width, height, image);
   MaxSize = width * height;

   int IMAGE_SIZE_IN_BYTES = sizeof(int) * MaxSize;

   allocHost(cpu_Result, cpu_Block, cpu_Thread, hist, totalnodes, nodes,
             HistSize_Byte);

   allocDevice(g_image, g_width, g_height, g_hist, g_nodes, g_totalnodes,
               gpu_Result, gpu_Block, gpu_Thread, IMAGE_SIZE_IN_BYTES, 
               HistSize_Byte);

   HostToDevice(g_image, g_width, g_height, g_hist, g_nodes, g_totalnodes,
                image, width, height, hist, nodes, totalnodes,
                IMAGE_SIZE_IN_BYTES, HistSize_Byte);


   initHist_cu<<<hist_num_blocks, hist_num_threads>>>(g_hist, gpu_Result, 
                                                      gpu_Block, gpu_Thread);

   ocurrence_cu<<<image_num_blocks,image_num_threads>>>(g_image);

   copy_data_from_shared<<<hist_num_blocks, hist_num_threads>>>(g_hist, gpu_Result, 
                                                               gpu_Block, gpu_Thread);

   nonZero_ocurrence_cu<<<hist_num_blocks, hist_num_threads>>>(g_nodes,gpu_Result, 
                                                               gpu_Block, gpu_Thread);

   minProp_cu<<<hist_num_blocks, hist_num_threads>>>(g_width, g_height, gpu_Result, 
                                                     gpu_Block, gpu_Thread);

   //maxcodelen = MaxLength_cu(p) - 3;
   totalNode<<<1,1>>>(g_totalnodes, g_nodes, gpu_Result, gpu_Block, gpu_Thread);
   cudaMemcpy(totalnodes, g_totalnodes, sizeof(int), cudaMemcpyDeviceToHost);
   cudaMemcpy(nodes,      g_nodes,      sizeof(int), cudaMemcpyDeviceToHost);

   pix_freq  = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * *totalnodes);

   cudaMalloc((void **)&g_pix_freq,   sizeof(pixfreq<25>) * *totalnodes);
   cudaMalloc((void **)&g_huffcodes,  sizeof(struct huffcode) * *nodes);

   InitStruct_cu<<<hist_num_blocks, hist_num_threads>>>(g_pix_freq, g_huffcodes, g_height, 
                                                        g_width, gpu_Result, gpu_Block, 
                                                        gpu_Thread);

   sortHist_cu<<<1,1>>>(g_huffcodes, g_nodes, gpu_Result, gpu_Block, gpu_Thread);

   BuildTree_cu<<<1,1>>>(g_pix_freq, g_huffcodes, g_nodes, gpu_Result, gpu_Block, 
                         gpu_Thread);
                         
   AssignCode_cu<<<1,*totalnodes - 1>>>(g_pix_freq, g_nodes, g_totalnodes, gpu_Result, 
                                        gpu_Block, gpu_Thread);
   
   cudaMemcpy(pix_freq,  g_pix_freq, sizeof(pixfreq<25>) * *totalnodes, cudaMemcpyDeviceToHost);
   PrintHuffmanCode(pix_freq, *nodes);
   calBitLength(pix_freq, *nodes);

   freeMem(g_image,g_width, g_height, g_hist, g_nodes, g_totalnodes,
           g_pix_freq, g_huffcodes, gpu_Result, gpu_Block, gpu_Thread,
           hist, cpu_Result, cpu_Block, cpu_Thread, image);

}

int main()
{
   // Main Driver
   
  
   cout<<"Using Local CPU"<<endl;
   auto start = high_resolution_clock::now();
   compressionDriver();
   auto end = high_resolution_clock::now();
   auto duration = chrono::duration<double>(end-start);

   cout<<"\nElapse Time without CUDA = "<<duration.count()<<"microseconds "<<endl;

   cout<<"\n\nUsing GPU"<<endl;
   start = high_resolution_clock::now();
   compressionDriver_cu();
   end = high_resolution_clock::now();
   duration = chrono::duration<double>(end-start);
   cout<<"Elapse Time with CUDA = "<<duration.count()<<"microseconds"<<endl;
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
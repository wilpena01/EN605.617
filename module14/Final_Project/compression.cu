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
   int i,j;
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
   int i,j;
   int width, height;
   int** image;
   int hist[HistSize];
   int nodes = 0;
   int maxcodelen, totalnodes;
   float p = 1.0; 
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;

   const int hist_num_blocks     = 1;
   const int hist_num_threads    = HistSize;

   const int image_num_blocks    = 2;
   const int image_num_threads   = 256;

   int *g_image;
   int* g_width, g_height, g_hist, g_nodes, g_totalnodes;
   float* g_p = 1.0;
   pixfreq<25>* g_pix_freq;
   huffcode* g_huffcodes;
   
   LoadImagePGM(width, height, image);

   int IMAGE_SIZE_IN_BYTES = sizeof(int) * width * height;
   
   cudaMalloc((void **)&g_image,       IMAGE_SIZE_IN_BYTES);
   cudaMalloc((void **)&g_width,       sizeof(int));
   cudaMalloc((void **)&g_height,      sizeof(int));
   cudaMalloc((void **)&g_hist,        HistSize*sizeof(int));
   cudaMalloc((void **)&g_nodes,       sizeof(int));
   cudaMalloc((void **)&g_p,           sizeof(float));
   cudaMalloc((void **)&g_totalnodes,  sizeof(int));


   cudaMemcpy(g_image,      image,      IMAGE_SIZE_IN_BYTES,   cudaMemcpyHostToDevice);
   cudaMemcpy(g_width,      width,      sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_height,     height,     sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_hist,       hist,       HistSize*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(g_nodes,      node,       sizeof(int),           cudaMemcpyHostToDevice);
   cudaMemcpy(g_totalnodes, totalnodes, sizeof(int),           cudaMemcpyHostToDevice);


   initHist_cu<<<hist_num_blocks, hist_num_threads>>>(g_hist);
   ocurrence_cu<<<image_num_blocks,image_num_threads>>>(g_hist, g_image);
   nonZero_ocurrence_cu<<<hist_num_blocks, hist_num_threads>>>(g_hist, g_nodes);
   minProp_cu<<<hist_num_blocks, hist_num_threads>>>(g_p, g_hist, g_width, g_height);
   //maxcodelen = MaxLength_cu(p) - 3;
   totalNode<<<1,1>>>(g_totalnodes,g_nodes);

   pix_freq  = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * totalnodes);
   huffcodes = (struct huffcode*)malloc(sizeof(struct huffcode) * nodes);

   cudaMalloc((void **)&g_pix_freq,   sizeof(pixfreq<25>*) * totalnodes);
   cudaMalloc((void **)&g_huffcodes,  sizeof(struct huffcode) * nodes);

   InitStruct_cu<<<hist_num_blocks, hist_num_threads>>>(g_pix_freq, g_huffcodes, g_hist, g_height, g_width);

   cudaMemcpy(image,       g_image,      IMAGE_SIZE_IN_BYTES,   cudaMemcpyDeviceToHost);
   cudaMemcpy(width,       g_width,      sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(height,      g_height,     sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(hist,        g_hist,       HistSize*sizeof(int),  cudaMemcpyDeviceToHost);
   cudaMemcpy(node,        g_nodes,      sizeof(int),           cudaMemcpyDeviceToHost);
   cudaMemcpy(totalnodes,  g_totalnodes, sizeof(int),           cudaMemcpyDeviceToHost);


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
}

int main()
{
   cout<<"Using Local CPU"<<endl;
   compressionDriver();

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
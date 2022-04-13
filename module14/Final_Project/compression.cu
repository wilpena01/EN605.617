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
}

void compressionDriver_CL()
{
   int i,j;
   int width, height;
   int** image_cl;
   int hist[256];
   int nodes, maxcodelen, totalnodes;
   float p = 1.0; 
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;
   
   LoadImagePGM(width, height, image_cl);
   ocurrence(hist, image_cl, width, height);
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
   free(image_cl);
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
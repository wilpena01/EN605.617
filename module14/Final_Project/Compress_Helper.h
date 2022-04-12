#ifndef COMPRESS_HELPER_H_
#define COMPRESS_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>

using namespace std;

void readBMPFILE(int &width, int &height, int** &image)
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
         image = (int**)malloc(height * sizeof(int*));

         for (i = 0; i < height; i++)
         {
            image[i] = (int*)malloc(width * sizeof(int));
         }

         // Reading the BMP File
         // into Image Array
         for (i = 0; i < height; i++)
         {
            for (j = 0; j < width; j++)
            {
               fread(&temp, 3, 1, image_file);

               // the Image is a
               // 24-bit BMP Image
               temp = temp & 0x0000FF;
               image[i][j] = temp;
            }
         }
      }
}

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
            if(image[i][j]>=256)
                cout<<"Este es el problema ="<<image[i][j]<<endl;
            hist[image[i][j]] += 1;
        }
    }
}

void nonZero_ocurrence(int* hist, int &node)
{
    // Finding number of
    // non-zero occurrences
    node=0;
    for (int i = 0; i < 256; i++)
      if (hist[i] != 0)
         node += 1;
}

void minProp(int* hist, int width, int height)
{
    // Calculating minimum probability
    float p = 1.0, ptemp;
    for (int i = 0; i < 256; i++)
    {
        ptemp = (hist[i] / (float)(height * width));
        if (ptemp > 0 && ptemp <= p)
            p = ptemp;
    }
}


#endif /* COMPRESS_HELPER_H_ */
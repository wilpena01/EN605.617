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

using namespace std;

void LoadImagePGM(int &width, int &height, int** &image)
{
   int i,j;
   npp::ImageCPU_8u_C1 hostImage;
   string name = "Lena.pgm";

   ifstream inputfile(name.data(), std::ifstream::in);

   if (inputfile.good())
   {
      cout << "assignmentNPP opened: <" << name.data() << "> successfully!" << endl;
      inputfile.close();
   }
   else
   {
      cout << "assignmentNPP unable to open: <" << name.data() << ">" << endl;
      inputfile.close();
      exit(EXIT_FAILURE);
   }



   // load gray-scale image from disk
   npp::loadImage(name, hostImage);

   height = hostImage.height();
   width = hostImage.width();

   // Creating Image array
   image = (int**)malloc(height * sizeof(int*));

   for (i = 0; i < height; i++)
   {
      image[i] = (int*)malloc(width * sizeof(int*));
   }

   for(i=0; i<height; i++)
      for(j=0; j<width;j++)
         image[i][j] = static_cast<int>(*hostImage.data(i,j));

}



#endif /* COMPRESS_HELPER_CU_H_ */
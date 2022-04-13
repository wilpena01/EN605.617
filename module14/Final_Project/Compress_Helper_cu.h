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

void LoadImagePGM(int &width, int &height, int** &image_cl)
{
   int i,j;

   string name = "Lena.pgm";
   FILE* inputfile = fopen(name, "rb");

   if (image_file != NULL)
   {
      cout << "assignmentNPP opened: <" << name.data() << "> successfully!" << endl;
   }
   else
   {
      cout << "assignmentNPP unable to open: <" << name.data() << ">" << endl;
      fclose(inputfile);
      exit(EXIT_FAILURE);
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

}



#endif /* COMPRESS_HELPER_CU_H_ */
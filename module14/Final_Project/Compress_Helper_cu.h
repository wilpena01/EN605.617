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


void LoadImagePGM(npp::ImageCPU_8u_C1 &hostImage)
{
    cout<<"Start...\n";
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

   // declare a host image object for an 8-bit grayscale image
    

    // load gray-scale image from disk
    npp::loadImage(name, hostImage);

}



#endif /* COMPRESS_HELPER_CU_H_ */
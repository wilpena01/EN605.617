#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>


int main()
{
    printf("Starting...\n\n");

    std::string sFilename = "mary.jpg";


    // if we specify the filename at the command line, then we only test sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
        std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
        file_errors = 0;
        infile.close();
    }
    else
    {
        std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
        file_errors++;
        infile.close();
    }

    if (file_errors > 0)
    {
        exit(EXIT_FAILURE);
    }

     

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
        sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_boxFilter.pgm";

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        printf("aqui no es...\n\n");

    // create struct with box-filter mask size
    NppiSize oMaskSize = {5, 5};

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // set anchor point inside the mask to (oMaskSize.width / 2, oMaskSize.height / 2)
    // It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    // run box filter
    nppiFilterBoxBorder_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                                    oSrcSize, oSrcOffset,
                                                    oDeviceDst.data(), oDeviceDst.pitch(),
                                                    oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE) ;

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

}
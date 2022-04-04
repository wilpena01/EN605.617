#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
#include <npp.h>

#include <string.h>
#include <fstream>
#include <iostream>

void NPPTest()
{

    cout<<"Start...\n";
    std::string name = "Lena.pgm";

    std::ifstream inputfile(name.data(), std::ifstream::in);

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

    string result = name;
    std::string::size_type dot = result.rfind('.');

    if (dot != std::string::npos)
    {
        result = result.substr(0, dot);
    }

    result += "_boxFilter.pgm";

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 hostImage;

    // load gray-scale image from disk
    npp::loadImage(name, hostImage);

    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 deviceImage(hostImage);


    // create struct with box-filter mask size
    NppiSize FilterSize = {5, 5};

    NppiSize oSrcSize = {(int)deviceImage.width(), (int)deviceImage.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize ROISize = {(int)deviceImage.width() , (int)deviceImage.height() };
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 device_imageDes(ROISize.width, ROISize.height);
    // set anchor point inside the mask to (FilterSize.width / 2, FilterSize.height / 2)
    // It should round down when odd
    NppiPoint host_anchor = {FilterSize.width / 2, FilterSize.height / 2};

    // run box filter
    nppiFilterBoxBorder_8u_C1R(deviceImage.data(), deviceImage.pitch(),
                                                    oSrcSize, oSrcOffset,
                                                    device_imageDes.data(), device_imageDes.pitch(),
                                                    ROISize, FilterSize, host_anchor, NPP_BORDER_REPLICATE) ;

    // declare a host image for the result
    npp::ImageCPU_8u_C1 host_imageDes(device_imageDes.size());
    // and copy the device result data into it
    device_imageDes.copyTo(host_imageDes.data(), host_imageDes.pitch());

    saveImage(result, host_imageDes);
    cout << "Saved image: " << result << endl;
}

int main()
{
    NPPTest();

}
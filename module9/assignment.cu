#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>

#include "./UtilNPP/ImagesCPU.h"
#include "UtilNPP/ImagesNPP.h"
#include "UtilNPP/ImageIO.h"
#include "UtilNPP/Exceptions.h"
#include "UtilNPP/npp.h"


#include <chrono>
#include <iostream>
#include <string>
#include "Utilities.h"

using namespace std;
using namespace std::chrono;

#define N 30

void addAnalysis(thrust::host_vector<int> A, thrust::host_vector<int> B, 
                 thrust::host_vector<int> &Z, microseconds &d1, microseconds &d2)
{
    // compute Z = X + Y
    thrust::device_vector<int> g_X = A;
    thrust::device_vector<int> g_Y = B;
    thrust::device_vector<int> temp;

    auto start = high_resolution_clock::now();
    thrust::transform(g_X.begin(), g_X.end(), g_Y.begin(), g_Y.begin(), thrust::plus<int>());
    auto stop = high_resolution_clock::now();
    d1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    for(int i = 0; i<N ; i++)
    {
        temp.push_back(g_X[i] + g_Y[i]);
    }
    stop = high_resolution_clock::now();
    d2 = duration_cast<microseconds>(stop - start);
    Z = g_Y;
}

void subAnalysis(thrust::host_vector<int> A, thrust::host_vector<int> B, 
                 thrust::host_vector<int> &Z, microseconds &d1, microseconds &d2)
{
    // compute Z = X - Y
    thrust::device_vector<int> g_X = A;
    thrust::device_vector<int> g_Y = B;
    thrust::device_vector<int> temp;

    auto start = high_resolution_clock::now();
    thrust::transform(g_X.begin(), g_X.end(), g_Y.begin(), g_Y.begin(), thrust::minus<int>());
    auto stop = high_resolution_clock::now();
    d1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    for(int i = 0; i<N ; i++)
    {
        temp.push_back(g_X[i] - g_Y[i]);
    }
    stop = high_resolution_clock::now();
    d2 = duration_cast<microseconds>(stop - start);
    Z = g_Y;
}

void mulAnalysis(thrust::host_vector<int> A, thrust::host_vector<int> B, 
                 thrust::host_vector<int> &Z, microseconds &d1, microseconds &d2)
{
    // compute Z = X * Y
    thrust::device_vector<int> g_X = A;
    thrust::device_vector<int> g_Y = B;
    thrust::device_vector<int> temp;

    auto start = high_resolution_clock::now();
    thrust::transform(g_X.begin(), g_X.end(), g_Y.begin(), g_Y.begin(), thrust::multiplies<int>());
    auto stop = high_resolution_clock::now();
    d1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    for(int i = 0; i<N ; i++)
    {
        temp.push_back(g_X[i] * g_Y[i]);
    }
    stop = high_resolution_clock::now();
    d2 = duration_cast<microseconds>(stop - start);
    Z = g_Y;
}

void modAnalysis(thrust::host_vector<int> A, thrust::host_vector<int> B, 
                 thrust::host_vector<int> &Z, microseconds &d1, microseconds &d2)
{
    // compute Z = X % Y
    thrust::device_vector<int> g_X = A;
    thrust::device_vector<int> g_Y = B;
    thrust::device_vector<int> temp;

    auto start = high_resolution_clock::now();
    thrust::transform(g_X.begin(), g_X.end(), g_Y.begin(), g_Y.begin(), thrust::modulus<int>());
    auto stop = high_resolution_clock::now();
    d1 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    for(int i = 0; i<N ; i++)
    {
         if(g_Y[i] != 0)
            temp.push_back(g_X[i] * g_Y[i]);
        else
            temp.push_back(-999);
    }
    stop = high_resolution_clock::now();
    d2 = duration_cast<microseconds>(stop - start);
    Z = g_Y;
}

void thrus()
{
    // allocate two host_vectors with N elements
    thrust::host_vector<int> X(N);
    thrust::host_vector<int> Y(N);

    // fill X & Y with randon numbers
    for(int i = 0; i<N ; i++)
    {
        X[i]=rand() % 10+1;
        Y[i]=rand() % 10+1;
    }
    
    microseconds add_d1, add_d2;
    microseconds sub_d1, sub_d2;
    microseconds mul_d1, mul_d2;
    microseconds mod_d1, mod_d2;

    thrust::host_vector<int> add(N);
    thrust::host_vector<int> sub(N);
    thrust::host_vector<int> mul(N);
    thrust::host_vector<int> mod(N); 

    addAnalysis(X,Y,add,add_d1,add_d2);
    subAnalysis(X,Y,sub,sub_d1,sub_d2);
    mulAnalysis(X,Y,mul,mul_d1,mul_d2);
    modAnalysis(X,Y,mod,mod_d1,mod_d2);

    string str[] ={"Thrus"};
    cout<<"X = ";   outputVec(X);
    cout<<"Y = ";   outputVec(Y);
    cout<<"add = "; outputVec(add); outputTime(add_d1,add_d2,str);
    cout<<"sub = "; outputVec(sub); outputTime(sub_d1,sub_d2,str);
    cout<<"mul = "; outputVec(mul); outputTime(mul_d1,mul_d2,str);
    cout<<"mod = "; outputVec(mod); outputTime(mod_d1,mod_d2,str);  
}








void NPP()
{
    printf("%s Starting...\n\n", argv[0]);

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
                                                    oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE) );

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

}


int main()
{
    //Main driver
    thrus();
    NPP();

 

    return 0;    
}
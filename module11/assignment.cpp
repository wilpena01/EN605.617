//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


// assignment.cpp
//
//    This is a simple example that demonstrates OpenCL platform, device, and context
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "Signal_mask.h"
#include <chrono>

using namespace std;
using namespace std::chrono;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

void outputT(microseconds d1, microseconds d2, microseconds d3, 
             microseconds d4, microseconds d5);

void setUp(cl_int &errNum,
    cl_uint &numPlatforms, cl_uint &numDevices,
    cl_platform_id * platformIDs,
	cl_device_id * deviceIDs,
    cl_context &context,
	cl_command_queue &queue,
	cl_program &program,
	cl_kernel &kernel,
	cl_mem &inputSignalBuffer, 
    cl_mem &outputSignalBuffer, 
    cl_mem &maskBuffer)
{

    // First, select an OpenCL platform to run on.  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}
    // Next, create an OpenCL context on the selected platform.  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

	// Create kernel object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		0,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
}


void runNominal(cl_mem& inputSignalBuffer, cl_mem& outputSignalBuffer, cl_mem& maskBuffer,
cl_kernel& kernel, cl_int& errNum, cl_command_queue& queue, cl_context& context)
{
    createBuffer(inputSignalBuffer, outputSignalBuffer, maskBuffer,
    context, errNum);
    exeKernel(errNum, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer, queue);
}

void runNominal100(cl_mem& inputSignalBuffer, cl_mem& outputSignalBuffer, cl_mem& maskBuffer,
cl_kernel& kernel, cl_int& errNum, cl_command_queue& queue, cl_context& context)
{
    createBuffer100(inputSignalBuffer, outputSignalBuffer, maskBuffer,
    context, errNum);
    exeKernel100(errNum, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer, queue);
}

void runNominal75(cl_mem& inputSignalBuffer, cl_mem& outputSignalBuffer, cl_mem& maskBuffer,
cl_kernel& kernel, cl_int& errNum, cl_command_queue& queue, cl_context& context)
{
    createBuffer75(inputSignalBuffer, outputSignalBuffer, maskBuffer,
    context, errNum);
    exeKernel75(errNum, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer, queue);
}

void runNominal50(cl_mem& inputSignalBuffer, cl_mem& outputSignalBuffer, cl_mem& maskBuffer,
cl_kernel& kernel, cl_int& errNum, cl_command_queue& queue, cl_context& context)
{
    createBuffer50(inputSignalBuffer, outputSignalBuffer, maskBuffer,
    context, errNum);
    exeKernel50(errNum, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer, queue);
}

void runNominal25(cl_mem& inputSignalBuffer, cl_mem& outputSignalBuffer, cl_mem& maskBuffer,
cl_kernel& kernel, cl_int& errNum, cl_command_queue& queue, cl_context& context)
{
    createBuffer25(inputSignalBuffer, outputSignalBuffer, maskBuffer,
    context, errNum);
    exeKernel25(errNum, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer, queue);
}

///
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms, numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer, outputSignalBuffer, maskBuffer;

    setUp(errNum, numPlatforms, numDevices, platformIDs, deviceIDs,
    context, queue, program, kernel, inputSignalBuffer, outputSignalBuffer, 
    maskBuffer);

    std::chrono::microseconds d1,d2,d3,d4,d5;

    auto start = high_resolution_clock::now();
    runNominal(inputSignalBuffer, outputSignalBuffer, maskBuffer, kernel,
    errNum, queue, context);
    auto stop = high_resolution_clock::now(); 
    d1 = duration_cast<microseconds>(stop - start);

    // Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

    start = high_resolution_clock::now();
    runNominal100(inputSignalBuffer, outputSignalBuffer, maskBuffer, kernel,
    errNum, queue, context);
    stop = high_resolution_clock::now(); d2 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    runNominal75(inputSignalBuffer, outputSignalBuffer, maskBuffer, kernel,
    errNum, queue, context);
    stop = high_resolution_clock::now(); d3 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    runNominal50(inputSignalBuffer, outputSignalBuffer, maskBuffer, kernel,
    errNum, queue, context);
    stop = high_resolution_clock::now(); d4 = duration_cast<microseconds>(stop - start);

    start = high_resolution_clock::now();
    runNominal25(inputSignalBuffer, outputSignalBuffer, maskBuffer, kernel,
    errNum, queue, context);
    stop = high_resolution_clock::now(); d5 = duration_cast<microseconds>(stop - start);

    outputT(d1,d2,d3,d4,d5);

    std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}


void outputT(microseconds d1, microseconds d2, microseconds d3, 
             microseconds d4, microseconds d5)
{
    std::cout<<"Signal 49x49 and filter 7x7   - Elapse Time = "<<d1.count()<<" microsecond\n";
    std::cout<<"Signal 49x49 and filter 24x24 - Elapse Time = "<<d2.count()<<" microsecond\n";
    std::cout<<"Signal 49x49 and filter 12x12 - Elapse Time = "<<d3.count()<<" microsecond\n";
    std::cout<<"Signal 49x49 and filter 8x8   - Elapse Time = "<<d4.count()<<" microsecond\n";
    std::cout<<"Signal 49x49 and filter 6x6   - Elapse Time = "<<d5.count()<<" microsecond\n";

}
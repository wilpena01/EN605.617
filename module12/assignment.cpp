//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

int NUM_BUFFER_ELEMENTS = 16;
#define NUM_SUBBUFFER_ELEMENTS 2

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int * inputOutput;
    float Average;
    float total = (float)NUM_BUFFER_ELEMENTS;
    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    auto start = high_resolution_clock::now();

    for (int i = 1; i < argc; i++)
    {
        std::string input(argv[i]);

        if (!input.compare("--platform"))
        {
            input = std::string(argv[++i]);
            std::istringstream buffer(input);
            buffer >> platform;
        }
        else if (!input.compare("--useMap"))
        {
            useMap = true;
        }
        else
        {
            std::cout << "usage: --platform n --useMap" << std::endl;
            return 0;
        }
    }


    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

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
        "-I.",
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

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);

    cl_mem sum = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float),
        0,
        &errNum);

    cl_mem arraySize = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float),
        0,
        &errNum);
    //cl_mem arraySize[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_READ_WRITE,
     //                              sizeof(int), NUM_BUFFER_ELEMENTS, NULL);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
    for (unsigned int i = 0; i < numDevices; i++)
    {
        cl_buffer_region region = 
            {
                NUM_SUBBUFFER_ELEMENTS * i * sizeof(int), 
                NUM_SUBBUFFER_ELEMENTS * sizeof(int)
            };
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // Create command queues
    for (unsigned int i = 0; i < numDevices; i++)
    {
        InfoDevice<cl_device_type>::display(
            deviceIDs[i], 
            CL_DEVICE_TYPE, 
            "CL_DEVICE_TYPE");

        cl_command_queue queue = 
            clCreateCommandQueue(
                context,
                deviceIDs[i],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "average",
            &errNum);
        checkErr(errNum, "clCreateKernel(average)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem),    (void *)&buffers[i]); 
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem),    (void *)&arraySize);
        errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem),    &sum);

        checkErr(errNum, "clSetKernelArg(average)");

        kernels.push_back(kernel);
    }

    if (useMap) 
    {
        
        errNum = clEnqueueReadBuffer(
            queues[numDevices - 1], 
            arraySize, 
            CL_TRUE,    
            0, 
            sizeof(float), 
            &total,
            0, 
            NULL, 
            NULL);

        errNum = clEnqueueReadBuffer(
            queues[numDevices - 1], 
            sum, 
            CL_TRUE,    
            0, 
            sizeof(float), 
            &Average,
            0, 
            NULL, 
            NULL);



        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            CL_MAP_WRITE,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);

             
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            mapPtr[i] = inputOutput[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            main_buffer,
            mapPtr,
            0,
            NULL,
            NULL);
        
        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            arraySize,
            &total,
            0,
            NULL,
            NULL);
            
        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            sum,
            &Average,
            0,
            NULL,
            NULL);


        checkErr(errNum, "clEnqueueUnmapMemObject(..)");
    }
    else 
    {
        // Write input data
        errNum = clEnqueueWriteBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput,
            0,
            NULL,
            NULL);

         errNum = clEnqueueWriteBuffer(
            queues[numDevices - 1],
            arraySize,
            CL_TRUE,
            0,
            sizeof(float),
            (void*)&total,
            0,
            NULL,
            NULL);
            
        errNum = clEnqueueWriteBuffer(
            queues[numDevices - 1],
            sum,
            CL_TRUE,
            0,
            sizeof(float),
            (void*)&Average,
            0,
            NULL,
            NULL);


    }
    std::vector<cl_event> events;
    // call kernel for each device
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_BUFFER_ELEMENTS;

        errNum = clEnqueueNDRangeKernel(
            queues[i], 
            kernels[i], 
            1, 
            NULL,
            (const size_t*)&gWI, 
            (const size_t*)NULL, 
            0, 
            0, 
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

    if (useMap)
    {
        cl_int * mapPtr = (cl_int*) clEnqueueMapBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            CL_MAP_READ,
            0,
            sizeof(cl_int) * NUM_BUFFER_ELEMENTS * numDevices,
            0,
            NULL,
            NULL,
            &errNum);
        checkErr(errNum, "clEnqueueMapBuffer(..)");

        for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
        {
            inputOutput[i] = mapPtr[i];
        }

        errNum = clEnqueueUnmapMemObject(
            queues[numDevices - 1],
            main_buffer,
            mapPtr,
            0,
            NULL,
            NULL);

        clFinish(queues[numDevices - 1]);
    }
    else 
    {
        // Read back computed data
        clEnqueueReadBuffer(
            queues[numDevices - 1],
            main_buffer,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput,
            0,
            NULL,
            NULL);

            clEnqueueReadBuffer(
            queues[numDevices - 1],
            arraySize,
            CL_TRUE,
            0,
            sizeof(float),
            (void*)&total,
            0,
            NULL,
            NULL);
            
            clEnqueueReadBuffer(
            queues[numDevices - 1],
            sum,
            CL_TRUE,
            0,
            sizeof(float),
            (void*)&Average,
            0,
            NULL,
            NULL);

            
    }

    auto stop = high_resolution_clock::now(); 
    auto d = duration_cast<microseconds>(stop - start);


    // Display output in rows
    for (unsigned i = 0; i < numDevices; i++)
    {
        for (unsigned elems = i * NUM_BUFFER_ELEMENTS; elems < ((i+1) * NUM_BUFFER_ELEMENTS); elems++)
        {
            std::cout << " " << inputOutput[elems];
        }

        std::cout << std::endl;
    }
    std::cout << "Average = " <<Average<< std::endl;
    std::cout << "Elapse Time = " <<d.count()<< std::endl;

    return 0;
}

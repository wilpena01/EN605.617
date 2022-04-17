#include "Signal_mask.h"
#include <iostream>

inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

void createBuffer(cl_mem &inputSignalBuffer,
	cl_mem &outputSignalBuffer,
	cl_mem &maskBuffer,  cl_context &context, cl_int &errNum)
{
    // Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");
}

void exeKernel(cl_int &errNum, cl_kernel &kernel, cl_mem &inputSignalBuffer, 
cl_mem &outputSignalBuffer, cl_mem &maskBuffer, cl_command_queue &queue)
{
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
}



void createBuffer100(cl_mem &inputSignalBuffer,
	cl_mem &outputSignalBuffer,
	cl_mem &maskBuffer,  cl_context &context, cl_int &errNum)
{
    // Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight100 * maskWidth100,
		static_cast<void *>(mask100),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");
}


void exeKernel100(cl_int &errNum, cl_kernel &kernel, cl_mem &inputSignalBuffer, 
cl_mem &outputSignalBuffer, cl_mem &maskBuffer, cl_command_queue &queue)
{
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth100);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
	errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
}


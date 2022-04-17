#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;
cl_uint inputSignal[inputSignalHeight][inputSignalWidth] =
{
	{3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 3, 1, 1, 4, 8, 2, 1, 3, 3, 1, 1, 4, 8, 2, 1, 3},
	{4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4, 2, 1, 1, 2, 1, 2, 3, 4},
	{4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 4, 4, 4, 4, 3, 2, 2, 2, 5},
	{9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 9, 8, 3, 8, 9, 0, 0, 0, 8},
	{9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 9, 3, 3, 9, 0, 0, 0, 0, 6},
	{0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0, 9, 0, 8, 0, 0, 0, 0, 0},
	{3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 3, 0, 8, 8, 9, 4, 4, 4, 0},
	{5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
    {5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 5, 9, 8, 1, 8, 1, 1, 1, 0},
};


const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_uint mask[maskHeight][maskWidth] =
{
	{1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0}, 
	{1, 1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0}, 
	{1, 1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 0},
};

const unsigned int maskWidth100  = 24;
const unsigned int maskHeight100 = 24;

cl_uint mask100[maskHeight100][maskWidth100] =
{
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
};

const unsigned int maskWidth75  = 12;
const unsigned int maskHeight75 = 12;

cl_uint mask75[maskHeight75][maskWidth75] =
{
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0}, 
	{1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0},
};

const unsigned int maskWidth50  = 8;
const unsigned int maskHeight50 = 8;

cl_uint mask50[maskHeight50][maskWidth50] =
{
	{1, 1, 1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1, 0, 0}, 
	{1, 1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 0}, 
	{1, 0, 1, 1, 0, 1, 0, 0}, 
	{1, 1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 0, 0},
};

const unsigned int maskWidth25  = 6;
const unsigned int maskHeight25 = 6;

cl_uint mask25[maskHeight25][maskWidth25] =
{
	{1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1}, 
	{1, 0, 1, 1, 0, 1}, 
	{1, 1, 1, 1, 1, 1},
};

const unsigned int outputSignalWidth  = 49;
const unsigned int outputSignalHeight = 49;

cl_uint outputSignal[outputSignalHeight][outputSignalWidth];







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


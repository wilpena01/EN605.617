#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <chrono>
#include <iostream>
#include <string>
#include "Utilities.h"

using namespace std;
using namespace std::chrono;

#define N 10

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
        temp.push_back(g_X[i] % g_Y[i]);
    }
    stop = high_resolution_clock::now();
    d2 = duration_cast<microseconds>(stop - start);
    Z = g_Y;
}


int main()
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
    //microseconds mod_d1, mod_d2;

    thrust::host_vector<int> add(N);
    thrust::host_vector<int> sub(N);
    thrust::host_vector<int> mul(N);
    //thrust::host_vector<int> mod(N); 

    addAnalysis(X,Y,add,add_d1,add_d2);
    subAnalysis(X,Y,sub,sub_d1,sub_d2);
    mulAnalysis(X,Y,mul,mul_d1,mul_d2);
    //modAnalysis(X,Y,mod,mod_d1,mod_d2);

    string str[] ={"Thrus"};
    cout<<"X = ";   outputVec(X);
    cout<<"Y = ";   outputVec(Y);
    cout<<"add = "; outputVec(add); outputTime(add_d1,add_d2,str);
    cout<<"sub = "; outputVec(sub); outputTime(sub_d1,sub_d2,str);
    cout<<"mul = "; outputVec(mul); outputTime(mul_d1,mul_d2,str);
    //cout<<"mod = "; outputVec(mod); outputTime(mod_d1,mod_d2,str);   

    return 0;    
}
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

#define N 3

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

    // fill X, Y with randon numbers
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

    // print X
    cout<<"X = ";
    thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, " ")); cout<<endl;
    // print Y
    cout<<"Y = ";
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, " ")); cout<<endl;

    
    // print add
    cout<<"add = ";
    thrust::copy(add.begin(), add.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print sub
    cout<<"sub = ";
    thrust::copy(sub.begin(), sub.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print mul
    cout<<"mul = ";
    thrust::copy(mul.begin(), mul.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print mod
    cout<<"mod = ";
    thrust::copy(mod.begin(), mod.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
   
    return 0;    
}
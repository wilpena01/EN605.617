#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>

#include <iostream>
using namespace std;

int main()
{
    int N = 3;
    // allocate two host_vectors with N elements
    thrust::host_vector<int> X(N);
    thrust::host_vector<int> Y(N);

    // fill X, Y with randon numbers
   for(int i = 0; i<N ; i++)
   {
       X[i]=rand() % 10;
       Y[i]=rand() % 10;
   }

    // print X
    cout<<"X = ";
    thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print Y
    cout<<"Y = ";
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;

    // copy host to device
    thrust::device_vector<int> g_X = X;
    thrust::device_vector<int> g_Y = Y;
    thrust::device_vector<int> g_Z;

    // compute Z = X + Y
    thrust::device_vector<int> add = Y;
    thrust::transform(g_X.begin(), g_X.end(), add.begin(), add.begin(), thrust::plus<int>());

    cout<<"add = ";
    thrust::copy(add.begin(), add.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;

    // compute Z = X - Y
    thrust::device_vector<int> sub = Y;
    thrust::transform(g_X.begin(), g_X.end(), sub.begin(), sub.begin(), thrust::minus<int>());

    // compute Z = X * Y
    thrust::device_vector<int> mul = Y;
    thrust::transform(g_X.begin(), g_X.end(), mul.begin(), mul.begin(), thrust::multiplies<int>());

    // compute Z = X % Y
    thrust::device_vector<int> mod = Y;
    thrust::transform(g_X.begin(), g_X.end(), mod.begin(), mod.begin(), thrust::modulus<int>());

    // print Y
    cout<<"add = ";
    thrust::copy(add.begin(), add.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print Y
    cout<<"sub = ";
    thrust::copy(sub.begin(), sub.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print Y
    cout<<"mul = ";
    thrust::copy(mul.begin(), mul.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
    // print Y
    cout<<"mod = ";
    thrust::copy(mod.begin(), mod.end(), std::ostream_iterator<int>(std::cout, "\t")); cout<<endl;
   
    return 0;    
}
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>

#include <iostream>

int main()
{
    int N = 20;
    // allocate two host_vectors with N elements
    thrust::host_vector<int> X(N);
    thrust::host_vector<int> Y(N);

    // fill X, Y with randon numbers
    thrust::generate(X.begin(), X.end(), rand);
    thrust::generate(Y.begin(), Y.end(), rand); 

    // print X
    thrust::copy(X.begin(), X.end(), std::ostream_iterator<int>(std::cout, "\n"));
    // print Y
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

    // copy host to device
    thrust::device_vector<int> g_X = X;
    thrust::device_vector<int> g_Y = Y;
    thrust::device_vector<int> g_Z;

    // compute Z = X + Y
    thrust::device_vector<int> add = Y;
    thrust::transform(g_X.begin(), g_X.end(), add.begin(), add.begin(), thrust::plus<int>());

    // compute Z = X - Y
    thrust::device_vector<int> sub = Y;
    thrust::transform(g_X.begin(), g_X.end(), sub.begin(), sub.begin(), thrust::negate<int>());

    // compute Z = X * Y
    thrust::device_vector<int> mul = Y;
    thrust::transform(g_X.begin(), g_X.end(), mul.begin(), mul.begin(), thrust::multiplies<int>());

    // compute Z = X % Y
    thrust::device_vector<int> mod = Y;
    thrust::transform(g_X.begin(), g_X.end(), mod.begin(), mod.begin(), thrust::modulus<int>());

    // print Y
    thrust::copy(add.begin(), add.end(), std::ostream_iterator<int>(std::cout, "\n"));
    // print Y
    thrust::copy(sub.begin(), sub.end(), std::ostream_iterator<int>(std::cout, "\n"));
    // print Y
    thrust::copy(mul.begin(), mul.end(), std::ostream_iterator<int>(std::cout, "\n"));
    // print Y
    thrust::copy(mod.begin(), mod.end(), std::ostream_iterator<int>(std::cout, "\n"));
   
    return 0;    
}
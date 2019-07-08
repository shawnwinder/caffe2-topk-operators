#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
using namespace std;

template<typename T>
struct absolute_value : public unary_function<T, T> {
    __host__ __device__ T operator()(const T &x) const {
        return x < T(0) ? -x : x;
    }
};

int main() {
    int data[6] = {-1, 0, -2, -2, 1, -3};
    // int result = thrust::transform_reduce(
    //         thrust::host, 
    //         data, 
    //         data + 6,
    //         absolute_value<int>(),
    //         0,
    //         thrust::maximum<int>());
    
    int result = thrust::transform_reduce(
            thrust::host, 
            data, 
            data + 6,
            absolute_value<int>(),
            0,
            thrust::plus<int>());
    cout << "result: " << result << endl;

    return 0;
}

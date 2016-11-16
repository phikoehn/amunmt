#pragma once

#ifndef NO_CUDA

#include <thrust/device_vector.h>

template<class T>
using DeviceVector = thrust::device_vector<T>;

template<class T>
using HostVector = thrust::host_vector<T>;

namespace algo = thrust;
namespace iteralgo = thrust;

static void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#else

#include <vector>
#include <algorithm>

template<class T>
using DeviceVector = std::vector<T>;

template<class T>
using HostVector = std::vector<T>;

namespace algo = std;
namespace iteralgo = std;


#endif


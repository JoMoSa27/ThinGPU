#pragma once

#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

// Order-preserving map between float and int32, so that atomicMin/atomicMax on
// integers can be used to reduce floats.
namespace atomic_float {

__host__ __device__ inline int32_t encode(float f) {
    int32_t b;
    memcpy(&b, &f, sizeof(float));
    if (b & 0x80000000) b ^= 0x7fffffff;
    return b;
}

__host__ __device__ inline float decode(int32_t b) {
    if (b & 0x80000000) b ^= 0x7fffffff;
    float f;
    memcpy(&f, &b, sizeof(float));
    return f;
}

constexpr float EMPTY_MIN = +1e30f;
constexpr float EMPTY_MAX = -1e30f;

}

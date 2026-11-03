#pragma once
#include <cuda_runtime.h>
#include <cstring>
#include <stdint.h>
#include "SpatialTags.cuh"
#include "AtomicFloat.cuh"

template<int Dim>
__global__ void bbox_init_k(int32_t* mn, int32_t* mx)
{
    for (int d = 0; d < Dim; d++) {
        mn[d] = atomic_float::encode(atomic_float::EMPTY_MIN);
        mx[d] = atomic_float::encode(atomic_float::EMPTY_MAX);
    }
}

template<int Dim>
__global__ void bbox_reduce_k(
    CoordPtrs<Dim> coords, uint32_t n, int32_t* mn, int32_t* mx)
{
    __shared__ int32_t s_mn[Dim];
    __shared__ int32_t s_mx[Dim];

    if (threadIdx.x < (uint32_t)Dim) {
        s_mn[threadIdx.x] = atomic_float::encode(atomic_float::EMPTY_MIN);
        s_mx[threadIdx.x] = atomic_float::encode(atomic_float::EMPTY_MAX);
    }
    __syncthreads();

    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        for (int d = 0; d < Dim; d++) {
            const int32_t e = atomic_float::encode(coords.c[d][id]);
            atomicMin(&s_mn[d], e);
            atomicMax(&s_mx[d], e);
        }
    }
    __syncthreads();

    if (threadIdx.x < (uint32_t)Dim) {
        atomicMin(&mn[threadIdx.x], s_mn[threadIdx.x]);
        atomicMax(&mx[threadIdx.x], s_mx[threadIdx.x]);
    }
}

// Bounding box of n points, reduced per block in shared memory and then globally.
template<int Dim>
inline void compute_bbox_gpu(
    CoordPtrs<Dim> d_coords, uint32_t n, float* h_mn, float* h_mx)
{
    int32_t* d_buf = nullptr;
    cudaMalloc(&d_buf, 2 * Dim * sizeof(int32_t));
    int32_t* d_mn = d_buf;
    int32_t* d_mx = d_buf + Dim;

    bbox_init_k<Dim><<<1, 1>>>(d_mn, d_mx);

    const uint32_t block = 256;
    bbox_reduce_k<Dim><<<(n + block - 1) / block, block>>>(d_coords, n, d_mn, d_mx);

    int32_t h_buf[2 * Dim];
    cudaMemcpy(h_buf, d_buf, 2 * Dim * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_buf);

    for (int d = 0; d < Dim; d++) {
        h_mn[d] = atomic_float::decode(h_buf[d]);
        h_mx[d] = atomic_float::decode(h_buf[Dim + d]);
    }
}

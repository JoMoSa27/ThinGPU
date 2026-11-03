#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// BVH node in heap layout: the root is 0, the children of i are 2i + 1 and 2i + 2,
// and leaves occupy [n - 1, 2n - 2]. A leaf has leftChild = -1 and rightChild = point index.
template<int Dim>
struct BVHNode {
    float mn[Dim];
    float mx[Dim];
    int   leftChild;
    int   rightChild;
};

static_assert(sizeof(BVHNode<2>) == 24, "BVHNode<2> size changed");
static_assert(sizeof(BVHNode<3>) == 32, "BVHNode<3> size changed");

struct RayHitResult {
    int   pointIdx;
    float t;
};

// Morton codes: interleave the bits of the quantized coordinates.
__device__ inline uint32_t expandBits32(uint32_t v) {
    v &= 0x0000FFFFu;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__device__ inline uint32_t expandBits32_3(uint32_t v) {
    v &= 0x000003FFu;
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template<int Dim> __device__ inline uint32_t mortonCode32(const float* p);

template<> __device__ inline uint32_t mortonCode32<2>(const float* p) {
    const uint32_t ix = (uint32_t)(p[0] * 65535.0f);
    const uint32_t iy = (uint32_t)(p[1] * 65535.0f);
    return expandBits32(ix) | (expandBits32(iy) << 1);
}

template<> __device__ inline uint32_t mortonCode32<3>(const float* p) {
    const uint32_t ix = (uint32_t)(p[0] * 1023.0f);
    const uint32_t iy = (uint32_t)(p[1] * 1023.0f);
    const uint32_t iz = (uint32_t)(p[2] * 1023.0f);
    return (expandBits32_3(ix) << 2) | (expandBits32_3(iy) << 1) | expandBits32_3(iz);
}

// Slab test between a ray and a box; updates [tMin, tMax].
template<int Dim>
__device__ inline bool rayAABBIntersect(
    const float* ro, const float* rd,
    const float* mn, const float* mx,
    float& tMin, float& tMax)
{
    const float eps = 1e-6f;

    #pragma unroll
    for (int d = 0; d < Dim; d++) {
        if (fabsf(rd[d]) < eps) {

            if (ro[d] < mn[d] || ro[d] > mx[d]) return false;
        } else {
            float t1 = (mn[d] - ro[d]) / rd[d];
            float t2 = (mx[d] - ro[d]) / rd[d];
            if (t1 > t2) { const float tmp = t1; t1 = t2; t2 = tmp; }
            tMin = fmaxf(tMin, t1);
            tMax = fminf(tMax, t2);
            if (tMin > tMax) return false;
        }
    }
    return tMin < tMax;
}

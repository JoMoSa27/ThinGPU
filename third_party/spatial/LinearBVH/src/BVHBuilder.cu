#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include "../include/BVHBuilder.cuh"

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[BVH] CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[BVH] kernel error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
} while(0)

// Linear BVH build: Morton codes, radix sort, leaves, then inner boxes bottom up.

namespace {

struct BuildBufs {
    uint32_t* keys_in   = nullptr;
    uint32_t* vals_in   = nullptr;
    uint32_t* keys_out  = nullptr;
    uint32_t* vals_out  = nullptr;
    void*     cub_tmp   = nullptr;
    size_t    cub_bytes = 0;
};

inline size_t cub_sort_bytes(uint32_t n)
{
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (int)n);
    return bytes;
}

// One plan for both measuring and laying out the scratch buffer.
inline BuildBufs plan_scratch(ScratchLayout& L, uint32_t n)
{
    BuildBufs b;
    b.keys_in   = L.take<uint32_t>(n);
    b.vals_in   = L.take<uint32_t>(n);
    b.keys_out  = L.take<uint32_t>(n);
    b.vals_out  = L.take<uint32_t>(n);
    b.cub_bytes = cub_sort_bytes(n);
    b.cub_tmp   = L.take_bytes(b.cub_bytes);
    return b;
}

// Morton code of each normalized point; vals keeps the original index.
template<int Dim>
__global__ void computeMortonCodesKernel(
    CoordPtrs<Dim> c, int n, MortonNorm<Dim> nb,
    uint32_t* keys, uint32_t* vals)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float p[Dim];
    #pragma unroll
    for (int d = 0; d < Dim; d++)
        p[d] = (c.c[d][idx] - nb.mn[d]) * nb.inv[d];

    keys[idx] = mortonCode32<Dim>(p);
    vals[idx] = (uint32_t)idx;
}

// One leaf per point in Morton order; a leaf's box is the point itself.
template<int Dim>
__global__ void createLeavesKernel(
    CoordPtrs<Dim> c, const uint32_t* order, int n, BVHNode<Dim>* nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const uint32_t orig = order[idx];
    BVHNode<Dim>&  leaf = nodes[n - 1 + idx];

    #pragma unroll
    for (int d = 0; d < Dim; d++)
        leaf.mn[d] = leaf.mx[d] = c.c[d][orig];

    leaf.leftChild  = -1;
    leaf.rightChild = (int)orig;
}

// Inner node i takes the union of the boxes of children 2i + 1 and 2i + 2.
template<int Dim>
__global__ void buildInternalNodesKernel(int startIdx, int count, BVHNode<Dim>* nodes)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const int i     = startIdx + tid;
    const int left  = 2*i + 1;
    const int right = 2*i + 2;

    const BVHNode<Dim>& L = nodes[left];
    const BVHNode<Dim>& R = nodes[right];
    BVHNode<Dim>&       P = nodes[i];

    #pragma unroll
    for (int d = 0; d < Dim; d++) {
        P.mn[d] = fminf(L.mn[d], R.mn[d]);
        P.mx[d] = fmaxf(L.mx[d], R.mx[d]);
    }
    P.leftChild  = left;
    P.rightChild = right;
}

}

template<int Dim>
size_t bvh_scratch_bytes(uint32_t n)
{
    if (n == 0) return 0;
    ScratchLayout sizer;
    plan_scratch(sizer, n);
    return sizer.total();
}

template<int Dim>
void build_bvh_gpu(
    BVHNode<Dim>*  d_nodes,
    CoordPtrs<Dim> coords,
    uint32_t       n,
    const float*   bmin,
    const float*   bmax,
    void*          scratch,
    cudaStream_t   stream)
{
    if (n == 0) return;

    MortonNorm<Dim> nb;
    for (int d = 0; d < Dim; d++) {
        const float range = bmax[d] - bmin[d];
        nb.mn[d]  = bmin[d];

        // A flat dimension maps to 0 instead of dividing by zero.
        nb.inv[d] = (range > 0.f) ? (1.f / range) : 0.f;
    }

    ScratchLayout layout(scratch);
    BuildBufs b = plan_scratch(layout, n);

    const int TPB   = 256;
    const int nBlks = ((int)n + TPB - 1) / TPB;

    computeMortonCodesKernel<Dim><<<nBlks, TPB, 0, stream>>>(
        coords, n, nb, b.keys_in, b.vals_in);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        b.cub_tmp, b.cub_bytes,
        b.keys_in, b.keys_out,
        b.vals_in, b.vals_out,
        (int)n, 0, (int)(sizeof(uint32_t) * 8), stream));

    createLeavesKernel<Dim><<<nBlks, TPB, 0, stream>>>(
        coords, b.vals_out, n, d_nodes);
    CUDA_CHECK_KERNEL();

    // Heap levels from the deepest up, so children are ready before their parent.
    if (n > 1) {
        int maxDepth = 0;
        for (int tmp = (int)n - 1; tmp > 1; tmp >>= 1) ++maxDepth;

        for (int d = maxDepth; d >= 0; --d) {
            const int startIdx = (1 << d) - 1;
            const int endIdx   = std::min(2*(1 << d) - 2, (int)n - 2);
            const int count    = endIdx - startIdx + 1;
            if (count <= 0) continue;
            buildInternalNodesKernel<Dim><<<(count + TPB - 1) / TPB, TPB, 0, stream>>>(
                startIdx, count, d_nodes);
            CUDA_CHECK_KERNEL();
        }
    }
}

template size_t bvh_scratch_bytes<2>(uint32_t);
template size_t bvh_scratch_bytes<3>(uint32_t);

template void build_bvh_gpu<2>(BVHNode<2>*, CoordPtrs<2>, uint32_t,
                               const float*, const float*, void*, cudaStream_t);
template void build_bvh_gpu<3>(BVHNode<3>*, CoordPtrs<3>, uint32_t,
                               const float*, const float*, void*, cudaStream_t);
